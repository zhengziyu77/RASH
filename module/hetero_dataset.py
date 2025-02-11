import torch
import numpy as np
import scipy.sparse as sp
from torch_geometric.data import HeteroData
from torch_geometric.datasets import IMDB, AMiner, HGBDataset
from torch_geometric.transforms import AddMetaPaths
from torch_geometric.utils import to_undirected, add_self_loops
import pickle
from collections import defaultdict
import torch_geometric.transforms as T

from torch_sparse import SparseTensor
from sklearn.preprocessing import StandardScaler
from data.load_liar import Liar
import random
import os

def scale_feats(x):
    scaler = StandardScaler()
    #scaler = MaxAbsScaler()
    
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats
def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples = labels.shape[0]
    num_classes = labels.max() + 1
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples = labels.shape[0]
    num_classes = labels.max() + 1
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(
            random_state, labels, train_examples_per_class)
    else:

        # select train examples with no respect to class distribution
        train_indices = random_state.choice(
            remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(
            random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(
            remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(
            remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)
               ) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)
               ) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate(
            (train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


def train_test_split(labels, seed, train_examples_per_class=None, val_examples_per_class=None,
                     test_examples_per_class=None, train_size=None, val_size=None, test_size=None):
    random_state = np.random.RandomState(seed)
    train_indices, val_indices, test_indices = get_train_val_test_split(
        random_state, labels, train_examples_per_class, val_examples_per_class, test_examples_per_class, train_size,
        val_size, test_size)

    # print('number of training: {}'.format(len(train_indices)))
    # print('number of validation: {}'.format(len(val_indices)))
    # print('number of testing: {}'.format(len(test_indices)))

    train_mask = np.zeros((labels.shape[0], 1), dtype=int)
    train_mask[train_indices, 0] = 1
    train_mask = np.squeeze(train_mask, 1)
    val_mask = np.zeros((labels.shape[0], 1), dtype=int)
    val_mask[val_indices, 0] = 1
    val_mask = np.squeeze(val_mask, 1)
    test_mask = np.zeros((labels.shape[0], 1), dtype=int)
    test_mask[test_indices, 0] = 1
    test_mask = np.squeeze(test_mask, 1)
    mask = {}
    mask['train'] = train_mask
    mask['val'] = val_mask
    mask['test'] = test_mask
    return mask

def preprocess_sp_features(features):
    features = features.tocoo()
    row = torch.from_numpy(features.row)
    col = torch.from_numpy(features.col)
    e = torch.stack((row, col))
    v = torch.from_numpy(features.data)
    x = torch.sparse_coo_tensor(e, v, features.shape).to_dense()
    x.div_(x.sum(dim=-1, keepdim=True).clamp_(min=1.))
    return x


def preprocess_th_features(features):
    x = features.to_dense()
    x.div_(x.sum(dim=-1, keepdim=True).clamp_(min=1.))
    return x


def nei_to_edge_index(nei, reverse=False):
    edge_indexes = []

    for src, dst in enumerate(nei):
        src = torch.tensor([src], dtype=dst.dtype, device=dst.device)
        src = src.repeat(dst.shape[0])
        if reverse:
            edge_index = torch.stack((dst, src))
        else:
            edge_index = torch.stack((src, dst))

        edge_indexes.append(edge_index)

    return torch.cat(edge_indexes, dim=1)


def sp_feat_to_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def sp_adj_to_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return indices


def make_sparse_eye(N):
    e = torch.arange(N, dtype=torch.long)
    e = torch.stack([e, e])
    o = torch.ones(N, dtype=torch.float32)
    return torch.sparse_coo_tensor(e, o, size=(N, N))
def get_adj_from_edges(edges, weights, size):
    adj = torch.zeros(size)
    adj[edges[0], edges[1]] = weights
    return adj

def make_sparse_tensor(x):
    row, col = torch.where(x == 1)
    e = torch.stack([row, col])
    o = torch.ones(e.shape[1], dtype=torch.float32)
    return torch.sparse_coo_tensor(e, o, size=x.shape)


data_folder = "/data/"
def load_dblp(ratio):
    path = data_folder + "dblp/"
    #ratio = [20, 40, 60]

    label = np.load(path + "labels.npy").astype('int32')
    # nei_p = np.load(path + "nei_p.npy", allow_pickle=True)
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_p = sp.load_npz(path + "p_feat.npz").astype("float32")
    apa = sp.load_npz(path + "apa.npz")
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")
    # pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = torch.LongTensor(label)
    # nei_p = nei_to_edge_index([torch.LongTensor(i) for i in nei_p], True)
    feat_p = preprocess_sp_features(feat_p)
    feat_a = preprocess_sp_features(feat_a)
    
    #feat_p = make_sparse_eye(14328)
    feat_c = make_sparse_eye(20)
    feat_t = make_sparse_eye(7723)
    #feat_a = preprocess_sp_features(feat_a)
    #feat_p = preprocess_th_features(feat_p)
    feat_c = preprocess_th_features(feat_c)
    feat_t = preprocess_th_features(feat_t)
  



    apa = sp_adj_to_tensor(apa)
    apcpa = sp_adj_to_tensor(apcpa)
    aptpa = sp_adj_to_tensor(aptpa)
    pa = np.genfromtxt(path + "pa.txt").astype('int64').T


    pc = np.genfromtxt(path + "pc.txt").astype('int64').T
    pt = np.genfromtxt(path + "pt.txt").astype('int64').T
    type_num = {'p':14328,'a':4057,'c':20, 't':7723}

    adj_pp = torch.zeros(type_num['p'], type_num['p'])
    adj_pa = get_adj_from_edges(pa,1,(type_num['p'], type_num['a']))
    adj_pc = get_adj_from_edges(pc,1,(type_num['p'], type_num['c']))
    adj_pt = get_adj_from_edges(pt,1,(type_num['p'], type_num['t']))
    adj_p = torch.cat([adj_pp, adj_pa, adj_pc, adj_pt], dim=1)

    p_edge_index = torch.nonzero(adj_p).T

    adj_dict = {}
    adj_dict['total'] = to_undirected(p_edge_index)



    # pos = sp_adj_to_tensor(pos)
    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]

    data = HeteroData()
    schemaData = HeteroData()
    mask = torch.tensor([False] * feat_a.shape[0])
    data['a'].x = feat_a
    data['a'].y = label


    train_mask_l = "train_mask"
    val_mask_l = "val_mask"
    test_mask_l = "test_mask"

    data['a'][train_mask_l] = []
    data['a'][val_mask_l] = []
    data['a'][test_mask_l] = []
    for r, tr, va, te in zip(ratio, train, val, test):
        train_mask_l = f"{r}_train_mask"
        train_mask = mask.clone()
        train_mask[tr] = True

        val_mask_l = f"{r}_val_mask"
        val_mask = mask.clone()
        val_mask[va] = True

        test_mask_l = f"{r}_test_mask"
        test_mask = mask.clone()
        test_mask[te] = True

        train_mask_l = "train_mask"
        val_mask_l = "val_mask"
        test_mask_l = "test_mask"

        data['a'][train_mask_l].append(train_mask)
        data['a'][val_mask_l].append(val_mask)
        data['a'][test_mask_l].append(test_mask)

    data['p'].x = feat_p
    data['c'].x = feat_c
    data['t'].x = feat_t
    
    #schemaData[('a', 'p')].edge_index = torch.tensor(ap)
    #schemaData[('a', 'p')].edge_index[[0, 1]] = schemaData[('a', 'p')].edge_index[[1, 0]]

    data[('p','to', 'a')].edge_index = torch.tensor(pa)
    data[('a','to', 'p')].edge_index = torch.tensor(pa)[[1,0]]
    data[('p', 'to','c')].edge_index = torch.tensor(pc)
    data[('c','to', 'p')].edge_index = torch.tensor(pc)[[1,0]]
    data[('p', 'to','t')].edge_index = torch.tensor(pt)
    data[('t','to', 'p')].edge_index = torch.tensor(pt)[[1,0]]

  
    #data[('a', 'p', 'a')].edge_index = apa
    #data[('a', 'pcp', 'a')].edge_index = apcpa
    #data[('a', 'ptp', 'a')].edge_index = aptpa
    # data[('a', 'pos', 'a')].edge_index = pos
    data["dataset"]="dblp"
    metapath_dict = {
        ('a', 'p', 'a'): None,
        ('a', 'pcp', 'a'): None,
        ('a', 'ptp', 'a'): None
    }

    schema_dict = {
        ('a', 'p'): None
    }

    data['metapath_dict'] = metapath_dict
    data['schema_dict'] = schema_dict
    schemaData['schema_dict'] = schema_dict
    data['main_node'] = 'a'
    data['use_nodes'] = ('a', 'p','t','c')
    node_types = ['p', 'a', 'c','t']
    data.node_types = node_types
    data.adj_dict = adj_dict
    node_idx = {}
    node_idx['p'] = torch.LongTensor([i for i in range(14328)])
    node_idx['a'] = torch.LongTensor([i for i in range(14328,18385)])
    node_idx['c'] = torch.LongTensor([i for i in range(18385, 18405)])
    node_idx['t'] = torch.LongTensor([i for i in range(18405, 26128)])

    init_idx = {}
    init_idx['p'] = torch.LongTensor([i for i in range(type_num['p'])])
    init_idx['a'] = torch.LongTensor([i for i in range(type_num['a'])])
    init_idx['c'] = torch.LongTensor([i for i in range(type_num['c'])])
    init_idx['t'] = torch.LongTensor([i for i in range(type_num['t'])])

    #id映射
    data.adj_dict = adj_dict
    data.node_idx = node_idx
    node_num = type_num['p']+type_num['a']+type_num['c'] + type_num['t']
    id_map={}

    id_map = torch.zeros((1,node_num),dtype=torch.long).squeeze()
    init_to_id ={}
    for k in init_idx:
        init_to_id[k] = torch.zeros((1,node_num),dtype=torch.long).squeeze()
    for node_type in init_idx:
        id_map[node_idx[node_type]] = init_idx[node_type]
        init_to_id[node_type][init_idx[node_type]] = node_idx[node_type]
    data.init_id_map = init_to_id
    data.id_map = id_map


    # x_dict = init_feat(node_types, 128, type_num)
    # for k in x_dict:
    #     data[k].x = x_dict[k]
    # x_dict = {}
    # for k in node_types:
    #     feat = make_sparse_eye(type_num[k])
    #     data[k].x = preprocess_th_features(feat)
    return data


def load_acm(ratio):
    path = data_folder + "acm/"
    #ratio = [20, 40, 60]
    label = np.load(path + "labels.npy").astype('int32')
    # nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    # nei_s = np.load(path + "nei_s.npy", allow_pickle=True)
    feat_p = sp.load_npz(path + "p_feat.npz").astype("float32")
    feat_p = preprocess_sp_features(feat_p)

    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")
    feat_a = preprocess_sp_features(feat_a)

    #feat_a = make_sparse_eye(7167)
    feat_s = make_sparse_eye(60)
    #feat_a = preprocess_th_features(feat_a)
    feat_s = preprocess_th_features(feat_s)
    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    # pos = sp.load_npz(path + "pos.npz")

    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = torch.LongTensor(label)
    # nei_a = nei_to_edge_index([torch.LongTensor(i) for i in nei_a])
    # nei_s = nei_to_edge_index([torch.LongTensor(i) for i in nei_s])

    pap = sp_adj_to_tensor(pap)
    psp = sp_adj_to_tensor(psp)
    # pos = sp_adj_to_tensor(pos)
    pa = np.genfromtxt(path + "pa.txt").astype('int64').T
    ps = np.genfromtxt(path + "ps.txt").astype('int64').T
    


    type_num = {'p':4019,'a':7167,'s':60}
    adj_pp = torch.zeros((4019,4019))
    adj_pa = get_adj_from_edges(pa,1,(type_num['p'], type_num['a']))
    adj_ps = get_adj_from_edges(ps,1,(type_num['p'], type_num['s']))
    adj_p = torch.cat([adj_pp, adj_pa, adj_ps], dim=1)
    
    p_edge_index = torch.nonzero(adj_p).T
    print(p_edge_index)
    adj_dict = {}
    adj_dict['total'] = to_undirected(p_edge_index)
    #adj_dict['total'] = p_edge_index

    #adj_dict['total'] = p_edge_index


    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]

    data = HeteroData()
    schemaData = HeteroData()
    mask = torch.tensor([False] * feat_p.shape[0])

    data['p'].x = feat_p
    data['a'].x = feat_a
    data['s'].x = feat_s
    data['p'].y = label


    data['p']["train_mask"] = []
    data['p']["val_mask"] = []
    data['p']["test_mask"] = []

    for r, tr, va, te in zip(ratio, train, val, test):
        train_mask_l = f"{r}_train_mask"
        train_mask = mask.clone()
        train_mask[tr] = True

        val_mask_l = f"{r}_val_mask"
        val_mask = mask.clone()
        val_mask[va] = True

        test_mask_l = f"{r}_test_mask"
        test_mask = mask.clone()
        test_mask[te] = True

        data['p'][train_mask_l] = train_mask
        data['p'][val_mask_l] = val_mask
        data['p'][test_mask_l] = test_mask
        train_mask_l = "train_mask"
        val_mask_l = "val_mask"
        test_mask_l = "test_mask"

        data['p'][train_mask_l].append(train_mask)
        data['p'][val_mask_l].append(val_mask)
        data['p'][test_mask_l].append(test_mask)

    schemaData[('p', 'a')].edge_index = torch.tensor(pa)
    schemaData[('p', 'a')].edge_index[[0, 1]] = schemaData[('p', 'a')].edge_index[[1, 0]]

    schemaData[('p', 's')].edge_index = torch.tensor(ps)
    schemaData[('p', 's')].edge_index[[0, 1]] = schemaData[('p', 's')].edge_index[[1, 0]]
    
   
    data[('p','to', 'a')].edge_index = torch.tensor(pa)
    data[('a','to', 'p')].edge_index = torch.tensor(pa)[[1,0]]
    data[('p', 'to','s')].edge_index = torch.tensor(ps)
    data[('s','to', 'p')].edge_index = torch.tensor(ps)[[1,0]]
 
    
    process_edge_dict = {
        ('p', 'to','a'): torch.tensor([pa[0],pa[1]+4018]),
        ('p', 'to','s'): torch.tensor([ps[0],ps[1]+4018+7167]),
        ('a', 'to','p'): torch.tensor([pa[1]+4018, pa[0]]),
        ('s', 'to','p'): torch.tensor([ps[1]+4018+7167, ps[0]])
    }
    data["process_edge_index_dict"]= process_edge_dict




    metapath_dict = {
        ('p', 'a', 'p'): None,
        ('p', 's', 'p'): None
    }

    schema_dict = {
        ('p', 'a'): None,
        ('p', 's'): None
    }
    # schema_dict = {
    #     ('p', 'a'): None,
    #     ('p', 's'): None
    # }

    #data['metapath_dict'] = metapath_dict
    #data['schema_dict'] = schema_dict
    schemaData['schema_dict'] = schema_dict
    data['main_node'] = 'p'
    data['use_nodes'] = ('p', 'a', 's')
    # data['use_nodes'] = ('p')
    data["dataset"] = "acm"
    data.adj_dict = adj_dict
    node_idx = {}
    node_idx['p'] = torch.LongTensor([i for i in range(4019)])
    node_idx['a'] = torch.LongTensor([i for i in range(4019, 11186)])
    node_idx['s'] = torch.LongTensor([i for i in range(11186, 11246)])
    data.node_idx = node_idx
    init_idx = {}
    init_idx['p'] = torch.LongTensor([i for i in range(type_num['p'])])
    init_idx['a'] = torch.LongTensor([i for i in range(type_num['a'])])
    init_idx['s'] = torch.LongTensor([i for i in range(type_num['s'])])
    #id映射
    data.adj_dict = adj_dict
    data.node_idx = node_idx
    node_num = type_num['p']+type_num['a']+type_num['s']
    id_map={}

    id_map = torch.zeros((1,node_num),dtype=torch.long).squeeze()

    init_to_id ={}
    for k in init_idx:
        init_to_id[k] = torch.zeros((1,node_num),dtype=torch.long).squeeze()
    for node_type in init_idx:
        id_map[node_idx[node_type]] = init_idx[node_type]
        init_to_id[node_type][init_idx[node_type]] = node_idx[node_type]
    data.init_id_map = init_to_id
    data.id_map = id_map
    data.node_num = node_num
    node_types = ['p', 'a', 's']
    data.node_types = node_types


    return data


def load_aminer(ratio):
    #ratio = [20, 40, 60]
    #ratio =[20]
    path = data_folder + "aminer/"
    label = np.load(path + "labels.npy").astype('int32')
    # nei_a = np.load(path + "nei_a.npy", allow_pickle=True)
    # nei_r = np.load(path + "nei_r.npy", allow_pickle=True)
    # Because none of P, A or R has features, we assign one-hot encodings to all of them.
    feat_p_one = make_sparse_eye(6564)
    feat_a_one = make_sparse_eye(13329)
    feat_r_one = make_sparse_eye(35890)
    pap = sp.load_npz(path + "pap.npz")
    prp = sp.load_npz(path + "prp.npz")
    pa = np.genfromtxt(path + "pa.txt").astype('int64').T
    pr = np.genfromtxt(path + "pr.txt").astype('int64').T

    type_num = {'p':6564,'a':13329,'r':35890}
    adj_pp = torch.zeros((6564, 6564))
    adj_pa = get_adj_from_edges(pa,1,(type_num['p'], type_num['a']))
    adj_pr = get_adj_from_edges(pr,1,(type_num['p'], type_num['r']))
    adj_p = torch.cat([adj_pp, adj_pa, adj_pr], dim=1)
    
    p_edge_index = torch.nonzero(adj_p).T
    print(p_edge_index)
    adj_dict = {}
    adj_dict['total'] = to_undirected(p_edge_index)



    # mw = np.genfromtxt(path + "mw.txt").astype('int64').T
    # pos = sp.load_npz(path + "pos.npz")
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = torch.LongTensor(label)
    # nei_a = nei_to_edge_index([torch.LongTensor(i) for i in nei_a])
    # nei_r = nei_to_edge_index([torch.LongTensor(i) for i in nei_r])
    feat_p = preprocess_th_features(feat_p_one)
    feat_a_one = preprocess_th_features(feat_a_one)
    feat_r_one = preprocess_th_features(feat_r_one)
    pap = sp_adj_to_tensor(pap)
    prp = sp_adj_to_tensor(prp)
    # pos = sp_adj_to_tensor(pos)
    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]

    data = HeteroData()
    schemaData = HeteroData()
    mask = torch.tensor([False] * feat_p.shape[0])


    data['p'].y = label
    node_types =['p','a','r']
    for i, node_type in enumerate(node_types):
        x = np.load(path+f'features_{i}.npy')
        #ft_dict[node_type] = scale_feats(torch.from_numpy(x).to(torch.float))
        data[node_type].x = torch.from_numpy(x).to(torch.float)

    #data['p'].x = feat_p
    #data['a'].x = feat_a_one
    #data['r'].x = feat_r_one
    data['p']["train_mask"] = []
    data['p']["val_mask"] = []
    data['p']["test_mask"] = []
    for r, tr, va, te in zip(ratio, train, val, test):
        train_mask_l = f"{r}_train_mask"
        train_mask = mask.clone()
        train_mask[tr] = True

        val_mask_l = f"{r}_val_mask"
        val_mask = mask.clone()
        val_mask[va] = True

        test_mask_l = f"{r}_test_mask"
        test_mask = mask.clone()
        test_mask[te] = True

        data['p'][train_mask_l] = train_mask
        data['p'][val_mask_l] = val_mask
        data['p'][test_mask_l] = test_mask

        train_mask_l = "train_mask"
        val_mask_l = "val_mask"
        test_mask_l = "test_mask"

        data['p'][train_mask_l].append(train_mask)
        data['p'][val_mask_l].append(val_mask)
        data['p'][test_mask_l].append(test_mask)

    data[('p','to', 'a')].edge_index = torch.tensor(pa)
    data[('a','to', 'p')].edge_index = torch.tensor(pa)[[1,0]]
    data[('p', 'to','r')].edge_index = torch.tensor(pr)
    data[('r','to', 'p')].edge_index = torch.tensor(pr)[[1,0]]

    #data[('p','rev_to', 'a')].edge_index = torch.tensor(pa)
    #data[('a','rev_to', 'p')].edge_index = torch.tensor(pa)[[1,0]]
    #data[('p', 'rev_to','r')].edge_index = torch.tensor(pr)
    #data[('r','rev_to', 'p')].edge_index = torch.tensor(pr)[[1,0]]
    #data[('p', 'a', 'p')].edge_index = pap
    #data[('p', 'r', 'p')].edge_index = prp
    # data[('p', 'pos', 'p')].edge_index = pos

    metapath_dict = {
        ('p', 'a', 'p'): None,
        ('p', 'r', 'p'): None
    }

    schema_dict = {
        ('p', 'a'): None,
        ('p', 'r'): None
    }

    data['metapath_dict'] = metapath_dict
    data['schema_dict'] = schema_dict
    schemaData['schema_dict'] = schema_dict
    data['main_node'] = 'p'
    data['use_nodes'] = ('p', 'a', 'r')
    data["dataset"] = "aminer"
    node_idx = {}
    node_idx['p'] = torch.LongTensor([i for i in range(6564)])
    node_idx['a'] = torch.LongTensor([i for i in range(6564, 19893)])
    node_idx['r'] = torch.LongTensor([i for i in range(19893, 55783)])
    init_idx = {}
    init_idx['p'] = torch.LongTensor([i for i in range(type_num['p'])])
    init_idx['a'] = torch.LongTensor([i for i in range(type_num['a'])])
    init_idx['r'] = torch.LongTensor([i for i in range(type_num['r'])])
        #id映射

    data.adj_dict = adj_dict
    data.node_idx = node_idx
    node_num = type_num['p']+type_num['a']+type_num['r']
    id_map={}

    id_map = torch.zeros((1,node_num),dtype=torch.long).squeeze()
    init_to_id ={}
    for k in init_idx:
        init_to_id[k] = torch.zeros((1,node_num),dtype=torch.long).squeeze()
    for node_type in init_idx:
        id_map[node_idx[node_type]] = init_idx[node_type]
        init_to_id[node_type][init_idx[node_type]] = node_idx[node_type]
    data.init_id_map = init_to_id
    data.id_map = id_map
    data.node_num = node_num
    #data.node_types = node_types

    return data



def load_imdb(ratio):
    #ratio = [20, 40, 60]
    data = IMDB(root='./data/imdb/')[0]
    node_types = ['m', 'd', 'a']
    # data[('m', 'pos', 'm')].edge_index = pos
    data['main_node'] = 'm'
    data['use_nodes'] = ('m', 'd', 'a')
    #data['use_nodes'] = ('m')

    data["dataset"] = "imdb"

    node_idx = {}
    type_num = {'m':4278, 'd':2081,'a':5257}

    adj_mm = torch.zeros((type_num['m'], type_num['m']))
    ma = data[('m','to','a')].edge_index

    md = data[('m','to','d')].edge_index

    adj_ma = get_adj_from_edges(ma,1,(type_num['m'], type_num['a']))
    adj_md = get_adj_from_edges(md,1,(type_num['m'], type_num['d']))
    adj_m = torch.cat([adj_mm, adj_md, adj_ma], dim=1)
    m_edge_index = torch.nonzero(adj_m).T
    adj_dict = {}
    adj_dict['total'] = to_undirected(m_edge_index)

    node_idx['m'] = torch.LongTensor([i for i in range(4278)])
    node_idx['d'] = torch.LongTensor([i for i in range(4278, 6359)])
    node_idx['a'] = torch.LongTensor([i for i in range(6359, 11616)])


    init_idx = {}
    init_idx['m'] = torch.LongTensor([i for i in range(type_num['m'])])
    init_idx['d'] = torch.LongTensor([i for i in range(type_num['d'])])
    init_idx['a'] = torch.LongTensor([i for i in range(type_num['a'])])

    #id映射
    

    data.adj_dict = adj_dict
    data.node_idx = node_idx
    node_num = type_num['m']+type_num['d']+type_num['a'] 
    id_map={}

    id_map = torch.zeros((1,node_num),dtype=torch.long).squeeze()
    init_to_id ={}
    for k in init_idx:
        init_to_id[k] = torch.zeros((1,node_num),dtype=torch.long).squeeze()
    for node_type in init_idx:
        id_map[node_idx[node_type]] = init_idx[node_type]
        init_to_id[node_type][init_idx[node_type]] = node_idx[node_type]
    data.init_id_map = init_to_id
    
    data.id_map = id_map
    data.node_num = node_num
    data.node_types = node_types
    
  
    path = data_folder+"imdb/"
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]
    mask = torch.tensor([False] * type_num['m'])



    data['m']['train_mask'] = []
    data['m']['val_mask'] = []
    data['m']['test_mask'] = []
    for r, tr, va, te in zip(ratio, train, val, test):
        train_mask_l = f"{r}_train_mask"
        train_mask = mask.clone()
        train_mask[tr] = True

        val_mask_l = f"{r}_val_mask"
        val_mask = mask.clone()
        val_mask[va] = True

        test_mask_l = f"{r}_test_mask"
        test_mask = mask.clone()
        test_mask[te] = True

        data['m'][train_mask_l] = train_mask
        data['m'][val_mask_l] = val_mask
        data['m'][test_mask_l] = test_mask
        train_mask_l = "train_mask"
        val_mask_l = "val_mask"
        test_mask_l = "test_mask"
        data['m'][train_mask_l].append(train_mask)
        data['m'][val_mask_l].append(val_mask)
        data['m'][test_mask_l].append(test_mask)


    """
    data['m']['train_mask'] = []
    data['m']['val_mask'] = []
    data['m']['test_mask'] = []

    for r in ratio:
        mask = train_test_split(
            data['m'].y.detach().cpu().numpy(), seed=0, train_examples_per_class=r,
            val_size=1000, test_size=1000)
        
        #train_mask_l = f"{r}_train_mask"
        #train_mask = mask['train'].astype(bool)
        #val_mask_l = f"{r}_val_mask"
        #val_mask = mask['val'].astype(bool)

        #test_mask_l = f"{r}_test_mask"
        #test_mask = mask['test'].astype(bool)
        
        train_mask_l = "train_mask"
        train_mask = mask['train'].astype(bool)
        val_mask_l = "val_mask"
        val_mask = mask['val'].astype(bool)

        test_mask_l = "test_mask"
        test_mask = mask['test'].astype(bool)
        data['m'][train_mask_l].append(train_mask)
        data['m'][val_mask_l].append(val_mask)
        data['m'][test_mask_l].append(test_mask)
    """
  
    x = np.load(path+f'features_movie.npy')

    #x = th.FloatTensor(x)
    data['m'].x = torch.from_numpy(x).to(torch.float)
    x = sp.load_npz(path + "features_1.npz").astype("float32")
    data['d'].x = torch.FloatTensor(preprocess_features(x))
    x = sp.load_npz(path + "features_2.npz").astype("float32")
    data['a'].x = torch.FloatTensor(preprocess_features(x))


    y = data['m'].y
    #torch.save(y,"m_labels.npy")
    #stop
    # x_dict = init_feat(['m','d','a'], 128, type_num)

    # for k in x_dict:
    #     data[k].x = x_dict[k]

    # feat_m = make_sparse_eye(4278)
    # feat_d = make_sparse_eye(2081)
    # feat_a = make_sparse_eye(5257)
    # feat_m = preprocess_th_features(feat_m)
    # feat_d = preprocess_th_features(feat_d)
    # feat_a = preprocess_th_features(feat_a)
    # data['m'].x = feat_m
    # data['d'].x = feat_d
    # data['a'].x = feat_a
    return data



    
def preprocess_features(features, norm=True):
    """Row-normalize feature matrix and convert to tuple representation"""
    if sp.issparse(features):
        features = features.toarray()
    if norm:
        #features[features>0] = 1
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1.0).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
    return torch.FloatTensor(features)
def load_yelp(ratio):

    path = data_folder +'/yelp"
    with open(path + '/meta_data.pkl', 'rb') as f:
        data = pickle.load(f)
    node_idx = {}
    for t in data['t_info'].keys():
        node_idx[t] = torch.LongTensor([i for p, i in data['node2gid'].items() if p.startswith(t)])
    with open(path + '/labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open(path + '/edges.pkl', 'rb') as f:
        edges = pickle.load(f)
    with open(path + "/node_features.pkl", "rb") as f:
        features = pickle.load(f)
    node_rel = defaultdict(list)
    for rel in edges:
        s, t = rel.split('-')
        node_rel[s].append(rel)

    features = preprocess_features(features, norm=True)
    label = np.load(path + "/labels.npy").astype('int32')
    print(label)
    data = HeteroData()

    graph = {t: [m for m in ms] for t, ms in edges.items()}
    #print(edges['b-s'])
    #print(edges['b-l'])
    edge_index_dict = {}
    for edge_type in edges:
        row, col = torch.nonzero(torch.LongTensor(edges[edge_type].toarray())).T
        row -= torch.min(row)
        col -= torch.min(col)
        row = row.view(1,-1)
        col = col.view(1,-1)
        src,_,dst = edge_type
        data[(src, 'to', dst)].edge_index = torch.cat((row, col), dim=0)

    node_types = ['b', 'u', 's', 'l']
    type_num = {'b':2614, 'u':1286,'s':4, 'l': 9}
    node_num = type_num['b']+type_num['u']+type_num['s'] + type_num['l'] 

    for edge_type, edge_index in data.edge_index_dict.items():
        src,_, dst = edge_type
    adj_bb = torch.zeros((type_num['b'], type_num['b']))
    adj_bu = get_adj_from_edges(data[('b', 'to', 'u')].edge_index, 1,(type_num['b'], type_num['u']))
    adj_bs = get_adj_from_edges(data[('b', 'to', 's')].edge_index, 1, (type_num['b'], type_num['s']))
    adj_bl = get_adj_from_edges(data[('b', 'to', 'l')].edge_index, 1, (type_num['b'], type_num['l']))
    adj_b = torch.cat([adj_bb, adj_bu, adj_bs, adj_bl], dim=1)
    b_edge_index = torch.nonzero(adj_b).T
    adj_dict = {}
    adj_dict['total'] = to_undirected(b_edge_index)


    for n_type in node_idx:
        data[n_type].x = features[node_idx[n_type]]
    #for i, node_type in enumerate(node_types):
    #    #x = np.load(path+f'/features_{i}.npz')
    #    x = sp.load_npz(path + "/features_0.npz").astype("float32")
    #    data[node_type].x = torch.FloatTensor(preprocess_features(x))
    # x_dict = init_feat(node_types, 128, type_num)
    # for k in x_dict:
    #     data[k].x = x_dict[k]
    # x_dict = {}
    # for k in node_types:
    #     feat = make_sparse_eye(type_num[k])
    #     data[k].x = preprocess_th_features(feat)
    #data['b'].y = labels[0] + labels[1] + labels[2]
    data['b'].y = torch.LongTensor(label)
    #features = preprocess_sp_features(features)
    #features = preprocess_th_features(features)
    init_idx = {}
    init_idx['b'] = torch.LongTensor([i for i in range(type_num['b'])])
    init_idx['u'] = torch.LongTensor([i for i in range(type_num['u'])])
    init_idx['s'] = torch.LongTensor([i for i in range(type_num['s'])])
    init_idx['l'] = torch.LongTensor([i for i in range(type_num['l'])])


    data.adj_dict = adj_dict
    data.node_idx = node_idx
    id_map={}

    id_map = torch.zeros((1,node_num),dtype=torch.long).squeeze()
    init_to_id ={}
    for k in init_idx:
        init_to_id[k] = torch.zeros((1,node_num),dtype=torch.long).squeeze()
    for node_type in init_idx:
        id_map[node_idx[node_type]] = init_idx[node_type]
        init_to_id[node_type][init_idx[node_type]] = node_idx[node_type]
    data.init_id_map = init_to_id
    
    data.id_map = id_map
    data.node_num = node_num
    data.node_types = node_types
    data.num_classes = 3
    train = [np.load(path + "/train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "/test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "/val_" + str(i) + ".npy") for i in ratio]

    train = [torch.LongTensor(i) for i in train]
    val = [torch.LongTensor(i) for i in val]
    test = [torch.LongTensor(i) for i in test]
    mask = torch.tensor([False] * data['b'].x.shape[0])



    data['b']["train_mask"] = []
    data['b']["val_mask"] = []
    data['b']["test_mask"] = []
    for r, tr, va, te in zip(ratio, train, val, test):
        train_mask_l = f"{r}_train_mask"
        train_mask = mask.clone()
        train_mask[tr] = True

        val_mask_l = f"{r}_val_mask"
        val_mask = mask.clone()
        val_mask[va] = True

        test_mask_l = f"{r}_test_mask"
        test_mask = mask.clone()
        test_mask[te] = True

        data['b']["train_mask"].append(train_mask)
        data['b']["val_mask"].append(val_mask)
        data['b']["test_mask"].append(test_mask)

    return data
  


def DHT( edge_index, batch, add_loops=True):

    num_edge = edge_index.size(1)
    device = edge_index.device
    ### Transform edge list of the original graph to hyperedge list of the dual hypergraph
    edge_to_node_index = torch.arange(0,num_edge,1, device=device).repeat_interleave(2).view(1,-1)
    
    #edge_to_node_index = torch.arange(0,num_edge,1, device=device).view(1,-1)

    hyperedge_index = edge_index.T.reshape(1,-1)
    #hyperedge_index = edge_index[1].T.reshape(1,-1)

    hyperedge_index = torch.cat([edge_to_node_index, hyperedge_index], dim=0).long() 
    ### Transform batch of nodes to batch of edges
    #edge_batch = hyperedge_index[1,:].reshape(-1,2)[:,0]
    edge_batch = 0
    #edge_batch = torch.index_select(batch, 0, edge_batch)

    ### Add self-loops to each node in the dual hypergraph
    if add_loops:
        bincount =  hyperedge_index[1].bincount()
        mask = bincount[hyperedge_index[1]]!=1
        max_edge = hyperedge_index[1].max()
        loops = torch.cat([torch.arange(0,num_edge,1,device=device).view(1,-1), 
                            torch.arange(max_edge+1,max_edge+num_edge+1,1,device=device).view(1,-1)], 
                            dim=0)

        hyperedge_index = torch.cat([hyperedge_index[:,mask], loops], dim=1)

    return hyperedge_index, edge_batch

def load_hetero_dataset(dataset: str):
    ratio = [20, 40, 60]
    if dataset == "ACM":
        data = load_acm(ratio)
    elif dataset == "DBLP":
        data = load_dblp(ratio)
    elif dataset == "Aminer":
        data = load_aminer(ratio)
    elif dataset == "IMDB":
        data = load_imdb(ratio)
    elif dataset == "Yelp":
        data = load_yelp(ratio)



    hyperedge_dict = {}

    for k in data.x_dict:
        hyperedge_dict[k] = {}

    batch_dict = {}
    init_map = data.init_id_map
    for edge_type, edge_index in data.edge_index_dict.items():
        src, rel, dst = edge_type
        batch = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)

        e_type = src+dst
        #print(edge_type)
        #print(edge_index)
        process_edge_index = edge_index.clone()
        process_edge_index[0] = init_map[src][edge_index[0]]
        process_edge_index[1] = init_map[dst][edge_index[1]]
        #print(edge_index)

        hyperedge_dict[src][dst], batch_dict[e_type] = DHT(process_edge_index, batch, add_loops=True)#放在预处理

    data.hyperedge_dict =hyperedge_dict
    data.batch_dict = batch_dict
    
    # total_edge_index = data.adj_dict['total']
    # total_batch = torch.zeros(total_edge_index.size(1), dtype=torch.long, device=total_edge_index.device)
    # total_hyperedge, total_batch = DHT(total_edge_index, total_batch, add_loops=True)
    # data.total_hyperedge = total_hyperedge
    # data.total_batch = total_batch
    # data.total_edge_index = total_edge_index
    # print(total_edge_index)
    # print(total_hyperedge)
    target_type = [t for t in data.node_types if data[t].get('y') is not None][0]
    #data[target_type].x = perturb_features(data[target_type].x,0.9)
    data.num_classes = max(data[target_type].y)+1

    return data


def build_id_mapping(original_ids, target_ids):
    """
    Builds a mapping dictionary from original IDs to target IDs.

    :param original_ids: List of original IDs.
    :param target_ids: List of target IDs corresponding to the original IDs.
    :return: Dictionary mapping original IDs to target IDs.
    """
    if len(original_ids) != len(target_ids):
        raise ValueError("The length of original_ids and target_ids must be the same.")
    
    id_mapping = dict(zip(original_ids, target_ids))
    return id_mapping



def init_feat(node_type,dim,node_num):
    x_dict = {}
    for k in node_type:
        x_dict[k] = torch.FloatTensor(node_num[k], dim)
        torch.nn.init.xavier_uniform_(x_dict[k].data, gain=1.414)
        #x_dict[k] = scale_feats(x_dict[k])
    return x_dict


