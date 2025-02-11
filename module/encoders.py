import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Node2Vec as n2v
from torch_geometric.nn import Sequential
from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv, HeteroConv, Sequential, Linear, HeteroDictLinear, HypergraphConv,SGConv, HGTConv
from torch_geometric.nn.module_dict import ModuleDict

from module.resolver import (activation_resolver, layer_resolver,
                            normalization_resolver)
from torch_geometric.utils import index_to_mask, degree, one_hot,remove_self_loops,add_self_loops
import math
from torch_scatter import scatter
import random
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import HGTConv, Linear

from torch_geometric.nn.models import JumpingKnowledge

def to_sparse_tensor(edge_index, num_nodes):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)






class HeteroGNNEncoder(nn.Module):
    def __init__(
        self,
        metadata,
        hidden_channels,
        out_channels=None,
        num_heads=4,
        num_layers=2,
        dropout=0.5,
        norm='batchnorm',
        layer="sage",
        activation="elu",
        add_last_act=True,
        add_last_bn=True,
    ):

        super().__init__()

        out_channels = out_channels or hidden_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers  
        self.add_last_act = add_last_act
        self.add_last_bn = add_last_bn
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.metadata = metadata

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)
        for i in range(num_layers):
            is_last_layer = i == num_layers - 1
            first_channels = -1 if i == 0 else hidden_channels
            second_channels = out_channels if is_last_layer else hidden_channels
            heads = 1 if i == num_layers - 1 or 'gat' not in layer else num_heads
            conv = HeteroConv({
                edge_type: layer_resolver(layer, first_channels,
                                  second_channels, heads)
                for edge_type in metadata[1]
            })#mean or sum
          
            if not is_last_layer or (is_last_layer and add_last_bn):
                norm_layer = nn.ModuleDict(
                    {
                        node_type: normalization_resolver(norm, second_channels*heads)
                        for node_type in metadata[0]
                    }
                )
                self.norms.append(norm_layer)
            if not is_last_layer or (is_last_layer and add_last_act):
                self.acts.append(activation_resolver(activation))                
            self.convs.append(conv)

        self.dropout = nn.Dropout(dropout)


        project = True
        if project:
            self.lin = nn.Sequential(Linear(-1, out_channels))  
                
                
        #self.proj = Linear(init_dim, hidden_channels,bias=False, weight_initializer="glorot")
        self.H_proj = {}


    def reset_parameters(self):
        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
                    
    def forward(self, x_dict, edge_index_dict):
        #x_dict = self.proj(x_dict)

        xs = [x_dict]
        #x_p = self.proj(x_dict['p'])
        h_dict ={}
        for k in x_dict:
            #h_dict[k] = self.lin_dict[k](self.dropout(x_dict[k]))
            h_dict[k] = self.dropout(x_dict[k])
            #h_dict[k] = x_dict[k]#ACM不用
        for i, conv in enumerate(self.convs):
            h_dict = conv(h_dict, edge_index_dict)

            h_dict = {
                key: self.dropout(self.acts[i](self.norms[i][key](x)))
                for key, x in h_dict.items()
            }

            #x_dict['p'] = self.hyperconv(x_dict['p'], edge_index_dict[('p','to','a')])
            xs.append(h_dict)
        
        
        #for k in x_dict:
        #    x_dict[k] = F.normalize(x_dict[k], dim=1)

        return xs

def x_to_dict(X,graph):
    h = {}
    for k in graph.x_dict:
        h[k] = X[graph.node_idx[k]]
    return h

from math import ceil

from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
class HR_encoder(nn.Module):#Heterogeneous Relation encoder
    def __init__(self,  node_type, edge_type_dict, num_edge_features, ehid, dropout):
        super().__init__()

        self.hyperconvs = nn.ModuleDict()
        self.scoreconvs = nn.ModuleDict()
        self.num_edge_features  = num_edge_features
        self.ehid = ehid
       
        self.edge_type_list = edge_type_dict
        self.dropout = dropout
        self.edge_ratio = 0.5
        self.bn = nn.ModuleDict()
        self.act = nn.ModuleDict()
        self.rel_act = nn.ModuleDict()
        self.rel_bn = nn.ModuleDict()
        for n_type in node_type:
            self.bn[n_type] = nn.BatchNorm1d(self.ehid)
            self.act[n_type] = torch.nn.ReLU()
        self.bn0 = nn.BatchNorm1d(num_edge_features//2)
        self.bn1 = nn.BatchNorm1d(self.ehid)
        self.bn2 = nn.BatchNorm1d(self.ehid//2)

        for edge_type in edge_type_dict:
            self.hyperconvs[edge_type] = HypergraphConv(self.num_edge_features, self.ehid)

            self.scoreconvs[edge_type] = HypergraphConv(self.ehid,1)
            self.rel_act[edge_type] = torch.nn.ReLU()
            self.rel_bn[edge_type] = torch.nn.BatchNorm1d(self.ehid)

 



    def forward(self,  x_dict, edge_index_dict, data, encoder, target_type):
  
        z = encoder(x_dict, edge_index_dict)
       
        x_dict = z[-1]


        for k in x_dict:
            x_dict[k] = self.bn[k](x_dict[k])
            x_dict[k] = self.act[k](x_dict[k])
        edge_attr_dict = {}
        hyperedge_dict = {}
        score_dict = {}
        for k in x_dict:
            edge_attr_dict[k] = {}#存储边特征
            hyperedge_dict[k] = {}
            score_dict[k] = {}




        #TODO 分类型

        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type

            #hyperedge_dict[src][dst] = self.DHT(edge_index, add_loops=True)#放在预处理
            hyperedge_dict[src][dst] = data.hyperedge_dict[src][dst]#放在预处理

            edge_attr_dict[src][dst] =  torch.cat((x_dict[src][edge_index[0, :]], x_dict[dst][edge_index[1, :]]), dim=1)

        edge_attr_list = []





        edge_h_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type
            e_type = src+dst
            edge_attr = edge_attr_dict[src][dst]
            hyperedge_index = hyperedge_dict[src][dst]
            edge_attr = self.hyperconvs[e_type](edge_attr, hyperedge_index)
            edge_attr = self.rel_bn[e_type](edge_attr)
            edge_attr = self.rel_act[e_type](edge_attr)
            edge_attr = F.dropout(edge_attr, self.dropout, training=self.training)


            edge_attr_list.append(edge_attr)
            edge_h_dict[edge_type] = edge_attr

            score_dict[src][dst] = gumbel_sampling( self.scoreconvs[e_type](edge_attr , hyperedge_index).squeeze() )#计算边得分


        init_edge_weight_dict = {}
        hetero_edge_weight_dict = {}

        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type
            score = score_dict[src][dst]
            init_edge_weight_dict[edge_type] = score
            hetero_edge_weight_dict[edge_type] = 1-score


        #full version
        adj_dict, multi_g = edge_dict_to_multi_g(data, edge_index_dict, init_edge_weight_dict, x_dict, target_type)
        _, hetero_multi_g = edge_dict_to_multi_g(data, edge_index_dict,hetero_edge_weight_dict, x_dict, target_type)

        return [multi_g,hetero_multi_g]
    def DHT(self, edge_index, add_loops=True):

        num_edge = edge_index.size(1)
        device = edge_index.device
        ### Transform edge list of the original graph to hyperedge list of the dual hypergraph
        edge_to_node_index = torch.arange(0,num_edge,1, device=device).repeat_interleave(2).view(1,-1)
        hyperedge_index = edge_index.T.reshape(1,-1)
        hyperedge_index = torch.cat([edge_to_node_index, hyperedge_index], dim=0).long() 
        ### Transform batch of nodes to batch of edges

        ### Add self-loops to each node in the dual hypergraph
        if add_loops:
            bincount =  hyperedge_index[1].bincount()
            mask = bincount[hyperedge_index[1]]!=1
            max_edge = hyperedge_index[1].max()
            loops = torch.cat([torch.arange(0,num_edge,1,device=device).view(1,-1), 
                                torch.arange(max_edge+1,max_edge+num_edge+1,1,device=device).view(1,-1)], 
                                dim=0)

            hyperedge_index = torch.cat([hyperedge_index[:,mask], loops], dim=1)

        return hyperedge_index



bias = 0.0001
temperature = 1.0
def gumbel_sampling(edges_weights_raw):
    eps = (bias - (1 - bias)) * torch.rand(edges_weights_raw.size()) + (1 - bias)
    gate_inputs = torch.log(eps) - torch.log(1 - eps)
    gate_inputs = gate_inputs.to(edges_weights_raw.device)
    gate_inputs = (gate_inputs + edges_weights_raw) / temperature
    return torch.sigmoid(gate_inputs).squeeze()


def edge_dict_to_multi_g(data, edge_index_dict, edge_weight_dict, x_dict, target_type):
    multi_g = {}
    sparse_edge_dict = {}
    adj_dict={}
    for k in x_dict:
        adj_dict[k] = {}
    for edge_type, cur_edge_index in edge_index_dict.items():
        src,_,dst = edge_type
        #if dst == target_type:
        row,col = cur_edge_index[0], cur_edge_index[1]
        #edge_weight = init_edge_weight_dict[edge_type]
        edge_weight = edge_weight_dict[edge_type]
        
        #edge_weight = torch.ones_like(edge_weight).to(edge_weight.device)

        sparse_edge_dict[edge_type] =  SparseTensor(row=row, col=col, value=edge_weight, sparse_sizes=(x_dict[src].shape[0], x_dict[dst].shape[0]))
        adj_dict[src][dst] = sparse_edge_dict[edge_type]
            #adj_dict[dst][src] =  SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(x_dict[dst].shape[0], x_dict[src].shape[0]))



    for src in adj_dict:
        for dst in adj_dict[src]:
            if src == target_type:
                g = adj_dict[src][dst].matmul(adj_dict[dst][src]).coo()
                row,col, edge_weight = g

                edge_index = torch.cat((row.view(1,-1),col.view(1, -1)), dim=0)


                edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x_dict[src].shape[0], False, True)
        
                g = torch.sparse_coo_tensor(edge_index, edge_weight,size=(x_dict[src].shape[0], x_dict[src].shape[0]))#SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, sparse_sizes=(x_dict[src].shape[0], x_dict[src].shape[0]))


                g_type = src + dst + src
                multi_g[g_type] = g
    return adj_dict, multi_g

