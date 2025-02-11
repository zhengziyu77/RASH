import argparse
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch_geometric.transforms as T
import datetime
import torch.nn.functional as F

# custom modules
from module.hetero_dataset import load_hetero_dataset
from module.encoders import HeteroGNNEncoder,  HR_encoder
from module.rash import RASH
from module.utils import set_seed
from evaluate import evaluate,evaluate_cluster,run_kmeans
from params import set_params
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="Yelp",
                    help="Datasets. (default: DBLP)")
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed for model and dataset. (default: 2024)')

parser.add_argument("--encode_layer", default="sage",
                    help="GNN layer, (default: sage)")

parser.add_argument("--encoder_activation", default="prelu",
                    help="Activation function for GNN encoder, (default: relu)")
parser.add_argument('--encoder_channels', type=int, default=128,
                    help='Channels of hidden representation. (default: 256)')#IMDB:1024
parser.add_argument('--hidden_channels', type=int, default=512,
                    help='Channels of hidden representation. (default: 256)')
parser.add_argument('--encoder_layers', type=int, default=2,
                    help='Number of layers for encoder. (default: 2)')
parser.add_argument('--encoder_dropout', type=float, default=0.2,#关键参数
                    help='Dropout probability of encoder. (default: 0.5)')
parser.add_argument("--encoder_norm",default="batchnorm", help="Normalization (default: batchnorm)")



parser.add_argument('--lr', type=float, default=0.0005,
                    help='Learning rate for training. (default: 0.005)')
parser.add_argument('--weight_decay', type=float, default=5e-5,
                    help='weight_decay for link prediction training. (default: 5e-5)')
parser.add_argument('--grad_norm', type=float, default=1.0,
                    help='grad_norm for training. (default: 1.0.)')

parser.add_argument('--nodeclas_lr', type=float, default=0.01,
                    help='Learning rate for training. (default: 0.01)')
parser.add_argument('--nodeclas_weight_decay', type=float, default=1e-3,
                    help='weight_decay for node classification training. (default: 1e-3)')

parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of training epochs. (default: 500)')
parser.add_argument('--runs', type=int, default=1,
                    help='Number of runs. (default: 1)')
parser.add_argument('--eval_steps', type=int, default=50, help='(default: 50)')


parser.add_argument('--hyper_dropout', type=float, default=0.2,
                    help='Dropout probability of hyperedge. (default: 0.2)')
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--tau', type=float, default=0.8)
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()

#set_seed(args.seed)


def train(args,device):

    set_seed(args.seed)



    data = load_hetero_dataset( args.dataset).to(device)
    #print(data)
    #stop
    edge_type_list = []
    for edge_type in data.edge_index_dict:
        src,rel,dst = edge_type
        edge_type_list.append(src+dst)
    print(edge_type_list)
    node_type = [t for t in data.node_types if data[t].get('y') is not None][0]
    target_type = node_type

    out = data.x_dict[target_type].shape[1]
    encoder = HeteroGNNEncoder(data.metadata(),
                            hidden_channels=args.hidden_channels,
                            out_channels=args.encoder_channels,
                            num_layers=args.encoder_layers,
                            dropout=args.encoder_dropout,
                            norm=args.encoder_norm,
                            layer=args.encode_layer,
                            activation=args.encoder_activation)

    import copy
    encoder2 = copy.deepcopy(encoder)


    input_dict ={}
    slove_node = []
    for k in data.x_dict:
        input_dict[k] = data.x_dict[k].shape[1]
        #if k!=target_type:
        slove_node.append(k)

    #ie-hgcn init
    input_layer_shape = dict([(k, args.encoder_channels,) for k in data.x_dict.keys()])
    out_layer_shape = dict.fromkeys(data.x_dict.keys(), out)
    input_layer_shape = dict([(k, data.x_dict[k].shape[1],) for k in data.x_dict.keys()])
    out_layer_shape = dict.fromkeys(data.x_dict.keys(), args.encoder_channels)
    # Model and optimizer

    adj_dict={}
    for k in data.x_dict:
        adj_dict[k] = {}
    for edge_type, cur_edge_index in data.edge_index_dict.items():
        src,_,dst = edge_type
        row,col = cur_edge_index[0], cur_edge_index[1]
        #edge_weight = edge_weight_dict[edge_type]
        #sparse_edge_dict[edge_type] =  SparseTensor(row=row, col=col, value=edge_weight, sparse_sizes=(x_dict[src].shape[0], x_dict[dst].shape[0]))
        adj_dict[src][dst] = None
    hr_encoder = HR_encoder(node_type = data.node_types, edge_type_dict = edge_type_list , num_edge_features=args.encoder_channels*2, ehid = args.encoder_channels,dropout=args.hyper_dropout)
    model = RASH(args, encoder, encoder2, hr_encoder, slove_node=slove_node).to(device)

    best_metric = None
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay)

    scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / args.epochs) ) * 0.5

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler) 

    pbar = tqdm(range(1, 1 + args.epochs))
    
    cnt_wait = 0
    best = 1e9
    period = 100
    best_epoch=0
    auc_list = {'0':[], '1':[], '2':[]}
    ma_list = {'0':[], '1':[], '2':[]}
    mi_list = {'0':[], '1':[], '2':[]}
    nmi_list = []
    ari_list = []

    starttime = datetime.datetime.now()
    ratio = [20, 40, 60]

    for epoch in pbar:
        model =model.to(device)
        data = data.to(device)
        

        optimizer.zero_grad()
        model.train()
        loss = model.train_step(args, epoch, data)
        loss.backward()
        if args.grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()
        scheduler.step()
        pbar.set_description(f'Loss: {loss.item():.4f}')
        
        """
        if epoch % args.eval_steps == 0:
            model.eval()
            embedding_dict = model(data.x_dict, data.edge_index_dict, data)[-1]

            embeddings = embedding_dict[target_type].detach()
            for i in range(len(data[target_type].train_mask)):
                evaluate(F.normalize(embeddings,p=2,dim=1), data[target_type].train_mask[i], data[target_type].val_mask[i], data[target_type].test_mask[i], data[target_type].y.squeeze().to(embeddings.device), data.num_classes, device, args.nodeclas_lr, args.nodeclas_weight_decay, args.dataset, ratio[i], epoch, isTest=True)
        
        """
   
        if best > loss.item():
            best = loss.item()
            cnt_wait = 0
            best_epoch = epoch
            torch.save(model.state_dict(), './checkpoint/'+args.dataset+'/best_'+str(args.seed)+str(args.device)+'.pth')
        else:
            cnt_wait += 1
        if cnt_wait >= args.patience:
            break


    model.load_state_dict(torch.load('./checkpoint/'+args.dataset+'/best_'+str(args.seed)+str(args.device)+'.pth'))
    model = model.to(device)
    epoch = best_epoch
    print("---------------------------------------------------")
    model.eval()
    embedding_dict = model(data.x_dict, data.edge_index_dict, data)[-1]

    embeddings = embedding_dict[target_type].detach()
    label = data[target_type].y.squeeze()

    nmi,ari = evaluate_cluster(F.normalize(embeddings.cpu(),p=2,dim=1), label.cpu(), int(data.num_classes.cpu().numpy()),0)
    print(nmi,ari)
    nmi, ari = run_kmeans(F.normalize(embeddings.cpu(),p=2,dim=1), label.cpu(), int(data.num_classes.cpu().numpy()), starttime, args.dataset, epoch + 1)
    nmi_list.append(nmi)
    ari_list.append(ari)
    for i in range(len(data[target_type].train_mask)):
        evaluate(F.normalize(embeddings,p=2,dim=1), data[target_type].train_mask[i], data[target_type].val_mask[i], data[target_type].test_mask[i], data[target_type].y.squeeze().to(embeddings.device), data.num_classes, device, args.nodeclas_lr, args.nodeclas_weight_decay, args.dataset, ratio[i], epoch, isTest=True)
        #evaluate(embeddings, data[target_type].train_mask[i], data[target_type].val_mask[i], data[target_type].test_mask[i], data[target_type].y.squeeze().to(embeddings.device), data.num_classes, device, args.nodeclas_lr, args.nodeclas_weight_decay, args.dataset, ratio[i], epoch, isTest=True)
    for i in range(len(ratio)):
        f = open("result_" + args.dataset + str(ratio[i]) + ".txt", "a")
        f.write(str(args) +  "\n")
        f.close()


if __name__ == '__main__':

    args = parser.parse_args()

    args = set_params(args.dataset)
    print(args)
    if args.device < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    train(args,device)
