from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree,to_undirected,shuffle_node, get_laplacian
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import HypergraphConv, Linear, GCNConv, GATConv,SAGEConv
import copy
from itertools import permutations
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_scatter import scatter
from .contrast import *

class SGC(nn.Module):
    def __init__(self, nlayers, in_dim, emb_dim, dropout):
        super(SGC, self).__init__()
        self.dropout = dropout
        self.linear = Linear(in_dim, emb_dim, bias=True, weight_initializer='glorot')
        self.k = nlayers

    def forward(self, x, g):
        x = F.dropout(x, p = self.dropout,training=self.training)
        x = torch.relu(self.linear(x))
        for _ in range(self.k):
            x = g.matmul(x)
        return x
class RASH(nn.Module):
    def __init__(
        self,
        args,
        encoder,
        encoder2,
        HR_encoder,
        slove_node = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder2 = encoder2
        self.hr_encoder = HR_encoder
        self.args = args


        self.node_type = slove_node




        self.tau = args.tau

        self.gcn_Encoder = nn.ModuleDict()
        self.heter_Encoder = nn.ModuleDict()
        self.project = nn.ModuleDict()
        self.CL_loss = nn.ModuleDict()
        self.homo_proj = nn.ModuleDict()
        self.heter_proj = nn.ModuleDict()

        g_type = args.g_type
        self.homo_layer = self.heter_layer = args.homo_layer
        for i in g_type:
            self.gcn_Encoder[i] = Linear(-1, self.encoder.out_channels,bias=False,weight_initializer='kaiming_uniform')
            self.heter_Encoder[i] = Linear(-1,self.encoder.out_channels,bias=False,weight_initializer='kaiming_uniform')

            self.homo_proj[i] = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.out_channels, self.encoder.out_channels, bias=True),
            torch.nn.BatchNorm1d(self.encoder.out_channels),
            torch.nn.ReLU(),
            #torch.nn.ELU(),
            torch.nn.Linear(self.encoder.out_channels, self.encoder.out_channels, bias=True),
            )
            self.heter_proj[i] = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.out_channels, self.encoder.out_channels, bias=True),
            torch.nn.BatchNorm1d(self.encoder.out_channels),
            torch.nn.ReLU(),
            #torch.nn.ELU(),
            torch.nn.Linear(self.encoder.out_channels, self.encoder.out_channels, bias=True),
            )


        self.contrast_l = Contrast2(self.encoder.out_channels,self.encoder.out_channels, 0,self.tau)


        self.act = torch.nn.ReLU()

        self.bn = nn.BatchNorm1d(self.encoder.out_channels)
        self.pos = args.n_pos
        self.homo_drop = nn.Dropout(args.homo_drop)
        self.target_proj = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.out_channels, self.encoder.out_channels, bias=True),
            torch.nn.BatchNorm1d(self.encoder.out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(self.encoder.out_channels, self.encoder.out_channels, bias=True),
            )

        


    
    def reset_parameters(self):
        self.encoder.reset_parameters()


    def forward(self, x_dict, edge_index_dict, graph,**kwargs):
        target_type = [t for t in graph.node_types if graph[t].get('y') is not None][0]

        #drop_edge_index_dict, remain_edge_index_dict, edge_weight_dict, adj_dict, multip_g = self.hyper_drop(1,1500, x_dict, edge_index_dict, graph, self.encoder2,target_type)
        z = self.encoder(x_dict, edge_index_dict, **kwargs)
    


     
        #z[-1][target_type] += z2
      
        #z = self.encoder(x, edge_index, **kwargs)
        return z
    def train_step(self, args, cur_epoch, graph: Union[Data, HeteroData], alpha: float = 0.) -> torch.Tensor:
        return self.train_step_hetero(args, cur_epoch,graph)

    def train_step_hetero(self, args, cur_epoch, graph: HeteroData) -> torch.Tensor:

        x_dict,edge_index_dict = graph.x_dict, graph.edge_index_dict

        init_x_dict = {}
        target_type = [t for t in graph.node_types if graph[t].get('y') is not None][0]

        for k in x_dict:
            init_x_dict[k] = x_dict[k].clone()



        multip_g = self.hr_encoder(init_x_dict, edge_index_dict, graph, self.encoder2,target_type)

        z = self.encoder(init_x_dict, edge_index_dict)

        z1 = z[-1]


        homo_mg, heter_mg = multip_g[0], multip_g[1]


    
        homo_list = []
        heter_list = []
        dis_loss = 0
        cl_loss= 0

        sub_h = {}
        homo_loss =[]
        heter_loss = []
        target_feat = init_x_dict[target_type]
        for i, g_type in enumerate(heter_mg):
            sub_h[g_type] = []
            if g_type[0] == target_type:
                #同配

                homo = target_feat.clone()
                for i in range(self.homo_layer):
                    homo = homo_mg[g_type].matmul(self.homo_drop(homo))

                homo = self.gcn_Encoder[g_type](homo)
                homo = self.act(homo)
                homo =self.bn(homo)

                sim_l = homo_mg[g_type].to_dense()
                pos = graph_construction(homo, sim_l, k_pos=self.pos)
                homo_list.append(homo)
                homo = self.homo_proj[g_type](homo)
                homo_cl =  self.contrast_l(z1[target_type], homo,pos)
                homo_loss.append(homo_cl)


             
                #异配
                edge_index = heter_mg[g_type].coalesce().indices()
                edge_weight = heter_mg[g_type].coalesce().values()
                edge_index, edge_weight = get_laplacian(edge_index=edge_index, edge_weight= edge_weight, normalization='sym')
                heterg = torch.sparse_coo_tensor(edge_index, edge_weight,size=(x_dict[target_type].shape[0], x_dict[target_type].shape[0]))#SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, sparse_sizes=(x_dict[src].shape[0], x_dict[src].shape[0]))
                heter = target_feat.clone()
            
                for i in range(self.heter_layer):
                    heter = heterg.matmul(self.homo_drop(heter))
                heter = self.heter_Encoder[g_type](heter)
                heter = self.act(heter)
                heter = self.bn(heter)

                pos = graph_construction(heter, sim_l, k_pos=self.pos)

                heter_list.append(heter)

                heter = self.heter_proj[g_type](heter)
                heter_cl = self.contrast_l(z1[target_type], heter, pos)

                heter_loss.append(heter_cl)



        total_loss = homo_loss + heter_loss
        for i in range(len(total_loss)):
            cl_loss += (total_loss[i] )#.reshape(-1,1)

        return cl_loss
