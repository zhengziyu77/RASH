from typing import Any, Optional, Union

import torch
from torch import nn, Tensor
from torch_geometric.nn import (GATConv, GATv2Conv, GCNConv, GINConv, Linear,
                                SAGEConv, GraphConv)
from torch_geometric.resolver import resolver
#from module.HGNN import GAT

def swish(x: Tensor) -> Tensor:
    return x * x.sigmoid()

def activation_resolver(query: Optional[Union[Any, str]] = 'relu', *args, **kwargs):
    if query is None or query == 'none':
        return torch.nn.Identity()
    base_cls = torch.nn.Module
    base_cls_repr = 'Act'
    acts = [
        act for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, base_cls)
    ]
    acts += [
        swish,
    ]
    act_dict = {}
    return resolver(acts, act_dict, query, base_cls, base_cls_repr, *args,
                    **kwargs)
    
def normalization_resolver(query: Optional[Union[Any, str]], *args, **kwargs):
    if query is None or query == 'none':
        return torch.nn.Identity()    
    import torch_geometric.nn.norm as norm
    base_cls = torch.nn.Module
    base_cls_repr = 'Norm'
    norms = [
        norm for norm in vars(norm).values()
        if isinstance(norm, type) and issubclass(norm, base_cls)
    ]
    norm_dict = {}
    return resolver(norms, norm_dict, query, base_cls, base_cls_repr, *args,
                    **kwargs)





def layer_resolver(name, first_channels, second_channels, heads=1):
    if name == "sage":
        layer = SAGEConv(first_channels, second_channels)
    elif name == "gcn":
        #layer = GCNConv(first_channels, second_channels)
        layer = GraphConv(first_channels, second_channels)

    elif name == "gin":
        layer = GINConv(nn.Sequential(Linear(first_channels, second_channels), 
                                      nn.LayerNorm(second_channels),
                                      nn.PReLU(),
                                      Linear(second_channels, second_channels),
                                      # nn.LayerNorm(second_channels),
                                     ), train_eps=True)
    elif name == "gat":
        layer = GATConv(-1, second_channels, heads=heads)
        #layer = GAT(first_channels, second_channels)
    elif name == "gat2":
        layer = GATv2Conv(-1, second_channels, heads=heads)
    elif name == 'linear':
        layer = Linear(first_channels, second_channels)
    else:
        raise ValueError(name)
    return layer


from torch_geometric.utils import softmax
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import SAGEConv, Sequential, Linear
from torch_geometric.nn.conv import MessagePassing




            
class GAT(MessagePassing):
    def __init__(self, in_channels, dropout, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.dropout = dropout

        self.att_src = Parameter(torch.Tensor(1, in_channels))
        self.att_dst = Parameter(torch.Tensor(1, in_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(in_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x, edge_index):
        x_src, x_dst = x

        alpha_src = (x_src * self.att_src).sum(-1)
        alpha_dst = (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=None)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)