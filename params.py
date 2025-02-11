import argparse
import torch
import sys



def acm_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ACM",
                        help="Datasets. (default: DBLP)")
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for model and dataset. (default: 2024)')

    parser.add_argument("--encode_layer", default="sage",
                        help="GNN layer, (default: sage)")
                        
    parser.add_argument("--decode_layer", default="sage",
                        help="GNN layer, (default: sage)")
    parser.add_argument("--encoder_activation", default="relu",
                        help="Activation function for GNN encoder, (default: relu)")
    parser.add_argument('--encoder_channels', type=int, default=128,
                        help='Channels of hidden representation. (default: 512)')#聚类维度第一点
    parser.add_argument('--hidden_channels', type=int, default=256,
                        help='Channels of hidden representation. (default: 512)')
    parser.add_argument('--encoder_layers', type=int, default=2,
                        help='Number of layers for encoder. (default: 2)')
    parser.add_argument('--encoder_dropout', type=float, default=0.2,#关键参数
                        help='Dropout probability of encoder. (default: 0.5)')
    parser.add_argument("--encoder_norm",default="batchnorm", help="Normalization (default: batchnorm)")

    parser.add_argument('--decoder_channels', type=int, default=32,
                        help='Channels of decoder layers. (default: 32)')
    parser.add_argument('--decoder_layers', type=int, default=2,
                        help='Number of layers for decoders. (default: 2)')
    parser.add_argument('--decoder_dropout', type=float, default=0.2,
                        help='Dropout probability of decoder. (default: 0.2)')
    parser.add_argument("--decoder_norm",
                        default="none", help="Normalization (default: none)")

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for training. (default: 0.0005)')
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='weight_decay for link prediction training. (default: 5e-5)')
    parser.add_argument('--grad_norm', type=float, default=1.0,
                        help='grad_norm for training. (default: 1.0.)')

    parser.add_argument('--p', type=float, default=0.2,
                        help='Mask ratio or sample ratio for HeteroMaskEdge')

    parser.add_argument("--mode", default="cat",
                        help="Embedding mode `last` or `cat` (default: `cat`)")
    parser.add_argument('--l2_normalize', action='store_true',
                        help='Whether to use l2 normalize output embedding. (default: False)')
    parser.add_argument('--nodeclas_lr', type=float, default=0.01,
                        help='Learning rate for training. (default: 0.01)')
    parser.add_argument('--nodeclas_weight_decay', type=float, default=1e-3,
                        help='weight_decay for node classification training. (default: 1e-3)')

    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs. (default: 500)')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs. (default: 1)')
    parser.add_argument('--eval_steps', type=int, default=50, help='(default: 50)')

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument('--hyper_dropout', type=float, default=0.2,
                    help='Dropout probability of hyperedge. (default: 0.2)')
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--g_type', type=str, default=['pap','psp'])

    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--homo_layer', type=int, default=2)
    parser.add_argument('--n_pos', type=int, default=0)
    parser.add_argument('--homo_drop', type=float, default=0.5)

    args = parser.parse_args()

    return args

def aminer_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Aminer",
                        help="Datasets. (default: DBLP)")
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for model and dataset. (default: 2024)')

    parser.add_argument("--encode_layer", default="sage",
                        help="GNN layer, (default: sage)")
                        
    parser.add_argument("--decode_layer", default="gat",
                        help="GNN layer, (default: sage)")
    parser.add_argument("--encoder_activation", default="relu",
                        help="Activation function for GNN encoder, (default: relu)")
    parser.add_argument('--encoder_channels', type=int, default=512,
                        help='Channels of hidden representation. (default: 256)')#IMDB:1024
    parser.add_argument('--hidden_channels', type=int, default=512,
                        help='Channels of hidden representation. (default: 256)')
    parser.add_argument('--encoder_layers', type=int, default=2,
                        help='Number of layers for encoder. (default: 2)')
    parser.add_argument('--encoder_dropout', type=float, default=0.5,#关键参数
                        help='Dropout probability of encoder. (default: 0.5)')
    parser.add_argument("--encoder_norm",default="batchnorm", help="Normalization (default: batchnorm)")

    parser.add_argument('--decoder_channels', type=int, default=32,
                        help='Channels of decoder layers. (default: 32)')
    parser.add_argument('--decoder_layers', type=int, default=2,
                        help='Number of layers for decoders. (default: 2)')
    parser.add_argument('--decoder_dropout', type=float, default=0.2,
                        help='Dropout probability of decoder. (default: 0.2)')
    parser.add_argument("--decoder_norm",
                        default="none", help="Normalization (default: none)")

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for training. (default: 0.005)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight_decay for link prediction training. (default: 5e-5)')
    parser.add_argument('--grad_norm', type=float, default=1.0,
                        help='grad_norm for training. (default: 1.0.)')

    parser.add_argument('--p', type=float, default=0.2,
                        help='Mask ratio or sample ratio for HeteroMaskEdge')

    parser.add_argument("--mode", default="cat",
                        help="Embedding mode `last` or `cat` (default: `cat`)")
    parser.add_argument('--l2_normalize', action='store_true',
                        help='Whether to use l2 normalize output embedding. (default: False)')
    parser.add_argument('--nodeclas_lr', type=float, default=0.01,
                        help='Learning rate for training. (default: 0.01)')
    parser.add_argument('--nodeclas_weight_decay', type=float, default=1e-3,
                        help='weight_decay for node classification training. (default: 1e-3)')

    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs. (default: 500)')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs. (default: 1)')
    parser.add_argument('--eval_steps', type=int, default=50, help='(default: 50)')

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument('--hyper_dropout', type=float, default=0.2,
                    help='Dropout probability of hyperedge. (default: 0.2)')
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--g_type', type=str, default=['pap','prp'])

    parser.add_argument('--tau', type=float, default=0.8)#0.7
    parser.add_argument('--homo_layer', type=int, default=2)
    parser.add_argument('--n_pos', type=int, default=2)
    parser.add_argument('--homo_drop', type=float, default=0.0)
    args = parser.parse_args()

    return args

def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="DBLP",
                        help="Datasets. (default: DBLP)")
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for model and dataset. (default: 2024)')

    parser.add_argument("--encode_layer", default="sage",
                        help="GNN layer, (default: sage)")
                        
    parser.add_argument("--decode_layer", default="gat",
                        help="GNN layer, (default: sage)")
    parser.add_argument("--encoder_activation", default="relu",
                        help="Activation function for GNN encoder, (default: relu)")
    parser.add_argument('--encoder_channels', type=int, default=512,
                        help='Channels of hidden representation. (default: 512)')#IMDB:1024
    parser.add_argument('--hidden_channels', type=int, default=256,
                        help='Channels of hidden representation. (default:256)')
    parser.add_argument('--encoder_layers', type=int, default=2,
                        help='Number of layers for encoder. (default: 2)')
    parser.add_argument('--encoder_dropout', type=float, default=0.2,#关键参数
                        help='Dropout probability of encoder. (default: 0.5)')
    parser.add_argument("--encoder_norm",default="batchnorm", help="Normalization (default: batchnorm)")

    parser.add_argument('--decoder_channels', type=int, default=32,
                        help='Channels of decoder layers. (default: 32)')
    parser.add_argument('--decoder_layers', type=int, default=2,
                        help='Number of layers for decoders. (default: 2)')
    parser.add_argument('--decoder_dropout', type=float, default=0.5,
                        help='Dropout probability of decoder. (default: 0.2)')
    parser.add_argument("--decoder_norm",
                        default="none", help="Normalization (default: none)")

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for training. (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight_decay for link prediction training. (default: 5e-5)')
    parser.add_argument('--grad_norm', type=float, default=1.0,
                        help='grad_norm for training. (default: 1.0.)')

    parser.add_argument('--p', type=float, default=0.2,
                        help='Mask ratio or sample ratio for HeteroMaskEdge')

    parser.add_argument("--mode", default="cat",
                        help="Embedding mode `last` or `cat` (default: `cat`)")
    parser.add_argument('--l2_normalize', action='store_true',
                        help='Whether to use l2 normalize output embedding. (default: False)')
    parser.add_argument('--nodeclas_lr', type=float, default=0.01,
                        help='Learning rate for training. (default: 0.01)')
    parser.add_argument('--nodeclas_weight_decay', type=float, default=1e-3,
                        help='weight_decay for node classification training. (default: 1e-3)')

    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs. (default: 500)')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs. (default: 1)')
    parser.add_argument('--eval_steps', type=int, default=50, help='(default: 50)')

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument('--hyper_dropout', type=float, default=0.2,
                    help='Dropout probability of hyperedge. (default: 0.2)')
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--g_type', type=str, default=['apa'])
    #parser.add_argument('--g_type', type=str, default=['pap','pcp','ptp'])

    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--homo_layer', type=int, default=2)
    parser.add_argument('--n_pos', type=int, default=2)
    parser.add_argument('--homo_drop', type=float, default=0.5)


    args = parser.parse_args()

    return args



def imdb_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="IMDB",
                        help="Datasets. (default: DBLP)")
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for model and dataset. (default: 2024)')

    parser.add_argument("--encode_layer", default="sage",
                        help="GNN layer, (default: sage)")
                        
    parser.add_argument("--decode_layer", default="gat",
                        help="GNN layer, (default: sage)")
    parser.add_argument("--encoder_activation", default="elu",
                        help="Activation function for GNN encoder, (default: relu)")
    parser.add_argument('--encoder_channels', type=int, default=512,
                        help='Channels of hidden representation. (default: 512)')#IMDB:1024
    parser.add_argument('--hidden_channels', type=int, default=512,
                        help='Channels of hidden representation. (default: 512)')
    parser.add_argument('--encoder_layers', type=int, default=2,
                        help='Number of layers for encoder. (default: 2)')
    parser.add_argument('--encoder_dropout', type=float, default=0.2,#关键参数
                        help='Dropout probability of encoder. (default: 0.5)')
    parser.add_argument("--encoder_norm",default="batchnorm", help="Normalization (default: batchnorm)")

    parser.add_argument('--decoder_channels', type=int, default=32,
                        help='Channels of decoder layers. (default: 32)')
    parser.add_argument('--decoder_layers', type=int, default=2,
                        help='Number of layers for decoders. (default: 2)')
    parser.add_argument('--decoder_dropout', type=float, default=0.2,
                        help='Dropout probability of decoder. (default: 0.2)')
    parser.add_argument("--decoder_norm",
                        default="none", help="Normalization (default: none)")

    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate for training. (default: 0.001)')#0.0005
    parser.add_argument('--weight_decay', type=float, default=5e-5,
                        help='weight_decay for link prediction training. (default: 1e-4)')#5e-5
    parser.add_argument('--grad_norm', type=float, default=1.0,
                        help='grad_norm for training. (default: 1.0.)')

    parser.add_argument('--p', type=float, default=0.2,
                        help='Mask ratio or sample ratio for HeteroMaskEdge')

    parser.add_argument("--mode", default="cat",
                        help="Embedding mode `last` or `cat` (default: `cat`)")
    parser.add_argument('--l2_normalize', action='store_true',
                        help='Whether to use l2 normalize output embedding. (default: False)')
    parser.add_argument('--nodeclas_lr', type=float, default=0.01,
                        help='Learning rate for training. (default: 0.01)')
    parser.add_argument('--nodeclas_weight_decay', type=float, default=1e-3,
                        help='weight_decay for node classification training. (default: 1e-3)')

    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs. (default: 500)')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs. (default: 1)')
    parser.add_argument('--eval_steps', type=int, default=50, help='(default: 50)')

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument('--hyper_dropout', type=float, default=0.2,
                    help='Dropout probability of hyperedge. (default: 0.2)')
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)#10?
    parser.add_argument('--tau', type=float, default=0.4)
    parser.add_argument('--g_type', type=str, default=['mam','mdm'])
    parser.add_argument('--homo_layer', type=int, default=2)
    parser.add_argument('--n_pos', type=int, default=2)
    parser.add_argument('--homo_drop', type=float, default=0.5)

    args = parser.parse_args()
    return args




def yelp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Yelp",
                        help="Datasets. (default: DBLP)")
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for model and dataset. (default: 2024)')

    parser.add_argument("--encode_layer", default="sage",
                        help="GNN layer, (default: sage)")
                        
    parser.add_argument("--decode_layer", default="gat",
                        help="GNN layer, (default: sage)")
    parser.add_argument("--encoder_activation", default="relu",
                        help="Activation function for GNN encoder, (default: relu)")
    parser.add_argument('--encoder_channels', type=int, default=512,
                        help='Channels of hidden representation. (default: 128)')#IMDB:1024
    parser.add_argument('--hidden_channels', type=int, default=512,
                        help='Channels of hidden representation. (default: 512)')
    parser.add_argument('--encoder_layers', type=int, default=2,
                        help='Number of layers for encoder. (default: 2)')
    parser.add_argument('--encoder_dropout', type=float, default=0.2,#关键参数
                        help='Dropout probability of encoder. (default: 0.5)')
    parser.add_argument("--encoder_norm",default="batchnorm", help="Normalization (default: batchnorm)")

    parser.add_argument('--decoder_channels', type=int, default=32,
                        help='Channels of decoder layers. (default: 32)')
    parser.add_argument('--decoder_layers', type=int, default=2,
                        help='Number of layers for decoders. (default: 2)')
    parser.add_argument('--decoder_dropout', type=float, default=0.2,
                        help='Dropout probability of decoder. (default: 0.2)')
    parser.add_argument("--decoder_norm",
                        default="none", help="Normalization (default: none)")

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for training. (default: 0.0005)')#classification:0.001,cluster:0.0005
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight_decay for link prediction training. (default: 5e-5)')#classification:5e-5,cluster:5e-4
    parser.add_argument('--grad_norm', type=float, default=1.0,
                        help='grad_norm for training. (default: 1.0.)')

    parser.add_argument('--p', type=float, default=0.2,
                        help='Mask ratio or sample ratio for HeteroMaskEdge')

    parser.add_argument("--mode", default="cat",
                        help="Embedding mode `last` or `cat` (default: `cat`)")
    parser.add_argument('--l2_normalize', action='store_true',
                        help='Whether to use l2 normalize output embedding. (default: False)')
    parser.add_argument('--nodeclas_lr', type=float, default=0.01,
                        help='Learning rate for training. (default: 0.01)')
    parser.add_argument('--nodeclas_weight_decay', type=float, default=1e-3,
                        help='weight_decay for node classification training. (default: 1e-3)')

    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs. (default: 500)')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs. (default: 1)')
    parser.add_argument('--eval_steps', type=int, default=50, help='(default: 50)')

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument('--hyper_dropout', type=float, default=0.2,
                    help='Dropout probability of hyperedge. (default: 0.2)')
    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--g_type', type=str, default=['bsb','bub','blb'])
    parser.add_argument('--homo_layer', type=int, default=1)
    parser.add_argument('--n_pos', type=int, default=0)
    parser.add_argument('--homo_drop', type=float, default=0.5)

    args = parser.parse_args()
    return args




def set_params(dataset):
    if dataset == "ACM":
        args = acm_params()
    elif dataset == "DBLP":
        args = dblp_params()
    elif dataset == 'Yelp':
        args = yelp_params()
    elif dataset == 'IMDB':
        args = imdb_params()
    elif dataset == 'Aminer':
        args = aminer_params()

    
    return args

