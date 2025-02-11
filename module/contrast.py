import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.functional import normalize
from torch_geometric.nn import Linear
import random


EPS = 1e-15

class Contrast2(nn.Module):
    def __init__(self, hidden_dim, project_dim, batch_size, tau):
        super(Contrast2, self).__init__()
        self.tau = tau
        self.proj_1 = nn.Sequential(
            nn.Linear(hidden_dim, project_dim),
            nn.BatchNorm1d(project_dim),
            nn.ReLU(),
            nn.Linear(project_dim, project_dim)
        )
        self.proj_2 = nn.Sequential(
            nn.Linear(hidden_dim, project_dim),
            nn.BatchNorm1d(project_dim),
            nn.ReLU(),
            nn.Linear(project_dim, project_dim)
        )
        self.batch_size = batch_size


        for model in self.proj_1:
            if isinstance(model, nn.Linear):
                nn.init.kaiming_normal_(model.weight)
                #nn.init.xavier_uniform_(model.weight)
       

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t()) + EPS
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix
    def similarity(self, h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()
    def infonce(self, anchor, sample, pos, neg):
     
        sim = self.similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim) * neg
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob * pos
        loss = loss.sum(dim=1) / pos.sum(dim=1)
        return -loss.mean()
    def batch_nce_loss(self, z1, z2, pos_mask=None, neg_mask=None):
        if pos_mask is None and neg_mask is None:
            pos_mask = self.pos_mask
            neg_mask = self.neg_mask

        nnodes = z1.shape[0]
        if (self.batch_size == 0) or (self.batch_size > nnodes):
            loss_0 = self.infonce(z1, z2, pos_mask, neg_mask)
            loss_1 = self.infonce(z2, z1, pos_mask, neg_mask)
            loss = (loss_0 + loss_1) / 2.0
        else:
            node_idxs = list(range(nnodes))
            random.shuffle(node_idxs)
            #node_idxs = torch.randperm(z1.shape[0])

            batches = split_batch(node_idxs, self.batch_size)
            loss = 0
            for b in batches:
                weight = len(b) / nnodes
                loss_0 = self.infonce(z1[b], z2[b], pos_mask[:,b][b,:], neg_mask[:,b][b,:])
                loss_1 = self.infonce(z2[b], z1[b], pos_mask[:,b][b,:], neg_mask[:,b][b,:])
                loss += (loss_0 + loss_1) / 2.0 * weight
        return loss
   
    def forward(self, z_1, z_2, pos):
        #pos = torch.eye(len(z_1)).to_sparse().to(z_2.device)
        pos = pos.to_dense()
        neg = 1-pos

        z_proj_1 = self.proj_1(z_1)
        
        #upper
        nnodes = z_1.shape[0]

        if (self.batch_size == 0) or (self.batch_size > nnodes):
            loss_0 = self.infonce(z_proj_1, z_2, pos, neg)
            loss_1 = self.infonce(z_2, z_proj_1, pos, neg)
            loss = (loss_0 + loss_1) / 2.0
            
        else:
            node_idxs = list(range(nnodes))
            random.shuffle(node_idxs)
            #node_idxs = torch.randperm(z1.shape[0])

            batches = split_batch(node_idxs, self.batch_size)
            loss = 0
            for b in batches:
                weight = len(b) / nnodes
                loss_0 = self.infonce(z_proj_1[b], z_2[b], pos[:,b][b,:], neg[:,b][b,:])
                loss_1 = self.infonce(z_2[b], z_proj_1[b], pos[:,b][b,:], neg[:,b][b,:])
                loss += (loss_0 + loss_1) / 2.0 * weight
        return loss
        #lori_1 = self.infonce(z_proj_1, z_2, pos, neg)
        #lori_2 = self.infonce(z_2, z_proj_1, pos, neg)

        #return (lori_1 + lori_2) / 2
    """
    def forward(self, z_1, z_2, pos):
        #pos = torch.eye(len(z_1)).to_sparse().to(z_2.device)

        z_proj_1 = self.proj_1(z_1)
        
        
        #z_proj_2 = self.proj_2(z_2)
        z_proj_2 = z_2
        matrix_1 = self.sim(z_proj_1, z_proj_2)
        matrix_2 = matrix_1.t()


        matrix_1 = matrix_1 / (torch.sum(matrix_1, dim=1).view(-1, 1) + EPS)
        lori_1 = -torch.log(matrix_1.mul(pos.to_dense()).sum(dim=-1)).mean()

        matrix_2 = matrix_2 / (torch.sum(matrix_2, dim=1).view(-1, 1) + EPS)
        lori_2 = -torch.log(matrix_2.mul(pos.to_dense()).sum(dim=-1)).mean()


        return (lori_1 + lori_2) / 2
        """

def get_top_k(sim_l, k1):
    _, k_indices_pos = torch.topk(sim_l, k=k1, dim=1)

    source = torch.tensor(range(len(sim_l))).reshape(-1, 1).to(sim_l.device)
    k_source_l = source.repeat(1, k1).flatten()
   
    k_indices_pos = k_indices_pos.flatten()
    k_indices_pos = torch.stack((k_source_l, k_indices_pos), dim=0)
    kg_pos = torch.sparse_coo_tensor(k_indices_pos, torch.ones((len(k_indices_pos[0]))).to(sim_l.device), ([len(sim_l), len(sim_l)]))
    #kg_pos = sim_l.to_sparse()

    return kg_pos


def graph_construction(x, sim_l, k_pos):
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    zero_indices = torch.nonzero(x_norm.flatten() == 0)
    x_norm[zero_indices] += EPS



    dot_numerator = torch.mm(x, x.t())
    dot_denominator = torch.mm(x_norm, x_norm.t())
  
    fea_sim = dot_numerator / dot_denominator
   

    sim_l = fea_sim * sim_l
    #sim_l = fea_sim
    #sim_l =  sim_l

    if k_pos <= 0:
        pos = torch.eye(len(x)).to_sparse().to(x.device)
    else:
        pos= get_top_k(sim_l, k_pos)
        pos = (pos.to_dense() + torch.eye(len(x)).to(x.device)).to_sparse()
    return pos

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list