import math
import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.io as sio
import random
import torch
import dgl
import torch.nn.functional as F


def generate_rwr_subgraph(dgl_graph, node, subgraph_size, restart_prob):
    reduced_size = subgraph_size - 1
    sampling_nodes = dgl.sampling.random_walk(
        dgl_graph, [node], length=subgraph_size*3, restart_prob=restart_prob)[0]
    
    sampling_nodes, inverse = torch.unique(sampling_nodes[sampling_nodes != -1], return_inverse=True)
    
    sampling_nodes[0], sampling_nodes[inverse[0]] = sampling_nodes[inverse[0]].item(), sampling_nodes[0].item()

    retry_time = 0
    while len(sampling_nodes) < reduced_size:
        cur_trace = dgl.sampling.random_walk(
            dgl_graph, [node], length=subgraph_size*5, restart_prob=restart_prob/2)[0]
        cur_trace, inverse = torch.unique(torch.cat([sampling_nodes, cur_trace[0][cur_trace[0] != -1]]), return_inverse=True)
        cur_trace[0], cur_trace[inverse[0]] = cur_trace[inverse[0]].item(), cur_trace[0].item()
        sampling_nodes = cur_trace
        retry_time += 1
        if retry_time > 5:
            break
            #subv[i] = (subv[i] * reduced_size)
    sampling_nodes = sampling_nodes[:reduced_size]
    subg = dgl_graph.subgraph(sampling_nodes)
    original_ids = subg.ndata[dgl.NID]
    deg = subg.in_degrees().float()
    hat_deg = deg + 1
    deg = deg.clamp(min=1)
    norm = torch.pow(deg, -0.5)
    subg.ndata['d'] = norm
    subg.ndata['hat_d'] = hat_deg
    subg.ndata['center'][(original_ids == node).nonzero()[0]] = True

           
    return subg


def perturb_labels(labels, p):
    # 独立翻转每个值
    flip_mask = (torch.rand(labels.shape) < p).to(labels.device)
    perturbed_labels = labels.clone()
    perturbed_labels = torch.where(flip_mask, 1 - perturbed_labels, perturbed_labels)
    return perturbed_labels, flip_mask

def output_cal(X, A, recon, center, num_nodes_per_subgraph, tau):
    eps = 1e-7
    low_X, high_X, low_A, high_A = recon['low_X'], recon['high_X'], recon['low_A'], recon['high_A']
        # 计算每个样本的欧几里得距离并取倒数
    dist_low_X = (torch.norm(X - low_X, p=2, dim=1) / X.shape[1])[center] + eps
    dist_high_X = (torch.norm(X - high_X, p=2, dim=1) / X.shape[1])[center] + eps
    low_A = (low_A + low_A.T) / 2
    high_A = (high_A + high_A.T) / 2
    num_nodes_per_subgraph = torch.cumsum(torch.tensor([0] + num_nodes_per_subgraph.tolist()), dim=0)
    unbatched_low_A = [low_A[num_nodes_per_subgraph[idx], num_nodes_per_subgraph[idx]:num_nodes_per_subgraph[idx + 1]] for idx in range(len(num_nodes_per_subgraph) - 1)]
    unbatched_high_A = [high_A[num_nodes_per_subgraph[idx], num_nodes_per_subgraph[idx]:num_nodes_per_subgraph[idx + 1]] for idx in range(len(num_nodes_per_subgraph) - 1)]
    unbatched_A = [A[num_nodes_per_subgraph[idx], num_nodes_per_subgraph[idx]:num_nodes_per_subgraph[idx + 1]] for idx in range(len(num_nodes_per_subgraph) - 1)]
    dist_low_As = []
    dist_high_As = []
    for unbatched_A, unbatched_low_A, unbatched_high_A in zip(unbatched_A, unbatched_low_A, unbatched_high_A):
        dist_low_A = torch.norm(unbatched_A - unbatched_low_A, p=1) / unbatched_A.shape[0] + eps
        dist_high_A = torch.norm(unbatched_A - unbatched_high_A, p=1) / unbatched_A.shape[0] + eps
        dist_low_As.append(dist_low_A)
        dist_high_As.append(dist_high_A)
    dist_low_As = torch.stack(dist_low_As)
    dist_high_As = torch.stack(dist_high_As)
    score_X = F.softmax(torch.stack([dist_low_X, dist_high_X], dim=1), dim=1)[:, 0]
    score_A = F.softmax(torch.stack([dist_low_As + dist_high_As], dim=1), dim=1)[:, 0]

    output = (score_X + score_A) / 2
    return output, torch.stack([dist_low_X, dist_high_X, dist_low_As, dist_high_As], dim=1)
    
def suplabel_lossv6neg(z1: torch.Tensor, z2: torch.Tensor, pos_mask: torch.Tensor, debias, tau):
        
    s_value = torch.exp(torch.mm(z1 , z1.t()) / tau)
    b_value = torch.exp(torch.mm(z1 , z2.t()) / tau)
    neg_mask = 1 - pos_mask
    #value_zi = b_value.diag().unsqueeze(0).T
    value_zi = (s_value + b_value) * pos_mask.float()
    value_zi = value_zi.sum(dim=1, keepdim=True)

    value_neg = (s_value + b_value) * neg_mask.float()
    value_neg = value_neg.sum(dim=1, keepdim=True)
    neg_sum = 2 * neg_mask.sum(dim=1, keepdim=True)
    value_neg = (value_neg - value_zi * neg_sum * debias) / (1 - debias)
    value_neg = torch.max(value_neg, neg_sum * math.exp(-1.0 / tau))
    value_mu = value_zi + value_neg
    
    loss = -torch.log(value_zi / value_mu)
    return loss


def cl_lossaug(h1: torch.Tensor, h2: torch.Tensor, pos_mask, debias, mean: bool = True, tau=1):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    
    loss1 = suplabel_lossv6neg(h1, h2, pos_mask, debias, tau)
    loss2 = suplabel_lossv6neg(h2, h1, pos_mask, debias, tau)
    ret = (loss1 + loss2) / 2

    ret = ret.mean() if mean else ret.sum()
    return ret

def accuracy(preds, labels):
    correct = preds == labels.double()
    correct = correct.sum()
    return correct

def weighted_cal(labels):
    class_counts = torch.bincount((labels > 0).int()).float()
    weight = torch.cat([torch.ones(1), (class_counts[0] / class_counts[1]).unsqueeze(0)])
    return weight
