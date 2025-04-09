import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from dgl import function as fn
from dgl.sampling import random_walk
import dgl
import numpy as np
import torch
import numpy as np


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, dropout, alpha, gain=1.414):
        super(GCNLayer, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_feats, out_feats)
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        self.alpha = alpha

    def forward(self, g, feature, freq='low'):
        if freq == 'low':
            D_ = torch.diag(g.ndata["hat_d"])
            I = torch.eye(D_.shape[0]).to(D_.device)
            h = feature
            A = g.adjacency_matrix().to_dense() + I
        
            h = D_ @ A @ D_ @ h
        elif freq == 'high':
            D_ = torch.diag(g.ndata["d"])
            I = torch.eye(D_.shape[0]).to(D_.device)
            A = g.adjacency_matrix().to_dense()
            h = (I - D_ @ A @ D_) @ feature
        else:
            raise ValueError("freq必须为low或者high")
        h = self.dropout(h)
        return self.linear(h)


class shareGCNLayer(nn.Module):
    def __init__(self, gcn_layer, freq):
        super(shareGCNLayer, self).__init__()
        self.shared_gcn_layer = gcn_layer
        self.freq = freq

    def forward(self, g, feature):
        return self.shared_gcn_layer(g, feature, self.freq)


class HLLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, num_heads, freq='low'):
        super(HLLayer, self).__init__()
        # self.g = g
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.in_linear = nn.Linear(in_dim, out_dim * num_heads)
        self.gates = nn.ModuleList([nn.Linear(2 * out_dim, 1) for _ in range(num_heads)])
        self.att_gates = nn.ModuleList([nn.Linear(out_dim, out_dim) for _ in range(num_heads)])
        self.freq = freq
        self.num_heads = num_heads
        self.out_linear = nn.Linear(out_dim * num_heads, out_dim)
        for gate in self.gates:
            nn.init.xavier_normal_(gate.weight, gain=1.414)
        for att_gate in self.att_gates:
            nn.init.xavier_normal_(att_gate.weight, gain=1.414)
        nn.init.xavier_normal_(self.in_linear.weight, gain=1.414)
        nn.init.xavier_normal_(self.out_linear.weight, gain=1.414)

        if freq == 'low':
            self.p_l = 0.1
        elif freq == 'high':
            self.p_h = 0.5
        

    def edge_applying(self, edges):
        alphas = []
        for head in range(self.num_heads):
            dst_h = self.att_gates[head](edges.dst['h'][:, head, :])
            src_h = self.att_gates[head](edges.src['h'][:, head, :])

            h2 = torch.cat([dst_h, src_h], dim=1)
            self.dropout(h2)
            _h = self.gates[head](h2)
            if self.freq == 'low':
                _g = F.relu(torch.where(_h>0, _h, -self.p_l*_h)).squeeze()
            elif self.freq == 'high':
                _g = -F.relu(torch.where(_h>0, _h, -self.p_h*_h)).squeeze()
            else:
                raise ValueError("freq必须为low或者high")
            # g_high = -F.relu(torch.where(_high>0, _high, -self.p_h*_high)).squeeze()   #原文章中的NegReLU和PosReLU
            alpha = _g * edges.dst['d'] * edges.src['d']
            alphas.append(alpha)
        alphas = torch.stack(alphas, dim=1).unsqueeze(-1)
        return {'alpha': alphas}
    
    def message_func(edges):
        # edges.src['h'] 的形状是 (num_edges, num_heads, hidden_dim)
        # 将 alpha 重复以匹配边的数量
        alpha_repeated = edges.dst['alpha'].repeat_interleave(edges.src['h'].size(0), dim=0)
        # 计算消息，即注意力加权的源节点特征
        return {'msg': edges.src['h'] * alpha_repeated}

    def forward(self, g, h):
        h = self.dropout(h)
        h = F.relu(self.in_linear(h).view(-1, self.num_heads, self.out_dim))
        g.ndata['h'] = h
        g.apply_edges(self.edge_applying)
        g.update_all(fn.u_mul_e('h', 'alpha', '_z'), fn.sum('_z', 'z'))
        h = g.ndata['z'].reshape(-1, self.out_dim * self.num_heads)

        return self.out_linear(h)


class Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout, eps, enc_layer_num, num_heads, alpha=1):
        super(Model, self).__init__()
        self.eps = eps
        self.enc_layer_num = enc_layer_num
        self.alpha = alpha
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        freqs = ['low', 'high']
        

        self.t1 = nn.Linear(in_dim, hidden_dim)
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        #self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)
        gcn_enc = nn.ModuleList([GCNLayer(
            hidden_dim, hidden_dim, self.dropout, alpha) for i in range(enc_layer_num)])

        self.enc_layers = nn.ModuleList([
            nn.ModuleDict({
                'low_GAT': HLLayer(hidden_dim, hidden_dim, self.dropout, self.num_heads, 'low'),
                'high_GAT': HLLayer(hidden_dim, hidden_dim, self.dropout, self.num_heads, 'high'),
                'low_GCN': shareGCNLayer(gcn_enc[i], 'low'),
                'high_GCN': shareGCNLayer(gcn_enc[i], 'high'),
                'low_GAT_BN': nn.BatchNorm1d(hidden_dim),
                'high_GAT_BN': nn.BatchNorm1d(hidden_dim),
                'low_GCN_BN': nn.BatchNorm1d(hidden_dim),
                'high_GCN_BN': nn.BatchNorm1d(hidden_dim)
            })
            for i in range(enc_layer_num)])
        self.dec_layers = nn.ModuleDict()
        for freq in freqs:
            self.dec_layers.update({
                freq + '_X': nn.Linear(hidden_dim, in_dim),
                freq: nn.Linear(2*hidden_dim, hidden_dim)
            })

    def forward(self, g, h):
        h = self.dropout(h)
        h = F.relu(self.t1(h))
        h = self.dropout(h)
        h_dict = {freq + '_GAT': h for freq in ['low', 'high']}
        h_dict.update({freq + '_GCN': h for freq in ['low', 'high']})
        # 编码
        for idx, layer in enumerate(self.enc_layers):
            for freq in ['low', 'high']:
                h_dict[freq + '_GAT'] = self.dropout(layer[freq + '_GAT'](g, h_dict[freq + '_GAT']))
                h_dict[freq + '_GAT'] = layer[freq + '_GAT_BN'](self.eps * h + F.relu(h_dict[freq + '_GAT']))
                h_dict[freq + '_GCN'] = self.dropout(layer[freq + '_GCN'](g, h_dict[freq + '_GCN']))
                h_dict[freq + '_GCN'] = layer[freq + '_GCN_BN'](self.eps * h + F.relu(h_dict[freq + '_GCN']))
            if idx == 0:
                self.low_h1 = h_dict['low_GCN']
                self.high_h1 = h_dict['high_GCN']
        self.low_cl, self.high_cl = h_dict['low_GCN'], h_dict['high_GCN']
        low_h = torch.cat([h_dict['low_GAT'], h_dict['low_GCN']], dim=1)
        high_h = torch.cat([h_dict['high_GAT'], h_dict['high_GCN']], dim=1)
        #low_h = self.batch_norm(low_h)
        #high_h = self.batch_norm(high_h)
        # 重建
        
        low_Z = self.dec_layers['low'](low_h)
        high_Z = self.dec_layers['high'](high_h)
        #print(torch.isnan(low_Z))
        
        low_X = self.dec_layers['low_X'](low_Z)
        high_X = self.dec_layers['high_X'](high_Z)
        
        low_A = F.sigmoid((low_Z @ low_Z.T))
        high_A = F.sigmoid((high_Z @ high_Z.T))
        g.ndata['low_X'] = low_X
        g.ndata['high_X'] = high_X
        return {'low_X': low_X, 'high_X': high_X, 'low_A': low_A, 'high_A': high_A}

    



class Discriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super(Discriminator, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.in_layer = nn.Linear(in_dim, hidden_dim[0])
        nn.init.xavier_uniform_(self.in_layer.weight)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_dim[i], hidden_dim[i+1]) for i in range(len(hidden_dim)-1)])
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim[i]) for i in range(len(hidden_dim))])
        self.out_layer = nn.Linear(hidden_dim[-1], 1)
        nn.init.xavier_uniform_(self.out_layer.weight)
    
    def forward(self, h, label):
        label = label.reshape(-1, 1)
        h = torch.cat([h, label], dim=1)
        h = self.dropout(h)
        h = F.relu(self.in_layer(h))
        bn = False
        if h.shape[0] > 1:
            bn = True
            h = self.bns[0](h)
        for idx, layer in enumerate(self.layers):
            h = F.relu(layer(h))
            if bn:
                h = self.bns[idx+1](h)
            
        return self.out_layer(h).squeeze()
