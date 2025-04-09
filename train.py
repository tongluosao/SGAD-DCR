import math
from multiprocessing import freeze_support
import os
import time
import argparse
import dgl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from dgl import DGLGraph
from dgl.data.utils import load_graphs, save_graphs
from utils import *
from model import Model, Discriminator
from process_data import *
import json
import pygod
from torchmetrics.functional import precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from pygod.utils import load_data
import scipy.sparse as sp
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime


def train(args):
    
    
    output_dir = args.output_dir
    device = torch.device(args.device)
    lr_gen = args.lr_gen
    lr_dis = args.lr_dis
    weight_decay = args.weight_decay
    epochs = args.epochs
    labeled_ratio = args.labeled_ratio
    train_ratio = args.train_ratio
    batch_size = args.batch_size
    flip_prob = 0.5
    threshold = 0.5
    D_final_threshold = 0.5
    debias = 0.5
    seed = args.seed
    random.seed(seed)
    noise_variance = args.noise_variance ** 0.5
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)

    clip = 0.1
    if args.resume != None:
        output_dir = args.resume
    else:
        output_dir = os.path.join(output_dir, str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'args.json'), 'w') as f:
            f.write(json.dumps(args.__dict__))
    # load data
    if args.dataset in ['tfinance', 'tsocial']:
        g = load_graphs('dataset/' + args.dataset)[0][0]
        features = g.ndata['feature'].float()
        if len(g.ndata['label'].shape) > 1:
            labels = g.ndata['label'][:, 1]
        else:
            labels = g.ndata['label']
    elif args.dataset in ['acmv9', 'dblpv7']:
        data_mat = sio.loadmat(f'dataset/{args.dataset}_both.mat')
        adj = sp.csr_matrix(data_mat['Network'])
        features = data_mat['Attributes'].tocoo()
        row = torch.tensor(features.row, dtype=torch.long)
        col = torch.tensor(features.col, dtype=torch.long)
        data = torch.tensor(features.data, dtype=torch.float32)

        # 构建 PyTorch 稀疏张量
        indices = torch.stack([row, col])
        features = torch.sparse_coo_tensor(indices, data, features.shape).to_dense()
        labels = torch.from_numpy(data_mat['Label'].flatten())
        g = dgl.from_scipy(adj)
    else:
        dataset = load_data(args.dataset, cache_dir='dataset')
        labels = dataset.y
        features = dataset.x
        g = dgl.graph((dataset.edge_index[0], dataset.edge_index[1]),
                num_nodes=dataset.num_nodes)
                
    normalization = Standardization(features)
    features = normalization.standardize(features)

    g.ndata['center'] = torch.zeros(g.num_nodes(), dtype=torch.bool)
    # split data
    nodes = list(range(g.num_nodes()))
    random.shuffle(nodes)

    train_nodes = nodes[:int(len(nodes) * train_ratio)]
    test_nodes = nodes[int(len(nodes) * labeled_ratio):]
    train_labels = labels[nodes[:int(len(nodes) * labeled_ratio)]]
    if labels[nodes[int(len(nodes) * labeled_ratio)]].sum() == 0:
        for i, idx in enumerate(test_nodes):
            if labels[idx] == 1:
                train_nodes.insert(0, idx)
                train_labels = torch.cat([torch.ones(1), train_labels])
                test_nodes.pop(i)
                break
    train_dataset = GraphDataset(
        g, features, train_nodes, train_labels, args.subgraph_size, args.restart_prob)
    test_dataset = GraphDataset(
        g, features, test_nodes, labels[test_nodes], args.subgraph_size, args.restart_prob)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=args.num_workers, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=args.num_workers, collate_fn=collate_fn, shuffle=False)

    model = Model(features.shape[1], args.hidden_model,
                  args.dropout, args.eps, args.layer_num, args.attention_heads, 0.1).to(device)
    discriminator = Discriminator(
        args.hidden_model + 1, args.hidden_discriminator, args.dropout).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    # print(f"Model structure: {model}")

    weight = weighted_cal(
        labels[train_nodes[:int(len(train_nodes) * labeled_ratio)]]).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    criterion_D = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr_gen, weight_decay=weight_decay)
    optimizer_discriminator = torch.optim.AdamW(
        discriminator.parameters(), lr=lr_dis, weight_decay=weight_decay)

    best_auc = 0
    best_F1 = 0
    val_acc_history = []
    val_f1_history = []
    val_auc_history = []
    val_loss_history = []
    train_loss_history = []
    train_acc_history = []
    train_f1_history = []
    train_auc_history = []
    resume_history = 0
    if args.resume:
        history = pd.read_csv(os.path.join(args.resume, 'result.csv'))
        val_acc_history = list(history['val_acc'])
        val_f1_history = list(history['val_f1'])
        val_auc_history = list(history['val_auc'])
        val_loss_history = list(history['val_loss'])
        train_loss_history = list(history['train_loss'])
        train_acc_history = list(history['train_acc'])
        train_f1_history = list(history['train_f1'])
        train_auc_history = list(history['train_auc'])
        best_auc = max(*val_auc_history)
        best_F1 = max(*val_f1_history)
        resume_history = len(val_auc_history)
        model.load_state_dict(torch.load(os.path.join(args.resume, 'model_last.pth')))
        discriminator.load_state_dict(torch.load(os.path.join(args.resume, 'discriminator_last.pth')))
    for epoch in range(resume_history, epochs):
        model.train()
        discriminator.train()
        pbar = tqdm(train_loader)
        D_threshold = D_final_threshold * np.exp(((epoch + 1) - epochs) / epochs)

        preds = []
        labels = []
        losss = 0
        valid_cnt = 0
        pbar.set_description(f'Epoch {epoch + 1}/{epochs}')

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('ACC')
        ax2.set_ylabel('Loss')
        total = min(len(train_loader), args.max_batchs)
        pbar.total = total
        for idx, (subgraph, node_feat, node_label) in enumerate(pbar):
            node_label = node_label.to(device)
            subgraph = subgraph.to(device)
            node_feat = node_feat.to(device)
            # 遮蔽
            masked_feat = node_feat.clone() + torch.randn_like(node_feat) * noise_variance
            masked_feat[subgraph.ndata['center']] = 0

            num_nodes_per_subgraph = subgraph.batch_num_nodes()
            
            # 模型前向传播
            recon = model(subgraph, masked_feat)
            output, recon_dists = output_cal(node_feat, subgraph.adjacency_matrix(
            ).to_dense(), recon, subgraph.ndata['center'], num_nodes_per_subgraph, args.tau)  # 计算概率值,[结构异常，属性异常]
            nonnans = ~output.isnan()
            output = output[nonnans]
            recon_dists = recon_dists[nonnans]
            node_label = node_label[nonnans]
            # 监督部分
            supervised_indices = node_label != -1
            supervised_label = node_label[supervised_indices]
            #supervised_recon_dists = recon_dists[supervised_indices]
            node_feat = node_feat[subgraph.ndata['center']]
            # supervised_node_feat = node_feat[supervised_indices]
            supervised_output = output[supervised_indices]
            low_cl, high_cl = model.low_cl[subgraph.ndata['center']][nonnans], model.high_cl[subgraph.ndata['center']][nonnans]
            wx = low_cl + high_cl
            supervised_wx = wx[supervised_indices]
            supervised_anomaly = (supervised_label > 0).int()

            preds.append(supervised_output)
            labels.append(supervised_label)

            # 无监督部分

            fake_wx = wx[~supervised_indices]
            fake_output = output[~supervised_indices]
            fake_recon_dists = recon_dists[~supervised_indices]
            fake_label_hat = torch.where(fake_output >= threshold, fake_output + (
                1 - fake_output).detach(), fake_output - fake_output.detach()).long()  # 根据阈值判断是否为伪标签，重参数化
            rd_anomaly_hat, flip_mask = perturb_labels(
                fake_label_hat, flip_prob)
            pseudo_hat = discriminator(
                fake_wx, rd_anomaly_hat)  # 判断随机翻转输入后的标签是否为伪标签
            flip_pseudo_hat = F.sigmoid(torch.where(
                flip_mask, -pseudo_hat, pseudo_hat))

            abandon_mask = flip_pseudo_hat > D_threshold  # 未标记节点且预测为伪标签的节点
            valid_output = fake_output[~abandon_mask]  # 判别正确的伪标签部分
            valid_wx = fake_wx[~abandon_mask]
            invalid_recon_dists = fake_recon_dists[abandon_mask]
            wx = torch.cat([supervised_wx, valid_wx], dim=0)
            output = torch.cat(
                [supervised_output, valid_output], dim=0)  # 拼接监督部分和伪标签部分
            label = torch.cat(
                [supervised_label, (valid_output >= threshold)], dim=0).long()
            #recon_dists = torch.cat([supervised_recon_dists, valid_recon_dists], dim=0)
            # 对比损失
            low_abandon_cl = low_cl[~supervised_indices][abandon_mask]
            high_abandon_cl = high_cl[~supervised_indices][abandon_mask]
            low_cl = torch.cat(
                [low_cl[supervised_indices], low_cl[~supervised_indices][~abandon_mask]], dim=0)
            high_cl = torch.cat(
                [high_cl[supervised_indices], high_cl[~supervised_indices][~abandon_mask]], dim=0)
            sp_label = label.unsqueeze(0)
            pos_mask = (sp_label == sp_label.T).int()
            cl_loss = cl_lossaug(low_cl, high_cl, pos_mask, debias)
            abandon_cl_loss = cl_lossaug(low_abandon_cl, high_abandon_cl, torch.eye(low_abandon_cl.shape[0]).to(device), debias=debias)
            cl_loss = (abandon_cl_loss * args.lambda_abandon_cl + cl_loss) / (1 + args.lambda_abandon_cl)
            out = output.unsqueeze(-1)
            # 分类损失
            loss_cls = criterion(torch.cat([1 - out, out], dim=1), label).mean()
            loss_recon_X = invalid_recon_dists[:, :2].mean()
            loss_recon_A = invalid_recon_dists[:, 2:].mean()
            loss_recon = (loss_recon_X + loss_recon_A) / 2         
            # 主干模型损失
            loss_main = loss_cls + cl_loss + loss_recon * args.lambda_recon
            losss += loss_main * out.shape[0]
            valid_cnt += out.shape[0]
            # flip_pseudo_hat = torch.where(flip_mask, pseudo_hat, 1 - pseudo_hat)        #将伪标签判断翻转回来
            # pseudo_mask = torch.where(~supervised_indices, 1, 0)
            rd_supervised_anomaly, supervised_flip_mask = perturb_labels(
                supervised_anomaly, flip_prob)
            supervised_pseudo_hat = discriminator(
                supervised_wx.detach(), rd_supervised_anomaly.detach())
            flip_supervised_pseudo_hat = torch.where(
                supervised_flip_mask, -supervised_pseudo_hat, supervised_pseudo_hat)
            supervised_loss_D = flip_supervised_pseudo_hat.mean()
            
            fake_label_hat = torch.where(fake_output >= threshold, fake_output + (
                1 - fake_output).detach(), fake_output - fake_output.detach()).long()  # 根据阈值判断是否为伪标签，重参数化
            rd_anomaly_hat, flip_mask = perturb_labels(
                fake_label_hat, flip_prob)
            pseudo_hat = discriminator(
                fake_wx.detach(), rd_anomaly_hat.detach())  # 判断随机翻转输入后的标签是否为伪标签
            flip_pseudo_hat = F.sigmoid(torch.where(
                flip_mask, -pseudo_hat, pseudo_hat))
            fake_loss_D = -flip_pseudo_hat.mean()

            loss_D = supervised_loss_D + fake_loss_D

            optimizer_discriminator.zero_grad()
            supervised_wx.detach()
            rd_supervised_anomaly.detach()
            fake_wx.detach()
            rd_anomaly_hat.detach()
            loss_D.backward()
            optimizer_discriminator.step()
            for p in discriminator.parameters():
                p.data.clamp_(-clip, clip)
            pbar.set_postfix({'loss_main': '{:.4f}'.format(
                loss_main.item()), 'loss_D': '{:.4f}'.format(loss_D.item())})
            optimizer.zero_grad()

            loss_main.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)    #设置剪裁阈值为5
            optimizer.step()
            torch.cuda.empty_cache()
            if idx >= total:
                break
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)
        losss /= valid_cnt
        preds = preds
        labels = labels
        precision, recall, thresholds = precision_recall_curve(
                preds, labels, task='binary')
        f1score_pos = 2 * (precision * recall) / (precision + recall + 1e-8)
        #precision, recall, _ = precision_recall_curve(
        #    1 - preds, 1 - labels, task='binary')
        #f1score_neg = torch.flip(2 * (precision * recall) / (precision + recall + 1e-8), [0])
        f1score = f1score_pos
        bestthreshold = thresholds[torch.argmax(f1score)]

        auc = roc_auc_score(labels.cpu().numpy(), preds.detach().cpu().numpy())
        
        acc = ((preds >= bestthreshold).int() ==
               labels).int().sum() / preds.shape[0]

        print(f'f1: {f1score.max():.4f}, acc: {acc:.4f}, auc: {auc:.4f}')
        train_acc_history.append(acc.item())
        train_auc_history.append(auc)
        train_f1_history.append(f1score.max().item())
        train_loss_history.append(losss.item())
        #ax1.plot(range(1, epoch + 2), train_acc_history,
        #         'b--', label='train_acc')
        ax1.plot(range(1, epoch + 2), train_auc_history,
                 'g--', label='train_auc')
        ax1.plot(range(1, epoch + 2), train_f1_history,
                 'y--', label='train_f1')
        ax2.plot(range(1, epoch + 2), train_loss_history,
                 'r--', label='train_loss')
                 
        model.eval()
        with torch.no_grad():
            acc = 0
            pbar = tqdm(test_loader)
            preds = []
            labels = []
            losss = 0
            total = min(len(test_loader), args.max_batchs)
            pbar.total = total
            for idx, (subgraph, node_feat, node_label) in enumerate(pbar):
                subgraph = subgraph.to(device)
                node_feat = node_feat.to(device)
                num_nodes_per_subgraph = subgraph.batch_num_nodes()
                masked_feat = node_feat.clone()
                masked_feat[subgraph.ndata['center']] = 0
                node_label = node_label.to(device)

                # 模型前向传播
                recon = model(subgraph, masked_feat)
                output, _ = output_cal(node_feat, subgraph.adjacency_matrix(
                ).to_dense(), recon, subgraph.ndata['center'], num_nodes_per_subgraph, args.tau)  # 计算概率值
                node_feat = node_feat[subgraph.ndata['center']]

                low_cl, high_cl = model.low_cl[subgraph.ndata['center']
                                               ], model.high_cl[subgraph.ndata['center']]
                sp_label = node_label.unsqueeze(0)
                pos_mask = (sp_label == sp_label.T).int()
                cl_loss = cl_lossaug(low_cl, high_cl, pos_mask, debias)
                out = output.unsqueeze(-1)
                loss_cls = criterion(
                    torch.cat([1 - out, out], dim=1), node_label).mean()       
                # 主干模型损失
                loss = loss_cls + cl_loss
                losss += loss * out.shape[0]
                pbar.set_postfix({'loss': '{:.4f}'.format(loss.item())})
                preds.append(output)
                labels.append(node_label)
                if idx >= total:
                    break

            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)
            losss /= min(len(test_dataset), batch_size * total)
            preds = preds
            labels = labels
            precision, recall, thresholds = precision_recall_curve(
                preds, labels, task='binary')
            f1score_pos = 2 * (precision * recall) / (precision + recall + 1e-8)
            #precision, recall, _ = precision_recall_curve(
            #    1 - preds, 1 - labels, task='binary')
            #f1score_neg = torch.flip(2 * (precision * recall) / (precision + recall + 1e-8), [0])
            f1score = f1score_pos
            bestthreshold = thresholds[torch.argmax(f1score)]
            auc = roc_auc_score(labels.cpu().numpy(),
                                preds.detach().cpu().numpy())
            acc = ((preds >= bestthreshold).int() ==
                   labels).int().sum() / preds.shape[0]
            
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), os.path.join(output_dir,'model_best_AUC.pth'))
            if f1score.max() > best_F1:
                best_F1 = f1score.max()
                torch.save(model.state_dict(), os.path.join(output_dir, 'model_best_F1.pth'))
            val_f1_history.append(f1score.max().item())
            val_acc_history.append(acc.item())
            val_auc_history.append(auc)
            val_loss_history.append(losss.item())
            print(f'f1: {f1score.max():.4f}, acc: {acc:.4f}, auc: {auc:.4f}')
            #ax1.plot(range(1, epoch + 2), val_acc_history, 'b-', label='acc')
            ax1.plot(range(1, epoch + 2), val_f1_history, 'y-', label='f1')
            ax1.plot(range(1, epoch + 2), val_auc_history, 'g-',  label='auc')
            ax2.plot(range(1, epoch + 2), val_loss_history, 'r-', label='loss')
        fig.legend()
        fig.savefig(os.path.join(output_dir, "result.png"))
        plt.close(fig)
        pd.DataFrame(np.array([val_acc_history,
                  val_f1_history,
                  val_auc_history,
                  val_loss_history,
                  train_loss_history,
                  train_acc_history,
                  train_f1_history,
                  train_auc_history]).T, columns=['val_acc', 'val_f1', 'val_auc', 'val_loss', 'train_loss', 'train_acc', 'train_f1', 'train_auc']).to_csv(os.path.join(output_dir, 'result.csv'), index=False)
        torch.save(model.state_dict(), os.path.join(output_dir, 'model_last.pth'))
        torch.save(discriminator.state_dict(), os.path.join(output_dir, 'discriminator_last.pth'))





if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='disney',
                        help='cornell texas wisconsin chameleon squirrel')
    parser.add_argument('--max_batchs', type=int, default=100)
    parser.add_argument('--lr_gen', type=float, default=6e-4,
                        help='Initial learning rate.')
    parser.add_argument('--lr_dis', type=float, default=1e-4,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.5,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int,
                        default=128, help='Batch size.')
    parser.add_argument('--num_workers', type=int,
                        default=0, help='Number of workers.')
    parser.add_argument('--device', default='cuda', help='Device to use.')
    parser.add_argument('--seed', type=int, default=50, help='Random seed.')
    parser.add_argument('--p_l', type=float, default=0.1,
                        help='Low frequency gradient')
    parser.add_argument('--p_h', type=float, default=0.5,
                        help='High frequency edge gradient')
    parser.add_argument('--subgraph_size', type=int,
                        default=5, help='Subgraph size')
    parser.add_argument('--restart_prob', type=float, default=0.2,
                        help='Subgraph sample restart probability')
    parser.add_argument('--hidden_model', type=int,
                        default=64, help='Number of hidden units.')
    parser.add_argument('--attention_heads', type=int, default=6, help='Number of attention heads.')
    parser.add_argument('--hidden_discriminator', type=tuple,
                        default=(256, 256, 128, 128, 64, 64, 32), help='Number of hidden discriminator units.')
    parser.add_argument('--dropout', type=float, default=0.8,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--eps', type=float, default=0.1,
                        help='Fixed scalar or learnable weight.')
    parser.add_argument('--layer_num', type=int,
                        default=2, help='Number of layers')
    parser.add_argument('--labeled_ratio', type=float,
                        default=0.025, help='Ratio of lebeled set')
    parser.add_argument('--train_ratio', type=float,
                        default=0.4, help='Ratio of training set')
    parser.add_argument('--noise_variance', type=float,
                        default=5e-1, help='Variance of noise injection')
    parser.add_argument('--tau', type=float, default=0.5, help='Temperature')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--lambda_recon', type=float, default=0.3)
    parser.add_argument('--lambda_abandon_cl', type=float, default=0.2)
    parser.add_argument('--resume', type=str, default='')
    args = parser.parse_args()
    
    train(args)
