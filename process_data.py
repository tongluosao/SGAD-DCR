import torch
from torch.utils.data import Dataset, DataLoader
from dgl.dataloading import Sampler
import dgl
from utils import *

class GraphDataset(Dataset):
    def __init__(self, g, features, nodes, labels, subgraph_size, restart_prob):
        """
        :param g: DGL图对象
        :param label_key: 节点标签在g.ndata中的键
        """
        super(GraphDataset, self).__init__()
        self.graph = g
        self.nodes = torch.tensor(nodes)  # 获取所有节点编号
        self.labels = labels  # 获取节点标签
        self.labels = torch.where(self.labels > 0, 1, 0)
        self.subgraph_size = subgraph_size  # 子图大小
        self.restart_prob = restart_prob  # 重启概率
        self.graph.ndata['feat'] = features  # 将节点特征存储在图中

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        node_id = self.nodes[idx]  # 获取节点ID
        if idx > len(self.labels) - 1:
            node_label = torch.tensor(-1, dtype=torch.long)  # 如果没有标签，则返回-1
        else:
            node_label = self.labels[idx]   # 获取该节点的标签
        subgraph = generate_rwr_subgraph(self.graph, node_id, self.subgraph_size, self.restart_prob)  # 生成子图
        return subgraph, subgraph.ndata['feat'], node_label  # 返回节点编号、特征和标签
    
def collate_fn(batch):
    """
    :param batch: 一个batch的数据，每个元素是一个元组，包含子图、节点特征和节点标签
    :return: 返回一个元组，包含所有子图、节点特征和节点标签
    """
    graphs, feats, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    batched_feats = torch.cat(feats, dim=0)
    batched_labels = torch.stack(labels, dim=0)
    return batched_graph, batched_feats, batched_labels
    
class Standardization():
    def __init__(self, data):
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)
        pass
    def standardize(self, data):
        notnan_mask = self.std != 0
        data = data[:, notnan_mask]
        std = self.std[notnan_mask]
        mean = self.mean[notnan_mask]
        return (data - mean) / std
    