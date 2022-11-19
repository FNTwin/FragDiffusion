import os

import torch
from torch.utils.data import random_split, Dataset
import torch_geometric.utils

from dgd.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

# TODO: Update
FRAG_GRAPH_FILE = "frag/mol_frag_graphs.pt" 
FRAG_INDEX_FILE = "frag/fragment_index.csv"
FRAG_EDGE_FILE = "frag/fragment_edge_index.csv"


class FragDataset(Dataset):
    def __init__(self, data_file):
        """ This class can be used to load the comm20, sbm and planar datasets. """
        base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'data')
        filename = os.path.join(base_path, data_file)
        self.graphs = torch.load(filename)
        print(f'Dataset {filename} loaded from file')

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        n_edges = data.edge_index.shape[-1]
        n_edge_classes = data.edge_attr.shape[-1]
        y = torch.zeros([1, 0]).float()
        data.idx = idx
        n_nodes  = data.num_nodes * torch.ones(1, dtype=torch.long)
        # Add edge type for "no edge"
        new_edge_attr = torch.zeros(n_edges, n_edge_classes+1, dtype=torch.float)
        new_edge_attr[:, 1:] = data.edge_attr
        data_out = torch_geometric.data.Data(x=data.x, edge_index=data.edge_index, edge_attr=new_edge_attr,
                                         y=y, idx=idx, n_nodes=n_nodes)
        return data_out


class FragDataModule(AbstractDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.file_name = FRAG_GRAPH_FILE
        self.prepare_data()
        self.inner = self.train_dataloader()

    def __getitem__(self, item):
        return self.inner[item]

    def prepare_data(self):
        graphs = FragDataset(self.file_name)
        test_len = int(round(len(graphs) * 0.2))
        train_len = int(round((len(graphs) - test_len) * 0.8))
        val_len = len(graphs) - train_len - test_len
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        splits = random_split(graphs, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(1234))

        datasets = {'train': splits[0], 'val': splits[1], 'test': splits[2]}
        super().prepare_data(datasets)


class FragDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = datamodule.node_types()
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)

