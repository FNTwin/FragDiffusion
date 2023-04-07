from pathlib import Path
import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import random_split, Dataset
import torch_geometric.utils

from dgd.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

# TODO: Update
FRAG_GRAPH_FILE = "zinc/mol_frag_graphs_250k_300_5.pt"
ATOM_GRAPH_FILE = "zinc/atom_graphs_250k_300_5.pt" #"frag/atom_graphs_100000.pt"
ATOM_DECODER_FILE = "frag/atom_decoder_250k.csv" #"frag/atom_decoder.csv"
SMILES_FILE = "frag/valid_smiles_250k.txt"
FRAG_INDEX_FILE = "frag/fragment_index.csv"
FRAG_EDGE_FILE = "frag/fragment_edge_index.csv"
SPLIT_IDX_FILE = "frag/split_idxs.npz"


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
        n_edge_classes = data.edge_attr.shape[-1]
        y = torch.zeros([1, 0]).float()
        data.idx = idx
        n_nodes  = data.num_nodes * torch.ones(1, dtype=torch.long)

        # Symmetrize
        edge_index, edge_attr = torch_geometric.utils.to_undirected(data.edge_index, data.edge_attr,  data.num_nodes)
        n_edges = edge_index.shape[-1]

        # Add edge type for "no edge"
        new_edge_attr = torch.zeros(n_edges, n_edge_classes+1, dtype=torch.float)
        new_edge_attr[:, 1:] = edge_attr
        data_out = torch_geometric.data.Data(x=data.x.float(), edge_index=edge_index, edge_attr=new_edge_attr,
                                         y=y, idx=idx, n_nodes=n_nodes)
        return data_out


class AtomDataset(FragDataset):
    def __getitem__(self, idx):
        data = self.graphs[idx]
        data.idx = idx

        return data


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
        splits = random_split(
            graphs,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(1234)
        )

        datasets = {'train': splits[0], 'val': splits[1], 'test': splits[2]}
        split_idxs = {
            '%s_idxs' % key: np.array([x.idx for x in val])
            for key, val in datasets.items()
        }

        cwd = Path.cwd()
        print('Saving split indices to %s' % cwd)
        np.savez(cwd / 'split_idxs', **split_idxs)

        super().prepare_data(datasets)


class AtomDataModule(AbstractDataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.file_name = ATOM_GRAPH_FILE
        self.prepare_data()
        self.inner = self.train_dataloader()

    def __getitem__(self, item):
        return self.inner[item]

    def prepare_data(self):
        graphs = AtomDataset(self.file_name)

        base_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            os.pardir,
            os.pardir,
            'data'
        )

        filename = os.path.join(base_path, SPLIT_IDX_FILE)
        split_idxs = np.load(filename)

        datasets = {
            key: [graphs[i] for i in split_idxs['%s_idxs' % key]]
            for key in ['train', 'val', 'test']
        }

        super().prepare_data(datasets)

class FragDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.name = 'nx_graphs'
        self.n_nodes = datamodule.node_counts()
        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)


class AtomDatasetInfos(FragDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        base_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            os.pardir,
            os.pardir,
            'data'
        )

        filename = os.path.join(base_path, ATOM_DECODER_FILE)

        self.atom_decoder = {
            row['index']: row['atom_name']
            for _, row in pd.read_csv(filename).iterrows()
        }
        self.valencies =  [1, 4, 1, 1, 1, 3, 2, 2]
        self.remove_h = dataset_config.remove_h
        self.max_weight = 1000

        self.atom_weights = {0: 79.9, 1: 12, 2: 35.45, 3: 19, 4: 126.9, 5: 14, 6: 16, 7: 32.06}
        super().__init__(datamodule, dataset_config)


def get_train_smiles(train_dataloader):
    base_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.pardir,
        os.pardir,
        'data'
    )

    filename = os.path.join(base_path, SMILES_FILE)
    with open(filename, 'r') as f:
        smiles_with_endline = f.readlines()

    all_smiles = [smiles.replace('\n', '') for smiles in smiles_with_endline]
    nested_idxs = [data['idx'] for data in train_dataloader]
    if torch.is_tensor(nested_idxs[0]):
        train_idxs = torch.cat(nested_idxs).tolist()
    else:
        train_idxs = [x for idxs in nested_idxs for x in idxs]

    return [all_smiles[idx] for idx in train_idxs]
