from rdkit import Chem
from typing import Dict, List, Tuple
import rdkit
import torch
import torch_geometric
import pandas as pd
import wandb
import torch.nn as nn
import torch.nn.functional as f

class FragmentEdgeToAtomEdgeConverter:
    def __init__(
        self,
        frag_id_to_name: Dict[int, str], 
        frag_edge_idx_df: pd.DataFrame
    ):
        self.frag_id_to_name = frag_id_to_name

        self._frag_edge_to_atom_edge = {}
        for _, row in frag_edge_idx_df.iterrows():
            frag_ident = (row['fragment_index_1'], row['fragment_index_2'])
            if frag_ident not in self._frag_edge_to_atom_edge:
                self._frag_edge_to_atom_edge[frag_ident] = {}

            atom_ident = (row['atom_idx_1'], row['atom_idx_2'])
            self._frag_edge_to_atom_edge[frag_ident][row['edge_id']] = atom_ident

    def frag_edge_to_atom_edge(
        self, 
        frag_id_pair: Tuple[int, int], 
        edge_type: int
    ) -> Tuple[int, int]:
        frag_names = [self.frag_id_to_name[frag_id] for frag_id in frag_id_pair]
        sorted_frag_names = list(sorted(frag_names))
        swapped_order = frag_names != sorted_frag_names

        if swapped_order:
            frag_id_pair = tuple(reversed(frag_id_pair))

        atom_pair = self._frag_edge_to_atom_edge[frag_id_pair][edge_type]
        if swapped_order:
            atom_pair = tuple(reversed(atom_pair))

        return atom_pair
    
def _frag_atom_string_to_tuple(frag_atom_str: str) -> Tuple[str]:
    # The atom string is serialized to a string like "('C', 'C', 'N')".
    # To get the tuple of strings, we need to remove the enclosing parenthesis,
# as we do below.
    no_parenthesis_str = frag_atom_str[1:-1]
    return tuple(no_parenthesis_str.split(', '))
    
def _build_frag_id_to_atoms_dict(frag_idx_df: pd.DataFrame) -> Dict[int, Tuple[str]]:
    return {
        row['fragment_index']: _frag_atom_string_to_tuple(row['fragment_atoms'])
        for _, row in frag_idx_df.iterrows()
    }

def _combine_mols(mols: List[rdkit.Chem.rdchem.Mol]) -> rdkit.Chem.rdchem.Mol:
    combined_mol = Chem.CombineMols(mols[0], mols[1])
    for i in range(2, len(mols)):
        combined_mol = Chem.CombineMols(combined_mol, mols[i])

    return combined_mol

class PyGGraphToMolConverter:
    # TODO: make these file paths constants
    def __init__(self, frag_idx_csv_name: str, frag_edge_idx_csv_name: str):
        frag_idx_df = pd.read_csv(frag_idx_csv_name)
        frag_edge_idx_df = pd.read_csv(frag_edge_idx_csv_name)
        
        self.frag_id_to_name = {
            row['fragment_index']: row['fragment_name']
            for _, row in frag_idx_df.iterrows()
        }
        
        self.frag_id_to_atoms = _build_frag_id_to_atoms_dict(frag_idx_df)
        
        self.edge_converter = FragmentEdgeToAtomEdgeConverter(
            self.frag_id_to_name, 
            pd.read_csv(frag_edge_idx_csv_name)
        )


    def frags_to_mol(
        self, frag_ids:torch.tensor, edge_index:torch.tensor, edge_ids:torch.tensor
    ) -> rdkit.Chem.rdchem.Mol:
        frag_names = [self.frag_id_to_name[frag_id.item()] for frag_id in frag_ids]
        frag_mols = [Chem.MolFromSmiles(smiles_str) for smiles_str in frag_names]

        combined_mol = _combine_mols(frag_mols)
        editable_mol = Chem.EditableMol(combined_mol)

        atom_bond_idxs = self._get_atom_bond_idxs(frag_ids, edge_index, edge_ids)
        for atom_bond in atom_bond_idxs:
            editable_mol.AddBond(*atom_bond)

        return editable_mol.GetMol()

    def graph_to_mol(
        self, 
        graph: torch_geometric.data.Data,
        count_non_edge=False
    ) -> rdkit.Chem.rdchem.Mol:
        frag_ids = graph.x.nonzero()[:, 1].flatten()
        edge_ids = graph.edge_attr.nonzero()[:, 1].flatten()

        # Remove duplicate links
        mask = graph.edge_index[0] > graph.edge_index[1]
        edge_index = graph.edge_index[:, mask]
        edge_ids = edge_ids[mask]

        if count_non_edge:
            # edge_id=0 indicates non-edge, so decrement edge_ids
            edge_ids -= 1
        return self.frags_to_mol(frag_ids, edge_index, edge_ids)

    def node_and_adj_to_mol(self, node_list, adjacency_matrix):
        '''
        node_list: numpy list dimension n
        adjacency_matrix: numpy matrix dimension n x n
        '''
        edge_index = adjacency_matrix.nonzero().T
        mask = edge_index[0] > edge_index[1]
        edge_index = edge_index[:, mask]
        edge_ids = adjacency_matrix[edge_index.split(1,dim=0)].squeeze()
        # Decrement edge indices because 0 indicates non-edge
        edge_ids -= 1
        return self.frags_to_mol(node_list, edge_index, edge_ids)

    def _get_atom_bond_idxs(
        self, 
        frag_ids: torch.Tensor, 
        edge_index: torch.Tensor, 
        edge_ids: torch.Tensor
    ) -> List[Tuple[int]]:
        edge_index = edge_index.T
        frag_atom_start_idx = self._get_frag_atom_start_idx(frag_ids)
        
        atom_bond_idxs = []
        for i in range(len(edge_index)):
            atom_edge = self.edge_converter.frag_edge_to_atom_edge(
                tuple(frag_ids[v].item() for v in edge_index[i]), 
                edge_ids[i].item()
            )
            
            mol_atom_edge = tuple(
                frag_atom_start_idx[edge_index[i, j]] + atom_edge[j]
                for j in range(2)
            )
            
            atom_bond_idxs.append(mol_atom_edge)
            
        return atom_bond_idxs
            
    def _get_frag_atom_start_idx(self, frag_ids: torch.Tensor) -> List[int]:
        curr_start_idx = 0
        start_idxs = []
        for frag_id in frag_ids:
            start_idxs.append(curr_start_idx)
            curr_start_idx += len(self.frag_id_to_atoms[frag_id.item()])
        
        return start_idxs


class FragSamplingMetrics(nn.Module):
    '''
    Module for computing statistics between the generated graphs and test graphs
    '''
    def __init__(self, dataloaders, metrics_list=[]):
        super().__init__()
        self.metrics_list = metrics_list

    def forward(self, generated_graphs: list, name, current_epoch, val_counter, save_graphs=True, test=False):
        '''
        Compare generated_graphs list with test graphs
        '''
        if 'example_metric' in self.metrics_list:
            print("Computing example_metric stats..")
            # example_metric = compute_example_metric()
            # wandb.run.summary['example_metric'] = example_metric

    def reset(self):
        pass
