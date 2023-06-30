import hydra
import omegaconf
import torch
import pandas as pd
import numpy as np
from omegaconf import ValueNode

from torch.utils.data import Dataset
from torch_geometric.data import Data

from p_tqdm import p_umap

# from chggen.common.utils import PROJECT_ROOT
from chggen.common.data_utils import (
    preprocess, preprocess_tensors, add_scaled_lattice_prop)


from pymatgen.core import Structure
from chgnet.graph.converter import CrystalGraphConverter

crys_converter = CrystalGraphConverter(atom_graph_cutoff= 5, 
                                       bond_graph_cutoff= 3)

def process_csv(input_file, num_workers, niggli, primitive, prop_list):
    df = pd.read_csv(input_file)
    def process_one(row, niggli, primitive, prop_list):
        crystal_str = row['cif']
        structure = Structure.from_str(crystal_str, fmt = 'cif')
        crys_graph = crys_converter(structure)

        properties = {k: row[k] for k in prop_list if k in row.keys()}
        result_dict = {
            'mp_id': row['material_id'],
            'cif': crystal_str,
            'lengths': np.array(structure.lattice.lengths),
            'angles': np.array(structure.lattice.angles), 
            'num_atoms': structure.num_sites,
            'crys_graph': crys_graph,
        }
        result_dict.update(properties)
        return result_dict

    unordered_results = p_umap(
        process_one,
        [df.iloc[idx] for idx in range(len(df))],
        [niggli] * len(df),
        [primitive] * len(df),
        [prop_list] * len(df),
        num_cpus=num_workers)

    mpid_to_results = {result['mp_id']: result for result in unordered_results}
    ordered_results = [mpid_to_results[df.iloc[idx]['material_id']]
                       for idx in range(len(df))]

    return ordered_results


class CHGNetDataset(Dataset):
    def __init__(self, name: str, path: str,
                 prop_list: list, niggli: bool = True, primitive: bool = False,
                 preprocess_workers: int = 16,
                 lattice_scale_method: str = 'scale_length',
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop_list = prop_list
        self.niggli = niggli
        self.primitive = primitive
        # self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = process_csv(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            prop_list= prop_list)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        prop_list = []
        for key in self.prop_list:
            prop = data_dict[key]
            prop_list.append(prop)

        crys_graph = data_dict['crys_graph']


        # (frac_coords, atom_types, lengths, angles, edge_indices,
        #  to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(crys_graph = crys_graph, y = torch.Tensor(prop_list).view(1, -1),
        )
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"




class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode,
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=[prop])

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        prop = self.scaler.transform(data_dict[self.prop])
        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
        )
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


# @hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
# def main(cfg: omegaconf.DictConfig):
#     from torch_geometric.data import Batch
#     from chggen.common.data_utils import get_scaler_from_data_list
#     dataset: CrystDataset = hydra.utils.instantiate(
#         cfg.data.datamodule.datasets.train, _recursive_=False
#     )
#     lattice_scaler = get_scaler_from_data_list(
#         dataset.cached_data,
#         key='scaled_lattice')
#     scaler = get_scaler_from_data_list(
#         dataset.cached_data,
#         key=dataset.prop)

#     dataset.lattice_scaler = lattice_scaler
#     dataset.scaler = scaler
#     data_list = [dataset[i] for i in range(len(dataset))]
#     batch = Batch.from_data_list(data_list)
#     return batch


# if __name__ == "__main__":
#     main()
