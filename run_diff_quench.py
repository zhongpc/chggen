from pathlib import Path
from datetime import datetime
from typing import List
from types import SimpleNamespace
import pickle
import numpy as np
import torch
from pymatgen.core import Structure, Lattice, Element


from chggen.pl_data.dataset import CHGNetDataset
from chggen.pl_modules.model_volume import CHGGen
from chggen.common.data_utils import get_scaler_from_data_list, get_scaler


import os

def mkdir(path: str):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("Folder exists")
    return path


device = torch.device('cuda')


with open('./test_models/lattice_scaler_perov', 'rb') as fp:
    lattice_scaler = pickle.load(fp)

chggen = CHGGen.load_from_checkpoint('./test_models/perov/epoch=4.ckpt')
chggen.to(device = device)

chggen.lattice_scaler = lattice_scaler




# langevin dynamics
ld_kwargs = SimpleNamespace(n_step_each = 5,
                            step_lr = 1e-4,
                            min_sigma = 0,
                            save_traj = False,
                            disable_bar = False,
                            compute_force = True,
                            beta_c = 0, # property update rate
                            beta_f = 0.01, # atomic force update rate
                            )


num_structures = 1
z = torch.rand(num_structures, 64, requires_grad= True, device = device)
results = chggen.diffusion_quench_guidance(z = z, 
                                           prop_guidance = torch.tensor(-0.05, device= device), 
                                           box_lengths = [8,8,8], #  [5,5,5],
                                           box_angles = [90, 90, 90],
                                           # gt_num_atoms = torch.ones(num_structures, device = device, dtype = torch.int64) * 5, # 
                                           # box_lengths = 4.1*1,
                                           # box_angles = 90,
                                           change_type = False, #False, 
                                           ld_kwargs= ld_kwargs)

mkdir('./test_models/gen_structures/')



# save the results from langevin dynamics
lengths = results['lengths']
angles= results['angles']
num_atoms = results['num_atoms']
frac_coords = results['frac_coords']
atom_types = results['atom_types']

batch = torch.arange(len(num_atoms), device = device)
batch = batch.repeat_interleave(num_atoms)
print(num_atoms)
for ii in range(len(num_atoms)):
    indices = torch.where(batch == ii)[0]
    # print(ii, indices, )
    # print("composition", crys_graph.composition)
    print("num atoms: ", len(indices))

    if len(indices) == 0:
        continue
    
    
    Latt = Lattice.from_parameters(a = lengths.cpu().detach().numpy()[ii,0], 
                                   b = lengths.cpu().detach().numpy()[ii,1], 
                                   c = lengths.cpu().detach().numpy()[ii,2],
                                   alpha= angles.cpu().detach().numpy()[ii, 0], 
                                   beta = angles.cpu().detach().numpy()[ii,1], 
                                   gamma= angles.cpu().detach().numpy()[ii, 2])
                                   
    frac_ = frac_coords[indices]
    type_ = atom_types[indices]
    species_ = [Element.from_Z(ele_Z) for ele_Z in type_]
    
    s_gen = Structure(lattice= Latt , species= species_, coords= frac_.cpu().detach().numpy(),
                      to_unit_cell=False,coords_are_cartesian=False);
    s_gen.sort()
    # print("previou compo: ", crys_graph.composition)
    print("reconst compo: ", s_gen.composition)
    s_gen.to(filename= './test_models/gen_structures/gen_' + str(ii) + '.cif')
print("Done")