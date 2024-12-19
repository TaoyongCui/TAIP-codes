# This code is used to process data for water and ice
from sys import argv
import ase
import torch
from torch_geometric.data.in_memory_dataset import InMemoryDataset
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
import torch.nn.functional as F
from torch_geometric.nn import radius_graph
import numpy as np
from ase.atoms import Atoms
from ase.io import read




if __name__ == '__main__':
    file2=argv[2]
    import os
    import pdb

    atoms_all_list = []
    atoms_all = read(argv[1], index=slice(None), format='extxyz')#[:100]
    atoms_all_list.append(atoms_all)
#
    data_list =[]
    for i in range(len(atoms_all_list)):
        for i, atoms in enumerate(atoms_all_list[i]):
           
            #shifting the potential energy by atomic energies in vaccum#
            atom_o = atoms.get_chemical_symbols().count('O')
            atom_h = atoms.get_chemical_symbols().count('H')
            energy_shift = atom_o*(-407.84007125944) + atom_h*(-0.16108140)
            ############################################################

            natom = len(atoms)
            pos_arr = atoms.positions

            pos = pos_arr.tolist()
            pbc_index = list(range(natom))
            pbc_image = [[0, 0, 0] for _ in range(natom)]
            atoms_numbers = atoms.numbers.tolist()
            nbr_dist = [[[i, 0.0]] for i in range(natom)]

            type_idx = []
            type_map = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Na': 5, 'P': 6}
            x = torch.as_tensor(atoms.numbers, dtype=torch.int)
            cell = torch.as_tensor(atoms.cell[:])
            pos=torch.as_tensor(atoms.positions, dtype=torch.float)


            data = Data(
                pos=pos,
                z=x,
                energy=torch.as_tensor(atoms.get_potential_energy() - energy_shift, dtype=torch.float),
                force=torch.as_tensor(atoms.get_forces(), dtype=torch.float),
                natoms=len(atoms.numbers),
                cell=cell
            )
            data_list.append(data)


    torch.save(data_list, file2)

