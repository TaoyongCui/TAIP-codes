import pdb
import numpy as np
import torch
import argparse
import yaml
import os
from dig.threedgraph.evaluation import ThreeDEvaluator as threedEvaluator
from tqdm import tqdm
from torch import nn
from torch_geometric.loader import DataLoader
from torch.optim import Adam, AdamW
import torch.nn.functional as F
from torch.autograd import grad
from torch.optim.lr_scheduler import StepLR
from TAIP.PaiNN import PainnModel as PaiNN
from TAIP.decoder import *
import math
from torch_scatter import scatter_add, scatter_max, scatter_min, scatter_mean
from TAIP_test import EquivariantDenoisePred
from easydict import EasyDict
save_dir = './checkpoint/'

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0")
epochs = 1

evaluation = threedEvaluator()
def extractor_from_layer(rep):
    layers = [rep.atom_embedding, rep.message_layers[0], rep.update_layers[0],rep.message_layers[1],rep.update_layers[1],rep.message_layers[2],rep.update_layers[2]]
    return nn.Sequential(*layers)


class ExtractorHead(nn.Module):
    def __init__(self, head, ext):
        super(ExtractorHead, self).__init__()
        self.ext = ext
        self.head = head
        self.cutoff = 6.0
        self.hidden_state_size = 128

    def forward(self, z, pos, batch_data):


        edge_index, cell_offsets, _, neighbors = radius_graph_pbc(
            data = batch_data, radius = self.cutoff, max_num_neighbors_threshold = 500
        )
        batch_data.edge_index = edge_index
        batch_data.cell_offsets = cell_offsets
        batch_data.neighbors = neighbors

        assert z.dim() == 1 and z.dtype == torch.long
        
        out = get_pbc_distances(
            pos,
            batch_data.edge_index,
            batch_data.cell,
            batch_data.cell_offsets,
            batch_data.natoms,
            return_distance_vec=True,
        )

        input_dict = batch_data
        edge = out["edge_index"].t()
        edge_diff = out['distance_vec']

        edge_dist = out["distances"]
        node_scalar = self.ext[0](z)
        node_vector = torch.zeros((input_dict.pos.shape[0], 3, self.hidden_state_size),
                                  device=edge_diff.device,
                                  dtype=edge_diff.dtype,
                                 )

        node_scalar, node_vector = self.ext[1](node_scalar, node_vector, edge, edge_diff, edge_dist)
        node_scalar, node_vector = self.ext[2](node_scalar, node_vector)
        node_scalar, node_vector = self.ext[3](node_scalar, node_vector, edge, edge_diff, edge_dist)
        node_scalar, node_vector = self.ext[4](node_scalar, node_vector)
        node_scalar, node_vector = self.ext[5](node_scalar, node_vector, edge, edge_diff, edge_dist)
        node_scalar, node_vector = self.ext[6](node_scalar, node_vector)
        v = self.head(node_scalar)

        return v,batch_data.pos,out["edge_index"],out["distances"]

def updated_model(test_datasets, net):
    test_datasets.pos.requires_grad_(True)
    # print(test_datasets)
    # exit()

    xl, _,_,_=net.model( test_datasets.x[:,0].long(),test_datasets.pos,test_datasets)
    
    xl = net.node_dec(xl)
    xg = scatter_add(xl, test_datasets.batch, dim = -2)
    energy_pre2 = net.graph_dec(xg)
    force_pre = -grad(outputs=energy_pre2, inputs=test_datasets.pos, grad_outputs=torch.ones_like(energy_pre2),create_graph=True,retain_graph=True)[0]

    return energy_pre2,force_pre

def train(net, ssh, optimizer, batch_data, device):
      
    net.train()
    ssh.train()


    optimizer.zero_grad()

    train_batch_data = batch_data.to(device)
    train_batch_data.cell =train_batch_data.cell.reshape(-1,3,3)


    
    loss_pred_noise, fm_loss, m_loss = net(train_batch_data)
    loss = fm_loss + m_loss + loss_pred_noise
    
    loss.backward()
    optimizer.step()



    e, f= updated_model(train_batch_data, net=net)
    result_dict = {'energy': e}
    result_dict['forces'] =  f

    return result_dict

from ase.calculators.calculator import Calculator, all_changes
import numpy as np
class MLCalculator_schnet(Calculator):
    implemented_properties = ["energy", "forces", "force_max"]

    def __init__(
        self,
        net,
        ssh,
        head,
        rep2,
        device,
        learning_rate = 0.000005,
        energy_scale=1,
        forces_scale=1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.net = net
        self.ssh = ssh
        self.head = head
        self.rep2 = rep2
        self.device = device
        self.learning_rate = learning_rate
        # self.model_device = next(model.parameters()).device
        # self.cutoff = model.cutoff

        self.energy_scale = energy_scale
        self.forces_scale = forces_scale
#        self.stress_scale = stress_scale

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)
        if atoms is not None:
            self.atoms = atoms.copy()       
        from torch_geometric.data import Data, DataLoader
        import torch

        x = torch.as_tensor(atoms.numbers, dtype=torch.int)
        b=torch.zeros_like(x)
        edge_attr=torch.stack((x,b),1).squeeze(-1)
        data = Data(
            pos=torch.as_tensor(self.atoms.positions,dtype=torch.float32),
            z=torch.as_tensor(atoms.numbers),
            x=edge_attr,
            cell=torch.as_tensor(atoms.cell[:]).unsqueeze(0),
            natoms=len(atoms.numbers),
        )
        data_list =[]
        data_list.append(data)
        atomic_data = DataLoader(data_list,4,shuffle=False)
        batch_data=list(atomic_data)[0]
                
            
        parameters = list(self.net.model.parameters())+list(self.head.parameters())+list(self.net.node_dec.parameters())+list(self.net.graph_dec.parameters())+list(self.net.noise_pred.parameters())

        loss_func = nn.L1Loss()

        optimizer = AdamW(self.ssh.parameters(), lr=self.learning_rate, weight_decay=0.0)

        scheduler = StepLR(optimizer, step_size=150, gamma=0.5)

        sigmas = torch.tensor(
        np.exp(np.linspace(np.log(10), np.log(0.01),
                            50)), dtype=torch.float32)

        sigmas = nn.Parameter(sigmas, requires_grad=False)
        energy_and_force=True
        model_results = train(self.net, self.net, optimizer, batch_data, 
                              self.device)

        results = {}
        results["forces"] = (
            model_results["forces"].detach().cpu().numpy() * self.forces_scale
        )
        results["energy"] = (
            model_results["energy"].detach().cpu().numpy().item()
            * self.energy_scale
        )
        results["force_max"] = (
            torch.max(abs(model_results["forces"])).detach().cpu().numpy() * self.forces_scale
        )
        if model_results.get("fps"):
            atoms.info["fps"] = model_results["fps"].detach().cpu().numpy()
    
        self.results = results




























