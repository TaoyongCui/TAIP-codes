# For testtime MD
# last modified TangChenyu 2024/1/10 22:01
import pdb
import numpy as np
import torch
import argparse
import yaml
import os
from dig.threedgraph.evaluation import threedEvaluator as ThreeDEvaluator
from tqdm import tqdm
from torch import nn
from torch_geometric.loader import DataLoader
from torch.optim import Adam, AdamW
import torch.nn.functional as F
from torch.autograd import grad
from torch.optim.lr_scheduler import StepLR
from AAAI.PaiNN import PainnModel as PaiNN
from AAAI.SchNet import *
import math
from AAAI_base import EquivariantDenoisePred
from easydict import EasyDict
from torch_scatter import scatter_add, scatter_max, scatter_min, scatter_mean
save_dir = './checkpoint/'

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0")
epochs = 1
evaluation = ThreeDEvaluator()

def extractor_from_layer(rep):
    layers = [rep.init_v, rep.update_es[0], rep.update_vs[0],rep.update_es[1],rep.update_vs[1],rep.update_es[2],rep.update_vs[2],rep.update_es[3],rep.update_vs[3],rep.update_es[4],rep.update_vs[4],rep.update_es[5],rep.update_vs[5]]
    return nn.Sequential(*layers)

class ExtractorHead(nn.Module):
    def __init__(self, head, ext):
        super(ExtractorHead, self).__init__()
        self.ext = ext
        self.head = head
        self.cutoff = 6.0
        self.dist_emb = emb(0.0, 6.0, 50)

    def forward(self, z, pos, batch_data):

        #pos.requires_grad_()
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
        )


        dist = out["distances"]
        dist_emb = self.dist_emb(dist.to(torch.long)) 
        v = self.ext[0](z.to(torch.long))

        e = self.ext[1](v.to(torch.float), dist.to(torch.float), dist_emb.to(torch.float), edge_index)
        v = self.ext[2](v.to(torch.float), e , edge_index)
        e = self.ext[3](v.to(torch.float), dist.to(torch.float), dist_emb.to(torch.float), edge_index)
        v = self.ext[4](v.to(torch.float), e , edge_index)
        e = self.ext[5](v.to(torch.float), dist.to(torch.float), dist_emb.to(torch.float), edge_index)
        v = self.ext[6](v.to(torch.float), e , edge_index)
        v = self.head(v)

        #e = scatter(v, batch, dim=0)

        return v,pos,edge_index,dist

def train(net, ssh, optimizer, optimizer2, optimizer3, scheduler, scheduler2, scheduler3, test_batch_data, train_batch_data, 
          energy_and_force, loss_func, device, rep2):
  
    net.train()
    ssh.train()
    loss_accum = 0
    tensor1 = []
    tensor2= []
    optimizer.zero_grad()
    optimizer2.zero_grad()
    optimizer3.zero_grad()

    train_batch_data = train_batch_data.to(device)
    train_batch_data.cell = train_batch_data.cell.reshape(-1,3,3)

    loss_denoise, loss_pred_noise,fmaskloss,m_loss = net(train_batch_data)

    loss = fmaskloss + loss_pred_noise

    loss.backward()
    optimizer.step()
    optimizer2.step()
    optimizer3.step()
    loss_accum += loss.detach().cpu().item()
    result_dict = rep2(test_batch_data)

    # ckpt = torch.load("tta_1_20_schnet.pt",map_location=device) 
    # graph_weights_dict = {}
    # net.graph_dec.load_state_dict(ckpt['graph_dec'],strict=False)
    # net.node_dec.load_state_dict(ckpt['node_dec'],strict=False)
    # net.noise_pred.load_state_dict(ckpt['noise_pred'],strict=False)
    # net.decoder.load_state_dict(ckpt['decoder'],strict=False)
    # net.decoder_force.load_state_dict(ckpt['decoder_force'],strict=False)
    # for k, v in net.graph_dec.state_dict().items():
    #     new_k = 'graph_dec.'+k
    #     graph_weights_dict[new_k] = v
    # node_weights_dict = {}
    # for k, v in net.node_dec.state_dict().items():
    #     new_k = 'node_dec.'+k
    #     node_weights_dict[new_k] = v
    
    # rep2.load_state_dict(node_weights_dict,strict=False)
    # rep2.load_state_dict(graph_weights_dict,strict=False)

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
        training_rate = 0.00005,
        energy_scale=1,
        forces_scale=1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.net = net
        self.ssh = ssh
        self.head = head
        self.rep2 = rep2
        self.training_rate = training_rate
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
        data = Data(
            pos=torch.as_tensor(self.atoms.positions,dtype=torch.float32),
            z=torch.as_tensor(atoms.numbers),
            cell=torch.as_tensor(atoms.cell[:]).unsqueeze(0),
            # energy=energy,
            # force=torch.as_tensor(forces, dtype=torch.float64),
            natoms=len(atoms.numbers),
        )
        data_list =[]
        data_list.append(data)
        atomic_data = DataLoader(data_list,4,shuffle=False)
        batch_data=list(atomic_data)[0]
                
        parameters = list(self.net.model.parameters())+list(self.head.parameters())+list(self.net.node_dec.parameters())+list(self.net.graph_dec.parameters())+list(self.net.noise_pred.parameters())

        loss_func = nn.L1Loss()

        optimizer = AdamW(parameters, lr=self.training_rate, weight_decay=0.0)
        optimizer2 = AdamW(self.net.decoder.parameters(), lr=0.0005, weight_decay=0.0)
        optimizer3 = AdamW(self.net.decoder_force.parameters(), lr=0.0005, weight_decay=0.0)
        scheduler = StepLR(optimizer, step_size=150, gamma=0.5)
        scheduler2 = StepLR(optimizer2, step_size=150, gamma=0.5)
        scheduler3 = StepLR(optimizer3, step_size=150, gamma=0.5)

        sigmas = torch.tensor(
        np.exp(np.linspace(np.log(10), np.log(0.01),
                            50)), dtype=torch.float32)

        sigmas = nn.Parameter(sigmas, requires_grad=False)
        energy_and_force=True
        model_results = train(self.net, self.net, optimizer, optimizer2, optimizer3, scheduler, scheduler2, scheduler3, batch_data, 
                              batch_data, energy_and_force, loss_func, device, self.rep2)

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






















