import pdb
import numpy as np
import torch
import argparse
import yaml
import os
from dig.threedgraph.evaluation import ThreeDEvaluator
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
from TAIP.TAIP_test import EquivariantDenoisePred
from easydict import EasyDict
save_dir = './checkpoint/'

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda")
epochs = 1

evaluation = ThreeDEvaluator()
def extractor_from_layer(rep):
    layers = [rep.atom_embedding, rep.message_layers[0], rep.update_layers[0],rep.message_layers[1],rep.update_layers[1],rep.message_layers[2],rep.update_layers[2]]
    return nn.Sequential(*layers)


class ExtractorHead(nn.Module):
    """
    ExtractorHead is a neural network module that processes node features 
    and positional information to extract relevant features for further 
    processing in a graph-based model.

    Attributes:
        ext (list): A list of extraction modules used to compute node features.
        head (nn.Module): A final head module that outputs predictions.
        cutoff (float): The cutoff distance for considering neighbors in the graph.
        hidden_state_size (int): The size of the hidden state for internal representations.
        pdb (bool): A flag to toggle between periodic boundary conditions and standard graph processing.

    Parameters:
        head (nn.Module): The final head module for predictions.
    """

    def __init__(self, head):
        """
        Initializes the ExtractorHead with the given head module.

        Args:
            head (nn.Module): The final head module for predictions.
        """
        super(ExtractorHead, self).__init__()
        self.ext = ext
        self.head = head
        self.cutoff = 6.0
        self.hidden_state_size = 128
        self.pdb = True

    def forward(self, z, pos, batch_data):
        """
        Forward pass to compute the output based on input features and positions.

        Args:
            z (torch.Tensor): A tensor of node features (scalar values).
            pos (torch.Tensor): A tensor of node positions in 3D space.
            batch_data (BatchData): An object containing data for graph processing.

        Returns:
            tuple: A tuple containing:
                - v (torch.Tensor): The final output predictions from the head module.
                - torch.Tensor: The positions of the nodes.
                - torch.Tensor: The edge indices of the graph.
                - torch.Tensor: The distances between connected nodes.

        Raises:
            AssertionError: If the dimensions of `z` are not as expected.
        """
        pos.requires_grad_()

        if self.pdb:
            edge_index, cell_offsets, _, neighbors = radius_graph_pbc(
                data=batch_data, radius=self.cutoff, max_num_neighbors_threshold=500
            )
            batch_data.edge_index = edge_index
            batch_data.cell_offsets = cell_offsets
            batch_data.neighbors = neighbors
            assert z.dim() == 1 and z.dtype == torch.long
            out = get_pbc_distances(
                batch_data.pos,
                batch_data.edge_index,
                batch_data.cell,
                batch_data.cell_offsets,
                batch_data.natoms,
            )
            input_dict = batch_data
            edge = out["edge_index"].t()
            edge_diff = out['distance_vec']
            edge_dist = out["distances"]
        else:
            edge_index = radius_graph(pos, r=self.cutoff, batch=batch_data.batch)
            row, col = edge_index
            edge = edge_index.T
            edge_diff = (pos[row] - pos[col])
            edge_dist = (pos[row] - pos[col]).norm(dim=-1)

        node_scalar = self.ext[0](z)
        node_vector = torch.zeros((pos.shape[0], 3, self.hidden_state_size),
                                  device=edge_diff.device,
                                  dtype=edge_diff.dtype)

        node_scalar, node_vector = self.ext[1](node_scalar, node_vector, edge, edge_diff, edge_dist)
        node_scalar, node_vector = self.ext[2](node_scalar, node_vector)
        node_scalar, node_vector = self.ext[3](node_scalar, node_vector, edge, edge_diff, edge_dist)
        node_scalar, node_vector = self.ext[4](node_scalar, node_vector)
        node_scalar, node_vector = self.ext[5](node_scalar, node_vector, edge, edge_diff, edge_dist)
        node_scalar, node_vector = self.ext[6](node_scalar, node_vector)
        v = self.head(node_scalar)

        return v, batch_data.pos, edge_index, edge_dist

def updated_model(test_datasets):
    test_datasets.pos.requires_grad_(True)

    xl, _,_,_=net.model( test_datasets.z.long(),test_datasets.pos,test_datasets)
    
    xl = net.node_dec(xl)
    xg = scatter_add(xl, test_datasets.batch, dim = -2)
    energy_pre2 = net.graph_dec(xg)
    force_pre = -grad(outputs=energy_pre2, inputs=test_datasets.pos, grad_outputs=torch.ones_like(energy_pre2),create_graph=True,retain_graph=True)[0]
    e_mae = nn.L1Loss()(energy_pre2.detach(), (test_datasets.energy.unsqueeze(1)).detach())
    f_mae = nn.L1Loss()(force_pre.detach(), test_datasets.force.detach())
    return e_mae,f_mae


parser = argparse.ArgumentParser(description='mgp')
parser.add_argument('--config_path', type=str, default = './config.yaml',help='path of dataset', required=True)
parser.add_argument('--dataset', type=str, default='processed/water_test.pt', help='overwrite config seed')

args = parser.parse_args()
with open(args.config_path, 'r') as f:
    config = yaml.safe_load(f)
config = EasyDict(config)



rep = PaiNN(num_interactions=3, hidden_state_size=128, cutoff=6.0).to(device)

energy_and_force = True
P = 1000
ext = extractor_from_layer(rep).to(device)
head = nn.Sequential(
            nn.Linear(128, 128), 
            nn.SiLU(),
            nn.Linear(128, 128),
        ).to(device)

ckpt = torch.load('checkpoint/TAIP_water.pt',map_location='cpu')        
rep.load_state_dict(ckpt['model'])
ssh = ExtractorHead(head).to(device)
net = EquivariantDenoisePred(config, rep, ssh).to(device)

net.model.load_state_dict(ckpt['model'])
net.graph_dec.load_state_dict(ckpt['graph_dec'])
net.node_dec.load_state_dict(ckpt['node_dec'])
head.load_state_dict(ckpt['head'])
net.noise_pred.load_state_dict(ckpt['noise_pred'])
net.decoder.load_state_dict(ckpt['decoder'])
net.decoder_force.load_state_dict(ckpt['decoder_force'])





test_dataset = torch.load(args.dataset)[:500]

parameters = list(net.model.parameters())+list(head.parameters())+list(net.node_dec.parameters())+list(net.graph_dec.parameters())+list(net.noise_pred.parameters())

loss_func = nn.L1Loss()

optimizer = AdamW(ssh.parameters(), lr=0.00001, weight_decay=0.0)

scheduler = StepLR(optimizer, step_size=150, gamma=0.5)


test_loader = DataLoader(test_dataset, 1, shuffle=True)


best_valid = float('inf')
best_e = float('inf')
best_f = float('inf')
best_test = float('inf')


sigmas = torch.tensor(
np.exp(np.linspace(np.log(10), np.log(0.01),
                    50)), dtype=torch.float32)

sigmas = nn.Parameter(sigmas, requires_grad=False)

force = []
energy = [] 

def train(net, ssh, optimizer, test_loader, energy_and_force, loss_func, device, steps):
  
    net.train()
    ssh.train()
    loss_accum = 0
    tensor1 = []
    tensor2= []
    for step, batch_data in enumerate(tqdm(test_loader)):
        optimizer.zero_grad()
        net.model.load_state_dict(ckpt['model'])
        train_batch_data = batch_data.to(device)
        train_batch_data.cell =train_batch_data.cell.reshape(-1,3,3)


        
        loss_pred_noise, fm_loss, m_loss = net(train_batch_data)
        loss = fm_loss + m_loss + loss_pred_noise
        
        loss.backward()
        optimizer.step()



        e_mae, f_mae= updated_model(train_batch_data)

        force.append(f_mae)
        energy.append(e_mae)
        loss_accum += loss.detach().cpu().item()
    print("energy mea:", sum(energy)/len(energy))
    print("force mea:", sum(force)/len(force))

    return loss_accum / (step + 1)





for epoch in range(1, epochs + 1):

    
    print('\nTesting...', flush=True)
    train_mae = train(net, net, optimizer, test_loader, energy_and_force, loss_func, device,steps=epoch)





























