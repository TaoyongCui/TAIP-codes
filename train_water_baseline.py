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
from TAIP.TAIP_baseline import EquivariantDenoisePred
from easydict import EasyDict
save_dir = './checkpoint/'
import random  
from sys import argv  
# 设置随机种子  

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
energy_and_force = True
device = torch.device("cuda")
P = 1000 # or P = 100


evaluation = ThreeDEvaluator()
def extractor_from_layer(rep):
    layers = [rep.atom_embedding, rep.message_layers[0], rep.update_layers[0],rep.message_layers[1],rep.update_layers[1],rep.message_layers[2],rep.update_layers[2]]
    return nn.Sequential(*layers)
class ExtractorHead(nn.Module):
    """
    ExtractorHead is a neural network module that processes node features 
    and positional information to extract relevant features as a encoder for further 
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

    def __init__(self, head,hidden_state_size):
        """
        Initializes the ExtractorHead with the given head module.

        Args:
            head (nn.Module): The final head module for predictions.
        """
        super(ExtractorHead, self).__init__()
        self.ext = ext
        self.head = head
        self.cutoff = 6.0
        self.hidden_state_size = hidden_state_size

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

        edge = out["edge_index"].t()
        edge_diff = out['distance_vec']
        edge_dist = out["distances"]

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


parser = argparse.ArgumentParser(description='mgp')
parser.add_argument('--config_path', type=str, default = './config.yaml',help='path of dataset', required=True)
parser.add_argument('--seed', type=int, default=42, help='overwrite config seed')
args = parser.parse_args()
with open(args.config_path, 'r') as f:
    config = yaml.safe_load(f)
config = EasyDict(config)
epochs = config.train.epochs

rep = PaiNN(num_interactions=config.model.n_layers, hidden_state_size=config.model.hidden_dim, cutoff=6.0).to(device)
ext = extractor_from_layer(rep).to(device)
head = nn.Sequential(
            nn.Linear(config.model.hidden_dim, config.model.hidden_dim), 
            nn.SiLU(),
            nn.Linear(config.model.hidden_dim, config.model.hidden_dim),
        ).to(device)
ssh = ExtractorHead(head,config.model.hidden_dim).to(device)
net = EquivariantDenoisePred(config, rep, ssh).to(device)


#Water dataset
train_dataset = torch.load('processed/water_train.pt')
valid_dataset = torch.load('processed/water_valid.pt')
print(train_dataset[0])

parameters = list(net.model.parameters())+list(head.parameters())+list(net.node_dec.parameters())+list(net.graph_dec.parameters())+list(net.noise_pred.parameters())

loss_func = nn.L1Loss()

optimizer = AdamW(parameters, lr=config.train.optimizer.lr, weight_decay=0.0)

scheduler = StepLR(optimizer, step_size=150, gamma=0.5)


train_loader = DataLoader(train_dataset, config.train.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, config.train.batch_size, shuffle=False)

best_valid = float('inf')


sigmas = torch.tensor(
np.exp(np.linspace(np.log(10), np.log(0.01),
                    50)), dtype=torch.float32)

sigmas = nn.Parameter(sigmas, requires_grad=False)

force = []
energy = [] 

def train(net, ssh, optimizer, train_loader, energy_and_force, loss_func, device, steps):
  
    net.train()
    ssh.train()
    loss_accum = 0
    for step, batch_data in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()


        train_batch_data = batch_data.to(device)
        train_batch_data.cell =train_batch_data.cell.reshape(-1,3,3)

        e_loss,f_loss = net(train_batch_data)
        loss = e_loss + P * f_loss
        
        loss.backward()


        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)


def val(net, ssh, valid_loader, energy_and_force, loss_func, device, steps):
  
    net.train()
    ssh.train()
    loss_accum = 0
    e_losses = 0
    f_losses = 0
    den_losses = 0
    for step, batch_data in enumerate(tqdm(valid_loader)):

        batch_data = batch_data.to(device)

        train_batch_data = batch_data.to(device)
        train_batch_data.cell =train_batch_data.cell.reshape(-1,3,3)

        e_loss,f_loss = net(train_batch_data)
        loss = e_loss + P * f_loss

        loss_accum += loss.detach().cpu().item()
        e_losses += e_loss.detach().cpu().item()
        f_losses += f_loss.detach().cpu().item()
        # den_losses = m_loss

   

        
    return e_losses/ (step + 1), f_losses/ (step + 1), loss_accum / (step + 1)


for epoch in range(1, epochs + 1):
    print("\n=====Epoch {}".format(epoch), flush=True)
    
    print('\nTraining...', flush=True)
    train_mae = train(net, net, optimizer, train_loader, energy_and_force, loss_func, device,steps=epoch)



    e_mae, f_mae, valid_mae = val(net, net, valid_loader, energy_and_force, loss_func, device, steps=epoch)

    print({'e_mae': e_mae})
    print({'f_mae': f_mae})

  
    if valid_mae < best_valid:
        best_valid = valid_mae
        if save_dir != '':
            print('Saving checkpoint...')
            checkpoint = {'epoch': epoch, 'net': rep.state_dict(), 'head': head.state_dict(),'noise_pred':net.noise_pred.state_dict(),'graph_dec':net.graph_dec.state_dict(),'node_dec':net.node_dec.state_dict(),'model':net.model.state_dict(),'decoder_force':net.decoder_force.state_dict(),'decoder':net.decoder.state_dict()}
            torch.save(checkpoint, os.path.join(save_dir, 'baseline_water.pt'))
    scheduler.step()






























