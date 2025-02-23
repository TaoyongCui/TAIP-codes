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
from TAIP.TAIP_trainMD17 import EquivariantDenoisePred
from easydict import EasyDict
save_dir = './checkpoint/'
import random  
  
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
        assert z.dim() == 1 and z.dtype == torch.long

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
        # node_scalar, node_vector = self.ext[5](node_scalar, node_vector, edge, edge_diff, edge_dist)
        # node_scalar, node_vector = self.ext[6](node_scalar, node_vector)
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

rep = PaiNN(num_interactions=config.model.n_layers, hidden_state_size=config.model.hidden_dim, cutoff=6.0,pdb=False).to(device)
ext = extractor_from_layer(rep).to(device)
head = nn.Sequential(
            nn.Linear(config.model.hidden_dim, config.model.hidden_dim), 
            nn.SiLU(),
            nn.Linear(config.model.hidden_dim, config.model.hidden_dim),
        ).to(device)
ssh = ExtractorHead(head,config.model.hidden_dim).to(device)
net = EquivariantDenoisePred(config, rep, ssh).to(device)


#Water dataset


parameters = list(net.model.parameters())+list(head.parameters())+list(net.node_dec.parameters())+list(net.graph_dec.parameters())+list(net.noise_pred.parameters())

loss_func = nn.L1Loss()

optimizer = AdamW(parameters, lr=config.train.optimizer.lr, weight_decay=0.0)
optimizer2 = AdamW(net.decoder.parameters(), lr=config.train.optimizer.lr, weight_decay=0.0)
optimizer3 = AdamW(net.decoder_force.parameters(), lr=config.train.optimizer.lr, weight_decay=0.0)
scheduler = StepLR(optimizer, step_size=150, gamma=0.5)
scheduler2 = StepLR(optimizer2, step_size=150, gamma=0.5)
scheduler3 = StepLR(optimizer3, step_size=150, gamma=0.5)

from MD17 import MD17
dataset = MD17(root='./', name='aspirin')
split_idx = dataset.get_idx_split(len(dataset.data.energy), train_size=1000, valid_size=1000, seed=42)
train_dataset, valid_dataset = dataset[split_idx['train']], dataset[split_idx['valid']]

train_loader = DataLoader(train_dataset, config.train.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, config.train.batch_size*16, shuffle=False)

best_valid = float('inf')



force = []
energy = [] 

def train(net, ssh, optimizer, train_loader, energy_and_force, loss_func, device, steps):
  
    net.train()
    ssh.train()
    loss_accum = 0
    for step, batch_data in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()

        train_batch_data = batch_data.to(device)

        loss_pred_noise, fm_loss, m_loss,e_loss,f_loss = net(train_batch_data)
        loss = fm_loss + m_loss + loss_pred_noise + e_loss + P * f_loss
        
        loss.backward()
        optimizer.step()
        optimizer2.step()
        optimizer3.step()
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

        loss_pred_noise, fm_loss, m_loss,e_loss,f_loss = net(train_batch_data)
        loss = e_loss + P * f_loss

        loss_accum += loss.detach().cpu().item()
        e_losses += e_loss.detach().cpu().item()
        f_losses += f_loss.detach().cpu().item()
        den_losses = m_loss

   

        
    return e_losses/ (step + 1), f_losses/ (step + 1),den_losses, loss_accum / (step + 1)


for epoch in range(1, epochs + 1):
    print("\n=====Epoch {}".format(epoch), flush=True)
    
    print('\nTraining...', flush=True)
    train_mae = train(net, net, optimizer, train_loader, energy_and_force, loss_func, device,steps=epoch)



    e_mae, f_mae, d_mae, valid_mae = val(net, net, valid_loader, energy_and_force, loss_func, device, steps=epoch)

    print({'e_mae': e_mae})
    print({'f_mae': f_mae})
    print({'d_mae': d_mae})
  
    if valid_mae < best_valid:
        best_valid = valid_mae
        if save_dir != '':
            print('Saving checkpoint...')
            checkpoint = {'epoch': epoch, 'net': rep.state_dict(), 'head': head.state_dict(),'noise_pred':net.noise_pred.state_dict(),'graph_dec':net.graph_dec.state_dict(),'node_dec':net.node_dec.state_dict(),'model':net.model.state_dict(),'decoder_force':net.decoder_force.state_dict(),'decoder':net.decoder.state_dict()}
            torch.save(checkpoint, os.path.join(save_dir, 'TAIP_MD17_aspirin_new.pt'))
    scheduler.step()
    scheduler2.step()
    scheduler3.step()





























