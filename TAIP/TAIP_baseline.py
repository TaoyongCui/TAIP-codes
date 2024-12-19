import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max, scatter_min, scatter_mean
from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from scipy.stats import truncnorm
from torch.autograd import grad
import math
import time
import random
from functools import partial
from dig.threedgraph.evaluation import ThreeDEvaluator
from TAIP.decoder import SchNetDecoder
torch.autograd.set_detect_anomaly(True)
loss_func = {
    "L1" : nn.L1Loss(),
    "L2" : nn.MSELoss(reduction='none'),
    "Cosine" : nn.CosineSimilarity(dim=-1, eps=1e-08),
    "CrossEntropy" : nn.CrossEntropyLoss(reduction='none')
}
def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss
criterion = partial(sce_loss, alpha=1.0)
class EquivariantDenoisePred(torch.nn.Module):
    """
    EquivariantDenoisePred is a neural network module designed for equivariant denoising 
    predictions in graph-based models. It integrates various models to predict noise levels 
    and forces acting on nodes, while also implementing perturbation strategies.

    Attributes:
        config (Config): Configuration object containing model parameters.
        hidden_dim (int): Dimensionality of hidden layers.
        edge_types (int): Number of edge types; determines if edge information is used.
        noise_type (str): Type of noise to be applied ('riemann', 'kabsch', or 'gaussian').
        pred_mode (str): Mode of prediction.
        model (torch.nn.Module): Representation model for node embeddings.
        ssh (torch.nn.Module): Model with shared parameters for additional processing.
        node_dec (nn.Sequential): Sequential model for node feature transformation.
        graph_dec (nn.Sequential): Sequential model for graph-level feature transformation.
        noise_pred (nn.Sequential): Sequential model for predicting noise levels.
        decoder (nn.Sequential): Sequential model for generating final outputs.
        sigmas (nn.Parameter): Predefined noise levels as learnable parameters.
        decoder_force (SchNetDecoder): Model for predicting forces based on node positions.

    Parameters:
        config (Config): Configuration object containing model parameters.
        rep_model (torch.nn.Module): Representation model for the initial processing of nodes.
        ssh_model (torch.nn.Module): Secondary model for additional processing.
    """

    def __init__(self, config, rep_model, ssh_model):
        """
        Initializes the EquivariantDenoisePred with the specified models and configuration.

        Args:
            config (Config): Configuration object containing model parameters.
            rep_model (torch.nn.Module): Representation model for node embeddings.
            ssh_model (torch.nn.Module): Secondary model for additional processing.
        """
        super(EquivariantDenoisePred, self).__init__()
        self.config = config
        self.hidden_dim = self.config.model.hidden_dim
        self.edge_types = 0 if config.model.no_edge_types else config.model.order + 1
        self.noise_type = config.model.noise_type
        self.pred_mode = config.model.pred_mode
        self.model = rep_model
        self.ssh = ssh_model
        self.node_dec = nn.Sequential(nn.Linear(self.hidden_dim  , self.hidden_dim),
                                      nn.SiLU(),
                                      nn.Linear(self.hidden_dim, self.hidden_dim))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                       nn.SiLU(),
                                       nn.Linear(self.hidden_dim, 1))

        self.noise_pred = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                                       nn.SiLU(),
                                       nn.Linear(self.hidden_dim, self.config.model.num_noise_level))
        
        self.decoder = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                       nn.SiLU(),
                                       nn.Linear(self.hidden_dim, 43))


        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_noise_level)), dtype=torch.float32)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False) # (num_noise_level)

        self.decoder_force = SchNetDecoder(num_layers=1, hidden_channels=256,out_channels=3)




    def get_energy_and_rep(self, x, pos, data, node2graph, return_pos=False, models=None):
        """
        Computes energy and representations for the given inputs.

        Args:
            x (torch.Tensor): Input node features.
            pos (torch.Tensor): Node positions in 3D space.
            data (Data): Input data object containing graph information.
            node2graph (torch.Tensor): Mapping from nodes to graphs.
            return_pos (bool): If True, return additional position data.
            models (Optional): Additional models for processing.

        Returns:
            tuple: Energy or representations, and optionally positional data if return_pos is True.
        """
        

        xl, posl,edge_index,distance = models(x, pos,data)


        xl = self.node_dec(xl)

        xg = scatter_add(xl, node2graph, dim = -2)

        e = self.graph_dec(xg)
        if return_pos:
            return xg.squeeze(-1), xg, posl,xl,edge_index,distance
        return e, xg

    @torch.no_grad()
    def get_distance(self, data: Data):
        """
        Computes distances between connected nodes and updates the edge lengths in the data object.

        Args:
            data (Data): Input data object containing node positions and edge indices.

        Returns:
            Data: Updated data object with edge lengths.
        """
        pos = data.pos
        row, col = data.edge_index
        d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1) # (num_edge, 1)
        data.edge_length = d
        return data

    @torch.no_grad()
    def truncated_normal(self, size, threshold=1):
        """
        Generates samples from a truncated normal distribution.

        Args:
            size (int): Number of samples to generate.
            threshold (float): Threshold for truncation.

        Returns:
            torch.Tensor: Samples drawn from the truncated normal distribution.
        """
        values = truncnorm.rvs(-threshold, threshold, size=size)
        return torch.from_numpy(values)
    @torch.no_grad()
    def mask(self, x, num_atom_type, mask_rate):
        """
        Masks a portion of the input features according to the specified mask rate.

        Args:
            x (torch.Tensor): Input features to be masked.
            num_atom_type (int): Number of atom types for one-hot encoding.
            mask_rate (float): Proportion of nodes to mask.

        Returns:
            tuple: Perturbed input features, node attribute labels, and indices of masked atoms.
        """

        

        num_atoms = x.size()[0]
        sample_size = int(num_atoms * mask_rate + 1)
        masked_atom_indices = random.sample(range(num_atoms), sample_size)
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(x[atom_idx].view(1, -1))
        mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        masked_atom_indices = torch.tensor(masked_atom_indices)

        atom_type = F.one_hot(mask_node_label[:, 0], num_classes=num_atom_type).float()
        node_attr_label = atom_type

        # modify the original node feature of the masked node
        for atom_idx in masked_atom_indices:
            x[atom_idx] = torch.tensor([num_atom_type])

        x_perturb = x

        return x_perturb,node_attr_label,masked_atom_indices
    @torch.no_grad()
    def get_force_target(self, perturbed_pos, pos, node2graph):
        """
        Computes the target forces based on the perturbed and original positions.

        Args:
            perturbed_pos (torch.Tensor): Perturbed node positions.
            pos (torch.Tensor): Original node positions.
            node2graph (torch.Tensor): Mapping from nodes to graphs.

        Returns:
            torch.Tensor: Target forces for the nodes.
        """
        # s = - (pos_p @ (pos_p.T @ pos_p) - pos @ (pos.T @ pos_p)) / (torch.norm(pos_p.T @ pos_p) + torch.norm(pos.T @ pos_p))
        if self.noise_type == 'riemann':
            v = pos.shape[-1]
            center = scatter_mean(pos, node2graph, dim = -2) # B * 3
            perturbed_center = scatter_mean(perturbed_pos, node2graph, dim = -2) # B * 3
            pos_c = pos - center[node2graph]
            perturbed_pos_c = perturbed_pos - perturbed_center[node2graph]
            perturbed_pos_c_left = perturbed_pos_c.repeat_interleave(v,dim=-1)
            perturbed_pos_c_right = perturbed_pos_c.repeat([1,v])
            pos_c_left = pos_c.repeat_interleave(v,dim=-1)
            ptp = scatter_add(perturbed_pos_c_left * perturbed_pos_c_right, node2graph, dim = -2).reshape(-1,v,v) # B * 3 * 3     
            otp = scatter_add(pos_c_left * perturbed_pos_c_right, node2graph, dim = -2).reshape(-1,v,v) # B * 3 * 3     
            ptp = ptp[node2graph]
            otp = otp[node2graph]
            tar_force = - 2 * (perturbed_pos_c.unsqueeze(1) @ ptp - pos_c.unsqueeze(1) @ otp).squeeze(1) / (torch.norm(ptp,dim=(1,2)) + torch.norm(otp,dim=(1,2))).unsqueeze(-1).repeat([1,3])
            return tar_force
        else:
            return pos - perturbed_pos

    @torch.no_grad()
    def gen_edge_onehot(self, edge_types):
        """
        Generates one-hot encoded representations for edge types.

        Args:
            edge_types (torch.Tensor): Tensor of edge types.

        Returns:
            torch.Tensor or None: One-hot encoded edge types, or None if edge types are not used.
        """
        if not self.edge_types:
            return None
        return F.one_hot(edge_types.long(), self.edge_types)

    @torch.no_grad()
    def fit_pos(self, perturbed_pos, pos, node2graph):
        """
        Aligns perturbed positions with the original positions using optimal rotation and translation.

        Args:
            perturbed_pos (torch.Tensor): Perturbed node positions.
            pos (torch.Tensor): Original node positions.
            node2graph (torch.Tensor): Mapping from nodes to graphs.

        Returns:
            torch.Tensor: Aligned perturbed positions.
        """
        v = pos.shape[-1]
        center = scatter_mean(pos, node2graph, dim = -2) # B * 3
        perturbed_center = scatter_mean(perturbed_pos, node2graph, dim = -2) # B * 3
        pos_c = pos - center[node2graph]
        perturbed_pos_c = perturbed_pos - perturbed_center[node2graph]
        pos_c = pos_c.repeat([1,v])
        perturbed_pos_c = perturbed_pos_c.repeat_interleave(v,dim=-1)
        H = scatter_add(pos_c * perturbed_pos_c, node2graph, dim = -2).reshape(-1,v,v) # B * 3 * 3
        U, S, V = torch.svd(H)
        # Rotation matrix
        R = V @ U.transpose(2,1)
        t = center - (perturbed_center.unsqueeze(1) @ R.transpose(2,1)).squeeze(1)
        R = R[node2graph]
        t = t[node2graph]
        p_aligned = (perturbed_pos.unsqueeze(1) @ R.transpose(2,1)).squeeze(1) + t
        return p_aligned

    @torch.no_grad()
    def perturb(self, pos, node2graph, used_sigmas, steps=1):
        """
        Perturbs the node positions using the specified noise type and parameters.

        Args:
            pos (torch.Tensor): Original node positions.
            node2graph (torch.Tensor): Mapping from nodes to graphs.
            used_sigmas (torch.Tensor): Noise levels to apply.
            steps (int): Number of steps for perturbation.

        Returns:
            torch.Tensor: Perturbed node positions.
        """
        if self.noise_type == 'riemann':
            pos_p = pos
            for t in range(1, steps + 1):
                alpha = 1 / (2 ** t)
                s = self.get_force_target(pos_p, pos, node2graph)
                pos_p = pos_p + alpha * s + torch.randn_like(pos) * math.sqrt(2 * alpha) * used_sigmas
            return pos_p
        elif self.noise_type == 'kabsch':
            pos_p = pos + torch.randn_like(pos) * used_sigmas
            pos_p = self.fit_pos(pos_p, pos, node2graph)
            return pos_p
        elif self.noise_type == 'gaussian':
            pos_p = pos + torch.randn_like(pos) * used_sigmas
            return pos_p


    def mask_force(self, force, mask_rate, used_sigmas):
        """
        Masks a portion of the force values according to the specified mask rate.

        Args:
            force (torch.Tensor): Force values to be masked.
            mask_rate (float): Proportion of forces to mask.
            used_sigmas (torch.Tensor): Noise levels to apply.

        Returns:
            tuple: Masked forces, list of masked node labels, and indices of masked atoms.
        """



        num_atoms = force.size()[0]
        sample_size = int(num_atoms * mask_rate + 1)
        masked_atom_indices = random.sample(range(num_atoms), sample_size)
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(force[atom_idx])
        masked_atom_indices = torch.tensor(masked_atom_indices)
        for atom_idx in masked_atom_indices:
            force[atom_idx] = force[atom_idx] + torch.randn_like(force[atom_idx])


        return force,mask_node_labels_list,masked_atom_indices
    def forward(self, data):
        """
        Performs a forward pass through the model.

        Args:
            data (Data): Input data object containing features and positions.

        Returns:
            tuple: Tuple of losses including force loss and energy loss.
        """
        self.device = self.sigmas.device
        
        node2graph = data.batch
        
        noise_level = torch.randint(0, self.sigmas.size(0), (data.num_graphs,), device=self.device) # (num_graph)
        used_sigmas = self.sigmas[noise_level] # (num_graph)

        used_sigmas = used_sigmas[node2graph].unsqueeze(-1) # (num_nodes, 1)

        pos = data.pos.clone()
        pos.requires_grad_(True)
        energy_pre, graph_rep_ori = self.get_energy_and_rep(data.z.long(), pos, data, node2graph, return_pos = False, models=self.model)
        
        energy_pre2 = energy_pre

        e_loss = loss_func['L1'](energy_pre2, (data.energy.unsqueeze(1)))

        
        force_pre = -grad(outputs=energy_pre2, inputs=pos, grad_outputs=torch.ones_like(energy_pre2),create_graph=True,retain_graph=True)[0]
        f_loss = loss_func['L1'](force_pre, data.force)

        



        

        return e_loss,f_loss