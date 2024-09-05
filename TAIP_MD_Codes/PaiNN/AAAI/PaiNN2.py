import torch
from torch import nn
import pdb
from torch_scatter import scatter
from .SchNet2 import *
def sinc_expansion(edge_dist: torch.Tensor, edge_size: int, cutoff: float):
    """
    calculate sinc radial basis function:
    
    sin(n *pi*d/d_cut)/d
    """
    n = torch.arange(edge_size, device=edge_dist.device) + 1
    return torch.sin(edge_dist.unsqueeze(-1) * n * torch.pi / cutoff) / edge_dist.unsqueeze(-1)

def cosine_cutoff(edge_dist: torch.Tensor, cutoff: float):
    """
    Calculate cutoff value based on distance.
    This uses the cosine Behler-Parinello cutoff function:
    f(d) = 0.5*(cos(pi*d/d_cut)+1) for d < d_cut and 0 otherwise
    """

    return torch.where(
        edge_dist < cutoff,
        0.5 * (torch.cos(torch.pi * edge_dist / cutoff) + 1),
        torch.tensor(0.0, device=edge_dist.device, dtype=edge_dist.dtype),
    )

class PainnMessage(nn.Module):
    """Message function"""
    def __init__(self, node_size: int, edge_size: int, cutoff: float):
        super().__init__()
        
        self.edge_size = edge_size
        self.node_size = node_size
        self.cutoff = cutoff
        
        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )
        
        self.filter_layer = nn.Linear(edge_size, node_size * 3)
        
    def forward(self, node_scalar, node_vector, edge, edge_diff, edge_dist):
        # remember to use v_j, s_j but not v_i, s_i        
        filter_weight = self.filter_layer(sinc_expansion(edge_dist, self.edge_size, self.cutoff))
        filter_weight = filter_weight * cosine_cutoff(edge_dist, self.cutoff).unsqueeze(-1)
        scalar_out = self.scalar_message_mlp(node_scalar)        
        filter_out = filter_weight * scalar_out[edge[:, 1]]
        
        gate_state_vector, gate_edge_vector, message_scalar = torch.split(
            filter_out, 
            self.node_size,
            dim = 1,
        )
        
        # num_pairs * 3 * node_size, num_pairs * node_size
        message_vector =  node_vector[edge[:, 1]] * gate_state_vector.unsqueeze(1) 
        edge_vector = gate_edge_vector.unsqueeze(1) * (edge_diff / edge_dist.unsqueeze(-1)).unsqueeze(-1)
        message_vector = message_vector + edge_vector
        
        # sum message
        residual_scalar = torch.zeros_like(node_scalar)
        residual_vector = torch.zeros_like(node_vector)
        residual_scalar.index_add_(0, edge[:, 0], message_scalar)
        residual_vector.index_add_(0, edge[:, 0], message_vector)
        
        # new node state
        new_node_scalar = node_scalar + residual_scalar
        new_node_vector = node_vector + residual_vector
        
        return new_node_scalar, new_node_vector

class PainnUpdate(nn.Module):
    """Update function"""
    def __init__(self, node_size: int):
        super().__init__()
        
        self.update_U = nn.Linear(node_size, node_size)
        self.update_V = nn.Linear(node_size, node_size)
        
        self.update_mlp = nn.Sequential(
            nn.Linear(node_size * 2, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )
        
    def forward(self, node_scalar, node_vector):
        Uv = self.update_U(node_vector)
        Vv = self.update_V(node_vector)
        
        Vv_norm = torch.linalg.norm(Vv, dim=1)
        mlp_input = torch.cat((Vv_norm, node_scalar), dim=1)
        mlp_output = self.update_mlp(mlp_input)
        
        a_vv, a_sv, a_ss = torch.split(
            mlp_output,                                        
            node_vector.shape[-1],                                       
            dim = 1,
        )
        
        delta_v = a_vv.unsqueeze(1) * Uv
        inner_prod = torch.sum(Uv * Vv, dim=1)
        delta_s = a_sv * inner_prod + a_ss
        
        return node_scalar + delta_s, node_vector + delta_v

class PainnModel(nn.Module):
    """PainnModel without edge updating"""
    def __init__(
        self, 
        num_interactions, 
        hidden_state_size, 
        cutoff,
        normalization=True,
        target_mean=[0.0],
        target_stddev=[1.0],
        atomwise_normalization=True, 
        **kwargs,
    ):
        super().__init__()
        
        num_embedding = 119   # number of all elements
        self.cutoff = cutoff
        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.edge_embedding_size = 20
        
        # Setup atom embeddings
        self.atom_embedding = nn.Embedding(num_embedding, hidden_state_size)

        # Setup message-passing layers
        self.message_layers = nn.ModuleList(
            [
                PainnMessage(self.hidden_state_size, self.edge_embedding_size, self.cutoff)
                for _ in range(self.num_interactions)
            ]
        )
        self.update_layers = nn.ModuleList(
            [
                PainnUpdate(self.hidden_state_size)
                for _ in range(self.num_interactions)
            ]            
        )
        
        # Setup readout function
        self.readout_mlp = nn.Sequential(
            nn.Linear(self.hidden_state_size, self.hidden_state_size),
            nn.SiLU(),
            nn.Linear(self.hidden_state_size, self.hidden_state_size),
        )

        # Normalisation constants
        self.register_buffer("normalization", torch.tensor(normalization))
        self.register_buffer("atomwise_normalization", torch.tensor(atomwise_normalization))
        self.register_buffer("normalize_stddev", torch.tensor(target_stddev[0]))
        self.register_buffer("normalize_mean", torch.tensor(target_mean[0]))

    def forward(self, x, pos,input_dict, compute_forces=True):

        if compute_forces:
            pos.requires_grad_()       
        edge_index, cell_offsets, _, neighbors = radius_graph_pbc(
            data = input_dict, radius = self.cutoff, max_num_neighbors_threshold = 500
        )
        input_dict.edge_index = edge_index
        input_dict.cell_offsets = cell_offsets
        input_dict.neighbors = neighbors
        z = x.long()
        assert z.dim() == 1 and z.dtype == torch.long

        out = get_pbc_distances(
            pos,
            input_dict.edge_index,
            input_dict.cell,
            input_dict.cell_offsets,
            input_dict.natoms,
        )
        edge = out["edge_index"].t()
        edge_diff = out['distance_vec']
        num_atoms = input_dict.num_atoms = torch.tensor([len(z)]) .to(edge.device)

        edge_dist = out["distances"]
        
        node_scalar = self.atom_embedding(z)
        node_vector = torch.zeros((pos.shape[0], 3, self.hidden_state_size),
                                  device=edge_diff.device,
                                  dtype=edge_diff.dtype,
                                 )
        
        for message_layer, update_layer in zip(self.message_layers, self.update_layers):
            node_scalar, node_vector = message_layer(node_scalar, node_vector, edge, edge_diff, edge_dist)
            node_scalar, node_vector = update_layer(node_scalar, node_vector)
        
        node_scalar = self.readout_mlp(node_scalar)

        return node_scalar,pos,edge_index,edge_dist