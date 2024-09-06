from math import pi as PI
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Embedding, Sequential, Linear, Dropout
from torch_scatter import scatter
from torch_geometric.nn import radius_graph
from torch_geometric.nn import GCNConv,GATConv,SAGEConv
from torch_geometric.nn import global_mean_pool
from torch.autograd import grad
import copy
import pdb
#torch.set_default_tensor_type(torch.DoubleTensor)
HARTREE_TO_KCAL_MOL = 627.509
EV_TO_KCAL_MOL = 23.06052

OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]

def radius_graph_pbc(data, radius, max_num_neighbors_threshold, topk_per_pair=None):
        """Computes pbc graph edges under pbc.
        topk_per_pair: (num_atom_pairs,), select topk edges per atom pair
        Note: topk should take into account self-self edge for (i, i)
        """
        atom_pos = data.pos
        num_atoms = data.natoms
        lattice = data.cell.long()
        batch_size = len(num_atoms)
        device = atom_pos.device
        # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
        num_atoms_per_image = num_atoms
        num_atoms_per_image_sqr = (num_atoms_per_image ** 2).long()

        # index offset between images
        index_offset = (
            torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
        )

        index_offset_expand = torch.repeat_interleave(
            index_offset, num_atoms_per_image_sqr
        )
        num_atoms_per_image_expand = torch.repeat_interleave(
            num_atoms_per_image, num_atoms_per_image_sqr
        )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
        num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
        index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
        )
        index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
        )
        atom_count_sqr = (
        torch.arange(num_atom_pairs, device=device) - index_sqr_offset
        )

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
        index1 = (torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode='trunc')
        ).long() + index_offset_expand
        index2 = (
        atom_count_sqr % num_atoms_per_image_expand
        ).long() + index_offset_expand
    # Get the positions for each atom
        pos1 = torch.index_select(atom_pos, 0, index1)
        pos2 = torch.index_select(atom_pos, 0, index2)
    
        unit_cell = torch.tensor(OFFSET_LIST, device=device).float()
        num_cells = len(unit_cell)
        unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(index2), 1, 1
        )
        unit_cell = torch.transpose(unit_cell, 0, 1)
        unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
        )

    # Compute the x, y, z positional offsets for each cell in each image
        data_cell = torch.transpose(lattice, 1, 2)
        # pdb.set_trace()
        # pbc_offsets = torch.bmm(data_cell.long(), unit_cell_batch.long())
        pbc_offsets = torch.bmm(data_cell.float(), unit_cell_batch)
        pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
        )

    # Expand the positions and indices for the 9 cells
        pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
        pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
        index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
        index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
        pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
        atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)

        if topk_per_pair is not None:
            assert topk_per_pair.size(0) == num_atom_pairs
            atom_distance_sqr_sort_index = torch.argsort(atom_distance_sqr, dim=1)
            assert atom_distance_sqr_sort_index.size() == (num_atom_pairs, num_cells)
            atom_distance_sqr_sort_index = (
                atom_distance_sqr_sort_index +
                torch.arange(num_atom_pairs, device=device)[:, None] * num_cells).view(-1)
            topk_mask = (torch.arange(num_cells, device=device)[None, :] <
                     topk_per_pair[:, None])
            topk_mask = topk_mask.view(-1)
            topk_indices = atom_distance_sqr_sort_index.masked_select(topk_mask)

            topk_mask = torch.zeros(num_atom_pairs * num_cells, device=device)
            topk_mask.scatter_(0, topk_indices, 1.)
            topk_mask = topk_mask.bool()

        atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
        mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
        mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
        mask = torch.logical_and(mask_within_radius, mask_not_same)
        index1 = torch.masked_select(index1, mask)
        index2 = torch.masked_select(index2, mask)
        unit_cell = torch.masked_select(
            unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)
        if topk_per_pair is not None:
            topk_mask = torch.masked_select(topk_mask, mask)

        num_neighbors = torch.zeros(len(atom_pos), device=device)
        num_neighbors.index_add_(0, index1, torch.ones(len(index1), device=device))
        num_neighbors = num_neighbors.long()
        max_num_neighbors = torch.max(num_neighbors).long()

    # Compute neighbors per image
        _max_neighbors = copy.deepcopy(num_neighbors)
        _max_neighbors[
            _max_neighbors > max_num_neighbors_threshold
        ] = max_num_neighbors_threshold
        _num_neighbors = torch.zeros(len(atom_pos) + 1, device=device).long()
        _natoms = torch.zeros(num_atoms.shape[0] + 1, device=device).long()
        _num_neighbors[1:] = torch.cumsum(_max_neighbors, dim=0)
        _natoms[1:] = torch.cumsum(num_atoms, dim=0)
        num_neighbors_image = (
        _num_neighbors[_natoms[1:]] - _num_neighbors[_natoms[:-1]]
        )

        atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)
    # return torch.stack((index2, index1)), unit_cell, atom_distance_sqr.sqrt(), num_neighbors_image    
    
    # If max_num_neighbors is below the threshold, return early
        if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
        ):
            return torch.stack((index2, index1)), unit_cell, atom_distance_sqr.sqrt(), num_neighbors_image
    # atom_distance_sqr.sqrt() distance

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with values greater than radius*radius so we can easily remove unused distances later.
        distance_sort = torch.zeros(
            len(atom_pos) * max_num_neighbors, device=device
        ).fill_(radius * radius + 1.0)

    # Create an index map to map distances from atom_distance_sqr to distance_sort
        index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
        index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
        )
        index_sort_map = (
        index1 * max_num_neighbors
        + torch.arange(len(index1), device=device)
        - index_neighbor_offset_expand
        )
        distance_sort.index_copy_(0, index_sort_map, atom_distance_sqr)
        distance_sort = distance_sort.view(len(atom_pos), max_num_neighbors)

    # Sort neighboring atoms based on distance
        distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
        distance_sort = distance_sort[:, :max_num_neighbors_threshold]
        index_sort = index_sort[:, :max_num_neighbors_threshold]

    # Offset index_sort so that it indexes into index1
        index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
            -1, max_num_neighbors_threshold
    )
    # Remove "unused pairs" with distances greater than the radius
        mask_within_radius = torch.le(distance_sort, radius * radius)
        index_sort = torch.masked_select(index_sort, mask_within_radius)

    # At this point index_sort contains the index into index1 of the closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
        mask_num_neighbors = torch.zeros(len(index1), device=device).bool()
        mask_num_neighbors.index_fill_(0, index_sort, True)

    # Finally mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
        unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)

        if topk_per_pair is not None:
            topk_mask = torch.masked_select(topk_mask, mask_num_neighbors)

        edge_index = torch.stack((index2, index1))   
        atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask_num_neighbors)
    
        return edge_index, unit_cell, atom_distance_sqr.sqrt(), num_neighbors_image
    
def get_n_edge(senders, n_node):
        """
        return number of edges for each graph in the batched graph. 
        Has the same shape as <n_node>.
        """
        index_offsets = torch.cat([torch.zeros(1).to(n_node.device), 
                             torch.cumsum(n_node, -1)], dim=-1)
        n_edge = torch.LongTensor([torch.logical_and(senders >= index_offsets[i], 
                                               senders < index_offsets[i+1]).sum() 
                             for i in range(len(n_node))]).to(n_node.device)
        return n_edge

        
def get_pbc_distances(
        pos,
        edge_index,
        lattice,
        cell_offsets,
        num_atoms,
        return_offsets=False,
        return_distance_vec=False,
    ):
        j_index, i_index = edge_index
        num_edges = get_n_edge(j_index, num_atoms)
        distance_vectors = pos[j_index] - pos[i_index]

    # correct for pbc
        lattice_edges = torch.repeat_interleave(lattice, num_edges, dim=0).float()
        offsets = torch.einsum('bi,bij->bj', cell_offsets.float(), lattice_edges.float())
        distance_vectors += offsets

    # compute distances
        distances = distance_vectors.norm(dim=-1)

        out = {
            "edge_index": edge_index,
            "distances": distances,
        }


        out["distance_vec"] = distance_vectors

        if return_offsets:
            out["offsets"] = offsets

        return out


class DenseNormalGamma(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DenseNormalGamma, self).__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        self.dense = nn.Linear(self.in_dim, 4 * self.out_dim)

    def evidence(self, x): 
        return F.softplus(x)
        #return F.sigmoid(x)

    def forward(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)

        v = F.softplus(logv)
        alpha = F.softplus(logalpha)+1
        beta = F.softplus(logbeta)

        aleatoric = beta / (alpha - 1)
        epistemic = beta / v * (alpha - 1)

        return mu, v, alpha, beta


class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff):
        super(update_e, self).__init__()
        self.cutoff = cutoff
        self.lin = Linear(hidden_channels, num_filters, bias=False)
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)

    def forward(self, v, dist, dist_emb, edge_index):
        j, _ = edge_index
        C = 0.5 * (torch.cos(dist * PI / self.cutoff) + 1.0)
        W = self.mlp(dist_emb) * C.view(-1, 1)
        v = self.lin(v)
        e = v[j] * W
        return e


class update_v(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters):
        super(update_v, self).__init__()
        self.act = ShiftedSoftplus()
        self.lin1 = Linear(num_filters, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, e, edge_index):
        _, i = edge_index
        out = scatter(e, i, dim=0)
        out = self.lin1(out)
        out = self.act(out)
        out = self.lin2(out)
        
        return v + out


class update_u(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, uncert_mode):
        super(update_u, self).__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels, out_channels)
        self.uncert_mode = uncert_mode
        #self.gcn = GCNConv(hidden_channels,hidden_channels//2)
        #self.agg = global_mean_pool()
        


        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, edge_index):
        

        
        v = self.lin1(v)
        v = self.act(v)  
        
                   
        e_atom = self.lin2(v)          

        
        return e_atom


class emb(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(emb, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class SchNetDecoder(torch.nn.Module):
    def __init__(self, num_layers=2, hidden_channels=128, out_channels=3, cutoff=6.0, num_filters=128, num_gaussians=50
                 ,uncertainty = None, use_pbc=False, device = 'cpu'
                 ):
        super(SchNetDecoder, self).__init__()


        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians
        self.cutoff = 6.0

        self.init_v = nn.Linear(3, hidden_channels)
        self.dist_emb = emb(0.0, cutoff, num_gaussians)

        self.update_vs = torch.nn.ModuleList([update_v(hidden_channels, num_filters) for _ in range(num_layers)])

        self.update_es = torch.nn.ModuleList([
            update_e(hidden_channels, num_filters, num_gaussians, cutoff) for _ in range(num_layers)])
        
        self.update_u = update_u(hidden_channels, out_channels, uncert_mode = uncertainty)
        self.use_pbc = use_pbc
        self.device = device

        self.reset_parameters()

    def reset_parameters(self):
        self.init_v.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()
        self.update_u.reset_parameters()
        
    
    
    

    def forward(self, z, edge_index, dist):
        
        dist_emb = self.dist_emb(dist.to(torch.long))

        
        v = self.init_v(z.float())

        for update_e, update_v in zip(self.update_es, self.update_vs):
            e = update_e(v.to(torch.float), dist.to(torch.float), dist_emb.to(torch.float), edge_index)
            v = update_v(v.to(torch.float), e , edge_index)

        out = self.update_u(v, edge_index)
        
        return out