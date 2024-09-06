import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max, scatter_min, scatter_mean
# from torch_sparse import coalesce
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph
from scipy.stats import truncnorm
from torch.autograd import grad
import math
import time
import random
from functools import partial
# from dig.threedgraph.evaluation import threedEvaluator
from TAIP.decoder import SchNetDecoder
from typing import Optional, Union, List

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

    def __init__(self, config, rep_model,ssh_model):
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




    def get_energy_and_rep(self, x, pos, data, node2graph, return_pos = False, models = None):

        xl, posl,edge_index,distance = models(x, pos,data)

        xl = self.node_dec(xl)

        xg = scatter_add(xl, node2graph, dim = -2)

        e = self.graph_dec(xg)
        
        return xg.squeeze(-1), xg, posl,xl,edge_index,distance
        

    @torch.no_grad()
    def get_distance(self, data: Data):
        pos = data.pos
        row, col = data.edge_index
        d = (pos[row] - pos[col]).norm(dim=-1).unsqueeze(-1) # (num_edge, 1)
        data.edge_length = d
        return data

    @torch.no_grad()
    def truncated_normal(self, size, threshold=1):
        values = truncnorm.rvs(-threshold, threshold, size=size)
        return torch.from_numpy(values)
    @torch.no_grad()
    def mask(self, x, num_atom_type, mask_rate):

        

        num_atoms = x.size()[0]
        sample_size = int(num_atoms * mask_rate + 1)
        masked_atom_indices = random.sample(range(num_atoms), sample_size)
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(x[atom_idx].view(1, -1))
        mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        masked_atom_indices = torch.tensor(masked_atom_indices)

        atom_type = F.one_hot(mask_node_label[:, 0], num_classes=num_atom_type).float()
        # data.node_attr_label = torch.cat((atom_type,atom_chirality), dim=1)
        node_attr_label = atom_type

        # modify the original node feature of the masked node
        for atom_idx in masked_atom_indices:
            x[atom_idx] = torch.tensor([num_atom_type])

        x_perturb = x

        return x_perturb,node_attr_label,masked_atom_indices
    @torch.no_grad()
    def get_force_target(self, perturbed_pos, pos, node2graph):
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
        if not self.edge_types:
            return None
        return F.one_hot(edge_types.long(), self.edge_types)

    @torch.no_grad()
    def fit_pos(self, perturbed_pos, pos, node2graph):
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


    def mask_force(self, force, mask_rate,used_sigmas):



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
        Input:
            data: torch geometric batched data object
        Output:
            loss
        """
        self.device = self.sigmas.device
        
        node2graph = data.batch
        
        noise_level = torch.randint(0, self.sigmas.size(0), (data.num_graphs,), device=self.device) # (num_graph)
        used_sigmas = self.sigmas[noise_level] # (num_graph)

        used_sigmas = used_sigmas[node2graph].unsqueeze(-1) # (num_nodes, 1)

        pos = data.pos.clone()

        perturbed_pos = self.perturb(pos, node2graph, used_sigmas, self.config.train.steps)

        target = self.get_force_target(perturbed_pos, pos, node2graph) / used_sigmas


        input_pos = perturbed_pos.clone()
        input_pos.requires_grad_(True)

        data.pos = input_pos
        


        _, graph_rep_noise, pred_pos, _,_,_ = self.get_energy_and_rep(data.z.long(), input_pos, data, node2graph, return_pos = True, models=self.ssh)
        



        tmp_pos = pos.clone()
        tmp_pos.requires_grad_(True)
        data.pos = tmp_pos


        mask_z,node_attr_label,masked_node_indices = self.mask(data.z.long().clone(),num_atom_type=43,mask_rate=0.15)
        energy, _, _, mask_rep,edge_index,distance = self.get_energy_and_rep(mask_z, tmp_pos, data, node2graph, return_pos = True, models=self.ssh)
        

        pred_node = self.decoder(mask_rep)
        m_loss = criterion(node_attr_label, pred_node[masked_node_indices])



        grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
        dy = grad(
                [energy],
                [tmp_pos],
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
            )[0]
    
        pred_noise = (-dy).view(-1, 3)



        mask_force,mask_node_labels_list,masked_atom_indices = self.mask_force(pred_noise, 0.05,used_sigmas)
        dmask_force = self.decoder_force(mask_force,edge_index,distance)
        

        fm_loss = loss_func['L1'](torch.cat(mask_node_labels_list, dim=0).view(-1, 3), dmask_force[masked_atom_indices].clone().detach())
        energy_pre, graph_rep_ori, _,_,_,_ = self.get_energy_and_rep(data.z.long(), tmp_pos, data, node2graph, return_pos = False, models=self.ssh)
    


        graph_rep = torch.cat([graph_rep_ori, graph_rep_noise], dim=1)

        pred_scale = self.noise_pred(graph_rep)

        loss_pred_noise = loss_func['CrossEntropy'](pred_scale, noise_level) 


        return loss_pred_noise.mean() , fm_loss, m_loss