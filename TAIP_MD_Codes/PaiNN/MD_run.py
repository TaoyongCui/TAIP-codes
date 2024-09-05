from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.andersen import Andersen
from ase.md.npt import NPT
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, write, Trajectory
from ase.io.trajectory import TrajectoryReader
from ase import Atoms

import numpy as np
import torch
import sys
import glob
from testtime import MLCalculator_schnet
from ase.constraints import FixAtoms
from TAIP.PaiNN2 import PainnModel as PaiNN
from torch import nn
from testtime import extractor_from_layer, ExtractorHead
import argparse
import yaml
from easydict import EasyDict
from TAIP_test import EquivariantDenoisePred

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, help='Model checkpoint for MLIP-MD')
parser.add_argument('--config', type=str, default='config.yaml', help='Model config')
parser.add_argument('--init_atoms', type=str, help='Initial atomic configuration')
parser.add_argument('--save_dir', type=str, help='Saving dirs for Log and MD trajectory')
parser.add_argument('--temp', type=int, help='Simulation temperature')
parser.add_argument('--steps', type=int, help='Simulation steps')

args = parser.parse_args()

checkpoint = agrs.checkpoint
savedir = agrs.save_dir
initatoms = args.init_atoms
config = args.config
temp = args.temp
steps = args.steps

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 读取模型 #

with open(config, 'r') as f:
    config = yaml.safe_load(f)
config = EasyDict(config)
config.train.seed = 2021

rep = PaiNN(num_interactions=3, hidden_state_size=128, cutoff=6.0).to(device)

P = 1000
ext = extractor_from_layer(rep).to(device)
head = nn.Sequential(
            nn.Linear(128, 128), 
            nn.SiLU(),
            nn.Linear(128, 128),
        ).to(device)

ckpt = torch.load(str(checkpoint),map_location=device)

rep.load_state_dict(ckpt['model'],strict=False)
ssh = ExtractorHead(head, ext=ext).to(device)
net = EquivariantDenoisePred(config, rep, ssh).to(device)
net.model.load_state_dict(ckpt['model'],strict=False)
net.graph_dec.load_state_dict(ckpt['graph_dec'])
net.node_dec.load_state_dict(ckpt['node_dec'])
head.load_state_dict(ckpt['head'])
net.noise_pred.load_state_dict(ckpt['noise_pred'])
net.decoder.load_state_dict(ckpt['decoder'])
net.decoder_force.load_state_dict(ckpt['decoder_force'])


# 读取MD初始构型 #

atoms_all = read(initatoms)
mlcalc = MLCalculator_schnet(net, ssh, head, rep, device)
atoms.calc = mlcalc
intervals = 1

steps = 0

# MD轨迹LOG #
 
def printenergy(a=atoms):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    ekin = a.get_kinetic_energy()
    epot = a.get_potential_energy()
    eforcemax = a.calc.get_property('force_max')
    force_mean = np.mean(abs(a.calc.get_property('forces')))
    force_std = np.std(a.calc.get_property('forces'))
    temp = ekin / (1.5 * units.kB) / a.get_global_number_of_atoms()
    global steps
    steps += intervals
    with open(savedir+'/log.testtime_'+str(checkpoint)+str(init_atoms)+str(temp)+, 'a') as f:
        f.write(
        f"Steps={steps:8.0f} Epot={epot:8.2f} Ekin={ekin:8.2f} force_max={eforcemax:8.2f} force_mean={force_mean:8.2f} force_std={force_std:8.2f} temperature={temp:8.2f}\n")


# 设置初始温度#

import numpy

numpy.random.seed(123)
temp = int(temp)
MaxwellBoltzmannDistribution(atoms, temperature_K=temp,rng=numpy.random)
 
# 设置热偶 MD轨迹保存#

dyn = NVTBerendsen(atoms, 0.5 * units.fs, temperature_K=temp, taut=100 * units.fs)

dyn.attach(printenergy, interval=intervals)
traj = Trajectory(savedir+'/MD_testtime'+str(checkpoint)+str(init_atoms)+str(temp)+'.traj', 'w', atoms)
dyn.attach(traj.write, interval=20)

dyn.run(steps)

