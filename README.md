# TAIP


This is the official implementation for the paper: "Online Test-time Adaptation for Interatomic Potentials". Please read and cite these manuscripts if using this example:
[ArXiv:2405.08308 (2024)](https://arxiv.org/abs/2405.08308)


- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [How to run this code](#how-to-run-this-code)

# Overview
Machine learning interatomic potentials (MLIPs) enable more efficient molecular dynamics (MD) simulations with ab initio accuracy, which have been used in various domains of physical science. However, distribution shift between training and test data causes deterioration of the test performance of MLIPs, and even leads to collapse of MD simulations. In this work, we propose an online Test-time Adaptation Interatomic Potential (TAIP) framework to improve the generalization on test data. Specifically, we design a dual-level self-supervised learning approach that leverages global structure and atomic local environment information to align the model with the test data. Extensive experiments demonstrate TAIP's capability to bridge the domain gap between training and test dataset without additional data. TAIP enhances the test performance on various benchmarks, from small molecule datasets to complex periodic molecular systems with various types of elements. Remarkably, it also enables stable MD simulations where the corresponding baseline models collapse.


# System Requirements
## Hardware requirements

A GPU is required for running this code base, and one RTX 4090 card have been tested.

## Software requirements
### OS Requirements
This code base is supported for *Linux* and has been tested on the following systems:
+ Linux: Ubuntu 20.04

### Python Version

Python 3.9 has been tested.

# Installation Guide:

### Install dependencies

```shell
conda create -y -n TAIP python=3.9
conda activate TAIP
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install pyg==2.1.0 -c pyg
pip install easydict
pip install dive-into-graphs
pip install ase ase[test] ogb
```


# Demo:

### You can view this demo notebook to see our complete training and testing process.

```
water.ipynb
```


# How to run this code:


### Download the dataset to raw_data and save processed files to processed:

```
python xyz2pt.py ./raw_data/newliquid_shifted_ev.xyz ./processed/newliquid_shifted_ev.pt
```

### Train model on liquid water

The datasets are sampled using our recently developed active learning method based on evidential deep learning to improve the diversity of atomic structures. We train the models with liquid water, using a training set of 1000 frames and a validation set of 100 frames, and report the test accuracy on randomly sampled 1,000 liquid water and ice structures from the remaining dataset, respectively. 

```
python train_new.py --config config.yaml
```

### Test model on liquid water

```
python test.py --config config.yaml
```

### Molecular dynamic simulation on liquid water


```
cd ./TAIP_MD_Codes/PaiNN
python MD_run.py --checkpoint PaiNN-TAIP.pt --config config.yaml --init_atoms test.xyz --save_dir ./ --temp 300 --steps 10000
```
