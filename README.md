# TAIP


This is the official implementation for the paper: "Online Test-time Adaptation for Interatomic Potentials". Please read and cite these manuscripts if using this example:
[ArXiv:2405.08308 (2024)](https://arxiv.org/abs/2405.08308)


- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [How to run this code](#how-to-run-this-code)

# Overview
Machine learning interatomic potentials (MLIPs) enable more efficient molecular dynamics (MD) simulations with ab initio accuracy, which have been used in various domains of physical science. However, distribution shift between training and test data causes deterioration of the test performance of MLIPs, and even leads to collapse of MD simulations. In this work, we propose an online Test-time Adaptation Interatomic Potential (TAIP) framework to improve the generalization on test data. Specifically, we design a dual-level self-supervised learning approach that leverages global structure and atomic local environment information to align the model with the test data. Extensive experiments demonstrate TAIP's capability to bridge the domain gap between training and test dataset without additional data. TAIP enhances the test performance on various benchmarks, from small molecule datasets to complex periodic molecular systems with various types of elements. TAIP also enables stable MD simulations where the corresponding baseline models collapse.


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
  The notebook 'Demo_water_TAIP.ipynb' includes processes of data preprocessing, PaiNN-TAIP training on the liquid water dataset, testing on the liquid water/ice test dataset, and molecular dynamics simulations.
  
  The notebook 'Demo_water_baseline.ipynb' includes processes of PaiNN baseline training on the liquid water dataset, testing on the liquid water/ice test dataset, and molecular dynamics simulations. 
  
  The notebook 'Demo_md17_TAIP.ipynb' includes processes of PaiNN-TAIP training on the non-periodic testcases (aspirin dataset), testing on the aspirin test dataset. 
  
  The notebook 'Demo_md17_baseline.ipynb' includes processes of PaiNN baseline training on the non-periodic testcases (aspirin dataset), testing on the aspirin test dataset. 
  
  The notebook 'MD_simulations.ipynb' includes processes of molecular dynamics simulations, as well as comparisons between the PaiNN-TAIP and the PaiNN baseline model. 
```


# How to run this code:


### Download the dataset to raw_data and save processed files to processed:

```
python xyz2pt.py raw_data/liquid_train.xyz processed/water_train.pt

python xyz2pt.py raw_data/liquid_validation.xyz processed/water_valid.pt

python xyz2pt.py raw_data/liquid_test.xyz processed/water_test.pt

python xyz2pt.py raw_data/ice_test.xyz processed/ice_test.pt
```

### Train model on liquid water

To train the PaiNN model on the liquid water dataset, you can execute the following command. Note that training the model on a single RTX 4090 GPU card will approximately take 4-5 days.

```
python train_water_TAIP.py --config config.yaml
```

### Test model on liquid water

```
python test_water_TAIP.py --config config.yaml --dataset processed/water_test.pt
```

### Molecular dynamic simulation on liquid water


```
python MD_simulation/MD_run.py --checkpoint checkpoint/TAIP_water.pt --config config.yaml --init_atoms test.xyz --save_dir ./MD --temp 300 --steps 1000000
```

### Molecular dynamic simulation on ice


```
python MD_simulation/MD_run.py --checkpoint checkpoint/TAIP_water.pt --config config.yaml --init_atoms test2.xyz --save_dir ./MD --temp 300 --steps 1000000
```

# License
This project is licensed under the Apache License 2.0. For more details about the Apache License 2.0, please refer to the [Apache License](http://www.apache.org/licenses/LICENSE-2.0).