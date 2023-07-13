# Unlocking the Heart Using Adaptive Locked Agnostic Networks

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<p align="center">
  <img width="800" src="https://github.com/AstraZeneca/UnlockingHeart/blob/master/pipeline.jpg">
</p>

This repository accompanies our paper Unlocking the Heart Using Adaptive Locked Agnostic Networks and enables replication of the key results.

## Overview

We introduced the Adaptive Locked Agnostic Network (ALAN)-concept for analyzing echocardiograms. That is, focusing on learning a backbone-model in order to lock it in for the future. This locked model can then be the foundation on which multiple and vastly different medical tasks can be solved using relatively simple models. Furthermore, we introduced the notion of parcelization, and developed a post-processing method that transforms self-supervised output into anatomically relevant segmentation.

We investigated [DINO ViT](https://arxiv.org/abs/2104.14294) and [STEGO](https://arxiv.org/abs/2203.08414) trained both on general (ImageNet) and echocardiography data domain.

## Installation

Clone this repository and enter the directory:
```bash
git clone https://github.com/AstraZeneca/UnlockingHeart.git
cd UnlockingHeart
```

### Clone external repositories

Create directory to storge external code.

```bash
mkdir -p lib
```

#### DINO ViT
1. Clone DINO repository
2. Download pretrained DINO ViT checkpoint

```bash
# Clone DINO repository
cd lib
git clone https://github.com/facebookresearch/dino
cd dino
git checkout cb711401860da580817918b9167ed73e3eef3dcf
cd ../..

# Download DINO ViT checkpoint
mkdir -p models/dino
wget https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth -P models/dino
```

#### EchoNet-Dynamic

Clone EchoNet-Dynamic repository

```bash
cd lib
git clone https://github.com/echonet/dynamic.git
cd dynamic
git checkout 108518f305dbd729c4dca5da636601ebd388f77d
```

### Set up enviroment

Make sure you have Python installed. The code is implemented for Python 3.x.x.

Create enviroment and install dependencies from base directory:

```bash
# Create and activate conda environment
conda env create -f environment.yml --prefix echohf
conda activate echohf

# Install src as python package
pip install -e .
```

## Datasets

Create base directory for needed data:

```bash
mkdir -p data/EchoNet-Dynamic/
mkdir -p data/CAMUS/
```

### EchoNet-Dynamic
1. Download the dataset from [EchoNet-Dynamic website](https://echonet.github.io/dynamic/index.html#dataset) into `data/EchoNet-Dynamic/` directory
2. Run the below script to preprocess data:

```bash
python src/data/dataloaders/echonet_dynamic.py --data_directory ./data/EchoNet-Dynamic/ --data_split train
```

### CAMUS
1. Download the dataset from [CAMUS challenge website](https://www.creatis.insa-lyon.fr/Challenge/camus/databases.html) into `data/CAMUS/` directory
2. Run the below script to preprocess data:

```bash
python src/data/dataloaders/CAMUS.py --data_directory ./data/CAMUS/ --data_split train
```

## Usage

TBA

## Citation

Please consider citing the paper for this repo if you find it useful:

```
<citation>
```
