# Unlocking the Heart Using Adaptive Locked Agnostic Networks

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<p align="center">
  <img width="800" src="https://github.com/AstraZeneca/UnlockingHeart/blob/master/pipeline.jpg">
</p>

This repository accompanies our paper [Unlocking the Heart Using Adaptive Locked Agnostic Networks](https://arxiv.org/abs/2309.11899) and enables replication of the key results.

## Overview

We introduced the Adaptive Locked Agnostic Network (ALAN)-concept for analyzing echocardiograms. That is, focusing on learning a backbone-model in order to lock it in for the future. This locked model can then be the foundation on which multiple and vastly different medical tasks can be solved using relatively simple models. Furthermore, we introduced the notion of parcelization, and developed a post-processing method that transforms self-supervised output into anatomically relevant segmentation.

We investigated [DINO ViT](https://arxiv.org/abs/2104.14294) and RAPTOR (modified [STEGO](https://arxiv.org/abs/2203.08414)) trained both on general (ImageNet) and echocardiography data domain.

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

Make sure you have Python installed. Please install PyTorch. This codebase has been developed with python version 3.10.8., PyTorch version 1.11.0, CUDA 11.6 and torchvision 0.12.0. The arguments to reproduce the models presented in our paper can be found in the config directory in `yaml` files and as default arguments in the scripts.

To set up enviroment and install dependencies from base directory, run:

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

### TMED

1. Download the dataset from [TMEDv2](https://tmed.cs.tufts.edu/tmed_v2.html) into `data/TMED/` directory
2. Preprocess data based on `image_labels.csv` file to create following directory structure:
```txt
    ./data/TMED/
    ├── image_labels.csv      <- List of used images with their labels and tags train/val.
    ├── train/                <- Training images.
    │ ├── A2C/                <- directory for A2C data
    │ ├── A4C/                <- directory for A4C data
    │ ├── PLAX/               <- directory for PLAX data
    │ └── PSAX/               <- directory for PSAX data
    └── val/                  <- Validation images.
     ├── A2C/                 <- directory for A2C data
     ├── A4C/                 <- directory for A4C data
     ├── PLAX/                <- directory for PLAX data
     └── PSAX/                <- directory for PSAX data
```

## Usage

### DINO ViT training

Run DINO with ViT-S/4 on the EchoNet-Dynamic dataset using all separate frames of the training sequences by runing script:
```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=9904 src/executive/main_dino_echo.py --arch vit_base --data_path data/EchoNet-Dynamic/ --output_dir results/trained_echo_dino --epochs 50 --teacher_temp 0.04 --warmup_teacher_temp_epochs 30 --norm_last_layer false --length None --patch_size 4 --use_fp16 false --batch_size_per_gpu 2 --img_size 112

```
or use pretrained on ImageNet checkpoint from [facebookresearch/dino](https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth).

### STEGO training

In next step, the frozen backbone were used to acquire latent features for the STEGO head input, fed by randomly selected sub-sequences of 16 frames of the EchoNet-Dynamic training subset. To train STEGO head run:


```bash
python src/executive/train_parcelization_on_regular_dino.py --num_classes 64 --patch_size 8 --resize_size 224 \
                                                            --dino_model_path models/dino/dino_deitsmall8_pretrain.pth \
                                                            --device cuda --num_epochs 50 --batch_size 5 \
                                                            --output_model_path models/STEGO_like/STEGO_like_from_ImageNet_pretrained_dino_num_classes_64_patch_size_8_resize_size_224.pth
```

### Parcel-to-segment

Final segmentation is achieved using parcel-to-segment method. We first identified parcels that overlap with manually labelled end-diastolic and end-systolic frames. Second, we calculated metrics such as DICE scores. Finally, we visualized selected instances.

```bash
# Parcels identification
python src/executive/identify_parcels.py --config_file configs/identify_parcels_external_pretrained_EchoNet.yml \
                                         --backend_path ./models/dino/dino_deitsmall8_pretrain.pth \
                                         --head_path ./models/STEGO_like/STEGO_like_from_external_pretrained_dino_num_classes_64_patch_size_8_resize_size_224_automatic.pth \
                                         --output_path ./models/parcel_classes_external_ECHODYNAMIC.json

# Evaluation
python src/executive/evaluation.py --config_file configs/eval_ViT_STEGO_EchoNet_external_pretrained.yml \
                                   --backend_path ./models/dino/dino_deitsmall8_pretrain.pth \
                                   --head_path ./models/STEGO_like/STEGO_like_from_external_pretrained_dino_num_classes_64_patch_size_8_resize_size_224_automatic.pth \
                                   --segmentation_path ./models/parcel_classes_external_ECHODYNAMIC.json \
                                   --output_path ./results/eval_ViT_STEGO_external_ECHODYNAMIC

# Visualization
python src/executive/visualize_example_sequences.py --config_file configs/eval_ViT_STEGO_EchoNet_external_pretrained.yml \
                                                    --backend_path ./models/dino/dino_deitsmall8_pretrain.pth \
                                                    --head_path ./models/STEGO_like/STEGO_like_from_external_pretrained_dino_num_classes_64_patch_size_8_resize_size_224_automatic.pth \
                                                    --segmentation_path ./models/parcel_classes_external_ECHODYNAMIC.json \
                                                    --output_path ./results/eval_ViT_STEGO_external_ECHODYNAMIC -i 5
```

### KNN Classification

To evaluate a simple k-NN classifier on the a standard, ImageNet pre-trained DINO outputs, and TMEDv2 dataset, run:

```bash
# ImageNet based DINO trained
python src/executive/eval_knn_echo.py --num_classes 4 --nb_knn 2 10 \
                                      --pretrained_weights models/dino/dino_deitsmall8_pretrain.pth \
                                      --use_cuda True --patch_size 8 --checkpoint_key teacher --data_path data/TMED/data/ \
                                      --img_size 224 --dump_features results/TMED_features_imagenet
```

If you would like to follow our publication and train DINO model on echo data, then you need to addapt `image_size`, `patch_size` parameters, as for example:

```bash
# EchoNet-Dynamic based DINO trained
python src/executive/eval_knn_echo.py --num_classes 4 --nb_knn 2 10 \
                                      --pretrained_weights models/dino/trained_echo_dino.pth \
                                      --use_cuda True --patch_size 4 --checkpoint_key teacher --data_path data/TMED/data/ \
                                      --img_size 112 --dump_features results/TMED_features_echo
```

## Citation

Please consider citing the paper for this repo if you find it useful:

```
@misc{majchrowska2023unlocking,
      title={Unlocking the Heart Using Adaptive Locked Agnostic Networks}, 
      author={Sylwia Majchrowska and Anders Hildeman and Philip Teare and Tom Diethe},
      year={2023},
      eprint={2309.11899},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Folder structure

```txt
    ./
    ├── cofigs/                 <- Yaml files with used parameters.
    ├── data/                   <- Data from third party sources.
    ├── lib/                    <- Library of shared and/or external code.
    ├── models/                 <- Trained and serialized models, model predictions, or model summaries.
    ├── results/                <- Directory to store results.
    ├── src/                    <- Source code for use in this project.
    │   ├── __init__.py         <- Makes src a Python module
    │   ├── data                <- directory for data related source code
    │   ├── executive           <- directory for executive scripts
    │   ├── models              <- directory for model related source code
    │   └── tools               <- directory for additional tools related source code
    ├── environment.yml         <- Conda environment stored as '.yml'.
    ├── pipeline.jpg            <- Graphical abstract of used ALAN methodology.
    ├── README.md               <- The top-level README for developers using this project.
    └── setup.py                <- The setup script.
```
