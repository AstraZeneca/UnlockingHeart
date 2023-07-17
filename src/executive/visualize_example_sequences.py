# This file holds executibe code for visualization of selected sequence

# Load packages
import argparse
import os
import sys
import numpy as np
import tqdm
import yaml
import json

# Import from torch
import torch
from torchvision import transforms as pth_transforms

# Import echonet dynamic loader
from src.data import DatasetWithIndices
from src.data.dataloaders import *

# Import own packagea
from src.models import DINO_backed
from src.models.DINO_backed import model_predict
from src.models import parcel_segmentation
from src.tools import preprocessing
from src.tools import segment_processing
from src.tools import misc
from src.tools import vis


def main(cfg, backend_path, head_path, segmentation_path, img_indices, output_path, disk_radius=1):
    # set parameters
    num_classes = cfg['head']['num_classes']
    patch_size = cfg['backend']['patch_size']
    dataset_name = cfg['dataset']['name']
    resize_size = cfg['dataloader']['resize_size']

    # Set seed
    misc.set_seed(cfg['seed'])

    # Setup model
    # Define device for copmutations
    device = torch.device("cpu")
    if "device" in cfg:
        device = torch.device(cfg["device"])
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        print("Using CPU")

    # Load DINO model
    dino_model = DINO_backed.load_dino(
        backend_path, device=device, patch_size=patch_size, img_size=resize_size)

    # Load DINO_backed model
    model = DINO_backed.get_class(head_path)(
        dino_model=dino_model, num_classes=num_classes, device=device, reshaper=(-1, 3, resize_size, resize_size))
    model.load_front_end(head_path, device=device)
    print(f"Embedded dimensions: {model.embed_dim}")

    # %% Setup dataset
    # Get dataset and data loader
    dataset_class = eval(dataset_name)

    normalization = dataset_class.get_normalization()
    if dataset_name == 'CAMUS':
        mean = 0.5 * normalization['2CH']['mean'] + 0.5 * normalization['4CH']['mean']
        std = np.sqrt(0.5 * normalization['2CH']['std']**2 + 0.5 * normalization['4CH']['std']**2)
    elif dataset_name == 'EchoNet_Dynamic':
        mean = normalization['mean']
        std = normalization['std']
    else:
        sys.exit(f'Not suported dataset type {dataset_name}')

    preprocessing_list = [preprocessing.Normalize(mean=mean, std=std)]
    mask_transform = [pth_transforms.Resize(resize_size, pth_transforms.InterpolationMode.NEAREST)]
    # Acquire pre-processing
    if 'transforms' in cfg['dataset']:
        for t in cfg['dataset']['transforms']:
            sname = t.pop('name')
            if sname == 'CLAHE' and dataset_name == 'CAMUS':
                t['scale'] /= std
                t['translate'] = - mean / std
            transform_process = preprocessing.__dict__[sname](**t)
            preprocessing_list.append(transform_process)
            if sname == 'VerticalFlip' or sname == 'HorizontalFlip' or sname == 'Rotation':
                mask_transform.append(transform_process)
        cfg['dataset'].pop('transforms')
    cfg['dataset']['preprocessing'] = preprocessing_list
    mask_transform = pth_transforms.Compose(mask_transform)

    DataWithIndices = DatasetWithIndices(dataset_class)
    dataset = DataWithIndices(resize_size=[resize_size, resize_size],
                              device=device,
                              **cfg['dataset'])

    unnormalize = preprocessing.Unnormalize(mean=mean, std=std)

    # Create segmenter
    if cfg['segmentation']['name'] not in parcel_segmentation.__dict__.keys():
        raise Exception("Segmentation method given in config file does not exist!")

    # Load segmentation
    with open(segmentation_path, 'r') as file_handle:
        segmentation_data = json.load(file_handle)

    # Acquire post-processing
    transforms_list = []
    for t in cfg['segmentation']['transforms']:
        sname = t.pop('name')
        transforms_list.append(segment_processing.__dict__[sname](**t))

    # Handle segmenter
    segmenter = parcel_segmentation.__dict__[
        cfg['segmentation']['name']](**segmentation_data,
                                     post_processing=transforms_list,
                                     **cfg['segmentation']['parameters'])

    # Go through dataset
    # Set model as under evaluation
    model.eval()
    # Set that gradients should not be computed
    model.requires_grad_(False)

    # Loop through selected sequences
    for idx in tqdm.tqdm(img_indices):
        imgs, targets, _ = dataset[idx]
        # Get dimensions of images
        imgs = imgs[None, :]
        B, Fr, C, H, W = imgs.size()

        _, hard_parcelization_larger = model_predict(model, imgs, num_classes,
                                                     resize_size, patch_size)
        # Move to cpu
        mask_transform = pth_transforms.Resize(resize_size, pth_transforms.InterpolationMode.NEAREST)
        ES_masks = mask_transform(torch.unsqueeze(torch.Tensor(targets[0]), dim=0))
        ED_masks = mask_transform(torch.unsqueeze(torch.Tensor(targets[1]), dim=0))

        if dataset_name == 'CAMUS':
            exp_name = f"{dataset.patient_paths[idx % len(dataset.patient_paths)].name}_{dataset.views[idx // len(dataset.patient_paths)]}"
        elif dataset_name == 'EchoNet_Dynamic':
            exp_name = f"{dataset.dataset.fnames[idx]}"
        else:
            exp_name = None
            print(f'Not suported dataset type {dataset_name}')

        # Extract image sequence
        img = imgs[0]
        img = unnormalize(img.cpu()).numpy()
        img = img / np.max(img)

        # The systolic and diastolic masks
        ES_mask = ES_masks[0].numpy()
        ED_mask = ED_masks[0].numpy()

        # Get specific frames of parcelization
        parcels_larger = hard_parcelization_larger[0]

        # Acquire segment from parcels
        segment, _ = segmenter(parcels_larger)

        # Define end-systole and end-diastole frames
        areas = np.sum(segment, axis=(1, 2))
        ES_frame = np.argmin(areas)
        ED_frame = np.argmax(areas)

        seq_save_path = os.path.join(output_path, f"{exp_name}_segments.png")
        vis.vis_end_segments(img, parcels_larger, segment,
                               ES_mask, ED_mask, ES_frame, ED_frame,
                               seq_save_path, disk_radius=disk_radius)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="Config file with evaluation parameters", required=True)
    parser.add_argument("-t", "--target_type",
                        default=["ESTrace", "EDTrace"], nargs='+', help="Name of ES and ED heart parts to segment", required=False)
    parser.add_argument("-i", "--img_indices", nargs='+', type=int, help="Indices of sequences to visualize", required=True)
    parser.add_argument("--backend_path", help="Path to backend model", required=True)
    parser.add_argument("--head_path", help="Path to head model", required=True)
    parser.add_argument("--segmentation_path", help="Path to segmentation model", required=True)
    parser.add_argument("--output_path", help="Path to output directory", required=True)
    parser.add_argument("--disk_radius", default=1, type=int, required=False)

    args = parser.parse_args()
    # Get parameters from file
    # Load config
    with open(args.config_file, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)
    cfg['dataset']['target_type'] = args.target_type

    main(cfg=cfg, backend_path=args.backend_path,
         head_path=args.head_path, segmentation_path=args.segmentation_path,
         img_indices=args.img_indices, output_path=args.output_path, disk_radius=args.disk_radius)
