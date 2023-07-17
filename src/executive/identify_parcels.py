# This file holds code related to parcels (segmenting left ventricle and left artium) identification

# Load packages
import argparse
from collections import Counter
import sys
import numpy as np
from matplotlib import pyplot as plt
import tqdm
import yaml
import json

# Import from torch
import torch
from torchvision import transforms as pth_transforms

# Import echonet dynamic loader
from src.data.dataloaders import *

# Import own packagea
from src.models import DINO_backed
from src.models.parcel_identification import get_intersection_parcels
from src.tools import misc
from src.tools import preprocessing
from src.tools.image_processing import Segmentation2Labels


def main(config_path, backend_path, head_path, output_path):
    # %% Load parameters
    # Load config
    with open(args.config_file, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    dataset_name = cfg['dataset'].pop('name')
    resize_size = cfg['dataloader']['resize_size']
    batch_size = cfg['dataloader']['batch_size']
    num_classes = cfg['head']['num_classes']
    patch_size = cfg['backend']['patch_size']
    valve_parameters = cfg['valve']
    interior_parameters = cfg['interior']
    exterior_parameters = cfg['exterior']

    # Set seed
    if "seed" in cfg:
        misc.set_seed(cfg['seed'])

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
    dino_model = DINO_backed.load_dino(backend_path, device=device,
                                       patch_size=patch_size, img_size=resize_size)

    # Load DINO_backed model
    model = DINO_backed.get_class(head_path)(
        dino_model=dino_model, num_classes=num_classes, device=device,
        reshaper=(-1, 3, resize_size, resize_size))
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

    # Get dataset
    dataset = dataset_class(resize_size=[resize_size, resize_size],
                            device=device,
                            gt=True,
                            **cfg['dataset'])
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    # Set model as under evaluation
    model.eval()
    # Set that gradients should not be computed
    model.requires_grad_(False)

    systolic_list = []
    svalve_list = []
    exterior_systolic_list = []
    present_systolic_list = []

    diastolic_list = []
    dvalve_list = []
    exterior_diastolic_list = []
    present_diastolic_list = []

    # Define end-systole and end-diastole frames
    ES_frame = 0
    ED_frame = 1
    # We require ESTrace and EDTrace order in a config file
    ES_trace = 0  # [ item_num for item_num, item in enumerate( cfg['dataset']["target_type"] ) if item == "ESTrace" ][0]
    ED_trace = 1  # [ item_num for item_num, item in enumerate( cfg['dataset']["target_type"] ) if item == "EDTrace" ][0]

    # Loop through testset
    for imgs, targets in tqdm.tqdm(iter(data_loader)):

        # Get dimensions of images
        B, Fr, C, H, W = imgs.size()

        # Get segmentation
        soft_parcelization = model(
            imgs, hard_classification=False, use_activation=False)[:, 1:, :].transpose(-1, -2).reshape(
            (-1, num_classes, resize_size // patch_size, resize_size // patch_size))
        # Transform image to larger size
        soft_parcelization = pth_transforms.Resize(
            size=resize_size, interpolation=pth_transforms.InterpolationMode.BILINEAR)(soft_parcelization).reshape(
            (B, Fr, num_classes, resize_size, resize_size)).cpu()
        # Hard segmentation larger
        hard_parcelization_larger = torch.argmax(soft_parcelization, dim=2, keepdim=False).numpy().astype(np.float32)

        # Move to cpu
        systolic_masks = mask_transform(targets[ES_trace].detach().cpu())
        diastolic_masks = mask_transform(targets[ED_trace].detach().cpu())

        # Loop through batches
        for it in range(B):

            # The systolic and diastolic masks
            systolic_mask = (systolic_masks[it] == 1)
            diastolic_mask = (diastolic_masks[it] == 1)

            # Get specific frames of parcelization
            parcels_larger = hard_parcelization_larger[it]

            # Get present classes
            systolic_present = np.unique(parcels_larger[ES_frame])
            present_systolic_list.append(systolic_present)
            diastolic_present = np.unique(parcels_larger[ED_frame])
            present_diastolic_list.append(diastolic_present)

            # Get labels from parcels
            systolic_labeling = Segmentation2Labels(parcels_larger[ES_frame])
            diastolic_labeling = Segmentation2Labels(parcels_larger[ED_frame])

            systolic_inner_labels, systolic_valve_labels = misc.compute_prospective_labels(
                systolic_mask, torch.from_numpy(systolic_labeling.labels),
                threshold=interior_parameters["overlap"],
                minimum_segment_ratio=interior_parameters["minimum_segment_ratio"],
                minimum_pixels=interior_parameters["minimum_pixels"])

            systolic_inner_classes = systolic_labeling.get_classes_of_labels(systolic_inner_labels)
            systolic_valve_classes = systolic_labeling.get_classes_of_labels(systolic_valve_labels)
            systolic_list.append(systolic_inner_classes)
            svalve_list.append(systolic_valve_classes)

            diastolic_inner_labels, diastolic_valve_labels = misc.compute_prospective_labels(
                diastolic_mask, torch.from_numpy(diastolic_labeling.labels),
                threshold=interior_parameters["overlap"],
                minimum_segment_ratio=interior_parameters["minimum_segment_ratio"],
                minimum_pixels=interior_parameters["minimum_pixels"])

            diastolic_inner_classes = diastolic_labeling.get_classes_of_labels(diastolic_inner_labels)
            diastolic_valve_classes = diastolic_labeling.get_classes_of_labels(diastolic_valve_labels)
            diastolic_list.append(diastolic_inner_classes)
            diastolic_list.append(diastolic_valve_classes)

            # Append classes exterior to true labeled segment
            exterior_systolic_unique, _ = get_intersection_parcels(
                ground_truth=~systolic_mask,
                parcels=torch.from_numpy(parcels_larger[ES_frame]),
                threshold=exterior_parameters["overlap"])
            exterior_systolic_list.append(
                [cur_parcel for cur_parcel in exterior_systolic_unique if (
                    cur_parcel not in systolic_inner_classes) and (cur_parcel not in systolic_valve_classes)])
            exterior_diastolic_unique, _ = get_intersection_parcels(
                ground_truth=~diastolic_mask,
                parcels=torch.from_numpy(parcels_larger[ED_frame]),
                threshold=exterior_parameters["overlap"])
            exterior_diastolic_list.append(
                [cur_parcel for cur_parcel in exterior_diastolic_unique if (
                    cur_parcel not in diastolic_inner_classes) and (cur_parcel not in diastolic_valve_classes)])

    present_counts = Counter(
        [item for sublist in present_systolic_list for item in sublist
        ] + [item for sublist in present_diastolic_list for item in sublist])
    valve_counts = Counter(
        [item for sublist in svalve_list for item in sublist
        ] + [item for sublist in dvalve_list for item in sublist])
    inside_counts = Counter(
        [item for sublist in systolic_list for item in sublist
        ] + [item for sublist in diastolic_list for item in sublist])
    outside_counts = Counter(
        [item for sublist in exterior_systolic_list for item in sublist
        ] + [item for sublist in exterior_diastolic_list for item in sublist])

    num_all = 2 * len(systolic_list)
    inside_counts = [int(k) for k, v in inside_counts.items() if (
        float(v) / present_counts[k] > interior_parameters["present_fraction"]) and (
        float(v) / num_all > interior_parameters["population_fraction"])]
    valve_counts = [int(k) for k, v in valve_counts.items() if (
        float(v) / present_counts[k] > valve_parameters["present_fraction"]) and (
        float(v) / num_all > valve_parameters["population_fraction"])]
    outside_counts = [int(k) for k, v in outside_counts.items() if (
        float(v) / present_counts[k] > exterior_parameters["present_fraction"]) and (
        float(v) / num_all > exterior_parameters["population_fraction"])]
    print("Inside: ", inside_counts)
    print("Valve: ", valve_counts)
    print("Outside: ", outside_counts)

    # Save values
    save_dict = {'interior_parcels': inside_counts, 'valve_parcels': valve_counts, 'exterior_parcels': outside_counts}
    with open(output_path, 'w') as file_handle:
        json.dump(save_dict, file_handle, indent=4)

    return save_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="Path to config file with needed parameters", required=True)
    parser.add_argument("--backend_path", help="Path to backend model", required=True)
    parser.add_argument("--head_path", help="Path to head model", required=True)
    parser.add_argument("--output_path", help="Path to save output", required=True)
    args = parser.parse_args()

    # Run parcel identifier
    save_dict = main(config_path=args.config_file, backend_path=args.backend_path,
                     head_path=args.head_path, output_path=args.output_path)
