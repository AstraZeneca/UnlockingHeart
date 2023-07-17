# This file holds executibe code for evaluation of the segmantation pipeline

# Load packages
import argparse
import os
import sys
import numpy as np
import tqdm
import yaml
import json
import csv

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
from src.tools import preprocessing, segment_processing, misc, vis

from src.tools.metrics import dice_metric, iou


def main(config, backend_path, head_path, segmentation_path, output_path, num_samples=5):
    # set parameters
    num_classes = config['head']['num_classes']
    patch_size = config['backend']['patch_size']
    dataset_name = config['dataset']['name']
    resize_size = config['dataloader']['resize_size']
    batch_size = config['dataloader']['batch_size']

    if output_path is not None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # opening the csv file
        data_file = open(os.path.join(output_path, f'{dataset_name}_results.csv'), 'w')
        writer = csv.DictWriter(data_file,
                                fieldnames=['ID', 'Name',
                                            'DICE_ES', 'DICE_ED',
                                            'IoU_ES', 'IoU_ED'])
        writer.writeheader()

    # Set seed
    if "seed" in cfg:
        misc.set_seed(cfg['seed'])

    # Setup model
    # Define device for copmutations
    device = torch.device("cpu")
    if "device" in config:
        device = torch.device(config["device"])
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
    if 'transforms' in config['dataset']:
        for t in config['dataset']['transforms']:
            sname = t.pop('name')
            if sname == 'CLAHE' and dataset_name == 'CAMUS':
                t['scale'] /= std
                t['translate'] = - mean / std
            transform_process = preprocessing.__dict__[sname](**t)
            preprocessing_list.append(transform_process)
            if sname == 'VerticalFlip' or sname == 'HorizontalFlip' or sname == 'Rotation':
                mask_transform.append(transform_process)
        config['dataset'].pop('transforms')
    config['dataset']['preprocessing'] = preprocessing_list
    mask_transform = pth_transforms.Compose(mask_transform)

    DataWithIndices = DatasetWithIndices(dataset_class)
    dataset = DataWithIndices(resize_size=[resize_size, resize_size],
                              device=device,
                              **config['dataset'])

    unnormalize = preprocessing.Unnormalize(mean=mean, std=std)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    # Create segmenter
    if config['segmentation']['name'] not in parcel_segmentation.__dict__.keys():
        raise Exception("Segmentation method given in config file does not exist!")

    # Load segmentation
    with open(segmentation_path, 'r') as file_handle:
        segmentation_data = json.load(file_handle)

    # Acquire post-processing
    transforms_list = []
    for t in config['segmentation']['transforms']:
        sname = t.pop('name')
        transforms_list.append(segment_processing.__dict__[sname](**t))

    # Handle segmenter
    segmenter = parcel_segmentation.__dict__[config['segmentation']['name']](**segmentation_data,
                                                                             post_processing=transforms_list,
                                                                             **config['segmentation']['parameters'])

    # Go through dataset
    # Set model as under evaluation
    model.eval()
    # Set that gradients should not be computed
    model.requires_grad_(False)

    # list to store results
    dice_metric_ES = []
    dice_metric_ED = []
    iou_metric_ES = []
    iou_metric_ED = []
    # Loop through testset
    for imgs, targets, idxs in tqdm.tqdm(iter(data_loader)):

        # Get dimensions of images
        B, Fr, C, H, W = imgs.size()

        _, hard_parcelization_larger = model_predict(model, imgs, num_classes,
                                                     resize_size, patch_size)
        # Move to cpu
        ES_masks = mask_transform(targets[0].detach().cpu())
        ED_masks = mask_transform(targets[1].detach().cpu())

        # Loop through batches
        item_data = {}
        for it in range(B):
            idx = idxs[it]
            if dataset_name == 'CAMUS':
                exp_name = f"{dataset.patient_paths[idx % len(dataset.patient_paths)].name}_{dataset.views[idx // len(dataset.patient_paths)]}"
            elif dataset_name == 'EchoNet_Dynamic':
                exp_name = f"{dataset.dataset.fnames[idx]}"
            else:
                exp_name = None
                print(f'Not suported dataset type {dataset_name}')
            item_data['ID'] = int(idx)
            item_data['Name'] = exp_name

            # Extract image sequence
            img = imgs[it]
            img = unnormalize(img.cpu()).numpy()
            img = img / np.max(img)

            # The systolic and diastolic masks
            ES_mask = ES_masks[it].numpy()
            ED_mask = ED_masks[it].numpy()

            # Get specific frames of parcelization
            parcels_larger = hard_parcelization_larger[it]

            # Acquire segment from parcels
            segment, _ = segmenter(parcels_larger)

            # Define end-systole and end-diastole frames
            areas = np.sum(segment, axis=(1, 2))
            ES_frame = np.argmin(areas)
            ED_frame = np.argmax(areas)

            # Append metrics
            dice_metric_ES.append(dice_metric(segment[ES_frame, :, :], ES_mask))
            dice_metric_ED.append(dice_metric(segment[ED_frame, :, :], ED_mask))
            iou_metric_ES.append(iou(segment[ES_frame, :, :], ES_mask))
            iou_metric_ED.append(iou(segment[ED_frame, :, :], ED_mask))
            item_data['DICE_ES'] = dice_metric_ES[-1]
            item_data['DICE_ED'] = dice_metric_ED[-1]
            item_data['IoU_ES'] = iou_metric_ES[-1]
            item_data['IoU_ED'] = iou_metric_ED[-1]

            if output_path is not None:
                writer.writerow(item_data)

            if 'plot' in config or idx < num_samples:
                seq_save_path = os.path.join(output_path,
                                             f"{int(dice_metric(segment[ES_frame, :, :], ES_mask)*100)}_{exp_name}.png")
                vis.vis_end_segments(img, parcels_larger, segment,
                                     ES_mask, ED_mask, ES_frame, ED_frame,
                                     seq_save_path, disk_radius=1)

    results_dict = {
        "DICE":
        {
            "ES":
            {
                "mean": float(np.nanmean(dice_metric_ES)),
                "std": float(np.nanstd(dice_metric_ES))
            },
            "ED":
            {
                "mean": float(np.nanmean(dice_metric_ED)),
                "std": float(np.nanstd(dice_metric_ED))
            },
        },
        "IOU":
        {
            "ES":
            {
                "mean": float(np.nanmean(iou_metric_ES)),
                "std": float(np.nanstd(iou_metric_ES))
            },
            "ED":
            {
                "mean": float(np.nanmean(iou_metric_ED)),
                "std": float(np.nanstd(iou_metric_ED))
            },
        }
    }

    if output_path is not None:
        with open(os.path.join(output_path, "evaluation_results.json"), "w") as fh:
            json.dump(results_dict, fh)
        # close results file
        data_file.close()
        if 'plot' in config or 'plot_hist' in config:
            # plot results
            hist_save_path = os.path.join(output_path, "hist_plots.png")
            vis.plot_histograms(dice_metric_ES, dice_metric_ED,
                                 hist_save_path)
    print("Metrics:")
    print(results_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="Config file with evaluation parameters", required=True)
    parser.add_argument("-t", "--target_type",
                        default=["ESTrace", "EDTrace"], nargs='+',
                        help="Name of ES and ED heart parts to segment, exacly in this order", required=False)
    parser.add_argument("--backend_path", help="Path to backend model", required=True)
    parser.add_argument("--head_path", help="Path to head model", required=True)
    parser.add_argument("--segmentation_path", help="Path to segmentation model", required=True)
    parser.add_argument("--output_path", default=None, help="Path to output directory", required=False)

    args = parser.parse_args()
    # Get parameters from file
    # Load config
    with open(args.config_file, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)
    cfg['dataset']['target_type'] = args.target_type

    main(config=cfg, backend_path=args.backend_path,
         head_path=args.head_path, segmentation_path=args.segmentation_path,
         output_path=args.output_path)
