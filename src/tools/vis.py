# Load packages
import os
from matplotlib import pyplot as plt
import numpy as np

# Import own packagea
from src.tools.image_processing import SegmentationVisualization


def vis_end_segments(sequence, all_parcels, segment,
                     ES_mask, ED_mask, ES_frame, ED_frame,
                     save_path, disk_radius=1):
    # Acquire boundaries
    parcels_borders = np.zeros(all_parcels.shape, dtype=bool)
    borderVisualizer = SegmentationVisualization(all_parcels)
    for it_frame in range(all_parcels.shape[0]):
        parcels_borders[it_frame, :, :] = borderVisualizer.bitmapped_boundaries(
            segments=all_parcels[it_frame, :, :], disk_radius=disk_radius)

    # Overlay true and predicted segmentation
    overlayES = sequence.transpose((0, 2, 3, 1))[ES_frame, :, :, :]
    overlayES = 0.7 * overlayES
    overlayES[:, :, 2] = np.clip(
        overlayES[:, :, 2] + 0.6 * np.clip(ES_mask, a_min=None, a_max=1), a_min=None, a_max=1)
    overlayES[:, :, 1] = np.clip(
        overlayES[:, :, 1] + 0.6 * np.clip(
            segment[ES_frame, :, :], a_min=None, a_max=1), a_min=None, a_max=1)
    overlayED = sequence.transpose((0, 2, 3, 1))[ED_frame, :, :, :]
    overlayED = 0.7 * overlayED
    overlayED[:, :, 2] = np.clip(
        overlayED[:, :, 2] + 0.6 * np.clip(ED_mask, a_min=None, a_max=1), a_min=None, a_max=1)
    overlayED[:, :, 1] = np.clip(
        overlayED[:, :, 1] + 0.6 * np.clip(
            segment[ED_frame, :, :], a_min=None, a_max=1), a_min=None, a_max=1)

    overlay_bordersS = sequence.transpose((0, 2, 3, 1))[ES_frame, :, :, :]
    overlay_bordersS = 0.7 * overlay_bordersS
    overlay_bordersS[:, :, 2] = np.clip(
        overlay_bordersS[:, :, 2] + 0.6 * parcels_borders[ES_frame, :, :], a_min=None, a_max=1)
    overlay_bordersS[:, :, 1] = np.clip(
        overlay_bordersS[:, :, 1] + 0.6 * np.clip(
            segment[ES_frame, :, :], a_min=None, a_max=1), a_min=None, a_max=1)
    overlay_bordersD = sequence.transpose((0, 2, 3, 1))[ED_frame, :, :, :]
    overlay_bordersD = 0.7 * overlay_bordersD
    overlay_bordersD[:, :, 2] = np.clip(
        overlay_bordersD[:, :, 2] + 0.6 * parcels_borders[ED_frame, :, :], a_min=None, a_max=1)
    overlay_bordersD[:, :, 1] = np.clip(
        overlay_bordersD[:, :, 1] + 0.6 * np.clip(
            segment[ED_frame, :, :], a_min=None, a_max=1), a_min=None, a_max=1)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(5, 5), dpi=120)
    axes[0, 0].imshow(overlayES)
    axes[0, 0].title.set_text('End-systole frame')
    axes[0, 1].imshow(overlayED)
    axes[0, 1].title.set_text('End-diastole frame')
    axes[1, 0].imshow(overlay_bordersS)
    axes[1, 1].imshow(overlay_bordersD)
    plt.savefig(save_path)
    plt.close()


def plot_histograms(dice_metric_ES, dice_metric_ED, save_path):
    fig, axs = plt.subplots(1, 2, figsize=(9, 6), dpi=120)
    axs[0].hist(dice_metric_ES)
    axs[0].set_title("End-systole DICE")
    axs[1].hist(dice_metric_ED)
    axs[1].set_title("End-diastole DICE")
    plt.savefig(os.path.join(save_path))
    plt.close()
