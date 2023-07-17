# This file holds helper functions

# Load packages
import numpy as np
import torch
import random

from src.models.parcel_identification import get_intersection_parcels


def set_seed(seed: int):
    '''
    Sets seed for randomization in all frameworks used
    '''
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def compute_prospective_labels(true_mask, labels, threshold=None,
                               minimum_segment_ratio=None, minimum_pixels=None):
    # Get areas
    intersection_area = {label: ((labels == label) * (true_mask)).sum() for label in labels.unique().numpy()}
    true_area = true_mask.sum().numpy()

    # Get unique labels that are intersecting enough
    intersecting_labels, valve_labels = get_intersection_parcels(ground_truth=true_mask, parcels=labels, threshold=threshold)
    if minimum_segment_ratio is not None:
        intersecting_labels = np.array(
            [label for label in intersecting_labels if intersection_area[label] / true_area >= minimum_segment_ratio], dtype=int)
    if minimum_pixels is not None:
        intersecting_labels = np.array(
            [label for label in intersecting_labels if intersection_area[label] >= minimum_pixels], dtype=int)

    return intersecting_labels, valve_labels
