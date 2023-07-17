# This file holds code related to tools for parcels identification

# Load packages
import numpy as np
import skimage

from src.tools.metrics import cover_fraction


def get_intersection_parcels(ground_truth, parcels, threshold=.5):
    """Calculate cover fraction between
    ground truth binary mask and predicted selected parcel mask
    Parameters
    ----------
    ground_truth: Tensor
        Tensor consists of 0 and 1 of ground truth mask
    parcels: Tensor
        Tensor of predicted parcels where each parcel correspond to a number for the specified pixel element.
    threshold: float
        Threshold for overlaping fraction
    Returns
    -------
    float
        Value of cover fraction in [0,1]
    list of ints
        indices that are associated with valves
    """
    mask_unique = np.unique(parcels[ground_truth == 1])
    inner_indices = [
        int(idx) for idx in mask_unique if cover_fraction(target=(ground_truth == 1), parcel=((parcels == idx)) >= threshold)]
    botom_border = (np.max(
        skimage.segmentation.find_boundaries(
            ground_truth, mode='iner').astype(int) * np.indices(ground_truth.shape)[0],
        axis=0))
    botom_borderx = np.arange(len(botom_border))[botom_border > 1]
    botom_bordery = botom_border[botom_border > 1]
    valve_indices = list(set(np.unique(parcels[botom_bordery, botom_borderx])) - set(inner_indices))
    return inner_indices, valve_indices
