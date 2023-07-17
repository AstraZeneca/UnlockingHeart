# This file holds functions required to calcule metrics


# Dice Coefficient
def dice_metric(inputs, target):
    """Calculate DICE metric between
    ground truth binary mask and predicted segmentation mask
    Parameters
    ----------
    inputs: Tensor
        Tensor consists of 0 and 1 of predicted mask
    target: Tensor
        Tensor consists of 0 and 1 of ground truth mask
    Returns
    -------
    float
        Value of DICE metric in [0,1]
    """
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0
    return intersection / union


# Jaccard’s Index (Intersection over Union, IoU)
def iou(inputs, target):
    """Calculate IoU metric between
    ground truth binary mask and predicted segmentation mask
    Parameters
    ----------
    inputs: Tensor
        Tensor consists of 0 and 1 of predicted mask
    target: Tensor
        Tensor consists of 0 and 1 of ground truth mask
    Returns
    -------
    float
        Value of IoU metric in [0,1]
    """
    intersection = (target * inputs).sum()
    union = target.sum() + inputs.sum() - intersection
    if union == 0 and intersection == 0:
        return 0
    if target.sum() + inputs.sum() == intersection:
        return 1.0
    return intersection / union


def cover_fraction(target, parcel):
    """Calculate cover fraction between
    ground truth binary mask and predicted selected parcel mask
    Parameters
    ----------
    target: Tensor
        Tensor consists of 0 and 1 of ground truth mask
    parcel: Tensor
        Tensor consists of 0 and 1 of selected parcel
    Returns
    -------
    float
        Value of cover fraction in [0,1]
    """
    return (target & parcel).sum() / parcel.sum()
