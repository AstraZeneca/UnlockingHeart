"""
Preprocessing functionality for images
"""
import numpy as np
import os
from PIL import Image
import skimage
import sys
import torch
from torchvision import transforms as pth_transforms
from abc import ABC, abstractmethod
import cv2
from skimage.util import random_noise

# Import dino package
sys.path.append(os.path.join(os.path.dirname(__file__), "../../lib/dino/"))
import utils as dino_utils


class BasePreProcessing(ABC):
    """
    Class representing a pre-processing operation
    """

    @abstractmethod
    def __call__(self, image):
        """
        perform pre-processing
        """
        raise NotImplementedError("Not implemented load_front_end() method")


class Rotation(BasePreProcessing):
    """Rotate the image by angle.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image[.Resampling].NEAREST``) are still accepted,
            but deprecated since 0.13 and will be removed in 0.15. Please use InterpolationMode enum.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (sequence, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number): Pixel fill value for the area outside the rotated
            image. Default is ``0``. If given a number, the value is used for all bands respectively.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """
    def __init__(self, degrees, interpolation=pth_transforms.InterpolationMode.NEAREST,
                 expand=False, center=None, fill=0):
        if type(degrees) == str:
            degrees = list([float(i) for i in degrees[1:-1].split(",")])
        self.degrees = degrees
        self.interpolation = interpolation
        self.expand = bool(expand)
        if type(center) == str:
            center = list([int(i) for i in center[1:-1].split(",")])
        self.center = center
        self.fill = fill
        self.transform = pth_transforms.RandomRotation(degrees=self.degrees, interpolation=self.interpolation,
                                                       expand=self.expand, center=self.center, fill=self.fill)

    def __call__(self, image):
        return self.transform(image)


class SaltPaper(BasePreProcessing):
    """
    Default ratio argument value is 0.5.
    This means that the ratio of the salt to pepper noise is going to be equal.
    """
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, image):
        return torch.tensor(random_noise(image, mode='salt', amount=self.ratio))


class VerticalFlip(BasePreProcessing):

    def __init__(self, p=0.5):
        self.p = p
        self.transform = pth_transforms.RandomVerticalFlip(p=self.p)

    def __call__(self, image):
        return self.transform(image)


class HorizontalFlip(BasePreProcessing):

    def __init__(self, p=0.5):
        self.p = p
        self.transform = pth_transforms.RandomHorizontalFlip(p=self.p)

    def __call__(self, image):
        return self.transform(image)


class Resize(BasePreProcessing):

    def __init__(self, resize_size):
        self.resize_size = resize_size
        self.transform = pth_transforms.Resize(resize_size)

    def __call__(self, image):
        return self.transform(image)


class Normalize(BasePreProcessing):

    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.transform = pth_transforms.Normalize(self.mean, self.std)

    def __call__(self, image):
        return self.transform(image)


class Unnormalize(BasePreProcessing):

    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.multiply = pth_transforms.Normalize(0, 1 / self.std)
        self.add = pth_transforms.Normalize(-self.mean, 1)

    def __call__(self, image):
        image = self.multiply(image)
        image = self.add(image)
        return image


class Transpose(BasePreProcessing):

    def __init__(self, switch1, switch2):
        self.switch1 = switch1
        self.switch2 = switch2

    def __call__(self, image):
        return image.transpose(self.switch1, self.switch2)


class MedianFilter(BasePreProcessing):
    """
    Median filter

    """

    def __init__(self, radius):

        self.radius = radius
        self.disk = skimage.morphology.disk(self.radius)
        self.disk = np.expand_dims(self.disk, (0))

    def __call__(self, image):

        image = image.numpy()
        if image.ndim == 3:
            image = np.expand_dims(image, 0)
        for it in range(image.shape[0]):
            image[it, :, :, :] = skimage.filters.median(
                image[it, :, :, :], footprint=self.disk, behavior='ndimage')
        image = torch.from_numpy(image)

        return image


class CLAHE(BasePreProcessing):
    """
    Contrast limited adaptive histogram equalization
    """

    def __init__(self, clipLimit, tileGridSize, translate=0, scale=1.0):
        """
        clipLimit : float
        A threshold value for where to clip histogram bins in order to avoid overamplification
        of contrasts in homogeneous regions. Expressed as a ratio of all pixel values
        tileGridSize : 2-tuple of integers
        The (row, column) size of sub-grids to perform histogram equialization over.
        """
        if type(tileGridSize) == str:
            tileGridSize = list([int(i) for i in tileGridSize[1:-1].split(",")])

        self.clahe = cv2.createCLAHE(clipLimit=clipLimit * tileGridSize[0] * tileGridSize[1], tileGridSize=tileGridSize)
        self.translate = translate
        self.scale = scale

    def __call__(self, image):

        image = image.numpy()
        if image.ndim == 3:
            image = np.expand_dims(image, 0)

        image = image.transpose((0, 2, 3, 1))
        image = np.clip((image - self.translate) / self.scale, a_min=0, a_max=255).astype(np.uint8)

        for it in range(image.shape[0]):
            lab = cv2.cvtColor(image[it, :, :, :], cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            lab = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            image[it, :, :, :] = lab

        image = image.transpose((0, 3, 1, 2))
        image = image.astype(np.float32)
        image = image * self.scale + self.translate
        image = torch.from_numpy(image)

        return image


# Code from: https://github.com/facebookresearch/dino/blob/main/main_dino.py#L419
class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, img_size):
        flip_and_color_jitter = pth_transforms.Compose([
            pth_transforms.RandomHorizontalFlip(p=0.5),
            pth_transforms.RandomApply(
                [pth_transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            pth_transforms.RandomGrayscale(p=0.2),
        ])
        normalize = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = pth_transforms.Compose([
            pth_transforms.RandomResizedCrop(img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            dino_utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = pth_transforms.Compose([
            pth_transforms.RandomResizedCrop(img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            dino_utils.GaussianBlur(0.1),
            dino_utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = pth_transforms.Compose([
            pth_transforms.RandomResizedCrop( np.round(img_size * np.mean(np.array(local_crops_scale))), scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            dino_utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
