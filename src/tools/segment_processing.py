"""
Preprocessing functionality for segmentation
"""
import numpy as np
from abc import ABC, abstractmethod
import scipy
from skimage import morphology


class BaseSegmentProcessing(ABC):
    """
    Class representing a post-processing step on a segmentation/parcel/binary mask
    """

    @abstractmethod
    def __call__(self, image):
        """
        perform processing
        """
        raise NotImplementedError("Not implemented load_front_end() method")


class SpatialMedianFilter(BaseSegmentProcessing):

    def __init__(self, radius):
        self.radius = radius

    def __call__(self, image):
        return scipy.ndimage.median_filter(image, footprint=morphology.disk(self.radius).reshape(
            (1, self.radius * 2 + 1, self.radius * 2 + 1)))


class TemporalMedianFilter(BaseSegmentProcessing):

    def __init__(self, radius):
        self.radius = radius

    def __call__(self, image):
        return scipy.ndimage.median_filter(image, footprint=np.ones(
            (self.radius)).reshape((-1, 1, 1)))


class ConvexHull(BaseSegmentProcessing):

    def __call__(self, image):
        return np.array([morphology.convex_hull_image(image[it_frame, :, :]) for it_frame in range(image.shape[0])])


class SpatialClosing(BaseSegmentProcessing):

    def __init__(self, radius):
        self.radius = radius

    def __call__(self, image):
        return morphology.binary_closing(image, footprint=morphology.disk(self.radius).reshape(
            (1, self.radius * 2 + 1, self.radius * 2 + 1)))
