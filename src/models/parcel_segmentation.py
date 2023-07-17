# This file holds code related to models for segmenting echocardiograms based on parcels

# Load packages
import numpy as np
import json
from abc import ABC

import cv2
from skimage import segmentation, draw, filters

from src.tools import segment_processing
from src.tools.image_processing import NeighborhoodGraph, Segmentation2Labels


class BaseSegmenter(ABC):
    '''
    Abstract class representing a segmentation method using parcels as input.
    This class serves as a blueprint for how to communicate with these segmenter objects.
    '''

    def __init__(self, post_processing=None, **kwargs):
        self.post_processing_steps = post_processing

    def __call__(self, x, **kwargs):

        # Call post-processing step
        return self.post_processing(x)

    def post_processing(self, segment):
        '''
        Class for post-processing
        '''

        if self.post_processing_steps is not None:
            for cur_post_processing in self.post_processing_steps:
                segment = cur_post_processing(segment)
        return segment


class StarlitSky(BaseSegmenter):
    '''
    Use bordering parcels, when available, together with shape regularization to confine left ventricle.
    '''

    def __init__(self, interior_parcels=None, exterior_parcels=None, file_path=None,
                 closing_radius=10, enclave_below_num_labels=8, active_contour=None,
                 **kwargs):
        '''
        Constructor

        :param interior_parcels: A list of integers. Each integer identifying a parcel which makes up the interior of left ventricle
        '''

        if file_path is not None:
            # Load config
            with open(file_path, 'r') as json_file:
                config = json.load(json_file)
            self.interior_parcels = config["interior_parcels"]
            self.exterior_parcels = config["exterior_parcels"]
        else:
            # Store parcel classes
            self.interior_parcels = interior_parcels
            self.exterior_parcels = exterior_parcels

        self.spatial_closing = None
        if closing_radius is not None:
            self.spatial_closing = segment_processing.SpatialClosing(radius=closing_radius)

        self.enclave_below_num_labels = enclave_below_num_labels
        self.active_contour = active_contour

        super(StarlitSky, self).__init__(**kwargs)

    def __call__(self, parcels, **kwargs):
        segment = parcels * False
        neighborhood_segment = parcels * False

        # Compute all sub segments
        for iter_t in range(segment.shape[0]):

            # Get neighborhood structure of parcels
            neighborhood = NeighborhoodGraph(parcels[iter_t, :, :], compute_graph=True)

            # Get total segment from interior parcels
            interior_segment = np.isin(parcels[iter_t, :, :], self.interior_parcels)
            # Get labels from interior segment
            interior_labeling = Segmentation2Labels(interior_segment, background=0)
            # Get dictionary of (interior) parcels associated with each interior label
            num_parcels_per_label = {label: np.sum(np.isin(neighborhood.get_parcels_of_mask(
                interior_labeling.get_mask_from_label(label)),
                self.interior_parcels)) for label in interior_labeling.label_list if label != 0}

            # Sort 'parcels_per_label' from largest to smallest
            sort_index = np.argsort(list(num_parcels_per_label.values()))[::-1]
            num_parcels_per_label = {
                label: num_parcels_per_label[label] for label in np.array(list(num_parcels_per_label.keys()), dtype=int)[sort_index]}

            # Find interior label with the largest number of interior parcels
            main_interior_label = list(num_parcels_per_label.keys())[0]
            # Get parcel labels of main interior label
            interior_labels = np.array(neighborhood.labeling.get_labels_from_mask(
                interior_labeling.get_mask_from_label(main_interior_label)))

            # Identify enclave parcels,i.e., parcels that are with a few numbers completely cutoff from
            # the outer world by the interior parcels
            enclave_labels = neighborhood.get_encapsulated_labels(
                interior_labels, below_num_labels=self.enclave_below_num_labels)
            interior_labels = np.append(interior_labels, enclave_labels)

            # Get segment from snake cropped by neighb mask
            segment[iter_t, :, :] = neighborhood.labeling.get_mask_from_labels(interior_labels)
            neighborhood_segment[iter_t, :, :] = neighborhood.labeling.get_mask_from_labels(enclave_labels)

        if self.spatial_closing is not None:
            segment = self.spatial_closing(segment)

        if self.active_contour is not None:
            # Compute active contours
            for iter_t in range(segment.shape[0]):
                contours, _ = cv2.findContours(segment[iter_t, :, :].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                verts = []
                for object in contours:
                    coords = []
                    for point in object:
                        coords.append([int(point[0][1]), int(point[0][0])])
                    verts.extend(coords)
                verts = np.array(verts)
                snake = segmentation.active_contour(filters.gaussian(segment[iter_t, :, :], 3, preserve_range=False), verts,
                    alpha=0.0, beta=self.active_contour['beta'], gamma=self.active_contour['gamma'],
                    w_line=0, w_edge=self.active_contour['w_edge'], max_num_iter=self.active_contour['max_num_iter'])
                # Get segment from snake cropped by neighb mask
                segment[iter_t, :, :] = draw.polygon2mask(segment.shape[1:], snake)

        # Call parent functor
        segment = super(StarlitSky, self).__call__(segment, **kwargs)
        return segment, neighborhood_segment
