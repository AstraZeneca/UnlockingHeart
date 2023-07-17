"""
Functionality for tracing parcels
"""

# Load packages
import math
import numpy as np
from skimage import measure, morphology
import networkx as nx


class SegmentationVisualization:
    """
    Class for operations related to visualizing segmentations
    """

    def __init__(self, segments: np.array):
        self.segments = segments
        self.classes = np.unique(segments).astype(int)

    def bitmapped_boundaries(self, segments=None, disk_radius=1):
        """
        Computes a bitmapped mask which highlights separations between segments.

        Parameters
        ----------
        segments : 2D-array of integers
            The segmented image to analyze. If not given, the segmented image from constructor is used instead. 
            Each element in the array correspond to a pixel and the value is the segment assigned to that pixel.
        disk_radius : int
            The radius of a disk-shaped kernel used for to acquire the separation of the segments using morphological erosion.
        
        Returns
        -------
        A mask of the same dimensions as 'segments'.
            Returns a mask where only the boundaries between segments are 'true'.
            
        """

        if segments is None:
            segments = self.segments

        disk_kernel = morphology.disk(disk_radius)

        # Get empty bitmap of borders
        borders = np.zeros(segments.shape, dtype=bool)

        # For each class, get border
        for it_class in self.classes:
            borders = borders + (morphology.binary_erosion((segments == it_class), disk_kernel) > 0)

        return ~borders

    def text_positions(self, segments=None):
        """
        Compute x- and y- positions of centroid of each segment.
        """
        if segments is None:
            segments = self.segments

        locations = {}

        # Loop through all classes
        for it_class in self.classes:
            # Get mask of current segment
            cur_segment = (segments == it_class)
            # Compute all sub segments
            labels = measure.label(cur_segment, background=0)
            # Generate props object
            props = measure.regionprops(labels, cur_segment)
            # Acquire centers of current segment
            locations[it_class] = np.array([getattr(prop, 'centroid') for prop in props])
        return locations


class Segmentation2Labels:    
    """
    Segmentations divides an image into mutually exclusive and exhaustive segments.
    However, a segment does not necessarily need to be connected.
    This class will create a relationship between all segments in the original segmentation
    and a new segmentation where each segments does not have a non-connected region.
    """

    def __init__(self, segments: np.array, background: int=None, identity: bool=False):
        
        self.props = None
        self.class_list = np.unique(segments)
        
        # If should not label connected
        if identity:
            self.labels = segments
            self.label_list = self.class_list
            self.mapping = np.identity(self.class_list.size, dtype=bool)
        else:

            if background is None:
                background = np.max(self.class_list) + 1

            self.labels = measure.label(segments, background=background)
            self.label_list = np.unique(self.labels)

            self.mapping = np.zeros((self.class_list.size, self.label_list.size), dtype=bool)

            for iter_classes_index in range( self.class_list.size ):
                self.mapping[iter_classes_index, :] = np.isin(
                    self.label_list, np.unique(self.labels[segments == self.class_list[iter_classes_index]]))

        self.num_labels = self.label_list.size

    def get_labels_of_classes(self, classes):
        '''
        Get which labels corresponds to given classes
        '''
        class_indices = self._get_class_indices_from_classes(classes)
        label_indices = np.nonzero(np.sum(self.mapping[ class_indices, : ], axis=0))[0]
        labels = self._get_labels_from_label_indices(label_indices)

        return labels

    def get_classes_of_labels(self, labels):
        '''
        Get which classes correspond to given labels
        '''
        label_indices = self._get_label_indices_from_labels(labels)
        class_indices = np.nonzero( np.sum( self.mapping[:, label_indices], axis=1))[0]
        classes = self._get_classes_from_class_indices(class_indices)
        return classes

    def _get_class_indices_from_classes(self, class_list):
        '''
        Get class indices from classes
        '''
        class_indices = [np.where(self.class_list == class_)[0] for class_ in class_list]
        class_indices = np.array(
            [index[0] for index in class_indices if len(index) > 0], dtype=int)
        return class_indices

    def _get_classes_from_class_indices(self, class_indices):
        '''
        Get class indices from classes
        '''
        return self.class_list[class_indices]

    def _get_label_indices_from_labels(self, label_list):
        '''
        Get label indices from labels
        '''
        label_indices = [np.where(self.label_list == label)[0] for label in label_list]
        label_indices = np.array(
            [index[0] for index in label_indices if len(index) > 0 ], dtype=int)
        return label_indices

    def _get_labels_from_label_indices(self, label_indices):
        '''
        Get label indices from labels
        '''
        return self.label_list[label_indices]

    def _measure_labels(self):
        '''
        Measure general properties of different labels
        '''

        if self.props is None:
            # Get properties of masked region
            self.props = measure.regionprops(self.labels)

        return self.props

    def measure_area(self, label_list : list = [], sort : bool = False):
        '''
        Measure area of each label. 

        Parameters
        ----------
        label_list : list of label ids
            If given, the labels for which area should be measured
        sort : bool
            Flag for if the label id:s should be sorted according to their area.

        Returns
        -------
        Dictionary of labels (as keys) and their areas (as values).
        '''

        props = self._measure_labels()

        if len(label_list) == 0:
            label_list = self.label_list

        # Compute area
        areas = {prop.label : prop.area for prop in props if prop.label in label_list}

        sort_index = None
        if sort:
            # Sort area from largest to smallest
            sort_index = np.argsort( list(areas.values()) )[::-1]
            areas = { label : areas[label] for label in np.array(list(areas.keys()), dtype=int)[sort_index]  }

        return areas

    def measure_com(self, label : int):
        '''
        Compute center of mass of label. 
        
        Parameters
        ----------
        label : int
            The label to return the center of mass for
    
        Returns
        -------
        dictionary of float
        Y- and x- values of centroid with keys: "y" and "x".
        
        '''
        props = self._measure_labels()

        # Get properties for chosen label
        prop = [ prop for prop in props if prop.label == label ]
        if len(prop) == 0:
            raise Exception("Label not present!")
        prop = prop[0]

        # Get centroid (as y, x)
        centroid = prop.centroid

        return {"y": centroid[0], "x": centroid[1]}

    def measure_bbox(self, label : int):
        '''
        Compute bounding box of label. 

        Parameters
        ----------
        label : int
            The label to return the perimeter of

        Returns
        -------
        Dictionary of the bounding box values with keys 'x_min', 'x_max', 'y_min', 'y_max'.

        '''   
        props = self._measure_labels()

        # Get properties for chosen label
        prop = [prop for prop in props if prop.label == label]
        if len(prop) == 0:
            raise Exception("Label not present!")
        prop = prop[0]

        # Get bounding box
        (min_row, min_col, max_row, max_col) = prop.bbox

        return {"x_min": min_col, "x_max": max_col, "y_min": min_row, "y_max": max_row}

    def measure_perimeter(self, label : int):
        '''
        Compute perimeter of label. 
        
        Parameters
        ----------
        label : int
            The label to return the perimeter of
            
        Returns
        -------
        Scalar of the perimeter of the label
        
        '''
        props = self._measure_labels()

        # Get properties for chosen label
        prop = [prop for prop in props if prop.label == label]
        if len(prop) == 0:
            raise Exception("Label not present!")
        prop = prop[0]

        # Get centroid (as y, x)
        perimeter = prop.perimeter
        
        return perimeter

    def get_mask_from_labels(self, labels):
        '''
        Get mask from joining labels
        '''
        return np.isin(self.labels, labels)

    def get_mask_from_label(self, label):
        '''
        Get mask from joining label
        '''
        return self.get_mask_from_labels([label])

    def get_labels_from_mask(self, mask):
        '''
        Get all labels which overlap the slightest with given mask
        '''

        return np.unique(self.labels[mask])


class NeighborhoodGraph:
    """
    Calculates and represents the neighborhood structure of segments
    """

    def __init__(self, segments: np.array, order: int=1,
                 separate_instances: bool=True, compute_graph: bool=False):
        """
        Computes the neighborhood graph of a segmented image

        Parameters
        ----------
        segments : 2D-array of integers
            The segmented image to analyze.
            Each element in the array correspond to a pixel and the value is the segment assigned to that pixel.
        order : integer
            The order of the neighborhood structure to consider.
            A value of 1 means that only pixels closest above or besides are neighbors to a pixel.
            A value of 2 means that also the diagonal pixels are included. For larger values one can think of
            the neighbors as being defined by the radius of a disk (measure in pixel lengths)
            stretching from the center of the origin pixel. Whichever pixel centers such a disk covers will be considered a neighbor. 
        separate_instances : bool
            If 'False', subsets of the same class that are completely separated in space (other classes in between)
            are considered as different entities. 
        compute_graph : bool
            If 'True', the graph structure will be computed.
        """

        # Generate neighborhood disk
        disk = morphology.disk(order)

        self.labeling = Segmentation2Labels(segments, identity=(not separate_instances))

        # Initialize neighborhood array
        self.neighbors = np.identity(self.labeling.num_labels, dtype=int)
        self.neighbors[self.neighbors == 0] = np.iinfo(np.int32).max
        self.neighbors[self.neighbors == 1] = 0

        # Loop through each label
        for iter_label in range(self.labeling.num_labels):
            temp = self.labeling.get_mask_from_label(self.labeling.label_list[iter_label])
            temp = morphology.binary_dilation( temp, disk )
            # Loop through each label again
            for iter_label2 in range(self.labeling.num_labels):
                if (iter_label2 != iter_label):
                    # If the dilated mask is overlapping with label iter_label2
                    if ( np.any( self.labeling.get_mask_from_label(self.labeling.label_list[iter_label2]) & temp)):
                        self.neighbors[iter_label2, iter_label] = 1

        # If neighbor of neighbors should be computed
        self.graph = None
        if compute_graph:
            self.compute_graph()

    def compute_graph(self):
        '''
        Compute graph from nearest neighbor information
        '''
        if self.graph is not None:
            return
        # Create empty graph
        self.graph = nx.Graph()
        # Create nodes from label list
        self.graph.add_nodes_from(range(self.labeling.num_labels))
        # go through each label index
        for iter_label_index in range(self.labeling.num_labels):
            # Find all first-order neighbors labels
            iter_neighb_labels =  np.where( self.neighbors[iter_label_index, :] == 1)[0]
            # Add edges between current node and its neighbors
            self.graph.add_edges_from([
                (iter_label_index, cur_neighb_label_index) for cur_neighb_label_index in iter_neighb_labels])
        # compute shortest path between all pairs
        lengths = dict(nx.all_pairs_shortest_path_length( self.graph ))
        # Populate neighbors matrix with number of edges between indices
        for iter_label_index in range(self.labeling.num_labels):
            for iter_label_index2 in range(self.labeling.num_labels):
                self.neighbors[iter_label_index, iter_label_index2] = lengths[iter_label_index][iter_label_index2]

    def get_neighbors_of(self, label_list:list, distance: int=1):
        """
        Get labels which are neighbors of labels in list

        Parameters
        ----------
        label_list : list of integers
            A list of label numbers. Give labels which are neighbors to those.

        """
        if distance > 1:
            self.compute_graph()

        # Translate indices of labels given
        indices = self.labeling._get_label_indices_from_labels(label_list)
        # Get indices which are 'distance' away from given labels
        indices = np.where(np.any(self.neighbors[indices, : ] == distance, axis=0))[0]
        # Get labels corresponding to given indices
        labels = self.labeling._get_labels_from_label_indices(indices)
        return labels

    def get_encapsulated_labels( self, exterior_labels, below_num_labels=None):
        '''
        Get labels which are encapsulated inside 'exterior_labels'
        '''
        # Compute graph, if not already
        self.compute_graph()
        # Get graphs when removing exterior labels
        graph = self.graph.copy()
        graph.remove_nodes_from( self.labeling._get_label_indices_from_labels( exterior_labels) )
        # Get subgraphs after removal
        subgraphs = list( nx.connected_components( graph ) )
        # If given, only use subgraphs below a value
        if below_num_labels is not None:
            subgraphs = [ subgraph for subgraph in subgraphs if len(subgraph) <= below_num_labels ]
        subgraphs = [ self.labeling._get_labels_from_label_indices( list( subgraph ) ) for subgraph in subgraphs  ]
        return subgraphs

    def get_parcels_of_mask(self, mask):
        """
        Get which parcels are part of mask.

        Parameters
        ----------
        mask : 2D-array of booleans
            The mask to investigate

        Returns
        -------
        A list of parcel numbers that are the slightest overlapping with mask
            
        """

        # Get label names of labels that are overlapping the slightest with mask
        labels = self.labeling.get_labels_from_mask(mask)
        # Get the parcels corresponding to these labels
        parcels = self.labeling.get_classes_of_labels(labels)

        return parcels
