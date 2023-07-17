# Base dataloader functionality

import torch
from torchvision import datasets

from src.tools.preprocessing import Resize
from abc import ABC, abstractmethod


def DatasetWithIndices(cls):
    """
    Modifies the given Dataset class to return a tuple: (data, target, index)
    instead of just: (data, target).
    """
    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls, ), {
        '__getitem__': __getitem__,
    })


class ReturnIndexDataset(datasets.ImageFolder):
    """
    Modifies the given Dataset class to return a tuple: (data, index).
    """
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


class BaseDataset(ABC):
    """
    Class representing the abstract class for a data loader of echocardiograms.
    """

    def __init__(self,
                 device: str = "cpu",
                 preprocessing: list = None,
                 resize_size: list = None,
                 **kwargs):
        """
        Constructor

        Parameters
        ----------
        device : str.
            Define which device data should be retrieved to. Options are: 'cpu', 'cuda'.
        preprocessing : list.
            A list of objects inherited from 'BasePreProcessing'.
            These objects will be used for preprocessing data, in the order from first to last in this list.
        """
        self.device = device
        self.preprocessing = preprocessing

        if self.preprocessing is None:
            self.preprocessing = self.get_default_preprocessing()

        self.resize_transform = None
        if resize_size is not None:
            self.resize_transform = Resize(resize_size)

    def preprocess(self, img):
        """
        Method to perform pre-processing on images

        Parameters
        ----------
        img : Fr x H x W.
            Image sequence to perform pre-processing on.

        Returns
        -------
        image
        """

        # Go through preprocessing
        for preprocessing in self.preprocessing:
            img = preprocessing(img)

        # send to device
        if self.device is not None:
            img = img.to(self.device)

        if self.resize_transform is not None:
            img = self.resize_transform(img)

        # Return preprocessed image
        return img

    def get_data_loader(self, batch_size=1, shuffle=True):
        """
        Method to acquire dataloader from dataset

        Parameters
        ----------
        batch_size : int.
            Number batches for dataloader.
        shuffle : bool.
            If true, the output should be randomized in the dataloader.

        Returns
        -------
        dataloader.
            A torch dataloader object.
        """

        return torch.utils.data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle)

    @abstractmethod
    def __len__(self):
        """
        Method to acquire number of items in dataset

        Returns
        -------
        length : int.
            The number of item in dataset
        """

        raise NotImplementedError("Not implemented load_front_end() method")

    @abstractmethod
    def __getitem__(self, index):
        """
        Method to acquire item from dataset

        Parameters
        ----------
        index : int.
            Number identifying data.

        Returns
        -------
        (img, label).
            First element is a sequence of images (Fr x H  x W).
            Second element is the labels acquired for the sequences.
        """

        raise NotImplementedError("Not implemented load_front_end() method")

    @staticmethod
    @abstractmethod
    def get_normalization():
        """
        Method to acquire normalization constants

        Returns
        -------
        dictionary.
            Returns a dictionary, with at least, 'mean'- and 'std'-vectors.
        """

        raise NotImplementedError("Not implemented load_front_end() method")

    @staticmethod
    @abstractmethod
    def get_default_preprocessing():
        """
        Method to acquire the list of "standard" preprocessing for the given dataset

        Returns
        -------
        list of BasePreProcessing derivative objects.
            List of preprocessing objects that are considered as "standard" for this specific dataset.
        """

        raise NotImplementedError("Not implemented get_default_preprocessing() method")
