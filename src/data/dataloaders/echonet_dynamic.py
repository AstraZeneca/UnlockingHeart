# Functionality related to the Echonet-Dynamic dataset
#
# Cite this paper for any use of the Echonet-Dynamic database
# David Ouyang , et al.
# "Video-based AI for beat-to-beat assessment of cardiac function", Nature (2020)

import os
import sys
import json
import argparse
import numpy as np

import torch

from src.data.dataloader import BaseDataset
from src.tools import misc, preprocessing

# Import echo-dynamic package
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../lib/dynamic"))
import echonet


class EchoNet_Dynamic(BaseDataset):
    """
    Dataloader for the EchoNet-Dynamic dataset.
    """

    # The path to dataset
    ECHONET_PATH = os.path.join(os.path.dirname(__file__), "../../../data/EchoNet-Dynamic")

    def __init__(self, split="all", target_type=[], gt=False, length=16,
                 period=2, max_length=250, pad=None, **kwargs):
        """
        Constructor

        Parameters
        ----------
        split : str.
            String of which subset of dataset to use, viz., "train", "val", "test", "all".
        """

        # Initiate parent
        super(EchoNet_Dynamic, self).__init__(**kwargs)

        # Set if load manualy segmented frames and transform them
        self.gt = gt
        if self.gt:
            if "ESFrame" in target_type:
                target_type.remove("ESFrame")
            if "EDFrame" in target_type:
                target_type.remove("EDFrame")
            target_type.append("ESFrame")  # set this as -2 element
            target_type.append("EDFrame")  # set this as -1 element

        # Rename target types
        rename_dict = {
            "ESIndex": "LargeIndex", "EDIndex": "SmallIndex",
            "ESTrace": "LargeTrace", "EDTrace": "SmallTrace",
            "ESFrame": "LargeFrame", "EDFrame": "SmallFrame",
            "path": "Filename"}

        for orig, replac in rename_dict.items():
            target_type = [word.replace(orig, replac) for word in target_type]

        self.dataset = echonet.datasets.Echo(root=EchoNet_Dynamic.ECHONET_PATH, split=split, target_type=target_type,
                                             length=length, period=period, max_length=max_length, pad=pad)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Function to acquire item from dataset

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

        # Get data from echonet code
        img, label = self.dataset.__getitem__(index)
        # Handle if output should be whole sequence or just end-systole and end-distole frames
        if self.gt:
            # Make into torch tensor stacked ESFrame and EDFrame
            img = torch.from_numpy(np.stack((label[-2], label[-1])))
        else:
            # Make into torch tensor and order the dimensions as Frames X Channels X Height X Width
            img = torch.from_numpy(img[:, :, :, :].transpose((1, 0, 2, 3)))

        # Preprocess image
        img = self.preprocess(img)

        # return image and labels
        return img, label

    @staticmethod
    def get_normalization():
        path = os.path.join(EchoNet_Dynamic.ECHONET_PATH, 'normalization.json')
        with open(path, 'r') as fd:
            normalization = json.load(fd)
        return normalization

    @staticmethod
    def get_default_preprocessing():
        normalization = EchoNet_Dynamic.get_normalization()
        preprocessing_list = [preprocessing.Normalize(mean=normalization["mean"], std=normalization["std"])]
        return preprocessing_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_directory',
        help="Path to directory of echonet-dynamic dataset.")
    parser.add_argument(
        '--data_split',
        help="Which data split to use for normalization ('train', 'val', 'test', 'all').",
        default="train")
    parser.add_argument(
        '--resize_size',
        help="Set resize size of images.")
    parser.add_argument(
        '--seed',
        type=int,
        help="Seed for random number generator. Defaults to 0.",
        default=0)
    args = parser.parse_args()

    # Seed RNGs
    misc.set_seed(args.seed)

    print("Initializing the EchoNet-Dynamic dataset")

    dataset = echonet.datasets.Echo(root=args.data_directory)
    img, *_ = dataset.__getitem__(0)
    print(f"Image shape: {img.shape}")
    if args.resize_size is None:
        args.resize_size = img.shape[-2:]
    else:
        args.resize_size = int(args.resize_size)
    print(f"Resize size: {args.resize_size}")

    # Get mean and standard deviation of pixel values in echonet dynamics training set
    echo_mean, echo_std = echonet.utils.get_mean_and_std(
        echonet.datasets.Echo(root=args.data_directory, split=args.data_split), samples=None)
    print(f"Mean value: {echo_mean}")
    print(f"Standard deviation value: {echo_std}")

    # Save values
    with open(os.path.join(args.data_directory, "normalization.json"), 'w') as file_handle:
        json.dump({"mean": echo_mean.tolist(), "std": echo_std.tolist(),
                   "resize_size": np.array(args.resize_size).tolist()},
                  file_handle, indent=4)
