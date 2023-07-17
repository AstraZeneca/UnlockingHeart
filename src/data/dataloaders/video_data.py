# Functionality related to echocardiogram datasets for DINO trainings
#
# Cite this paper for any use of the Echonet-Dynamic database
# David Ouyang , et al.
# "Video-based AI for beat-to-beat assessment of cardiac function", Nature (2020)
#
# Cite this paper for any use of the CAMUS database
# S. Leclerc, E. Smistad, J. Pedrosa, A. Ostvik, et al.
# "Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography",
# IEEE Transactions on Medical Imaging, vol. 38, no. 9, pp. 2198-2210, Sept. 2019.
# doi: 10.1109/TMI.2019.2900516

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from torchvision import transforms as pth_transforms
import yaml

# Import echo-dynamic package
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../lib/dynamic"))
import echonet

# Import own package
from src.data.dataloaders import *
from src.tools.preprocessing import DataAugmentationDINO

VIEWS_CLASSES = {'2CH': 0, '4CH': 1, 'PLAX': 2}


def none_or_int(value):
    if value == 'None':
        return None
    return int(value)


class EchoVideoDataset(echonet.datasets.Echo):
    def __init__(self, root, split, length=16, period=2, max_length=250, pad=None,
                 mean=None, std=None, resize_size=None, transform=None):
        # Call base class constructor
        super(EchoVideoDataset, self).__init__(root=root, split=split,
                                               mean=mean, std=std,
                                               length=length, period=period,
                                               max_length=max_length, pad=pad)
        # all videos in echonet-dynamic present 4ch view
        self.view = '4CH'
        if self.length is None:
            # take all frames in outcome whole data for given subset is stored
            num_frames_per_video = [f[-2] for f in self.outcome]
            self.num_frames = [i for index, i in enumerate(
                [*range(len(self.fnames))]) for _ in range(num_frames_per_video[index])]
        elif self.length > 1:
            # multiply the list of the videos in case the length > 1
            self.fnames = [i for i in self.fnames for _ in range(self.length)]

        # add class 4CH idx 0 as samples: list of (sample path, class_index) tuples
        self.samples = [(s, VIEWS_CLASSES[self.view]) for s in self.fnames]

        # Load normalization parameters
        with open(os.path.join(root, "normalization.json"), 'r') as file_handle:
            self.transform_parameters = json.load(file_handle)

        if mean == 0.:
            self.transform_parameters["mean"] = mean
        if std == 1.:
            self.transform_parameters["std"] = std
        if resize_size is not None:
            self.transform_parameters["resize_size"] = resize_size

        # Transform data
        if transform is None:
            self.transform = pth_transforms.Compose([
                    pth_transforms.ToTensor(),
                    pth_transforms.Resize(self.transform_parameters["resize_size"]),
                    pth_transforms.Normalize(self.transform_parameters["mean"],
                                             self.transform_parameters["std"])])
        else:
            self.transform = transform

    def __getitem__(self, index):
        # Find filename of video and proper frame index
        if self.length is None:
            video = os.path.join(self.root, "Videos", self.fnames[self.num_frames[index]])
            # Load video into numpy ndarray of shape F x H x W x C
            video = echonet.utils.loadvideo(video).astype(np.float32).transpose((1, 2, 3, 0))
            # Get required frame from video in PIL format
            idx = index - self.num_frames.index(self.num_frames[index])
            vio = video[idx, :, :, :]
        else:
            video = os.path.join(self.root, "Videos", self.fnames[index])
            # Load video into numpy ndarray of shape F x H x W x C
            video = echonet.utils.loadvideo(video).astype(np.float32).transpose((1, 2, 3, 0))
            # Get required frame from video in PIL format
            vio = video[np.random.choice(video.shape[0]), :, :, :]

        img = PIL.Image.fromarray(vio.astype('uint8'), 'RGB')

        # Transform data
        if self.transform is not None:
            img = self.transform(img)

        # return image and index of the frame
        return img, index

    def __len__(self):
        if self.length is None:
            return len(self.num_frames)

        return len(self.fnames)


class CAMUSVideoDataset(CAMUS):
    """
    Dataloader for the CAMUS dataset for dino backend.
    """

    def __init__(self, split, target_type=['NumberOfFrames', 'view'],
                 views=['2CH', '4CH'], gt=False, length=None, pad=None, **kwargs):
        """
        Constructor

        Parameters
        ----------
        split : str.
            Can be 'training', 'test', or 'all'.
        target_type : list of strings.
            Targets to acquire. Can be:
                'ESTrace' : End-systole segmentation.
                'EDTrace' : End-diastole segmentation.
                'ESFrame' : End-systole echocardiogram frame.
                'EDFrame' : End-diastole echocardiogram frame.
        views : list of strings.
            Which views to include. Can be: {'2CH' or '4CH'}
        """
        # Call base class constructor
        super(CAMUSVideoDataset, self).__init__(split=split, views=views, target_type=target_type,
                                                length=length, pad=pad, **kwargs)

        self.samples = []
        num_frames_per_video = []
        for index in range(len(self.patient_paths) * len(self.views)):
            # Get view
            view = self.views[index // len(self.patient_paths)]
            # Get path to target
            path = self.patient_paths[index % len(self.patient_paths)]
            with open(path / f'Info_{view}.cfg', 'r') as file:
                info = yaml.safe_load(file)
            num_frames_per_video.append(info["NbFrame"])
            self.samples.append((os.path.join(path, f"{path.name}_{view}_sequence.mhd"), VIEWS_CLASSES[view]))

        if self.sequence_length is None:
            self.num_frames = [i for index, i in enumerate(
                [*range(len(self.samples))]) for _ in range(num_frames_per_video[index])]
            self.samples = [i for index, i in enumerate(self.samples) for _ in range(num_frames_per_video[index])]
        elif self.sequence_length > 1:
            # multiply the list of the videos in case the length > 1
            self.samples = [i for i in self.samples for _ in range(self.sequence_length)]

    def __getitem__(self, index):
        # Find filename of video and proper frame index
        video_path = self.samples[index][0]
        # Load video into tensor of shape F x C x H x W
        video = CAMUS.open_mhd(video_path)
        # Get required frame from video
        if self.sequence_length is None:
            idx = index - self.num_frames.index(self.num_frames[index])
            vio = video[idx, :, :, :]
        else:
            # Get required frame from video
            vio = video[np.random.choice(video.shape[0]), :, :, :]

        img = torch.from_numpy(vio)

        # Preprocess image
        img = self.preprocess(img)

        # return image and index of the frame
        return img, index

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='EchoNet-Dynamics', type=str, help='Please specify dataset type tu use.')
    parser.add_argument('--data_directory', help="Path to directory of echonet-dynamic dataset.")
    parser.add_argument('--data_split', default="train", help="Which data split to use for normalization ('train', 'val', 'test', 'all').")
    parser.add_argument('--length', default=1, type=none_or_int, help='Number of frames from video to load.')
    parser.add_argument('--batch_size', default=4, type=int, help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--aug', action='store_true', help='Launch augumentations.')  # on/off flag
    args = parser.parse_args()

    if args.data == "EchoNet-Dynamics":
        print("Initializing the EchoNet-Dynamic dataset")
        # ============ preparing data ... ============
        if args.aug:
            transform = DataAugmentationDINO(
                (0.4, 1.),
                (0.05, 0.4),
                8,
                112
            )
        else:
            transforms = [pth_transforms.Resize((112, 112)),
                          pth_transforms.ToTensor()]
            transform = pth_transforms.Compose(transforms)
        dataset = EchoVideoDataset(root=args.data_directory, split="train",
                                   length=args.length, transform=transform)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True
        )

        plt.figure(figsize=(12, 12))
        for i in range(4):
            img, _ = next(iter(data_loader))
            if args.aug: img = img[0]
            for j in range(4):
                plt.subplot(4, 4, (i * 4) + j + 1)
                plt.imshow(img[j, ...].permute(1, 2, 0))
                plt.axis("off")
        plt.savefig("plot.png")
    else:
        dataset = CAMUSVideoDataset(split="test", target_type=[],
                                    length=args.length,
                                    resize_size=[112, 112])
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True
        )

        plt.figure(figsize=(12, 12))
        for i in range(4):
            img, _ = next(iter(data_loader))
            if args.aug: img = img[0]
            for j in range(4):
                plt.subplot(4, 4, (i * 4) + j + 1)
                plt.imshow(img[j, ...].permute(1, 2, 0))
                plt.axis("off")
        plt.savefig("plotcamus.png")
