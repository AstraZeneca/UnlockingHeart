# Functionality related to the CAMUS dataset
#
# Cite this paper for any use of the CAMUS database
# S. Leclerc, E. Smistad, J. Pedrosa, A. Ostvik, et al.
# "Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography",
# IEEE Transactions on Medical Imaging, vol. 38, no. 9, pp. 2198-2210, Sept. 2019.
# doi: 10.1109/TMI.2019.2900516

import os
from pathlib import Path
import json
import yaml
import argparse
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

import torch

from src.data.dataloader import BaseDataset
from src.tools import misc, preprocessing


class CAMUS(BaseDataset):
    """
    Dataloader for the CAMUS dataset.
    """

    CAMUS_PATH = os.path.join(os.path.dirname(__file__), "../../../data/CAMUS/")

    def __init__(self, split="all", target_type=[], views=[
                 '2CH', '4CH'], gt=False, length=16, pad=None, **kwargs):
        """
        Constructor

        Parameters
        ----------
        split : str.
            Can be 'train', 'test', or 'all'.
        target_type : list of strings.
            Targets to acquire. Can be:
                'ESILVTrace' : End-systole endo segmentation.
                'EDILVTrace' : End-diastole endo segmentation.
                'ESOLVTrace' : End-systole epi segmentation.
                'EDOLVTrace' : End-diastole epi segmentation.
                'ESLATrace' : End-systole left atrium segmentation.
                'EDLATrace' : End-diastole left atrium segmentation.
                'ESFrame' : End-systole echocardiogram frame.
                'EDFrame' : End-diastole echocardiogram frame.
                'ESV' : End-systole left-ventricular volume.
                'EDV' : End-diastole left-ventricular volume.
                'ESIndex' : The frame index corresponding to the end-systole frame
                'EDIndex' : The frame index corresponding to the end-diastole frame
                'EF' : Left ventricular ejection fraction
                'NumberOfFrames' : Number of frames in sequence
                'sex' : Sex of patient
                'age' : Age of patient
                'quality' : Quality of echocardiogram acquisition
                'view' : Which view class
                'path' : The path to this specific patients files

        views : list of strings.
            Which views to include. Can be: {'2CH' or '4CH'}

        """
        # Initiate parent
        super(CAMUS, self).__init__(**kwargs)

        self.pad = pad
        self.sequence_length = length

        for view in views:
            if view not in ['2CH', '4CH']:
                raise Exception(f"View {view} not supported!")
        self.views = views

        # Set if load manualy segmented frames and transform them
        self.gt = gt
        if self.gt:
            # the order: target_type["LargeTrace","SmallTrace", "LargeFrame", "SmallFrame"]
            if "EDFrame" in target_type:
                target_type.remove("EDFrame")
            if "ESFrame" in target_type:
                target_type.remove("ESFrame")
            target_type.append("ESFrame")  # set this as -2 element
            target_type.append("EDFrame")  # set this as -1 element

        self.target_type = target_type
        print(target_type)

        # Get home path
        home_path = Path(CAMUS.CAMUS_PATH)

        if split not in ['train', 'test', 'all']:
            raise Exception(f"Split {split} not supported!")

        # Get patient paths
        self.patient_paths = []
        if (split == 'train') or (split == 'all'):
            self.patient_paths = [*self.patient_paths,
                                  *list(home_path.glob("training/patient*"))]
        if (split == 'test') or (split == 'all'):
            self.patient_paths = [*self.patient_paths,
                                  *list(home_path.glob("testing/patient*"))]

    @staticmethod
    def open_mhd(path, resize_transform=None):
        """
        Open mhd-file and introduce (artificial) color channel
        """
        try:
            img = sitk.GetArrayFromImage(sitk.ReadImage(str(path), sitk.sitkFloat32))
        except Exception as e:
            print(path)
            raise e
        if resize_transform is not None:
            img = torch.from_numpy(img)
            img = resize_transform(img)
            img = img.numpy()

        img = np.stack((img, ) * 3, axis=-1)
        img = img.transpose((0, 3, 1, 2))
        return img

    @staticmethod
    def get_labels(path, view, target_type, resize_transform=None):
        """
        Get sought after labels for given patient
        """
        if view == '2CH':
            with open(path / 'Info_2CH.cfg', 'r') as file:
                info = yaml.safe_load(file)
        elif view == '4CH':
            with open(path / 'Info_4CH.cfg', 'r') as file:
                info = yaml.safe_load(file)
        else:
            raise Exception(f"View {view} not supported!")

        # Go through all target labels
        labels = []
        for label in target_type:
            if label == 'ESILVTrace':
                img = CAMUS.open_mhd(path / f"{path.name}_{view}_ES_gt.mhd", resize_transform)[0, 0, :, :]
                labels.append(img == 1)
            elif label == 'EDILVTrace':
                img = CAMUS.open_mhd(path / f"{path.name}_{view}_ED_gt.mhd", resize_transform)[0, 0, :, :]
                labels.append(img == 1)
            elif label == 'ESOLVTrace':
                img = CAMUS.open_mhd(path / f"{path.name}_{view}_ES_gt.mhd", resize_transform)[0, 0, :, :]
                labels.append((img == 1) + (img == 2))
            elif label == 'EDOLVTrace':
                img = CAMUS.open_mhd(path / f"{path.name}_{view}_ED_gt.mhd", resize_transform)[0, 0, :, :]
                labels.append((img == 1) + (img == 2))
            elif label == 'ESLATrace':
                img = CAMUS.open_mhd(path / f"{path.name}_{view}_ES_gt.mhd", resize_transform)[0, 0, :, :]
                labels.append(img == 3)
            elif label == 'EDLATrace':
                img = CAMUS.open_mhd(path / f"{path.name}_{view}_ED_gt.mhd", resize_transform)[0, 0, :, :]
                labels.append(img == 3)
            elif label == 'ESFrame':
                img = CAMUS.open_mhd(path / f"{path.name}_{view}_ES.mhd", resize_transform)[0, :, :, :]
                labels.append(img)
            elif label == 'EDFrame':
                img = CAMUS.open_mhd(path / f"{path.name}_{view}_ED.mhd", resize_transform)[0, :, :, :]
                labels.append(img)
            elif label == 'EDIndex':
                labels.append(info["ED"])
            elif label == 'ESIndex':
                labels.append(info["ES"])
            elif label == 'NumberOfFrames':
                labels.append(info["NbFrame"])
            elif label == 'sex':
                labels.append(info["Sex"])
            elif label == 'age':
                labels.append(info["Age"])
            elif label == 'quality':
                labels.append(info["ImageQuality"])
            elif label == 'EDV':
                labels.append(info["LVedv"])
            elif label == 'ESV':
                labels.append(info["LVesv"])
            elif label == 'EF':
                labels.append(info["LVef"])
            elif label == 'view':
                labels.append(view)
            elif label == 'path':
                labels.append(str(path))
            else:
                raise Exception(f"Target type: {label} not present in {path}!")

        return labels

    def __len__(self):
        return len(self.patient_paths) * len(self.views)

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

        # Get view
        view = self.views[index // len(self.patient_paths)]
        # Get path to target
        path = self.patient_paths[index % len(self.patient_paths)]

        # Get labels
        labels = CAMUS.get_labels(path, view, self.target_type, resize_transform=self.resize_transform)

        # Handle if output should be whole sequence or just end-systole and end-distole frames
        if self.gt:
            # Make into torch tensor stacked ESFrame and EDFrame
            img = torch.from_numpy(np.stack((labels[-2], labels[-1])))
        else:
            # Make into torch tensor and order the dimensions as Frames X Channels X Height X Width
            img = CAMUS.open_mhd(path / f"{path.name}_{view}_sequence.mhd")

            # Get length of sequence
            sequence_length = img.shape[0]
            # If below length, pad with initial frame
            if sequence_length < self.sequence_length:
                img = [img[0:1, :, :, :] for it in range(int(self.sequence_length - sequence_length + 1))] + img
                img = np.concatenate(img, axis=0)
                sequence_length = img.shape[0]
            # Get start frame
            start_frame = np.random.randint(low=0, high=(sequence_length - self.sequence_length + 1),
                                            size=None, dtype=int)
            # Take subsequence
            img = img[start_frame:(start_frame + self.sequence_length)]
            img = torch.from_numpy(img)

        # Preprocess image
        img = self.preprocess(img)

        # return image and labels
        return img, labels

    @staticmethod
    def get_normalization():
        path = os.path.join(CAMUS.CAMUS_PATH, 'normalization.json')
        with open(path, 'r') as fd:
            normalization = json.load(fd)
        return normalization

    @staticmethod
    def get_default_preprocessing():
        normalization = CAMUS.get_normalization()
        mean = 0.5 * normalization['2CH']['mean'] + \
            0.5 * normalization['4CH']['mean']
        std = np.sqrt(0.5 * normalization['2CH']['std']**2 + 0.5 * normalization['4CH']['std']**2)
        resize_size = normalization['resize_size']

        preprocessing_list = [
            preprocessing.Normalize(
                mean=mean,
                std=std),
            preprocessing.Resize(resize_size)]

        return preprocessing_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_directory',
        required=True,
        help="Path to directory of CAMUS dataset.")
    parser.add_argument(
        '--data_split',
        help="Which data split to use for normalization ('train', 'test', 'all').",
        default="train")
    parser.add_argument(
        '--resize_size',
        help="Set resize size of images. If left blank, no resize will take place")
    parser.add_argument(
        '--seed',
        type=int,
        help="Seed for random number generator. Defaults to 0.",
        default=0)
    args = parser.parse_args()

    # Seed RNGs
    misc.set_seed(args.seed)

    save_dict = {'resize_size': [112]}
    if args.resize_size is not None:
        if isinstance(args.resize_size, list):
            save_dict['resize_size'] = [int(x) for x in args.resize_size]
        else:
            save_dict['resize_size'] = [int(args.resize_size)]

    print("Initializing the CAMUS dataset 2 chamber views")

    dataset = CAMUS(
        split=args.data_split,
        target_type=[],
        views=["2CH"],
        preprocessing=[])
    img, *_ = dataset.__getitem__(0)
    print(f"Image shape: {img.shape}")

    # Get mean and standard deviation
    mean_2CH = 0
    var_2CH = 0
    for iteration in tqdm(
            range(int(np.round(len(dataset.patient_paths) * 1)))):
        img, *_ = dataset.__getitem__(iteration)
        mean_2CH = mean_2CH * iteration / \
            (iteration + 1) + np.nanmean(img) / (iteration + 1)
        var_2CH = var_2CH * iteration / \
            (iteration + 1) + np.nanvar(img) / (iteration + 1)
    save_dict['2CH'] = {
        "mean": mean_2CH.tolist(),
        "std": np.sqrt(var_2CH).tolist()}
    print(f"Mean value: {mean_2CH}")
    print(f"Standard deviation value: {np.sqrt(var_2CH)}")
    print("")

    print("Initializing the CAMUS dataset 4 chamber views")

    dataset = CAMUS(
        split=args.data_split,
        target_type=[],
        views=["4CH"],
        preprocessing=[])
    img, *_ = dataset.__getitem__(0)
    print(f"Image shape: {img.shape}")

    # Get mean and standard deviation
    mean_4CH = 0
    var_4CH = 0
    for iteration in tqdm(
            range(int(np.round(len(dataset.patient_paths) * 1)))):
        img, *_ = dataset.__getitem__(iteration)
        mean_4CH = mean_4CH * iteration / \
            (iteration + 1) + np.nanmean(img) / (iteration + 1)
        var_4CH = var_4CH * iteration / \
            (iteration + 1) + np.nanvar(img) / (iteration + 1)
    save_dict['4CH'] = {
        "mean": mean_4CH.tolist(),
        "std": np.sqrt(var_4CH).tolist()}
    print(f"Mean value: {mean_4CH}")
    print(f"Standard deviation value: {np.sqrt(var_4CH)}")

    # Save values
    with open(os.path.join(args.data_directory, "normalization.json"), 'w') as file_handle:
        json.dump(save_dict, file_handle, indent=4)
