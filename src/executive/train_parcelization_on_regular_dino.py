# Script for training a STEGO model with regular DINO backend using EchoNet-Dynamic

# Load packages
import os
import numpy as np
import argparse
from matplotlib import pyplot as plt
import csv
import torch

# Import echonet dynamic dataloader
from src.data.dataloaders.echonet_dynamic import EchoNet_Dynamic

# Import STEGO model
from src.models import DINO_backed
from src.tools import misc


class Train_parcelization:

    def __init__(self, num_classes, patch_size, resize_size, sequence_length, dino_model_path,
                 device="cpu", dataset='EchoNet_Dynamic', model_path=None, standardize_dino_output=False):
        print(num_classes)
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        # Define device for training
        self.device = torch.device("cpu")
        if device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("Using CUDA")
            else:
                raise Exception("GPU was not available!")
        else:
            print("Using CPU")

        if dataset == 'EchoNet_Dynamic':
            # Load EchoNet-Dynamics dataset
            self.dataset = EchoNet_Dynamic(
                split="train", target_type=[], device=self.device, length=self.sequence_length,
                resize_size=[resize_size, resize_size])
        else:
            raise Exception(f"Dataset {dataset} is not supported!")

        # Load DINOBased model
        print("")
        print("Load STEGO-DINOBased model")
        dino_model = DINO_backed.load_dino(dino_model_path, device=device,
                                           patch_size=patch_size, img_size=resize_size)

        self.model = DINO_backed.StegoLikeModel_3_layer_marginal(
            dino_model=dino_model, num_classes=num_classes,
            device=device, reshaper=(-1, 3, resize_size, resize_size),
            b_param={"self": 0.3, "similarity": 0.3, "contrastive": 0.3},
            loss_weighting={"self": 1.0, "similarity": 0.5, "contrastive": 1.0})

        # See if file exists
        if model_path is not None:
            print(f"Loading initial model from: {model_path}")
            # Load model
            self.model.load_front_end(model_path, device=self.device)
        # If model has not been pre-trained
        elif standardize_dino_output:
            print("")
            print("Compute loc- and scale-parameter after DINO-backbone!")
            # Get dataloader
            data_loader = self.dataset.get_data_loader(batch_size=1, shuffle=True)
            # Compute location and scaling vectors
            self.model.compute_loc_and_scale(data_loader, device=self.device)

    def train(self, model_path, num_epochs=1, batch_size=1, learning_rate=5e-3, saveckp_freq=10):
        # If directory of output path does not exist
        if not os.path.isdir(os.path.dirname(model_path)):
            # Create it
            os.mkdir(os.path.dirname(model_path))

        # Get dataloader
        data_loader = self.dataset.get_data_loader(batch_size=batch_size, shuffle=True)

        # Train STEGO
        print("")
        print("Perform training")
        losses = np.zeros(0)

        def epoch_callback(x):
            # Define callback function for saving model after every epoch
            # Add current loss
            nonlocal losses
            losses = np.append(losses, x)

            # Save model
            self.model.save_front_end(model_path)

            # Generate .png file of the training
            fig = plt.figure(figsize=(9, 5), dpi=100)
            plt.plot(losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            fig.savefig(model_path + ('.png'))
            plt.close()

            # Generate .csv file of the training performance
            with open(model_path + ('.csv'), "w") as f:
                write = csv.writer(f)
                write.writerow(["Epoch", "Loss"])
                write.writerows([[it, x] for it, x in enumerate(losses)])

        # Start training
        _ = self.model.fit(
            data_loader, num_epochs=num_epochs, lr=learning_rate, hard_classification=False,
            verbose=True, epoch_callback=epoch_callback, B=batch_size, Fr=self.sequence_length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', required=True, help="Number of classes to train.", type=int)
    parser.add_argument('--patch_size', required=True, help="Patch size of initial convolution in DINO pipeline.", type=int)
    parser.add_argument('--dino_model_path', required=True, help="Path to DINO model to base STEGO on.")
    parser.add_argument('--device', required=True, help="Which device to compute with ('cpu' or 'cuda').")
    parser.add_argument('--output_model_path', required=True, help="Path to save output at.")
    parser.add_argument('--num_epochs', required=True, help="Number of epochs to train for.", type=int)
    parser.add_argument('--batch_size', required=True, help="Number of sequences in one batch.", type=int)
    parser.add_argument('--sequence_length', required=False, default=16, help="Number of frames in one sequence.", type=int)
    parser.add_argument('--resize_size', required=True, help="Size to resize images to.", type=int)
    parser.add_argument('--standardize_dino_output', action='store_true', help="If dino output should be standardized")
    parser.add_argument('--initial_model_path', help="Path initial model weights.")
    parser.add_argument('--learning_rate', type=float, default=5e-3, help="Which learning rate to use for Adam optimizer.")
    parser.add_argument('--seed', default=1, type=int, help='Set seed.')

    args = parser.parse_args()

    # Set seed
    misc.set_seed(args.seed)

    print("Initializing training of STEGO_like on regular DINO backend trained on EchoNet-Dynamic!")

    # Initialize model
    model = Train_parcelization(num_classes=args.num_classes, patch_size=args.patch_size,
                                resize_size=args.resize_size, sequence_length=args.sequence_length,
                                dino_model_path=args.dino_model_path, device=args.device,
                                model_path=args.initial_model_path, standardize_dino_output=args.standardize_dino_output)
    # Train model
    model.train(args.output_model_path, num_epochs=args.num_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
