# This file holds code related to a STEGO model with a DINO backend and an arbitrary frontend.
# The DINO model is assumed to be a torch module.

# Load packages
import os
import sys
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
import torch
from torchvision import transforms as pth_transforms

from ..tools.nn_blocks import MarginalBlock

# Import dino package
sys.path.append(os.path.join(os.path.dirname(__file__), "../../lib/dino/"))
import vision_transformer as vit
import utils as dino_utils


# for now it states here
def model_predict(model, input_data, num_classes, resize_size, patch_size):
    # Get dimensions of images
    B, Fr, C, H, W = input_data.size()

    # Get segmentation
    soft_parcelization = model(input_data, hard_classification=False,
                               use_activation=False)[:, 1:, :].transpose(-1, -2).reshape(
        (-1, num_classes, resize_size // patch_size, resize_size // patch_size))
    # Transform image to larger size
    soft_parcelization = pth_transforms.Resize(size=resize_size,
                                               interpolation=pth_transforms.InterpolationMode.BILINEAR)(
        soft_parcelization).reshape((B, Fr, num_classes, resize_size, resize_size)).cpu()
    # Hard segmentation larger
    hard_parcelization_larger = torch.argmax(soft_parcelization, dim=2, keepdim=False).numpy().astype(np.float32)
    return soft_parcelization, hard_parcelization_larger


def load_dino(model_path, device, patch_size, img_size):
    """
    Load dino-model from file into dino_model object.
    """
    # See if file exists
    if model_path is not None:
        if os.path.exists(model_path):
            # Create visual transformer
            dino_model = vit.vit_small(patch_size=patch_size, img_size=[img_size])
            # Load weights
            dino_utils.load_pretrained_weights(dino_model, model_path, "teacher", None, None)
            # Freeze backbone
            dino_model.requires_grad = False
            return dino_model
        else:
            raise Exception("No initial training file exists!")
    else:
        raise Exception("No initial training file exists!")

    return


def get_class(model_path):
    '''
    Acquire class of DINO-backed model
    '''

    parameter_checkpoint = torch.load(model_path, map_location="cpu")
    class_name = parameter_checkpoint["class_name"]

    return eval(class_name)


class DINOBasedModel(torch.nn.Module, ABC):
    '''
    Class for representing model based on a DINO backend
    '''

    def __init__(self, dino_model, front_end, loc=None, scale=None, device='cpu', reshaper=None):
        """
        Constructor
        :param dino_model: A dino model object.
        :param sequential: A 'torch.nn.Sequential' object to use as front-end on top of dino architecture.
        """

        # Initiate parent
        super().__init__()

        self.reshaper = reshaper

        # Get embedding dimensions
        self.embed_dim = dino_model.embed_dim
        # set dino model
        self.dino_model = dino_model.to(device)
        # Set internal projection layers
        self.proj = front_end
        if front_end is not None:
            self.proj = self.proj.to(device)

        self.loc = loc
        if self.loc is None:
            self.loc = torch.zeros((self.embed_dim), device=device, requires_grad=False)

        self.scale = scale
        if self.scale is None:
            self.scale = torch.ones((self.embed_dim), device=device, requires_grad=False)

        # Set model to evaluation mode
        self.eval()
        # Make sure that gradients are not required
        self.requires_grad_(False)

    def dino_forward(self, x, n=1):
        '''
        Compute dino features (from n:th hidden layer counted from the last)
        '''
        # run through dino model
        return self.dino_model.get_intermediate_layers(x, n=n)[-1]

    def dino_attention(self, x, n=None):
        '''
        Compute dino attention (from n:th hidden layer counted from the last)
        '''

        # Handle n-value
        if (n is None):
            n = 0
        if n > len(self.dino_model.blocks):
            n = len(self.dino_model.blocks) - 1

        x = self.dino_model.prepare_tokens(x)
        for i, blk in enumerate(self.dino_model.blocks):
            if (i < len(self.dino_model.blocks) - 1 - n):
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def compute_loc_and_scale(self, data_loader, device=None, num_iterations=None):
        '''
        Compute location- and scaling-vectors for dino output given current data loader.
        '''

        # Acquire scale and location parameters
        self.loc = torch.zeros((self.embed_dim), requires_grad=True)
        self.scale = torch.zeros((self.embed_dim), requires_grad=True)

        if device is not None:
            self.loc = self.loc.to(device)
            self.scale = self.scale.to(device)

        def _update_scale_and_loc(x):
            # Compute last-1 layer of dino model
            x = self.dino_forward(x, n=1).detach()

            loc = torch.mean(x, dim=(0, 1))
            scale = torch.std(x, dim=(0, 1), unbiased=True)

            return (loc, scale)

        for loader_iter, imgs_and_targets in enumerate(tqdm(data_loader)):

            if num_iterations is not None:
                if loader_iter > num_iterations:
                    break

            x = imgs_and_targets[0]

            if self.reshaper is not None:
                x = x.reshape(self.reshaper)

            loc, scale = _update_scale_and_loc(x.detach())

            self.loc = (self.loc * loader_iter + loc) / (loader_iter + 1)
            self.scale = (self.scale * loader_iter + scale) / (loader_iter + 1)

    @abstractmethod
    def load_front_end(self, model_path, device):
        '''
        Abstract method for loading fron-end model
        '''
        raise NotImplementedError("Not implemented load_front_end() method")

    def _load_front_end_file(self, model_path, device):
        '''
        Load front-end file and return data structure.
        '''
        # See if file exists
        if model_path is not None:
            if os.path.exists(model_path):
                # Load trained parameters
                parameter_checkpoint = torch.load(model_path, map_location=device)

                self.loc = torch.zeros((self.embed_dim), device=device, requires_grad=False)
                if "loc" in parameter_checkpoint.keys():
                    self.loc = parameter_checkpoint["loc"]

                self.scale = torch.ones((self.embed_dim), device=device, requires_grad=False)
                if "scale" in parameter_checkpoint.keys():
                    self.scale = parameter_checkpoint["scale"]

                # Assert dimensions
                if self.loc.shape[0] != self.embed_dim:
                    raise Exception(
                        f"Embedded dimensionality was wrong. Expected {self.embed_dim} but found {self.model.loc.shape[0]} in loc-parameter.")
                if self.scale.shape[0] != self.embed_dim:
                    raise Exception(
                        f"Embedded dimensionality was wrong. Expected {self.embed_dim} but found {self.model.scale.shape[0]} in scale-parameter.")

                # Return further parameters
                return parameter_checkpoint
            else:
                raise Exception("No initial training file exists!")

    def _save_front_end_file(self, model_path, objects_dict):
        '''
        Save front-end file.
        '''
        objects_dict["class_name"] = self.__class__.__name__
        objects_dict["loc"] = self.loc
        objects_dict["scale"] = self.scale
        # Save dictionary
        torch.save(objects_dict, model_path)

    @abstractmethod
    def save_front_end(self, model_path):
        '''
        Abstract function for saving model
        '''
        raise NotImplementedError("Not implemented save_front_end() method")

    def front_end_forward(self, x, *args, **kwargs):
        '''
        How to forward through front end
        '''
        # run through front end
        x = self.proj.forward(x, *args, **kwargs)
        return x

    def forward(self, x, dino_n=1, bypass_front_end=False, **kwargs):
        '''
        Send input through dino and then front end
        '''
        if self.reshaper is not None:
            x = x.reshape(self.reshaper)
        # run through dino model
        x = self.dino_forward(x, n=dino_n)

        # Acquire dimensions
        B, N, C = x.size()
        # Normalize
        x = x - self.loc.reshape((1, 1, C))
        x = x / self.scale.reshape((1, 1, C))

        if not bypass_front_end:
            # run through front end
            x = self.front_end_forward(x, **kwargs)

        return x


class DINOBasedHeadLearner(DINOBasedModel):
    '''
    A learner subclass of DINOBased
    '''
    @abstractmethod
    def _loss_fn():
        '''
        Abstract method for loss function
        '''
        raise NotImplementedError("Not implemented _loss_fn() method")

    def fit(self, dataloader, num_epochs=100, retain_graph=True, num_iterations=None, lr=5e-3, epoch_callback=None, **kwargs):
        '''
        Method for fitting using Adam optimization
        '''
        # Set model as under training
        self.train()

        # Train all parameters of this module and submodules
        self.requires_grad_(True)
        # Freeze dino_model
        self.dino_model.requires_grad_(False)

        # choose Adam optimizer
        optimizer = torch.optim.Adam(list(self.parameters()), lr=lr)

        # Loop through epochs of training
        losses = np.zeros(num_epochs)
        for iter_epoch in range(num_epochs):
            avg_loss = 0.0

            for loader_iter, loader in enumerate(tqdm(dataloader)):

                if num_iterations is not None:
                    if loader_iter > num_iterations:
                        break

                cur_sample, cur_classes = loader

                # erase old gradient
                optimizer.zero_grad()

                # Compute loss of current state
                loss = self._loss_fn(cur_sample, **kwargs)
                avg_loss = (avg_loss * loader_iter + loss.detach().cpu()) / (loader_iter + 1)
                # Compute gradients of current state through backpropogation
                loss.backward(retain_graph=retain_graph)
                # Step with optimizer
                optimizer.step()

            losses[iter_epoch] = avg_loss

            if epoch_callback is not None:
                epoch_callback(avg_loss)

            if "verbose" in kwargs.keys():
                if kwargs["verbose"]:
                    print(f"Finished epoch: {iter_epoch} with loss: {avg_loss}")

        # Set model to evaluation mode
        self.eval()
        # Make sure that gradients are not required
        self.requires_grad_(False)

        return losses


class StegoLikeModel(DINOBasedHeadLearner):
    '''
    Parent class for STEGO_like models
    '''
    def __init__(self, dino_model, front_end, num_classes,
                 b_param={"self": 0.3, "similarity": 0.3, "contrastive": 0.3},
                 loss_weighting={"self": 1.0, "similarity": 0.5, "contrastive": 1.0},
                 **kwargs):

        super(StegoLikeModel, self).__init__(dino_model=dino_model, front_end=front_end, **kwargs)

        self.b_param = b_param
        self.loss_weighting = loss_weighting
        self.num_classes = num_classes

    def _feature_correspondence_matrix(self, f, g):
        # Compute the feature correspondence matrix (from the STEGO paper) between two tensors

        assert (f.size()[-2:] == g.size()[-2:])

        # Matrix multiplication between last layers of the two images using the DINO architecture
        mult = f @ g.transpose(-1, -2)
        # Normalize matrix with norm of each images last layers
        f_abs = torch.norm(f, dim=-1, p=2, keepdim=True)
        g_abs = torch.norm(g, dim=-1, p=2, keepdim=True)

        return mult / (f_abs * g_abs.transpose(-1, -2))

    def dino_feature_correspondence(self, img1, img2, n=1):
        # Compute the feature correspondence matrix between last layers of two images using the internal DINO architecture

        # Compute last-1 layer of dino model
        f = self.dino_forward(img1, n=n)
        g = self.dino_forward(img2, n=n)

        # Compute feature correspondence matrix
        return self._feature_correspondence_matrix(f, g)

    def segmentation_feature_correspondence(self, img1, img2):
        # Compute the feature correspondence matrix between the segmentation output of two images

        # Compute last-1 layer of dino model
        f = self.forward(img1)
        g = self.forward(img2)

        # Compute feature correspondence matrix
        return self._feature_correspondence_matrix(f, g)

    def dino_and_segmentation_feature_correspondence(self, img1, img2, n=1):
        # Compute the feature correspondence matrix for both features and segmentation

        # Compute feature layer of dino model
        f = self.dino_forward(img1, n)
        g = self.dino_forward(img2, n)
        # Compute feature correspondence
        F = self._feature_correspondence_matrix(f, g)

        # Compute segmentations
        sf = self.proj(f)
        sg = self.proj(g)
        # Copmute feature correspondence matrix
        S = self._feature_correspondence_matrix(sf, sg)

        return {"F": F, "S": S}

    def _loss_fn(self, x, B=None, Fr=None, **kwargs):
        """
        Loss function
        The loss function as given in the STEGO paper.
        :param x: The input signal
        :param b_self: The b-parameter (negative pressure) value for the self-similarity.
        :param b_similar: The b-parameter (negative pressure) value for the similarity.
        :param b_contrastive: The b-parameter (negative pressure) value for contrastive.
        :param loss_weighting_self: The weighting for the self-similarity loss.
        :param loss_weighting_similar: The weighting for the similarity loss.
        :param loss_weighting_contrastive: The weighting for the contrastive loss.
        :returns: loss-value
        """
        L = 0
        similarity_histogram = torch.zeros((4, int(np.ceil(2.0 / 0.05))))
        similarity_histogram[0, :] = 0.5 * (torch.arange(-1, 1.001, step=0.05)[:-1] + torch.arange(-1, 1.001, step=0.05)[1:])

        # Compute through backbone, but skip head
        f = self.forward(x, n=1, bypass_front_end=True)
        # Compute segmentation
        s = self.front_end_forward(f)

        if B is None:
            raise Exception("B was not given!")
        if Fr is None:
            raise Exception("Fr was not given!")
        b_self = self.b_param["self"]
        b_contrastive = self.b_param["contrastive"]
        loss_weighting_self = self.loss_weighting["self"]
        loss_weighting_similar = self.loss_weighting["similarity"]
        loss_weighting_contrastive = self.loss_weighting["contrastive"]

        # Loop through sequences in batch
        for iter_sequence in np.arange(stop=B):

            # Look at self
            if (loss_weighting_self > 0):
                # Compute feature correspondence between frames and themselves in current sequence
                F = self._feature_correspondence_matrix(
                    f[(Fr * iter_sequence):(Fr * (iter_sequence + 1))],
                    f[(Fr * iter_sequence):(Fr * (iter_sequence + 1))])
                # Compute segmentation correspondence between frames and themselves in current sequence
                S = self._feature_correspondence_matrix(
                    s[(Fr * iter_sequence):(Fr * (iter_sequence + 1))],
                    s[(Fr * iter_sequence):(Fr * (iter_sequence + 1))])

                # Compute spatial centering feature correspondence
                F = F - torch.mean(F, dim=-1, keepdim=True)
                # Compute, normalized, feature correspondence similarity
                similarity_histogram[1, :] =+ torch.histogram(
                    torch.flatten((F.detach() * S.detach()).cpu()),
                    bins=torch.arange(-1, 1.001, step=0.05))[0]
                # Update loss function with contribution from comparing current sequence with itself
                L += - loss_weighting_self * torch.mean((F - b_self) * torch.clamp(S, min=0))

            # Now look at similarity between one image and all other in sequence
            if (loss_weighting_similar > 0):
                chosen_frame = np.random.randint(low=0, high=Fr)

                # Compute feature correspondence between chosen frame and all other frames in current sequence
                F = self._feature_correspondence_matrix(
                    f[(Fr * iter_sequence + chosen_frame):(Fr * iter_sequence + chosen_frame + 1)],
                    f[(Fr * iter_sequence):(Fr * (iter_sequence + 1))])
                # Compute segmentation correspondence between chosen frame and all other frames in current sequence
                S = self._feature_correspondence_matrix(
                    s[(Fr * iter_sequence + chosen_frame):(Fr * iter_sequence + chosen_frame + 1)],
                    s[(Fr * iter_sequence):(Fr * (iter_sequence + 1))])

                # Compute spatial centering feature correspondence
                F = F - torch.mean(F, dim=-1, keepdim=True)
                # Compute, normalized, feature correspondence similarity
                similarity_histogram[2, :] =+ torch.histogram(
                    torch.flatten((F.detach() * S.detach()).cpu()),
                    bins=torch.arange(-1, 1.001, step=0.05))[0]
                # Update loss function with contribution from comparing one frame with all other in same sequence
                L += - loss_weighting_similar * torch.mean((F - b_self) * torch.clamp(S, min=0))

            # Now look at contrast with other sequence
            if (loss_weighting_contrastive > 0):

                # Get random other sequence
                other_sequence = np.random.choice(list(set(np.arange(stop=B)) - set([iter_sequence])), size=1)[0]

                # Compute feature correspondence between frames and themselves in current sequence to other sequence
                F = self._feature_correspondence_matrix(f[(Fr * iter_sequence):(Fr * (iter_sequence + 1))],
                                                        f[(Fr * other_sequence):(Fr * (other_sequence + 1))])
                # Compute segmentation correspondence between frames and themselves in current sequence to other sequence
                S = self._feature_correspondence_matrix(
                    s[(Fr * iter_sequence):(Fr * (iter_sequence + 1))],
                    s[(Fr * other_sequence):(Fr * (other_sequence + 1))])
                # Compute spatial centering feature correspondence
                F = F - torch.mean(F, dim=(-1), keepdim=True)
                # Compute, normalized, feature correspondence similarity
                similarity_histogram[3, :] =+ torch.histogram(torch.flatten((F.detach() * S.detach()).cpu()),
                                                              bins=torch.arange(-1, 1.001, step=0.05))[0]
                # Update loss function with contribution from comparing current sequence with other sequence
                L += - loss_weighting_contrastive * torch.mean((F - b_contrastive) * torch.clamp(S, min=0))
        return L

    def load_front_end(self, model_path, device):
        '''
        Load model from file.
        '''
        parameter_checkpoint = super(StegoLikeModel, self)._load_front_end_file(model_path, device)

        # Load hyper-parameters
        self.proj.load_state_dict(parameter_checkpoint["NN_model_front_end"])
        self.b_param = parameter_checkpoint["b"]
        self.loss_weighting = parameter_checkpoint["loss_weights"]

    def save_front_end(self, model_path):
        '''
        Save front_end model
        '''
        # Define parameters to save
        objects_dict = {"NN_model_front_end": self.proj.state_dict(), "b": self.b_param, "loss_weights": self.loss_weighting}
        # Packet and save
        self._save_front_end_file(model_path, objects_dict)

    def front_end_forward(self, x, num_layers=None, use_activation=True, hard_classification=False, **kwargs):
        '''
        Compute segmentation (using the n:th hidden layer counted from the last in the DINO architecture)
        '''
        # Loop through all blocks in sequential
        for iter, iter_block in enumerate(self.proj):

            # See if current block is the last
            stop = (iter + 1 >= len(self.proj))

            # If defined another stopping criteria
            if num_layers is not None:
                if num_layers >= 0 and iter + 1 >= num_layers:
                    stop = True
                elif num_layers < 0 and iter + 1 >= len(self.proj) + num_layers:
                    stop = True
            # If time to stop
            if stop:
                # evaluate the last one and choose to use activation function or not
                x = iter_block(x, use_activation=use_activation)
                break
            else:
                # Continue running through blocks
                x = iter_block(x)

        # If should classify as hard number
        if hard_classification:
            # Acquire hard classification
            x = torch.argmax(x, dim=-1)

        return x


class StegoLikeModel_3_layer_marginal(StegoLikeModel):
    '''
    Class representing a 3 layer marginal architecture for a StegoLikeModel
    '''
    def __init__(self, dino_model, num_classes, **kwargs):

        super(StegoLikeModel_3_layer_marginal, self).__init__(dino_model=dino_model, front_end=torch.nn.Sequential(
                    MarginalBlock(in_dim=dino_model.embed_dim, out_dim=dino_model.embed_dim, dropout_p=0.2,
                                  skip_connection=True, activation=torch.nn.GELU(),
                                  initialization_function=torch.nn.init.xavier_uniform_,
                                  norm_layer=torch.nn.LayerNorm(dino_model.embed_dim)),
                    MarginalBlock(in_dim=dino_model.embed_dim, out_dim=dino_model.embed_dim, dropout_p=0.2,
                                  skip_connection=True, activation=torch.nn.ReLU(),
                                  initialization_function=torch.nn.init.kaiming_uniform_,
                                  norm_layer=torch.nn.LayerNorm(dino_model.embed_dim)),
                    MarginalBlock(in_dim=dino_model.embed_dim, out_dim=num_classes, dropout_p=0.2,
                                  skip_connection=False, activation=torch.nn.Softmax(dim=-1),
                                  initialization_function=torch.nn.init.xavier_uniform_,
                                  norm_layer=torch.nn.LayerNorm(dino_model.embed_dim))), num_classes=num_classes, **kwargs)
