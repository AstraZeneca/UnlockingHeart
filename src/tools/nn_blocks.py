# Class of a block in the segmentation chain

# Load packages
import torch


class MarginalBlock(torch.nn.Module):
    # Class of a block in the segmentation chain

    def __init__(self, in_dim, out_dim,
                 activation=None,
                 dropout_p=0.0,
                 skip_connection=True,
                 initialization_function=torch.nn.init.xavier_uniform_,
                 norm_layer=None):
        """
        Constructor
        :param in_dim: The number of dimensions in input.
        :param out_dim: The number of dimensions in output.
        :param activation: An activation_function object to use for the output of the block.
        :param dropout_p: The probability of a randomly chosen weight to be droped in a backpropagation step. Default '0.0', i.e., no dropout.
        :param skip_connection: 'True' if the block should have a skip connection.
        If in_dim != out_dim this will be set to false regardless. Default 'True'
        :param initialization_function: What function is used to set the initial values of the weights of
        the linear layer (not the bias). Default 'torch.nn.init.xavier_uniform_'.
        :param norm_layer: A normalization layer object.

        """

        # Initialize parent
        super().__init__()

        self.skip_connection = False
        if in_dim == out_dim:
            self.skip_connection = skip_connection

        # Set normalization layer type
        self.norm = norm_layer

        # Set first internal projection layer
        self.linear_layer = torch.nn.Linear(in_dim, out_dim) if out_dim > 0 else torch.nn.Identity()
        # Set drop out layer
        self.dropout = torch.nn.Dropout(dropout_p)
        # Set activation function
        self.act = activation

        initialization_function(self.linear_layer.weight)
        self.linear_layer.bias.data.fill_(0.0)

    def forward(self, x, use_activation=True):
        # Feed 'x' forward
        y = x

        # Normalize
        if self.norm is not None:
            y = self.norm(y)

        # Go through linear layer
        y = self.linear_layer(y)
        # Apply dropout
        y = self.dropout(y)
        # Run through activation function
        if self.act is not None and use_activation:
            y = self.act(y)
        # Apply skip connection
        if self.skip_connection:
            x = x + y
        else:
            x = y
        return x
