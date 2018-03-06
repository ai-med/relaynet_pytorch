"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - kernel_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride_conv: Stride for the convolution layer.
        - stride_pool: Stride for the max pooling layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ClassificationCNN, self).__init__()
        channels, height, width = input_dim

        ########################################################################
        # TODO: Initialize the necessary trainable layers to resemble the      #
        # ClassificationCNN architecture  from the class docstring.            #
        #                                                                      #
        # In- and output features should not be hard coded which demands some  #
        # calculations especially for the input of the first fully             #
        # convolutional layer.                                                 #
        #                                                                      #
        # The convolution should use "same" padding which can be derived from  #
        # the kernel size and its weights should be scaled. Layers should have #
        # a bias if possible.                                                  #
        #                                                                      #
        # Note: Avoid using any of PyTorch's random functions or your output   #
        # will not coincide with the Jupyter notebook cell.                    #
        ########################################################################
        padding = int((kernel_size - 1) / 2)

        self.conv = nn.Conv2d(channels, num_filters, kernel_size, padding=padding, stride=stride_conv)
        conv_out_width, conv_out_height = int(width + 2 * padding - kernel_size / stride_conv + 1), int(
            (height + 2 * padding - kernel_size / stride_conv) + 1)
        # print('Conv output size')
        # print(conv_out_width, conv_out_height)

        self.max_pool = nn.MaxPool2d(pool, stride=stride_pool)
        max_pool_out_width, max_pool_out_height = int(((conv_out_width - pool) / stride_pool) + 1), int(
            ((conv_out_height - pool) / stride_pool) + 1)

        # print('Pool output size')
        # print(max_pool_out_width, max_pool_out_height)

        max_pool_output_size = max_pool_out_width * max_pool_out_height
        # print('Max pool est size :', str(max_pool_output_size))
        self.fc1 = nn.Linear(num_filters * max_pool_output_size, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but by
        calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        ########################################################################
        # TODO: Chain our previously initialized fully-connected neural        #
        # network layers to resemble the architecture drafted in the class     #
        # docstring. Have a look at the Variable.view function to make the     #
        # transition from the spatial input image to the flat fully connected  #
        # layers.                                                              #
        ########################################################################
        # Max pooling over a (2, 2) window
        x_conv = F.relu(self.conv(x))
        # print('Conv shape :')
        # print(x_conv.shape)
        x_pool = self.max_pool(x_conv)
        # print('Pool shape :')
        # print(x_pool.shape)
        x_flatten = x_pool.view(-1, self.num_flat_features(x_pool))
        # print('Flatten shape :')
        # print(x_flatten.shape)
        x_fc1 = F.relu(self.dropout(self.fc1(x_flatten)))
        # print('x_fc1 shape :')
        # print(x_fc1.shape)
        x = self.fc2(x_fc1)
        # print('x_final shape :')
        # print(x.shape)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return x

    def num_flat_features(self, x):
        """
        Computes the number of features if the spatial input x is transformed
        to a 1D flat input.
        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
