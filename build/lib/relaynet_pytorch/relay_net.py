"""ClassificationCNN"""
import torch
import torch.nn as nn

from relaynet_pytorch.net_api import sub_module as sm


class ReLayNet(nn.Module):
    """
    A PyTorch implementation of ReLayNet
    Coded by Shayan and Abhijit

    param ={
        'num_channels':1,
        'num_filters':64,
        'num_channels':64,
        'kernel_h':7,
        'kernel_w':3,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':10
    }

    """

    def __init__(self, params):
        super(ReLayNet, self).__init__()

        self.encode1 = sm.EncoderBlock(params)
        params['num_channels'] = 64
        self.encode2 = sm.EncoderBlock(params)
        # params['num_channels'] = 64  # This can be used to change the numchannels for each block
        self.encode3 = sm.EncoderBlock(params)
        self.bottleneck = sm.BasicBlock(params)
        params['num_channels'] = 128
        self.decode1 = sm.DecoderBlock(params)
        self.decode2 = sm.DecoderBlock(params)
        self.decode3 = sm.DecoderBlock(params)
        params['num_channels'] = 64
        self.classifier = sm.ClassifierBlock(params)

    def forward(self, input):
        e1, out1, ind1 = self.encode1.forward(input)
        e2, out2, ind2 = self.encode2.forward(e1)
        e3, out3, ind3 = self.encode3.forward(e2)
        bn = self.bottleneck.forward(e3)

        d3 = self.decode1.forward(bn, out3, ind3)
        d2 = self.decode2.forward(d3, out2, ind2)
        d1 = self.decode3.forward(d2, out1, ind1)
        prob = self.classifier.forward(d1)

        return prob

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
