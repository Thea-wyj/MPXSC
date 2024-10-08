"""
A general convolution network.
"""
import torch
import numpy as np

from torch import nn
from torchvision import transforms


def generate_conv_layer(input_layer_par: tuple, pool_size=None, ave_pool=True):
    """
    Generate a 2D-convolution layer with pytorch Conv2d, AvgPool2d/MaxPool2d and Sequential class.

    Args:
        input_layer_par: Conclude three parameters, num of channels in the input image(int),
        number of channels produced by the convolution(int), stride of the convolution(int or tuple) and
        size of the convolve kernel(int or tuple), respectively.
        pool_size: A tuple or None, the former representative the size of the pool.
        ave_pool: A bool value, True represents use average pool, else max pool.

    Returns:
        A 2D convolution layer.
    """
    input_num, output_num, kernel_size, stride = input_layer_par
    conv = nn.Conv2d(input_num, output_num, kernel_size, stride, padding=1)
    if pool_size is not None:
        pool = nn.AvgPool2d(pool_size) if ave_pool else nn.MaxPool2d(pool_size)
        return nn.Sequential(conv, nn.ReLU(), pool)
    else:
        return nn.Sequential(conv, nn.ReLU())


class CnnExtractor(nn.Module):
    def __init__(self, img_size: tuple):
        """
        A 2D convolution network used for atari video games with three convolution layers,
        a hidden layer and a output layer.
        The activation functions is Relu.
        In three convolution layers, the first layer has 32 output channels generated by (8*8) kernel
        with (4,4) stride; the second layer has 64 output channels generated by (4*4) kernel with (2,2) stride;
        and the last layer has 64 output channels generated by (3*3) kernel with (1,1) stride.

        Args:
            img_size: A tuple in (in_channels, height, width) form.
        """
        super().__init__()
        self.conv1 = generate_conv_layer(input_layer_par=(img_size[0], 32, (8, 8), (4, 4)))
        self.conv2 = generate_conv_layer(input_layer_par=(32, 64, (4, 4), (2, 2)))
        self.conv3 = generate_conv_layer(input_layer_par=(64, 64, (3, 3), (1, 1)))
        self.conv_set = (self.conv1, self.conv2, self.conv3)
        self.flatten_size = self._cal_size((1, *img_size))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _cal_size(self, img_size):
        """
        Calculate num of the elements after convolution layers.
        """
        x = torch.zeros(img_size)
        for conv in self.conv_set:
            x = conv(x)
        return np.prod(x.shape[1:])

    def conv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate convolution of the input images.
        """
        for conv in self.conv_set:
            x = conv(x)
        x = x.view(-1, np.prod(x.shape[1:]))
        return x
