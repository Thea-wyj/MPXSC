"""
A deep Q-learning network for Deep Q-Learning algorithm.
"""
import torch
import numpy as np

from torch import nn, optim

from general.general_net import CnnExtractor


class DqlNet(CnnExtractor):
    def __init__(self, img_size: tuple, out_channels: int):
        """
        A 2D convolution network, and the hidden layer use linear function with 512 neurons.
        The lose function is Huber loss, and the optimiser is Adam algorithm.

        Args:
            img_size: A tuple in (in_channels, height, width) form.
            out_channels: Num of the output action.
        """
        super().__init__(img_size)
        self.fully_net = nn.Sequential(
            nn.Linear(self.flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, out_channels))

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001, eps=1e-6)
        self.loss = nn.SmoothL1Loss()
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict action through input images.

        Args:
            x: A sequence includes the information of images.

        Returns:
            prediction: A sequence includes the prediction value of all actions.
        """
        x = self.conv(x)
        prediction = self.fully_net(x)
        prediction = nn.functional.softmax(prediction, dim=1)
        return prediction
