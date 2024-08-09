"""
Two networks used for dql algorithm.
"""
import torch
from torch import nn, optim

from general.general_net import CnnExtractor


class QNet(nn.Module):
    def __init__(self, img_size: tuple, output_num=1):
        """
        A fully connected network, including two layers, each layer has 512 neurons.

        Args:
            img_size: A tuple in (in_channels, height, width) form.
            output_num: The number of results elements.
        """
        super().__init__()
        self.conv = CnnExtractor(img_size)
        self.fully_net = nn.Sequential(
            nn.Linear(self.conv.flatten_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_num))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict results according input tensor.
        """
        x = self.conv.conv(x)
        x = self.fully_net(x)
        return x


class PolicyNetwork(QNet):
    def __init__(self, img_size: tuple, output_num=1):
        """
        The policy network for Actor-Critics algorithm.

        Args:
            img_size: A tuple in (in_channels, height, width) form.
            output_num: The number of results elements.
        """
        super(PolicyNetwork, self).__init__(img_size, output_num)
        self.to(self.device)

    def get_best_action(self, s: torch.Tensor) -> torch.Tensor:
        """
        Get the best action about certain state.

        Args:
            s: The state of the environment.

        Returns:
            action: The best action for this state.
        """
        action_value = self.forward(s)
        action = torch.argmax(action_value).item()
        return action

    def sample_action(self, s: torch.Tensor):
        """
        Sample action with their probabilities.

        Args:
            s: The state of the environment.

        Returns:
            action: The action sample from certain probabilities.
            action_prob: The probabilities for different actions.
            log_action_prob: Rescale actions' probabilities in log scale.
        """
        action_prob = torch.nn.functional.softmax(self.forward(s), dim=1)
        action_distribution = torch.distributions.Categorical(action_prob)
        action = action_distribution.sample()
        z = (action_prob == 0.0).float() * 1e-8
        log_action_prob = torch.log(action_prob + z)
        return action, action_prob, log_action_prob
