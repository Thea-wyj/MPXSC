"""
Memory space contains sates, actions, rewards, next states and terminal symbols.
"""
import torch
from numpy.random import randint


class Memory:
    def __init__(self, memory_size: int, state_size: tuple):
        """
        The memory space used for saving and sampling experience,
        including sates, actions, rewards, next states and terminal symbols.

        Args:
            memory_size: The size of the memory space.
            state_size: The size of the states.
        """
        self.memory_size = memory_size
        self.current_index = 0
        self.state = torch.zeros(size=(memory_size, *state_size), dtype=torch.uint8)
        self.state_ = torch.zeros(size=(memory_size, *state_size), dtype=torch.uint8)
        self.action = torch.zeros(memory_size, dtype=torch.int8)
        self.reward = torch.zeros(memory_size, dtype=torch.int)
        self.terminal = torch.zeros(memory_size, dtype=torch.bool)

    def store_sars_(self, s, a, r, s_, done):
        """
        Store memory.
        """
        index = self.current_index % self.memory_size
        self.state[index] = s
        self.state_[index] = s_
        self.action[index] = a
        self.reward[index] = r
        self.terminal[index] = done
        self.current_index += 1

    def sample(self, batch_size: int):
        """
        Sample batch size memory form memory space.
        """
        batch = randint(min(self.current_index, self.memory_size), size=batch_size)
        a_batch = self.action[batch]
        s_batch = self.state[batch]
        s_batch_ = self.state_[batch]
        r_batch = self.reward[batch]
        t_batch = self.terminal[batch]
        return s_batch, a_batch, r_batch, s_batch_, t_batch
