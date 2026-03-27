import torch
import torch.nn as nn

from abc import ABC, abstractmethod

class DiscreteTimeDynamicsModel(nn.Module, ABC):
    """Abstract class for discrete time dynamics models."""
    def __init__(
            self,
            state_dim: int,
            action_dim: int, 
            device: str,
            history_horizon_length: int = 1,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_horizon_length = history_horizon_length
        self.device = device

    @abstractmethod()
    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        pass
