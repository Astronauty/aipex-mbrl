# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.networks import HiddenState
from rsl_rl.utils import split_and_pad_trajectories

# Rollout storage for dynamics model learning


class DynamicsRolloutStorage:
    """Storage for the data collected during a rollout.

    The rollout storage is populated by adding transitions during the rollout phase. It then returns a generator for
    learning, depending on the algorithm and the policy architecture.
    """

    class Transition:
        """Storage for a single state transition."""

        def __init__(self) -> None:
            self.observations: TensorDict | None = None
            self.actions: torch.Tensor | None = None
            self.hidden_states: tuple[HiddenState, HiddenState] = (None, None)
            self.contact_mode: torch.Tensor | None = None

        def clear(self) -> None:
            self.__init__()

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape

        # Core
        self.observations = TensorDict(
            {key: torch.zeros(num_transitions_per_env, *value.shape, device=device) for key, value in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )

        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.contact_mode = torch.zeros(num_transitions_per_env, num_envs, dtype=torch.int64, device=self.device)
        
        # Counter for the number of transitions stored
        self.step = 0

    def add_transition(self, transition: Transition) -> None:
        # Check if the transition is valid
        # if self.step >= self.num_transitions_per_env:
        #     raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")
        
        
        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        # self.contact_mode[self.step].copy_(transition.contact_mode)
        self.step += 1

    def get_item(self, idx):
        return
    
    

    def clear(self) -> None:
        self.step = 0

