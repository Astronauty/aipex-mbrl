from __future__ import annotations
from abc import ABC, abstractmethod

import os 
import torch
import statistics
import time
import warnings
from collections import deque
from tensordict import TensorDict

import rsl_rl
from rsl_rl.env import VecEnv

from rsl_rl.utils.logger import Logger

class MBRLRunner(ABC):
    """Generic runner for model based reinforcement learning. Implemented as an abstract class with
    core functionality of:
    1. Handling training configurations
    2. Handling logging"""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        self.env=env
        self.cfg = train_cfg
        # Policy cfg only instantiated in children where policy is required (offline planning, and some online planning like TD-MPC)
        self.alg_cfg = train_cfg["algorithm"]
        self.device = device
        
        self._configure_multi_gpu()

        # Get observations from env to establish size of obs space
        obs = self.env.get_observations()
        # self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], self._get_default_obs_sets())

        # Construct the dynamics model used for MBRL algorithms
        self.dynamics_model_cfg = self.alg_cfg["dynamics_model"]

        DynamicsClass = getattr(rsl_rl.modules, self.dynamics_model_cfg["class_name"]) # Find the correct dynamics class based on config
        self.dynamics_model = DynamicsClass(
             state_dim=self.dynamics_model_cfg["state_dim"],
             action_dim=self.dynamics_model_cfg["action_dim"],
             device=self.device,
             history_horizon_length=self.dynamics_model_cfg["history_horizon_length"]
        )

        # Initialize storage for imagined rollouts w/ the learned dynamics mdoel
        self.dynamics_data_buffer = 


        # Initialize storage for simulated rollouts (via IsaacLAB)
        
        # Define value function? 

        # Initialize policy


        # Initialize logger
        self.logger = Logger(log_dir=log_dir,
                             cfg=self.cfg,
                             num_envs=self.env.num_envs,
                             is_distributed=self.is_distributed,
                             gpu_world_size=self.gpu_world_size,
                             gpu_global_rank=self.gpu_global_rank,
                             device=self.device)


        self.current_learning_iteration = 0
        pass

    @abstractmethod
    def learn(self):
        """
        Main learning loop for MBRL algorithms.
        """

        pass

    @abstractmethod
    def update_dynamics_model(self, dynamics_model, traj) -> None:
        """Update self.dynamics model based on sampled trajectories. These trajectories can be sampled 
        either from the sim environment or a replay buffer.

        Returns
        - M : updated dynamics model       
        """
        return NotImplemented


    @abstractmethod
    def update_policy(self, policy, value_function, traj):
        """Update self."""
        return NotImplemented
    

    @abstractmethod 
    def rollout(self, state, policy, ):
        return NotImplemented


    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Register a repository path whose git status should be logged."""
        self.logger.git_status_repos.append(repo_file_path)
    
    # Helper functions
    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.cfg["multi_gpu"] = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.cfg["multi_gpu"] = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-GPU configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # Initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # Set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)


    @abstractmethod
    def _construct_algorithm(self, obs: TensorDict):
        return NotImplementedError