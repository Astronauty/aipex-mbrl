# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL using a dummy environment (no IsaacLab dependency)."""

import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import torch
from datetime import datetime
from tensordict import TensorDict

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.runners import MBRLOnPolRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class DummyEnv(gym.Env):
    """Dummy env."""

    def __init__(self, num_envs=4, obs_dim=10, action_dim=4, device="cpu"):
        super().__init__()
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.num_actions = action_dim
        self.device = device

        self.action_space = spaces.Box(low=-1, high=1, shape=(num_envs, action_dim), dtype=np.float32)
        # Observation space needs to be dicttionary for rslrl
        self.observation_space = spaces.Dict({
            "policy": spaces.Box(low=-np.inf, high=np.inf, shape=(num_envs, obs_dim), dtype=np.float32)
        })

        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        obs_data = torch.randn(self.num_envs, self.obs_dim, dtype=torch.float32, device=self.device)
        self.obs = TensorDict({"policy": obs_data})
        return self.obs, {}

    def step(self, actions):
        # Random transition not using actions
        obs_data = torch.randn(self.num_envs, self.obs_dim, dtype=torch.float32, device=self.device)
        self.obs = TensorDict({"policy": obs_data})
        rewards = torch.randn(self.num_envs, dtype=torch.float32, device=self.device)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        return self.obs, rewards, dones, truncated, {}

    def close(self):
        pass

class SimpleVecEnvWrapper(gym.Env):
    """Simple wrapper to make environment compatible with RSL-RL Runner definitions."""

    def __init__(self, env, clip_actions=True):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.num_envs = env.num_envs
        self.num_actions = env.num_actions
        self.device = env.device
        self.clip_actions = clip_actions
        # Create a simple cfg object just to fool runner 
        class EnvConfig:
            def __init__(self, num_envs, num_actions):
                self.num_envs = num_envs
                self.num_actions = num_actions

        self.cfg = EnvConfig(env.num_envs, env.num_actions)

        # Track episode lengths for each environment
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int32, device=env.device)
        self.max_episode_length = 1000  # Maximum steps before episode truncation

    def reset(self, seed=None):
        self.episode_length_buf.fill(0)
        return self.env.reset(seed=seed)

    def step(self, actions):

        obs, rewards, dones, truncated, info = self.env.step(actions)

        # Update episode length buf and reset for done environments
        self.episode_length_buf += 1
        done_mask = dones | truncated
        self.episode_length_buf[done_mask] = 0

        return obs, rewards, dones, info

    def close(self):
        self.env.close()

    def render(self):
        pass

    def get_observations(self):
        return self.env.obs


class SimpleAgentConfig(dict):
    """Simple agent configuration compatible with OnPolicyRunner."""

    def __init__(self, max_iterations=100, seed=0, device="cpu"):
        super().__init__()
        self.update({
            "experiment_name": "dummy_test",
            "num_steps_per_env": 24,
            "run_name": "test_run",
            "save_interval": 50,
            "max_iterations": max_iterations,
            "seed": seed,
            "device": device,
            "class_name": "OnPolicyRunner",
            "clip_actions": True,
            "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
            "actor": {
                "class_name": "MLPModel",
                "hidden_dims": [256],
                "activation": "elu",
                "obs_normalization": False,
                "distribution_cfg": {
                    "class_name": "GaussianDistribution",
                    "init_std": 1.0,
                },
            },
            "critic": {
                "class_name": "MLPModel",
                "hidden_dims": [256],
                "activation": "elu",
                "obs_normalization": False,
            },
            "algorithm": {
                "class_name": "MBPO",
                "value_loss_coef": 1.0,
                "use_clipped_value_loss": True,
                "clip_param": 0.2,
                "entropy_coef": 0.005,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "learning_rate": 1.0e-3,
                "schedule": "adaptive",
                "gamma": 0.99,
                "lam": 0.95,
                "desired_kl": 0.01,
                "max_grad_norm": 1.0,
            },
            "replay_buffer": {
                "num_episodes_replay": 100,
            },
        })

    def to_dict(self):
        return dict(self)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")



def main():
    """Train with RSL-RL agent using dummy environment."""
    parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL using dummy environment.")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
    parser.add_argument("--obs_dim", type=int, default=10, help="Observation dimension.")
    parser.add_argument("--action_dim", type=int, default=4, help="Action dimension.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--max_iterations", type=int, default=100, help="RL Policy training iterations.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda).")

    args_cli = parser.parse_args()

    agent_cfg = SimpleAgentConfig(
        max_iterations=args_cli.max_iterations,
        seed=args_cli.seed if args_cli.seed is not None else 0,
        device=args_cli.device
    )

    

    # don't save logs
    log_dir = None
    ## specify dirctory to trigger logging visualizatyion
    # # specify directory for logging experiments
    # log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    # log_root_path = os.path.abspath(log_root_path)
    # print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # if agent_cfg.run_name:
    #     log_dir += f"_{agent_cfg.run_name}"
    # log_dir = os.path.join(log_root_path, log_dir)
    # Create dummy environment
    print(f"[INFO] Creating dummy environment with {args_cli.num_envs} parallel environments")
    env = DummyEnv(
        num_envs=args_cli.num_envs,
        obs_dim=args_cli.obs_dim,
        action_dim=args_cli.action_dim,
        device=args_cli.device
    )

    # wrap around environment for rsl-rl
    env = SimpleVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    print(f"[INFO] Creating OnPolicyRunner")
    # runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner = MBRLOnPolRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # run training
    print(f"[INFO] Starting training for {agent_cfg.max_iterations} iterations")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the environment
    env.close()
    print(f"[INFO] Training completed. Logs saved to {log_dir}")


if __name__ == "__main__":
    main()
