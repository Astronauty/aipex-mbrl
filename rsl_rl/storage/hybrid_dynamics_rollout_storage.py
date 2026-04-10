
import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.networks import HiddenState
from rsl_rl.utils import split_and_pad_trajectories

class HybridDynamicsRolloutStorage:
    def __init__(self, num_modes, obs_dim, act_dim, history_len_obs, history_len_act, max_size, device):
        self.num_modes = num_modes
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.na = history_len_obs
        self.nb = history_len_act
        self.max_size = max_size
        self.device = device

        # Storage per mode
        self.buffers = {}
        
        # for mode in range(num_modes):
        #     self.buffers[mode] = {
        #         # Storing Flattened history features directly to save computation during training
        #         # Shape: [max_size, obs_dim * na + act_dim * nb]
        #         "features": torch.zeros((max_size, obs_dim * self.na + act_dim * self.nb), device=device),
        #         # Shape: [max_size, obs_dim] (Target next state)
        #         "next_obs": torch.zeros((max_size, obs_dim), device=device),
        #         "ptr": 0,
        #         "size": 0
        #     }
        
        # Explicitly disable inference mode so the buffer arrays evaluate as normal mutable tensors
        with torch.inference_mode(False):
            for mode in range(num_modes):
                self.buffers[mode] = {
                    # Storing Flattened history features directly to save computation during training
                    # Shape: [max_size, obs_dim * na + act_dim * nb]
                    "features": torch.zeros((max_size, obs_dim * self.na + act_dim * self.nb), device=device),
                    # Shape: [max_size, obs_dim] (Target next state)
                    "next_obs": torch.zeros((max_size, obs_dim), device=device),
                    "ptr": 0,
                    "size": 0
                }

    def add(self, mode_indices, obs_history_window, act_history_window, next_obs):
        """
        mode_indices: [Batch]
        obs_history_window: [Batch, na, obs_dim]
        act_history_window: [Batch, nb, act_dim]
        next_obs: [Batch, obs_dim]
        """
        # Flatten features: [Batch, Feature_Dim]
        obs_flat = obs_history_window.flatten(1, 2)
        act_flat = act_history_window.flatten(1, 2)
        features = torch.cat([obs_flat, act_flat], dim=-1)

        # Force mode_indices to be 1-dimensional and cast to integer types to prevent crashes 
        # when incorrect multidimensional observation tensors are erroneously passed as mode labels.
        if mode_indices.dim() > 1:
            if mode_indices.shape[-1] == 1:
                mode_indices = mode_indices.squeeze(-1)
            else:
                mode_indices = mode_indices.reshape(mode_indices.shape[0], -1)[:, 0]
        
        mode_indices = mode_indices.long()

        unique_modes = torch.unique(mode_indices)
        
        for mode in unique_modes:
            mode_idx = mode.item()
            # Failsafe bounds check to avoid KeyError if labels are derived from unclipped states
            if mode_idx < 0 or mode_idx >= self.num_modes: continue
            
            mask = (mode_indices == mode)
            n_samples = mask.sum().item()
            if n_samples == 0: continue

            # Extract data for this mode. 
            # Cloning explicitly strips the inference tensor flag to prevent inplace update RuntimeErrors.
            mode_feats = features[mask].clone()
            mode_next = next_obs[mask].clone()

            # If the current batch of samples is larger than the buffer capacity, 
            # keep only the most recent max_size samples to prevent out-of-bounds wrap-around.
            if n_samples > self.max_size:
                mode_feats = mode_feats[-self.max_size:]
                mode_next = mode_next[-self.max_size:]
                n_samples = self.max_size

            # Buffer logic
            buf = self.buffers[mode_idx]
            ptr = buf["ptr"]
            
            # Handle wrapping if batch is larger than remaining space
            remaining = self.max_size - ptr
            if n_samples <= remaining:
                buf["features"][ptr : ptr+n_samples] = mode_feats
                buf["next_obs"][ptr : ptr+n_samples] = mode_next
                buf["ptr"] = (ptr + n_samples) % self.max_size
                buf["size"] = min(buf["size"] + n_samples, self.max_size)
            else:
                # Split writing
                # 1. Fill to end
                buf["features"][ptr:] = mode_feats[:remaining]
                buf["next_obs"][ptr:] = mode_next[:remaining]
                # 2. Wrap to beginning
                overflow = n_samples - remaining
                buf["features"][:overflow] = mode_feats[remaining:]
                buf["next_obs"][:overflow] = mode_next[remaining:]
                
                buf["ptr"] = overflow
                buf["size"] = self.max_size


    def sample_balanced(self, batch_size_per_mode=256):
        """Returns concatenated batch from all modes"""
        features_list = []
        targets_list = []
        labels_list = []

        for mode in range(self.num_modes):
            buf = self.buffers[mode]
            if buf["size"] == 0: continue
            
            # Random indices
            indices = torch.randint(0, buf["size"], (batch_size_per_mode,), device=self.device)
            
            features_list.append(buf["features"][indices])
            targets_list.append(buf["next_obs"][indices])
            labels_list.append(torch.full((batch_size_per_mode,), mode, device=self.device, dtype=torch.long))

        if not features_list:
            return None, None, None

        return torch.cat(features_list), torch.cat(labels_list), torch.cat(targets_list)