from envs.base import BaseEnv
import torch
from tensordict import TensorDict

class Go2FlatEnv(BaseEnv):
    def __init__(
        self, 
        num_envs, 
        max_episode_length, 
        step_dt, 
        reward_term_weights, 
        device, 
        uncertainty_penalty_weight, 
        observation_noise, 
        command_resample_interval_range, 
        event_interval_range,
        debug_viz=False
    ):
        super().__init__(
            num_envs=num_envs, 
            max_episode_length=max_episode_length, 
            step_dt=step_dt, 
            reward_term_weights=reward_term_weights, 
            device=device, 
            uncertainty_penalty_weight=uncertainty_penalty_weight, 
            observation_noise=observation_noise, 
            command_resample_interval_range=command_resample_interval_range, 
            event_interval_range=event_interval_range
        )
        
        self.debug_viz = debug_viz
        
        # Go2 Joint Limits
        self.dof_pos_limits = torch.stack([
            torch.tensor([-1.0472, -1.5708, -2.7227] * 4, device=device),
            torch.tensor([ 1.0472,  3.4907, -0.8378] * 4, device=device)
        ], dim=1)

        # State indices (Adjusted for 33 dims: 0-9 Base, 9-21 Pos, 21-33 Vel)
        self.idx_lin_vel = slice(0, 3)
        self.idx_ang_vel = slice(3, 6)
        self.idx_projected_gravity = slice(6, 9)
        self.idx_joint_pos = slice(9, 21)
        self.idx_joint_vel = slice(21, 33)

    def _init_additional_imagination_attributes(self):
        self.last_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.current_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_contact_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.current_contact_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.obs_last_action = torch.zeros(self.num_envs, 12, device=self.device)

    def _reset_additional_imagination_attributes(self, env_ids):
        if len(env_ids) > 0:
            self.last_air_time[env_ids] = 0.0
            self.current_air_time[env_ids] = 0.0
            self.last_contact_time[env_ids] = 0.0
            self.current_contact_time[env_ids] = 0.0
            self.obs_last_action[env_ids] = 0.0

    def _init_imagination_command(self):
        r = torch.empty(self.num_envs, device=self.device)
        self.base_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        # Randomize commands: X [-1, 1], Y [-1, 1], Yaw [-1, 1]
        self.base_velocity[:, 0] = r.uniform_(-1.0, 1.0)
        self.base_velocity[:, 1] = r.uniform_(-1.0, 1.0)
        self.base_velocity[:, 2] = r.uniform_(-1.0, 1.0)

    def _reset_imagination_command(self, env_ids):
        if len(env_ids) > 0:
            r = torch.empty(len(env_ids), device=self.device)
            self.base_velocity[env_ids, 0] = r.uniform_(-1.0, 1.0)
            self.base_velocity[env_ids, 1] = r.uniform_(-1.0, 1.0)
            self.base_velocity[env_ids, 2] = r.uniform_(-1.0, 1.0)

    def get_imagination_observation(self, state_history, action_history):
        state_history_denormalized, action_history_denormalized = self.dataset.denormalize(state_history[:, -1], action_history[:, -1])
        
        # 33-Dim State Parsing
        obs_base_lin_vel = state_history_denormalized[:, 0:3]
        obs_base_ang_vel = state_history_denormalized[:, 3:6]
        obs_projected_gravity = state_history_denormalized[:, 6:9]
        obs_joint_pos = state_history_denormalized[:, 9:21]   # Indices corrected for 33 dim input
        obs_joint_vel = state_history_denormalized[:, 21:33]  # Indices corrected for 33 dim input
        self.obs_last_action = action_history_denormalized

        if self.observation_noise:
            obs_base_lin_vel += 2 * (torch.rand_like(obs_base_lin_vel) - 0.5) * 0.1
            obs_base_ang_vel += 2 * (torch.rand_like(obs_base_ang_vel) - 0.5) * 0.2
            obs_projected_gravity += 2 * (torch.rand_like(obs_projected_gravity) - 0.5) * 0.05
            obs_joint_pos += 2 * (torch.rand_like(obs_joint_pos) - 0.5) * 0.01
            obs_joint_vel += 2 * (torch.rand_like(obs_joint_vel) - 0.5) * 1.5

        # Construct 48-Dim Observation for Policy
        # [Lin(3), Ang(3), Grav(3), CMD(3), Pos(12), Vel(12), Act(12)]
        obs = torch.cat([
            obs_base_lin_vel, 
            obs_base_ang_vel, 
            obs_projected_gravity, 
            self.base_velocity, 
            obs_joint_pos, 
            obs_joint_vel, 
            self.obs_last_action
        ], dim=1)
        
        obs = TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)
        self.last_obs = obs
        return obs

    def _parse_imagination_states(self, imagination_states_denormalized):
        # 33-Dim Slicing
        base_lin_vel = imagination_states_denormalized[:, 0:3]
        base_ang_vel = imagination_states_denormalized[:, 3:6]
        projected_gravity = imagination_states_denormalized[:, 6:9]
        joint_pos = imagination_states_denormalized[:, 9:21]
        joint_vel = imagination_states_denormalized[:, 21:33]
        
        # Zero fill torque as we don't track it in 33-dim state
        joint_torque = torch.zeros_like(joint_vel)

        parsed_imagination_states = {
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_torque": joint_torque,
        }
        return parsed_imagination_states

    def _parse_extensions(self, extensions):
        return None

    def _parse_contacts(self, contacts):
        if contacts is not None:
             thigh_contact = torch.sigmoid(contacts[:, 0:4]).round()
             foot_contact = torch.sigmoid(contacts[:, 4:8]).round()
             return {"thigh_contact": thigh_contact, "foot_contact": foot_contact}
        return {"thigh_contact": None, "foot_contact": None}
    
    def _parse_terminations(self, terminations):
        if terminations is not None:
             return torch.sigmoid(terminations).squeeze(-1).round().bool()
        return None

    def _compute_imagination_reward_terms(self, parsed_imagination_states, rollout_action, parsed_extensions, parsed_contacts):
        # Extract
        base_lin_vel = parsed_imagination_states["base_lin_vel"]
        base_ang_vel = parsed_imagination_states["base_ang_vel"]
        projected_gravity = parsed_imagination_states["projected_gravity"]
        joint_pos = parsed_imagination_states["joint_pos"]
        joint_vel = parsed_imagination_states["joint_vel"]
        joint_torque = parsed_imagination_states["joint_torque"]
        
        # Approximate Acceleration
        if hasattr(self, "last_obs") and self.last_obs is not None:
             # prev_joint_vel corresponds to 21:33 in State, or 24:36 in Obs
             # Using State logic here (21:33) might be tricky if last_obs is Obs (48 dims)
             prev_joint_vel = self.last_obs["policy"][:, 24:36] 
             joint_acc = (joint_vel - prev_joint_vel) / self._step_dt
        else:
             joint_acc = torch.zeros_like(joint_vel)

        thigh_contact = parsed_contacts["thigh_contact"]
        foot_contact = parsed_contacts["foot_contact"]

        # Reward Terms
        lin_vel_error = torch.sum(torch.square(self.base_velocity[:, :2] - base_lin_vel[:, :2]), dim=1)
        track_lin_vel_xy_exp = torch.exp(-lin_vel_error / 0.25)
        
        ang_vel_error = torch.square(self.base_velocity[:, 2] - base_ang_vel[:, 2])
        track_ang_vel_z_exp = torch.exp(-ang_vel_error / 0.25)
        
        lin_vel_z_l2 = torch.square(base_lin_vel[:, 2]) 
        ang_vel_xy_l2 = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)
        dof_torques_l2 = torch.sum(torch.square(joint_torque), dim=1)
        dof_acc_l2 = torch.sum(torch.square(joint_acc), dim=1)
        action_rate_l2 = torch.sum(torch.square(self.obs_last_action - rollout_action), dim=1)
        
        flat_orientation_l2 = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
        
        out_of_limits = (joint_pos < self.dof_pos_limits[:, 0]) | (joint_pos > self.dof_pos_limits[:, 1])
        dof_pos_limits = torch.sum(out_of_limits.float(), dim=1)

        # Update dict: (Keys must match reward_term_weights in Config)
        self.imagination_reward_per_step = {
            "track_lin_vel_xy_exp": track_lin_vel_xy_exp,
            "track_ang_vel_z_exp": track_ang_vel_z_exp,
            "lin_vel_z_l2": lin_vel_z_l2,
            "ang_vel_xy_l2": ang_vel_xy_l2,
            "dof_torques_l2": dof_torques_l2,
            "dof_acc_l2": dof_acc_l2,
            "action_rate_l2": action_rate_l2,
            "feet_air_time": torch.zeros_like(track_lin_vel_xy_exp), # Placeholder if not computing contact timing
            "undesired_contacts": torch.sum(thigh_contact, dim=1) if thigh_contact is not None else torch.zeros_like(track_lin_vel_xy_exp),
            "stand_still": torch.zeros_like(track_lin_vel_xy_exp), # Placeholder
            "flat_orientation_l2": flat_orientation_l2,
            "dof_pos_limits": dof_pos_limits,
        }

    def _apply_interval_events(self, imagination_states_denormalized, parsed_imagination_states, event_ids):
        if len(event_ids) > 0:
            base_lin_vel = parsed_imagination_states["base_lin_vel"]
            velocity_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
            r = torch.empty(len(event_ids), device=self.device)
            base_lin_vel[event_ids, 0] += r.uniform_(*velocity_range["x"])
            base_lin_vel[event_ids, 1] += r.uniform_(*velocity_range["y"])
            imagination_states_denormalized[event_ids, 0:3] = base_lin_vel[event_ids, 0:3]
            
            # Re-normalize just the modified states (or return raw if base handles it)
            if hasattr(self.dataset, 'normalize'):
                 input_tensor, _ = self.dataset.normalize(imagination_states_denormalized, None)
            else:
                 input_tensor = imagination_states_denormalized
            return input_tensor
            
        # Return full normalized state if no events
        normalized_state, _ = self.dataset.normalize(imagination_states_denormalized, None)
        return normalized_state

    @property
    def state_dim(self):
        # 3(Lin) + 3(Ang) + 3(Grav) + 12(Pos) + 12(Vel) = 33
        # return 33
        return 45
    
    @property
    def observation_dim(self):
        # 48 Dims for Policy
        return 48
    
    @property
    def action_dim(self):
        return 12