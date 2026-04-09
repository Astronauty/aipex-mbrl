from .base_cfg import BaseConfig
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Go2FlatConfig(BaseConfig):
    experiment_name: str = "offline_go2"
    
    @dataclass
    class ExperimentConfig(BaseConfig.ExperimentConfig):
        environment: str = "go2_flat" # This name is used to finding the dataset folder later
    
    @dataclass
    class EnvironmentConfig(BaseConfig.EnvironmentConfig):
        # Weights for the rewards calculated in the MBRL Env class (Step 1)
        reward_term_weights: Dict[str, float] = field(default_factory=lambda: {
            "track_lin_vel_xy_exp": 1.5,
            "track_ang_vel_z_exp": 0.8,
            "lin_vel_z_l2": -2.0,
            "ang_vel_xy_l2": -0.05,
            # "joint_torques_l2": -2.5e-5, # Uncomment if you added torque prediction
            "dof_acc_l2": -2.5e-7,
            "action_rate_l2": -0.01,
            "feet_air_time": 0.5,
            "undesired_contacts": -1.0,
            "stand_still": -1.0,
            "flat_orientation_l2": -5.0,
            "dof_pos_limits": -1.0,
        })
        uncertainty_penalty_weight: float = -0.5 # Penalty for visiting states where the model is unsure
        
        # Resampling ranges
        command_resample_interval_range: List[int] | None = field(default_factory=lambda: [100, 150])
        event_interval_range: List[int] = field(default_factory=lambda: [50, 100])
    
    @dataclass
    class DataConfig(BaseConfig.DataConfig):
        dataset_root: str = "assets"
        dataset_folder: str = "data" # Store your collected CSVs here
        batch_data_size: int = 10000
        
        # DEFINES THE STATE VECTOR STRUCTURE
        # Must match Step 1 indices and Step 2 ObsGroup order!
        state_idx_dict = {
            "base_ang_vel": range(0, 3),
            "projected_gravity": range(3, 6),
            "commands": range(6, 9),
            "joint_pos": range(9, 21),
            "joint_vel": range(21, 33),
            "last_action": range(33, 45)
        }
        # NORMALIZATION STATS (Estimated - Update after data collection!)
        # 33 Dimensions Total
        # state_data_mean: List[float] = field(default_factory=lambda: [
        #     0.0, 0.0, 0.0,       # Lin Vel
        #     0.0, 0.0, 0.0,       # Ang Vel
        #     0.0, 0.0, -1.0,      # Gravity (Z is usually -1 in projected frame on flat ground)
        #     # Joint Pos (Relative to default, so mean approx 0)
        #     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        #     # Joint Vel (Mean approx 0)
        #     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        # ])
        state_data_mean: List[float] = field(default_factory=lambda: [
            0.0, 0.0, 0.0,       # base_ang_vel (3)
            0.0, 0.0, -1.0,      # projected_gravity (3)
            0.0, 0.0, 0.0,       # commands (3)
            # joint_pos (12)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # joint_vel (12)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            # last_action (12)
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])
        # state_data_std: List[float] = field(default_factory=lambda: [
        #     1.0, 1.0, 1.0,       # Lin Vel
        #     1.0, 1.0, 1.0,       # Ang Vel
        #     0.1, 0.1, 0.1,       # Gravity (Small variance)
        #     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # Joint Pos
        #     2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, # Joint Vel
        # ])

        state_data_std: List[float] = field(default_factory=lambda: [
            1.0, 1.0, 1.0,       # base_ang_vel
            0.1, 0.1, 0.1,       # projected_gravity
            1.0, 1.0, 1.0,       # commands
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # joint_pos
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, # joint_vel
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # last_action
        ])
        
        # Action stats (Unitree SDK usually clips to specific ranges, standard RL uses [-1, 1])
        action_data_mean: List[float] = field(default_factory=lambda: [0.0] * 12)
        action_data_std: List[float] = field(default_factory=lambda: [1.0] * 12)

    @dataclass
    class ModelArchitectureConfig(BaseConfig.ModelArchitectureConfig):
        history_horizon: int = 15 # History window input
        forecast_horizon: int = 5 # Prediction into future
        ensemble_size: int = 5    # Number of models in ensemble
        
        # Configure output shapes
        contact_dim: int = 8      # 4 feet + 4 thighs
        termination_dim: int = 1  # Base contact
        
        architecture_config: Dict[str, object] = field(default_factory=lambda: {
            "type": "rnn",
            "rnn_type": "gru",
            "rnn_num_layers": 2,
            "rnn_hidden_size": 256,
            "state_mean_shape": [128],    # Latent size
            "state_logstd_shape": [128],
            "extension_shape": [128],
            "contact_shape": [128],
            "termination_shape": [128],
        })
        # resume_path: str | None = None # Set this if restarting training
        resume_path: str | None = "rsl_rl/2026-03-02_22-19-28/model_5000.pt"

    @dataclass
    class PolicyArchitectureConfig(BaseConfig.PolicyArchitectureConfig):
        # Change 33 to 45
        observation_dim: int = 45 + 15  # State dim + History/Latent details usually
        action_dim: int = 12
        resume_path: str | None = None

    @dataclass
    class PolicyAlgorithmConfig(BaseConfig.PolicyAlgorithmConfig):
        learning_rate: float = 3e-4 # Standard PPO LR
        entropy_coef: float = 0.001

    @dataclass
    class PolicyTrainingConfig(BaseConfig.PolicyTrainingConfig):
        save_interval: int = 10
        max_iterations: int = 1000 # Short run for testing
    
    experiment_config: ExperimentConfig = field(default_factory=ExperimentConfig)
    environment_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    model_architecture_config: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    policy_architecture_config: PolicyArchitectureConfig = field(default_factory=PolicyArchitectureConfig)
    policy_algorithm_config: PolicyAlgorithmConfig = field(default_factory=PolicyAlgorithmConfig)
    policy_training_config: PolicyTrainingConfig = field(default_factory=PolicyTrainingConfig)