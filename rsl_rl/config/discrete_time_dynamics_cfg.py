from dataclasses import MISSING

class DiscreteTimeDynamicsModelCfg:
    """Configuration for the system dynamics networks."""
    obs_dim : int = MISSING
    action_dim : int = MISSING
    history_horizon : int = MISSING # Prior number of timesteps for states and actions
    prediction_horizon : int = MISSING # Number of timesteps to predict into the future for states

    