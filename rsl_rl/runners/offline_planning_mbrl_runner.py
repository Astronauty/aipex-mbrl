import torch
from rsl_rl.runners.mbrl_runner import MBRLRunner

class OnlinePlanningMBRLRunner(MBRLRunner):
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):

        super().__init__(env, train_cfg, log_dir, device)

        # Check if train cfg contains algorithm
        if "algorithm" in self.cfg:
            self.alg_cfg = self.cfg["algorithm"]
        else:
            raise ValueError("Training configuration file not compatible with offline planning MBRL runner. " \
                             "Please make sure to include an 'algorithm' key in the training configuration file.")

        pass

    
    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        """Load the models and training state from a given path.

        Args:
            path (str): Path to load the model from.
            load_cfg (dict | None): Optional dictionary that defines what models and states to load. If None, all
                models and states are loaded.
            strict (bool): Whether state_dict loading should be strict.
            map_location (str | None): Device mapping for loading the model.
        """
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]