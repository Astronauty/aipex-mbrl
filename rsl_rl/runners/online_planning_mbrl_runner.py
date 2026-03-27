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

    def
