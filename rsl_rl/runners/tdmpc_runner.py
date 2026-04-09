from rsl_rl.runners.online_planning_mbrl_runner import OnlinePlanningMBRLRunner


class TDMPCRunner(OnlinePlanningMBRLRunner):
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        
        super().__init__(env, train_cfg, log_dir, device)