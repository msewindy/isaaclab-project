import gymnasium as gym

from . import agents

gym.register(
        id="Comnova-manipulator-draw-v1",
    entry_point=f"{__name__}.manipulator_draw_env_cfg_3:ManipulatorDrawEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.manipulator_draw_env_cfg_3:ManipulatorDrawEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)