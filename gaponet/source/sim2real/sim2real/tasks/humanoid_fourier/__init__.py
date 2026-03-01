import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Humanoid-Operator-Delta-Action-Fourior",
    entry_point=f"{__name__}.humanoid_operator_env_fourior:HumanoidOperatorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_operator_env_cfg_fourior:HumanoidOperatorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_operator_cfg:HumanoidOperatorFourierRunnerCfg",
    },
)

