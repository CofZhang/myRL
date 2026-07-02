from legged_gym.utils.task_registry import task_registry

from zof_gym.envs.zof_robot import ZofRobot
from zof_gym.envs.zof_config import ZofFlatCfg, ZofFlatCfgPPO


task_registry.register(
    "zof_flat",
    ZofRobot,
    ZofFlatCfg(),
    ZofFlatCfgPPO(),
)
