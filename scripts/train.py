import os
import sys

MYRL_ROOT = "/home/zhz/Desktop/myRL"
if MYRL_ROOT not in sys.path:
    sys.path.insert(0, MYRL_ROOT)

import isaacgym  # noqa: F401
import legged_gym.envs  # noqa: F401
import zof_gym.envs  # noqa: F401

from legged_gym.utils import get_args, task_registry


def train(args):
    env, _ = task_registry.make_env(name=args.task, args=args)

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        log_root=os.path.join(MYRL_ROOT, "runs", args.task),
    )

    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    args = get_args()

    if args.task == "anymal_c_flat":
        args.task = "zof_flat"

    train(args)
