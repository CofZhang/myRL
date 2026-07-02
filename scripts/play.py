import os
import sys

MYRL_ROOT = "/home/zhz/Desktop/myRL"
if MYRL_ROOT not in sys.path:
    sys.path.insert(0, MYRL_ROOT)

import isaacgym  # noqa: F401
import legged_gym.envs  # noqa: F401
import zof_gym.envs  # noqa: F401

import torch

from legged_gym.utils import get_args, task_registry


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 64)
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.measure_heights = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(
        name=args.task,
        args=args,
        env_cfg=env_cfg,
    )

    obs = env.get_observations()

    train_cfg.runner.resume = True

    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=os.path.join(MYRL_ROOT, "runs", args.task),
    )

    policy = ppo_runner.get_inference_policy(device=env.device)

    for _ in range(10 * int(env.max_episode_length)):
        env.commands[:, 0] = 0.5
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0

        env.compute_observations()
        obs = env.get_observations()

        with torch.no_grad():
            actions = policy(obs.detach())
            obs, _, _, _, _ = env.step(actions.detach())


if __name__ == "__main__":
    args = get_args()

    if args.task == "anymal_c_flat":
        args.task = "zof_flat"

    play(args)
