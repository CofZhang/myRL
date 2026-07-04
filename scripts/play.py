import os
import sys

MYRL_ROOT = "/home/zhz/Desktop/myRL"
if MYRL_ROOT not in sys.path:
    sys.path.insert(0, MYRL_ROOT)

import isaacgym  # noqa: F401
import legged_gym.envs  # noqa: F401
import zof_gym.envs  # noqa: F401

import argparse

import torch

from legged_gym.utils import get_args, task_registry


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = 16
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

    print(f"[play] command: vx={args.cmd_vx:.3f}, vy=0.000, wz={args.cmd_wz:.3f}")

    for _ in range(10 * int(env.max_episode_length)):
        env.commands[:, 0] = args.cmd_vx
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = args.cmd_wz

        env.compute_observations()
        obs = env.get_observations()

        with torch.no_grad():
            actions = policy(obs.detach())
            obs, _, _, _, _ = env.step(actions.detach())

 
def parse_play_command_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cmd_vx", type=float, default=0.5)
    parser.add_argument("--cmd_wz", type=float, default=0.0)
    play_args, remaining = parser.parse_known_args()

    # legged_gym 的 get_args() 从 sys.argv 解析；移除自定义参数后再交给它。
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    args.cmd_vx = play_args.cmd_vx
    args.cmd_wz = play_args.cmd_wz
    return args


if __name__ == "__main__":
    args = parse_play_command_args()

    if args.task == "anymal_c_flat":
        args.task = "zof_flat"

    play(args)
