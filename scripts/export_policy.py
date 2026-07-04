import os
import shutil
import sys

MYRL_ROOT = "/home/zhz/Desktop/myRL"
if MYRL_ROOT not in sys.path:
    sys.path.insert(0, MYRL_ROOT)

import isaacgym  # noqa: F401
import legged_gym.envs  # noqa: F401
import zof_gym.envs  # noqa: F401

from legged_gym.utils import export_policy_as_jit, get_args, task_registry


def export_policy(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = 1
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

    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        log_root=os.path.join(MYRL_ROOT, "runs", args.task),
    )

    load_run = str(args.load_run)
    checkpoint = str(args.checkpoint)
    export_dir = os.path.join(
        MYRL_ROOT,
        "resources",
        "policies",
        args.task,
        load_run,
        f"checkpoint_{checkpoint}",
    )
    os.makedirs(export_dir, exist_ok=True)

    export_policy_as_jit(runner.alg.actor_critic, export_dir)

    src = os.path.join(export_dir, "policy_1.pt")
    dst = os.path.join(export_dir, "policy.pt")
    if os.path.exists(src):
        shutil.copyfile(src, dst)

    print("Exported policy:", dst)
    print("Observation dim:", env.num_obs)
    print("Action dim:", env.num_actions)
    print("Controller note: policy outputs RL residuals; deployment must also run zof IK gait reference.")


if __name__ == "__main__":
    args = get_args()

    if args.task == "anymal_c_flat":
        args.task = "zof_flat"

    export_policy(args)
