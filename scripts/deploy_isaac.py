"""
deploy_isaac.py — 基于训练环境的部署验证程序
=============================================
和 play.py 的区别：
  - 不走 ppo_runner，直接 torch.jit.load("policy.pt")
  - 单机器人，headless 或 viewer 均可
  - 打印每步高度/速度，方便调试

用法：
    conda activate /home/zhz/pan1/leggedgym
    cd /home/zhz/Desktop/myRL
    python scripts/deploy_isaac.py --task zof_flat --headless
    python scripts/deploy_isaac.py --task zof_flat          # 带 viewer（需 X11）
"""

import os
import sys

MYRL_ROOT = "/home/zhz/Desktop/myRL"
if MYRL_ROOT not in sys.path:
    sys.path.insert(0, MYRL_ROOT)

# isaacgym 必须在 torch 之前 import
import isaacgym        # noqa: F401
import legged_gym.envs # noqa: F401
import zof_gym.envs    # noqa: F401

import torch
from legged_gym.utils import get_args, task_registry

POLICY_PATH = os.path.join(
    MYRL_ROOT,
    "resources", "policies", "zof_flat",
    "Jul02_16-33-17_ppo_v5_ik_residual_bezier",
    "checkpoint_3000", "policy.pt",
)


def deploy(args):
    # ------------------------------------------------------------------
    # 1. 建环境（复用训练环境，保证所有参数和训练完全一致）
    # ------------------------------------------------------------------
    env_cfg, _ = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = 1          # 只要一个机器人
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.measure_heights = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # ------------------------------------------------------------------
    # 2. 加载 policy.pt（不需要 ppo_runner）
    # ------------------------------------------------------------------
    print(f"[deploy] 加载策略: {POLICY_PATH}")
    policy = torch.jit.load(POLICY_PATH, map_location=env.device)
    policy.eval()

    # ------------------------------------------------------------------
    # 3. 运行循环
    # ------------------------------------------------------------------
    obs = env.get_observations()
    last_action = torch.zeros(1, 12, device=env.device)

    steps = 10 * int(env.max_episode_length)
    print(f"[deploy] 开始运行 {steps} 步 ({steps * env.dt:.1f} s 仿真时间)")

    for step in range(steps):
        # 固定速度指令：向前 0.5 m/s
        env.commands[:, 0] = 0.5
        env.commands[:, 1] = 0.0
        env.commands[:, 2] = 0.0

        # 策略推理
        with torch.no_grad():
            actions = policy(obs.detach())

        # 执行一步（内部包含 IK 基准 + 残差 + PD + 物理步）
        obs, _, _, _, _ = env.step(actions.detach())

        # 每 100 步打印一次状态
        if (step + 1) % 100 == 0:
            h  = env.root_states[0, 2].item()
            vx = env.base_lin_vel[0, 0].item()
            print(f"  step={step+1:4d}  height={h:.3f} m  vx={vx:.3f} m/s")

    print("[deploy] 完成")


if __name__ == "__main__":
    args = get_args()
    args.task = "zof_flat"
    deploy(args)
