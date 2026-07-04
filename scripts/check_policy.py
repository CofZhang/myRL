import torch
from pathlib import Path

POLICY_PATH = Path(
    "/home/zhz/Desktop/myRL/resources/policies/zof_flat/"
    "Jul02_16-33-17_ppo_v5_ik_residual_bezier/"
    "checkpoint_3000/policy.pt"
)


def main():
    print("Loading policy:", POLICY_PATH)

    policy = torch.jit.load(str(POLICY_PATH), map_location="cpu")
    policy.eval()

    obs = torch.zeros(3, 45, dtype=torch.float32)

    # 假设机器人是正常直立状态
    obs[:, 3:6] = torch.tensor([0.0, 0.0, -1.0])

    # 三种命令：站住、慢速前进、较快前进
    obs[0, 6:9] = torch.tensor([0.0, 0.0, 0.0])
    obs[1, 6:9] = torch.tensor([0.3, 0.0, 0.0])
    obs[2, 6:9] = torch.tensor([0.8, 0.0, 0.0])

    with torch.no_grad():
        action = policy(obs)

    print("obs shape:", obs.shape)
    print("action shape:", action.shape)
    print("action:", action.numpy())
    print("action min:", action.min().item())
    print("action max:", action.max().item())
    print("has nan:", torch.isnan(action).any().item())

    joint_names = [
        "FL_hip", "FL_thigh", "FL_calf",
        "FR_hip", "FR_thigh", "FR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
    ]

    residual_scale = torch.tensor([
        0.06, 0.16, 0.16,
        0.06, 0.16, 0.16,
        0.06, 0.16, 0.16,
        0.06, 0.16, 0.16,
    ])

    # 先看第二组命令 vx=0.3 对应的 action
    a = action[1]
    delta_q = a * residual_scale

    print()
    print("Action table for command vx=0.3:")
    for i, name in enumerate(joint_names):
        print(
            f"{i:2d} {name:10s} "
            f"action={a[i].item(): .4f} "
            f"delta_q={delta_q[i].item(): .4f} rad"
        )

    default_q = torch.tensor([
        -0.1, 1.1, -1.5,
         0.1, 1.1, -1.5,
        -0.1, 1.3, -1.5,
         0.1, 1.3, -1.5,
    ])

    target_q = default_q + delta_q

    print()
    print("Target q if reference_q = default_q:")
    for i, name in enumerate(joint_names):
        print(
            f"{i:2d} {name:10s} "
            f"default={default_q[i].item(): .4f} "
            f"delta={delta_q[i].item(): .4f} "
            f"target={target_q[i].item(): .4f} rad"
        )

    kp = 20.0
    kd = 0.5

    current_q = default_q.clone()
    current_dq = torch.zeros(12)

    torque_limits = torch.tensor([
        23.7, 23.7, 35.55,
        23.7, 23.7, 35.55,
        23.7, 23.7, 35.55,
        23.7, 23.7, 35.55,
    ])

    torque = kp * (target_q - current_q) - kd * current_dq
    torque_clipped = torch.clamp(torque, -torque_limits, torque_limits)

    print()
    print("PD torque if current_q = default_q, current_dq = 0:")
    for i, name in enumerate(joint_names):
        clipped = abs(torque[i].item() - torque_clipped[i].item()) > 1e-6
        print(
            f"{i:2d} {name:10s} "
            f"torque={torque[i].item(): .4f} Nm "
            f"limit={torque_limits[i].item(): .2f} "
            f"clipped={clipped}"
        )


if __name__ == "__main__":
    main()
