from pathlib import Path

import torch


POLICY_PATH = Path(__file__).resolve().parents[1] / "policies" / "zof_wasd_policy.pt"


def main():
    print("policy:", POLICY_PATH)

    policy = torch.jit.load(str(POLICY_PATH), map_location="cpu")
    policy.eval()

    obs = torch.zeros(1, 45, dtype=torch.float32)
    obs[0, 3:6] = torch.tensor([0.0, 0.0, -1.0])  # projected gravity
    obs[0, 6:9] = torch.tensor([0.0, 0.0, 0.0])   # stand command

    with torch.no_grad():
        action = policy(obs)

    print("obs shape:", tuple(obs.shape))
    print("action shape:", tuple(action.shape))
    print("action:", action.numpy())
    print("action min:", action.min().item())
    print("action max:", action.max().item())
    print("has nan:", torch.isnan(action).any().item())


if __name__ == "__main__":
    main()

