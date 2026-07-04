# Policy Export and Deploy Contract

## Current exported policy

- Task: `zof_flat`
- Run: `Jul02_16-33-17_ppo_v5_ik_residual_bezier`
- Checkpoint: `3000`
- Exported policy:
  `/home/zhz/Desktop/myRL/resources/policies/zof_flat/Jul02_16-33-17_ppo_v5_ik_residual_bezier/checkpoint_3000/policy.pt`

## Policy I/O

Input observation: 45 dimensions.

- `0-2`: base angular velocity, body frame, scaled by `obs_scales.ang_vel`
- `3-5`: projected gravity
- `6-8`: command `vx, vy, wz`, scaled by `commands_scale`
- `9-20`: joint position error, `q - default_q`
- `21-32`: joint velocity, scaled by `obs_scales.dof_vel`
- `33-44`: last policy action

Output action: 12 dimensions.

Important: the exported actor outputs **RL residual actions**, not final joint targets.
Deployment must reproduce the same IK gait reference used in `zof_gym/envs/zof_robot.py`:

```text
final_target_q = ik_gait_reference_q + action * residual_scale
pd_tau = kp * (final_target_q - q) - kd * dq
```

## Why this matters

Older policies used direct joint target offsets. The current good-walking policy uses an IK trot reference plus residual RL. If a deployment program loads `policy.pt` but sends the 12 outputs directly to motors as joint targets, the robot will move incorrectly.

## Next step

Build a Python deployment verifier that:

1. loads `policy.pt`,
2. constructs the same 45D observation,
3. computes the same IK gait reference,
4. applies residual action and PD control,
5. runs one robot in simulation before any real robot test.
