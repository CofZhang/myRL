import numpy as np


JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]

DEFAULT_Q = np.array([
    -0.1, 1.1, -1.5,
     0.1, 1.1, -1.5,
    -0.1, 1.3, -1.5,
     0.1, 1.3, -1.5,
], dtype=np.float32)

RESIDUAL_SCALE = np.array([
    0.06, 0.16, 0.16,
    0.06, 0.16, 0.16,
    0.06, 0.16, 0.16,
    0.06, 0.16, 0.16,
], dtype=np.float32)

TORQUE_LIMITS = np.array([
    23.7, 23.7, 35.55,
    23.7, 23.7, 35.55,
    23.7, 23.7, 35.55,
    23.7, 23.7, 35.55,
], dtype=np.float32)

OBS_ANG_VEL_SCALE = 0.25
OBS_DOF_POS_SCALE = 1.0
OBS_DOF_VEL_SCALE = 0.05
CMD_LIN_VEL_SCALE = 2.0
CMD_ANG_VEL_SCALE = 0.25

POLICY_HZ = 50.0
POLICY_DT = 1.0 / POLICY_HZ

# Conservative first real-robot command limits.
CMD_VX_LIMIT = 0.2
CMD_WZ_LIMIT = 0.2

# MuJoCo baseline was kp=40, kd=1. Start lower on real hardware.
REAL_KP_INIT = 10.0
REAL_KD_INIT = 0.3

