from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class ZofFlatCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 1024
        num_observations = 45
        num_privileged_obs = None
        num_actions = 12
        env_spacing = 2.0
        episode_length_s = 20

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        measure_heights = False
        curriculum = False

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]

        default_joint_angles = {
            "FL_hip_joint": -0.1,
            "FL_thigh_joint": 1.1,
            "FL_calf_joint": -1.5,

            "FR_hip_joint": 0.1,
            "FR_thigh_joint": 1.1,
            "FR_calf_joint": -1.5,

            "RL_hip_joint": -0.1,
            "RL_thigh_joint": 1.3,
            "RL_calf_joint": -1.5,

            "RR_hip_joint": 0.1,
            "RR_thigh_joint": 1.3,
            "RR_calf_joint": -1.5,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"

        stiffness = {
            "hip_joint": 20.0,
            "thigh_joint": 20.0,
            "calf_joint": 20.0,
        }

        damping = {
            "hip_joint": 0.5,
            "thigh_joint": 0.5,
            "calf_joint": 0.5,
        }

        # In ZofRobot actions are residuals around an IK gait reference.
        # The residual amplitude is configured in gait.*_residual_scale.
        action_scale = 1.0
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "/home/zhz/Desktop/myRL/resources/robots/zof/urdf/zof.urdf"
        name = "zof"
        foot_name = "foot"

        penalize_contacts_on = ["hip", "thigh", "calf"]
        terminate_after_contacts_on = ["base"]

        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False

        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05

        clip_observations = 100.0
        clip_actions = 100.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            ang_vel = 0.2
            gravity = 0.05
            dof_pos = 0.01
            dof_vel = 1.5

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = True
        tracking_sigma = 0.10
        soft_dof_pos_limit = 0.9
        base_height_target = 0.32
        max_contact_force = 100.0

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 3.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            torques = -0.00002
            dof_acc = -2.5e-7
            action_rate = -0.01
            collision = -1.0
            dof_pos_limits = -10.0
            feet_air_time = 0.15
            stand_still_when_commanded = -2.0
            hip_deviation = -0.8
            hip_abduction_limit = -6.0
            calf_too_straight = -1.5
            long_air_time = -1.0
            ik_residual = -0.02
            foot_bezier_tracking = -3.0
            foot_lateral_deviation = -8.0
            contact_schedule = -0.6

    class gait:
        # Trot reference: FL/RR and FR/RL move as diagonal pairs.
        frequency = 2.2
        duty_factor = 0.58
        phase_offsets = [0.0, 0.5, 0.5, 0.0]

        # Approximate dimensions from the zof URDF. Keep these conservative.
        thigh_length = 0.220
        calf_length = 0.219
        hip_offsets = [
            [0.1403, 0.0498, 0.0],
            [0.1403, -0.0498, 0.0],
            [-0.1403, 0.0498, 0.0],
            [-0.1403, -0.0498, 0.0],
        ]
        foot_side_signs = [1.0, -1.0, 1.0, -1.0]
        foot_lateral_offset = 0.02

        step_length = 0.055
        step_length_command_gain = 0.050
        min_step_length = 0.045
        max_step_length = 0.105
        swing_height = 0.055

        hip_residual_scale = 0.06
        leg_residual_scale = 0.16

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        num_commands = 4
        resampling_time = 10.0
        heading_command = False

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.3, 0.8]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [-3.14, 3.14]


class ZofFlatCfgPPO(LeggedRobotCfgPPO):
    seed = 1

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        activation = "elu"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24
        max_iterations = 1500
        save_interval = 50
        experiment_name = "zof_flat"
        run_name = "ppo_v1"
        resume = False
        load_run = -1
        checkpoint = -1
