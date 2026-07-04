import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym.envs.base.legged_robot import LeggedRobot


class ZofRobot(LeggedRobot):
    def _init_buffers(self):
        super()._init_buffers()

        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies, 13
        )

        self.gait_target_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        self.gait_target_foot_pos_body = torch.zeros(
            self.num_envs, 4, 3, dtype=torch.float, device=self.device
        )

        self.leg_phase_offsets = torch.tensor(
            self.cfg.gait.phase_offsets, dtype=torch.float, device=self.device
        ).view(1, 4)
        self.hip_ids = torch.tensor([0, 3, 6, 9], dtype=torch.long, device=self.device)
        self.thigh_ids = torch.tensor([1, 4, 7, 10], dtype=torch.long, device=self.device)
        self.calf_ids = torch.tensor([2, 5, 8, 11], dtype=torch.long, device=self.device)

        self.hip_offsets_body = torch.tensor(
            self.cfg.gait.hip_offsets,
            dtype=torch.float,
            device=self.device,
        ).view(1, 4, 3)
        self.foot_side_sign = torch.tensor(
            self.cfg.gait.foot_side_signs,
            dtype=torch.float,
            device=self.device,
        ).view(1, 4)

        self.residual_scale_vec = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device
        )
        self.residual_scale_vec[self.hip_ids] = self.cfg.gait.hip_residual_scale
        self.residual_scale_vec[self.thigh_ids] = self.cfg.gait.leg_residual_scale
        self.residual_scale_vec[self.calf_ids] = self.cfg.gait.leg_residual_scale

        self.default_dof_pos_1d = self.default_dof_pos[0]
        self.default_foot_x, self.default_foot_z = self._forward_kinematics_xz(
            self.default_dof_pos_1d[self.thigh_ids].view(1, 4),
            self.default_dof_pos_1d[self.calf_ids].view(1, 4),
        )

    def post_physics_step(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        super().post_physics_step()

    def check_termination(self):
        super().check_termination()

        fallen = self.projected_gravity[:, 2] > -0.3
        too_low = self.root_states[:, 2] < 0.18

        self.reset_buf |= fallen
        self.reset_buf |= too_low

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)

        stand_probability = getattr(self.cfg.commands, "stand_probability", 0.0)
        if stand_probability <= 0.0 or len(env_ids) == 0:
            return

        stand = torch.rand(len(env_ids), device=self.device) < stand_probability
        stand_env_ids = env_ids[stand]
        if len(stand_env_ids) > 0:
            self.commands[stand_env_ids, 0:3] = 0.0

    def _stand_command_mask(self):
        return (
            (torch.abs(self.commands[:, 0]) < 0.05)
            & (torch.abs(self.commands[:, 1]) < 0.05)
            & (torch.abs(self.commands[:, 2]) < 0.10)
        )

    def _get_gait_phase(self):
        phase = self.episode_length_buf.float().unsqueeze(1) * self.dt
        phase = phase * self.cfg.gait.frequency + self.leg_phase_offsets
        return torch.remainder(phase, 1.0)

    def _forward_kinematics_xz(self, thigh, calf):
        l1 = self.cfg.gait.thigh_length
        l2 = self.cfg.gait.calf_length
        x = l1 * torch.sin(thigh) + l2 * torch.sin(thigh + calf)
        z = -l1 * torch.cos(thigh) - l2 * torch.cos(thigh + calf)
        return x, z

    def _inverse_kinematics_xz(self, x, z):
        l1 = self.cfg.gait.thigh_length
        l2 = self.cfg.gait.calf_length
        eps = 1e-6

        r = torch.sqrt(torch.square(x) + torch.square(z)).clamp(
            min=abs(l1 - l2) + 1e-4,
            max=l1 + l2 - 1e-4,
        )
        cos_calf = (torch.square(r) - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        cos_calf = torch.clamp(cos_calf, -1.0 + eps, 1.0 - eps)
        calf = -torch.acos(cos_calf)

        cos_alpha = (l1 * l1 + torch.square(r) - l2 * l2) / (2.0 * l1 * r)
        cos_alpha = torch.clamp(cos_alpha, -1.0 + eps, 1.0 - eps)
        thigh = torch.atan2(x, -z) + torch.acos(cos_alpha)
        return thigh, calf

    def _compute_reference_targets(self):
        phase = self._get_gait_phase()
        duty = self.cfg.gait.duty_factor
        swing = phase >= duty

        cmd_x = torch.clamp(self.commands[:, 0], min=0.0).unsqueeze(1)
        stand = self._stand_command_mask().unsqueeze(1)

        step_length = self.cfg.gait.step_length + self.cfg.gait.step_length_command_gain * cmd_x
        step_length = torch.clamp(
            step_length,
            min=self.cfg.gait.min_step_length,
            max=self.cfg.gait.max_step_length,
        )

        stance_s = torch.clamp(phase / duty, 0.0, 1.0)
        swing_s = torch.clamp((phase - duty) / (1.0 - duty), 0.0, 1.0)
        swing_curve = swing_s * swing_s * (3.0 - 2.0 * swing_s)

        x_stance = 0.5 * step_length - step_length * stance_s
        x_swing = -0.5 * step_length + step_length * swing_curve
        z_swing = self.cfg.gait.swing_height * 4.0 * swing_s * (1.0 - swing_s)

        x_delta = torch.where(swing, x_swing, x_stance)
        z_delta = torch.where(swing, z_swing, torch.zeros_like(z_swing))

        # stand mask：命令速度低于阈值时关闭踏步，原地站立
        x_delta = torch.where(stand, torch.zeros_like(x_delta), x_delta)
        z_delta = torch.where(stand, torch.zeros_like(z_delta), z_delta)

        foot_x = self.default_foot_x + x_delta
        foot_z = self.default_foot_z + z_delta
        thigh, calf = self._inverse_kinematics_xz(foot_x, foot_z)

        target = self.default_dof_pos.repeat(self.num_envs, 1).clone()
        target[:, self.thigh_ids] = thigh
        target[:, self.calf_ids] = calf

        lower = self.dof_pos_limits[:, 0].unsqueeze(0)
        upper = self.dof_pos_limits[:, 1].unsqueeze(0)
        target = torch.maximum(torch.minimum(target, upper), lower)

        foot_y = self.foot_side_sign * self.cfg.gait.foot_lateral_offset
        self.gait_target_foot_pos_body[:, :, 0] = self.hip_offsets_body[:, :, 0] + foot_x
        self.gait_target_foot_pos_body[:, :, 1] = self.hip_offsets_body[:, :, 1] + foot_y
        self.gait_target_foot_pos_body[:, :, 2] = self.hip_offsets_body[:, :, 2] + foot_z
        self.gait_target_dof_pos = target
        return target

    def _get_foot_positions_body(self):
        foot_pos_world = self.rigid_body_states[:, self.feet_indices, 0:3]
        rel_pos_world = foot_pos_world - self.root_states[:, None, 0:3]
        quat = self.base_quat[:, None, :].repeat(1, 4, 1).reshape(-1, 4)
        rel_pos = rel_pos_world.reshape(-1, 3)
        return quat_rotate_inverse(quat, rel_pos).view(self.num_envs, 4, 3)

    def _compute_torques(self, actions):
        reference = self._compute_reference_targets()
        residual = actions * self.residual_scale_vec.unsqueeze(0)
        target = reference + residual

        lower = self.dof_pos_limits[:, 0].unsqueeze(0)
        upper = self.dof_pos_limits[:, 1].unsqueeze(0)
        target = torch.maximum(torch.minimum(target, upper), lower)

        torques = self.p_gains * (target - self.dof_pos) - self.d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reward_stand_still_when_commanded(self):
        commanded = self.commands[:, 0] > 0.2
        too_slow = self.base_lin_vel[:, 0] < 0.15
        return (commanded & too_slow).float()

    def _reward_hip_deviation(self):
        hip_error = self.dof_pos[:, self.hip_ids] - self.default_dof_pos_1d[self.hip_ids]
        return torch.sum(torch.square(hip_error), dim=1)

    def _reward_hip_abduction_limit(self):
        hip_error = torch.abs(
            self.dof_pos[:, self.hip_ids] - self.default_dof_pos_1d[self.hip_ids]
        )
        return torch.sum(torch.square(torch.clamp(hip_error - 0.12, min=0.0)), dim=1)

    def _reward_calf_too_straight(self):
        return torch.sum(torch.clamp(self.dof_pos[:, self.calf_ids] + 1.0, min=0.0), dim=1)

    def _reward_long_air_time(self):
        return torch.sum(torch.clamp(self.feet_air_time - 0.55, min=0.0), dim=1)

    def _reward_ik_residual(self):
        return torch.sum(torch.square(self.actions), dim=1)

    def _reward_foot_bezier_tracking(self):
        foot_pos = self._get_foot_positions_body()
        error = foot_pos - self.gait_target_foot_pos_body
        weights = torch.tensor([1.0, 0.25, 1.0], dtype=torch.float, device=self.device)
        return torch.sum(torch.square(error) * weights.view(1, 1, 3), dim=(1, 2))

    def _reward_foot_lateral_deviation(self):
        foot_pos = self._get_foot_positions_body()
        lateral_error = torch.abs(foot_pos[:, :, 1] - self.gait_target_foot_pos_body[:, :, 1])
        return torch.sum(torch.square(torch.clamp(lateral_error - 0.035, min=0.0)), dim=1)

    def _reward_contact_schedule(self):
        phase = self._get_gait_phase()
        expected_contact = phase < self.cfg.gait.duty_factor
        actual_contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        return torch.mean(torch.abs(expected_contact.float() - actual_contact.float()), dim=1)

    def _reward_stand_base_motion(self):
        stand = self._stand_command_mask().float()
        lin_xy = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
        yaw = torch.square(self.base_ang_vel[:, 2])
        return stand * (lin_xy + yaw)

    def _reward_stand_orientation_flat(self):
        stand = self._stand_command_mask().float()
        return stand * torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_stand_joint_posture(self):
        stand = self._stand_command_mask().float()
        error = self.dof_pos - self.default_dof_pos
        return stand * torch.sum(torch.square(error), dim=1)

    def _reward_stand_foot_stance(self):
        stand = self._stand_command_mask().float()
        foot_pos = self._get_foot_positions_body()
        target = self.gait_target_foot_pos_body
        error = foot_pos - target
        weights = torch.tensor([1.0, 0.5, 1.0], dtype=torch.float, device=self.device)
        return stand * torch.sum(torch.square(error) * weights.view(1, 1, 3), dim=(1, 2))

    def _reward_stand_foot_contact(self):
        stand = self._stand_command_mask().float()
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        missing_contacts = 4.0 - torch.sum(contact.float(), dim=1)
        return stand * missing_contacts

    def compute_observations(self):
        self.obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )

        if self.add_noise:
            self.obs_buf += (
                2 * torch.rand_like(self.obs_buf) - 1
            ) * self.noise_scale_vec

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise

        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.0
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0.0

        return noise_vec
