from isaacgym import gymapi
import os


MYRL_ROOT = "/home/zhz/Desktop/myRL"

ASSET_ROOT = os.path.join(MYRL_ROOT, "resources", "robots", "zof")
ASSET_FILE = "urdf/zof.urdf"



def create_sim():
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt = 0.005 #仿真步长200Hz
    sim_params.substeps = 1
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = False

    compute_device_id = 0
    graphics_device_id = 0
    physics_engine = gymapi.SIM_PHYSX

    sim = gym.create_sim(
        compute_device_id,
        graphics_device_id,
        physics_engine,
        sim_params,
    )

    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym sim")

    return gym, sim


def load_robot_asset(gym, sim):
    asset_options = gymapi.AssetOptions()

    asset_options.default_dof_drive_mode =int(gymapi.DOF_MODE_EFFORT)
    asset_options.collapse_fixed_joints = True
    asset_options.replace_cylinder_with_capsule = True
    asset_options.flip_visual_attachments = True
    asset_options.fix_base_link = False
    asset_options.disable_gravity = False

    asset_options.density = 0.001
    asset_options.angular_damping = 0.0
    asset_options.linear_damping = 0.0
    asset_options.max_angular_velocity = 1000.0
    asset_options.max_linear_velocity = 1000.0
    asset_options.armature = 0.0
    asset_options.thickness = 0.01

    robot_asset = gym.load_asset(
        sim,
        ASSET_ROOT,
        ASSET_FILE,
        asset_options,
    )

    if robot_asset is None:
        raise RuntimeError("Failed to load robot asset")

    num_dofs = gym.get_asset_dof_count(robot_asset)
    num_bodies = gym.get_asset_rigid_body_count(robot_asset)

    print("Robot asset loaded successfully")
    print("num_dofs:", num_dofs)
    print("num_bodies:", num_bodies)

    return robot_asset


def print_body_names(gym, robot_asset):
    body_names = gym.get_asset_rigid_body_names(robot_asset)

    print("\nBody names:")
    for i, name in enumerate(body_names):
        print(f"{i}: {name}")

    foot_names = [name for name in body_names if "foot" in name]
    base_names = [name for name in body_names if "base" in name]

    print("\nDetected foot bodies:")
    for name in foot_names:
        print(name)

    print("\nDetected base bodies:")
    for name in base_names:
        print(name)

    if len(foot_names) != 4:
        print("WARNING: expected 4 foot bodies, got", len(foot_names))
    else:
        print("Foot body count OK")

    if len(base_names) < 1:
        print("WARNING: no base body found")
    else:
        print("Base body found")


def print_dof_limits(gym, robot_asset):
    dof_names = gym.get_asset_dof_names(robot_asset)
    dof_props = gym.get_asset_dof_properties(robot_asset)

    print("\nDOF limits:")
    for i, name in enumerate(dof_names):
        lower = dof_props["lower"][i]
        upper = dof_props["upper"][i]
        effort = dof_props["effort"][i]
        velocity = dof_props["velocity"][i]

        print(
            f"{i}: {name} "
            f"lower={lower:.4f}, "
            f"upper={upper:.4f}, "
            f"effort={effort:.4f}, "
            f"velocity={velocity:.4f}"
        )


def print_dof_names(gym, robot_asset):
    dof_names = gym.get_asset_dof_names(robot_asset)

    print("\nDOF names:")
    for i, name in enumerate(dof_names):
        print(f"{i}: {name}")

    expected_dof_names = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ]

    print("\nCompare with expected order:")
    if list(dof_names) == expected_dof_names:
        print("DOF order OK")
    else:
        print("DOF order MISMATCH")
        print("Expected:")
        for i, name in enumerate(expected_dof_names):
            print(f"{i}: {name}")


def main():
    print("MYRL_ROOT:", MYRL_ROOT)
    print("ASSET_ROOT:", ASSET_ROOT)
    print("ASSET_FILE:", ASSET_FILE)

    gym, sim = create_sim()
    print("Isaac Gym sim created successfully")

    robot_asset = load_robot_asset(gym, sim)
    print_dof_names(gym, robot_asset)
    print_body_names(gym, robot_asset)
    print_dof_limits(gym, robot_asset)

    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
