import mujoco
from pathlib import Path

XML_PATH = Path("/home/zhz/Desktop/myRL/resources/robots/zof/xml/zof_deploy_from_urdf.xml")


def get_name(model, obj_type, obj_id):
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name if name is not None else "<unnamed>"


def main():
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))

    print("XML:", XML_PATH)
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)
    print("njnt:", model.njnt)
    print()

    print("Joints:")
    for jid in range(model.njnt):
        name = get_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        jtype = model.jnt_type[jid]
        qpos_adr = model.jnt_qposadr[jid]
        dof_adr = model.jnt_dofadr[jid]
        print(
            f"{jid:2d} "
            f"name={name:20s} "
            f"type={jtype} "
            f"qpos_adr={qpos_adr:2d} "
            f"dof_adr={dof_adr:2d}"
        )

    print()
    print("Actuators:")
    for aid in range(model.nu):
        actuator_name = get_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
        joint_id = model.actuator_trnid[aid, 0]
        joint_name = get_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        print(f"{aid:2d} name={actuator_name:20s} joint={joint_name}")

    print()
    print("Geoms that may be feet:")
    for gid in range(model.ngeom):
        geom_name = get_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        body_id = model.geom_bodyid[gid]
        body_name = get_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        geom_type = model.geom_type[gid]
        size = model.geom_size[gid]

        text = (geom_name + " " + body_name).lower()
        if "foot" in text or "calf" in text or "sphere" in text:
            print(
                f"{gid:2d} "
                f"geom={geom_name:24s} "
                f"body={body_name:20s} "
                f"type={geom_type} "
                f"size={size}"
            )


if __name__ == "__main__":
    main()
