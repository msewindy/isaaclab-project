import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg


MANIPULATOR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"file:///home/lfw/下载/ur5e_peg_2.usda",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0, -0, 0),
        rot=(1, 0, 0, 0),
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -2.15,
            "elbow_joint": -2.007,
            "wrist_1_joint": -3.4907, 
            "wrist_2_joint": -1.5708,
            "wrist_3_joint": 0.0,
        },
        joint_vel={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": 0.0,
            "elbow_joint": 0.0,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=10.0,
            effort_limit=80.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)