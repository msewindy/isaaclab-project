import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.assets import RigidObjectCfg


# RealSense D455配置
REALSENSE_D455_CFG = RigidObjectCfg(
    spawn = sim_utils.UsdFileCfg(
        usd_path=f"omniverse://localhost/NVIDIA/Assets/Isaac/4.2/Isaac/Sensors/Intel/RealSense/rsd455.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
        ),
        activate_contact_sensors=False  
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos = (0.5, -0.9, 0.8),
        rot = (0.6532815, -0.27059805, 0.27059805, 0.6532815),
    )
)