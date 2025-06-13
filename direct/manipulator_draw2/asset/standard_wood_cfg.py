import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

STANDARDWOOD_CFG=RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"omniverse://localhost/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/Shipping/Wood_Crates/Standard_A/StandardWoodCrate_A27_60x50x50m_PR_NV_01.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        activate_contact_sensors=False,
        scale=(0.027, 0.038, 0.01)
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.5000000074505806, 0.5000000074505806, 0),
        rot=(6.123234e-17, 1, 0, 0),
    )
)