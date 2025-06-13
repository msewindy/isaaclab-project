# flake8: noqa


import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.assets import AssetBaseCfg
from .manipulator_cfg import MANIPULATOR_CFG
from .realsense_d455_cfg import REALSENSE_D455_CFG
from omni.isaac.lab.managers import SceneEntityCfg
import numpy as np
import torch
from pxr import Usd, Gf, UsdGeom, UsdShade
from .texture_manager import TextureManager
import math
import os




@configclass
class SceneCfg(InteractiveSceneCfg):

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="file:///home/lfw/下载/table_paper.usda",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )


    robot = MANIPULATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/UR5e")
    #realsense = REALSENSE_D455_CFG.replace(prim_path="{ENV_REGEX_NS}/rsd455")
