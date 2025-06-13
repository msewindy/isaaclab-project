# flake8: noqa
import argparse
from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.assets import AssetBaseCfg
from manipulator_cfg import MANIPULATOR_CFG
from realsense_d455_cfg import REALSENSE_D455_CFG
from omni.isaac.lab.managers import SceneEntityCfg
import numpy as np
import torch
from pxr import Usd, Gf, UsdGeom, UsdShade
from texture_manager import TextureManager
import math
import os
import cv2
from omni.isaac.sensor import Camera


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
         prim_path="/World/ground",
         spawn=sim_utils.GroundPlaneCfg(),
         init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path="file:///home/lfw/下载/table_paper.usda",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    light = AssetBaseCfg(
         prim_path="/World/light",
         spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    robot = MANIPULATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/UR5e")
    #realsense = REALSENSE_D455_CFG.replace(prim_path="{ENV_REGEX_NS}/rsd455")


# 到达指定位置
def reachPosition(robot, joint_pos_des):
    
    # 设置误差阈值
    POS_THRESHOLD = 0.25 # 位置误差阈值（弧度）
    VEL_THRESHOLD = 0.1  # 速度误差阈值（弧度/秒）
    
    # 获取当前关节状态
    current_joint_pos = robot.data.joint_pos.clone()  # 使用joint_pos属性获取当前关节位置
    current_joint_vel = robot.data.joint_vel.clone()  # 使用joint_vel属性获取当前关节速度
    
    # 计算位置误差
    pos_error = torch.abs(current_joint_pos - joint_pos_des)
    
    # 计算速度的绝对值
    vel_abs = torch.abs(current_joint_vel)
    
    # 判断是否所有关节都满足条件
    pos_reached = torch.all(pos_error < POS_THRESHOLD)
    vel_reached = torch.all(vel_abs < VEL_THRESHOLD)
    
    # 添加调试信息
    print(f"[DEBUG] 目标位置: {joint_pos_des}")
    print(f"[DEBUG] 当前位置: {current_joint_pos}")
    print(f"[DEBUG] 位置误差: {pos_error}")
    print(f"[DEBUG] 当前速度: {current_joint_vel}")
    print(f"[DEBUG] 位置到达: {pos_reached}, 速度到达: {vel_reached}")
    
    # 返回是否到达目标位置
    return pos_reached 


def main():  
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)   
    sim.reset()


    



    robot = scene.articulations["robot"]

    

    joint_pos_des_set = [
        [[0.0, -2.2689, -2.3562, -3.4907, -1.5708, 0.0]],
        [[0.0, -2.2689, -1.2217, -3.4907, -1.5708, 0.0]],
        [[0.0, -2.5000, -1.2217, -3.4907, -1.5708, 0.0]],
        [[0.0, -2.5000, -2.3562, -3.4907, -1.5708, 0.0]]
    ]
    joint_pos_des_set = torch.tensor(joint_pos_des_set, device=sim.device)

    current_goal_idx = 0
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["tool0"])
    robot_entity_cfg.resolve(scene)

    print("[INFO]: Setup complete...")


    # 初始化机器人状态
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
    robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
    scene.write_data_to_sim()

  

    #从初始位置开始运行robot
    joint_pos_des = joint_pos_des_set[current_goal_idx][:,robot_entity_cfg.joint_ids].clone()
    # 更新机器人状态
    robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
    scene.write_data_to_sim()    
    # 执行仿真步骤
    sim.step()
    scene.update(sim.get_physics_dt())

    while simulation_app.is_running():  
        if reachPosition(robot, joint_pos_des_set[current_goal_idx][:,robot_entity_cfg.joint_ids].clone()):
             current_goal_idx = (current_goal_idx + 1) % len(joint_pos_des_set)
             joint_pos_des = joint_pos_des_set[current_goal_idx][:,robot_entity_cfg.joint_ids].clone() 
             print("[INFO] current_goal_idx is " + str(current_goal_idx))
             # 更新机器人状态
             robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
             scene.write_data_to_sim()             
             current_goal_idx += 1
             if current_goal_idx % 4 == 0:
                 current_goal_idx = 0
        sim.step()
        scene.update(sim.get_physics_dt())

if __name__ == "__main__":
    main()
    simulation_app.close()