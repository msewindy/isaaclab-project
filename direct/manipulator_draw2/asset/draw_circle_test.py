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

from omni.isaac.debug_draw import _debug_draw
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
# 添加相机相关的导入
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.prims import get_prim_at_path
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



def pos_pen(scene: InteractiveScene, sim, current_time):
    # 获取笔的prim
    pen_prim = scene.stage.GetPrimAtPath('/World/envs/env_0/UR5e/ur5e/tool0/Cone')
    if not pen_prim:
        print("[ERROR] 无法获取笔的prim")
        return None, None
    
    try:
        # 使用Isaac Sim的API获取笔的位置和方向
        robot = scene.articulations["robot"]
        
        # 获取所有链接名称
        link_names = robot.body_names
        print(f"[DEBUG] 机器人链接名称: {link_names}")
        
        # 找到tool0的索引
        tool0_index = None
        for i, name in enumerate(link_names):
            if name == "tool0":
                tool0_index = i
                break
                
        if tool0_index is None:
            print("[ERROR] 未找到tool0链接")
            return None, None
            
        # 获取tool0的世界位置和方向
        world_pos = robot.data.body_pos_w[0, tool0_index].cpu().numpy()  # 获取第一个环境中的tool0位置
        world_quat = robot.data.body_quat_w[0, tool0_index].cpu().numpy()  # 获取第一个环境中的tool0方向
        
        # 将四元数转换为旋转矩阵
        w, x, y, z = world_quat
        world_rot_matrix = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        
        # Cone相对于tool0的局部变换
        cone_local_pos = np.array([0.0, 0.0, 0.05])  # Cone在tool0坐标系中的位置
        cone_local_scale = np.array([0.025, 0.025, 0.1])  # Cone的缩放
        
        # 计算Cone在世界坐标系中的位置
        # 1. 将局部位置从tool0坐标系转换到世界坐标系
        cone_world_pos = world_pos + world_rot_matrix @ cone_local_pos
        
        # 2. 计算Cone在世界坐标系中的方向（与tool0相同，因为Cone没有额外的旋转）
        cone_world_z = world_rot_matrix @ np.array([0.0, 0.0, 1.0])
        
        # 添加更多调试信息
        print(f"[DEBUG] 笔的prim路径: {pen_prim.GetPath()}")
        print(f"[DEBUG] 笔的prim类型: {pen_prim.GetTypeName()}")
        print(f"[DEBUG] 笔的prim是否有效: {pen_prim.IsValid()}")
        print(f"[DEBUG] 笔的prim是否激活: {pen_prim.IsActive()}")
        print(f"[DEBUG] tool0的索引: {tool0_index}")
        print(f"[DEBUG] tool0的世界位置: {world_pos}")
        print(f"[DEBUG] tool0的世界旋转四元数: {world_quat}")
        print(f"[DEBUG] tool0的世界旋转矩阵:\n{world_rot_matrix}")
        print(f"[DEBUG] Cone的局部位置: {cone_local_pos}")
        print(f"[DEBUG] Cone的局部缩放: {cone_local_scale}")
        print(f"[DEBUG] Cone的世界位置: {cone_world_pos}")
        print(f"[DEBUG] Cone的世界方向: {cone_world_z}")
        
        # 获取笔中心在世界坐标系中的位置
        pen_origin = np.array([
            [cone_world_pos[0]],  # x
            [cone_world_pos[1]],  # y
            [cone_world_pos[2]],  # z
            [1.0]
        ])
        
        # 获取笔local坐标系z轴(笔尖方向)在世界坐标系中的方向
        pen_z = np.array([
            [cone_world_z[0]],  # z轴在世界坐标系x分量
            [cone_world_z[1]],  # z轴在世界坐标系y分量
            [cone_world_z[2]]   # z轴在世界坐标系z分量
        ])
        
        # 确保方向向量是单位向量
        pen_z = pen_z / np.linalg.norm(pen_z)
        
        # 添加更多调试信息
        print(f"[DEBUG] 当前时间: {current_time}")
        print(f"[DEBUG] 笔的世界位置(中心点): {pen_origin[:3].T}")
        print(f"[DEBUG] 笔的z轴方向(单位向量): {pen_z.T}")
        
        return pen_origin, pen_z
        
    except Exception as e:
        print(f"[ERROR] 获取笔的位置和方向时发生错误: {str(e)}")
        import traceback
        print(f"[ERROR] 错误堆栈: {traceback.format_exc()}")
        return None, None


def is_angle_greater_than(v1, v2, angle_min=45):  
    # 将向量展平为一维数组
    v1_flat = v1.flatten()
    v2_flat = v2.flatten()
    
    # 计算单位向量
    v1_u = v1_flat / np.linalg.norm(v1_flat)  
    v2_u = v2_flat / np.linalg.norm(v2_flat)  
    
    # 计算点积
    cos_angle = np.dot(v1_u, v2_u)  
    cos_min = np.cos(np.radians(angle_min))  
    
    # 添加调试信息
    print(f"[DEBUG] 向量1: {v1_u}")
    print(f"[DEBUG] 向量2: {v2_u}")
    print(f"[DEBUG] 角度: {np.degrees(np.arccos(cos_angle))}, 最小角度: {angle_min}")
    
    # 修改：放宽角度限制
    return cos_angle > 0.707, cos_angle  # 改为固定阈值0.5，对应约60度


def calculate_point(p0, v, h):  
    v_norm = np.linalg.norm(v)  
    if v_norm == 0:  
        raise ValueError("方向向量 v 不能是零向量。")  
    v_unit = v / v_norm  
    
    # 修改：确保计算的点在正确的方向上
    p = p0[:3] - h * v_unit  # 改为减去高度，使点更靠近桌面
    
    # 添加调试信息
    print(f"[DEBUG] 计算点: 起点={p0[:3]}, 方向={v_unit}, 高度={h}, 结果={p}")
    
    return p


def pen_paper_rad(delta_h):
    # 增大笔迹半径
    return 0.025 * (0.5 - delta_h / 0.1)  # 将0.1改成0.5


def rad_trait(scene: InteractiveScene, sim, current_time):
    pen_origin, pen_z = pos_pen(scene, sim, current_time)
    
    # 修改参考向量，使其与桌面平行
    reference_vector = np.array([[0.0, 0.0, -1.0]]).T  # 改为负z方向
    
    result, cos_angle = is_angle_greater_than(pen_z, reference_vector)
    
    if result:
        # 修改长度计算方式
        length = pen_origin[2][0]  # 使用z坐标的绝对值
        if 0 < length < 0.05:  # 添加高度阈值检查
            pen_trait_center = calculate_point(pen_origin, pen_z, length)
            pen_trait_rad = pen_paper_rad(length)
            
            # 添加调试信息
            print(f"[DEBUG] 计算得到的长度: {length}")
            print(f"[DEBUG] 笔迹中心: {pen_trait_center}")
            print(f"[DEBUG] 笔迹半径: {pen_trait_rad}")
            
            return pen_trait_center, pen_trait_rad
    return None, None


def _get_material_and_texture(prim, scene, time_code):
    """获取材质和纹理信息"""
    print(f"[DEBUG] 开始获取材质和纹理，prim路径: {prim.GetPath()}")
    
    try:
        # 获取材质绑定API
        material_binding_api = UsdShade.MaterialBindingAPI(prim)
        if not material_binding_api:
            print("[WARNING] 无法获取材质绑定API")
            return None, None, None
            
        # 获取所有绑定的材质
        collection = material_binding_api.GetDirectBindingRel()
        if not collection:
            print("[WARNING] 无法获取材质绑定关系")
            return None, None, None
            
        # 获取最强绑定的材质
        targets = collection.GetTargets()
        if not targets or len(targets) == 0:
            print("[WARNING] 未找到绑定的材质")
            return None, None, None
            
        # 获取最后绑定的材质（应该是我们代码中绑定的）
        material_path = targets[-1]
        bound_material = UsdShade.Material(prim.GetStage().GetPrimAtPath(material_path))
        if not bound_material:
            print("[WARNING] 无法获取绑定的材质")
            return None, None, None
            
        # 遍历材质的所有着色器
        for shader in Usd.PrimRange(bound_material.GetPrim()):
            if shader.IsA(UsdShade.Shader):
                shader_node = UsdShade.Shader(shader)
                
                # 获取纹理文件路径
                file_input = shader_node.GetInput("diffuse_texture")
                if file_input:
                    texture_path = file_input.Get(time_code)
                    print(f"[DEBUG] 找到纹理路径: {texture_path}")
                    return bound_material.GetPrim(), str(texture_path), "st"
        
        print("[WARNING] 未找到纹理节点")
        return None, None, None
        
    except Exception as e:
        print(f"[ERROR] 获取材质和纹理时发生错误: {str(e)}")
        import traceback
        print(f"[ERROR] 错误堆栈: {traceback.format_exc()}")
        return None, None, None


def _get_uv_at_hit_point(plane_prim, center, radius, uv_set, time_code):
    """计算UV坐标点"""
    try:
        mesh = UsdGeom.Mesh(plane_prim)
        # 使用当前时间获取变换矩阵
        xformable = UsdGeom.Xformable(plane_prim)
        world_to_local = xformable.ComputeLocalToWorldTransform(time_code).GetInverse()
        
        # 转换世界坐标到局部坐标
        hit_point = Gf.Vec3d(float(center[0][0]), float(center[1][0]), float(center[2][0]))
        local_center = world_to_local.Transform(hit_point)
        
        # 获取缩放信息 - 只关注XY方向的缩放（假设平面主要在XY平面上）
        # 从世界到局部的变换矩阵的XY行的长度表示缩放系数
        scale_x = world_to_local.GetRow3(0).GetLength()
        scale_y = world_to_local.GetRow3(1).GetLength()
        # 注意：对于平面，X和Y的缩放通常是相同的
        plane_scale = (scale_x + scale_y) / 2.0  # 取平均值以获得更稳定的结果
        print(f"[DEBUG] 平面缩放系数: X: {scale_x:.4f}, Y: {scale_y:.4f}, 平均: {plane_scale:.4f}")
        
        # 计算局部坐标系中的半径 - 将世界半径乘以缩放系数
        local_radius = radius * plane_scale
        print(f"[DEBUG] 世界半径: {radius:.4f} -> 局部半径: {local_radius:.4f}")
        
        # 使用 PrimvarsAPI 获取 UV 坐标
        primvars_api = UsdGeom.PrimvarsAPI(mesh)
        uv_primvar = primvars_api.GetPrimvar(uv_set)
        if not uv_primvar or not uv_primvar.HasValue():
            print(f"[WARNING] 无法获取 UV 坐标数据: {uv_set}")
            return None
            
        # 获取点和 UV 数据
        points_attr = mesh.GetPointsAttr()
        if not points_attr:
            print("[WARNING] 无法获取点数据")
            return None
            
        points = points_attr.Get(time=time_code)
        uvs = uv_primvar.Get(time=time_code)
        
        # 检查points和uvs是否为None或bool类型，确保可以进行len操作
        if (points is None or isinstance(points, bool) or uvs is None or isinstance(uvs, bool) or 
            not hasattr(points, "__len__") or not hasattr(uvs, "__len__") or
            len(points) < 4 or len(uvs) < 4):
            print("[WARNING] 点数据或 UV 数据不足或格式不正确")
            return None
            
        # 获取平面的四个角点
        p0, p1, p2, p3 = points[0], points[1], points[2], points[3]  # 左下、右下、左上、右上
        
        # 计算局部坐标系中的相对位置
        local_pos = np.array([local_center[0], local_center[1], local_center[2]])
        
        # 计算平面的宽度和高度向量
        width_vector = p1 - p0
        height_vector = p2 - p0
        plane_width = np.linalg.norm(width_vector)
        plane_height = np.linalg.norm(height_vector)
        
        print(f"[DEBUG] 平面尺寸: 宽度: {plane_width:.4f}, 高度: {plane_height:.4f}")
        
        # 计算相对位置
        width_dir = width_vector / plane_width
        height_dir = height_vector / plane_height
        
        rel_x = np.dot(local_pos - p0, width_dir)
        rel_y = np.dot(local_pos - p0, height_dir)
        
        # 归一化到[0,1]范围
        rel_x = rel_x / plane_width
        rel_y = rel_y / plane_height
        
        # 确保相对坐标在[0,1]范围内
        rel_x = np.clip(rel_x, 0.0, 1.0)
        rel_y = np.clip(rel_y, 0.0, 1.0)
        
        # 计算UV坐标
        center_u = rel_x
        center_v = rel_y
        
        # 计算UV空间中的半径 - 使用转换后的局部半径
        # 局部半径已经是平面局部坐标系下的值，可以直接除以平面宽度和高度
        uv_radius_u = local_radius
        uv_radius_v = local_radius
        
        print(f"[DEBUG] UV中心点: ({center_u:.4f}, {center_v:.4f})")
        print(f"[DEBUG] UV半径: U方向: {uv_radius_u:.4f}, V方向: {uv_radius_v:.4f}")
        
        # 返回计算结果，以元组形式
        return (float(center_u), float(center_v)), (float(uv_radius_u), float(uv_radius_v)), (plane_width, plane_height)
        
    except Exception as e:
        print(f"[ERROR] 计算 UV 点时发生错误: {str(e)}")
        import traceback
        print(f"[ERROR] 错误堆栈: {traceback.format_exc()}")
        return None, None, None

# 到达指定位置
def reachPosition(robot, joint_pos_des):
    
    # 设置误差阈值
    POS_THRESHOLD = 0.1 # 位置误差阈值（弧度）
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
    return pos_reached and vel_reached

def main():  
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)   
    sim.reset()
    current_time = 0.0
    current_step = 0

    # 获取相机Prim路径
    rgb_camera_path = '/World/envs/env_0/rsd455/RSD455/Camera_OmniVision_OV9782_Color'
    depth_camera_path = '/World/envs/env_0/rsd455/RSD455/Camera_Pseudo_Depth'
    
    # 创建相机传感器
    rgb_camera = Camera(
        prim_path=rgb_camera_path,
        resolution=(1920, 1080)
    )
    
    depth_camera = Camera(
        prim_path=depth_camera_path,
         resolution=(1920, 1080) # 使用USD中定义的位置
    )
    
    rgb_camera.initialize()
    depth_camera.initialize()
    
    print("[INFO] 相机传感器初始化完成")
    print(f"[DEBUG] RGB相机路径: {rgb_camera_path}")
    print(f"[DEBUG] 深度相机路径: {depth_camera_path}")
    
    # 等待相机初始化
    sim.step()
    scene.update(sim.get_physics_dt())


    robot = scene.articulations["robot"]
    plane_prim = scene.stage.GetPrimAtPath('/World/envs/env_0/table/table_instanceable/plane/Plane')
    
    # 检查平面的材质绑定
    print(f"[DEBUG] 平面prim路径: {plane_prim.GetPath()}")
    print(f"[DEBUG] 平面类型: {plane_prim.GetTypeName()}")
    
    # 检查所有属性
    print("[DEBUG] 平面属性列表:")
    for attr in plane_prim.GetAttributes():
        print(f"  - {attr.GetName()}: {attr.Get()}")
    
    # 检查材质绑定
    material_binding_api = UsdShade.MaterialBindingAPI(plane_prim)
    if material_binding_api:
        bound_material = material_binding_api.ComputeBoundMaterial()[0]  # 获取第一个返回值
        if bound_material:
            print(f"[DEBUG] 平面已绑定材质: {bound_material.GetPath()}")
        else:
            print("[WARNING] 材质prim无效")
    else:
        print("[WARNING] 平面无法获取材质绑定API")

    
    # 创建纹理管理器
    texture_manager = TextureManager()
    
    # 创建并绑定新材质
    material_path = "/World/envs/env_0/table/Looks/DrawingMaterial"
    material = texture_manager.create_material(scene.stage, material_path)
    
    if material:
        # 绑定材质到平面，使用最强的绑定强度
        material_binding_api = UsdShade.MaterialBindingAPI(plane_prim)
        
        # 清除所有现有的材质绑定
        material_binding_api.UnbindAllBindings()
        
        # 使用Bind方法直接绑定材质，指定最强的绑定强度
        success = material_binding_api.Bind(
            UsdShade.Material(material),  # 确保传入的是UsdShade.Material类型
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose=UsdShade.Tokens.allPurpose
        )
        
        if success:
            print(f"[DEBUG] 成功绑定新材质到平面: {material_path}")
            
            # 验证绑定结果
            bound_material = material_binding_api.ComputeBoundMaterial(materialPurpose=UsdShade.Tokens.allPurpose)
            if bound_material:
                print(f"[DEBUG] 当前绑定的材质: {bound_material[0].GetPath()}")
            else:
                print("[WARNING] 材质绑定可能未成功")
                
            # 强制更新场景
            scene.write_data_to_sim()
            
            # 等待一帧以确保绑定生效
            sim.step()
            scene.update(sim.get_physics_dt())
        else:
            print("[ERROR] 材质绑定失败")
    else:
        print("[ERROR] 无法创建新材质")
        return
    

    joint_pos_des_set = [
        [[0.0, -2.15, -2.007, -3.4907, -1.5708, 0.0]],
        [[0.7854, -2.15, -2.007, -3.4907, -1.5708, 0.0]],
        [[0.0, -2.15, -2.007, -3.4907, -1.5708, 0.0]],
        [[-0.75, -2.15, -2.007, -3.4907, -1.5708, 0.0]],
        [[-0.0, -2.15, -2.007, -3.4907, -1.5708, 0.0]]
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

    # 添加测试代码
    print("[INFO] 测试笔迹检测系统...")
    test_pen_z = np.array([[0.0, 0.0, -1.0]]).T
    test_result, test_cos_angle = is_angle_greater_than(test_pen_z, np.array([[0.0, 0.0, -1.0]]).T)
    print(f"[TEST] 测试结果: {test_result}, 角度: {np.degrees(np.arccos(test_cos_angle))}")
  

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

            # 获取相机图像
            rgb_frame = rgb_camera.get_current_frame()

            if rgb_frame is not None and "rgba" in rgb_frame:
                rgb_data = rgb_frame["rgba"]
                print("[DEBUG] RGB图像尺寸:", rgb_data.shape)
                print("[DEBUG] RGB数据类型:", type(rgb_data))
                
                # 根据数据类型进行相应的处理
                if isinstance(rgb_data, torch.Tensor):
                    rgb_data_np = rgb_data.cpu().numpy()
                else:
                    rgb_data_np = rgb_data
                
                depth_min, depth_max = np.min(rgb_data_np), np.max(rgb_data_np)
                if depth_max > depth_min:
                    depth_normalized = ((rgb_data_np - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                    os.makedirs("/tmp/isaac_textures/debug", exist_ok=True)
                    cv2.imwrite("/tmp/isaac_textures/debug/camera.png", depth_normalized)              

        
        # 更新当前时间
        current_time += sim.get_physics_dt()
        time_code = Usd.TimeCode(current_time)
        current_step += 1  # 增加时间步计数
        
        # 检测笔迹
        pen_trait_center, pen_trait_rad = rad_trait(scene, sim, current_time)
        if pen_trait_center is not None and pen_trait_rad is not None:
            print(f"[DEBUG] 检测到笔迹 - 中心: {pen_trait_center}, 半径: {pen_trait_rad}")
            
            # 计算像素半径（确保是数值类型）
            pixel_radius = max(1, int(float(pen_trait_rad) * 100))  # 增大系数，确保至少为1像素
            print(f"[DEBUG] 笔迹像素半径: {pixel_radius}")
            
            # 获取材质和纹理信息 - 使用NULL_TIME获取当前材质状态，而不依赖于特定时间点
            # NULL_TIME表示与时间无关的属性值
            material, texture_path, uv_set = _get_material_and_texture(plane_prim, scene, Usd.TimeCode.Default())
            if material and texture_path:
                print(f"[DEBUG] 获取到材质和纹理 - 材质: {material.GetPath()}, 纹理: {texture_path}")
                
                # 计算UV点，传入当前时间
                uv_points_center, uv_points_radius, plane_size = _get_uv_at_hit_point(plane_prim, pen_trait_center, pen_trait_rad, uv_set, time_code)
                if uv_points_center and uv_points_radius and plane_size:
                    print(f"[DEBUG] 计算UV点成功: 中心={uv_points_center}, 半径={uv_points_radius}, 平面尺寸={plane_size}")
                    
                    # 修改纹理 - 使用当前时间步
                    temp_texture_path = texture_manager._modify_texture(uv_points_center, uv_points_radius, plane_size, current_step)
                    if temp_texture_path:
                        print(f"[DEBUG] 纹理修改成功，新路径: {temp_texture_path}")
                        
                        # 检查文件是否存在和可读
                        if os.path.exists(temp_texture_path):
                            file_size = os.path.getsize(temp_texture_path)
                            print(f"[DEBUG] 纹理文件大小: {file_size} bytes")
                            if file_size > 0:
                                # 更新材质，传入当前时间码
                                if texture_manager._update_material(material, temp_texture_path, time_code):
                                    print(f"[DEBUG] 材质更新成功，时间点: {time_code}")
                                    # 强制更新场景
                                    scene.write_data_to_sim()
                                else:
                                    print("[WARNING] 材质更新失败")
                            else:
                                print("[WARNING] 纹理文件为空")
                        else:
                            print(f"[WARNING] 纹理文件不存在: {temp_texture_path}")
                    else:
                        print("[WARNING] 纹理修改失败")
                else:
                    print("[WARNING] UV点计算失败")
            else:
                print("[WARNING] 未找到材质或纹理")
        else:
            print("[DEBUG] 未检测到笔迹") 
                    # 执行仿真步骤
        sim.step()
        scene.update(sim.get_physics_dt())

if __name__ == "__main__":
    main()
    simulation_app.close()