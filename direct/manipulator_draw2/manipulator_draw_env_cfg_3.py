from __future__ import annotations

import torch
from collections.abc import Sequence
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.scene.interactive_scene_cfg import InteractiveSceneCfg
from .asset.scene_cfg import SceneCfg
from pxr import Usd, Gf, UsdGeom, UsdShade
from .asset.texture_manager import TextureManager
from omni.isaac.lab.utils.math import sample_uniform
import omni.isaac.lab.sim as sim_utils
from omni.isaac.debug_draw import _debug_draw
import cv2
import numpy as np
from PIL import Image
import os
import datetime
import shutil
import math

@configclass
class ManipulatorDrawEnvCfg(DirectRLEnvCfg):
   """
   """
    # env
   decimation = 1  #sim仿真执行2个周期，action control执行一次
   episode_length_s = 33.332#一段记录的长度，episode_length_step = ceil(episode_length_s / (decimation_rate * physics_time_step))

#每隔125步存储一下笔迹中心点坐标，这个作为观察值
    # simulation
   sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, #物理世界的更新间隔，默认1.0/60
        render_interval=decimation, #在渲染一帧时，物理世界更新的步数
    )
   pen_trace_record_step = 125
   pen_trace_record_space =  math.ceil(episode_length_s / (decimation * sim.dt)) // pen_trace_record_step
   action_space = 6 #机械臂6个关节
   observation_space = 14 + pen_trace_record_space * 2 #6个关节位置 6*1 + 6个关节速度 6*1 + 笔尖高度 1 + 笔尖方向 1 + 笔迹中心点坐标
   state_space = 0 #状态空间 tbd

    # scene
   scene: InteractiveSceneCfg = SceneCfg(num_envs=1024, env_spacing=3, replicate_physics=True)
   

class ManipulatorDrawEnv(DirectRLEnv):
    """
    """
    cfg: ManipulatorDrawEnvCfg
    texture_manager:TextureManager
    
    #环境初始化配置
    def __init__(self, cfg: ManipulatorDrawEnvCfg, render_mode: str|None = None, **kwargs):
        # 创建纹理管理器
        self.texture_manager = TextureManager()
        super().__init__(cfg, render_mode, **kwargs)
        self.current_time = 0.0
        self.current_step = 0
        self.robot = self.scene.articulations["robot"]       
        self.actions = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        #笔尖的世界位置坐标
        self.pen_pos = torch.zeros((self.num_envs, 3), device=self.device)
        #笔尖世界坐标方向
        self.pen_dir = torch.zeros((self.num_envs, 3), device=self.device)
        #笔持续在plane上的步数
        self.pen_on_plane = torch.zeros(self.num_envs, device=self.device)

        #各环境table上plane的中心点世界坐标
        self.table_center_pos = self.calculate_table_center_pos()

        # 每个环境建一个和贴图尺寸一致的像素矩阵，在笔迹更新的时候，更新此矩阵；使用此矩阵计算笔迹得分
        self.plane_pixel_size = 256
        self.plane_pixel_matrix = torch.zeros((self.num_envs, self.plane_pixel_size, self.plane_pixel_size), device=self.device)

        # 日志级别控制
        self.log_level = "INFO"  # 可选：DEBUG, INFO, WARNING, ERROR
        self.log_frequency = 1  # 每隔多少步记录一次详细日志
        
        # 设置日志系统
        self._setup_logging()

        # 机械臂各关节的活动范围
        self.robot_dof_lower_limits = torch.tensor([-0.7854,-2.5000,-2.3562, -3.1416, -3.1416, -3.1416], device=self.device)
        self.robot_dof_upper_limits = torch.tensor([0.7854,-2.2689,-1.2217, 3.1416, 3.1416, 3.1416], device=self.device)
        # 机械臂各关节的速度范围
        self.joint_vel_scale = 10.0     # 假设速度范围在±10

        #获取各环境plane的prim的world to lcoal转换矩阵
        #获取世界坐标到各环境plane相对坐标的缩放系数（各环境是一样的）
        #获取各环境plane相对坐标下的大小（各环境是一样的）
        self.world_to_local_matrix, self.world_to_local_scale, self.plane_p0, self.plane_width, self.plane_height = self.get_plane_world2local()

        #绘制图形的评价系数
        self.similarity_score = torch.zeros((self.num_envs), device=self.device)
        
        # 获取debug draw接口，避免反复创建
        self.is_debug_draw = True
        self.debug_draw = _debug_draw.acquire_debug_draw_interface()
        
        # 存储各环境的笔迹点信息，用于重置时选择性清除
        # 字典结构: {env_id: {'points': [...], 'colors': [...], 'sizes': [...]}}
        self.pen_trace_points = {}

        #每隔cfg.pen_trace_record_space步存储一下笔迹中心点坐标，这个作为观察值
        
        self.pen_trace_points_history = torch.zeros((self.num_envs, cfg.pen_trace_record_space, 2), device=self.device)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits) * 0.5
        self.robot_dof_speed_scales[self.robot.find_joints("wrist_1_joint")[0]] = 0.1
        self.robot_dof_speed_scales[self.robot.find_joints("wrist_2_joint")[0]] = 0.1
        self.robot_dof_speed_scales[self.robot.find_joints("wrist_3_joint")[0]] = 0.1

        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)

        #删除debug_images目录
        debug_dir = os.path.join(os.getcwd(), "debug_images")
        if os.path.exists(debug_dir):
            shutil.rmtree(debug_dir)
        
        #删除logs下的以.log结尾的日志文件
        log_dir = os.path.join(os.getcwd(), "logs")
        for file in os.listdir(log_dir):
            if file.endswith(".log"):
                os.remove(os.path.join(log_dir, file))



    def _setup_logging(self):
        """设置日志系统"""
        import os
        import datetime
        
        # 日志设置
        self.log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 时间戳
        self.log_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 日志文件路径
        self.log_file_path = os.path.join(self.log_dir, f"manipulator_draw_{self.log_timestamp}.log")
        
        # 日志文件大小限制 (10MB)
        self.max_log_size = 10 * 1024 * 1024
        
        # 日志计数器用于轮换
        self.log_counter = 0
        
        # 日志刷新频率 (每100条记录刷新一次文件)
        self.flush_frequency = 100
        self.log_count_since_flush = 0
        
        # 创建初始日志文件
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write(f"=== 机械臂绘制环境日志 开始于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(f"环境数量: {self.num_envs}\n")
            f.write(f"日志级别: {self.log_level}\n\n")
        
        print(f"[INFO] 日志文件创建于: {self.log_file_path}")

    #构建场景
    def _setup_scene(self):
        """
        """
        # 获取相机Prim路径
        """         rgb_camera_path = '/World/envs/env_0/rsd455/RSD455/Camera_OmniVision_OV9782_Color'
        
        # 创建相机传感器
        rgb_camera = Camera(
            prim_path=rgb_camera_path,
            resolution=(1920, 1080)
        )        
        
        rgb_camera.initialize()
        
        print("[INFO] 相机传感器初始化完成")
        print(f"[DEBUG] RGB相机路径: {rgb_camera_path}")  """

        plane_prim = self.scene.stage.GetPrimAtPath('/World/envs/env_0/table/table_instanceable/plane/Plane')        

        
        # 检查材质绑定
        """         material_binding_api = UsdShade.MaterialBindingAPI(plane_prim)
        if material_binding_api:
            bound_material = material_binding_api.ComputeBoundMaterial()[0]  # 获取第一个返回值
            if bound_material:
                print(f"[DEBUG] 平面已绑定材质: {bound_material.GetPath()}")
            else:
                print("[WARNING] 材质prim无效")
        else:
            print("[WARNING] 平面无法获取材质绑定API") """

        
        # 创建并绑定新材质
        material_path = "/World/envs/env_0/table/Looks/DrawingMaterial"
        material = self.texture_manager.create_material(self.scene.stage, material_path)
        
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
                    
            else:
                print("[ERROR] 材质绑定失败")
        else:
            print("[ERROR] 无法创建新材质")
        
        terrain = sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )

        terrain.func("/World/ground", terrain, (0.0, 0.0, -1.05))
        
        # 创建并绑定新光源
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)        
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        

    #每个迭代step中，在物理环境执行之前，执行的逻辑
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        """
        self.current_step += 1
        self.last_pen_pos = self.pen_pos
        self.actions = actions.clone().clamp(-1.0, 1.0) 
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * self.actions * 5.0
        self.robot_dof_targets[:] = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)   
        self._log("_pre_physics_step", "执行动作", self.robot_dof_targets, "DEBUG")

    
    #计算观察结果
    def _get_observations(self) -> dict:
        """
        """
        #可能会执行reset动作，所以需要重新计算指标
        self.compute_intermediate_values()

        joint_pos_norm = (
            2.0
            * (self.robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )  # 归一化到[-1,1]
        joint_vel_norm = self.robot.data.joint_vel / self.joint_vel_scale  # 归一化到[-1,1]
        pen_height = self.pen_pos[:, 2]
    # 接近平面(0-0.01)范围扩展到(0-0.05)，远离平面范围压缩
        pen_height_norm = torch.where(
            (pen_height > -0.05) & (pen_height < 0),
            (pen_height + 0.05) * 5,  # 近距离放大5倍
            -0.5  # 远距离压缩
        ).unsqueeze(1)

        pen_dir_norm = self.pen_dir

        #笔迹是在plane上的，笔迹的坐标可以根据各环境的table_center_pos和plane的大小进行归一化
        pen_trace_points_history_norm = torch.clip((self.pen_trace_points_history - self.table_center_pos[:, 0:2].unsqueeze(1)) / 0.4, -1.0, 1.0)

        # 将形状从 (num_envs, pen_trace_record_space, 2) 变为 (num_envs, pen_trace_record_space*2)
        pen_trace_points_history_norm_flat = pen_trace_points_history_norm.reshape(self.num_envs, -1)

        obs = torch.cat(
            (                
                joint_pos_norm,    # [-1,1]
                joint_vel_norm,    # [-1,1]
                pen_height_norm,   # [0,1]特别处理
                pen_dir_norm, 
                pen_trace_points_history_norm_flat                 
            ),
            dim=-1,
        )
        self._log("_get_observations", "观察值", {
            "关节位置": joint_pos_norm,
            "关节速度": joint_vel_norm,
            "笔尖位置": pen_height_norm,
            "笔尖方向": pen_dir_norm,
            "笔迹中心点坐标": pen_trace_points_history_norm_flat,
        }, "DEBUG")

        
        # 检查观察结果中是否有NaN值
        if torch.isnan(obs).any() or torch.isinf(obs).any() or torch.isneginf(obs).any():
            #如果当obs中的变量出现nan时，输出日志，打印出是哪个变量为nan
            # 检查各个变量是否存在NaN值
            variables = {
                "joint_pos_norm": joint_pos_norm,
                "joint_vel_norm": joint_vel_norm,
                "pen_height_norm": pen_height_norm,
                "pen_dir_norm": pen_dir_norm,
            }
            # 记录NaN出现的环境索引
            nan_env_indices = torch.where(torch.isnan(obs).any(dim=1))[0]
            inf_env_indices = torch.where(torch.isinf(obs).any(dim=1))[0]
            ninf_env_indices = torch.where(torch.isneginf(obs).any(dim=1))[0]
            
            # 记录各变量中的NaN情况
            nan_variables = {}  
            inf_variables = {}
            ninf_variables = {}
            for var_name, var_tensor in variables.items():
                if torch.isnan(var_tensor).any():
                    nan_envs = torch.where(torch.isnan(var_tensor).any(dim=1))[0].tolist()
                    nan_variables[var_name] = nan_envs
                if torch.isinf(obs).any():
                    inf_envs = torch.where(torch.isinf(obs).any(dim=1))[0].tolist()
                    inf_variables[var_name] = inf_envs
                if torch.isneginf(obs).any():
                    ninf_envs = torch.where(torch.isneginf(obs).any(dim=1))[0].tolist()
                    ninf_variables[var_name] = ninf_envs
            
            # 输出详细日志
            self._log("_get_observations", "观察值中检测到NaN", {
                "NaN环境索引": nan_env_indices.tolist(),
                "各变量NaN情况": nan_variables,
                "inf环境索引": inf_env_indices.tolist(),
                "各变量inf情况": inf_variables,
                "ninf环境索引": ninf_env_indices.tolist(),
                "各变量ninf情况": ninf_variables,
            }, "ERROR")
                
            # 将NaN替换为0，避免训练崩溃
            #obs = torch.nan_to_num(obs, nan=0.0)

        return {"policy": obs}

    
    #定义action执行逻辑
    def _apply_action(self) -> None:
        """
        """
        self._log("_apply_action", "应用动作到机械臂", self.robot_dof_targets, "DEBUG")
        self.robot.set_joint_position_target(self.robot_dof_targets)

    #计算奖励
    def _get_rewards(self) -> torch.Tensor:
        """
        计算rewards
        """
        rewards = self.compute_rewards()
        self._log("_get_rewards", "计算的奖励值", rewards, "DEBUG")
        return rewards

    #定义迭代结束逻辑
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        """
        ##action执行后，在判断是否终止前，需要计算指标
        self.compute_intermediate_values()
        #如果笔高度(距离plane太近)、笔的位置超出了plane范围，则终止
        pen_height_terminate = (self.pen_pos[:, 2] < -0.06)
        # 判断笔是否在plane范围内
        # 创建每个维度的范围判断
        x_in_range = (self.pen_pos[:, 0] < self.table_center_pos[:, 0] - 0.4) | (self.pen_pos[:, 0] > self.table_center_pos[:, 0] + 0.4)
        y_in_range = (self.pen_pos[:, 1] < self.table_center_pos[:, 1] - 0.4) | (self.pen_pos[:, 1] > self.table_center_pos[:, 1] + 0.4)

        # 综合判断所有维度是否都在范围内
        pen_in_plane_range = x_in_range | y_in_range
        #绘制完成       
        terminated = (self.similarity_score > 0.90) | pen_height_terminate | pen_in_plane_range
        #达到最大迭代次数
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        
        self._log("_get_dones", "终止状态", {
            "pen_height_terminate": pen_height_terminate,
            "pen_in_plane_range": pen_in_plane_range,
            "terminated": terminated, 
            "truncated": truncated, 
            "sucess_steps": self.pen_on_plane
        }, "DEBUG")
        
        return terminated, truncated

    #定义仿真实例重置逻辑
    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        """
        if env_ids is not None:
            
            super()._reset_idx(env_ids)
            #重置机械臂关节位置,随机生成关节位置,使用self.robot_dof_lower_limits和self.robot_dof_upper_limits作为上下限            
            joint_pos = torch.rand(self.num_envs, self.robot.num_joints, device=self.device)[env_ids] * (self.robot_dof_upper_limits - self.robot_dof_lower_limits) + self.robot_dof_lower_limits
            joint_vel = torch.zeros_like(joint_pos)
            
            
            self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

            # 清除被重置环境的笔迹点数据
            for env_idx in env_ids:
                # 将张量类型的 env_idx 转换为整数
                env_id = env_idx.item() if isinstance(env_idx, torch.Tensor) else int(env_idx)

                self.pen_trace_points_history[env_idx] = torch.zeros((self.cfg.pen_trace_record_space, 2), device=self.device)
                
                # 使用整数类型的键检查和删除笔迹点数据
                if env_id in self.pen_trace_points:
                    del self.pen_trace_points[env_id]
                
                try:
                    #重置笔迹和圆相似度指标
                    self.similarity_score[env_idx] = 0.0
                    #重置笔在plane上的步数
                    self.pen_on_plane[env_idx] = 0
                    self.plane_pixel_matrix[env_idx] = 0

                except Exception as e:
                    self._log("_reset_idx", f"重置pen on plane 上步数错误: {str(e)}", None, "ERROR", env_idx)
                    import traceback
                    self._log("_reset_idx", f"错误堆栈: {traceback.format_exc()}", None, "ERROR", env_idx)
            
            # 重新绘制所有未被重置环境的笔迹点
            if self.is_debug_draw:
                self._redraw_all_pen_traces()

    #重置观察、奖励使用的变量
    def compute_intermediate_values(self):
        """
        """
        update_mask = self.update_pen_on_plane()
        if update_mask is not None or not torch.any(update_mask):
            self.compare_origin_draw(update_mask)
                  

    #计算rewards的实现
    def compute_rewards(self):
        """
        计算rewards的实现 - 归一化版本
        """
        # 提取原始数据
        pen_height = self.pen_pos[:, 2]
        pen_dir_factor = self.pen_dir[:, 0]          
        
        # 1. 高度奖励归一化处理 - 越接近平面越好(0.0最好)
        # 使用与观察空间相似的非线性变换放大接近平面的信
        
        # 2. 方向奖励处理 - 指向平面越好(-1最好,1最差)
        # 归一化到[-1,0]范围，直接使用
        dir_reward = pen_dir_factor
            
        # 4. 任务阶段判断
        contact_reward = torch.where(
             (pen_height > -0.03) & (pen_height < 0.0),  # 接触平面的阈值
            torch.ones_like(pen_height) * -100.0,  # 接触奖励
            -10.0  # 未接触无奖励
        )


        #circ_reward = torch.pow(self.similarity_score, 0.5) * 7.0

        # 初始化奖励
        reward = torch.zeros_like(pen_height)

        reward = dir_reward * 0.5 + contact_reward * 0.1 + self.pen_on_plane * 1.0 #+ circ_reward * 1.0
        
        # 惩罚计算 - 保持原有结构但调整值
        penalties = {
            "笔位置低于平面": (pen_height < -0.05, -20.0) ,
             "所有机械臂关节运动小于0.01": (torch.all(self.actions < 0.001), -10.0) # 减小惩罚强度
        }
        
        # todo，修改成tensor操作，不使用for循环，应用惩罚(保持不变)
        for name, (condition, penalty) in penalties.items():
            if torch.any(condition):
                old_reward = reward.clone()
                reward = torch.where(condition, old_reward + penalty, old_reward)
                # 记录惩罚
                if self.current_step % self.log_frequency == 0:
                    applied_count = condition.sum().item()
                    self._log("compute_rewards", f"应用惩罚: {name}", 
                            f"数量: {applied_count}/{self.num_envs}, 惩罚值: {penalty}", 
                            "INFO")
        
        # 记录最终奖励
        self._log("compute_rewards", "最终奖励最大值", torch.max(reward), "INFO")
        
        return reward
    
    #计算table上plane的中心点世界坐标
    def calculate_table_center_pos(self):
        """
        计算所有env中table上plane的中心点世界坐标
        """

        #获取所有环境table的prim
        table_prims = torch.zeros((self.num_envs, 3), device=self.device)

        # todo，修改成tensor操作，不要使用for循环
        for env_idx in range(self.num_envs):
             table_prim = self.scene.stage.GetPrimAtPath(f'/World/envs/env_{env_idx}/table/table_instanceable/plane/Plane')
             if table_prim:
                xformable = UsdGeom.Xformable(table_prim)
                # 获取Local到World的变换矩阵
                # time_code用于指定时间点，可以使用Usd.TimeCode.Default()表示默认时间
                local_to_world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode(self.current_time))

                # 获取prim原点的世界坐标
                # 创建一个表示局部坐标系原点的向量
                local_origin = Gf.Vec3f(0.0, 0.0, 0.0)

                # 将局部坐标转换为世界坐标
                world_origin = local_to_world_transform.Transform(local_origin)
                
                # 将Gf.Vec3f类型转换为torch.Tensor的更简洁写法
                table_prims[env_idx] = torch.tensor([float(world_origin[0]), float(world_origin[1]), float(world_origin[2])], device=self.device)
        
        # #返回所有env中table的中心点世界坐标
        return table_prims
    
            
    def get_plane_world2local(self):
        """
        获取所有环境plane的prim的world to local转换矩阵
        获取世界坐标到各环境plane相对坐标的缩放系数（各环境是一样的）
        获取各环境plane相对坐标下的大小（各环境是一样的）
        """
        # 初始化变换矩阵 [num_envs, 4, 4]
        world_to_local_matrix = torch.zeros((self.num_envs, 4, 4), device=self.device)
        world_to_local_scale = 0.0
        plane_width = 0.0
        plane_height = 0.0
        p0 = torch.zeros((self.num_envs, 3), device=self.device)
        
        # 获取第一个环境的plane prim，用于计算缩放系数和平面大小
        first_env_plane_prim = self.scene.stage.GetPrimAtPath(f'/World/envs/env_0/table/table_instanceable/plane/Plane')
        if first_env_plane_prim:
            xformable = UsdGeom.Xformable(first_env_plane_prim)
            # 获取Local到World的变换矩阵
            local_to_world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode(self.current_time))
            # 获取World到Local的变换矩阵（取逆）
            world_to_local_transform = local_to_world_transform.GetInverse()
            
            # 计算缩放系数
            scale_x = world_to_local_transform.GetRow3(0).GetLength()
            scale_y = world_to_local_transform.GetRow3(1).GetLength()
            world_to_local_scale = (scale_x + scale_y) / 2.0
            
            # 获取平面的宽度和高度
            mesh = UsdGeom.Mesh(first_env_plane_prim)
            points_attr = mesh.GetPointsAttr()
            if points_attr:
                points = points_attr.Get(Usd.TimeCode(self.current_time))
                p0_usd, p1, p2 = points[0], points[1], points[2]
                
                # 计算平面的宽度和高度
                width_vector = p1 - p0_usd
                height_vector = p2 - p0_usd
                plane_width = width_vector.GetLength()
                plane_height = height_vector.GetLength()
                
                # 将p0转换为torch.Tensor
                p0_tensor = torch.tensor([float(p0_usd[0]), float(p0_usd[1]), float(p0_usd[2])], device=self.device)
                
                # 将p0赋值给所有环境
                for i in range(self.num_envs):
                    p0[i] = p0_tensor
        
        # 为每个环境计算world to local变换矩阵
        for env_idx in range(self.num_envs):
            plane_prim = self.scene.stage.GetPrimAtPath(f'/World/envs/env_{env_idx}/table/table_instanceable/plane/Plane')
            if plane_prim:
                xformable = UsdGeom.Xformable(plane_prim)
                # 获取Local到World的变换矩阵
                local_to_world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode(self.current_time))
                # 获取World到Local的变换矩阵（取逆）
                world_to_local_transform = local_to_world_transform.GetInverse()
                
                # 将Gf.Matrix4d转换为torch.Tensor
                # 注意：USD的Matrix4d是按行存储的，而PyTorch的矩阵是按列存储的
                # 因此，我们需要转置矩阵以确保正确的矩阵乘法顺序
                matrix_data = []
                for i in range(4):
                    row = []
                    for j in range(4):
                        row.append(float(world_to_local_transform[i, j]))
                    matrix_data.append(row)
                
                # 转换为torch.Tensor并转置
                world_to_local_tensor = torch.tensor(matrix_data, device=self.device)
                
                # 将转换后的tensor赋值给world_to_local_matrix
                world_to_local_matrix[env_idx] = world_to_local_tensor
        
        return world_to_local_matrix, world_to_local_scale, p0, plane_width, plane_height

   
       #local，计算贴图上黑色像素是否是一个圆形
    def compare_origin_draw(self, update_mask):
        """
        判断所有环境贴图上的黑色笔迹是否为圆形并计算相似度
        
        优化：使用self.plane_pixel_matrix直接计算，避免磁盘IO
        
        参数:
            update_mask (Tensor): 布尔掩码，指示哪些环境需要更新
        """                  
   
        # 只处理需要更新的环境
        if update_mask is None or not torch.any(update_mask):
            return

        # 获取需要更新的环境索引
        env_indices = torch.nonzero(update_mask).squeeze(-1).tolist()
        self._log("compare_origin_draw", f"需更更新笔迹的环境", env_indices, "DEBUG")

        # 使用PyTorch操作批量处理所有需要更新的环境
        for env_idx in env_indices:
            try:
                # 获取当前环境的像素矩阵
                pixel_matrix = self.plane_pixel_matrix[env_idx].cpu().numpy()                
                
                # 将像素矩阵转换为二值图像 (0-1 -> 0-255)
                binary = (pixel_matrix * 255).astype(np.uint8)

                #每过50 step，保存一下像素矩阵为图片
                if self.current_step % 50 == 0:
                    # 创建debug_images目录（如果不存在）
                    debug_dir = os.path.join(os.getcwd(), "debug_images")
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    # 为每个环境创建子目录
                    env_dir = os.path.join(debug_dir, f"env_{env_idx}")
                    os.makedirs(env_dir, exist_ok=True)
                    
                    # 保存图像，文件名包含步数信息
                    image_path = os.path.join(env_dir, f"step_{self.current_step}.png")
                    cv2.imwrite(image_path, binary)
                    
                    self._log("compare_origin_draw", f"保存笔迹图像", {
                        "env_idx": env_idx,
                        "image_path": image_path,
                        "step": self.current_step
                    }, "DEBUG", env_idx)               
                
                #根据记录的笔迹中心点坐标，计算笔迹和圆的相似度
                # 1. 计算拟合圆
                points = self.pen_trace_points_history[env_idx]  # 已经是tensor
                # 移除零值点
                valid_mask = ~torch.all(points == 0, dim=1)
                valid_points = points[valid_mask]
                
                if len(valid_points) >= 3:  # 至少需要3个点才能拟合圆
                    # 使用最小二乘法拟合圆
                    def fit_circle_tensor(points):
                        x = points[:, 0]
                        y = points[:, 1]
                        n = len(x)
                        x_mean = torch.mean(x)
                        y_mean = torch.mean(y)
                        u = x - x_mean
                        v = y - y_mean
                        Suv = torch.sum(u*v)
                        Suu = torch.sum(u**2)
                        Svv = torch.sum(v**2)
                        Suuv = torch.sum(u**2 * v)
                        Suvv = torch.sum(u * v**2)
                        Suuu = torch.sum(u**3)
                        Svvv = torch.sum(v**3)
                        
                        A = torch.tensor([[Suu, Suv], [Suv, Svv]], device=self.device)
                        B = torch.tensor([(Suuu + Suvv)/2, (Svvv + Suuv)/2], device=self.device)
                        uc, vc = torch.linalg.solve(A, B)
                        xc = x_mean + uc
                        yc = y_mean + vc
                        r = torch.sqrt(uc**2 + vc**2 + (Suu + Svv)/n)
                        return xc, yc, r
                    
                    xc, yc, r = fit_circle_tensor(valid_points)
                    
                    # 2. 计算IoU (交并比)
                    # 计算点到圆心的距离
                    distances = torch.sqrt((valid_points[:, 0] - xc)**2 + (valid_points[:, 1] - yc)**2)
                    # 计算在圆内的点比例
                    points_in_circle = torch.sum(distances <= r) / len(valid_points)
                    env_iou = points_in_circle
                    
                    # 3. 计算圆度
                    # 计算点集的周长和面积
                    diff_points = torch.diff(valid_points, dim=0)
                    perimeter = torch.sum(torch.sqrt(torch.sum(diff_points**2, dim=1)))
                    area = torch.pi * r**2
                    # 圆度 = 4π * 面积 / 周长^2
                    env_circularity = 4 * torch.pi * area / (perimeter**2) if perimeter > 0 else 0                    
                    
                    # 综合评分
                    env_similarity_score = 0.4 * env_iou  + 0.6 * env_circularity
                else:
                    env_similarity_score = torch.tensor(0.0, device=self.device)
                
                # 更新指标
                self.similarity_score[env_idx] = env_similarity_score
                
            except Exception as e:
                self._log("compare_origin_draw", f"分析笔迹像素矩阵时发生错误: {str(e)}", None, "ERROR", env_idx)
                import traceback
                self._log("compare_origin_draw", f"错误堆栈: {traceback.format_exc()}", None, "ERROR", env_idx)
        
        # 记录总体统计
        valid_scores = self.similarity_score > 0
        if torch.any(valid_scores):
            self._log("compare_origin_draw", "圆形检测结果", {
                "最终相似度分数": torch.max(self.similarity_score),
                "最终相似度分数最大值的env_idx": torch.argmax(self.similarity_score)
            }, "INFO")        
        # 不需要返回任何值
        return

    #更新plane贴图
    def update_pen_on_plane(self):
        """
        更新所有env下的plane的贴图
        
        优化：使用顺序处理代替多线程处理
        """
        
        # 获取当前时间码
        time_code = Usd.TimeCode(self.current_time)
        
        # 获取tool0链接的索引
        tool0_index = None
        for i, name in enumerate(self.robot.body_names):
            if name == "tool0":
                tool0_index = i
                break
                
        if tool0_index is None:
            self._log("update_pen_on_plane", "未找到tool0链接", None, "WARNING")
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 1. 批量获取所有环境的笔的位置和方向 (一次性在GPU上计算)
        # 获取所有tool0的世界位置和方向
        world_pos = self.robot.data.body_pos_w[:, tool0_index]  # [num_envs, 3]
        world_quat = self.robot.data.body_quat_w[:, tool0_index]  # [num_envs, 4]
        
        self._log("update_pen_on_plane", "笔世界位置", world_pos, "DEBUG")
        self._log("update_pen_on_plane", "笔世界旋转", world_quat, "DEBUG")
        
        # 计算旋转矩阵 (批量处理)
        w, x, y, z = world_quat[:, 0], world_quat[:, 1], world_quat[:, 2], world_quat[:, 3]
        
        # 构建每个环境的旋转矩阵元素 (批量)
        rot_xx = 1 - 2*y*y - 2*z*z
        rot_xy = 2*x*y - 2*w*z
        rot_xz = 2*x*z + 2*w*y
        rot_yx = 2*x*y + 2*w*z
        rot_yy = 1 - 2*x*x - 2*z*z
        rot_yz = 2*y*z - 2*w*x
        rot_zx = 2*x*z - 2*w*y
        rot_zy = 2*y*z + 2*w*x
        rot_zz = 1 - 2*x*x - 2*y*y
        
        # Cone相对于tool0的局部位置 (所有环境相同)
        cone_local_pos = torch.tensor([0.0, 0.0, 0.05], device=self.device)
        
        # 计算所有环境中Cone在世界坐标系中的位置
        cone_world_pos = torch.zeros((self.num_envs, 3), device=self.device)
        cone_world_pos[:, 0] = world_pos[:, 0] + rot_xx*cone_local_pos[0] + rot_xy*cone_local_pos[1] + rot_xz*cone_local_pos[2]
        cone_world_pos[:, 1] = world_pos[:, 1] + rot_yx*cone_local_pos[0] + rot_yy*cone_local_pos[1] + rot_yz*cone_local_pos[2]
        cone_world_pos[:, 2] = world_pos[:, 2] + rot_zx*cone_local_pos[0] + rot_zy*cone_local_pos[1] + rot_zz*cone_local_pos[2]
        
        # 计算所有环境中Cone在世界坐标系中的z轴方向
        cone_world_z = torch.zeros((self.num_envs, 3), device=self.device)
        cone_world_z[:, 0] = rot_xz  # z轴的x分量
        cone_world_z[:, 1] = rot_yz  # z轴的y分量
        cone_world_z[:, 2] = rot_zz  # z轴的z分量
        
        # 计算单位方向向量
        cone_world_z_norm = torch.norm(cone_world_z, dim=1, keepdim=True)
        cone_world_z = cone_world_z / cone_world_z_norm  # [num_envs, 3]
        
        # 2. 判断所有环境中哪些满足绘制条件
        # 计算参考向量 (负z方向)
        reference_vector = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        
        # 计算所有环境中笔与参考向量的夹角余弦值
        cos_angles = torch.sum(cone_world_z * reference_vector, dim=1)  # [num_envs]
        
        
        # 修复pen_dir的维度问题，使用适当的方法
        # 方案1：将cos_angles扩展为每个环境的特征值
        self.pen_dir = cos_angles.unsqueeze(1)  # [num_envs, 3]
        # 或者使用实际的方向向量
        # self.pen_dir = cone_world_z  # [num_envs, 3]
        
        # 确定哪些环境满足角度条件 (余弦值>0.707，约45度)
        angle_mask = cos_angles > 0.707  # [num_envs]
        
        # 根据笔中心高度，笔的朝向，计算所有环境中笔尖到平面的高度 (z坐标)
        self.pen_pos = self.calculate_endpoint(cone_world_pos, cone_world_z, 0.05)
        pen_origin_z = self.pen_pos[:, 2]  # [num_envs]
        
        # 确定哪些环境满足高度条件 (0<高度<0.05)
        height_mask = (pen_origin_z > -0.05) & (pen_origin_z < 0)  # [num_envs]
        
        x_in_range = (self.pen_pos[:, 0] >= self.table_center_pos[:, 0] - 0.4) & (self.pen_pos[:, 0] <= self.table_center_pos[:, 0] + 0.4)
        y_in_range = (self.pen_pos[:, 1] >= self.table_center_pos[:, 1] - 0.4) & (self.pen_pos[:, 1] <= self.table_center_pos[:, 1] + 0.4)

        # 综合条件，确定哪些环境需要更新步数
        update_mask = angle_mask & height_mask & x_in_range & y_in_range # [num_envs]
        
        # 如果没有环境需要更新，直接返回
        if not torch.any(update_mask):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        self.pen_on_plane = torch.where(update_mask, self.pen_on_plane + 1, torch.zeros_like(self.pen_on_plane))  

        #根据笔的高度，笔和plane的距离，计算笔和plane接触截面的中心位置
        pen_trait_center = torch.where(update_mask.unsqueeze(1).expand(-1, 3), 
                                      self.calculate_endpoint(cone_world_pos, cone_world_z, target_z=0), 
                                      torch.zeros_like(cone_world_pos))
        
        # 使用episode_length_buf来判断每个环境的步数，确保环境重置后正确记录笔迹中心点坐标
        record_indices = torch.where(self.episode_length_buf % self.cfg.pen_trace_record_step == 0)[0]
        if len(record_indices) > 0:
            # 计算每个环境应该存储的索引位置
            storage_indices = self.episode_length_buf[record_indices] // self.cfg.pen_trace_record_step
            # 确保索引不超出范围
            valid_indices = storage_indices < self.cfg.pen_trace_record_space
            if torch.any(valid_indices):
                valid_record_indices = record_indices[valid_indices]
                valid_storage_indices = storage_indices[valid_indices]
                # 存储笔迹中心点坐标
                self.pen_trace_points_history[valid_record_indices, valid_storage_indices, :] = pen_trait_center[valid_record_indices, 0:2]

        # 创建非零掩码
        non_zero_mask = torch.any(pen_trait_center != 0, dim=1)

        # 过滤非零值
        pen_trait_center_filtered = pen_trait_center[non_zero_mask].cpu().numpy()

        # 转换为列表格式
        pen_trait_center_draw = [(pos[0], pos[1], 0.0) for pos in pen_trait_center_filtered]

        #计算笔迹半径，目前是世界坐标的半径，可能会尺寸存在问题
        # 先将整个张量移到CPU，再转换为numpy
        pen_trait_rad = (0.025 * (0.5 - (pen_origin_z + 0.05) / 0.1)).cpu().numpy()
        
        # 计算可视化点的大小，根据笔迹半径按比例缩放
        points_sizes = []
        points_colors = []
        env_indices = []
        
        for i, env_idx in enumerate(torch.nonzero(non_zero_mask).squeeze(-1).tolist()):
            # 将半径转换为可视化点的大小，最小为5，最大为20
            # 确保使用正确的索引来获取对应环境的半径值
            radius = pen_trait_rad[env_idx]
            #point_size = max(5, min(20, int(radius * 800)))
            point_size = 5
            points_sizes.append(point_size)
            
            # 根据笔在平面上的停留步数确定颜色
            point_color = (1.0, 0.0, 0.0, 1.0)
            
            points_colors.append(point_color)
            env_indices.append(env_idx)
            
            # 为每个环境保存笔迹点信息
            # 对每个环境，将点信息附加到现有点列表中
            point = pen_trait_center_draw[i]
            
            # 确保使用整数类型作为字典键
            env_id = int(env_idx)
            
            if env_id not in self.pen_trace_points:
                self.pen_trace_points[env_id] = {'points': [], 'colors': [], 'sizes': []}
                
            # 限制每个环境存储的点数，避免内存占用过大（可选）
            max_points_per_env = 800  # 每个环境最多存储500个点
            if len(self.pen_trace_points[env_id]['points']) >= max_points_per_env:
                # 移除最旧的点
                self.pen_trace_points[env_id]['points'].pop(0)
                self.pen_trace_points[env_id]['colors'].pop(0)
                self.pen_trace_points[env_id]['sizes'].pop(0)
                
            # 添加新的点
            self.pen_trace_points[env_id]['points'].append(point)
            self.pen_trace_points[env_id]['colors'].append(point_color)
            self.pen_trace_points[env_id]['sizes'].append(point_size)

        # 重新绘制所有笔迹点
        if self.is_debug_draw:
            self._redraw_all_pen_traces()

        #更新各环境plane像素矩阵   
        #笔迹中心点在像素矩阵上的坐标  
        # 创建齐次坐标 [num_envs, 1, 4]
        homogeneous_coords = torch.zeros((self.num_envs, 1, 4), device=self.device)
        homogeneous_coords[:, 0, 0:3] = pen_trait_center
        homogeneous_coords[:, 0, 3] = 1.0       

        
        # 使用批量矩阵乘法进行坐标变换 [num_envs, 1, 4] @ [num_envs, 4, 4] -> [num_envs, 1, 4]
        # 注意：这里我们需要确保矩阵乘法的顺序与USD的Transform函数一致
        # 在USD中，Transform函数是 M * v，其中M是变换矩阵，v是向量
        # 但在PyTorch中，bmm是批量矩阵乘法，我们需要确保矩阵和向量的顺序正确
        transformed_coords = torch.bmm(homogeneous_coords, self.world_to_local_matrix)       

        
        # 将transformed_coords从[num_envs, 1, 4]转换为[num_envs, 4, 1]
        transformed_coords = transformed_coords.transpose(1, 2)
        
        # 提取前三个坐标，丢弃w分量
        local_coords = transformed_coords[:, 0:3, 0]

        
        # 计算相对于平面原点的位置并转换到像素坐标
        # 注意：这里我们需要确保plane_p0的维度与local_coords匹配
        # 如果plane_p0是[num_envs, 3]，则可以直接相减
        # 如果plane_p0是[3]，则需要广播到[num_envs, 3]
        relative_pen_trait_center_pixel = torch.floor((local_coords - self.plane_p0) / self.plane_width * self.plane_pixel_size).to(torch.int32)
        
        #计算笔迹在像素矩阵上的半径大小 
        # 注意：这里我们需要确保pen_trait_rad的维度与non_zero_mask匹配
        # 如果pen_trait_rad是[num_envs]，则可以直接使用
        # 如果pen_trait_rad是标量，则需要广播到[num_envs]
        # 确保pen_trait_rad是PyTorch张量
        if isinstance(pen_trait_rad, np.ndarray):
            pen_trait_rad = torch.tensor(pen_trait_rad, device=self.device)
        pen_trait_rad_pixel = torch.ceil(pen_trait_rad * self.world_to_local_scale / self.plane_width * self.plane_pixel_size).to(torch.int32)


        if torch.any(non_zero_mask):

            #从non_zero_mask中获取所有有效环境的索引
            valid_env_indices = torch.nonzero(non_zero_mask).squeeze(-1).tolist()

            # 获取所有有效环境的圆心和半径
            centers_x = relative_pen_trait_center_pixel[non_zero_mask, 0]
            centers_y = relative_pen_trait_center_pixel[non_zero_mask, 1]
            # 修复：确保pen_trait_rad_pixel是张量而不是NumPy数组
            if isinstance(pen_trait_rad_pixel, np.ndarray):
                pen_trait_rad_pixel = torch.tensor(pen_trait_rad_pixel, device=self.device)
            radii = pen_trait_rad_pixel[non_zero_mask]

            # 创建跟踪统计信息的张量
            pixels_updated = torch.zeros(len(valid_env_indices), dtype=torch.int64, device=self.device)  
            
            # 按批次处理环境，避免一次性占用过多内存
            batch_size = 64  # 每批处理的环境数量
            num_batches = (len(pixels_updated) + batch_size - 1) // batch_size
            
            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, len(pixels_updated))
                
                batch_env_indices = valid_env_indices[start_idx:end_idx]
                batch_centers_x = centers_x[start_idx:end_idx]
                batch_centers_y = centers_y[start_idx:end_idx]
                batch_radii = radii[start_idx:end_idx]
                
                # 对批次中的每个环境并行处理
                for i, (env_idx, center_x, center_y, radius) in enumerate(
                    zip(batch_env_indices, 
                        batch_centers_x.tolist(),
                        batch_centers_y.tolist(),
                        batch_radii.tolist())
                ):
                    # 确定绘制圆的边界框
                    x_min = max(0, int(center_x - radius))
                    x_max = min(self.plane_pixel_size - 1, int(center_x + radius))
                    y_min = max(0, int(center_y - radius))
                    y_max = min(self.plane_pixel_size - 1, int(center_y + radius))
                    
                    # 检查边界框是否有效
                    if x_min <= x_max and y_min <= y_max:
                        # 创建圆的像素坐标（使用预先计算的网格加速）
                        y_grid, x_grid = torch.meshgrid(
                            torch.arange(y_min, y_max + 1, device=self.device),
                            torch.arange(x_min, x_max + 1, device=self.device),
                            indexing="ij"
                        )
                        
                        # 使用向量化操作计算距离
                        distances = torch.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
                        circle_mask = distances <= radius
                        
                        # 应用掩码更新像素矩阵
                        if torch.any(circle_mask):
                            # 使用高效的张量索引批量更新
                            self.plane_pixel_matrix[env_idx, y_grid[circle_mask], x_grid[circle_mask]] = 1.0
                            
                            # 记录更新的像素数
                            pixels_updated[i] = torch.sum(circle_mask).item()
                            
                            # 记录单个环境的更新
                            if self.current_step % self.log_frequency == 0:
                                self._log("update_pen_on_plane", "更新笔迹像素矩阵", {
                                    "env_idx": env_idx,
                                    "center": (center_x, center_y), 
                                    "radius": radius,
                                    "坐标范围": f"x:[{x_min},{x_max}], y:[{y_min},{y_max}]",
                                    "更新像素数": pixels_updated[i].item()
                                }, "DEBUG", env_idx)
                        else:
                            self._log("update_pen_on_plane", "计算得到的圆内无像素", {
                                "env_idx": env_idx,
                                "center": (center_x, center_y), 
                                "radius": radius,
                                "坐标范围": f"x:[{x_min},{x_max}], y:[{y_min},{y_max}]"
                            }, "WARNING", env_idx)      

            # 记录总体更新状态
            self._log("update_pen_on_plane", "像素矩阵更新完成", {
                "更新环境数": len(non_zero_mask),
                "更新总像素数": torch.sum(pixels_updated).item(),
                "pixel_matrix_非零值": torch.sum(self.plane_pixel_matrix > 0).item()
            }, "DEBUG")
        
        return update_mask   

    def _redraw_all_pen_traces(self):
        """重新绘制所有环境的笔迹点"""
        # 清除当前所有点
        self.debug_draw.clear_points()
        self.debug_draw.clear_lines()
        
        # 如果没有任何笔迹点，直接返回
        if not self.pen_trace_points:
            return
            
        # 收集所有点的信息
        all_points = []
        all_colors = []
        all_sizes = []
        
        # 收集所有线的信息
        all_line_starts = []
        all_line_ends = []
        all_line_colors = []
        all_line_widths = []
        
        # 从所有环境收集点和线的信息
        for env_id, trace_data in self.pen_trace_points.items():
            if 'points' in trace_data and trace_data['points']:
                # 添加点
                all_points.extend(trace_data['points'])
                all_colors.extend(trace_data['colors'])
                all_sizes.extend(trace_data['sizes'])
                
                # 添加线（如果有多个点）
                points = trace_data['points']
                if len(points) > 1:
                    starts = points[:-1]
                    ends = points[1:]
                    all_line_starts.extend(starts)
                    all_line_ends.extend(ends)
                    
                    # 使用同样的蓝色半透明线条
                    line_colors = [(1.0, 0.0, 0.0, 1.0)] * len(starts)
                    line_widths = [3] * len(starts)
                    all_line_colors.extend(line_colors)
                    all_line_widths.extend(line_widths)
        
        # 绘制所有点
        if all_points:
            self.debug_draw.draw_points(all_points, all_colors, all_sizes)
            
        # 绘制所有线
        if all_line_starts:
            self.debug_draw.draw_lines(all_line_starts, all_line_ends, all_line_colors, all_line_widths)

    def _log(self, function_name, message, data=None, level="INFO", env_idx=None):
        """辅助函数：记录格式化日志，并同时保存到文件
        
        Args:
            function_name: 调用日志的函数名
            message: 日志消息
            data: 要记录的数据，会以简洁方式格式化
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
            env_idx: 特定环境ID，默认为None表示所有环境
        """
        # 日志级别检查
        levels = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
        if levels.get(level, 0) < levels.get(self.log_level, 1):
            return
        
        # 每隔指定步数记录详细日志，或者始终记录非INFO级别日志
        if level == "DEBUG" and self.current_step % self.log_frequency != 0:
            return
        
        # 准备环境ID信息
        env_info = f"all:{self.current_step}" if env_idx is None else f"{env_idx}:{self.current_step}"
        
        # 数据格式化
        data_str = ""
        if data is not None:
            if isinstance(data, torch.Tensor):
                # 打印张量的形状和完整内容
                try:
                    # 尝试将张量转换为NumPy数组以便更好地显示
                    data_str = f"shape={data.shape}, content={data.cpu().numpy()}"
                except Exception as e:
                    # 如果转换失败，使用字符串表示
                    data_str = f"shape={data.shape}, content={str(data)}"
            elif isinstance(data, dict):
                # 字典简化显示
                data_str = {}
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        try:
                            data_str[k] = f"shape={v.shape}, content={v.cpu().numpy()}"
                        except Exception as e:
                            data_str[k] = f"shape={v.shape}, content={str(v)}"
                    else:
                        data_str[k] = v
            else:
                data_str = str(data)
        
        # 构建完整日志消息
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_message = f"[{timestamp}][{level}][{function_name}][{env_info}] {message}: {data_str}"
        
        # 打印到控制台
        print(f"[RL_LOG]{log_message}")
        
        # 写入日志文件
        try:
            # 检查当前日志文件是否需要轮换
            if self._check_log_rotation():
                # 如果_check_log_rotation返回True，表示已经切换到新文件
                # 无需重复打开文件，直接写入即可
                pass
                
            # 写入日志
            with open(self.log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"{log_message}\n")
                # 更新计数器
                self.log_count_since_flush += 1
                
                # 定期刷新文件缓冲区，确保日志及时写入磁盘
                if self.log_count_since_flush >= self.flush_frequency:
                    log_file.flush()
                    import os
                    os.fsync(log_file.fileno())  # 强制操作系统将缓冲区数据写入磁盘
                    self.log_count_since_flush = 0
                
        except Exception as e:
            print(f"[ERROR] 写入日志文件失败: {str(e)}")
    
    def _check_log_rotation(self):
        """检查是否需要轮换日志文件
        
        返回:
            bool: 如果轮换了日志文件，返回True；否则返回False
        """
        import os
        import datetime
        
        # 检查文件是否存在并获取大小
        rotated = False
        if os.path.exists(self.log_file_path):
            file_size = os.path.getsize(self.log_file_path)
            
            # 如果超过大小限制，创建新的日志文件
            if file_size >= self.max_log_size:
                # 先关闭当前日志文件（如果有打开的文件句柄）
                try:
                    # 最后一次强制刷新旧文件
                    with open(self.log_file_path, 'a', encoding='utf-8') as old_file:
                        old_file.write(f"\n=== 日志文件大小已达到 {file_size/1024/1024:.2f}MB，分割至新文件 ===\n")
                        old_file.flush()
                        import os
                        os.fsync(old_file.fileno())
                except Exception as e:
                    print(f"[ERROR] 关闭旧日志文件时出错: {str(e)}")
                
                # 更新计数器并创建新文件路径
                self.log_counter += 1
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                old_log_file = self.log_file_path
                self.log_file_path = os.path.join(
                    self.log_dir, 
                    f"manipulator_draw_{self.log_timestamp}_{self.log_counter}.log"
                )
                
                # 创建新的日志文件
                with open(self.log_file_path, 'w', encoding='utf-8') as new_file:
                    new_file.write(f"=== 机械臂绘制环境日志 继续于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (第{self.log_counter}部分) ===\n")
                    new_file.write(f"=== 延续自: {old_log_file} ===\n\n")
                
                print(f"[INFO] 日志文件轮换，新文件: {self.log_file_path}")
                self.log_count_since_flush = 0
                rotated = True
        
        return rotated

    def calculate_endpoint(self, cone_world_pos, cone_world_z, length=None, target_z=None):
        """计算从cone_world_pos出发，沿着cone_world_z方向延伸的终点坐标
        
        参数:
            cone_world_pos (torch.Tensor): 起点坐标，形状为[num_envs, 3]
            cone_world_z (torch.Tensor): 方向向量，形状为[num_envs, 3]
            length (float, optional): 延伸距离，如果提供则使用此距离计算终点
            target_z (float, optional): 目标z坐标，如果提供则计算到达此z坐标所需的距离
            
        返回:
            torch.Tensor: 终点坐标，形状为[num_envs, 3]
        """
        # 确保方向向量是单位向量
        direction_norm = torch.norm(cone_world_z, dim=1, keepdim=True)
        normalized_direction = cone_world_z / (direction_norm + 1e-6)  # 添加小量避免除零
        
        if length is not None:
            # 如果指定了长度，直接计算终点
            endpoint = cone_world_pos + normalized_direction * length
        elif target_z is not None:
            # 如果指定了目标z坐标，计算到达该z坐标所需的距离
            # 对于每个环境，计算从起点到目标z平面的距离
            # 使用向量方程: P = P0 + t*D，其中P0是起点，D是方向向量，t是参数
            # 对于z坐标: target_z = P0_z + t*D_z
            # 因此: t = (target_z - P0_z) / D_z
            
            # 避免除零，添加小量
            z_direction = normalized_direction[:, 2] + 1e-6
            
            # 计算参数t
            t = (target_z - cone_world_pos[:, 2]) / z_direction
            
            # 计算终点坐标
            endpoint = cone_world_pos + normalized_direction * t.unsqueeze(1)
        else:
            # 如果既没有指定长度也没有指定目标z坐标，则返回起点
            endpoint = cone_world_pos
        
        return endpoint

    def close(self):
        """关闭环境时的清理操作"""
        
        # 记录统计信息
        self._log("close", "训练统计", {
            "总环境数": self.num_envs,
            "总训练步数": self.current_step,
            "完成圆形的环境数": torch.sum(self.similarity_score > 0.8).item(),
            "最高相似度": torch.max(self.similarity_score).item() if self.num_envs > 0 else 0
        }, "INFO")
        
        # 关闭日志文件
        try:
            import datetime
            with open(self.log_file_path, 'a', encoding='utf-8') as f:
                f.write(f"\n=== 训练完成于 {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                f.write(f"总环境数: {self.num_envs}\n")
                f.write(f"总训练步数: {self.current_step}\n")
                if self.num_envs > 0:
                    f.write(f"完成圆形的环境数: {torch.sum(self.similarity_score > 0.8).item()}\n")
                    f.write(f"最高相似度: {torch.max(self.similarity_score).item()}\n")
            
            # 最后一次强制刷新
            import os
            os.fsync(open(self.log_file_path, 'a').fileno())
            print(f"[INFO] 日志文件已关闭: {self.log_file_path}")
        except Exception as e:
            print(f"[ERROR] 关闭日志文件时出错: {str(e)}")
        
        # 调用父类的关闭方法
        super().close()