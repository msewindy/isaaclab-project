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
from omni.isaac.lab.utils.math import sample_uniform
import omni.isaac.lab.sim as sim_utils
import cv2
import numpy as np
from PIL import Image

@configclass
class ManipulatorDrawEnvCfg(DirectRLEnvCfg):
   """
   """
    # env
   decimation = 2  #sim仿真执行2个周期，action control执行一次
   episode_length_s = 16.666#一段记录的长度，episode_length_step = ceil(episode_length_s / (decimation_rate * physics_time_step))

    # simulation
   sim: SimulationCfg = SimulationCfg(
        dt=1 / 60, #物理世界的更新间隔，默认1.0/60
        render_interval=decimation, #在渲染一帧时，物理世界更新的步数
    )
   action_space = 6 #机械臂6个关节
   observation_space = 19
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
        #笔的世界位置坐标
        self.pen_pos = torch.zeros((self.num_envs, 3), device=self.device)
        #笔尖世界坐标方向
        self.pen_dir = torch.zeros((self.num_envs, 3), device=self.device)
        #笔持续在plane上的步数
        self.pen_on_plane = torch.zeros(self.num_envs, device=self.device)

        #各环境table上plane的中心点世界坐标
        self.table_center_pos = self.calculate_table_center_pos()

        # 日志级别控制
        self.log_level = "WARNING"  # 可选：DEBUG, INFO, WARNING, ERROR
        self.log_frequency = 2  # 每隔多少步记录一次详细日志

        # 添加观察归一化所需变量
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.joint_pos_scale = 3.14159  # 假设关节范围在±π
        self.joint_vel_scale = 10.0     # 假设速度范围在±10

        #绘制图形的评价系数
        self.similarity_score = torch.zeros((self.num_envs), device=self.device)
        self.iou = torch.zeros((self.num_envs), device=self.device)
        self.hu_similarity = torch.zeros((self.num_envs), device=self.device)
        self.circularity = torch.zeros((self.num_envs), device=self.device)



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
        self._log("_pre_physics_step", "_pre_physics_step", level="WARNING")
        self.current_step += 1
        self.last_pen_pos = self.pen_pos
        self.actions = torch.clamp(actions, self.robot_dof_lower_limits, self.robot_dof_upper_limits)     
        self._log("_pre_physics_step", "执行动作", self.actions, "DEBUG")

    
    #计算观察结果
    def _get_observations(self) -> dict:
        """
        """
        self._log("_get_observations", "_get_observations", level="WARNING")
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
            (pen_height > 0.02) & (pen_height < 0.05),
            pen_height * 5,  # 近距离放大5倍
            -0.5  # 远距离压缩
        ).unsqueeze(1)

        pen_dir_norm = self.pen_dir

        iou_norm = torch.pow(self.iou, 0.5).unsqueeze(1)          # 开方放大小值
        hu_sim_norm = torch.pow(self.hu_similarity, 0.5).unsqueeze(1)
        circ_norm = torch.pow(self.circularity, 0.5).unsqueeze(1)

        obs = torch.cat(
            (                
                joint_pos_norm,    # [-1,1]
                joint_vel_norm,    # [-1,1]
                pen_height_norm,   # [0,1]特别处理
                pen_dir_norm, 
                iou_norm,          # [0,1]
                hu_sim_norm,       # [0,1]
                circ_norm                   
            ),
            dim=-1,
        )         

        return {"policy": obs}

    
    #定义action执行逻辑
    def _apply_action(self) -> None:
        """
        """
        self._log("_apply_action", "_apply_action", level="WARNING")
        self._log("_apply_action", "应用动作到机械臂", self.actions, "DEBUG")
        self.robot.set_joint_position_target(self.actions)

    #计算奖励
    def _get_rewards(self) -> torch.Tensor:
        """
        计算rewards
        """
        self._log("_get_rewards", "_get_rewards", level="WARNING")
        rewards = self.compute_rewards()
        self._log("_get_rewards", "计算的奖励值", rewards, "DEBUG")
        return rewards

    #定义迭代结束逻辑
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        """
        self._log("_get_dones", "_get_dones", level="WARNING")
        ##action执行后，在判断是否终止前，需要计算指标
        self.compute_intermediate_values()
        #绘制完成       
        terminated = self.similarity_score > 0.80
        #达到最大迭代次数
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        
        self._log("_get_dones", "终止状态", {
            "terminated": terminated, 
            "truncated": truncated, 
            "sucess_steps": self.pen_on_plane
        }, "INFO")
        
        return terminated, truncated

    #定义仿真实例重置逻辑
    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        """
        self._log("_reset_idx", "_reset_idx", level="WARNING")
        if env_ids is not None:
            self._log("_reset_idx", "重置环境", env_ids, "INFO")
            
            super()._reset_idx(env_ids)
            #重置机械臂关节位置
            joint_pos = self.robot.data.default_joint_pos[env_ids] \
             + sample_uniform(
                 -0.125,
                 0.125,
                 (len(env_ids), self.robot.num_joints),
                 self.device,
             )
            joint_vel = torch.zeros_like(joint_pos)
            
            self._log("_reset_idx", "新关节位置", joint_pos, "INFO")
            
            self.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

            #重置
            for env_idx in env_ids:
                try:
                    #获取当前环境下plane的material
                    plane_prim = self.scene.stage.GetPrimAtPath(f'/World/envs/env_{env_idx}/table/table_instanceable/plane/Plane')
                    # 获取材质绑定API
                    material_binding_api = UsdShade.MaterialBindingAPI(plane_prim)
                    if not material_binding_api:
                        self._log("_reset_idx", "无法获取材质绑定API", None, "WARNING", env_idx)
                        continue
                        
                    # 获取所有绑定的材质
                    collection = material_binding_api.GetDirectBindingRel(materialPurpose=UsdShade.Tokens.allPurpose)
                    if not collection:
                        self._log("_reset_idx", "无法获取材质绑定关系", None, "WARNING", env_idx)
                        continue
                        
                    # 获取最强绑定的材质
                    targets = collection.GetTargets()
                    if not targets or len(targets) == 0:
                        self._log("_reset_idx", "未找到绑定的材质", None, "WARNING", env_idx)
                        continue
                        
                    # 获取最后绑定的材质（应该是我们代码中绑定的）
                    material_path = targets[-1]
                    bound_material = UsdShade.Material(plane_prim.GetStage().GetPrimAtPath(material_path))
                    
                    #更新贴图为默认白色贴图
                    self.texture_manager._update_material_per_env(env_idx, bound_material.GetPrim(), self.texture_manager.default_texture_path, Usd.TimeCode(self.current_time), self.current_step)
                    self._log("_reset_idx", "已重置贴图", {"env_idx": env_idx, "material": material_path}, "INFO", env_idx)

                    #重置笔迹和圆相似度指标
                    self.iou[env_idx] = 0.0
                    self.hu_similarity[env_idx] = 0.0
                    self.circularity[env_idx] = 0.0
                    self.similarity_score[env_idx] = 0.0
                    #重置笔在plane上的步数
                    self.pen_on_plane[env_idx] = 0

                except Exception as e:
                    self._log("_reset_idx", f"重置pen on plane 上步数错误: {str(e)}", None, "ERROR", env_idx)
                    import traceback
                    self._log("_reset_idx", f"错误堆栈: {traceback.format_exc()}", None, "ERROR", env_idx)

    #重置观察、奖励使用的变量
    def compute_intermediate_values(self):
        """
        """
        update_texture_ids =self.update_pen_on_plane()
        if update_texture_ids is not None:
            self.compare_origin_draw(update_texture_ids)
                  

    #计算rewards的实现
    def compute_rewards(self):
        """
        计算rewards的实现 - 归一化版本
        """
        # 提取原始数据
        pen_height = self.pen_pos[:, 2]
        pen_dir_factor = self.pen_dir[:, 0]

        #距离奖励
        # 计算笔到plane区域中心(0,0,0)在xy平面的距离
        pen_to_plane_distance = torch.norm(self.pen_pos[:, :2], dim=-1)
        # 和上一步笔到原点的距离对比
        last_pen_to_plane_distance = torch.norm(self.last_pen_pos[:, :2], dim=-1)
        pen_to_plane_distance_diff = pen_to_plane_distance - last_pen_to_plane_distance
        ##笔位置靠近奖励
        pen_pos_reward = torch.where(pen_to_plane_distance_diff < 0.0, pen_to_plane_distance_diff * -1.0, pen_to_plane_distance_diff * 100.0)
          
        
        # 1. 高度奖励归一化处理 - 越接近平面越好(0.0最好)
        # 使用与观察空间相似的非线性变换放大接近平面的信
        
        # 2. 方向奖励处理 - 指向平面越好(-1最好,1最差)
        # 归一化到[-1,0]范围，直接使用
        dir_reward = pen_dir_factor
            
        # 4. 任务阶段判断
        contact_reward = torch.where(
             (pen_height > 0.02) & (pen_height < 0.05),  # 接触平面的阈值
            torch.ones_like(pen_height) * 0.5,  # 接触奖励
            torch.zeros_like(pen_height)  # 未接触无奖励
        )
        # 判断笔是否在plane范围内
        # 创建每个维度的范围判断
        x_in_range = (self.pen_pos[:, 0] >= self.table_center_pos[:, 0] - 0.4) & (self.pen_pos[:, 0] <= self.table_center_pos[:, 0] + 0.4)
        y_in_range = (self.pen_pos[:, 1] >= self.table_center_pos[:, 1] - 0.4) & (self.pen_pos[:, 1] <= self.table_center_pos[:, 1] + 0.4)

        # 综合判断所有维度是否都在范围内
        pen_in_plane_range = x_in_range & y_in_range

        iou_reward = torch.pow(self.iou, 0.5) * 5.0
        hu_reward = torch.pow(self.hu_similarity, 0.5) * 6.0
        circ_reward = torch.pow(self.circularity, 0.5) * 7.0

        # 初始化奖励
        reward = torch.zeros_like(pen_height)

        # 为每个环境单独计算奖励
        for env_idx in range(self.num_envs):
            # 阶段0,笔在plane xy平面范围内
            if not pen_in_plane_range[env_idx]:
                reward[env_idx] = pen_pos_reward[env_idx]
            elif dir_reward[env_idx] < 0.85:
                reward[env_idx] = 1.0 + dir_reward[env_idx] * 3.0
            else:
                reward[env_idx] = 4.0 + contact_reward[env_idx] * 1.0 + self.pen_on_plane[env_idx] * 0.01 + iou_reward[env_idx] + hu_reward[env_idx] + circ_reward[env_idx]

        action_penalty = torch.sum(self.actions**2, dim=-1)
        reward -= action_penalty * 0.05
        
        # 惩罚计算 - 保持原有结构但调整值
        penalties = {
            "笔位置低于平面": (pen_height < 0.0, -20.0)  # 减小惩罚强度
        }
        
        # 应用惩罚(保持不变)
        for name, (condition, penalty) in penalties.items():
            if torch.any(condition):
                old_reward = reward.clone()
                reward = torch.where(condition, old_reward + penalty, old_reward)
                # 记录惩罚
                if self.current_step % self.log_frequency == 0:
                    applied_count = condition.sum().item()
                    self._log("compute_rewards", f"应用惩罚: {name}", 
                            f"数量: {applied_count}/{self.num_envs}, 惩罚值: {penalty}", 
                            "INFO" if applied_count == 0 else "INFO")
        
        # 记录最终奖励
        self._log("compute_rewards", "最终奖励", reward, "INFO")
        
        return reward
    
    #计算table上plane的中心点世界坐标
    def calculate_table_center_pos(self):
        """
        计算所有env中table上plane的中心点世界坐标
        """

        #获取所有环境table的prim
        table_prims = torch.zeros((self.num_envs, 3), device=self.device)
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

   
       #local，计算贴图上黑色像素是否是一个圆形
    def compare_origin_draw(self, update_texture_ids):
        """
        判断所有环境贴图上的黑色笔迹是否为圆形并计算相似度
        
        优化：严格控制并行度，减少内存使用
        
        返回:
            tuple[Tensor]: 包含所有环境的(iou, hu_similarity, circularity, 综合相似度分数)
        """            
        self._log("compare_origin_draw", "开始比较绘制结果", None, "INFO")        
   
        import os
        
        # 直接使用类变量，不再创建临时变量
        self.iou.zero_()
        self.hu_similarity.zero_()
        self.circularity.zero_()
        self.similarity_score.zero_() 

        # 收集需要更新贴图的环境的纹理路径
        texture_paths = []
        for env_idx in update_texture_ids:
            env_dir = self.texture_manager._get_env_dir(env_idx)
            texture_files = [f for f in os.listdir(env_dir) 
                            if f.startswith("runtime_texture_step_")]
            
            if not texture_files:
                self._log("compare_origin_draw", "没有找到纹理文件，使用默认纹理", None, "WARNING", env_idx)
                texture_path = self.texture_manager._get_default_texture_for_env(env_idx)
            else:
                # 按步骤号排序并获取最新的
                texture_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]), reverse=True)
                texture_path = os.path.join(env_dir, texture_files[0])
                self._log("compare_origin_draw", "使用纹理路径", texture_path, "INFO", env_idx)
            
            texture_paths.append((env_idx, texture_path))
        
        # 使用常规函数顺序处理图像
        self._log("compare_origin_draw", "开始处理图像", {"纹理数量": len(texture_paths)}, "INFO")
        
        # 定义处理逻辑
        def process_image(args):
            env_idx, texture_path = args
            try:
                if env_idx in update_texture_ids:
                    # 读取图像
                    img = np.array(Image.open(texture_path).convert("RGB"))
                    
                    # 转换为灰度图并二值化
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                    
                    # 寻找轮廓
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if not contours:
                        self._log("compare_origin_draw", "未检测到任何轮廓", None, "WARNING", env_idx)
                        return env_idx, (0.0, 0.0, 0.0, 0.0)
                    
                    # 找到最大轮廓
                    max_contour = max(contours, key=cv2.contourArea)
                    contour_area = cv2.contourArea(max_contour)
                    
                    if contour_area < 10000:  # 轮廓太小不计算奖励，仿真将一个点看作圆
                        self._log("compare_origin_draw", "检测到的轮廓面积太小", {"area": contour_area}, "WARNING", env_idx)
                        return env_idx, (0.0, 0.0, 0.0, 0.0)
                    
                    # 拟合圆并计算相关指标
                    (x, y), radius = cv2.minEnclosingCircle(max_contour)
                    center = (int(x), int(y))
                    radius = int(radius)
                    
                    # 计算指标
                    # 1. 创建掩码
                    circle_mask = np.zeros_like(binary)
                    contour_mask = np.zeros_like(binary)
                    
                    cv2.circle(circle_mask, center, radius, (255, 255, 255), -1)
                    cv2.drawContours(contour_mask, [max_contour], 0, (255, 255, 255), -1)
                    
                    # 2. 计算IoU
                    overlap = cv2.bitwise_and(circle_mask, contour_mask)
                    union = cv2.bitwise_or(circle_mask, contour_mask)
                    
                    overlap_area = cv2.countNonZero(overlap)
                    union_area = cv2.countNonZero(union)
                    
                    env_iou = overlap_area / union_area if union_area > 0 else 0.0
                    
                    # 3. 使用Hu矩计算形状相似度
                    contour_moments = cv2.moments(max_contour)
                    contour_hu = cv2.HuMoments(contour_moments)
                    
                    ideal_circle = np.zeros_like(binary)
                    cv2.circle(ideal_circle, center, radius, (255, 255, 255), -1)
                    ideal_moments = cv2.moments(ideal_circle)
                    ideal_hu = cv2.HuMoments(ideal_moments)
                    
                    hu_distance = np.sum(np.abs(np.log(np.abs(contour_hu) + 1e-10) - np.log(np.abs(ideal_hu) + 1e-10)))
                    env_hu_similarity = np.exp(-hu_distance)
                    
                    # 4. 计算圆形度
                    perimeter = cv2.arcLength(max_contour, True)
                    env_circularity = (4 * np.pi * contour_area) / (perimeter ** 2) if perimeter > 0 else 0
                    
                    # 5. 计算综合得分
                    env_similarity_score = 0.4 * env_iou + 0.3 * env_hu_similarity + 0.3 * env_circularity
                    
                    # 保存调试图像 - 减少文件IO，仅保存得分高的图像
                    if env_similarity_score > 0.5:
                        debug_dir = os.path.join(os.path.dirname(texture_path), "circle_detection")
                        os.makedirs(debug_dir, exist_ok=True)
                        
                        result_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.drawContours(result_img, [max_contour], 0, (0, 255, 0), 2)
                        cv2.circle(result_img, center, radius, (0, 0, 255), 2)
                        
                        debug_path = os.path.join(debug_dir, f"circle_detection_{os.path.basename(texture_path)}")
                        cv2.imwrite(debug_path, result_img)
                    
                    return env_idx, (env_iou, env_hu_similarity, env_circularity, env_similarity_score)
                else:
                    return env_idx, (0.0, 0.0, 0.0, 0.0)
            except Exception as e:
                self._log("compare_origin_draw", f"分析纹理时发生错误: {str(e)}", None, "ERROR", env_idx)
                import traceback
                self._log("compare_origin_draw", f"错误堆栈: {traceback.format_exc()}", None, "ERROR", env_idx)
                return env_idx, (0.0, 0.0, 0.0, 0.0)
            finally:
                # 手动清理占用内存的大型对象
                if 'img' in locals(): del img
                if 'gray' in locals(): del gray
                if 'binary' in locals(): del binary
                if 'circle_mask' in locals(): del circle_mask
                if 'contour_mask' in locals(): del contour_mask
                if 'result_img' in locals(): del result_img


        all_results = []

        for texture_path in texture_paths:
            result = process_image(texture_path)
            all_results.append(result)
        
        # 将结果转移到相应的张量中
        for env_idx, (iou_val, hu_val, circ_val, sim_val) in all_results:
            if env_idx in update_texture_ids:
                self.iou[env_idx] = iou_val
                self.hu_similarity[env_idx] = hu_val
                self.circularity[env_idx] = circ_val
                self.similarity_score[env_idx] = sim_val                
                # 只记录显著结果，减少日志量
                if sim_val > 0.5:
                    self._log("compare_origin_draw", "圆形检测结果", {
                        "env_idx": env_idx,
                        "IoU": iou_val,
                        "Hu矩相似度": hu_val,
                        "圆形度": circ_val,
                        "最终相似度分数": sim_val
                    }, "INFO")
        
        # 直接返回类变量的引用，而不是创建新张量
        return self.iou, self.hu_similarity, self.circularity, self.similarity_score

    #更新plane贴图
    def update_pen_on_plane(self):
        """
        更新所有env下的plane的贴图
        
        优化：使用顺序处理代替多线程处理
        """
        self._log("update_pen_on_plane", "开始更新update_pen_on_plane", None, "INFO")
        
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
            return
        
        # 1. 批量获取所有环境的笔的位置和方向 (一次性在GPU上计算)
        # 获取所有tool0的世界位置和方向
        world_pos = self.robot.data.body_pos_w[:, tool0_index]  # [num_envs, 3]
        world_quat = self.robot.data.body_quat_w[:, tool0_index]  # [num_envs, 4]
        
        self._log("update_pen_on_plane", "笔世界位置", world_pos, "INFO")
        self._log("update_pen_on_plane", "笔世界旋转", world_quat, "INFO")
        
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

        # 将PyTorch张量转换为Isaac Sim期望的格式
        #cone_world_pos_np = cone_world_pos.cpu().numpy()
        #cone_world_pos_list = [(pos[0], pos[1], pos[2]) for pos in cone_world_pos_np]

        # 绘制cone
        #draw = _debug_draw.acquire_debug_draw_interface()
        #draw.clear_points()
        #draw.draw_points(cone_world_pos_list, [(1, 0, 0, 1)] * len(cone_world_pos_list), [10] * len(cone_world_pos_list))
        
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
        
        # 存储当前笔的位置信息 (用于观察)
        self.pen_pos = cone_world_pos
        
        # 修复pen_dir的维度问题，使用适当的方法
        # 方案1：将cos_angles扩展为每个环境的特征值
        self.pen_dir = cos_angles.unsqueeze(1).expand(-1, 3)  # [num_envs, 3]
        # 或者使用实际的方向向量
        # self.pen_dir = cone_world_z  # [num_envs, 3]
        
        # 确定哪些环境满足角度条件 (余弦值>0.707，约45度)
        angle_mask = cos_angles > 0.707  # [num_envs]
        
        # 计算所有环境中笔尖到平面的高度 (z坐标)
        pen_origin_z = cone_world_pos[:, 2]  # [num_envs]
        
        # 确定哪些环境满足高度条件 (0<高度<0.05)
        height_mask = (pen_origin_z > 0.02) & (pen_origin_z < 0.05)  # [num_envs]
        
        # 综合条件，确定哪些环境需要更新步数
        update_mask = angle_mask & height_mask  # [num_envs]
        
        # 如果没有环境需要更新，直接返回
        if not torch.any(update_mask):
            return
            
        # 获取需要更新的环境索引
        update_env_indices = torch.nonzero(update_mask).squeeze(-1).cpu().numpy()
        
        for env_idx in range(self.num_envs):
            if env_idx in update_env_indices:
                self.pen_on_plane[env_idx] += 1
            else:
                self.pen_on_plane[env_idx] = 0   

                # 创建处理纹理的函数
        def process_env_texture(env_idx, length, pen_trait_center, cone_world_z_val):
            try:
                # 构建平面prim路径
                plane_prim_path = f'/World/envs/env_{env_idx}/table/table_instanceable/plane/Plane'
                plane_prim = self.scene.stage.GetPrimAtPath(plane_prim_path)
                
                if not plane_prim:
                    self._log("update_plane_texture", f"未找到平面 prim: {plane_prim_path}", None, "WARNING", env_idx)
                    return env_idx, None
                    
                # 计算笔迹半径 (根据高度动态计算)
                pen_trait_rad = 0.025 * (0.5 -length / 0.1)
                
                # 转换为numpy以便后续处理
                pen_trait_center_np = pen_trait_center.cpu().numpy().reshape(3, 1)
                pen_trait_rad_np = float(pen_trait_rad.cpu().numpy())
                
                # 获取材质和纹理信息
                material, texture_path, uv_set = self._get_material_and_texture(plane_prim, time_code)
                
                if not (material and texture_path):
                    self._log("update_plane_texture", f"未找到材质或纹理", None, "WARNING", env_idx)
                    return env_idx, None
                
                # 计算UV点
                uv_points_center, uv_points_radius, plane_size = self._get_uv_at_hit_point(plane_prim, pen_trait_center_np, pen_trait_rad_np, uv_set, time_code)
                
                if not uv_points_center or not uv_points_radius or not plane_size:
                    self._log("update_plane_texture", f"UV点计算失败", None, "WARNING", env_idx)
                    return env_idx, None                

                temp_texture_path = self.texture_manager._modify_texture_per_env(env_idx,uv_points_center, uv_points_radius, plane_size, self.current_step)
                if temp_texture_path:
                    return env_idx, (material, temp_texture_path)
                else:
                    self._log("update_plane_texture", f"纹理修改失败", None, "WARNING", env_idx)
                    return env_idx, None
            
            except Exception as e:
                self._log("update_plane_texture", f"处理贴图时发生错误: {str(e)}", None, "ERROR", env_idx)
                import traceback
                self._log("update_plane_texture", f"错误堆栈: {traceback.format_exc()}", None, "ERROR", env_idx)
                return env_idx, None
        
        # 顺序处理每个需要更新的环境
        all_results = []
        for env_idx in update_env_indices:
            # 获取当前环境的笔尖高度
            length = pen_origin_z[env_idx]
            
            # 计算笔迹中心点 (笔尖在平面上的投影)
            v_unit = cone_world_z[env_idx]
            pen_trait_center = cone_world_pos[env_idx] - length * v_unit
            
            # 处理当前环境的纹理
            result = process_env_texture(env_idx, length, pen_trait_center, v_unit)
            if result is not None:
                all_results.append(result)
        
        # 批量更新材质 (单独处理以避免USD API的并发问题)
        for env_idx, result in all_results:
            if result:
                material, temp_texture_path = result
                # 更新材质 - 使用环境特定的函数
                if not self.texture_manager._update_material_per_env(env_idx, material, temp_texture_path, Usd.TimeCode(self.current_time), self.current_step):
                    self._log("update_plane_texture", f"材质更新失败", None, "WARNING", env_idx)
                
        return update_env_indices

    def _get_material_and_texture(self, prim, time_code):
        """获取材质和纹理信息"""
        try:
            # 获取材质绑定API
            material_binding_api = UsdShade.MaterialBindingAPI(prim)
            if not material_binding_api:
                self._log("get_material_and_texture", "无法获取材质绑定API", None, "WARNING")
                return None, None, None
                
            # 获取所有绑定的材质
            collection = material_binding_api.GetDirectBindingRel(materialPurpose=UsdShade.Tokens.allPurpose)
            if not collection:
                self._log("get_material_and_texture", "无法获取材质绑定关系", None, "WARNING")
                return None, None, None
                
            # 获取最强绑定的材质
            targets = collection.GetTargets()
            if not targets or len(targets) == 0:
                self._log("get_material_and_texture", "未找到绑定的材质", None, "WARNING")
                return None, None, None
                
            # 获取最后绑定的材质（应该是我们代码中绑定的）
            material_path = targets[-1]
            bound_material = UsdShade.Material(prim.GetStage().GetPrimAtPath(material_path))
            if not bound_material:
                self._log("get_material_and_texture", "无法获取绑定的材质", None, "WARNING")
                return None, None, None
                
            # 遍历材质的所有着色器
            for shader in Usd.PrimRange(bound_material.GetPrim()):
                if shader.IsA(UsdShade.Shader):
                    shader_node = UsdShade.Shader(shader)
                    
                    # 获取纹理文件路径
                    file_input = shader_node.GetInput("diffuse_texture")
                    if file_input:
                        texture_path = file_input.Get(time=time_code)
                        return bound_material.GetPrim(), str(texture_path), "st"
            
            self._log("get_material_and_texture", "未找到纹理节点", None, "WARNING")
            return None, None, None
            
        except Exception as e:
            self._log("get_material_and_texture", f"获取材质和纹理时发生错误: {str(e)}", None, "ERROR")
            import traceback
            self._log("get_material_and_texture", f"错误堆栈: {traceback.format_exc()}", None, "ERROR")
            return None, None, None
    
    def _get_uv_at_hit_point(self, plane_prim, center, radius, uv_set, time_code):
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
            ##print(f"[DEBUG] 平面缩放系数: X: {scale_x:.4f}, Y: {scale_y:.4f}, 平均: {plane_scale:.4f}")
            
            # 计算局部坐标系中的半径 - 将世界半径乘以缩放系数
            local_radius = radius * plane_scale
            #print(f"[DEBUG] 世界半径: {radius:.4f} -> 局部半径: {local_radius:.4f}")
            
            # 使用 PrimvarsAPI 获取 UV 坐标
            primvars_api = UsdGeom.PrimvarsAPI(mesh)
            uv_primvar = primvars_api.GetPrimvar(uv_set)
            if not uv_primvar or not uv_primvar.HasValue():
                print(f"[WARNING] 无法获取 UV 坐标数据: {uv_set}")
                return None, None, None
                
            # 获取点和 UV 数据
            points_attr = mesh.GetPointsAttr()
            if not points_attr:
                print("[WARNING] 无法获取点数据")
                return None, None, None
                
            points = points_attr.Get(time=time_code)
            uvs = uv_primvar.Get(time=time_code)
            
            # 检查points和uvs是否为None或bool类型，确保可以进行len操作
            if (points is None or isinstance(points, bool) or uvs is None or isinstance(uvs, bool) or 
                not hasattr(points, "__len__") or not hasattr(uvs, "__len__") or
                len(points) < 4 or len(uvs) < 4):
                print("[WARNING] 点数据或 UV 数据不足或格式不正确")
                return None, None, None
                
            # 获取平面的四个角点
            p0, p1, p2, p3 = points[0], points[1], points[2], points[3]  # 左下、右下、左上、右上
            
            # 计算局部坐标系中的相对位置
            local_pos = np.array([local_center[0], local_center[1], local_center[2]])
            
            # 计算平面的宽度和高度向量
            width_vector = p1 - p0
            height_vector = p2 - p0
            plane_width = np.linalg.norm(width_vector)
            plane_height = np.linalg.norm(height_vector)
            
            #print(f"[DEBUG] 平面尺寸: 宽度: {plane_width:.4f}, 高度: {plane_height:.4f}")
            
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
            
            #print(f"[DEBUG] UV中心点: ({center_u:.4f}, {center_v:.4f})")
            #print(f"[DEBUG] UV半径: U方向: {uv_radius_u:.4f}, V方向: {uv_radius_v:.4f}")
            
            # 返回计算结果，以元组形式
            return (float(center_u), float(center_v)), (float(uv_radius_u), float(uv_radius_v)), (plane_width, plane_height)
            
        except Exception as e:
            print(f"[ERROR] 计算 UV 点时发生错误: {str(e)}")
            import traceback
            print(f"[ERROR] 错误堆栈: {traceback.format_exc()}")
            return None, None, None

    def _log(self, function_name, message, data=None, level="INFO", env_idx=None):
        """辅助函数：记录格式化日志
        
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
                # 对张量简化显示
                if data.numel() <= 50 or level == "INFO":
                    # 小张量完整显示
                    data_str = str(data.detach().cpu().numpy())
                #else:
                    # 大张量只显示形状和部分值
                    #data_str = f"shape={data.shape}, mean={data.mean().item():.4f}, std={data.std().item():.4f}, min={data.min().item():.4f}, max={data.max().item():.4f}"
            elif isinstance(data, dict):
                # 字典简化显示
                data_str = {k: v.shape if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            else:
                data_str = str(data)
        
        print(f"[RL_LOG][{function_name}][{env_info}] {message}: {data_str}")

