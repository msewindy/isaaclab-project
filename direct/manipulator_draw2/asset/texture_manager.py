# flake8: noqa
import os
import numpy as np
import torch
from PIL import Image, ImageDraw
from pxr import Sdf, UsdShade, Gf, Usd, UsdGeom
from omni.usd._impl import create_material_input
import carb

class TextureManager:
    def __init__(self):
        self.temp_dir = "/tmp/isaac_textures"
        self.max_history_files = 10  # 保留最近的文件数量
        self.runtime_texture = None
        self.default_texture_path = os.path.join(self.temp_dir, "default_texture.png")
        
        # 确保临时目录存在
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 为每个环境创建单独的目录
        self.env_dirs = {}
        
        # 清理所有旧的纹理文件
        self._cleanup_all_files()
        
        # 创建默认纹理
        self._create_default_texture()
        
    def _cleanup_all_files(self):
        """清理所有旧的纹理文件"""
        try:
            print(f"[DEBUG] 开始清理临时目录: {self.temp_dir}")
            for filename in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"[DEBUG] 已删除文件: {file_path}")
                    elif os.path.isdir(file_path) and "env_" in filename:
                        # 清理环境子目录
                        for subfile in os.listdir(file_path):
                            sub_path = os.path.join(file_path, subfile)
                            if os.path.isfile(sub_path):
                                os.remove(sub_path)
                                print(f"[DEBUG] 已删除文件: {sub_path}")
                except Exception as e:
                    print(f"[WARNING] 删除文件 {file_path} 时出错: {str(e)}")
            print("[DEBUG] 临时目录清理完成")
        except Exception as e:
            print(f"[ERROR] 清理临时目录时出错: {str(e)}")
            
    def _create_default_texture(self):
        """创建默认的白色纹理"""
        try:
            print("[DEBUG] 开始创建默认纹理")
            # 创建一个白色背景的RGBA图像 - 使用整数表示白色
            img = Image.new('RGBA', (1024, 1024), 0xFFFFFFFF)  # 白色，带完全不透明的alpha
            img.save(self.default_texture_path)
            print(f"[DEBUG] 默认纹理已创建: {self.default_texture_path}")
        except Exception as e:
            print(f"[ERROR] 创建默认纹理时出错: {str(e)}")
            
    def _get_texture_path_for_step(self, step):
        """根据时间步生成纹理文件路径"""
        return os.path.join(self.temp_dir, f"runtime_texture_step_{step}.png")
    
    def _get_env_dir(self, env_idx):
        """获取指定环境的目录，如果不存在则创建"""
        if env_idx not in self.env_dirs:
            env_dir = os.path.join(self.temp_dir, f"env_{env_idx}")
            os.makedirs(env_dir, exist_ok=True)
            self.env_dirs[env_idx] = env_dir
            # 复制默认纹理到环境目录
            default_env_texture = os.path.join(env_dir, "default_texture.png")
            if not os.path.exists(default_env_texture):
                import shutil
                shutil.copy(self.default_texture_path, default_env_texture)
                print(f"[DEBUG] 为环境 {env_idx} 创建默认纹理: {default_env_texture}")
        return self.env_dirs[env_idx]
    
    def _get_texture_path_for_env_step(self, env_idx, step):
        """根据环境和时间步生成纹理文件路径"""
        env_dir = self._get_env_dir(env_idx)
        return os.path.join(env_dir, f"runtime_texture_step_{step}.png")
    
    def _get_default_texture_for_env(self, env_idx):
        """获取指定环境的默认纹理路径"""
        env_dir = self._get_env_dir(env_idx)
        return os.path.join(env_dir, "default_texture.png")

    def _cleanup_old_files(self):
        """清理旧的纹理文件，只保留最近的几个"""
        try:
            # 获取所有纹理文件
            texture_files = [f for f in os.listdir(self.temp_dir) if f.startswith("runtime_texture_step_")]
            texture_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            
            # 删除旧文件
            while len(texture_files) > self.max_history_files:
                old_file = texture_files.pop(0)
                try:
                    os.remove(os.path.join(self.temp_dir, old_file))
                    print(f"[DEBUG] 已删除旧文件: {old_file}")
                except Exception as e:
                    print(f"[WARNING] 删除文件 {old_file} 时出错: {str(e)}")
        except Exception as e:
            print(f"[ERROR] 清理旧文件时出错: {str(e)}")

    def _cleanup_old_env_files(self, env_idx):
        """清理指定环境的旧纹理文件，只保留最近的几个"""
        try:
            env_dir = self._get_env_dir(env_idx)
            # 获取所有纹理文件
            texture_files = [f for f in os.listdir(env_dir) if f.startswith("runtime_texture_step_")]
            texture_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            
            # 删除旧文件
            while len(texture_files) > self.max_history_files:
                old_file = texture_files.pop(0)
                try:
                    os.remove(os.path.join(env_dir, old_file))
                    print(f"[DEBUG] 已删除环境 {env_idx} 的旧文件: {old_file}")
                except Exception as e:
                    print(f"[WARNING] 删除环境 {env_idx} 的文件 {old_file} 时出错: {str(e)}")
        except Exception as e:
            print(f"[ERROR] 清理环境 {env_idx} 的旧文件时出错: {str(e)}")
            
    def _modify_texture(self, uv_points_center, uv_points_radius, plane_size, current_step):
        """修改纹理 - 在当前纹理上画圆"""
        try:
            # 首先检查参数是否为None
            if uv_points_center is None or uv_points_radius is None or plane_size is None:
                print("[ERROR] UV点参数或平面尺寸为None，无法修改纹理")
                return None
                
            # 获取当前纹理路径
            current_texture_path = self._get_texture_path_for_step(current_step)
            
            # 检查是否有有效的纹理可用
            valid_prev_texture = None
            for step in range(current_step - 1, -1, -1):
                check_path = self._get_texture_path_for_step(step)
                if os.path.exists(check_path):
                    valid_prev_texture = check_path
                    print(f"[DEBUG] 找到最近的有效纹理文件: {check_path}")
                    break
            
            if valid_prev_texture is None:
                print("[DEBUG] 未找到任何有效的纹理文件，使用默认纹理")
                valid_prev_texture = self.default_texture_path
            
            # 使用找到的有效纹理文件
            print(f"[DEBUG] 使用纹理文件作为基础: {valid_prev_texture}")
            self.runtime_texture = Image.open(valid_prev_texture).convert("RGBA")
            
            # 创建绘图对象
            draw = ImageDraw.Draw(self.runtime_texture)
            
            # 解包中心点、半径信息和平面尺寸
            center_u, center_v = uv_points_center
            radius_u, radius_v = uv_points_radius
            plane_width, plane_height = plane_size
            
            # 转换为图像坐标
            x = int(center_u * self.runtime_texture.width)
            y = int((1.0 - center_v) * self.runtime_texture.height)  # 反转Y坐标
            
            # 计算图像空间半径 - 考虑平面尺寸进行归一化
            # 计算真实半径对应的UV空间半径（以像素为单位）
            # 如果radius_u已经是归一化的UV空间半径，就直接乘以纹理宽度
            if plane_width and plane_width > 0:
                radius_x = max(1, int(radius_u * self.runtime_texture.width / plane_width))
            else:
                radius_x = max(1, int(radius_u * self.runtime_texture.width))
                
            if plane_height and plane_height > 0:
                radius_y = max(1, int(radius_v * self.runtime_texture.height / plane_height))
            else:
                radius_y = max(1, int(radius_v * self.runtime_texture.height))

            # 取x和y方向半径的平均值作为最终半径
            effective_radius = max(1, int((radius_x + radius_y) / 2))
            print(f"[DEBUG] 计算得到的像素半径: X={radius_x}, Y={radius_y}, 最终={effective_radius}")
            
            # 确保坐标在有效范围内
            x = max(effective_radius, min(x, self.runtime_texture.width - effective_radius))
            y = max(effective_radius, min(y, self.runtime_texture.height - effective_radius))
            
            # 添加调试信息
            print(f"[DEBUG] UV坐标: ({center_u:.4f}, {center_v:.4f}) -> 图像坐标: ({x}, {y})")
            print(f"[DEBUG] UV半径: ({radius_u:.4f}, {radius_v:.4f}) -> 像素半径: {effective_radius}")
            print(f"[DEBUG] 平面尺寸: 宽度={plane_width:.4f}, 高度={plane_height:.4f}")
            
            # 绘制椭圆
            draw.ellipse(
                [x - effective_radius, y - effective_radius, 
                x + effective_radius, y + effective_radius], 
                fill=(0, 0, 0, 255),  # 黑色
                outline=(0, 0, 0, 255)  # 黑色边框
            )
            
            # 保存修改后的纹理
            self.runtime_texture.save(current_texture_path)
            print(f"[DEBUG] 新纹理已保存: {current_texture_path}")
            
            return current_texture_path
            
        except Exception as e:
            print(f"[ERROR] 修改纹理时发生错误: {str(e)}")
            import traceback
            print(f"[ERROR] 错误堆栈: {traceback.format_exc()}")
            return None

    
    def _modify_texture_per_env(self, env_idx, uv_points_center, uv_points_radius, plane_size, current_step):
        """修改特定环境的纹理，添加新的笔迹"""
        try:
            # 首先检查参数是否为None
            if uv_points_center is None or uv_points_radius is None or plane_size is None:
                print("[ERROR] UV点参数或平面尺寸为None，无法修改纹理")
                return None     
                
            #print(f"[DEBUG] 开始修改环境 {env_idx} 的纹理，当前步数: {current_step}")
            
            # 获取当前和上一个纹理的路径
            current_texture_path = self._get_texture_path_for_env_step(env_idx, current_step)
            prev_texture_path = self._get_texture_path_for_env_step(env_idx, current_step - 1)
            
            print(f"[DEBUG] 环境 {env_idx} 当前纹理路径: {current_texture_path}")
            print(f"[DEBUG] 环境 {env_idx} 上一个纹理路径: {prev_texture_path}")
            
            # 查找最近的有效纹理文件
            valid_prev_texture = None
            for step in range(current_step - 1, -1, -1):
                check_path = self._get_texture_path_for_env_step(env_idx, step)
                if os.path.exists(check_path):
                    valid_prev_texture = check_path
                    print(f"[DEBUG] 找到环境 {env_idx} 最近的有效纹理文件: {check_path}")
                    break
            
            if valid_prev_texture is None:
                print(f"[DEBUG] 未找到环境 {env_idx} 的任何有效纹理文件，使用默认纹理")
                valid_prev_texture = self._get_default_texture_for_env(env_idx)
            
            # 使用找到的有效纹理文件
            print(f"[DEBUG] 使用环境 {env_idx} 的纹理文件作为基础: {valid_prev_texture}")
            env_texture = Image.open(valid_prev_texture).convert("RGBA")


                        # 创建绘图对象
            draw = ImageDraw.Draw(env_texture)
            
            # 解包中心点、半径信息和平面尺寸
            center_u, center_v = uv_points_center
            radius_u, radius_v = uv_points_radius
            plane_width, plane_height = plane_size
            
            # 转换为图像坐标
            x = int(center_u * env_texture.width)
            y = int((1.0 - center_v) * env_texture.height)  # 反转Y坐标
            
            # 计算图像空间半径 - 考虑平面尺寸进行归一化
            # 计算真实半径对应的UV空间半径（以像素为单位）
            # 如果radius_u已经是归一化的UV空间半径，就直接乘以纹理宽度
            if plane_width and plane_width > 0:
                radius_x = max(1, int(radius_u * env_texture.width / plane_width))
            else:
                radius_x = max(1, int(radius_u * env_texture.width))
                
            if plane_height and plane_height > 0:
                radius_y = max(1, int(radius_v * env_texture.height / plane_height))
            else:
                radius_y = max(1, int(radius_v * env_texture.height))

            # 取x和y方向半径的平均值作为最终半径
            effective_radius = max(1, int((radius_x + radius_y) / 2))
            print(f"[DEBUG] 计算得到的像素半径: X={radius_x}, Y={radius_y}, 最终={effective_radius}")
            
            # 确保坐标在有效范围内
            x = max(effective_radius, min(x, env_texture.width - effective_radius))
            y = max(effective_radius, min(y, env_texture.height - effective_radius))
            
            # 添加调试信息
            print(f"[DEBUG] UV坐标: ({center_u:.4f}, {center_v:.4f}) -> 图像坐标: ({x}, {y})")
            print(f"[DEBUG] UV半径: ({radius_u:.4f}, {radius_v:.4f}) -> 像素半径: {effective_radius}")
            print(f"[DEBUG] 平面尺寸: 宽度={plane_width:.4f}, 高度={plane_height:.4f}")
            
            # 绘制椭圆
            draw.ellipse(
                [x - effective_radius, y - effective_radius, 
                x + effective_radius, y + effective_radius], 
                fill=(0, 0, 0, 255),  # 黑色
                outline=(0, 0, 0, 255)  # 黑色边框
            )
            
            # 保存修改后的纹理
            env_texture.save(current_texture_path)
            print(f"[DEBUG] 新纹理已保存: {current_texture_path}")

            # 清理临时文件
            self._cleanup_old_env_files(env_idx)
            
            return current_texture_path
            
        except Exception as e:
            print(f"[ERROR] 修改纹理时发生错误: {str(e)}")
            import traceback
            print(f"[ERROR] 错误堆栈: {traceback.format_exc()}")
            return None

            
    def _update_material(self, material, texture_path, time_code):
        """更新材质的纹理（旧的单一环境版本，为兼容性保留）"""
        try:
            print(f"[DEBUG] 开始更新材质，纹理路径: {texture_path}")
            print(f"[DEBUG] 时间码: {time_code}")
            
            # 获取材质的所有着色器
            for shader in Usd.PrimRange(material):
                if shader.IsA(UsdShade.Shader):
                    shader_node = UsdShade.Shader(shader)
                    
                    # 获取纹理输入
                    file_input = shader_node.GetInput("diffuse_texture")
                    if file_input:
                        # 设置新的纹理路径
                        abs_texture_path = os.path.abspath(texture_path)
                        print(f"[DEBUG] 设置纹理路径: {abs_texture_path}")
                        file_input.Set(abs_texture_path, time=time_code)
                        print("[DEBUG] 材质更新成功")
                        return True
            
            print("[WARNING] 未找到合适的着色器节点")
            return False
            
        except Exception as e:
            print(f"[ERROR] 更新材质时发生错误: {str(e)}")
            import traceback
            print(f"[ERROR] 错误堆栈: {traceback.format_exc()}")
            return False
    
    def _update_material_per_env(self, env_idx, material, texture_path, time_code, step):
        """更新特定环境的材质纹理"""
        try:
            print(f"[DEBUG] 开始更新环境 {env_idx} 的材质，纹理路径: {texture_path}")
            print(f"[DEBUG] 环境 {env_idx} 时间码: {time_code}")

            #如果texture_path为默认纹理路径，则将此贴图报错到环境目录下，以step命名
            if texture_path == self.default_texture_path:
                texture_path_new = self._get_texture_path_for_env_step(env_idx, step)
                env_texture = Image.open(self.default_texture_path).convert("RGBA")
                env_texture.save(texture_path_new)
                print(f"[DEBUG] 默认纹理已保存: {texture_path_new}")
            
            # 获取材质的所有着色器
            for shader in Usd.PrimRange(material):
                if shader.IsA(UsdShade.Shader):
                    shader_node = UsdShade.Shader(shader)
                    
                    # 获取纹理输入
                    file_input = shader_node.GetInput("diffuse_texture")
                    if file_input:
                        # 设置新的纹理路径
                        abs_texture_path = os.path.abspath(texture_path)
                        print(f"[DEBUG] 环境 {env_idx} 设置纹理路径: {abs_texture_path}")
                        file_input.Set(abs_texture_path, time=time_code)
                        print(f"[DEBUG] 环境 {env_idx} 材质更新成功")
                        return True
            
            print(f"[WARNING] 环境 {env_idx} 未找到合适的着色器节点")
            return False
            
        except Exception as e:
            print(f"[ERROR] 更新环境 {env_idx} 材质时发生错误: {str(e)}")
            import traceback
            print(f"[ERROR] 错误堆栈: {traceback.format_exc()}")
            return False

    def create_material(self, stage, material_path):
        """创建带有纹理的材质"""
        try:
            print(f"[DEBUG] 开始创建材质: {material_path}")
            
            # 创建材质
            material = UsdShade.Material.Define(stage, material_path)
            if not material:
                print("[ERROR] 无法创建材质")
                return None
                
            # 创建OmniPBR shader
            shader = UsdShade.Shader.Define(stage, f"{material_path}/OmniPBR")
            if not shader:
                print("[ERROR] 无法创建shader")
                return None
                
            # 设置shader的实现源和标识符
            shader.GetImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
            shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
            shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
            
            # 设置默认颜色为白色
            shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
            
            # 设置默认纹理
            if os.path.exists(self.default_texture_path):
                abs_texture_path = os.path.abspath(self.default_texture_path)
                shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(abs_texture_path)
                print(f"[DEBUG] 设置默认纹理: {abs_texture_path}")
            else:
                print("[WARNING] 默认纹理文件不存在")
            
            # 连接shader到材质
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            
            print("[DEBUG] 材质创建成功")
            return material.GetPrim()
            
        except Exception as e:
            print(f"[ERROR] 创建材质时发生错误: {str(e)}")
            import traceback
            print(f"[ERROR] 错误堆栈: {traceback.format_exc()}")
            return None
