# manipulator_draw_enf_cfg逻辑
## env step过程
1. Pre-process the actions before stepping through the physics.
    新一轮步进，current_step + 1；  
    记录当前位置为last_pen_pos；  
    actions的范围为-1～1,将action缩放到合适值，用于指定机械臂关节位置（关节为以弧度表示），因此aciton需要乘以系数3.14159得到self.actions（关节旋转一周弧度为2派）
2. Apply the actions to the simulator and step through the physics in a decimated manner.
    将self.actions驱动机械臂
3. Compute done signals.
    执行compute_intermediate_values()，更新plane贴图，用最新贴图计算笔迹和圆形的相似度；  
    根据笔迹和圆形相似度（大于阈值），执行训练的轮数（大于阈值），判断是否完成此次训练
4. Compute the reward.
    采用分段学习，不同阶段采用不同奖励机制。
    第一阶段目标，机械臂终端（笔）从任意位置运动到plane平面的xy范围内，奖励机制为笔的位置坐标和plane中心点（xy）距离持续变小，奖励越大  
    第二阶段目标，调整笔尖朝向，和-z轴一致奖励最大
    第三阶段目标，笔和plane接触，实现笔迹绘制
    第四阶段目标，笔迹和圆的相似度，实现圆形绘制
    除了奖励之外，各阶段学习都有惩罚机制，笔的位置在plane平面以下，进行惩罚
5. Reset environments that have terminated or reached the maximum episode length.
    根据第三步的结果，对于已经达到训练完成的env，进行初始化。  
    重置机械臂各关节位置为初始状态  
    重置plane的贴图，清除笔迹  
6. Apply interval events if they are enabled.
    执行一些循环事件，这里没有
7. Compute observations.
    计算当前状态
    由于执行了evn重置，所有需要在执行compute_intermediate_values()，获取当前机械臂、plane的最新状态
    状态包括机械臂各关节位置、速度、笔高度、朝向、笔迹圆度指标
## 存在问题
1. 分阶段学习策略可以借鉴，缩小学习空间，但是第一阶段的学习任务以笔的位置和plane的距离d作为奖励输入，由于机械臂结构特点来看，从机械臂本体初始位置到目标位置，d的变化存在极点，通过随机使得学习突破极点的难度有些大，因此需要改进，改进方法：
    - 仍以笔的位置作为奖励输入，通过规定机械臂初始化位置，限制机械臂运动范围来避开极点
    - 改变奖励输入，使得机械臂运动过程中，输入不存在极点或者存在缓慢极点，增大随机突破的可能


# manipulator_draw_env_cfg_1逻辑
解决不能创建thread的错误：
1. 首先将上面代码中的多线程方案改成单线程运行，在cursor中运行报错
2. 拆分任务，定位报错原因。首先将笔迹相关部分删除，实现笔和plane平面持续接触，在加入笔迹，最后实现圆的绘制
3. 使用marker显示笔的实时位置

# manipulator_draw_env_cfg_2逻辑
1. 更新修改贴图的方式，耗时严重，长时间运行的bug可能和使用贴图的方式有关，将笔迹呈现方式使用sim debug draw的方式实现
2. 每个环境建一个和贴图尺寸一致的像素矩阵，在笔迹更新的时候，更新此矩阵；使用此矩阵计算笔迹得分
3. 当笔迹需要重置时，重置此矩阵

# manipulator_draw_env_cfg_3逻辑
从训练过程来看，分阶段训练存在一些问题，在后面的阶段尝试探索时，会回到前面阶段，前面阶段的reward计算条件设置不完善，导致陷入局部最优，出现reward先上升后持续下降最后保持持平的问题。因此需要合理设计reward计算逻辑，避免在机械臂运动时，reward出现过大的非线性区间，从而陷入局部最优。
对于机械臂绘制图像来说，根据环境和任务，合理设置机械臂初始状态和关节运动范围，从而简化机械臂学习任务的复杂度。
## 机械臂关节运动范围确定
## 机械臂初始位置在各关节运动范围内随机生成
## 重新设计奖励函数。笔的高度，笔的方向，笔和plane持续接触，笔迹和圆的相似度。
## 终止条件，笔超出plane，超过最大迭代轮数，笔迹和圆相似度超过一定值

