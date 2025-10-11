import os
import json
import random
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from .base_generator import BaseGenerator


class CountHillsAndValleysInArrayGenerator(BaseGenerator):
    
    def __init__(self, output_folder="output/count_hills_and_valleys"):
        """
        初始化生成器并设置默认输出文件夹。
        """
        super().__init__(output_folder)

    def _get_difficulty_params(self, difficulty: int) -> Dict[str, Tuple[int, int]]:
        """
        根据难度级别（1-5）获取相应的参数配置。
        """
        if difficulty == 1:
            return { 
                'data_range': (1, 20),
                'list_length_range': (5, 10)
            }
        elif difficulty == 2:
            return {
                'data_range': (1, 40),
                'list_length_range': (10, 20)
            }
        elif difficulty == 3:
            return {
                'data_range': (1, 60),
                'list_length_range': (20, 30)
            }
        elif difficulty == 4:
            return {
                'data_range': (1, 80),
                'list_length_range': (30, 40)
            }
        else:  # difficulty 5
            return {
                'data_range': (1, 100),
                'list_length_range': (40, 70)
            }

    def generate(self, num_cases: int, difficulty: int, output_folder: Optional[str] = None):
        """
        生成指定数量和难度的“计算山峰和山谷”问题。
        """

        # 确定最终的输出路径
        output_folder = output_folder if output_folder else self.output_folder
        images_dir = os.path.join(output_folder, "images")
        os.makedirs(images_dir, exist_ok=True)

        # 获取该难度的参数
        params = self._get_difficulty_params(difficulty)
        
        annotations = []
        for i in range(1, num_cases + 1):
            # 使用 time.time() 作为随机种子
            random.seed(time.time())
            # 生成地形高度列表
            terrain = self._generate_height_list(
                params['list_length_range'], 
                params['data_range'], 
                difficulty
            )
            
            # 定义问题名称和图片路径
            question_name = f"CountHillsAndValleys_difficulty{difficulty}_{i}"
            image_path = os.path.join(images_dir, f"{question_name}.png")
            
            # 生成可视化图像
            self._generate_plot(terrain, image_path)
            
            # 计算答案和复杂度
            answer = self.solve(terrain)
            step_complexity = self._calculate_step_complexity(terrain, difficulty)
            
            # 创建标注信息
            puzzle_data = {
                "index": question_name,
                "category": "CountHillsAndValleys",
                "question": self.prompt_func_image_text(),
                "image": os.path.join("images", f"{question_name}.png"), # 使用相对路径
                "question_language": self.prompt_func_text(terrain),
                "answer": str(answer),
                "initial_state": {"terrain": terrain},
                "difficulty": difficulty,
                "solution": answer,
            }
            annotations.append(puzzle_data)
        
        self.save_annotations(annotations, output_folder)
        return annotations

    def _generate_height_list(self, list_length_range: Tuple[int, int], data_range: Tuple[int, int], difficulty: int) -> List[int]:
        """生成地形高度列表"""
        length = random.randint(*list_length_range)
        
        if difficulty <= 2:
            # 低难度：生成比较明显的波浪形山峰和山谷
            terrain = []
            trend = random.choice(["up", "down"])
            current_height = random.randint(data_range[0], data_range[0] + 3)
            
            while len(terrain) < length:
                if trend == "up":
                    current_height += random.randint(0, 2)
                    if current_height > data_range[1] - 3 or random.random() < 0.3:
                        trend = "down"
                else:  # trend == "down"
                    current_height -= random.randint(0, 2)
                    if current_height < data_range[0] + 3 or random.random() < 0.3:
                        trend = "up"
                
                current_height = max(data_range[0], min(current_height, data_range[1]))
                terrain.append(current_height)
                
                # 偶尔添加平台
                if random.random() < 0.2 and len(terrain) < length:
                    terrain.append(current_height)
            
            return terrain[:length]
        else:
            # 高难度：生成更随机的地形，包含更多的平台
            terrain = [random.randint(*data_range) for _ in range(length)]
            
            num_plateaus = random.randint(1, 3)
            for _ in range(num_plateaus):
                if length >= 3:
                    start_idx = random.randint(0, length - 3)
                    plateau_length = random.randint(2, min(4, length - start_idx))
                    plateau_height = random.randint(*data_range)
                    for i in range(start_idx, start_idx + plateau_length):
                        terrain[i] = plateau_height
            return terrain

    def _generate_plot(self, data: List[int], output_path: str):
            """生成绿色山脉主题的可视化（相邻数值不同时圆润转角，相同时保持平坦）"""
            if not data: 
                    return
            # 设置背景色调 - 使用浅蓝天空色
            background_color = '#e6f3ff'
            plt.rcParams['axes.facecolor'] = background_color
            plt.rcParams['figure.facecolor'] = background_color
            fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
            fig.patch.set_facecolor(background_color)
            ax.set_facecolor(background_color)
        
            # 创建插值点，使曲线在数值变化处圆润，在平台处保持平直
            x_original = np.arange(len(data))
            y_original = np.array(data)
        
            # 创建精细的插值点
            x_plot = []
            y_plot = []
        
            for i in range(len(data)):
                    # 添加原始点
                    x_plot.append(i)
                    y_plot.append(data[i])
                
                    # 如果不是最后一个点，添加插值点
                    if i < len(data) - 1:
                            # 如果相邻两点高度不同，添加插值点实现圆润效果
                            if data[i] != data[i+1]:
                                    # 在两点之间添加额外的点来创建圆润效果
                                    for j in range(1, 5):
                                            t = j / 5
                                            # 使用三次方插值实现圆润效果
                                            x_interp = i + t
                                            y_interp = data[i] + (data[i+1] - data[i]) * (3*t**2 - 2*t**3)
                                            x_plot.append(x_interp)
                                            y_plot.append(y_interp)
                                        
            # 转换为numpy数组以便绘图
            x_plot = np.array(x_plot)
            y_plot = np.array(y_plot)
        
            # 使用绿色渐变
            mountain_colors = ['#90cfa0','#b0efc0']
        
            # 创建多层山脉效果
            num_layers = 10
            alpha_values = np.linspace(0.95, 0.4, num_layers)
        
            # 调整offset，使得层与层之间的间距更小
            base_offset = 2 
            for i in range(num_layers):
                    offset = i * (base_offset / num_layers) * 4
                    y_offset = np.maximum(y_plot - offset, 0)
                
                    # 使用更平滑的颜色渐变
                    color_index = int(i / (num_layers - 1) * (len(mountain_colors) - 1))
                    current_color = mountain_colors[color_index]
                
                    # 使用插值后的数据点绘制
                    ax.fill_between(x_plot, 0, y_offset, 
                                                color=current_color, 
                                                alpha=alpha_values[i], 
                                                linewidth=0)
                
            # 绘制山脉主轮廓线
            ax.plot(x_plot, y_plot, color='#90cfa0', linewidth=2, alpha=0.8, zorder=5)
        
            # 添加数值标签（在原始数据点位置）
            max_val = max(data)
            for i, val in enumerate(data):
                    ax.text(i, val + max_val * 0.03, f'{int(val)}', 
                                    ha='center', va='bottom', 
                                    fontsize=8, fontweight='bold', 
                                    color='red', zorder=15)
                
            # 设置坐标轴
            ax.set_xlim(-0.5, len(data) - 0.5)
            ax.set_ylim(0, max_val * 1.35)
        
            # 隐藏坐标轴刻度和边框
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                    spine.set_visible(False)
                
            plt.tight_layout(pad=0.5)
            plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                                    facecolor=background_color, edgecolor='none')
            plt.close(fig)

    def solve(self, terrain: List[int]) -> int:
        """计算地形中山峰和山谷的数量"""
        return self.count_hills_and_valleys(terrain)

    def _calculate_step_complexity(self, terrain: List[int], difficulty: int) -> int:
        """计算计算山峰山谷问题的解题步骤复杂度"""
        n = len(terrain)
        if n < 3: return 10

        # 1. 操作复杂度
        traversal_operations = n
        # 简化版复杂度，因为最坏情况不常出现
        comparison_operations = n * (n / 4) 
        base_complexity = n * 2
        
        # 2. 数据特性复杂度
        plateau_count = 0
        if n > 1:
            # 计算去重后的数组长度，差值反映平台数量
            unique_successive_len = len([terrain[i] for i in range(len(terrain)) if i == 0 or terrain[i] != terrain[i-1]])
            plateau_count = n - unique_successive_len
            
        actual_count = self.count_hills_and_valleys(terrain)
        height_changes = sum(abs(terrain[i] - terrain[i-1]) for i in range(1, n))
        avg_change = height_changes / (n-1)
        
        # 3. 级别调整
        level_factor = 1 + (difficulty - 1) * 0.2
        
        # 4. 计算最终复杂度
        data_factor = 1 + (plateau_count / n * 2) + (actual_count / 5) + (avg_change / 10)
        
        step_complexity = int(
            (base_complexity + traversal_operations + comparison_operations) * data_factor * level_factor
        )
        return max(10, step_complexity)

    @staticmethod
    def count_hills_and_valleys(nums: List[int]) -> int:
        """计算数组中山峰和山谷的数量"""
        # 首先移除连续的重复项，因为它们属于同一个平台
        if not nums:
            return 0
        unique_nums = [nums[0]]
        for i in range(1, len(nums)):
            if nums[i] != nums[i-1]:
                unique_nums.append(nums[i])

        if len(unique_nums) < 3:
            return 0
            
        count = 0
        for i in range(1, len(unique_nums) - 1):
            left, mid, right = unique_nums[i-1], unique_nums[i], unique_nums[i+1]
            # 检查山峰
            if mid > left and mid > right:
                count += 1
            # 检查山谷
            elif mid < left and mid < right:
                count += 1
        return count


    @staticmethod
    def prompt_func_image_text() -> str:
        return (
            "Here is a terrain made of bars. "
            "Hill: A flat or raised area where the land right before it is lower, and the land right after it is lower too. "
            "Valley: A flat or dipped area where the land right before is higher, and the land right after is higher too. "
            "Neighboring bars with the same height count as part of the same hill/valley. "
            "Calculate the number of hills and valleys and return the sum."
        )

    @staticmethod
    def prompt_func_text(terrain: List[int]) -> str:
        return (
            f"Given a terrain represented by the integer array {terrain}. "
            "Hill: A flat or raised area where the land right before it is lower, and the land right after it is lower too. "
            "Valley: A flat or dipped area where the land right before is higher, and the land right after is higher too. "
            "Neighboring bars with the same height count as part of the same hill/valley. "
            "Calculate the number of hills and valleys and return the sum."
        )

# =============================================================================
# 3. 示例用法 (Updated)
# =============================================================================
if __name__ == "__main__":
    # 实例化生成器
    generator = CountHillsAndValleysInArrayGenerator()
    # generator.init(output_folder="newtasks/CountHillsAndValleys")
    
    num_cases_per_level = 5 # 为每个级别生成5个案例
    
    # 为每个难度级别生成测试用例
    for difficulty_level in range(1, 6):
        print(f"--- 正在生成 CountHillsAndValleys 难度级别 {difficulty_level} 的测试用例 ---")
        generator.generate(
            num_cases=num_cases_per_level,
            difficulty=difficulty_level
        )
    
    print("\n所有难度级别的测试用例生成完成！")