import os
import random
import re
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Optional


from .base_generator import BaseGenerator

class LargestRectangleInHistogramGenerator(BaseGenerator):
    def __init__(self, output_folder="output/largest_rectangle_in_histogram"):
        """
        初始化直方图最大矩形问题生成器
        Args:
            output_folder: 输出文件夹路径
        """
        super().__init__(output_folder)

    def _get_difficulty_params(self, difficulty):
        """
        根据难度级别获取相应的参数配置。
        Args:
            difficulty: 难度级别（1-5）
        Returns:
            dict: 包含难度参数的字典
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

    def generate(self, num_cases, difficulty, output_folder=None):
        """
        生成指定数量的直方图最大矩形问题
        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别
        Returns:
            生成的问题列表
        """

        # 获取难度参数
        params = self._get_difficulty_params(difficulty)
        data_range = params['data_range']
        list_length_range = params['list_length_range']
        
        # 确保主输出目录和 images 子目录存在
        os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
        
        problems = []
        for i in range(1, num_cases + 1):
            # 生成谜题
            puzzle = self._generate_single_puzzle(i, data_range, list_length_range, difficulty, output_folder)
            problems.append(puzzle)
        self.save_annotations(problems, output_folder)
        return problems

    def _generate_single_puzzle(self, case_id, data_range, list_length_range, difficulty=None, output_folder=None):
        """生成单个直方图最大矩形问题"""
        # 生成高度列表
        heights = self._generate_height_list(data_range, list_length_range)
        
        # 生成图像
        puzzle_image_path = self.visualize(heights, difficulty=difficulty, output_folder=output_folder, case_id=case_id)
        
        # 计算答案
        answer = self.solve(heights)
        
        # 计算解题步骤复杂度
        step_complexity = self._calculate_step_complexity(heights, difficulty)
        
        # 创建数据点
        return {
            "index": f"LargestRectangleInHistogram_difficulty{difficulty}_{case_id}",
            "category": "LargestRectangleInHistogram",
            "question": self.prompt_func_image_text(heights),
            "image": puzzle_image_path,
            "question_language": self.prompt_func_text(heights),
            "answer": answer,
            "initial_state": {"heights": heights},
            "difficulty": difficulty,
#           "solution": answer,
#           "step": step_complexity  # 添加解题步骤复杂度
        }

    def _generate_height_list(self, data_range, list_length_range) -> List[int]:
        """生成高度列表"""
        # 设置随机种子
        random.seed(time.time())
        
        length = random.randint(*list_length_range)
        return [random.randint(*data_range) for _ in range(length)]

    def visualize(self, heights, difficulty=None, output_folder=None, **kwargs):
        """生成可视化图像并返回相对路径"""
        case_id = kwargs.get('case_id', 0)
        
        # 定义新的图片文件名格式
        image_filename = f"LargestRectangleInHistogram_difficulty_{difficulty}_{case_id}.png"
        
        # 创建图片的完整保存路径
        full_image_path = os.path.join(output_folder, "images", image_filename)
        
        # 调用绘图函数
        self._generate_plot(heights, full_image_path, case_id, difficulty)
        
        # 返回JSON文件中要使用的相对路径
        return os.path.join("images", image_filename).replace("\\", "/")

    def _generate_plot(self, data, output_path, case_id=None,output_folder=None):
        """生成柱状图可视化"""
        # 蓝色渐变
        gradient_colors = [(0.1, 0.5, 0.8), (0.2, 0.7, 0.9)]  # 深蓝到浅蓝
        custom_cmap = LinearSegmentedColormap.from_list('custom_gradient', gradient_colors)
        
        # 深蓝背景
        background_color = '#1A1A2E'
        plt.rcParams['axes.facecolor'] = background_color
        plt.rcParams['figure.facecolor'] = background_color
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        fig.patch.set_facecolor(background_color)
        ax.set_facecolor(background_color)
        
        # 创建渐变柱体色
        base_colors = custom_cmap(np.linspace(0, 1, len(data)))
        
        # 添加柱状图
        for i in range(len(data)):
            # 阴影效果
            ax.bar(
                i, data[i] * 0.98, width=1.0, color='black', alpha=0.3,
                zorder=1, edgecolor=None, align='center'
            )
            # 主柱体（渐变效果）
            ax.bar(
                i, data[i], width=1.0, color=base_colors[i], edgecolor='white',
                linewidth=0.5, alpha=0.9, zorder=2, align='center'
            )
            
        # 添加数值标签
        for i, val in enumerate(data):
            text = ax.text(
                i, val + max(data) * 0.03, f'{int(val)}', ha='center', va='bottom',
                fontsize=7, fontweight='bold', color='white', zorder=4
            )
            
        # 网格和样式
        ax.grid(axis='y', linestyle='-', alpha=0.1, color='white')
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xticks([])
        ax.set_ylim(0, max(data) * 1.15 if data else 1)
        ax.yaxis.set_ticks([])
        
        # 背景渐变
        if data:
            gradient = np.linspace(0, 1, 100).reshape(-1, 1)
            gradient = np.tile(gradient, (1, 10))
            ax.imshow(
                gradient, extent=[ax.get_xlim()[0], ax.get_xlim()[1], 0, max(data) * 1.15],
                aspect='auto', alpha=0.1, cmap='Blues', zorder=0
            )
            
        plt.tight_layout(pad=1.5)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=background_color)
        plt.close(fig)

    def solve(self, heights, **kwargs):
        """计算直方图中最大矩形面积的解答"""
        return self.largest_rectangle_in_histogram(heights)

    def _calculate_step_complexity(self, heights: List[int], difficulty: int = None) -> int:
        """计算直方图最大矩形问题的解题步骤复杂度"""
        n = len(heights)
        if n == 0:
            return 10
        
        # 1. 基本操作复杂度计算
        stack_operations = n * 4
        area_calculations = n
        
        # 2. 基础复杂度
        base_complexity = n * 2
        
        # 3. 数据特性复杂度
        height_changes = sum(abs(heights[i] - heights[i-1]) for i in range(1, n)) if n > 1 else 0
        avg_change = height_changes / (n-1) if n > 1 else 0
        
        increasing_segments = 0
        decreasing_segments = 0
        current_direction = None
        
        for i in range(1, n):
            if heights[i] > heights[i-1]:
                if current_direction != "up":
                    increasing_segments += 1
                    current_direction = "up"
            elif heights[i] < heights[i-1]:
                if current_direction != "down":
                    decreasing_segments += 1
                    current_direction = "down"
        
        direction_changes = increasing_segments + decreasing_segments
        
        # 4. 最大矩形复杂度
        max_height = max(heights) if heights else 0
        max_possible_area = max_height * n
        area_factor = min(100, max_possible_area) / 50
        
        # 5. 特殊模式复杂度
        has_plateau = any(i > 0 and i < n-1 and heights[i-1] < heights[i] and heights[i] == heights[i+1] for i in range(1, n-1))
        has_peak = any(i > 0 and i < n-1 and heights[i-1] < heights[i] and heights[i] > heights[i+1] for i in range(1, n-1))
        
        pattern_factor = 1.0
        if has_plateau: pattern_factor *= 1.2
        if has_peak: pattern_factor *= 1.1
        
        # 6. 级别调整
        level_factor = 1 + (difficulty - 1) * 0.2 if difficulty is not None else 1
        
        # 7. 计算最终复杂度
        data_factor = 1 + (avg_change / 5) + (direction_changes / 5) + (area_factor * 0.3)
        step_complexity = int(
            (base_complexity + stack_operations + area_calculations) * data_factor * pattern_factor * level_factor
        )
        
        return max(10, step_complexity)

    @staticmethod
    def largest_rectangle_in_histogram(heights: List[int]) -> int:
        """计算直方图中最大矩形的面积"""
        n = len(heights)
        if n == 0:
            return 0
        left, right = [0] * n, [0] * n
        mono_stack = list()
        
        for i in range(n):
            while mono_stack and heights[mono_stack[-1]] >= heights[i]:
                mono_stack.pop()
            left[i] = mono_stack[-1] if mono_stack else -1
            mono_stack.append(i)
        
        mono_stack = list()
        for i in range(n - 1, -1, -1):
            while mono_stack and heights[mono_stack[-1]] >= heights[i]:
                mono_stack.pop()
            right[i] = mono_stack[-1] if mono_stack else n
            mono_stack.append(i)
        
        ans = max((right[i] - left[i] - 1) * heights[i] for i in range(n))
        
        return ans

    @staticmethod
    def prompt_func_image_text(heights: List[int]) -> str:
        return (
            "Here is a histogram made of bars where each 1 unit wide and packed tightly together. What's the biggest rectangle you can fit entirely inside the histogram?"
        )

    @staticmethod
    def prompt_func_text(heights: List[int]) -> str:
        return (
            f"Here is a histogram made of bars {heights} where each 1 unit wide and packed tightly together. What's the biggest rectangle you can fit entirely inside the histogram?"
        )

    @staticmethod
    def extract_output(output: str) -> Optional[int]:
        """从模型输出中提取答案"""
        numbers = re.findall(r"-?\d+", output)
        return int(numbers[-1]) if numbers else None

    @classmethod
    def _verify_correction(cls, solution: int, identity: Dict) -> bool:
        return solution == int(identity['answer'])

    
    