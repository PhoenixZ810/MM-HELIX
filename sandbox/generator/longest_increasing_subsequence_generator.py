import os
import json
import random
import io
import base64
import re
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from functools import reduce
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from abc import ABC, abstractmethod
from .base_generator import BaseGenerator

class LongestIncreasingSubsequenceGenerator(BaseGenerator):
    def __init__(self, output_folder):
        """
        初始化最长递增子序列问题生成器
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
        生成指定数量的最长递增子序列问题
        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别
            output_folder: 输出文件夹路径，覆盖构造函数中设置的路径
        Returns:
            生成的问题列表
        """
        # 如果提供了新的输出文件夹，则更新
        if not output_folder:
            output_folder = self.output_folder
            
        # 获取难度参数
        params = self._get_difficulty_params(difficulty)
        data_range = params['data_range']
        list_length_range = params['list_length_range']
            
        problems = []
        for i in range(1, num_cases + 1):
            # 生成谜题
            puzzle = self._generate_single_puzzle(i, data_range=data_range, list_length_range=list_length_range, difficulty=difficulty, output_folder=output_folder)
            problems.append(puzzle)
        self.save_annotations(problems, output_folder)
        return problems

    def _generate_single_puzzle(self, case_id, data_range, list_length_range, difficulty, output_folder):
        """生成单个最长递增子序列问题"""
        # 生成数据序列
        sequence = self._generate_height_list(data_range=data_range, list_length_range=list_length_range)
            
        # 保存原始数据 (optional, as per original code, but not requested for final output structure)
        # with open(os.path.join(self.category_dir, f"rawdata/rawdata_{case_id}.txt"), 'w') as f:
        #     f.write(str(sequence))
            
        # 生成图像
        # New image path format: images/{taskname}_{level}_{index}.png
        puzzle_image_path = self.visualize(sequence, output_folder=output_folder, case_id=case_id, level=difficulty, task_name="longest_increasing_subsequence")
            
        # 计算答案
        answer = self.solve(sequence)
            
        # 计算解题步骤复杂度
        step_complexity = self._calculate_step_complexity(sequence, difficulty)
            
        # 创建数据点
        return {
            "index": f"longest_increasing_subsequence_difficulty_{difficulty}_{case_id}",
            "category": "longest_increasing_subsequence",
            "question": self.prompt_func_image_text(sequence),
            "image": puzzle_image_path, # Updated image path
            "question_language": self.prompt_func_text(sequence),
            "answer": answer,
            "initial_state": {"sequence": sequence},
            "difficulty": difficulty,
#           "solution": answer,
#           "step": step_complexity # 添加解题步骤复杂度
        }

    def _generate_height_list(self, data_range, list_length_range) -> List[int]:
        """生成数据序列"""
        # 设置随机种子
        random.seed(time.time())
            
        length = random.randint(*list_length_range)
        return [random.randint(*data_range) for _ in range(length)]

    def visualize(self, sequence, output_folder, **kwargs):
        """生成可视化图像"""
        case_id = kwargs.get('case_id', 0)
        level = kwargs.get('level', 0)
        task_name = kwargs.get('task_name', "UnknownTask")
            
        # Create the full output path for images
        # The base path for images is output/{taskname}/images/
        base_images_dir = os.path.join(output_folder, task_name, "images")
        os.makedirs(base_images_dir, exist_ok=True) # Ensure the images directory exists

        # New image file name format: {taskname}_{level}_{index}.png
        image_filename = f"{task_name}_{level}_{case_id}.png"
        full_image_path = os.path.join(base_images_dir, image_filename)
            
        # The 'image' field in JSON should be "images/{taskname}_{level}_{index}"
        json_image_path = os.path.join("images", image_filename)
            
        # 调用绘图函数
        self._generate_plot(sequence, full_image_path, case_id, output_folder)
            
        return json_image_path

    def _generate_plot(self, data, output_path, case_id=None,output_folder=None):
        # """生成折线图可视化 (白色背景)"""
        
        # --- 1. 初始化设置 ---
        # 清除之前的样式设置，恢复matplotlib默认的白色背景风格
        plt.rcdefaults() 
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        fig.patch.set_facecolor('white') # 确保整个图像背景为白色
        ax.set_facecolor('white')      # 确保绘图区域背景为白色
        
        # --- 2. 绘制折线图 ---
        x_values = range(len(data))
        ax.plot(x_values, data, 
                color='#A8A8A8',      # 线条颜色为黑色
                marker='o',         # 数据点使用圆形标记
                markersize=6,       # 标记大小
                markerfacecolor='white', # 标记填充色为白色
                markeredgecolor='black', # 标记边框为黑色
                linewidth=1.5)      # 线条宽度
        
        # --- 3. 添加数值标签 ---
        for i, val in enumerate(data):
            # 在每个数据点的正上方添加数值，手动计算偏移位置
            ax.text(i, val + (max(data) - min(data)) * 0.05, f'{int(val)}', 
                    ha='center',        # 水平居中
                    va='bottom',        # 垂直对齐到底部
                    fontsize=8, 
                    fontweight='bold', 
                    color='red')
            
        # --- 4. 调整样式和布局 (留出空隙) ---
        # 移除顶部和右侧的边框线，使图表更简洁
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 设置坐标轴范围，在四周留出空隙
        # X轴：左右各留出0.5个单位的空隙
        ax.set_xlim(-0.5, len(data) - 0.5) 
        
        # Y轴：根据数据范围动态调整，上下留出10%的空隙
        if data:
            data_min, data_max = min(data), max(data)
            margin = (data_max - data_min) * 0.1 if (data_max - data_min) > 0 else 1
            ax.set_ylim(data_min - margin, data_max + margin * 1.5) # 顶部留更多空间给数值标签
            
        # 隐藏坐标轴刻度，保持图像的纯粹性
        ax.set_xticks([])
        ax.set_yticks([])
        
        # --- 5. 保存图像 ---
        plt.tight_layout(pad=1.5) # 自动调整布局，确保所有元素可见
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def solve(self, sequence, **kwargs):
        """计算最长递增子序列的解答"""
        return self.lis_std(sequence)

    def _calculate_step_complexity(self, sequence: List[int], difficulty: int) -> int:
        """计算最长递增子序列问题的解题步骤复杂度"""
        n = len(sequence)
            
        # 1. 基本操作复杂度计算
        # 动态规划的操作数：
        # 外层循环 n 次
        # 内层循环平均 n/2 次
        # 每次内层循环有一次比较和一次可能的更新
        dp_operations = n * (n/2) * 2
            
        # 最终找最大值的操作
        max_operation = n
            
        # 2. 基础复杂度 - 主要依赖于序列长度
        base_complexity = n * 2
            
        # 3. 数据特性复杂度
        # 分析序列的变化程度 - 上下波动
        changes = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, n)) if n > 1 else 0
        avg_change = changes / (n-1) if n > 1 else 0
            
        # 序列值的范围
        value_range = max(sequence) - min(sequence) if sequence else 0
            
        # 序列的递增/递减趋势
        increasing_pairs = sum(1 for i in range(1, n) if sequence[i] > sequence[i-1])
        decreasing_pairs = sum(1 for i in range(1, n) if sequence[i] < sequence[i-1])
            
        # 趋势一致性 - 值越高表示序列越不一致，求解越复杂
        trend_inconsistency = min(increasing_pairs, decreasing_pairs) / (n-1) if n > 1 else 0
            
        # 4. 算法复杂度因子
        algo_complexity = 1 + trend_inconsistency * 2 + avg_change / 10
            
        # 5. 级别调整
        level_factor = 1 + (difficulty - 1) * 0.2 if difficulty is not None else 1
            
        # 6. 计算最终复杂度
        step_complexity = int(
            (base_complexity + dp_operations + max_operation) *
            algo_complexity *
            level_factor
        )
            
        # 设置最小值
        step_complexity = max(10, step_complexity)
            
        return step_complexity

    @staticmethod
    def lis_std(nums: List[int]) -> int:
        """标准动态规划解法计算最长递增子序列"""
        if not nums:
            return 0
            
        dp = []
        for i in range(len(nums)):
            dp.append(1)
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
            
        ans = max(dp)
        return ans

    @staticmethod
    def prompt_func_image_text(sequence: List[int]) -> str:
        return (
            "Here is a row of bars each with some height. Pick a subset of these bars where each one is strictly taller than the last and they appear in order from left to right. What's the longest such sequence you can find?"
        )

    @staticmethod
    def prompt_func_text(sequence: List[int]) -> str:
        return (
            f"Here is a row of bars each with some height {sequence}. Pick a subset of these bars where each one is strictly taller than the last and they appear in order from left to right. What's the longest such sequence you can find?"
        )

    @staticmethod
    def extract_output(output: str) -> Optional[int]:
        """从模型输出中提取答案"""
        numbers = re.findall(r"-?\d+", output)
        return int(numbers[-1]) if numbers else None

    @classmethod
    def _verify_correction(cls, solution: int, identity: Dict) -> bool:
        return solution == int(identity['answer'])
    