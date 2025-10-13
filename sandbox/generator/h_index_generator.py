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

class HIndexGenerator(BaseGenerator):
    def __init__(self, output_folder="output/hindex"):
        """
        初始化H指数问题生成器
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
        生成指定数量的H指数问题
        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别
        Returns:
            生成的问题列表
        """
        # 根据难度获取参数
        if output_folder is None:
            output_folder = self.output_folder
        params = self._get_difficulty_params(difficulty)
        self.level = difficulty
        self.data_range = params['data_range']
        self.list_length_range = params['list_length_range']

        # 创建主输出目录和 images 子目录（如果不存在）
        os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
        
        problems = []
        for i in range(1, num_cases + 1):
            # 生成谜题
            puzzle = self._generate_single_puzzle(i)
            problems.append(puzzle)

        self.save_annotations(problems, output_folder)
            
        return problems

    def _generate_single_puzzle(self, case_id):
        """生成单个H指数问题"""
        # 生成引用次数列表
        citations = self._generate_citations_list()
        
        # 生成图像并获取其相对路径
        puzzle_image_path = self.visualize(citations, case_id=case_id)
        
        # 计算答案
        answer = self.solve(citations)
        
        # 计算解题步骤复杂度
        step_complexity = self._calculate_step_complexity(citations)
        
        # 创建数据点
        return {
            "index": f"HIndex_difficulty{self.level}_{case_id}",
            "category": "HIndex",
            "question": self.prompt_func_image_text(citations),
            "image": puzzle_image_path, # 使用新的相对路径格式
            "question_language": self.prompt_func_text(citations),
            "answer": str(answer),
            "initial_state": {"citations": citations},
            "difficulty": self.level,
            "solution": answer,
#           "step": step_complexity   添加解题步骤复杂度
        }

    def _generate_citations_list(self) -> List[int]:
        """生成引用次数列表，确保有合理的H指数"""
        # 设置随机种子
        random.seed(time.time())
        
        length = random.randint(*self.list_length_range)
        # 首先生成一个随机的引用列表
        citations = []
        
        # 根据难度级别调整生成策略
        if self.level <= 2:
            # 低难度：生成较明显的H指数
            # 确定一个目标H指数
            target_h = random.randint(3, min(7, length))
            
            # 生成至少h篇有h或更多引用的论文
            for i in range(target_h):
                citations.append(random.randint(target_h, self.data_range[1]))
            
            # 生成剩余的低引用论文
            for i in range(length - target_h):
                citations.append(random.randint(0, target_h - 1))
            
            # 随机打乱顺序
            random.shuffle(citations)
        else:
            # 高难度：生成更随机的引用分布
            citations = [random.randint(*self.data_range) for _ in range(length)]
            
            # 对于较高难度，确保有一些高引用论文
            num_high_citations = random.randint(2, length // 3)
            high_indices = random.sample(range(length), num_high_citations)
            for idx in high_indices:
                citations[idx] = random.randint(int(self.data_range[1] * 0.7), self.data_range[1])
        
        return citations

    def visualize(self, citations, **kwargs):
        """生成可视化图像并返回相对路径"""
        case_id = kwargs.get('case_id', 0)
        
        # 定义新的图片文件名格式
        image_filename = f"hindex_difficulty_{self.level}_{case_id}.png"
        
        # 创建图片的完整保存路径
        full_image_path = os.path.join(self.output_folder, "images", image_filename)
        
        # 调用绘图函数
        self._generate_plot(citations, full_image_path, case_id)
        
        # 返回JSON文件中要使用的相对路径
        return os.path.join("images", image_filename).replace("\\", "/")


    def _generate_plot(self, data, output_path, case_id=None):
        """生成柱状图可视化"""
        # 使用white_gray_bar样式配色
        background_color = 'white'
        plt.rcParams['axes.facecolor'] = background_color
        plt.rcParams['figure.facecolor'] = background_color
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        fig.patch.set_facecolor(background_color)
        ax.set_facecolor(background_color)
        
        # 添加柱状图（纯灰色，无渐变）
        for i in range(len(data)):
            # 阴影效果
            ax.bar(
                i, data[i] * 0.98, width=1.0, color='#E0E0E0', alpha=0.3,
                zorder=1, edgecolor=None, align='center'
            )
            # 主柱体（纯灰色）
            ax.bar(
                i, data[i], width=1.0, color='#666666', edgecolor='#666666',
                linewidth=0.5, alpha=0.9, zorder=2, align='center'
            )
            
        # 添加数值标签
        for i, val in enumerate(data):
            text = ax.text(
                i, val + max(data) * 0.03, f'{int(val)}', ha='center', va='bottom',
                fontsize=7, fontweight='bold', color='#333333', zorder=4
            )
            
        # 网格和样式
        ax.grid(axis='y', linestyle='-', alpha=0.1, color='#BBBBBB')
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xticks([])
        ax.set_ylim(0, max(data) * 1.15)
        ax.yaxis.set_ticks([])
        
        # 移除背景渐变
        plt.tight_layout(pad=1.5)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=background_color)
        plt.close(fig)

    def solve(self, citations, **kwargs):
        """计算研究者的H指数"""
        return self.calculate_h_index(citations)

    def _calculate_step_complexity(self, citations: List[int]) -> int:
        """计算H指数问题的解题步骤复杂度"""
        n = len(citations)
        
        # 1. 基本操作复杂度计算
        # 排序操作复杂度
        sorting_operations = n * (n // 2)  # 近似排序的操作数
        
        # 计算H指数的操作
        h_index_operations = n  # 比较和更新H值的操作
        
        # 2. 基础复杂度 - 主要依赖于论文数量
        base_complexity = n * 2
        
        # 3. 数据特性复杂度
        # 实际H指数 - H指数接近论文数量的一半时最复杂
        h_index = self.calculate_h_index(citations)
        h_index_ratio = h_index / n if n > 0 else 0
        h_complexity_factor = 4 * h_index_ratio * (1 - h_index_ratio)  # 当h_index=n/2时最大
        
        # 引用分布分析 - 引用数接近H指数的论文越多，越难确定准确H值
        sorted_citations = sorted(citations, reverse=True)
        near_h_papers = sum(1 for c in sorted_citations if abs(c - h_index) <= 2)
        near_h_factor = near_h_papers / n if n > 0 else 0
        
        # 4. 级别调整
        level_factor = 1 + (self.level - 1) * 0.2
        
        # 5. 计算最终复杂度
        # 数据特性因子
        data_factor = 1 + h_complexity_factor + (near_h_factor * 0.5)
        step_complexity = int(
            (base_complexity + sorting_operations / 4 + h_index_operations) * data_factor * level_factor
        )
        
        # 设置最小值
        step_complexity = max(10, step_complexity)
        
        return step_complexity

    @staticmethod
    def calculate_h_index(citations: List[int]) -> int:
        """计算H指数"""
        sorted_citation = sorted(citations, reverse=True)
        h = 0
        i = 0
        n = len(citations)
        
        while i < n and sorted_citation[i] > h:
            h += 1
            i += 1
        
        return h

    @staticmethod
    def prompt_func_image_text(citations: List[int]) -> str:
        return (
            "Here is a bar chart showing how many times each of a researcher's papers was cited. Determine the researcher's h-index: the largest value h such that at least h papers have at least h citations each."
        )

    @staticmethod
    def prompt_func_text(citations: List[int]) -> str:
        return (
            f"Here is a bar chart {citations} showing how many times each of a researcher's papers was cited. Determine the researcher's h-index: the largest value h such that at least h papers have at least h citations each."
        )

    @staticmethod
    def extract_output(output: str) -> Optional[int]:
        """从模型输出中提取答案"""
        numbers = re.findall(r"-?\d+", output)
        return int(numbers[-1]) if numbers else None

    @classmethod
    def _verify_correction(cls, solution: int, identity: Dict) -> bool:
        return solution == int(identity['answer'])


if __name__ == "__main__":
    # 定义不同难度级别的参数
    difficulties = [1, 2, 3, 4, 5]
    case_num = 5
    
    all_puzzles_across_difficulties = []
    
    # 实例化一次生成器
    # 所有文件将保存在 'output/hindex' 目录下
    generator = HIndexGenerator(output_folder="output/hindex")
    
    # 为每个难度级别生成测试用例
    for difficulty in difficulties:
        print(f"正在生成 HIndex 难度级别 {difficulty} 的测试用例...")
        puzzles = generator.generate(case_num, difficulty)
        all_puzzles_across_difficulties.extend(puzzles)
        print(f"HIndex 难度级别 {difficulty} 的测试用例生成完成！")
    
    # 所有难度生成完毕后，将结果统一保存到一个 annotations.json 文件中
    generator.save_annotations(all_puzzles_across_difficulties, generator.output_folder)
    