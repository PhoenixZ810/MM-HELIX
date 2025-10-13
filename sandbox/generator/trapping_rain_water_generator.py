import os
import json
import random
import re
from sqlite3 import paramstyle
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Optional

from .base_generator import BaseGenerator


class TrappingRainWaterGenerator(BaseGenerator):
    def __init__(self, output_folder):
        """
        初始化接雨水问题生成器
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

    def generate(self, num_cases, difficulty, output_folder: str = None):
        """
        生成指定数量的接雨水问题
        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别
            global_case_index_start: 全局用例索引的起始值，用于生成唯一的图片文件名
        Returns:
            生成的问题列表
        """

        output_folder = output_folder if output_folder else self.output_folder
        images_dir = os.path.join(output_folder, "images")
        os.makedirs(images_dir, exist_ok=True)

        problems = []
        for i in range(num_cases):
            # 生成谜题
            puzzle = self._generate_single_puzzle(case_id=i, difficulty=difficulty, output_folder=output_folder, images_dir=images_dir)
            problems.append(puzzle)

        self.save_annotations(problems, output_folder)
        return problems

    def _generate_single_puzzle(self, case_id, difficulty, output_folder, images_dir):
        """生成单个接雨水问题"""
        # 生成高度列表
        height = self._generate_height_list(difficulty=difficulty)

        # 生成图像
        # 图片地址格式为 images/{taskname}_{level}_{index}
        image_filename = f"trapping_rain_water_level_{difficulty}_{case_id}.png"
        puzzle_image_path = os.path.join(images_dir, image_filename)
        # 在JSON中，image字段表示为 "images/{taskname}_{level}_{index}.png"
        json_image_path = f"images/trapping_rain_water_level_{difficulty}_{case_id}.png"

        self._generate_plot(height, puzzle_image_path) # 保存图片到指定路径

        # 计算答案
        water = self.solve(height)

        # 计算解题步骤复杂度
#       step_complexity = self._calculate_step_complexity(height)
        step_complexity = self._calculate_step_complexity(height, difficulty)

        # 创建数据点
        return {
            "index": f"trapping_rain_water_difficulty_{difficulty}_{case_id}",
            "category": "trapping_rain_water",
            "question": self.prompt_func_image_text(height),
            "image": json_image_path, # JSON中的image路径
            "question_language": self.prompt_func_text(height),
            "answer": water,
            "initial_state": {"height": height},
            "difficulty": difficulty,
        }

    def _generate_height_list(self, difficulty) -> List[int]:
        """生成高度列表"""
        # 设置随机种子
        random.seed(time.time())

        params = self._get_difficulty_params(difficulty)
        list_length_range = params['list_length_range']
        length = random.randint(*list_length_range)
        return [random.randint(0, 20) for _ in range(length)]

    def visualize(self, height, **kwargs):
        """生成可视化图像 (此方法不再直接使用，而是通过 _generate_single_puzzle 调用 _generate_plot)"""
        # 此方法不再需要直接返回路径，因为路径在 _generate_single_puzzle 中构建
        pass

    def _generate_plot(self, data, output_path):
        """生成柱状图可视化"""
        # 设置为纯黑色柱体，无渐变
        background_color = 'white'
        plt.rcParams['axes.facecolor'] = background_color
        plt.rcParams['figure.facecolor'] = background_color
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        fig.patch.set_facecolor(background_color)
        ax.set_facecolor(background_color)
        
        # 添加柱状图
        for i in range(len(data)):
            # 阴影效果
            ax.bar(
                i, data[i] * 0.98, width=1.0, color='#E0E0E0', alpha=0.3,
                zorder=1, edgecolor=None, align='center'
            )
            # 主柱体（纯黑色，无渐变）
            ax.bar(
                i, data[i], width=1.0, color='#ADD8E6', edgecolor='#ADD8E6',
                linewidth=0.5, alpha=0.9, zorder=2, align='center'
            )
            
        # 添加数值标签
        for i, val in enumerate(data):
            text = ax.text(
                i, val + max(data) * 0.03, f'{int(val)}', ha='center', va='bottom',
                fontsize=7, fontweight='bold', color='black', zorder=4
            )
            
        # 网格和样式
        ax.grid(axis='y', linestyle='-', alpha=0.1, color='#AAAAAA')
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xticks([])
        ax.set_ylim(0, max(data) * 1.15)
        ax.yaxis.set_ticks([])
        
        # 背景渐变
        gradient = np.linspace(0, 1, 100).reshape(-1, 1)
        gradient = np.tile(gradient, (1, 10))
        ax.imshow(
            gradient, extent=[ax.get_xlim()[0], ax.get_xlim()[1], 0, max(data) * 1.15],
            aspect='auto', alpha=0.1, cmap='Greys', zorder=0
        )
        
        plt.tight_layout(pad=1.5)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=background_color)
        plt.close(fig)

    def solve(self, height, **kwargs):
        """计算接雨水的解答"""
        return self.trap_water(height)

    @staticmethod
    def trap_water(height: List[int]) -> int:
        """计算可以接多少雨水"""
        if not height:
            return 0

        n = len(height)
        leftMax = [height[0]] + [0] * (n - 1)
        for i in range(1, n):
            leftMax[i] = max(leftMax[i - 1], height[i])

        rightMax = [0] * (n - 1) + [height[n - 1]]
        for i in range(n - 2, -1, -1):
            rightMax[i] = max(rightMax[i + 1], height[i])

        ans = sum(min(leftMax[i], rightMax[i]) - height[i] for i in range(n))
        return ans

    def _calculate_step_complexity(self, height: List[int], difficulty: int = 1) -> int:
        """计算接雨水问题的解题步骤复杂度"""
        n = len(height)
        
        # 1. 基本操作复杂度计算
        # 构建leftMax数组的操作数
        left_max_operations = n - 1  # n-1次比较
        
        # 构建rightMax数组的操作数
        right_max_operations = n - 1  # n-1次比较
        
        # 计算trapped water的操作数
        # 每个位置需要: 1次min比较 + 1次减法 + 1次加法(sum)
        water_calculation_operations = n * 3
        
        # 2. 基础复杂度 - 主要依赖于数组长度
        base_complexity = n * 2  # 数组长度是关键因素
        
        # 3. 操作复杂度 - 所有关键操作的总和
        operations_complexity = left_max_operations + right_max_operations + water_calculation_operations
        
        # 4. 空间复杂度影响 - 需要额外空间存储左右最大值数组
        space_complexity_factor = 2  # 两个额外数组
        
        # 5. 问题特性复杂度
        # 数组高度值的大小范围会影响解题思考难度
        height_range = max(height) - min(height) if height else 0
        height_factor = min(20, height_range) / 10  # 将高度差异归一化，最大影响为2
        
        # 数组的形状特征 - 波动性
        # 计算相邻高度的差异总和，波动越大越复杂
        height_changes = sum(abs(height[i] - height[i-1]) for i in range(1, n)) if n > 1 else 0
        volatility_factor = min(20, height_changes) / 10  # 归一化，最大影响为2
        
        # 6. 级别调整
        level_factor = 1 + (difficulty - 1) * 0.2  # 根据难度级别调整
        
        # 7. 计算最终复杂度
        # 基础部分 + 操作复杂度 * 问题特性因子 * 级别因子
        step_complexity = int(
            base_complexity +
            operations_complexity * (1 + height_factor * 0.3 + volatility_factor * 0.2) *
            level_factor
        )
        
        # 设置最小值
        step_complexity = max(10, step_complexity)
        
        return step_complexity

    @staticmethod
    def prompt_func_image_text(height: List[int]) -> str:
        return (
            "Here is a bunch of bars lined up side by side, where the width of each bar is 1 and consecutive bars are adjacent with no gaps between them. Compute how much water it can trap after raining."
        )

    @staticmethod
    def prompt_func_text(height: List[int]) -> str:
        return (
            f"Here is a bunch of bars {height} lined up side by side, where the width of each bar is 1 and consecutive bars are adjacent with no gaps between them. Compute how much water it can trap after raining."
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
    case_num_per_difficulty = 5 # 每个难度生成5个用例
    output_folder = "output"
    task_name = "trapping_rain_water"

    all_puzzles = []
    global_case_index = 0 # 用于跟踪所有用例的全局唯一索引

    # 为每个难度级别生成测试用例
    for difficulty in difficulties:
        # 实例化生成器时传入任务名称
        generator = TrappingRainWaterGenerator(output_folder=output_folder, task_name=task_name)
        
        # generate方法现在会返回当前难度级别生成的所有puzzle
        puzzles_for_difficulty = generator.generate(case_num_per_difficulty, difficulty, global_case_index_start=global_case_index)
        all_puzzles.extend(puzzles_for_difficulty)
        global_case_index += case_num_per_difficulty # 更新全局索引

    # 所有难度级别生成完成后，统一保存到 annotations.json
    generator.save_all_puzzles_to_json(all_puzzles)