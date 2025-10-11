import os
import json
import random
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from .base_generator import BaseGenerator

class ContainerWithMostWaterGenerator(BaseGenerator):
    
    def __init__(self, output_folder="output/container_with_most_water"):
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
        生成指定数量和难度的“盛最多水的容器”问题。
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
            # 生成高度列表
            heights = self._generate_height_list(
                params['list_length_range'], 
                params['data_range'], 
                difficulty
            )
            
            # 定义问题名称和图片路径
            question_name = f"ContainerWithMostWater_difficulty{difficulty}_{i}"
            image_path = os.path.join(images_dir, f"{question_name}.png")
            
            # 生成可视化图像
            self._generate_plot(heights, image_path)
            
            # 计算答案和复杂度
            answer = self.solve(heights)
            step_complexity = self._calculate_step_complexity(heights, difficulty)
            
            # 创建标注信息
            puzzle_data = {
                "index": question_name,
                "category": "ContainerWithMostWater",
                "question": self.prompt_func_image_text(),
                "image": os.path.join("images", f"{question_name}.png"), # 使用相对路径
                "question_language": self.prompt_func_text(heights),
                "answer": str(answer),
                "initial_state": {"heights": heights},
                "difficulty": difficulty,
                "solution": answer
            }
            annotations.append(puzzle_data)

        self.save_annotations(annotations, output_folder)
        return annotations

    def _generate_height_list(self, list_length_range: Tuple[int, int], data_range: Tuple[int, int], difficulty: int) -> List[int]:
        """生成高度列表"""
        length = random.randint(*list_length_range)
        heights = [random.randint(*data_range) for _ in range(length)]
        
        # 确保有一些较高的柱子，增加问题的趣味性
        if difficulty >= 3:
            # 随机选择2-3个位置，增加高度
            num_peaks = random.randint(2, 3)
            if length >= num_peaks:
                peak_positions = random.sample(range(length), num_peaks)
                for pos in peak_positions:
                    heights[pos] = random.randint(int(data_range[1] * 0.7), data_range[1])
        
        return heights

    def _generate_plot(self, data: List[int], output_path: str):
        """生成柱状图可视化"""
        # 浅蓝色渐变（pastel_sky风格）
        gradient_colors = [(0.8, 0.9, 1.0), (0.5, 0.7, 0.9)]  # 浅蓝到中蓝渐变
        custom_cmap = LinearSegmentedColormap.from_list('custom_gradient', gradient_colors)
        
        # 白色背景
        background_color = '#FFFFFF'
        plt.rcParams['axes.facecolor'] = background_color
        plt.rcParams['figure.facecolor'] = background_color
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        fig.patch.set_facecolor(background_color)
        ax.set_facecolor(background_color)
        
        base_colors = custom_cmap(np.linspace(0, 1, len(data)))
        
        for i in range(len(data)):
            # 移除黑色阴影效果
            ax.bar(i, data[i], width=1.0, color=base_colors[i], 
                   edgecolor='white', linewidth=0.5, alpha=0.9, zorder=2, align='center')
            
        # 数据标签改为深灰色
        for i, val in enumerate(data):
            ax.text(i, val + max(data) * 0.03, f'{int(val)}', ha='center', va='bottom',
                    fontsize=7, fontweight='bold', color='#333333', zorder=4)
            
        # 调整网格线为浅蓝色
        ax.grid(axis='y', linestyle='-', alpha=0.2, color='#B0C4DE')
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xticks([])
        ax.set_ylim(0, max(data) * 1.15 if data else 10)
        ax.yaxis.set_ticks([])
        
        # 移除深色背景渐变
        plt.tight_layout(pad=1.5)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=background_color)
        plt.close(fig)

    def solve(self, heights: List[int]) -> int:
        """计算盛最多水的容器问题的解答"""
        return self.max_area(heights)

    def _calculate_step_complexity(self, heights: List[int], difficulty: int) -> int:
        """计算盛最多水的容器问题的解题步骤复杂度"""
        n = len(heights)
        if n < 2:
            return 10

        # 1. 操作复杂度
        pointer_operations = n * 2
        area_calculations = n
        max_comparisons = n
        base_complexity = n * 2
        
        # 2. 数据特性复杂度
        height_changes = sum(abs(heights[i] - heights[i-1]) for i in range(1, n))
        avg_change = height_changes / (n-1)
        height_range = max(heights) - min(heights)
        combinations = n * (n - 1) // 2
        
        # 3. 最优解特性
        _, optimal_left, optimal_right = self.max_area_with_indices(heights)
        optimal_position_factor = 1.0
        if optimal_left > 0 or optimal_right < n - 1:
            optimal_position_factor = 1.2
            
        # 4. 级别调整
        level_factor = 1 + (difficulty - 1) * 0.2
        
        # 5. 计算最终复杂度
        data_factor = 1 + (avg_change / 10) + (height_range / 20) + (combinations / 100)
        
        step_complexity = int(
            (base_complexity + pointer_operations + area_calculations + max_comparisons) *
            data_factor * optimal_position_factor * level_factor
        )
        return max(10, step_complexity)

    @staticmethod
    def max_area(heights: List[int]) -> int:
        """使用双指针算法计算盛最多水的容器"""
        return ContainerWithMostWaterGenerator.max_area_with_indices(heights)[0]
    
    @staticmethod
    def max_area_with_indices(heights: List[int]) -> Tuple[int, int, int]:
        """计算最大面积并返回面积和对应的左右指针位置"""
        left, right = 0, len(heights) - 1
        max_area_val = 0
        opt_left, opt_right = 0, 0
        
        while left < right:
            width = right - left
            height = min(heights[left], heights[right])
            area = width * height
            
            if area > max_area_val:
                max_area_val = area
                opt_left, opt_right = left, right
                
            if heights[left] <= heights[right]:
                left += 1
            else:
                right -= 1
        
        return max_area_val, opt_left, opt_right

    @staticmethod
    def prompt_func_image_text() -> str:
        return (
            "Given a row of vertical bars where consecutive bars are adjacent with no gaps between them. "
            "Pick any two bars and form the sides of a water container, with the x-axis as the base. "
            "How much water can the biggest possible container hold?"
        )

    @staticmethod
    def prompt_func_text(heights: List[int]) -> str:
        return (
            f"Given a list of non-negative integers {heights}, where each number represents a point at coordinate (i, ai). "
            "n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). "
            "Find two lines, which together with x-axis forms a container, such that the container contains the most water. "
            "Return the maximum amount of water a container can store."
        )

# =============================================================================
# 3. 示例用法 (Updated)
# =============================================================================
if __name__ == "__main__":
    # 实例化生成器
    # 所有生成的问题将保存在 'newtasks/ContainerWithMostWater' 文件夹下
    generator = ContainerWithMostWaterGenerator()
    # generator.init(output_folder="newtasks/ContainerWithMostWater")
    
    num_cases_per_level = 5 # 为每个级别生成5个案例
    
    # 为每个难度级别生成测试用例
    for difficulty_level in range(1, 6):
        print(f"--- 正在生成 ContainerWithMostWater 难度级别 {difficulty_level} 的测试用例 ---")
        # 调用 generate 方法，它将处理文件的创建和保存
        generator.generate(
            num_cases=num_cases_per_level,
            difficulty=difficulty_level
        )
    
    print("\n所有难度级别的测试用例生成完成！")