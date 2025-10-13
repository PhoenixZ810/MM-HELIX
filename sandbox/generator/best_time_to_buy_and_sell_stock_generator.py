import os
import json
import random
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from .base_generator import BaseGenerator

class BestTimeToBuyAndSellStockGenerator(BaseGenerator):
    
    def __init__(self, output_folder="output/best_time_to_buy_and_sell_stock"):
        """
        初始化生成器并设置默认输出文件夹。
        """
        super().__init__(output_folder)

    def _get_difficulty_params(self, difficulty: int) -> Dict[str, Tuple[int, int]]:
        """
        根据难度级别（1-5）获取相应的参数配置。
        """
        if difficulty == 1:
            return {"data_range": (1, 20), "list_length_range": (5, 10)}
        elif difficulty == 2:
            return {"data_range": (1, 40), "list_length_range": (10, 20)}
        elif difficulty == 3:
            return {"data_range": (1, 60), "list_length_range": (20, 30)}
        elif difficulty == 4:
            return {"data_range": (1, 80), "list_length_range": (30, 50)}
        else:  # difficulty 5
            return {"data_range": (1, 100), "list_length_range": (50, 70)}

    def generate(self, num_cases: int, difficulty: int, output_folder: Optional[str] = None):
        """
        生成指定数量和难度的股票买卖问题。
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
            # 生成价格列表
            prices = self._generate_price_list(
                params['list_length_range'], 
                params['data_range'], 
                difficulty
            )
            
            # 定义问题名称和图片路径
            question_name = f"BestTimeToBuyAndSellStock_difficulty{difficulty}_{i}"
            image_path = os.path.join(images_dir, f"{question_name}.png")
            
            # 生成可视化图像
            self._generate_plot(prices, image_path)
            
            # 计算答案和复杂度
            answer = self.solve(prices)
            step_complexity = self._calculate_step_complexity(prices, difficulty)
            
            # 创建标注信息
            puzzle_data = {
                "index": question_name,
                "category": "BestTimeToBuyAndSellStock",
                "question": self.prompt_func_image_text(),
                "image": os.path.join("images", f"{question_name}.png"), # 使用相对路径
                "question_language": self.prompt_func_text(prices),
                "answer": str(answer),
                "initial_state": {"prices": prices},
                "difficulty": difficulty,
                "solution": answer,
            }
            annotations.append(puzzle_data)
        
        self.save_annotations(annotations, output_folder)
        
        return annotations

    def _generate_price_list(self, list_length_range: Tuple[int, int], data_range: Tuple[int, int], difficulty: int) -> List[int]:
        """生成股票价格列表"""
        length = random.randint(*list_length_range)
        prices = [random.randint(*data_range) for _ in range(length)]
        
        # 确保生成的价格数据有一定的波动性
        if difficulty >= 3:
            # 对于较高难度级别，确保有更多的买入卖出机会
            for i in range(1, length):
                # 有30%的概率让价格上涨
                if random.random() < 0.3:
                    prices[i] = prices[i-1] + random.randint(1, 5)
                # 有30%的概率让价格下跌
                elif random.random() < 0.6:
                    prices[i] = max(1, prices[i-1] - random.randint(1, 5))
        
        return prices

    def _generate_plot(self, data: List[int], output_path: str):
        """
        生成股票价格折线图可视化 (增强科技感背景)
        """
        # 设置图表风格和背景色
        background_color = '#1A1A2E' # 深蓝紫色背景
        plt.style.use('dark_background') 
        plt.rcParams['axes.facecolor'] = background_color
        plt.rcParams['figure.facecolor'] = background_color
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        
        x_data = np.arange(len(data))
        
        
        # 首先设置坐标轴范围以便在正确区域生成背景
        ax.set_xlim(-0.5, len(data) - 0.5)
        if data:
            y_min, y_max = min(data) * 0.9, max(data) * 1.1
            ax.set_ylim(y_min, y_max)
        else:
            y_min, y_max = 0, 100
            ax.set_ylim(y_min, y_max)
            
        # 创建网格状分布的数字背景
        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        
        # 密集网格：每0.3个单位放一个数字
        grid_spacing_x = x_range / 35  # X方向约35个网格
        grid_spacing_y = y_range / 20  # Y方向约20个网格
        
        for i in range(40):  # X方向
            for j in range(25):  # Y方向
                # 基础网格位置 + 随机偏移
                base_x = ax.get_xlim()[0] + i * grid_spacing_x
                base_y = ax.get_ylim()[0] + j * grid_spacing_y
        
                # 添加随机偏移，避免过于规整
                rand_x = base_x + random.uniform(-grid_spacing_x*0.3, grid_spacing_x*0.3)
                rand_y = base_y + random.uniform(-grid_spacing_y*0.3, grid_spacing_y*0.3)
        
                # 确保坐标在图表范围内
                if ax.get_xlim()[0] <= rand_x <= ax.get_xlim()[1] and ax.get_ylim()[0] <= rand_y <= ax.get_ylim()[1]:
                    # 生成不同类型的数字和字符
                    char_pool = ['0','1','2','3','4','5','6','7','8','9','.','$','%']
                    char = random.choice(char_pool)
                    
                    # 不同区域使用不同颜色，增加层次感
                    colors = ['#006666', '#004d4d', '#003333', '#002626', '#001a1a']  # 青色系渐变
                    color = random.choice(colors)
                    
                    ax.text(rand_x, rand_y, char, 
                           color=color, 
                           alpha=random.uniform(0.03, 0.08),  # 更淡的透明度
                           fontsize=random.randint(6, 10),    # 较小的字体
                           ha='center', va='center', 
                           rotation=random.uniform(-15, 15),  # 轻微旋转增加动感
                           zorder=0)
                    
                    
        # --- 绘制网格 (在背景元素之上，折线之下) ---
        # 添加比默认更精细和透明的网格
        ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, color='white', alpha=0.2, zorder=1)
        ax.grid(True, which='major', axis='x', linestyle='--', linewidth=0.5, color='white', alpha=0.1, zorder=1)
        
        
        # --- 绘制折线和填充 (在网格之上) ---
        line_color = '#00BFFF'  # DeepSkyBlue，保持清晰
        # 绘制带有标记点的折线
        ax.plot(x_data, data, color=line_color, linewidth=2, marker='o', markersize=5, markerfacecolor='white', markeredgecolor=line_color, zorder=3)
        
        # 填充线下方的区域以增加视觉效果
        ax.fill_between(x_data, data, color=line_color, alpha=0.1, zorder=2)
        
        # --- 隐藏和美化坐标轴 ---
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        if data:
            ax.set_ylim(min(data) * 0.9, max(data) * 1.1)
            
        # 在每个数据点上添加价格标签
        for i, val in enumerate(data):
            offset = max(data) * 0.02 if max(data) > 0 else 0.5
            ax.text(i, val + offset, f'{int(val)}', ha='center', va='bottom', fontsize=7, fontweight='bold', color='white', zorder=4)
            
        # --- 优化布局并保存 ---
        plt.tight_layout(pad=1.5)
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=background_color)
        plt.close(fig)

    def solve(self, prices: List[int]) -> int:
        """计算股票买卖的最大利润"""
        return self.max_profit(prices)

    def _calculate_step_complexity(self, prices: List[int], difficulty: int) -> int:
        """计算股票买卖问题的解题步骤复杂度"""
        n = len(prices)
        if n == 0:
            return 10

        greedy_operations = n - 1
        comparison_operations = (n - 1) * 2
        base_complexity = n * 2
        
        price_changes = sum(abs(prices[i] - prices[i-1]) for i in range(1, n)) if n > 1 else 0
        avg_change = price_changes / (n-1) if n > 1 else 0
        
        buy_opportunities = sum(1 for i in range(1, n) if prices[i] < prices[i-1])
        sell_opportunities = sum(1 for i in range(1, n) if prices[i] > prices[i-1])
        trade_opportunities = min(buy_opportunities, sell_opportunities)
        
        trend_changes = 0
        current_trend = None
        for i in range(1, n):
            if prices[i] > prices[i-1]:
                if current_trend != "up":
                    trend_changes += 1
                    current_trend = "up"
            elif prices[i] < prices[i-1]:
                if current_trend != "down":
                    trend_changes += 1
                    current_trend = "down"
        
        level_factor = 1 + (difficulty - 1) * 0.2
        data_factor = 1 + (avg_change / 10) + (trade_opportunities / 5) + (trend_changes / 5)
        
        step_complexity = int((base_complexity + greedy_operations + comparison_operations) * data_factor * level_factor)
        return max(10, step_complexity)

    @staticmethod
    def max_profit(prices: List[int]) -> int:
        """计算多次交易的最大利润 (贪心算法)"""
        profit = 0
        for i in range(1, len(prices)):
            profit_today = prices[i] - prices[i - 1]
            if profit_today > 0:
                profit += profit_today
        return profit

    @staticmethod
    def prompt_func_image_text() -> str:
        return (
            "Given a bar chart of stock prices over time and each bar's height is the price on that day. "
            "You can buy and sell as much as you want, but can only hold one stock at a time. "
            "Calculate the maximum profit you can get from this transaction. If you cannot get any profit, answer 0."
        )

    @staticmethod
    def prompt_func_text(prices: List[int]) -> str:
        return (
            f"Given a list of stock prices {prices} over time. "
            "You can buy and sell as much as you want, but can only hold one stock at a time. "
            "Calculate the maximum profit you can get from this transaction. If you cannot get any profit, answer 0."
        )

# =============================================================================
# 3. 示例用法 (Updated)
# =============================================================================
if __name__ == "__main__":
    # 实例化生成器
    # 所有生成的问题将保存在 'newtasks/BestTimeToBuyAndSellStock' 文件夹下
    generator = BestTimeToBuyAndSellStockGenerator()
    # generator.init(output_folder="newtasks/BestTimeToBuyAndSellStock")
    
    num_cases_per_level = 5 # 为每个级别生成5个案例
    
    # 为每个难度级别生成测试用例
    for difficulty_level in range(1, 6):
        print(f"--- 正在生成 BestTimeToBuyAndSellStock 难度级别 {difficulty_level} 的测试用例 ---")
        # 调用 generate 方法，它将处理文件的创建和保存
        generator.generate(
            num_cases=num_cases_per_level,
            difficulty=difficulty_level
        )
    
    print("\n所有难度级别的测试用例生成完成！")