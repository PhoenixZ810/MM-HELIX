import os
import json
import random
import re
import io
import base64
import time
from functools import reduce
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
from abc import ABC, abstractmethod

from .base_generator import BaseGenerator

class KukurasuGenerator(BaseGenerator):
    def __init__(self, output_folder="output/kukurasu"):
        super().__init__(output_folder)
        # 配置参数
        self.image_size = (800, 800)
        self.grid_color = "#333333"
        self.cell_color = "#ffffff"
        self.black_cell_color = "#000000"
        self.highlight_color = "#3498db"
        self.bg_color = "#f5f5f5"
        self.text_color = "#2c3e50"
        # 查找系统字体
        self.font_path = self._find_system_font()
    
    def _get_difficulty_params(self, difficulty):
        """
        根据难度级别获取相应的参数配置。
        Args:
            difficulty: 难度级别（1-5）
        Returns:
            dict: 包含难度参数的字典
        """
        params = {
            "difficulty": difficulty,
            "min_size": 3,
            "max_size": 3
        }
        
        if difficulty == 1:
            params["min_size"] = 3
            params["max_size"] = 3
        elif difficulty == 2:
            params["min_size"] = 4
            params["max_size"] = 4
        elif difficulty == 3:
            params["min_size"] = 5
            params["max_size"] = 5
        elif difficulty == 4:
            params["min_size"] = 6
            params["max_size"] = 6
        else:  # level 5
            params["min_size"] = 7
            params["max_size"] = 7
        
        return params
    
    def _find_system_font(self):
        """查找系统中可用的字体"""
        common_fonts = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Linux
            '/System/Library/Fonts/Helvetica.ttc',  # macOS
            'C:/Windows/Fonts/arial.ttf',  # Windows
            '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'  # Ubuntu
        ]
        for font in common_fonts:
            if os.path.exists(font):
                return font
        return None

    def generate(self, num_cases, difficulty, output_folder=None):
        """
        生成指定数量的 Kukurasu 谜题
        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别
        Returns:
            生成的问题列表
        """
        if output_folder is None:
            output_folder = self.output_folder
        # 设置随机种子
        random.seed(time.time())
        
        # 获取难度级别参数
        params = self._get_difficulty_params(difficulty)
        min_size = params["min_size"]
        max_size = params["max_size"]
        
        # 确保主输出目录和 images 子目录存在
        os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
        
        problems = []
        for i in range(1, num_cases + 1):
            # 生成谜题
            case = self._generate_single_puzzle(i, min_size=min_size, max_size=max_size, difficulty=difficulty, output_folder=output_folder)
            problems.append(case)
        
        self.save_annotations(problems, output_folder)
        return problems
    
    def _generate_single_puzzle(self, case_id, min_size=None, max_size=None, difficulty=None, output_folder=None):
        """生成单个 Kukurasu 谜题"""
        # 随机选择网格大小
        n = random.randint(min_size, max_size)
        # 生成有效网格
        grid = self._generate_valid_grid(n)
        # 计算行和列的约束条件
        row_sums = [sum(j+1 for j in range(n) if grid[i][j]) for i in range(n)]
        col_sums = [sum(i+1 for i in range(n) if grid[i][j]) for j in range(n)]
        # 确保至少每行/列有1个填充
        if 0 in row_sums or 0 in col_sums:
            return self._generate_single_puzzle(case_id, min_size=min_size, max_size=max_size, difficulty=difficulty, output_folder=output_folder) # 重新生成
        # 创建谜题数据
        puzzle_data = {
            "row_sums": row_sums,
            "col_sums": col_sums,
            "size": n,
            "solution": grid
        }
        # 生成图像
        puzzle_image_path = self._visualize_puzzle(puzzle_data, case_id, min_size=min_size, max_size=max_size, difficulty=difficulty, output_folder=output_folder)
        
        # 创建数据点
        return self._create_datapoint(case_id, grid, puzzle_data, puzzle_image_path, min_size=min_size, max_size=max_size, difficulty=difficulty)

    def _generate_valid_grid(self, n):
        """生成有效的初始解"""
        grid = []
        for _ in range(n):
            # 动态调整填充概率
            base_prob = 0.3 + random.random() * 0.4  # 30%-70%
            grid.append([
                1 if random.random() < base_prob else 0
                for _ in range(n)
            ])
        return grid
        
    def _visualize_puzzle(self, puzzle, case_id, min_size=None, max_size=None, difficulty=None, output_folder=None):
        """生成谜题的可视化图像并返回相对路径"""
        # 生成谜题图像数据
        puzzle_image_data = self._generate_puzzle_image(
            puzzle["size"], 
            puzzle["row_sums"], 
            puzzle["col_sums"]
        )
        
        # 定义新的图片文件名格式
        image_filename = f"kukurasu_difficulty_{difficulty}_{min_size}x{max_size}_{case_id}.png"
        # 创建图片的完整保存路径
        full_image_path = os.path.join(output_folder, "images", image_filename)

        with open(full_image_path, "wb") as f:
            f.write(base64.b64decode(puzzle_image_data))

        # 返回JSON文件中要使用的相对路径
        return os.path.join("images", image_filename).replace("\\", "/")
    
    def _generate_puzzle_image(self, n, row_sums, col_sums):
        """
        生成 Kukurasu 谜题的图像表示
        参数:
            n: 网格尺寸
            row_sums: 行和约束
            col_sums: 列和约束
        返回:
            base64编码的图像
        """
        # 确定图像尺寸和边距
        width, height = self.image_size
        margin = 80  # 边距
        title_margin = 30  # 标题顶部边距
        # 计算单元格大小
        cell_size = min((width - 2 * margin) // (n + 1), (height - 2 * margin - 50) // (n + 1))
        # 创建基础图像
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)
        # 设置字体
        title_font_size = 36
        number_font_size = 24
        cell_font_size = 20
        try:
            if self.font_path:
                title_font = ImageFont.truetype(self.font_path, title_font_size)
                number_font = ImageFont.truetype(self.font_path, number_font_size)
                cell_font = ImageFont.truetype(self.font_path, cell_font_size)
            else:
                title_font = ImageFont.load_default()
                number_font = ImageFont.load_default()
                cell_font = ImageFont.load_default()
        except:
            title_font = ImageFont.load_default()
            number_font = ImageFont.load_default()
            cell_font = ImageFont.load_default()
        # 绘制标题
        title = "Kukurasu Puzzle"
        title_width = draw.textlength(title, font=title_font) if hasattr(draw, 'textlength') else title_font_size * len(title) * 0.6
        draw.text(((width - title_width) // 2, title_margin), title, fill=self.text_color, font=title_font)
        # 计算网格和约束区域的总宽度和高度
        constraint_space = 50  # 约束数字的空间
        total_grid_width = n * cell_size
        total_grid_height = n * cell_size
        # 计算网格起始位置(整体居中，包括约束)
        start_x = (width - (total_grid_width + constraint_space)) // 2
        # 留出上方的空间用于标题和说明
        title_space = title_margin + title_font_size + 20
        # 计算网格垂直居中位置，考虑到标题空间
        start_y = title_space + ((height - title_space - total_grid_height - constraint_space) // 2)
        # 绘制网格
        for i in range(n + 1):
            # 绘制水平线
            y = start_y + i * cell_size
            draw.line([(start_x, y), (start_x + total_grid_width, y)], fill=self.grid_color, width=2)
            # 绘制垂直线
            x = start_x + i * cell_size
            draw.line([(x, start_y), (x, start_y + total_grid_height)], fill=self.grid_color, width=2)
        # 绘制单元格内容 (X)
        for i in range(n):
            for j in range(n):
                x = start_x + j * cell_size
                y = start_y + i * cell_size
                cell_text = "X"
                cell_width = draw.textlength(cell_text, font=cell_font) if hasattr(draw, 'textlength') else cell_font_size * 0.6
                draw.text((x + (cell_size - cell_width) // 2, y + (cell_size - cell_font_size) // 2), 
                          cell_text, fill=self.text_color, font=cell_font)
        # 绘制行和约束
        for i in range(n):
            y = start_y + i * cell_size + cell_size // 2
            x = start_x + total_grid_width + 10
            row_sum = str(row_sums[i])
            draw.text((x, y - number_font_size // 2), row_sum, fill=self.highlight_color, font=number_font)
        # 绘制列和约束
        for j in range(n):
            x = start_x + j * cell_size + cell_size // 2
            y = start_y + total_grid_height + 10
            col_sum = str(col_sums[j])
            col_width = draw.textlength(col_sum, font=number_font) if hasattr(draw, 'textlength') else number_font_size * 0.6
            draw.text((x - col_width // 2, y), col_sum, fill=self.highlight_color, font=number_font)
        # 绘制坐标提示
        # 列号 (1-n)
        for j in range(n):
            x = start_x + j * cell_size + cell_size // 2
            y = start_y - 30
            col_num = str(j + 1)
            col_width = draw.textlength(col_num, font=number_font) if hasattr(draw, 'textlength') else number_font_size * 0.6
            draw.text((x - col_width // 2, y), col_num, fill="#999999", font=number_font)
        # 行号 (1-n)
        for i in range(n):
            y = start_y + i * cell_size + cell_size // 2
            x = start_x - 30
            row_num = str(i + 1)
            row_width = draw.textlength(row_num, font=number_font) if hasattr(draw, 'textlength') else number_font_size * 0.6
            draw.text((x - row_width // 2, y - number_font_size // 2), row_num, fill="#999999", font=number_font)
        # 添加边框
        border_margin = 5
        draw.rectangle([border_margin, border_margin, width - border_margin, height - border_margin], 
                       fill=None, outline=self.highlight_color, width=3)
        # 转换为base64编码的图像
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = base64.b64encode(buffer.getvalue()).decode()
        return image_data

    def _create_datapoint(self, case_id, grid, puzzle_data, puzzle_image_path, min_size=None, max_size=None, difficulty=None):
        """创建数据点，并添加step字段评估解题步骤复杂度"""
        # 计算解题步骤复杂度
        step_complexity = self._calculate_step_complexity(puzzle_data, difficulty=difficulty)
        return {
            "index": f"Kukurasu_difficulty{difficulty}_{case_id}",
            "category": "Kukurasu",
            "question": self._generate_question_text(puzzle_data, min_size, max_size),
            "image": puzzle_image_path,
            "question_language": self._generate_detailed_question_text(puzzle_data, min_size, max_size),
            "answer": grid,
            "initial_state": puzzle_data,
            "difficulty": difficulty,
#           "grid": grid,
#           "solution": grid,
#           "step": step_complexity   新增字段：评估解题步骤的复杂程度
        }
    
    def _calculate_step_complexity(self, puzzle_data, difficulty=None):
        """计算Kukurasu谜题解题步骤的复杂度"""
        n = puzzle_data["size"]
        row_sums = puzzle_data["row_sums"]
        col_sums = puzzle_data["col_sums"]
        solution = puzzle_data["solution"]
        # 1. 基础复杂度 - 基于网格大小
        base_complexity = n ** 2
        # 2. 约束密度 - 评估黑格子数量与网格总数的比例
        black_count = sum(sum(row) for row in solution)
        black_density = black_count / (n * n) if n > 0 else 0
        density_factor = 4 * black_density * (1 - black_density)
        # 3. 约束相互关系
        max_sum = sum(j + 1 for j in range(n))
        if max_sum == 0: max_sum = 1 # 避免除以零
        row_constraint_ratio = [s / max_sum for s in row_sums]
        col_constraint_ratio = [s / max_sum for s in col_sums]
        row_constraint_tightness = sum(4 * r * (1 - r) for r in row_constraint_ratio) / n
        col_constraint_tightness = sum(4 * r * (1 - r) for r in col_constraint_ratio) / n
        # 4. 解的唯一性评估
        row_black_counts = [sum(row) for row in solution]
        col_black_counts = [sum(solution[i][j] for i in range(n)) for j in range(n)]
        row_repeats = len(row_black_counts) - len(set(row_black_counts))
        col_repeats = len(col_black_counts) - len(set(col_black_counts))
        repeat_factor = 1 + (row_repeats + col_repeats) / (2 * n) * 0.5
        # 5. 相邻单元格分析
        adjacency_count = 0
        for i in range(n):
            for j in range(n):
                if solution[i][j] == 1:
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < n and solution[ni][nj] == 1:
                            adjacency_count += 1
        max_adjacency = 2 * black_count
        adjacency_ratio = (adjacency_count / 2 / max_adjacency) if max_adjacency > 0 else 0
        dispersion_factor = 1 - adjacency_ratio
        # 6. 级别调整
        level_factor = difficulty * 0.5 if difficulty is not None else 1
        # 组合所有因素计算最终复杂度
        complexity_factors = [
            base_complexity * 0.5,
            base_complexity * density_factor * 0.15,
            base_complexity * (row_constraint_tightness + col_constraint_tightness) * 0.15,
            base_complexity * repeat_factor * 0.1,
            base_complexity * dispersion_factor * 0.1
        ]
        step_complexity = int(sum(complexity_factors) * (1 + level_factor))
        return max(10, step_complexity)

    def _generate_question_text(self, puzzle_data, min_size=None, max_size=None):
        """生成带图像的问题描述"""
        n = puzzle_data["size"]
        return (
            f"This is a {n}x{n} Kukurasu puzzle. You need to fill the grid with black cells according to the following rules:\n" 
            f"1. The sum of column positions (1 to {n}) of black cells in each row must equal the number on the right.\n"
            f"2. The sum of row positions (1 to {n}) of black cells in each column must equal the number at the bottom.\n" 
            f"Please solve the puzzle and provide the solution as a two-dimensional array, using 0 for white cells and 1 for blackcells.\n"
            f"Example answer format: [[1, 1, 0], [1, 0, 1], [0, 0, 1]]."
        )
    
    def _generate_detailed_question_text(self, puzzle_data, min_size=None, max_size=None):
        """生成纯文本问题描述"""
        n = puzzle_data["size"]
        row_sums = puzzle_data["row_sums"]
        col_sums = puzzle_data["col_sums"]
        text_lines = [
            f"This is a {n}x{n} Kukurasu puzzle. You need to fill the grid with black cells according to the following rules:\n" 
            f"1. The sum of column positions (1 to {n}) of black cells in each row must equal the number on the right.\n"
            f"2. The sum of row positions (1 to {n}) of black cells in each column must equal the number at the bottom.\n" 
        ]
        for i, row_sum in enumerate(row_sums):
            text_lines.append(f"Row {i+1}: {row_sum}")
        text_lines.append("\nColumn constraints (sum of row positions of black cells in each column):")
        for j, col_sum in enumerate(col_sums):
            text_lines.append(f"Column {j+1}: {col_sum}")
        text_lines.append("\nPlease solve the puzzle and provide the solution as a two-dimensional array, using 0 for white cells and 1 for blackcells.\n"
            "Example answer format: [[1, 1, 0], [1, 0, 1], [0, 0, 1]].")
        return '\n'.join(text_lines)


if __name__ == "__main__":
    # 定义不同难度级别的参数
    difficulties = [1, 2, 3, 4, 5]
    case_num = 5
    
    all_puzzles_across_difficulties = []
    
    # 实例化一次生成器
    # 所有文件将保存在 'output/kukurasu' 目录下
    generator = KukurasuGenerator(output_folder="output/kukurasu")
    
    # 为每个难度级别生成测试用例
    for difficulty in difficulties:
        print(f"正在生成 Kukurasu 难度级别 {difficulty} 的测试用例...")
        puzzles = generator.generate(case_num, difficulty)
        all_puzzles_across_difficulties.extend(puzzles)
        print(f"Kukurasu 难度级别 {difficulty} 的测试用例生成完成！")
    
    # 所有难度生成完毕后，将结果统一保存到一个 annotations.json 文件中
    generator.save_annotations(all_puzzles_across_difficulties, generator.output_folder)
    