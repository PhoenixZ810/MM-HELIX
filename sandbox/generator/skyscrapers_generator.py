#!/usr/bin/env python3

import os
import json
import random
import io
import base64
import time # 导入time模块
from typing import Dict, List, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
from abc import ABC, abstractmethod

from .base_generator import BaseGenerator


class SkyscrapersGenerator(BaseGenerator):
    # 难度级别配置定义
    DIFFICULTY_CONFIGS = {
        1: {"n": 3},  # Level 1: 3x3 网格
        2: {"n": 4},  # Level 2: 4x4 网格
        3: {"n": 5},  # Level 3: 5x5 网格
        4: {"n": 6},  # Level 4: 6x6 网格
        5: {"n": 7}  # Level 5: 7x7 网格
    }

    def __init__(self, output_folder="output/skyscrapers"):
        """
        初始化摩天楼谜题生成器。
    
        Args:
            output_folder: 输出文件夹路径
        """
        super().__init__(output_folder)
        
        # 从难度配置中获取参数，这里我们在generate方法中处理difficulty
        # self.level = kwargs.get('level', 1) # 不再在__init__中直接设置level
        # difficulty_config = self.DIFFICULTY_CONFIGS.get(self.level, self.DIFFICULTY_CONFIGS[1])
        
        # 初始化参数的默认值，这些参数可以在generate方法中根据difficulty调整
        self.n = 0  # 初始为0，会在generate方法中根据难度设置
        self.image_size = (800, 800)
        self.cell_size = 80
        self.bg_color = "#f8f9fa"
        self.grid_color = "#343a40"
        self.text_color = "#212529"
        self.clue_color = "#e74c3c"
        
        # 摩天楼颜色渐变
        self.skyscraper_colors = [
            "#caf0f8",  # 最矮
            "#90e0ef",
            "#00b4d8",
            "#0077b6",
            "#03045e"  # 最高
        ]
        
        # 查找系统字体
        self.font_path = self._find_system_font()      
        # 初始化annotations列表，用于存储所有难度级别的JSON描述
        self.annotations = []
        
    def generate(self, num_cases: int, difficulty: int, output_folder: str = None):
        """
        生成指定数量的摩天楼谜题。
    
        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别
            output_folder: 输出文件夹路径，覆盖构造函数中设置的路径
        """
        output_folder = output_folder if output_folder else self.output_folder
        images_dir = os.path.join(output_folder, "images")
        os.makedirs(images_dir, exist_ok=True)
        

        
        # 获取难度参数
        difficulty_config = self._get_difficulty_params(difficulty)
        n = difficulty_config["n"]
        
        problems = []
        for i in range(1, num_cases + 1):
            # 设置随机种子
            random.seed(time.time())
            # 生成谜题
            puzzle = self._generate_single_puzzle(i, difficulty=difficulty, n=n, output_folder=output_folder)
            problems.append(puzzle)
            
        self.save_annotations(problems, output_folder)
        return problems
    
    def _generate_single_puzzle(self, case_id, difficulty, n, output_folder):
        """生成单个摩天楼谜题"""
        solution = self._generate_solution(n)
        
        # 计算四边的可见性
        left = [self._count_visible(row) for row in solution]
        right = [self._count_visible(list(reversed(row))) for row in solution]
        top = []
        bottom = []
        
        for col in range(n):
            column = [solution[row][col] for row in range(n)]
            top.append(self._count_visible(column))
            bottom.append(self._count_visible(list(reversed(column))))
            
        # 创建谜题数据
        puzzle_data = {
            'n': n,
            'top': top,
            'bottom': bottom,
            'left': left,
            'right': right,
            'solution': solution
        }
        
        # 计算解题步骤复杂度
        step_complexity = self._calculate_step_complexity(puzzle_data, difficulty)
        
        # 生成图像
        puzzle_image_path, solution_image_path = self._visualize(puzzle_data, case_id=case_id, difficulty=difficulty, output_folder=output_folder, n=n)
        
        # 创建数据点
        return {
            "index": f"Skyscrapers_difficulty{difficulty}_{case_id}",
            "category": "Skyscrapers",
            "level": difficulty,
            "question": self._generate_question_text(puzzle_data),
            "image": puzzle_image_path,
            "question_language": self._generate_detailed_question_text(puzzle_data),
            "answer": solution,
#           "n": self.n,
#           "top": top,
#           "bottom": bottom,
#           "left": left,
#           "right": right,
            "initial_state": {
                "n": n,
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right,
            },
#           "solution": solution,
#           "step": step_complexity
        }
    
    def _visualize(self, puzzle_data, output_folder, **kwargs):
        """生成谜题和解答的可视化图像"""
        case_id = kwargs.get('case_id', 0)
        difficulty = kwargs.get('difficulty', 1)
        taskname = "skyscrapers"
        
        puzzle_image_path = f"images/{taskname}_level{difficulty}_{case_id}.png"
        puzzle_image_full_path = os.path.join(output_folder, puzzle_image_path)
        
        puzzle_image_data = self._generate_puzzle_image(puzzle_data, difficulty)
        with open(puzzle_image_full_path, "wb") as f:
            f.write(base64.b64decode(puzzle_image_data))
            
        solution_image_path = f"images/{taskname}_level{difficulty}_{case_id}_solution.png"
        solution_image_full_path = os.path.join(output_folder, solution_image_path)
        
        solution_image_data = self._generate_solution_image(puzzle_data, difficulty)
        with open(solution_image_full_path, "wb") as f:
            f.write(base64.b64decode(solution_image_data))
            
        return puzzle_image_path, solution_image_path

    def _get_difficulty_params(self, difficulty: int) -> Dict[str, Any]:
        """
        根据难度级别获取相应的参数配置。

        Args:
            difficulty: 难度级别（1-5）

        Returns:
            dict: 包含难度参数的字典
        """
        if difficulty not in self.DIFFICULTY_CONFIGS:
            raise ValueError(f"Difficulty level {difficulty} not supported. Choose from 1 to 5.")
        return self.DIFFICULTY_CONFIGS[difficulty]

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

    
    def _calculate_step_complexity(self, puzzle_data, difficulty):
        """计算摩天楼谜题解题步骤的复杂度"""
        n = puzzle_data['n']
        top = puzzle_data['top']
        bottom = puzzle_data['bottom']
        left = puzzle_data['left']
        right = puzzle_data['right']
        solution = puzzle_data['solution']

        # 1. 基础复杂度 - 基于网格大小
        base_complexity = n ** 2.5

        # 2. 可见摩天楼约束强度
        def constraint_strength(values):
            strengths = []
            for value in values:
                dist_to_min = value - 1
                dist_to_max = n - value
                min_dist = min(dist_to_min, dist_to_max)
                normalized_dist = min_dist / ((n - 1) / 2)
                strength = 1 - normalized_dist
                strengths.append(strength)
            return sum(strengths) / len(strengths)

        top_strength = constraint_strength(top)
        bottom_strength = constraint_strength(bottom)
        left_strength = constraint_strength(left)
        right_strength = constraint_strength(right)

        avg_constraint_strength = (top_strength + bottom_strength + left_strength + right_strength) / 4

        # 3. 约束稀疏度 - 约束越少，问题越难
        missing_constraints = []
        for direction in [top, bottom, left, right]:
            missing_constraints.extend([1 for v in direction if v == 0])

        missing_factor = 1 + len(missing_constraints) / (4 * n) * 0.5

        # 4. 冲突约束分析 - 互相矛盾的约束导致更困难的推理
        conflict_score = 0
        for i in range(n):
            if left[i] > n / 2 and right[i] > n / 2:
                conflict_score += (left[i] + right[i]) / (2 * n)
            if top[i] > n / 2 and bottom[i] > n / 2:
                conflict_score += (top[i] + bottom[i]) / (2 * n)

        conflict_factor = 1 + (conflict_score / n) * 0.5

        # 5. 极端约束分析 - 极端值(1或n)提供强力约束，简化问题
        extreme_constraints = 0
        for direction in [top, bottom, left, right]:
            extreme_constraints += sum(1 for v in direction if v == 1 or v == n)

        extreme_factor = 1 - (extreme_constraints / (4 * n)) * 0.3

        # 6. 级别调整
        level_factor = 1 + (difficulty - 1) * 0.5

        # 综合计算复杂度
        complexity_factors = [
            base_complexity,
            1 + avg_constraint_strength * 0.5,
            missing_factor,
            conflict_factor,
            extreme_factor
        ]

        total_factor = 1
        for factor in complexity_factors[1:]:
            total_factor *= factor

        step_complexity = int(base_complexity * total_factor * level_factor)

        step_complexity = max(10, step_complexity)

        return step_complexity

    def _generate_solution(self, n):
        """生成符合摩天楼规则的解答"""

        base = [[(i + j) % n + 1 for j in range(n)] for i in range(n)]

        random.shuffle(base)

        cols = list(range(n))
        random.shuffle(cols)
        solution = []
        for row in base:
            new_row = [row[col] for col in cols]
            solution.append(new_row)

        for _ in range(n):
            i, j = random.sample(range(n), 2)
            solution[i], solution[j] = solution[j], solution[i]

        for _ in range(n):
            i, j = random.sample(range(n), 2)
            for row in solution:
                row[i], row[j] = row[j], row[i]

        return solution

    @staticmethod
    def _count_visible(sequence):
        """计算序列中可见的摩天楼数量"""
        max_h, count = 0, 0
        for num in sequence:
            if num > max_h:
                count += 1
                max_h = num
        return count

    

    def _generate_puzzle_image(self, puzzle_data, difficulty):
        """为摩天楼谜题生成图像"""
        n = puzzle_data['n']
        top = puzzle_data['top']
        bottom = puzzle_data['bottom']
        left = puzzle_data['left']
        right = puzzle_data['right']

        width, height = self.image_size
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)

        title_font_size = 36
        clue_font_size = 28

        try:
            if self.font_path:
                title_font = ImageFont.truetype(self.font_path, title_font_size)
                clue_font = ImageFont.truetype(self.font_path, clue_font_size)
            else:
                title_font = ImageFont.load_default()
                clue_font = ImageFont.load_default()
        except:
            title_font = ImageFont.load_default()
            clue_font = ImageFont.load_default()

        title = "Skyscrapers Puzzle"
        title_width = draw.textlength(title, font=title_font) if hasattr(draw, 'textlength') else title_font_size * len(title) * 0.6
        draw.text(((width - title_width) // 2, 40), title, fill=self.text_color, font=title_font)

        grid_width = (n + 2) * self.cell_size
        grid_height = (n + 2) * self.cell_size
        start_x = (width - grid_width) // 2 + self.cell_size
        start_y = 160

        grid_center_y = start_y + (n * self.cell_size) // 2

        for i, num in enumerate(top):
            x = start_x + i * self.cell_size + self.cell_size // 2
            y = start_y - self.cell_size // 2
            clue_width = draw.textlength(str(num), font=clue_font) if hasattr(draw, 'textlength') else clue_font_size * 0.6
            draw.text((x - clue_width // 2, y - clue_font_size // 2),
                      str(num), fill=self.clue_color, font=clue_font)

        for i, num in enumerate(bottom):
            x = start_x + i * self.cell_size + self.cell_size // 2
            y = start_y + n * self.cell_size + self.cell_size // 2
            clue_width = draw.textlength(str(num), font=clue_font) if hasattr(draw, 'textlength') else clue_font_size * 0.6
            draw.text((x - clue_width // 2, y - clue_font_size // 2),
                      str(num), fill=self.clue_color, font=clue_font)

        for i, num in enumerate(left):
            x = start_x - self.cell_size // 2
            y = start_y + i * self.cell_size + self.cell_size // 2
            clue_width = draw.textlength(str(num), font=clue_font) if hasattr(draw, 'textlength') else clue_font_size * 0.6
            draw.text((x - clue_width // 2, y - clue_font_size // 2),
                      str(num), fill=self.clue_color, font=clue_font)

        for i, num in enumerate(right):
            x = start_x + n * self.cell_size + self.cell_size // 2
            y = start_y + i * self.cell_size + self.cell_size // 2
            clue_width = draw.textlength(str(num), font=clue_font) if hasattr(draw, 'textlength') else clue_font_size * 0.6
            draw.text((x - clue_width // 2, y - clue_font_size // 2),
                      str(num), fill=self.clue_color, font=clue_font)

        for i in range(n):
            for j in range(n):
                x = start_x + j * self.cell_size
                y = start_y + i * self.cell_size

                draw.rectangle([x, y, x + self.cell_size, y + self.cell_size],
                                outline=self.grid_color, width=2, fill="#ffffff")

                x_text = "X"
                x_width = draw.textlength(x_text, font=clue_font) if hasattr(draw, 'textlength') else clue_font_size * 0.6
                draw.text((x + (self.cell_size - x_width) // 2, y + (self.cell_size - clue_font_size) // 2),
                          x_text, fill=self.text_color, font=clue_font)

        arrow_size = 24

        draw.polygon([(width // 2, start_y - self.cell_size),
                      (width // 2 - arrow_size // 2, start_y - self.cell_size + arrow_size),
                      (width // 2 + arrow_size // 2, start_y - self.cell_size + arrow_size)],
                     fill=self.clue_color)

        draw.polygon([(width // 2, start_y + n * self.cell_size + self.cell_size),
                      (width // 2 - arrow_size // 2, start_y + n * self.cell_size + self.cell_size - arrow_size),
                      (width // 2 + arrow_size // 2, start_y + n * self.cell_size + self.cell_size - arrow_size)],
                     fill=self.clue_color)

        draw.polygon([(start_x - self.cell_size, grid_center_y),
                      (start_x - self.cell_size + arrow_size, grid_center_y - arrow_size // 2),
                      (start_x - self.cell_size + arrow_size, grid_center_y + arrow_size // 2)],
                     fill=self.clue_color)

        draw.polygon([(start_x + n * self.cell_size + self.cell_size, grid_center_y),
                      (start_x + n * self.cell_size + self.cell_size - arrow_size, grid_center_y - arrow_size // 2),
                      (start_x + n * self.cell_size + self.cell_size - arrow_size, grid_center_y + arrow_size // 2)],
                     fill=self.clue_color)

        level_text = f"Level: {difficulty}"
        level_font_size = clue_font_size
        try:
            level_font = ImageFont.truetype(self.font_path, level_font_size) if self.font_path else ImageFont.load_default()
        except:
            level_font = ImageFont.load_default()

        level_width = draw.textlength(level_text, font=level_font) if hasattr(draw, 'textlength') else level_font_size * len(level_text) * 0.6
        draw.text((width - level_width - 20, 20), level_text, fill=self.text_color, font=level_font)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = base64.b64encode(buffer.getvalue()).decode()

        return image_data

    def _generate_solution_image(self, puzzle_data, difficulty):
        """为摩天楼谜题的解答生成图像"""
        n = puzzle_data['n']
        top = puzzle_data['top']
        bottom = puzzle_data['bottom']
        left = puzzle_data['left']
        right = puzzle_data['right']
        solution = puzzle_data['solution']

        width, height = self.image_size
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)

        title_font_size = 36
        clue_font_size = 28
        number_font_size = 32

        try:
            if self.font_path:
                title_font = ImageFont.truetype(self.font_path, title_font_size)
                clue_font = ImageFont.truetype(self.font_path, clue_font_size)
                number_font = ImageFont.truetype(self.font_path, number_font_size)
            else:
                title_font = ImageFont.load_default()
                clue_font = ImageFont.load_default()
                number_font = ImageFont.load_default()
        except:
            title_font = ImageFont.load_default()
            clue_font = ImageFont.load_default()
            number_font = ImageFont.load_default()

        title = "Skyscrapers Solution"
        title_width = draw.textlength(title, font=title_font) if hasattr(draw, 'textlength') else title_font_size * len(title) * 0.6
        draw.text(((width - title_width) // 2, 40), title, fill=self.text_color, font=title_font)

        grid_width = (n + 2) * self.cell_size
        grid_height = (n + 2) * self.cell_size
        start_x = (width - grid_width) // 2 + self.cell_size
        start_y = 180

        grid_center_y = start_y + (n * self.cell_size) // 2

        for i, num in enumerate(top):
            x = start_x + i * self.cell_size + self.cell_size // 2
            y = start_y - self.cell_size // 2
            clue_width = draw.textlength(str(num), font=clue_font) if hasattr(draw, 'textlength') else clue_font_size * 0.6
            draw.text((x - clue_width // 2, y - clue_font_size // 2),
                      str(num), fill=self.clue_color, font=clue_font)

        for i, num in enumerate(bottom):
            x = start_x + i * self.cell_size + self.cell_size // 2
            y = start_y + n * self.cell_size + self.cell_size // 2
            clue_width = draw.textlength(str(num), font=clue_font) if hasattr(draw, 'textlength') else clue_font_size * 0.6
            draw.text((x - clue_width // 2, y - clue_font_size // 2),
                      str(num), fill=self.clue_color, font=clue_font)

        for i, num in enumerate(left):
            x = start_x - self.cell_size // 2
            y = start_y + i * self.cell_size + self.cell_size // 2
            clue_width = draw.textlength(str(num), font=clue_font) if hasattr(draw, 'textlength') else clue_font_size * 0.6
            draw.text((x - clue_width // 2, y - clue_font_size // 2),
                      str(num), fill=self.clue_color, font=clue_font)

        for i, num in enumerate(right):
            x = start_x + n * self.cell_size + self.cell_size // 2
            y = start_y + i * self.cell_size + self.cell_size // 2
            clue_width = draw.textlength(str(num), font=clue_font) if hasattr(draw, 'textlength') else clue_font_size * 0.6
            draw.text((x - clue_width // 2, y - clue_font_size // 2),
                      str(num), fill=self.clue_color, font=clue_font)

        for i in range(n):
            for j in range(n):
                x = start_x + j * self.cell_size
                y = start_y + i * self.cell_size

                height_value = solution[i][j]

                color_index = min(height_value - 1, len(self.skyscraper_colors) - 1)
                skyscraper_color = self.skyscraper_colors[color_index]

                draw.rectangle([x, y, x + self.cell_size, y + self.cell_size],
                                outline=self.grid_color, width=2, fill="#ffffff")

                self._draw_skyscraper(draw, x, y, self.cell_size, height_value, n, skyscraper_color, alpha=0.5)

                number_text = str(height_value)
                number_width = draw.textlength(number_text, font=number_font) if hasattr(draw, 'textlength') else number_font_size * 0.6

                text_color = "#000000"
                outline_color = "#ffffff"
                outline_width = 2

                for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                    draw.text((x + (self.cell_size - number_width) // 2 + dx * outline_width,
                                y + (self.cell_size - number_font_size) // 2 + dy * outline_width),
                            number_text, fill=outline_color, font=number_font)

                draw.text((x + (self.cell_size - number_width) // 2, y + (self.cell_size - number_font_size) // 2),
                          number_text, fill=text_color, font=number_font)

        verification_y = start_y + grid_height + 30

        row_valid = all(sorted(row) == list(range(1, n+1)) for row in solution)

        columns = [[solution[i][j] for i in range(n)] for j in range(n)]
        col_valid = all(sorted(col) == list(range(1, n+1)) for col in columns)

        visibility_valid = True
        for i in range(n):
            if self._count_visible(solution[i]) != left[i]:
                visibility_valid = False
                break
            if self._count_visible(list(reversed(solution[i]))) != right[i]:
                visibility_valid = False
                break

        if visibility_valid: # Only check column visibility if row visibility is valid to avoid redundant breaks
            for j in range(n):
                column = [solution[i][j] for i in range(n)]
                if self._count_visible(column) != top[j]:
                    visibility_valid = False
                    break
                if self._count_visible(list(reversed(column))) != bottom[j]:
                    visibility_valid = False
                    break

        if row_valid and col_valid and visibility_valid:
            verification_text = "VALID SOLUTION: All constraints satisfied"
            verification_color = "#27ae60"
        else:
            verification_text = "INVALID SOLUTION"
            if not row_valid:
                verification_text += ": Row constraint violated"
            elif not col_valid:
                verification_text += ": Column constraint violated"
            else:
                verification_text += ": Visibility constraint violated"
            verification_color = "#e74c3c"

        verification_width = draw.textlength(verification_text, font=clue_font) if hasattr(draw, 'textlength') else clue_font_size * len(verification_text) * 0.6
        draw.text(((width - verification_width) // 2, verification_y),
                  verification_text, fill=verification_color, font=clue_font)

        level_text = f"Level: {difficulty}"
        level_font_size = clue_font_size
        try:
            level_font = ImageFont.truetype(self.font_path, level_font_size) if self.font_path else ImageFont.load_default()
        except:
            level_font = ImageFont.load_default()

        level_width = draw.textlength(level_text, font=level_font) if hasattr(draw, 'textlength') else level_font_size * len(level_text) * 0.6
        draw.text((width - level_width - 20, 20), level_text, fill=self.text_color, font=level_font)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = base64.b64encode(buffer.getvalue()).decode()

        return image_data

    def _draw_skyscraper(self, draw, x, y, cell_size, height, max_height, color, alpha=1.0):
        """
        绘制一个摩天楼图标
        """
        building_width = cell_size * 0.6
        building_height = cell_size * 0.7 * (height / max_height)

        building_x = x + (cell_size - building_width) // 2
        building_y = y + cell_size - building_height - 5

        if alpha < 1.0:
            if color.startswith('#'):
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
            else:
                r, g, b = 176, 224, 230

            bg_r, bg_g, bg_b = 255, 255, 255
            r = int(r * alpha + bg_r * (1 - alpha))
            g = int(g * alpha + bg_g * (1 - alpha))
            b = int(b * alpha + bg_b * (1 - alpha))

            color = f"#{r:02x}{g:02x}{b:02x}"

        draw.rectangle([building_x, building_y, building_x + building_width, y + cell_size - 5],
                       fill=color, outline="#333333", width=1)

        window_size = max(3, int(building_width / 5))
        window_margin = max(2, int(window_size / 2))
        windows_per_floor = max(1, int(building_width / (window_size + window_margin)) - 1)
        floors = max(1, int(building_height / (window_size + window_margin)) - 1)

        window_color = "#ffffff"
        if alpha < 1.0:
            window_color = "#f0f0f0"

        for floor in range(floors):
            for window in range(windows_per_floor):
                window_x = building_x + window_margin + window * (window_size + window_margin)
                window_y = building_y + window_margin + floor * (window_size + window_margin)
                draw.rectangle([window_x, window_y, window_x + window_size, window_y + window_size],
                              fill=window_color, outline=None)

    def solve(self, puzzle: Dict[str, Any]) -> List[List[int]]:
        """
        返回谜题的解答。
        """
        return puzzle["solution"]

    def _generate_question_text(self, puzzle_data):
        """生成带图像的问题描述"""
        n = puzzle_data['n']

        return (
            f"Arrange skyscrapers of heights 1-{n} on this {n}x{n} grid. The rules are as follows: \n"
            f"1. Each row and column must contain exactly one of each height (1 to {n}).\n"
            f"2. The numbers around the grid indicate how many skyscrapers are visible when looking from that direction, with taller buildings obscuring shorter ones behind them.\n"
            f"Please provide your answer as a two-dimensional list.\n"
            f"Example answer format: [[1, 2, 3, 4], [4, 3, 2, 1], [2, 1, 4, 3], [3, 4, 1, 2]]."
        )

    def _generate_detailed_question_text(self, puzzle_data):
        """生成纯文本问题描述"""
        n = puzzle_data['n']
        top = puzzle_data['top']
        bottom = puzzle_data['bottom']
        left = puzzle_data['left']
        right = puzzle_data['right']

        grid_layout = "Grid Layout:\n"

        grid_layout += " " * 2 + " ".join(map(str, top)) + "\n"

        for i in range(n):
            left_clue = left[i]
            right_clue = right[i]
            grid_layout += f"{left_clue} " + " ".join(["X"] * n) + f" {right_clue}\n"

        grid_layout += " " * 2 + " ".join(map(str, bottom)) + "\n"

        return (
            f"Arrange skyscrapers of heights 1-{n} on this {n}x{n} grid. The rules are as follows: \n"
            f"1. Each row and column must contain exactly one of each height (1 to {n}).\n"
            f"2. The numbers around the grid indicate how many skyscrapers are visible when looking from that direction, with taller buildings obscuring shorter ones behind them.\n"
            f"Grid: {grid_layout}\n"
            f"Please provide your answer as a two-dimensional list.\n"
            f"Example answer format: [[1, 2, 3, 4], [4, 3, 2, 1], [2, 1, 4, 3], [3, 4, 1, 2]]."
        )

    def _format_solution(self, solution):
        """将解决方案格式化为答案字符串"""
        return "[[" + ", ".join([" ".join(map(str, row)) for row in solution]) + "]]"


if __name__ == "__main__":
    # 定义不同难度级别的参数
    level_configs = [
        {"difficulty": 1},
        {"difficulty": 2},
        {"difficulty": 3},
        {"difficulty": 4},
        {"difficulty": 5},
    ]
    num_cases_per_difficulty = 5 # 每个难度级别生成5个用例

    # 为每个难度级别生成测试用例
    for config in level_configs:
        current_difficulty = config["difficulty"]

        # 实例化生成器，只传入 output_folder
        generator_Skyscrapers = SkyscrapersGenerator()
        # 调用 generate 方法，传入 num_cases 和 difficulty
        generator_Skyscrapers.generate(num_cases_per_difficulty, current_difficulty)
        print(f"Skyscrapers 难度级别 {current_difficulty} 的测试用例生成完成！")

    print("所有难度级别的测试用例生成完成！")