#!/usr/bin/env python3
import os
import json
import random
import re
import io
import base64
import time
from itertools import permutations
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
from math import pi
from abc import ABC, abstractmethod

from .base_generator import BaseGenerator


class TwentyFourPointsGenerator(BaseGenerator):
    def __init__(self, output_folder, task_name="24_points"):
        """
        初始化24点问题生成器
        Args:
            output_folder: 输出文件夹路径
        """
        super().__init__(output_folder)

        # 图像参数
        self.card_width = 160
        self.card_height = 240
        self.image_size = (1000, 500)
        self.bg_color = "#F0F4F8"
        self.text_color = "#2D3748"
        self.card_color = "#FFFFFF"
        self.accent_color = "#4C51BF"
        self.font_size_factor = 0.45
        
        # 卡片花色
        self.suits = ["♠", "♥", "♦", "♣"]
        self.suit_colors = {"♠": "#2D3748", "♥": "#E53E3E", "♦": "#E53E3E", "♣": "#2D3748"}
        
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
        if difficulty == 1:
            return {
                'min_num': 1,
                'max_num': 9,
                'allow_repeats': True
            }
        elif difficulty == 2:
            return {
                'min_num': 1,
                'max_num': 13,
                'allow_repeats': True
            }
        elif difficulty == 3:
            return {
                'min_num': 1,
                'max_num': 20,
                'allow_repeats': True
            }
        elif difficulty == 4:
            return {
                'min_num': 1,
                'max_num': 20,
                'allow_repeats': False
            }
        else:  # difficulty 5
            return {
                'min_num': 3,
                'max_num': 25,
                'allow_repeats': False
            }

    def generate(self, num_cases, difficulty, output_folder: str = None):
        """
        生成指定数量的 24 点谜题
        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别
        Returns:
            生成的问题列表
        """
        output_folder = output_folder if output_folder else self.output_folder
        images_dir = os.path.join(output_folder, "images")
        os.makedirs(images_dir, exist_ok=True)

        problems = []
        for i in range(1, num_cases + 1):
            # 生成谜题
            case = self._generate_single_puzzle(i, difficulty=difficulty, output_folder=output_folder, images_dir=images_dir)
            problems.append(case)

        self.save_annotations(problems, output_folder)
        return problems

    def _find_system_font(self):
        """查找系统中可用的字体"""
        common_fonts = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', # Linux
            '/System/Library/Fonts/Helvetica.ttc', # macOS
            'C:/Windows/Fonts/arialbd.ttf', # Windows Bold
            '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf' # Ubuntu
        ]
        for font in common_fonts:
            if os.path.exists(font):
                return font
        return None

    def _generate_single_puzzle(self, case_id, difficulty, output_folder, images_dir):
        """生成单个 24 点谜题"""
        # 设置随机种子
        random.seed(time.time())
        
        MAX_ATTEMPTS = 1000
        params = self._get_difficulty_params(difficulty=difficulty)
        min_num = params['min_num']
        max_num = params['max_num']
        allow_repeats = params['allow_repeats']
        for _ in range(MAX_ATTEMPTS):
            # 根据参数生成不同特性的数字组合
            if allow_repeats:
                numbers = [random.randint(min_num, max_num) for _ in range(4)]
            else:
                numbers = random.sample(range(min_num, max_num+1), 4)
            
            if self._has_solution(numbers):
                # 对数字排序
                sorted_numbers = sorted(numbers)
                # 尝试查找有效解
                solution = self._find_a_solution(sorted_numbers)
                
                # 生成图像，并获取相对路径
                puzzle_image_path, solution_image_path = self.visualize(
                    {'numbers': sorted_numbers, 'solution': solution},
                    difficulty=difficulty,
                    case_id=case_id, 
                    output_folder=output_folder,
                    images_dir=images_dir
                )
                
                # 创建数据点
                return self._create_datapoint(case_id, sorted_numbers, solution, puzzle_image_path, difficulty)
        
        raise RuntimeError("Failed to generate valid case after maximum attempts")

    def visualize(self, puzzle, **kwargs):
        """生成谜题和解答的可视化图像"""
        case_id = kwargs.get('case_id', 0)
        numbers = puzzle['numbers']
        solution = puzzle['solution']
        difficulty = kwargs.get('difficulty', 1)
        output_folder = kwargs.get('output_folder', self.output_folder)
        images_dir = kwargs.get('images_dir', os.path.join(output_folder, "images"))
        
        # 构建新的图片文件名，符合 {taskname}_{level}_{index} 格式
        puzzle_image_name = f"24_points_difficulty_{difficulty}_{case_id}.png"
        solution_image_name = f"solution_24_points_difficulty_{difficulty}_{case_id}.png"
        
        # 构建新的文件路径，都指向统一的 images 文件夹
        puzzle_image_path = os.path.join(images_dir, puzzle_image_name)
        solution_image_path = os.path.join(images_dir, solution_image_name)
        
        # 生成谜题图像
        puzzle_image_data = self._generate_puzzle_image(numbers)
        with open(puzzle_image_path, "wb") as f:
            f.write(base64.b64decode(puzzle_image_data))
        
        # 生成解答图像
        solution_image_data = self._generate_solution_image(numbers, solution)
        with open(solution_image_path, "wb") as f:
            f.write(base64.b64decode(solution_image_data))
        
        # 返回 JSON 中所需的图片相对路径
        relative_image_path = f"images/{puzzle_image_name}"
        return relative_image_path, solution_image_path

    def _has_solution(self, numbers):
        """检查给定的数字是否有解"""
        ops = ['+', '-', '*', '/']
        
        def is_valid(expr_str):
            """检查是否为有效表达式"""
            try:
                return abs(eval(expr_str) - 24) < 1e-6
            except ZeroDivisionError:
                return False
            except Exception as e:
                return False # 避免其他错误
        
        # 生成带括号的表达式
        for a, b, c, d in permutations(numbers):
            for op1 in ops:
                for op2 in ops:
                    for op3 in ops:
                        # 尝试所有可能的括号组合
                        expressions = [
                            f"({a}{op1}{b}){op2}({c}{op3}{d})",
                            f"({a}{op1}({b}{op2}{c})){op3}{d}",
                            f"{a}{op1}({b}{op2}({c}{op3}{d}))",
                            f"({a}{op1}{b}{op2}{c}){op3}{d}",
                            f"{a}{op1}({b}{op2}{c}{op3}{d})",
                            f"({a}{op1}{b}){op2}{c}{op3}{d}",
                            f"{a}{op1}({b}{op2}{c}){op3}{d}"
                        ]
                        for expr in expressions:
                            valid_expr = expr.replace('×', '*').replace('÷', '/').replace(' ', '')
                            if is_valid(valid_expr):
                                return True
        return False

    def _find_a_solution(self, numbers):
        """返回其中一个可行解"""
        ops = ['+', '-', '*', '/']
        
        def is_valid(expr_str):
            """检查是否为有效表达式"""
            try:
                return abs(eval(expr_str) - 24) < 1e-6
            except ZeroDivisionError:
                return False
            except Exception as e:
                return False # 避免其他错误
        
        # 生成带括号的表达式
        for a, b, c, d in permutations(numbers):
            for op1 in ops:
                for op2 in ops:
                    for op3 in ops:
                        # 尝试所有可能的括号组合
                        expressions = [
                            f"({a}{op1}{b}){op2}({c}{op3}{d})",
                            f"({a}{op1}({b}{op2}{c})){op3}{d}",
                            f"{a}{op1}({b}{op2}({c}{op3}{d}))",
                            f"({a}{op1}{b}{op2}{c}){op3}{d}",
                            f"{a}{op1}({b}{op2}{c}{op3}{d})",
                            f"({a}{op1}{b}){op2}{c}{op3}{d}",
                            f"{a}{op1}({b}{op2}{c}){op3}{d}"
                        ]
                        for expr in expressions:
                            valid_expr = expr.replace('×', '*').replace('÷', '/').replace(' ', '')
                            if is_valid(valid_expr):
                                # 美化表达式 - 添加空格和使用更美观的符号
                                pretty_expr = valid_expr.replace('*', ' × ').replace('/', ' ÷ ').replace('+', ' + ').replace('-', ' - ')
                                # 数字后跟括号表示乘法，需要转义括号
                                pretty_expr = re.sub(r'(\d)\(', r'\1 × (', pretty_expr)
                                return pretty_expr
        return None

    def _generate_puzzle_image(self, numbers):
        """生成包含卡片图像的谜题图像"""
        # 图像尺寸
        width, height = self.image_size
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # 绘制背景纹理
        try:
            self._draw_background_pattern(draw, width, height)
        except Exception:
            pass
        
        # 字体大小与卡片高度的比例
        font_size = int(self.card_height * self.font_size_factor)
        try:
            if self.font_path:
                number_font = ImageFont.truetype(self.font_path, font_size)
                small_font = ImageFont.truetype(self.font_path, int(font_size * 0.4))
                title_font = ImageFont.truetype(self.font_path, int(font_size * 0.7))
            else:
                number_font = ImageFont.load_default()
                small_font = ImageFont.load_default()
                title_font = ImageFont.load_default()
        except:
            number_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # 计算卡片起始位置
        total_card_width = 4 * self.card_width
        spacing = (width - total_card_width) / 5 # 左右边距和卡片间距相等
        start_x = spacing
        start_y = (height - self.card_height) / 2
        
        # 绘制标题
        title_text = "24 POINTS PUZZLE"
        title_width = draw.textlength(title_text, font=title_font) if hasattr(draw, 'textlength') else font_size * len(title_text) * 0.6
        draw.text(((width - title_width) // 2, 30), title_text, fill=self.text_color, font=title_font)
        
        # 绘制每张卡片
        for i, num in enumerate(numbers):
            # 选择花色
            suit = self.suits[i % 4]
            suit_color = self.suit_colors[suit]
            
            # 卡片位置
            card_x = start_x + i * (self.card_width + spacing)
            card_y = start_y
            
            # 绘制卡片阴影
            shadow_offset = 6
            shadow_color = "#CBD5E0"
            self._draw_rounded_rectangle(draw,
                (card_x + shadow_offset, card_y + shadow_offset,
                 card_x + self.card_width + shadow_offset,
                 card_y + self.card_height + shadow_offset),
                radius=15, fill=shadow_color, outline=None)
            
            # 绘制卡片背景 - 添加光泽效果
            self._draw_card_with_gloss(draw,
                (card_x, card_y, card_x + self.card_width, card_y + self.card_height),
                radius=15, fill=self.card_color, outline="#A0AEC0", width=2)
            
            # 绘制数字
            text = str(num)
            text_width = draw.textlength(text, font=number_font) if hasattr(draw, 'textlength') else font_size * 0.6
            text_x = card_x + (self.card_width - text_width) / 2
            text_y = card_y + (self.card_height - font_size) / 2 - 10 # 向上偏移一点
            draw.text((text_x, text_y), text, fill=self.text_color, font=number_font)
            
            # 在左上角和右下角绘制花色
            corner_margin = 15
            suit_size = small_font.size if hasattr(small_font, 'size') else 12
        
        # 转换为base64编码的图像
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = base64.b64encode(buffer.getvalue()).decode()
        
        return image_data

    def _generate_solution_image(self, numbers, solution):
        """生成 24点 谜题解答图像"""
        # 图像尺寸
        width, height = self.image_size
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)
        
        # 绘制背景纹理
        try:
            self._draw_background_pattern(draw, width, height)
        except Exception:
            pass
        
        # 字体大小与卡片高度的比例
        font_size = int(self.card_height * self.font_size_factor)
        try:
            if self.font_path:
                number_font = ImageFont.truetype(self.font_path, font_size)
                small_font = ImageFont.truetype(self.font_path, int(font_size * 0.4))
                title_font = ImageFont.truetype(self.font_path, int(font_size * 0.7))
                solution_font = ImageFont.truetype(self.font_path, int(font_size * 0.8))
            else:
                number_font = ImageFont.load_default()
                small_font = ImageFont.load_default()
                title_font = ImageFont.load_default()
                solution_font = ImageFont.load_default()
        except:
            number_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            solution_font = ImageFont.load_default()
        
        # 计算卡片起始位置
        total_card_width = 4 * self.card_width
        spacing = (width - total_card_width) / 5 # 左右边距和卡片间距相等
        start_x = spacing
        start_y = (height - self.card_height) / 2 + 30 # 向下移动一点，为解答腾出空间
        
        # 绘制标题
        title_text = "24 POINTS SOLUTION"
        title_width = draw.textlength(title_text, font=title_font) if hasattr(draw, 'textlength') else len(title_text) * 10
        draw.text(((width - title_width) // 2, 30), title_text, fill=self.text_color, font=title_font)
        
        # 绘制每张卡片
        for i, num in enumerate(numbers):
            # 选择花色
            suit = self.suits[i % 4]
            suit_color = self.suit_colors[suit]
            
            # 卡片位置
            card_x = start_x + i * (self.card_width + spacing)
            card_y = start_y
            
            # 绘制卡片阴影
            shadow_offset = 6
            shadow_color = "#CBD5E0"
            self._draw_rounded_rectangle(draw,
                (card_x + shadow_offset, card_y + shadow_offset,
                 card_x + self.card_width + shadow_offset,
                 card_y + self.card_height + shadow_offset),
                radius=15, fill=shadow_color, outline=None)
            
            # 绘制卡片背景 - 添加光泽效果
            self._draw_card_with_gloss(draw,
                (card_x, card_y, card_x + self.card_width, card_y + self.card_height),
                radius=15, fill=self.card_color, outline="#A0AEC0", width=2)
            
            # 绘制数字
            text = str(num)
            text_width = draw.textlength(text, font=number_font) if hasattr(draw, 'textlength') else font_size * 0.6
            text_x = card_x + (self.card_width - text_width) / 2
            text_y = card_y + (self.card_height - font_size) / 2 - 10 # 向上偏移一点
            draw.text((text_x, text_y), text, fill=self.text_color, font=number_font)
            
            # 在左上角和右下角绘制花色
            corner_margin = 15
            suit_size = small_font.size if hasattr(small_font, 'size') else 12
        
        # 绘制解法
        if solution:
            # 创建一个半透明的背景框来突出显示解答
            solution_box_height = 60
            solution_box_y = height - solution_box_height - 30
            
            # 绘制解答文本
            solution_width = draw.textlength(solution, font=solution_font) if hasattr(draw, 'textlength') else len(solution) * 12
            font_height = getattr(solution_font, 'size', font_size * 0.8)
            draw.text(((width - solution_width) / 2, solution_box_y + (solution_box_height - font_height) / 2),
                      solution, fill=self.accent_color, font=solution_font)
        else:
            # 如果没有解答
            solution_text = "No solution found"
            solution_width = draw.textlength(solution_text, font=solution_font) if hasattr(draw, 'textlength') else len(solution_text) * 12
            
            # 创建一个背景框显示"无解"
            solution_box_height = 60
            solution_box_y = height - solution_box_height - 30
            self._draw_rounded_rectangle(draw,
                (width * 0.3, solution_box_y,
                 width * 0.7, solution_box_y + solution_box_height),
                radius=10, fill="#E53E3E33", outline="#E53E3E", width=2)
                
            font_height = getattr(solution_font, 'size', font_size * 0.8)
            draw.text(((width - solution_width) / 2, solution_box_y + (solution_box_height - font_height) / 2),
                      solution_text, fill="#E53E3E", font=solution_font)
        
        # 转换为base64编码的图像
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = base64.b64encode(buffer.getvalue()).decode()
        
        return image_data

    def _draw_rounded_rectangle(self, draw, xy, radius, fill=None, outline=None, width=1):
        """绘制圆角矩形"""
        x1, y1, x2, y2 = xy
        diameter = 2 * radius
        
        # 绘制填充区域
        if fill:
            draw.rectangle([x1 + radius, y1, x2 - radius, y2], fill=fill)
            draw.rectangle([x1, y1 + radius, x2, y2 - radius], fill=fill)
            draw.ellipse([x1, y1, x1 + diameter, y1 + diameter], fill=fill)
            draw.ellipse([x2 - diameter, y1, x2, y1 + diameter], fill=fill)
            draw.ellipse([x1, y2 - diameter, x1 + diameter, y2], fill=fill)
            draw.ellipse([x2 - diameter, y2 - diameter, x2, y2], fill=fill)
        
        # 绘制轮廓
        if outline:
            draw.arc([x1, y1, x1 + diameter, y1 + diameter], 180, 270, fill=outline, width=width)
            draw.arc([x2 - diameter, y1, x2, y1 + diameter], 270, 0, fill=outline, width=width)
            draw.arc([x1, y2 - diameter, x1 + diameter, y2], 90, 180, fill=outline, width=width)
            draw.arc([x2 - diameter, y2 - diameter, x2, y2], 0, 90, fill=outline, width=width)
            draw.line([x1 + radius, y1, x2 - radius, y1], fill=outline, width=width)
            draw.line([x1 + radius, y2, x2 - radius, y2], fill=outline, width=width)
            draw.line([x1, y1 + radius, x1, y2 - radius], fill=outline, width=width)
            draw.line([x2, y1 + radius, x2, y2 - radius], fill=outline, width=width)

    def _draw_card_with_gloss(self, draw, xy, radius, fill=None, outline=None, width=1):
        """绘制带有光泽效果的卡片"""
        # 先绘制基本圆角矩形
        self._draw_rounded_rectangle(draw, xy, radius, fill, outline, width)
        
        # 添加光泽效果 - 顶部半透明白色渐变
        x1, y1, x2, y2 = xy
        gloss_height = int((y2 - y1) // 3) # 确保是整数
        
        # 创建局部区域的半透明渐变效果
        for i in range(gloss_height):
            # 计算当前位置的透明度 - 从顶部（较白）到底部（完全透明）
            alpha = int(150 * (1 - i / gloss_height)) if gloss_height > 0 else 0 # 防止除零错误
            gloss_color = f"#FFFFFF{alpha:02X}"
            
            # 绘制水平线条，每行透明度递减
            y = y1 + i
            
            # 确保在圆角范围内绘制
            if i < radius:
                # 在圆角区域宽度减小
                offset = int(radius * (1 - ((i + 1) / radius) * 0.5))
                draw.line([x1 + offset, y, x2 - offset, y], fill=gloss_color, width=1)
            else:
                draw.line([x1, y, x2, y], fill=gloss_color, width=1)

    def _draw_background_pattern(self, draw, width, height):
        """绘制背景纹理"""
        # 绘制轻微的网格纹理
        grid_color = "#E2E8F0" # 浅色网格
        grid_spacing = 20
        
        # 横线
        for y in range(0, height, grid_spacing):
            draw.line([(0, y), (width, y)], fill=grid_color, width=1)
        
        # 竖线
        for x in range(0, width, grid_spacing):
            draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
        
        # 添加一些随机的装饰点
        num_dots = 50
        dot_color = "#CBD5E0"
        for _ in range(num_dots):
            x = random.randint(0, width)
            y = random.randint(0, height)
            dot_size = random.randint(2, 5)
            draw.ellipse([x-dot_size, y-dot_size, x+dot_size, y+dot_size], fill=dot_color)

    def _create_datapoint(self, case_id, numbers, solution, puzzle_image_path, difficulty):
        """创建数据点，并添加step字段评估解题步骤复杂度"""
        # 计算解题步骤复杂度
        step_complexity = self._calculate_step_complexity(numbers, solution, difficulty)
        
        return {
            "index": f"24Points_difficulty{difficulty}_{case_id}",
            "category": "24Points",
            "question": self._generate_question_text(numbers),
            "image": puzzle_image_path,
            "question_language": self._generate_detailed_question_text(numbers),
            "answer": f"{solution}" if solution else "[[No solution found]]",
            "initial_state": {"numbers": numbers},
            "difficulty": difficulty,
#           "solution": solution,
#           "step": step_complexity # 新增字段：评估解题步骤的复杂程度
        }

    def _calculate_step_complexity(self, numbers, solution, difficulty):
        """计算24点游戏解题步骤的复杂度"""
        # 如果没有解答，设置最高复杂度
        if not solution:
            return 100
        
        # 1. 数字大小因子 - 较大的数字通常更难处理
        number_size_factor = sum(min(num, 15) for num in numbers) / len(numbers)
        
        # 2. 数字多样性 - 不同数字的数量影响复杂度
        unique_count = len(set(numbers))
        diversity_factor = unique_count / len(numbers) # 数字越多样，复杂度越高
        
        # 3. 解法复杂度分析
        # 分析解法中的运算符数量和括号嵌套深度
        if solution:
            # 运算符复杂度 - 乘除法比加减法复杂
            op_complexity = {
                '+': 1.0,
                '-': 1.2,
                '×': 1.5,
                '÷': 2.0
            }
            
            # 计算各类运算符的数量
            op_counts = {
                '+': solution.count('+'),
                '-': solution.count('-'),
                '×': solution.count('×'),
                '÷': solution.count('÷')
            }
            
            # 加权运算符复杂度
            weighted_op_complexity = sum(op_counts[op] * op_complexity[op] for op in op_counts)
            
            # 括号嵌套深度分析
            max_bracket_depth = 0
            current_depth = 0
            for char in solution:
                if char == '(':
                    current_depth += 1
                    max_bracket_depth = max(max_bracket_depth, current_depth)
                elif char == ')':
                    current_depth -= 1
            
            # 括号复杂度因子
            bracket_factor = 1 + max_bracket_depth * 0.5
            
            # 解法长度因子
            length_factor = len(solution) / 20 # 标准化长度
            
            # 组合运算的复杂度
            solution_complexity = weighted_op_complexity * bracket_factor * length_factor
        else:
            solution_complexity = 1.0 # 默认值
        
        # 4. 特殊情况分析
        special_case_factor = 1.0
        
        # 包含1的情况通常较简单
        if 1 in numbers:
            special_case_factor *= 0.9
        
        # 包含特定组合的情况
        if 8 in numbers and 3 in numbers: # 8和3常见组合
            special_case_factor *= 0.9
        if 6 in numbers and 4 in numbers: # 6和4常见组合
            special_case_factor *= 0.9
        
        # 全部是奇数或全部是偶数的情况通常更复杂
        if all(num % 2 == 0 for num in numbers) or all(num % 2 == 1 for num in numbers):
            special_case_factor *= 1.2
        
        # 5. 运算路径分析
        # 分析解法中是否需要中间结果为分数或较大数字
        path_complexity = 1.0
        try:
            # 尝试解析表达式中的运算步骤
            if solution:
                # 简化表达式分析 - 检查是否有除法
                if '÷' in solution:
                    path_complexity *= 1.3 # 含除法运算路径通常更复杂
                # 检查是否有连续的乘除运算
                if '×' in solution and '÷' in solution:
                    path_complexity *= 1.2
        except:
            # 解析失败，使用默认值
            path_complexity = 1.0
        
        # 6. 级别调整
        level_factor = 1 + (difficulty - 1) * 0.4 # 根据难度级别1-5调整
        
        # 组合所有因子计算最终复杂度
        base_complexity = 15 # 基础复杂度
        complexity_factors = [
            number_size_factor * 0.5, # 数字大小影响
            diversity_factor * 5, # 数字多样性影响
            solution_complexity * 2, # 解法复杂度影响
            special_case_factor, # 特殊情况影响
            path_complexity # 运算路径影响
        ]
        
        # 计算总和作为修正因子
        adjustment_factor = sum(complexity_factors)
        
        # 最终复杂度计算
        step_complexity = int(base_complexity * adjustment_factor * level_factor)
        
        # 设置最小值为10
        step_complexity = max(10, step_complexity)
        
        return step_complexity

    def _generate_question_text(self, numbers):
        """生成带图像的问题描述"""
        return (
            f"Use these numbers exactly once, and combine them with +, -, ×, ÷, and parentheses to make 24.\n"
            f"Please provide your answer as an expression that includes only numbers, operators, and parentheses.\n"
            f"Example answer format: (9 - 3) × 8 ÷ 2."
        )

    def _generate_detailed_question_text(self, numbers):
        """生成纯文本问题描述"""
        example = "(9 - 5) × (7 - 2)" if 9 in numbers else "8 × (3 + 1) - 8"
        return (
            f"Use these numbers exactly once, and combine them with +, -, ×, ÷, and parentheses to make 24.\n"
            f"Numbers: {numbers}\n"
            f"Please provide your answer as an expression that includes only numbers, operators, and parentheses.\n"
            f"Example answer format: (9 - 3) × 8 ÷ 2."
        )

    def solve(self, puzzle, **kwargs):
        """求解 24 点谜题 - 返回已计算出的解答"""
        return puzzle.get('solution', None)

    @staticmethod
    def extract_output(output):
        """从模型输出中提取答案"""
        if isinstance(output, dict) and "text" in output:
            output = output["text"]
        matches = re.findall(r'\[(.*?)]', output)
        return matches[-1].strip() if matches else None

    def verify_solution(self, solution_str, puzzle_data):
        """验证提取的答案是否正确"""
        try:
            # 提取并验证数字使用
            input_nums = sorted(puzzle_data["numbers"])
            used_nums = sorted(map(int, re.findall(r'\b\d+\b', solution_str)))
            if input_nums != used_nums:
                return False
            
            # 数学表达式验证
            expr = solution_str.replace('×', '*').replace('÷', '/').replace(' ', '')
            return abs(eval(expr) - 24) < 1e-6
        except:
            return False

