import os
import json
import random
import io
import base64
from functools import reduce
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
from abc import ABC, abstractmethod
from .base_generator import BaseGenerator


class CalcudokuGenerator(BaseGenerator):
    def __init__(self, output_folder="output/calcudoku", **kwargs):
        super().__init__(output_folder)
      
        # 配置参数
        self.image_size = kwargs.get('image_size', (800, 800))
        self.cell_size = kwargs.get('cell_size', 80)
        self.bg_color = kwargs.get('bg_color', "#f8f9fa")
        self.grid_color = kwargs.get('grid_color', "#343a40")
        self.text_color = kwargs.get('text_color', "#212529")
      
        # 区域颜色
        self.region_colors = kwargs.get('region_colors', [
            "#e6f2ff", "#ffe6e6", "#e6ffe6", "#fff2e6", 
            "#f2e6ff", "#e6e6ff", "#ffe6f2", "#f2ffe6",
            "#e6fff2", "#ffe6e6", "#e6e6e6", "#fff9e6",
            "#e6f9ff", "#f9e6ff", "#ffe6f9", "#f9ffe6"
        ])
      
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
        params_map = {
            1: {"grid_size": 3},  # 3x3
            2: {"grid_size": 4},  # 4x4
            3: {"grid_size": 5},  # 5x5
            4: {"grid_size": 6},  # 6x6
            5: {"grid_size": 8},  # 8x8
        }
        return params_map.get(difficulty, {"grid_size": 3})
    
    def generate(self, num_cases, difficulty, output_folder=None):
        """
        生成问题的方法。
        
        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别
            output_folder: 输出文件夹路径，覆盖构造函数中设置的路径

        Returns:
            生成的问题列表
        """
        # 使用传入的output_folder或默认的
        current_output_folder = output_folder if output_folder is not None else self.output_folder
        
        # 获取难度参数
        difficulty_params = self._get_difficulty_params(difficulty)
        self.grid_size = difficulty_params["grid_size"]
        
        # 创建输出目录结构
        self.category_dir = current_output_folder
        os.makedirs(os.path.join(current_output_folder, "images"), exist_ok=True)
        
        puzzles = []
        annotations = []
      
        for i in range(1, num_cases + 1):
            # 生成谜题
            case = self._generate_single_puzzle(i, difficulty)
            puzzles.append(case)
            
            # 准备annotation数据
            annotation = {
                "index": case["index"],
                "category": case["category"],
                "question": case["question"],
                "question_language": case["question_language"],
                "difficulty": difficulty,
                "image": case["image"],
                "answer": case["answer"],
                "initial_state": case["initial_state"],
                "solution": case["solution"],
            }
            annotations.append(annotation)
  
        self.save_annotations(annotations, current_output_folder)
              
        return puzzles
      
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
  
    def _generate_single_puzzle(self, case_id, difficulty, output_folder=None):
        """生成单个Calcudoku谜题"""
        # 步骤1: 生成有效的数独解
        grid = self._generate_valid_sudoku()
      
        # 步骤2: 基于数独解划分连续区域
        regions = self._divide_into_valid_regions(grid)
      
        # 步骤3: 为每个区域生成运算条件
        puzzle_data = self._generate_region_operations(grid, regions)
      
        # 步骤4: 生成图像（修改为新的命名方式）
        puzzle_image_path, solution_image_path = self.visualize(
            {'grid': grid, 'puzzle_data': puzzle_data}, 
            case_id=case_id,
            difficulty=difficulty
        )
          
        # 步骤6: 生成数据点
        return self._create_datapoint(case_id, grid, puzzle_data, puzzle_image_path, difficulty)
  
    def _generate_valid_sudoku(self):
        """生成一个有效的数独解"""
        n = self.grid_size
      
        # 使用Fisher-Yates洗牌算法初始化第一行
        first_row = list(range(1, n+1))
        random.shuffle(first_row)
      
        # 创建空网格
        grid = [[0 for _ in range(n)] for _ in range(n)]
      
        # 填充第一行
        for j in range(n):
            grid[0][j] = first_row[j]
          
        # 使用回溯算法填充剩余单元格
        def backtrack(row, col):
            # 如果达到最后一行之后，说明填充完成
            if row == n:
                return True
          
            # 计算下一个单元格位置
            next_row = row + 1 if col == n-1 else row
            next_col = 0 if col == n-1 else col + 1
          
            # 如果当前单元格已有数字，继续下一个
            if grid[row][col] != 0:
                return backtrack(next_row, next_col)
          
            # 尝试填充当前单元格
            candidates = list(range(1, n+1))
            random.shuffle(candidates)  # 随机尝试，增加随机性
          
            for num in candidates:
                # 检查是否符合数独规则
                if self._is_valid_sudoku_position(grid, row, col, num):
                    grid[row][col] = num
                  
                    # 递归填充下一个单元格
                    if backtrack(next_row, next_col):
                        return True
                  
                    # 如果失败，回溯
                    grid[row][col] = 0
                  
            # 无法填充有效数字
            return False
      
        # 从第一行第一列开始回溯填充（第一行已填充，所以从第二行开始）
        if not backtrack(1, 0):
            # 这种情况理论上不会发生，因为数独总是有解的
            print("警告: 无法生成有效的数独解! 这不应该发生。")
            # 使用简单的Latin方阵作为后备方案
            for i in range(n):
                for j in range(n):
                    grid[i][j] = (i + j + 1) % n
                    if grid[i][j] == 0:
                        grid[i][j] = n
                      
        return grid
  
    def _is_valid_sudoku_position(self, grid, row, col, num):
        """检查在数独网格的指定位置放置数字是否有效"""
        n = self.grid_size
      
        # 检查同一行是否有重复
        for j in range(n):
            if grid[row][j] == num:
                return False
          
        # 检查同一列是否有重复
        for i in range(n):
            if grid[i][col] == num:
                return False
          
        return True
  
    def _divide_into_valid_regions(self, grid):
        """将数独网格划分为连续区域"""
        n = self.grid_size
      
        # 确定每个区域的大小范围
        min_region_size = 2
        max_region_size = min(4, n)  # 区域大小上限，不超过网格大小
      
        # 标记单元格是否已分配区域
        assigned = [[False for _ in range(n)] for _ in range(n)]
        regions = []
      
        # 返回给定单元格的未分配邻居
        def get_unassigned_neighbors(i, j):
            neighbors = []
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:  # 上下左右
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n and not assigned[ni][nj]:
                    neighbors.append((ni, nj))
            return neighbors
      
        # 随机排序的单元格列表
        cells = [(i, j) for i in range(n) for j in range(n)]
        random.shuffle(cells)
      
        # 开始划分区域
        for start_i, start_j in cells:
            if assigned[start_i][start_j]:
                continue
          
            # 创建新区域
            region = [(start_i, start_j)]
            assigned[start_i][start_j] = True
          
            # 随机决定这个区域的大小
            target_size = random.randint(min_region_size, max_region_size)
          
            # 扩展区域
            candidates = get_unassigned_neighbors(start_i, start_j)
            while candidates and len(region) < target_size:
                # 随机选择一个邻居
                i, j = random.choice(candidates)
                region.append((i, j))
                assigned[i][j] = True
              
                # 更新候选列表
                candidates = []
                for cell_i, cell_j in region:
                    candidates.extend(get_unassigned_neighbors(cell_i, cell_j))
                # 去重
                candidates = list(set(candidates))
              
            regions.append(region)
          
        # 处理剩余的单个单元格（确保每个单元格都属于某个区域）
        for i in range(n):
            for j in range(n):
                if not assigned[i][j]:
                    # 查找相邻区域
                    for region in regions:
                        for ri, rj in region:
                            if abs(ri - i) + abs(rj - j) == 1:  # 曼哈顿距离为1
                                region.append((i, j))
                                assigned[i][j] = True
                                break
                        if assigned[i][j]:
                            break
                      
                    # 如果仍未分配，创建新区域
                    if not assigned[i][j]:
                        regions.append([(i, j)])
                        assigned[i][j] = True
                      
        return regions
  
    def _generate_region_operations(self, grid, regions):
        """为每个区域生成合适的运算条件"""
        operations = []
      
        for region in regions:
            # 提取区域中的数字
            nums = [grid[i][j] for i, j in region]
          
            # 如果区域只有一个单元格，则直接使用其值作为结果
            if len(nums) == 1:
                operations.append({
                    'cells': [(i+1, j+1) for i, j in region],  # 转换为1-indexed
                    'operator': '+',
                    'target': nums[0]
                })
                continue
          
            # 尝试不同的运算符，并选择适合的
            candidates = []
          
            # 加法总是有效的
            candidates.append(('+', sum(nums)))
          
            # 乘法，避免结果过大
            product = reduce(lambda x, y: x * y, nums)
            if product < 100:  # 避免结果过大
                candidates.append(('*', product))  # 这里依然使用'*'作为内部存储，只在显示时转换为'×'
          
            # 减法（仅对2个单元格有效）
            if len(nums) == 2:
                candidates.append(('-', abs(nums[0] - nums[1])))
              
                # 除法（仅当整除时有效）
                a, b = nums
                if max(a, b) % min(a, b) == 0:
                    candidates.append(('÷', max(a, b) // min(a, b)))
                  
            # 随机选择一个候选运算
            op, target = random.choice(candidates)
          
            operations.append({
                'cells': [(i+1, j+1) for i, j in region],  # 转换为1-indexed
                'operator': op,
                'target': target
            })
          
        return {'size': self.grid_size, 'regions': operations}
  
    def visualize(self, puzzle, **kwargs):
        """生成谜题和解答的可视化图像"""
        case_id = kwargs.get('case_id', 0)
        difficulty = kwargs.get('difficulty', 1)
        grid = puzzle['grid']
        puzzle_data = puzzle['puzzle_data']
      
        # 生成谜题图像（修改文件名为question_name格式）
        puzzle_image_data = self._generate_puzzle_image(puzzle_data)
        puzzle_image_path = os.path.join(self.category_dir, f"images/puzzle_difficulty_{difficulty}_{case_id}.png")
        with open(puzzle_image_path, "wb") as f:
            f.write(base64.b64decode(puzzle_image_data))
          
        # 生成解答图像
        solution_image_data = self._generate_solution_image(grid, puzzle_data)
        solution_image_path = os.path.join(self.category_dir, f"images/solution_difficulty_{difficulty}_{case_id}.png")
        with open(solution_image_path, "wb") as f:
            f.write(base64.b64decode(solution_image_data))
          
        return puzzle_image_path, solution_image_path
  
    def _generate_puzzle_image(self, puzzle_data):
        """为计算数独谜题生成图像"""
        size = puzzle_data['size']
        regions = puzzle_data['regions']
      
        # 创建图像
        width, height = self.image_size
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)
      
        # 设置字体
        title_font_size = 36
        target_font_size = 24
        operator_font_size = 20
      
        try:
            if self.font_path:
                title_font = ImageFont.truetype(self.font_path, title_font_size)
                target_font = ImageFont.truetype(self.font_path, target_font_size)
                operator_font = ImageFont.truetype(self.font_path, operator_font_size)
            else:
                title_font = ImageFont.load_default()
                target_font = ImageFont.load_default()
                operator_font = ImageFont.load_default()
        except:
            title_font = ImageFont.load_default()
            target_font = ImageFont.load_default()
            operator_font = ImageFont.load_default()
          
        # 绘制标题
        title = "Calcudoku Puzzle"
        title_width = draw.textlength(title, font=title_font) if hasattr(draw, 'textlength') else title_font_size * len(title) * 0.6
        draw.text(((width - title_width) // 2, 40), title, fill=self.text_color, font=title_font)
      
        # 计算网格起始位置 - 确保居中
        grid_width = size * self.cell_size
        grid_height = size * self.cell_size
        start_x = (width - grid_width) // 2
        start_y = (height - grid_height) // 2  # 确保网格垂直居中
      
        # 创建区域映射
        region_map = {}
        for i, region in enumerate(regions):
            color_index = i % len(self.region_colors)
            color = self.region_colors[color_index]
          
            for cell in region['cells']:
                r, c = cell
                region_map[(r-1, c-1)] = {
                    'color': color,
                    'target': region['target'],
                    'operator': region['operator']
                }
              
        # 绘制单元格和区域
        for i in range(size):
            for j in range(size):
                x = start_x + j * self.cell_size
                y = start_y + i * self.cell_size
              
                # 绘制单元格背景
                region_info = region_map.get((i, j))
                if region_info:
                    color = region_info['color']
                else:
                    color = "#ffffff"  # 默认白色
                  
                draw.rectangle([x, y, x + self.cell_size, y + self.cell_size], 
                               fill=color, outline=self.grid_color, width=1)
              
                # 每个区域第一个单元格绘制目标值和运算符
                if region_info and (i, j) == self._get_top_left_cell(region_map, i, j):
                    target = region_info['target']
                    operator = region_info['operator']
                  
                    # 将*替换为×符号
                    if operator == '*':
                        operator = '×'
                      
                    # 绘制目标值
                    target_text = str(target)
                    target_x = x + 5
                    target_y = y + 5
                    draw.text((target_x, target_y), target_text, fill=self.text_color, font=target_font)
                  
                    # 绘制运算符
                    op_x = x + target_font_size + 10
                    op_y = y + 5
                    draw.text((op_x, op_y), operator, fill=self.text_color, font=operator_font)
                  
        # 绘制粗网格线
        for i in range(size + 1):
            line_width = 3 if i == 0 or i == size else 1
          
            # 绘制水平线
            draw.line([(start_x, start_y + i * self.cell_size), 
                      (start_x + grid_width, start_y + i * self.cell_size)], 
                     fill=self.grid_color, width=line_width)
          
            # 绘制垂直线
            draw.line([(start_x + i * self.cell_size, start_y), 
                      (start_x + i * self.cell_size, start_y + grid_height)], 
                     fill=self.grid_color, width=line_width)
          
        # 绘制区域边界线
        for i in range(size):
            for j in range(size):
                x = start_x + j * self.cell_size
                y = start_y + i * self.cell_size
              
                # 检查右边界
                if j < size - 1:
                    if region_map.get((i, j)) != region_map.get((i, j+1)):
                        draw.line([(x + self.cell_size, y), (x + self.cell_size, y + self.cell_size)], 
                                 fill=self.grid_color, width=3)
                      
                # 检查下边界
                if i < size - 1:
                    if region_map.get((i, j)) != region_map.get((i+1, j)):
                        draw.line([(x, y + self.cell_size), (x + self.cell_size, y + self.cell_size)], 
                                 fill=self.grid_color, width=3)
                      
        # 转换为base64编码的图像
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = base64.b64encode(buffer.getvalue()).decode()
      
        return image_data
  
    def _generate_solution_image(self, solution_grid, puzzle_data):
        """为计算数独谜题的解答生成图像"""
        size = puzzle_data['size']
        regions = puzzle_data['regions']
      
        # 创建图像
        width, height = self.image_size
        img = Image.new('RGB', (width, height), self.bg_color)
        draw = ImageDraw.Draw(img)
      
        # 设置字体
        title_font_size = 36
        target_font_size = 24
        operator_font_size = 20
        number_font_size = 32
      
        try:
            if self.font_path:
                title_font = ImageFont.truetype(self.font_path, title_font_size)
                target_font = ImageFont.truetype(self.font_path, target_font_size)
                operator_font = ImageFont.truetype(self.font_path, operator_font_size)
                number_font = ImageFont.truetype(self.font_path, number_font_size)
            else:
                title_font = ImageFont.load_default()
                target_font = ImageFont.load_default()
                operator_font = ImageFont.load_default()
                number_font = ImageFont.load_default()
        except:
            title_font = ImageFont.load_default()
            target_font = ImageFont.load_default()
            operator_font = ImageFont.load_default()
            number_font = ImageFont.load_default()
          
        # 绘制标题
        title = "Calcudoku Solution"
        title_width = draw.textlength(title, font=title_font) if hasattr(draw, 'textlength') else title_font_size * len(title) * 0.6
        draw.text(((width - title_width) // 2, 40), title, fill=self.text_color, font=title_font)
      
        # 计算网格起始位置 - 确保居中
        grid_width = size * self.cell_size
        grid_height = size * self.cell_size
        start_x = (width - grid_width) // 2
        start_y = (height - grid_height) // 2  # 确保网格垂直居中
      
        # 创建区域映射
        region_map = {}
        for i, region in enumerate(regions):
            color_index = i % len(self.region_colors)
            color = self.region_colors[color_index]
          
            for cell in region['cells']:
                r, c = cell
                region_map[(r-1, c-1)] = {
                    'color': color,
                    'target': region['target'],
                    'operator': region['operator'],
                    'region_id': i
                }
              
        # 验证每个区域的计算是否正确
        region_valid = {}
        for i, region in enumerate(regions):
            cells = [(r-1, c-1) for r, c in region['cells']]
            values = [solution_grid[r][c] for r, c in cells]
          
            # 检查运算
            op = region['operator']
            target = region['target']
          
            if op == '+':
                valid = sum(values) == target
            elif op == '*':
                valid = reduce(lambda x, y: x*y, values) == target
            elif op == '-':
                valid = abs(values[0] - values[1]) == target
            elif op == '÷':
                valid = max(values) / min(values) == target
            else:
                valid = False
              
            region_valid[i] = valid
          
        # 绘制单元格、区域和解答
        for i in range(size):
            for j in range(size):
                x = start_x + j * self.cell_size
                y = start_y + i * self.cell_size
              
                # 绘制单元格背景
                region_info = region_map.get((i, j))
                if region_info:
                    color = region_info['color']
                    region_id = region_info['region_id']
                    # 如果区域验证失败，添加淡红色覆盖
                    if not region_valid.get(region_id, True):
                        color = self._blend_colors(color, "#ffcccc", 0.5)
                else:
                    color = "#ffffff"  # 默认白色
                  
                draw.rectangle([x, y, x + self.cell_size, y + self.cell_size], 
                               fill=color, outline=self.grid_color, width=1)
              
                # 每个区域第一个单元格绘制目标值和运算符
                if region_info and (i, j) == self._get_top_left_cell(region_map, i, j):
                    target = region_info['target']
                    operator = region_info['operator']
                  
                    # 将*替换为×符号
                    if operator == '*':
                        operator = '×'
                      
                    # 绘制目标值
                    target_text = str(target)
                    target_x = x + 5
                    target_y = y + 5
                    draw.text((target_x, target_y), target_text, fill=self.text_color, font=target_font)
                  
                    # 绘制运算符
                    op_x = x + target_font_size + 10
                    op_y = y + 5
                    draw.text((op_x, op_y), operator, fill=self.text_color, font=operator_font)
                  
                # 绘制数字
                number = solution_grid[i][j]
                number_text = str(number)
              
                number_width = draw.textlength(number_text, font=number_font) if hasattr(draw, 'textlength') else number_font_size * 0.6
                number_x = x + (self.cell_size - number_width) // 2
                number_y = y + (self.cell_size - number_font_size) // 2 + 10
              
                draw.text((number_x, number_y), number_text, fill=self.text_color, font=number_font)
              
        # 绘制粗网格线
        for i in range(size + 1):
            line_width = 3 if i == 0 or i == size else 1
          
            # 绘制水平线
            draw.line([(start_x, start_y + i * self.cell_size), 
                      (start_x + grid_width, start_y + i * self.cell_size)], 
                     fill=self.grid_color, width=line_width)
          
            # 绘制垂直线
            draw.line([(start_x + i * self.cell_size, start_y), 
                      (start_x + i * self.cell_size, start_y + grid_height)], 
                     fill=self.grid_color, width=line_width)
          
        # 绘制区域边界线
        for i in range(size):
            for j in range(size):
                x = start_x + j * self.cell_size
                y = start_y + i * self.cell_size
              
                # 检查右边界
                if j < size - 1:
                    if region_map.get((i, j)) != region_map.get((i, j+1)):
                        draw.line([(x + self.cell_size, y), (x + self.cell_size, y + self.cell_size)], 
                                 fill=self.grid_color, width=3)
                      
                # 检查下边界
                if i < size - 1:
                    if region_map.get((i, j)) != region_map.get((i+1, j)):
                        draw.line([(x, y + self.cell_size), (x + self.cell_size, y + self.cell_size)], 
                                 fill=self.grid_color, width=3)
                      
        # 转换为base64编码的图像
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_data = base64.b64encode(buffer.getvalue()).decode()
      
        return image_data
  
    def _get_top_left_cell(self, region_map, row, col):
        """获取区域中的左上角单元格"""
        current_region = region_map.get((row, col))
        if not current_region:
            return (row, col)
      
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if region_map.get((i, j)) == current_region:
                    if i < row or (i == row and j < col):
                        return self._get_top_left_cell(region_map, i, j)
                  
        return (row, col)
  
    def _blend_colors(self, color1, color2, ratio=0.5):
        """混合两种颜色"""
        r1 = int(color1[1:3], 16)
        g1 = int(color1[3:5], 16)
        b1 = int(color1[5:7], 16)
      
        r2 = int(color2[1:3], 16)
        g2 = int(color2[3:5], 16)
        b2 = int(color2[5:7], 16)
      
        r = int(r1 * (1 - ratio) + r2 * ratio)
        g = int(g1 * (1 - ratio) + g2 * ratio)
        b = int(b1 * (1 - ratio) + b2 * ratio)
      
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _calculate_step_complexity(self, grid, puzzle_data, difficulty):
        """计算解题步骤的复杂程度（返回整数）"""
        # 基础复杂度：基于网格大小
        size = puzzle_data['size']
        base_complexity = size ** 2
        
        # 区域因子：基于区域数量和特性
        regions = puzzle_data['regions']
        region_count = len(regions)
        
        # 运算复杂度因子
        operation_complexity = 0
        for region in regions:
            op = region['operator']
            cells_count = len(region['cells'])
            
            # 根据运算类型和区域大小调整复杂度
            if op == '+':
                op_factor = 1
            elif op == '*':
                op_factor = 2
            elif op == '-':
                op_factor = 2
            elif op == '÷':
                op_factor = 3
            else:
                op_factor = 1
                
            # 区域大小影响：大区域需要考虑更多组合
            size_factor = cells_count
            
            # 累加此区域的复杂度
            operation_complexity += op_factor * size_factor
            
        # 交互复杂度：评估区域间的关联
        # 简化版本：基于网格大小和区域数量
        interaction_complexity = int(size * (region_count / size))
        
        # 最终步骤复杂度计算（确保为整数）
        step_complexity = int(base_complexity * 0.3 + operation_complexity * 0.5 + interaction_complexity * 0.2)
        
        # 根据难度级别进行最终调整
        # 确保不同级别的难度有明显区分
        level_factor = difficulty * 1.2
        step_complexity = int(step_complexity * level_factor)
        
        return step_complexity
  
    def _create_datapoint(self, case_id, grid, puzzle_data, puzzle_image_path, difficulty):
        """创建数据点"""
        # 转换网格为字符串形式，用于答案
        grid_str = "[[" + ", ".join([" ".join(map(str, row)) for row in grid]) + "]]"
        
        # 计算解题步骤复杂度（整数值）
        step_complexity = self._calculate_step_complexity(grid, puzzle_data, difficulty)
      
        return {
            "index": f"Calcudoku_difficulty_{difficulty}_{case_id}",
            "category": "Calcudoku",
            "question": self._generate_question_text(puzzle_data),
            "image": f"images/puzzle_difficulty_{difficulty}_{case_id}.png",  # 修改为相对路径，符合新的文件结构
            "question_language": self._generate_detailed_question_text(puzzle_data),
            "answer": grid,
            "initial_state": puzzle_data,
            "difficulty": difficulty,
            "grid": grid,
            "solution": grid,
        }
  
    def _generate_question_text(self, puzzle_data):
        """生成带图像的问题描述"""
        size = puzzle_data['size']
      
        return (
            f"This is a {size}x{size} Calcudoku puzzle. Each row and column must contain the numbers 1 to {size} exactly once.\n"
            f"The grid is divided into regions, each with a target number and a specified operation.\n"
            f"The numbers within each region must be combined using the given operation to achieve the target number.\n"
            f"Please solve the puzzle and provide the solution as a two-dimensional array.\n"
            f"Example answer format: [[1, 2, 3, 4], [4, 3, 2, 1], [2, 1, 4, 3], [3, 4, 1, 2]]."
        )
  
    def _generate_detailed_question_text(self, puzzle_data):
        """生成纯文本问题描述"""
        size = puzzle_data['size']
        regions = puzzle_data['regions']
      
        text_lines = [
            f"This is a {size}x{size} Calcudoku puzzle. Each row and column must contain the numbers 1 to {size} exactly once. "
            f"The grid is divided into regions, each with a target number and a specified operation.\n"
            f"The numbers within each region must be combined using the given operation to achieve the target number.\n"
        ]
      
        for region in regions:
            coords = ', '.join(f"({r},{c})" for (r, c) in region['cells'])
            operator = region['operator']
            if operator == '*':
                operator = '×'
            text_lines.append(f"Region at {coords}: {region['target']}{operator}")
          
        text_lines.append("Please solve the puzzle and provide the solution as a two-dimensional array.\n")
        text_lines.append("Answer with a two-dimensional list.\nExample answer format: [[1, 2, 3, 4], [4, 3, 2, 1], [2, 1, 4, 3], [3, 4, 1, 2]].")
      
        return '\n'.join(text_lines)
  
    def solve(self, puzzle, **kwargs):
        """求解计算数独谜题 - 在这个情况下返回提供的解答"""
        return puzzle['grid']


if __name__ == "__main__":
    # 定义不同难度级别的参数
    difficulties = [1, 2, 3, 4, 5]
    case_num = 6
    
    # 为每个难度级别生成测试用例
    for difficulty in difficulties:
        print(f"正在生成 Calcudoku 难度级别 {difficulty} 的测试用例...")
        
        # 创建生成器
        output_folder = "newtasks"
        generator_calcudoku = CalcudokuGenerator(output_folder)
        
        # 生成测试用例
        puzzles = generator_calcudoku.generate(case_num - 1, difficulty)  # case_num-1 因为原代码是range(1, case_num)
        
        print(f"Calcudoku 难度级别 {difficulty} 的测试用例生成完成！")

    print("所有难度级别的测试用例生成完成！")