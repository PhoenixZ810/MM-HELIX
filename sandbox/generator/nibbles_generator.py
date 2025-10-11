import numpy as np
import matplotlib
# 设置matplotlib使用非交互式后端，解决多线程GUI问题
matplotlib.use('Agg')  # 必须在pyplot导入之前设置
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import json
import os
import random
import time
import sys
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import uuid
import shutil
import heapq
from matplotlib.patches import Rectangle
from heapq import heappush, heappop
import concurrent.futures
import threading
import fcntl
from abc import ABC, abstractmethod
from generator.base_generator import BaseGenerator
from utils.constants import PROMPT_MAZE_IMAGE, PROMPT_15PUZZLE_IMAGE, PROMPT_HANOI_IMAGE, PROMPT_WORDSEARCH_IMAGE, PROMPT_NUMBRIX_IMAGE, PROMPT_MINESWEEPER_IMAGE, PROMPT_EULERO_IMAGE, PROMPT_SNAKE_IMAGE
from utils.constants import PROMPT_MAZE, PROMPT_15PUZZLE, PROMPT_HANOI, PROMPT_WORDSEARCH, PROMPT_NUMBRIX, PROMPT_MINESWEEPER, PROMPT_EULERO, PROMPT_SNAKE
class NibblesGenerator(BaseGenerator):
    def __init__(self, output_folder):
        super().__init__(output_folder)
        # 初始化通用状态并创建输出目录结构
        self.seed = int(time.time())
        os.makedirs(self.output_folder, exist_ok=True)
        self.images_dir = os.path.join(self.output_folder, 'images')
        os.makedirs(self.images_dir, exist_ok=True)

    def _get_difficulty_params(self, difficulty):
        """
        根据难度级别获取Snake puzzle的参数配置。

        Args:
            difficulty: 难度级别（1-5）

        Returns:
            dict: 包含难度参数的字典
        """
        # Map difficulty to size and parameters
        difficulty_params = {
            1: {'size': 6, 'apples': 1, 'initial_length': 2, 'difficulty': '1'},
            2: {'size': 7, 'apples': 2, 'initial_length': 2, 'difficulty': '2'},
            3: {'size': 8, 'apples': 3, 'initial_length': 2, 'difficulty': '3'},
            4: {'size': 9, 'apples': 4, 'initial_length': 2, 'difficulty': '4'},
            5: {'size': 10, 'apples': 5, 'initial_length': 2, 'difficulty': '5'},
        }

        if difficulty in difficulty_params:
            return difficulty_params[difficulty]
        else:
            # Default parameters for unknown difficulty
            return {'size': 8, 'apples': 3, 'initial_length': 2, 'difficulty': '3'}

    def generate(self, num_cases, difficulty, output_folder=None):
        """
        生成Snake puzzles

        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别（1-5）
            output_folder: 输出文件夹路径，覆盖构造函数中设置的路径

        Returns:
            生成的问题列表
        """
        if output_folder is not None:
            self.output_folder = output_folder
            os.makedirs(self.output_folder, exist_ok=True)
            self.images_dir = os.path.join(self.output_folder, 'images')
            os.makedirs(self.images_dir, exist_ok=True)

        # Get parameters for this difficulty
        params = self._get_difficulty_params(difficulty)
        size = params['size']

        generated_count = 0
        max_attempts = 10000
        temp_puzzles = []  # 临时存储生成的puzzles（作为annotations）

        print(f"Generating {num_cases} Snake puzzles with size {size}x{size}, difficulty {difficulty}")

        for case_idx in range(num_cases):
            for attempt in range(max_attempts):
                try:
                    # Use timestamp-based seed with case index for variety
                    current_seed = self.seed + case_idx * 1000 + attempt
                    random.seed(current_seed)
                    np.random.seed(current_seed)

                    puzzle = self._generate_single_puzzle(size, size, params)
                    if puzzle is None:
                        continue

                    # Generate puzzle ID based on size and seed
                    puzzle_id = f"nibbles_{size}_{current_seed}_{case_idx}"
                    puzzle['id'] = puzzle_id
                    puzzle['difficulty'] = params['difficulty']
                    puzzle['step_count'] = len(puzzle['answer'])

                    # Create visualization
                    img_filename = f"nibbles_{size}_{current_seed}_{case_idx}.png"
                    img_path = os.path.join(self.images_dir, img_filename)
                    self.visualize(puzzle, filename=img_path)

                    # Format initial state
                    initial_state = self._format_puzzle_question(puzzle)

                    # Generate COT reasoning
                    cot_result = self.generate_cot(puzzle)

                    # Create the formatted puzzle data
                    formatted_puzzle = {
                        "index": puzzle_id,
                        "category": "nibbles",
                        "image": f"images/{img_filename}",
                        "question": PROMPT_SNAKE_IMAGE,
                        "question_language": self._format_question_language(initial_state),
                        "answer": self._format_actions(puzzle['answer']),
                        "initial_state": initial_state,
                        "difficulty": params['difficulty'],
                        "cot": cot_result['cot'],
                        "cot_step1_all": cot_result['cot_steps']['cot_step1_all'],
                        "cot_step2_all": cot_result['cot_steps']['cot_step2_all'],
                        "cot_step3_all": cot_result['cot_steps']['cot_step3_all']
                    }

                    # 收集到临时列表，稍后统一保存
                    temp_puzzles.append(formatted_puzzle)

                    print(f"Successfully generated Snake puzzle {puzzle_id} with {len(puzzle['answer'])} steps")
                    generated_count += 1
                    break

                except Exception as e:
                    print(f"Attempt {attempt + 1} for case {case_idx + 1} failed: {e}")
                    continue

            if generated_count >= num_cases:
                break

        # 所有puzzle生成完成后，保存到annotations.json（追加去重）
        if temp_puzzles:
            self.save_annotations(temp_puzzles, self.output_folder)

        print(f"Generated {generated_count} Snake puzzles successfully")
        return temp_puzzles

    def _generate_single_puzzle(self, rows, cols, params):
        """Generate a single traditional Snake puzzle"""
        apples_count = params['apples']
        initial_length = params['initial_length']
        
        max_attempts = 10000
        for attempt in range(max_attempts):
            # 随机生成初始蛇的位置和方向
            snake, direction = self._generate_initial_snake(rows, cols, initial_length)
            if snake is None:
                continue
                
            # 随机放置苹果
            apples = self._place_apples(rows, cols, snake, apples_count)
            if len(apples) < apples_count:
                continue
                
            # 尝试找到解决方案
            solution = self._find_solution(rows, cols, snake, direction, apples)
            if solution:
                # 创建puzzle字典
                puzzle = {
                    'grid_size': {'rows': rows, 'cols': cols},
                    'initial_snake': snake,
                    'initial_direction': direction,
                    'apples': apples,
                    'apples_count': apples_count,
                    'answer': solution
                }
                return puzzle
        
        return None
    
    def _generate_initial_snake(self, rows, cols, length):
        """生成初始蛇的位置，确保不会立即撞墙"""
        directions = ['up', 'down', 'left', 'right']
        
        for _ in range(5000):  # 最多尝试5000次
            direction = random.choice(directions)
            
            # 根据方向确定蛇头的可能位置范围，并确保身体在“头的后方”与朝向一致
            if direction == 'up':
                # 头朝上，身体在头的下方（行号增大方向）
                head_row = random.randint(0, rows - length)
                head_col = random.randint(0, cols - 1)
                snake = [(head_row + i, head_col) for i in range(length)]
            elif direction == 'down':
                # 头朝下，身体在头的上方（行号减小方向）
                head_row = random.randint(length - 1, rows - 1)
                head_col = random.randint(0, cols - 1)
                snake = [(head_row - i, head_col) for i in range(length)]
            elif direction == 'left':
                # 头朝左，身体在头的右侧（列号增大方向）
                head_row = random.randint(0, rows - 1)
                head_col = random.randint(0, cols - length)
                snake = [(head_row, head_col + i) for i in range(length)]
            else:  # right
                # 头朝右，身体在头的左侧（列号减小方向）
                head_row = random.randint(0, rows - 1)
                head_col = random.randint(length - 1, cols - 1)
                snake = [(head_row, head_col - i) for i in range(length)]
            
            # 验证所有位置都在边界内
            if all(0 <= r < rows and 0 <= c < cols for r, c in snake):
                return snake, direction
        
        return None, None
    
    def _place_apples(self, rows, cols, snake, count):
        """随机放置苹果，确保不与蛇重叠"""
        snake_positions = set(snake)
        available_positions = [(r, c) for r in range(rows) for c in range(cols) 
                             if (r, c) not in snake_positions]
        
        if len(available_positions) < count:
            return []
        
        return random.sample(available_positions, count)
    
    def _find_solution(self, rows, cols, initial_snake, initial_direction, apples):
        """使用BFS寻找解决方案"""
        from collections import deque
        
        # 状态：(snake, direction, remaining_apples, actions)
        initial_state = (tuple(initial_snake), initial_direction, tuple(sorted(apples)), [])
        queue = deque([initial_state])
        visited = set()
        visited.add((tuple(initial_snake), initial_direction, tuple(sorted(apples))))
        
        directions = ['up', 'down', 'left', 'right']
        direction_deltas = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        
        max_steps = 10000  # 限制最大步数防止无限搜索
        
        while queue and len(queue[0][3]) < max_steps:
            snake, current_dir, remaining_apples, actions = queue.popleft()
            
            # 如果所有苹果都吃完了，找到解决方案
            if not remaining_apples:
                return actions
            
            # 尝试所有可能的方向
            for new_direction in directions:
                # 不能直接反向
                if self._is_opposite_direction(current_dir, new_direction):
                    continue
                
                # 计算新的头部位置
                head_r, head_c = snake[0]
                dr, dc = direction_deltas[new_direction]
                new_head = (head_r + dr, head_c + dc)
                
                # 检查是否撞墙
                if not (0 <= new_head[0] < rows and 0 <= new_head[1] < cols):
                    continue
                
                # 检查是否撞到身体
                if new_head in snake:
                    continue
                
                # 计算新的蛇身
                new_snake = list(snake)
                new_snake.insert(0, new_head)
                
                # 检查是否吃到苹果
                ate_apple = new_head in remaining_apples
                if ate_apple:
                    # 吃到苹果，蛇变长，移除这个苹果
                    new_remaining_apples = tuple(a for a in remaining_apples if a != new_head)
                else:
                    # 没吃到苹果，去掉尾巴
                    new_snake.pop()
                    new_remaining_apples = remaining_apples
                
                # 创建新状态
                new_state_key = (tuple(new_snake), new_direction, new_remaining_apples)
                if new_state_key not in visited:
                    visited.add(new_state_key)
                    new_actions = actions + [new_direction]
                    queue.append((tuple(new_snake), new_direction, new_remaining_apples, new_actions))
        
        return None  # 没有找到解决方案
    
    def _is_opposite_direction(self, dir1, dir2):
        """检查两个方向是否相反"""
        opposites = {
            'up': 'down',
            'down': 'up',
            'left': 'right',
            'right': 'left'
        }
        return opposites.get(dir1) == dir2

    def _draw_continuous_snake(self, ax, snake, direction):
        """Draw snake as a continuous body with clear head direction"""
        from matplotlib.patches import Circle, Polygon
        import numpy as np
        
        if not snake:
            return
            
        # Define snake body width
        snake_width = 0.6
        
        # Calculate path points (centers of each grid cell)
        path_points = [(c + 0.5, r + 0.5) for r, c in snake]
        
        if len(snake) == 1:
            # Single segment snake - just draw head
            self._draw_snake_head(ax, snake[0], direction, snake)
            return
            
        # Draw snake body as continuous segments
        for i in range(len(path_points) - 1):
            start_point = path_points[i]
            end_point = path_points[i + 1]
            
            # Calculate perpendicular direction for body width
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            
            # Normalize and get perpendicular
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx /= length
                dy /= length
            
            # Perpendicular vector
            perp_x = -dy * snake_width / 2
            perp_y = dx * snake_width / 2
            
            # Create rectangle for this segment
            segment_points = [
                (start_point[0] + perp_x, start_point[1] + perp_y),
                (start_point[0] - perp_x, start_point[1] - perp_y),
                (end_point[0] - perp_x, end_point[1] - perp_y),
                (end_point[0] + perp_x, end_point[1] + perp_y)
            ]
            
            # Color intensity decreases toward tail
            intensity = 1.0 - (i / len(snake)) * 0.3
            body_color = (0.1, 0.8 * intensity, 0.25 * intensity)
            
            # Draw body segment
            segment = Polygon(segment_points, facecolor=body_color, 
                            edgecolor='#39ff14', linewidth=2.2, alpha=0.98)
            # Neon glow stroke for clarity
            segment.set_path_effects([
                pe.Stroke(linewidth=5.5, foreground='#1aff80'),
                pe.Normal()
            ])
            ax.add_patch(segment)
        
        # Draw circles at joints to make it smoother
        for i, (x, y) in enumerate(path_points):
            if i == 0:
                continue  # Skip head, we'll draw it separately
                
            intensity = 1.0 - (i / len(snake)) * 0.3
            body_color = (0.1, 0.8 * intensity, 0.25 * intensity)
            
            joint_circle = Circle((x, y), snake_width/2, 
                                facecolor=body_color, 
                                edgecolor='#39ff14', 
                                linewidth=2.2, alpha=0.98)
            joint_circle.set_path_effects([
                pe.Stroke(linewidth=5.0, foreground='#1aff80'),
                pe.Normal()
            ])
            ax.add_patch(joint_circle)
        
        # Draw snake head last (on top)
        self._draw_snake_head(ax, snake[0], direction, snake)
    
    def _draw_snake_head(self, ax, head_pos, direction, snake=None):
        """Draw snake head with clear directional indication based on body direction"""
        from matplotlib.patches import Circle, Polygon
        import numpy as np
        
        r, c = head_pos
        center_x, center_y = c + 0.5, r + 0.5
        
        # Head is larger and more prominent
        head_radius = 0.35
        
        # Draw main head circle
        head_circle = Circle((center_x, center_y), head_radius, 
                           facecolor='#39ff14', 
                           edgecolor='#eafff0', 
                           linewidth=3.2, alpha=1.0)
        head_circle.set_path_effects([
            pe.Stroke(linewidth=6.5, foreground='#1aff80'),
            pe.Normal()
        ])
        ax.add_patch(head_circle)
        
        # Add inner glow
        inner_circle = Circle((center_x, center_y), head_radius * 0.6, 
                            color='#7fff00', alpha=0.6)
        ax.add_patch(inner_circle)
        
        # Determine actual head direction based on snake body arrangement
        actual_direction = direction
        if snake and len(snake) > 1:
            head_r, head_c = snake[0]
            body_r, body_c = snake[1]
            
            # Head points away from body (背对身体方向)
            if head_r < body_r:
                actual_direction = 'up'    # 头在身体上方，朝向上方（远离身体）
            elif head_r > body_r:
                actual_direction = 'down'  # 头在身体下方，朝向下方（远离身体）
            elif head_c < body_c:
                actual_direction = 'left'  # 头在身体左边，朝向左边（远离身体）
            elif head_c > body_c:
                actual_direction = 'right' # 头在身体右边，朝向右边（远离身体）
        
        # Draw directional nose/snout
        nose_length = 0.15
        nose_width = 0.1
        
        if actual_direction == 'up':
            nose_tip = (center_x, center_y - head_radius - nose_length)
            nose_points = [
                (center_x - nose_width, center_y - head_radius),
                (center_x + nose_width, center_y - head_radius),
                nose_tip
            ]
            eye1_pos = (center_x - 0.15, center_y - 0.1)
            eye2_pos = (center_x + 0.15, center_y - 0.1)
        elif actual_direction == 'down':
            nose_tip = (center_x, center_y + head_radius + nose_length)
            nose_points = [
                (center_x - nose_width, center_y + head_radius),
                (center_x + nose_width, center_y + head_radius),
                nose_tip
            ]
            eye1_pos = (center_x - 0.15, center_y + 0.1)
            eye2_pos = (center_x + 0.15, center_y + 0.1)
        elif actual_direction == 'left':
            nose_tip = (center_x - head_radius - nose_length, center_y)
            nose_points = [
                (center_x - head_radius, center_y - nose_width),
                (center_x - head_radius, center_y + nose_width),
                nose_tip
            ]
            eye1_pos = (center_x - 0.1, center_y - 0.15)
            eye2_pos = (center_x - 0.1, center_y + 0.15)
        else:  # right
            nose_tip = (center_x + head_radius + nose_length, center_y)
            nose_points = [
                (center_x + head_radius, center_y - nose_width),
                (center_x + head_radius, center_y + nose_width),
                nose_tip
            ]
            eye1_pos = (center_x + 0.1, center_y - 0.15)
            eye2_pos = (center_x + 0.1, center_y + 0.15)
        
        # Draw nose/snout
        nose = Polygon(nose_points, facecolor='#2eb82e', 
                      edgecolor='#ffffff', linewidth=2.2, alpha=1.0)
        nose.set_path_effects([
            pe.Stroke(linewidth=4.5, foreground='#1aff80'),
            pe.Normal()
        ])
        ax.add_patch(nose)
        
        # Draw eyes
        eye1 = Circle(eye1_pos, 0.055, color='black')
        eye2 = Circle(eye2_pos, 0.055, color='black')
        ax.add_patch(eye1)
        ax.add_patch(eye2)

    def _draw_continuous_victory_snake(self, ax, snake):
        """Draw victory snake as continuous body with celebration effects"""
        from matplotlib.patches import Circle, Polygon
        import numpy as np
        
        if not snake:
            return
            
        # Victory snake is slightly wider and brighter
        snake_width = 0.65
        
        # Calculate path points
        path_points = [(c + 0.5, r + 0.5) for r, c in snake]
        
        if len(snake) == 1:
            # Single segment victory snake
            self._draw_victory_snake_head(ax, snake[0], snake)
            return
            
        # Draw victory snake body segments
        for i in range(len(path_points) - 1):
            start_point = path_points[i]
            end_point = path_points[i + 1]
            
            # Calculate perpendicular direction for body width
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            
            # Normalize and get perpendicular
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx /= length
                dy /= length
            
            # Perpendicular vector
            perp_x = -dy * snake_width / 2
            perp_y = dx * snake_width / 2
            
            # Create rectangle for this segment
            segment_points = [
                (start_point[0] + perp_x, start_point[1] + perp_y),
                (start_point[0] - perp_x, start_point[1] - perp_y),
                (end_point[0] - perp_x, end_point[1] - perp_y),
                (end_point[0] + perp_x, end_point[1] + perp_y)
            ]
            
            # Victory colors - brighter and less fade
            intensity = 1.0 - (i / len(snake)) * 0.2
            body_color = (0.1, 0.95 * intensity, 0.4 * intensity)
            
            # Draw body segment with victory glow
            segment = Polygon(segment_points, facecolor=body_color, 
                            edgecolor='#39ff14', linewidth=3.2, alpha=1.0)
            segment.set_path_effects([
                pe.Stroke(linewidth=6.5, foreground='#66ff99'),
                pe.Normal()
            ])
            ax.add_patch(segment)
        
        # Draw circles at joints with victory glow
        for i, (x, y) in enumerate(path_points):
            if i == 0:
                continue  # Skip head
                
            intensity = 1.0 - (i / len(snake)) * 0.2
            body_color = (0.1, 0.95 * intensity, 0.4 * intensity)
            
            joint_circle = Circle((x, y), snake_width/2, 
                                facecolor=body_color, 
                                edgecolor='#39ff14', 
                                linewidth=3.0, alpha=1.0)
            joint_circle.set_path_effects([
                pe.Stroke(linewidth=6.0, foreground='#66ff99'),
                pe.Normal()
            ])
            ax.add_patch(joint_circle)
            
            # Add victory sparkle effect
            sparkle_circle = Circle((x, y), snake_width/2 * 0.6, 
                                  color='#7fff00', alpha=0.7)
            ax.add_patch(sparkle_circle)
        
        # Draw victory snake head last
        self._draw_victory_snake_head(ax, snake[0], snake)
    
    def _draw_victory_snake_head(self, ax, head_pos, snake=None):
        """Draw victory snake head with celebration expression and correct direction"""
        from matplotlib.patches import Circle, Polygon
        import numpy as np
        
        r, c = head_pos
        center_x, center_y = c + 0.5, r + 0.5
        
        # Victory head is larger and more radiant
        head_radius = 0.4
        
        # Draw main head circle with victory glow
        head_circle = Circle((center_x, center_y), head_radius, 
                           facecolor='#39ff14', 
                           edgecolor='#ffffff', 
                           linewidth=4.2, alpha=1.0)
        head_circle.set_path_effects([
            pe.Stroke(linewidth=7.2, foreground='#66ff99'),
            pe.Normal()
        ])
        ax.add_patch(head_circle)
        
        # Add bright inner victory glow
        inner_circle = Circle((center_x, center_y), head_radius * 0.7, 
                            color='#7fff00', alpha=0.8)
        ax.add_patch(inner_circle)
        
        # Add outer celebration glow
        outer_glow = Circle((center_x, center_y), head_radius * 1.2, 
                          color='#ccff00', alpha=0.3)
        ax.add_patch(outer_glow)
        
        # Determine head direction based on body position
        actual_direction = 'up'  # default
        if snake and len(snake) > 1:
            head_r, head_c = snake[0]
            body_r, body_c = snake[1]
            
            # Head points away from body
            if head_r < body_r:
                actual_direction = 'up'    # 头在身体上方，朝向上方
            elif head_r > body_r:
                actual_direction = 'down'  # 头在身体下方，朝向下方
            elif head_c < body_c:
                actual_direction = 'left'  # 头在身体左边，朝向左边
            elif head_c > body_c:
                actual_direction = 'right' # 头在身体右边，朝向右边
        
        # Draw directional nose for victory head
        nose_length = 0.12
        nose_width = 0.08
        
        if actual_direction == 'up':
            nose_tip = (center_x, center_y - head_radius - nose_length)
            nose_points = [
                (center_x - nose_width, center_y - head_radius),
                (center_x + nose_width, center_y - head_radius),
                nose_tip
            ]
        elif actual_direction == 'down':
            nose_tip = (center_x, center_y + head_radius + nose_length)
            nose_points = [
                (center_x - nose_width, center_y + head_radius),
                (center_x + nose_width, center_y + head_radius),
                nose_tip
            ]
        elif actual_direction == 'left':
            nose_tip = (center_x - head_radius - nose_length, center_y)
            nose_points = [
                (center_x - head_radius, center_y - nose_width),
                (center_x - head_radius, center_y + nose_width),
                nose_tip
            ]
        else:  # right
            nose_tip = (center_x + head_radius + nose_length, center_y)
            nose_points = [
                (center_x + head_radius, center_y - nose_width),
                (center_x + head_radius, center_y + nose_width),
                nose_tip
            ]
        
        # Draw victory nose
        nose = Polygon(nose_points, facecolor='#7fff00', 
                      edgecolor='#ffffff', linewidth=2.2, alpha=1.0)
        nose.set_path_effects([
            pe.Stroke(linewidth=4.5, foreground='#66ff99'),
            pe.Normal()
        ])
        ax.add_patch(nose)
        
        # Draw happy eyes
        eye1 = Circle((center_x - 0.15, center_y - 0.08), 0.06, color='black')
        eye2 = Circle((center_x + 0.15, center_y - 0.08), 0.06, color='black')
        ax.add_patch(eye1)
        ax.add_patch(eye2)
        
        # Draw big happy smile
        ax.text(center_x, center_y + 0.15, '‿', ha='center', va='center', 
                fontsize=24, fontweight='bold', color='#000000')

    def visualize(self, puzzle, filename=None):
        """Visualize the Snake puzzle with realistic snake and apple shapes"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle, Circle, Ellipse, FancyBboxPatch, Polygon
            import numpy as np
            
            rows = puzzle['grid_size']['rows']
            cols = puzzle['grid_size']['cols']
            snake = puzzle['initial_snake']
            apples = puzzle['apples']
            direction = puzzle['initial_direction']
            
            # Create figure with neutral style (no green background or glow)
            fig, ax = plt.subplots(figsize=(max(10, cols * 0.95), max(10, rows * 0.95)))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')
            
            # Draw game board background with subtle pattern and crisp borders
            for i in range(rows):
                for j in range(cols):
                    base_color = '#f2f2f2' if (i + j) % 2 == 0 else '#e6e6e6'
                    rect = Rectangle((j, i), 1, 1, fill=True, color=base_color,
                                     edgecolor='#cfcfcf', linewidth=0.8, alpha=1.0)
                    ax.add_patch(rect)
            
            # Draw snake as a continuous body
            self._draw_continuous_snake(ax, snake, direction)
            
            # Draw apples with glossy style and glow
            for r, c in apples:
                # Apple body - slightly oval shape
                apple_body = Ellipse((c + 0.5, r + 0.55), 0.6, 0.65,
                                     facecolor='#d83232',
                                     edgecolor='#8a1a1a',
                                     linewidth=1.8,
                                     alpha=1.0)
                ax.add_patch(apple_body)
                
                # Apple highlight/shine
                shine = Ellipse((c + 0.35, r + 0.4), 0.15, 0.2,
                                facecolor='#ffd0d0', alpha=0.85)
                ax.add_patch(shine)
                
                # Apple stem (small brown rectangle)
                stem = Rectangle((c + 0.47, r + 0.15), 0.06, 0.15,
                                 facecolor='#7a3812',
                                 edgecolor='#5a2a0e', linewidth=1.2)
                ax.add_patch(stem)
                
                # Apple leaf (small green oval)
                leaf = Ellipse((c + 0.55, r + 0.25), 0.12, 0.08,
                                angle=30, facecolor='#1fad1f',
                                edgecolor='#006400', linewidth=1.2)
                ax.add_patch(leaf)

                # Subtle glow ring around apple
                # Remove strong glow to avoid greenish tint
            
            # Add subtle grid lines for better visualization
            for i in range(rows + 1):
                ax.axhline(i, color='#cfcfcf', linewidth=0.6, alpha=0.6)
            for j in range(cols + 1):
                ax.axvline(j, color='#cfcfcf', linewidth=0.6, alpha=0.6)
            
            # Set limits and aspect - only game area
            ax.set_xlim(0, cols)
            ax.set_ylim(rows, 0)  # Reverse y-axis
            ax.set_aspect('equal')
            
            # Remove ticks and spines for clean look
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Remove neon outer frame to avoid green accents
            
            # Save or show - only game area (high DPI, no unrelated text)
            if filename:
                plt.savefig(filename, bbox_inches='tight', facecolor=fig.get_facecolor(),
                            edgecolor='none', dpi=300, pad_inches=0)
                plt.close()
            else:
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Error in visualize: {e}")

    def visualize_solution(self, puzzle, filename=None):
        """Visualize the solution showing the snake's path with realistic shapes"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Ellipse
            import numpy as np
            
            rows = puzzle['grid_size']['rows']
            cols = puzzle['grid_size']['cols']
            initial_snake = puzzle['initial_snake']
            apples = puzzle['apples']
            actions = puzzle['answer']
            initial_direction = puzzle['initial_direction']
            
            # Simulate the game to show final state
            final_snake, path_positions = self._simulate_game(
                initial_snake, initial_direction, apples, actions, rows, cols)
            
            # Create figure with enhanced style
            fig, ax = plt.subplots(figsize=(max(10, cols * 0.9), max(10, rows * 0.9)))
            fig.patch.set_facecolor('#0a0a1a')
            ax.set_facecolor('#1a1a2e')
            
            # Draw game board with pattern
            for i in range(rows):
                for j in range(cols):
                    if (i + j) % 2 == 0:
                        color = '#16213e'
                    else:
                        color = '#1a2540'
                    rect = Rectangle((j, i), 1, 1, fill=True, color=color, 
                                   edgecolor='#2c3e50', linewidth=0.8, alpha=0.9)
                    ax.add_patch(rect)
            
            # Draw movement path with enhanced arrows
            for i in range(len(path_positions) - 1):
                start = path_positions[i]
                end = path_positions[i + 1]
                
                # Draw glowing path arrow
                arrow = FancyArrowPatch((start[1] + 0.5, start[0] + 0.5),
                                      (end[1] + 0.5, end[0] + 0.5),
                                      arrowstyle='->', mutation_scale=20,
                                      color='#ffd700', alpha=0.8, linewidth=3)
                ax.add_patch(arrow)
                
                # Add subtle glow effect
                glow_arrow = FancyArrowPatch((start[1] + 0.5, start[0] + 0.5),
                                           (end[1] + 0.5, end[0] + 0.5),
                                           arrowstyle='->', mutation_scale=25,
                                           color='#ffff00', alpha=0.3, linewidth=5)
                ax.add_patch(glow_arrow)
            
            # Draw final snake as continuous body with victory style
            self._draw_continuous_victory_snake(ax, final_snake)
            
            # Mark eaten apple positions with realistic eaten apple shape
            for r, c in apples:
                # Eaten apple - faded but still apple-shaped
                eaten_apple = Ellipse((c + 0.5, r + 0.55), 0.5, 0.55, 
                                    facecolor='#90ee90', 
                                    edgecolor='#006400', 
                                    linewidth=2,
                                    alpha=0.7)
                ax.add_patch(eaten_apple)
                
                # Add eaten apple stem
                eaten_stem = Rectangle((c + 0.47, r + 0.2), 0.06, 0.12, 
                                     facecolor='#654321', 
                                     edgecolor='#4a2c2a', linewidth=1, alpha=0.7)
                ax.add_patch(eaten_stem)
                
                # Success checkmark overlay
                ax.text(c + 0.5, r + 0.55, '✓', ha='center', va='center', 
                       fontsize=20, fontweight='bold', color='#006400')
            
            # Add subtle grid lines
            for i in range(rows + 1):
                ax.axhline(i, color='#34495e', linewidth=0.5, alpha=0.3)
            for j in range(cols + 1):
                ax.axvline(j, color='#34495e', linewidth=0.5, alpha=0.3)
            
            ax.set_xlim(0, cols)
            ax.set_ylim(rows, 0)
            ax.set_aspect('equal')
            
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            if filename:
                plt.savefig(filename, bbox_inches='tight', facecolor='#0a0a1a', 
                           edgecolor='none', dpi=200, pad_inches=0)
                plt.close()
            else:
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"Error in visualize_solution: {e}")
    
    def _simulate_game(self, initial_snake, initial_direction, apples, actions, rows, cols):
        """模拟游戏过程，返回最终蛇的状态和移动路径"""
        snake = list(initial_snake)
        direction = initial_direction
        remaining_apples = set(apples)
        path_positions = [snake[0]]  # 记录蛇头移动路径
        
        direction_deltas = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        
        for action in actions:
            direction = action
            dr, dc = direction_deltas[direction]
            head_r, head_c = snake[0]
            new_head = (head_r + dr, head_c + dc)
            
            snake.insert(0, new_head)
            path_positions.append(new_head)
            
            # 检查是否吃到苹果
            if new_head in remaining_apples:
                remaining_apples.remove(new_head)
                # 蛇变长，不移除尾巴
            else:
                # 没吃到苹果，移除尾巴
                snake.pop()
        
        return snake, path_positions

    def generate_cot(self, puzzle):
        """Generate enhanced Rule-Based CoT with 4 detailed structured steps and progressive parts."""
        rows = puzzle['grid_size']['rows']
        cols = puzzle['grid_size']['cols']
        snake = puzzle['initial_snake']
        apples = puzzle['apples']
        direction = puzzle['initial_direction']
        solution = puzzle['answer']

        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def half_truncate_words(text: str) -> str:
            words = text.split()
            if not words:
                return text
            half_count = max(1, len(words) // 2)
            truncated = " ".join(words[:half_count])
            return truncated
        
        def generate_grid_representation():
            """Generate a visual representation of the initial game state"""
            grid = [['.' for _ in range(cols)] for _ in range(rows)]
            
            # Mark snake body
            for i, (r, c) in enumerate(snake):
                if i == 0:
                    grid[r][c] = 'H'  # Head
                else:
                    grid[r][c] = 'S'  # Snake body
            
            # Mark apples
            for r, c in apples:
                grid[r][c] = 'A'
            
            # Convert to string
            grid_str = "\n".join([''.join(row) for row in grid])
            return grid_str

        intro = "I need to solve this Snake puzzle by carefully analyzing the image and applying strategic reasoning. Let me work through this step by step."

        # Step 1: Enhanced game rules understanding
        step1_lines = [
            "### Step 1: Understanding the Game Rules and Mechanics",
            "",
            "Let me first clarify the fundamental rules of this Snake puzzle:",
            "",
            "**Core Movement Rules:**",
            "- The snake moves one cell at a time in four directions: up, down, left, right",
            "- The snake cannot immediately reverse direction (e.g., if moving right, cannot move left next)",
            "- Each move, the snake's head advances to the next cell in the chosen direction",
            "",
            "**Collision Rules:**",
            "- The snake dies if it hits any wall (moves outside the grid boundaries)",
            "- The snake dies if its head collides with any part of its own body",
            "- These are the only two failure conditions",
            "",
            "**Apple and Growth Mechanics:**",
            f"- There are {len(apples)} apple(s) placed on the {rows}×{cols} grid",
            "- When the snake's head reaches an apple's position, the apple is consumed",
            "- Upon eating an apple, the snake's length increases by exactly 1 segment",
            "- The tail doesn't move for one turn when an apple is eaten (snake grows)",
            "",
            "**Victory Condition:**",
            f"- **Primary Goal:** Consume all {len(apples)} apple(s) without any collisions",
            "- The puzzle is solved when no apples remain on the board",
            "",
            "**Strategic Implications:**",
            "- As the snake grows longer, navigation becomes more constrained",
            "- Path planning must account for future snake length",
            "- Early moves must preserve space for later maneuvers"
        ]
        step1_text = "\n".join(step1_lines)

        # Step 2: Enhanced visual analysis with grid representation
        head_r, head_c = snake[0]
        apples_sorted = sorted(list(apples), key=lambda a: manhattan((head_r, head_c), a))
        grid_repr = generate_grid_representation()
        
        step2_lines = [
            "### Step 2: Careful Image Analysis and Initial State Reading",
            "",
            "Now I will read the image carefully to extract the precise initial game state:",
            "",
            "**Grid Layout Analysis:**",
            f"- Grid dimensions: {rows} rows × {cols} columns",
            f"- Total cells available: {rows * cols}",
            f"- Coordinate system: (row, col) with (0,0) at top-left",
            "",
            "**Visual Grid Representation:**",
            "```",
            grid_repr,
            "```",
            "Legend: H = Snake Head, S = Snake Body, A = Apple, . = Empty Cell",
            "",
            "**Snake Initial Configuration:**",
            f"- Snake body positions: {snake}",
            f"- Snake length: {len(snake)} segments",
            f"- Head position: ({head_r}, {head_c})",
            f"- Current facing direction: {direction}",
            f"- Tail position: {snake[-1]}",
            "",
            "**Apple Locations Analysis:**",
            f"- Apple positions: {list(apples)}",
            f"- Number of apples to collect: {len(apples)}",
            f"- Apples sorted by Manhattan distance from head: {apples_sorted}",
            "",
            "**Distance Analysis:**",
        ]
        
        # Add distance calculations for each apple
        for i, apple_pos in enumerate(apples_sorted):
            dist = manhattan((head_r, head_c), apple_pos)
            step2_lines.append(f"  - Apple at {apple_pos}: {dist} moves minimum (Manhattan distance)")
        
        step2_lines.extend([
            "",
            "**State Reading Verification:**",
            "Let me double-check my reading of the initial state:",
            f"✓ Grid size correctly identified as {rows}×{cols}",
            f"✓ Snake head at ({head_r}, {head_c}) facing {direction}",
            f"✓ Snake body has {len(snake)} segments total",
            f"✓ Found all {len(apples)} apples on the grid",
            f"✓ No overlapping positions between snake and apples",
            "",
            "**Spatial Constraints:**",
            f"- Available empty cells: {rows * cols - len(snake) - len(apples)}",
            f"- Grid boundaries: rows [0, {rows-1}], columns [0, {cols-1}]",
            "- Movement constraints: cannot reverse direction, cannot hit walls or self"
        ])
        step2_text = "\n".join(step2_lines)

        # Step 3: Enhanced strategic exploration and reasoning
        step3_lines = [
            "### Step 3: Strategic Exploration and Detailed Reasoning Process",
            "",
            "Now I'll develop a comprehensive strategy to solve this puzzle through systematic exploration:",
            "",
            "**Initial Strategic Assessment:**"
        ]
        
        # Analyze each apple strategically
        if apples_sorted:
            step3_lines.extend([
                "",
                "**Apple-by-Apple Strategic Analysis:**"
            ])
            
            for i, apple_pos in enumerate(apples_sorted, 1):
                dist = manhattan((head_r, head_c), apple_pos)
                step3_lines.append(f"")
                step3_lines.append(f"**Target {i}: Apple at {apple_pos}**")
                step3_lines.append(f"- Manhattan distance: {dist} moves")
                step3_lines.append(f"- Direct path analysis: Need to move {apple_pos[0] - head_r} vertically, {apple_pos[1] - head_c} horizontally")
                
                # Analyze potential obstacles
                if i == 1:  # First apple - most detailed analysis
                    step3_lines.append("- Path considerations:")
                    step3_lines.append("  * Check if snake body blocks direct routes")
                    step3_lines.append("  * Evaluate alternative paths if direct route blocked")
                    step3_lines.append("  * Consider space requirements for future growth")
                    step3_lines.append("  * Ensure tail clearance for return paths")
        
        # Simulate reasoning through initial moves
        step3_lines.extend([
            "",
            "**Move-by-Move Strategic Reasoning:**",
            "",
            "Let me trace through the solution logic step by step:"
        ])
        
        # Analyze the solution moves in detail
        if solution:
            current_snake = list(snake)
            current_dir = direction
            remaining_apples = set(apples)
            
            for i, move in enumerate(solution[:min(8, len(solution))]):  # Show first 8 moves in detail
                step3_lines.append(f"")
                step3_lines.append(f"**Move {i+1}: {move}**")
                
                # Calculate new head position
                direction_deltas = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
                dr, dc = direction_deltas[move]
                old_head = current_snake[0]
                new_head = (old_head[0] + dr, old_head[1] + dc)
                
                step3_lines.append(f"- Current head: {old_head} → New head: {new_head}")
                step3_lines.append(f"- Direction change: {current_dir} → {move}")
                
                # Check for apple consumption
                ate_apple = new_head in remaining_apples
                if ate_apple:
                    remaining_apples.remove(new_head)
                    step3_lines.append(f"- ✓ Apple consumed at {new_head}! Snake grows to length {len(current_snake) + 1}")
                    current_snake.insert(0, new_head)
                else:
                    step3_lines.append(f"- No apple at {new_head}, snake maintains length {len(current_snake)}")
                    current_snake.insert(0, new_head)
                    current_snake.pop()
                
                # Safety checks
                step3_lines.append(f"- Safety check: position {new_head} is within bounds and collision-free")
                
                current_dir = move
                
                if i >= 7 and len(solution) > 8:
                    step3_lines.append(f"")
                    step3_lines.append(f"... (continuing for {len(solution) - 8} more moves)")
                    break
        
        step3_lines.extend([
            "",
            "**Strategic Heuristics Applied:**",
            "- **Nearest-First Strategy:** Prioritize closest apples to minimize travel distance",
            "- **Space Preservation:** Avoid moves that create dead-ends or trap the snake",
            "- **Growth Planning:** Account for snake length increase when planning paths",
            "- **Boundary Awareness:** Use grid edges strategically to guide movement",
            "- **Tail Management:** Ensure the tail has clearance for complex maneuvers",
            "",
            "**Alternative Path Consideration:**",
            "- If direct path to nearest apple is blocked, consider perimeter approach",
            "- Evaluate trade-offs between immediate apple collection vs. strategic positioning",
            "- Plan escape routes before entering confined spaces",
            "",
            "**Risk Assessment:**",
            "- Monitor available space as snake grows longer",
            "- Avoid creating situations where the snake blocks its own path to remaining apples",
            "- Maintain flexibility for direction changes throughout the solution"
        ])
        step3_text = "\n".join(step3_lines)

        # Step 4: Enhanced validation and reflection
        final_length = len(snake) + len(apples)
        step4_lines = [
            "### Step 4: Solution Validation and Comprehensive Reflection",
            "",
            "Now I'll thoroughly validate the solution and reflect on the reasoning process:",
            "",
            "**Solution Completeness Verification:**",
            f"✓ All {len(apples)} apples are successfully collected",
            f"✓ Solution contains {len(solution)} moves",
            f"✓ Snake length progression: {len(snake)} → {final_length} (growth of {len(apples)})",
            "",
            "**Collision Safety Verification:**",
            "Let me verify each move is safe:"
        ]
        
        # Simulate the entire solution to verify safety
        if solution:
            current_snake = list(snake)
            current_dir = direction
            remaining_apples = set(apples)
            direction_deltas = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
            
            all_safe = True
            for i, move in enumerate(solution):
                dr, dc = direction_deltas[move]
                old_head = current_snake[0]
                new_head = (old_head[0] + dr, old_head[1] + dc)
                
                # Check bounds
                if not (0 <= new_head[0] < rows and 0 <= new_head[1] < cols):
                    all_safe = False
                    step4_lines.append(f"✗ Move {i+1} ({move}): Wall collision at {new_head}")
                    break
                
                # Check self-collision
                if new_head in current_snake:
                    all_safe = False
                    step4_lines.append(f"✗ Move {i+1} ({move}): Self-collision at {new_head}")
                    break
                
                # Update snake
                current_snake.insert(0, new_head)
                if new_head in remaining_apples:
                    remaining_apples.remove(new_head)
                else:
                    current_snake.pop()
                
                current_dir = move
            
            if all_safe:
                step4_lines.append("✓ All moves verified safe - no wall or self-collisions detected")
                step4_lines.append(f"✓ Final snake length: {len(current_snake)} (expected: {final_length})")
                step4_lines.append(f"✓ Remaining apples: {len(remaining_apples)} (expected: 0)")
        
        step4_lines.extend([
            "",
            "**Strategic Reflection:**",
            "- The solution demonstrates effective application of nearest-first strategy",
            "- Space management was crucial for avoiding self-trapping scenarios",
            "- Growth planning ensured the snake maintained maneuverability throughout",
            "",
            "**Algorithm Efficiency Analysis:**",
            f"- Solution efficiency: {len(solution)} moves for {len(apples)} apples",
            f"- Average moves per apple: {len(solution) / max(1, len(apples)):.1f}",
            "- Path optimality: Solution balances directness with safety constraints",
            "",
            "**Final Answer Confidence:**",
            f"Based on thorough analysis and verification, the solution is: **{' '.join(solution)}**",
            "",
            "This solution successfully guides the snake to collect all apples while avoiding all collision risks.",
            "The strategic approach demonstrates systematic planning and careful execution of the Snake puzzle mechanics."
        ])
        step4_text = "\n".join(step4_lines)

        # Compose progressive CoTs per requirement
        cot_step1_all = "\n\n".join([intro, step1_text])
        cot_step1_part = "\n\n".join([intro, half_truncate_words(step1_text)])

        cot_step2_all = "\n\n".join([intro, step1_text, step2_text])
        cot_step2_part = "\n\n".join([intro, step1_text, half_truncate_words(step2_text)])

        cot_step3_all = "\n\n".join([intro, step1_text, step2_text, step3_text])
        cot_step3_part = "\n\n".join([intro, step1_text, step2_text, half_truncate_words(step3_text)])

        full_cot = "\n\n".join([intro, step1_text, step2_text, step3_text, step4_text])

        return {
            'cot': full_cot,
            'cot_steps': {
                'cot_step1_part': cot_step1_part,
                'cot_step1_all': cot_step1_all,
                'cot_step2_part': cot_step2_part,
                'cot_step2_all': cot_step2_all,
                'cot_step3_part': cot_step3_part,
                'cot_step3_all': cot_step3_all,
            }
        }

    def _format_question_language(self, initial_state):
        """Format question_language by replacing image reference with initial_state"""
        # Extract only the essential game state information for question_language
        lines = initial_state.split('\n')
        essential_info = []
        
        for line in lines:
            if line.startswith('Grid:') or line.startswith('Snake:') or line.startswith('Direction:') or line.startswith('Apples:'):
                essential_info.append(line)
        
        formatted_state = '\n'.join(essential_info)
        
        # Use PROMPT_SNAKE and include the essential state information
        question_template = """
        You are a puzzle solver focusing on Snake puzzles.

### Game Rules:
1. Control a snake to move around the grid using directional commands (up, down, left, right)
2. The snake must eat all apples on the grid to win
3. When the snake eats an apple, it grows longer by one segment
4. The snake cannot collide with walls or itself
5. The snake moves one cell at a time in the chosen direction

### Initial state:
{}

### Input:
- A grid showing the initial state with the snake and apples
- Snake head is marked with 'H', snake body with 'S', apples with 'A', empty cells with '.'

### Goal:
Find a sequence of directional moves to eat all apples without the snake colliding with walls or itself.

### Output Format Requirements:
Your answer should be a sequence of directional moves separated by spaces.
Valid moves are: up, down, left, right
Example: up right down left up
        """
        
        return question_template.format(formatted_state)


    def _format_puzzle_question(self, puzzle):
        """Format the puzzle question as a string representation"""
        rows = puzzle['grid_size']['rows']
        cols = puzzle['grid_size']['cols']
        snake = puzzle['initial_snake']
        apples = puzzle['apples']
        direction = puzzle['initial_direction']
        apples_count = puzzle['apples_count']
        
        # Create a string representation
        question_lines = []
        
        # Add grid size
        question_lines.append(f"Grid: {rows}x{cols}")
        
        # Add initial snake
        snake_str = " ".join([f"({r},{c})" for r, c in snake])
        question_lines.append(f"Snake: {snake_str}")
        
        # Add initial direction
        question_lines.append(f"Direction: {direction}")
        
        # Add apples
        apples_str = " ".join([f"({r},{c})" for r, c in apples])
        question_lines.append(f"Apples: {apples_str}")
        
        # Add goal
        question_lines.append(f"Goal: Eat {apples_count} apple{'s' if apples_count > 1 else ''}")
        
        return "\n".join(question_lines)

    def _format_actions(self, actions):
        """Format the actions as a string"""
        return " ".join(actions)

    def solve(self, puzzle, **kwargs):
        """Solve a Snake puzzle using BFS"""
        rows = puzzle['grid_size']['rows']
        cols = puzzle['grid_size']['cols']
        initial_snake = puzzle['initial_snake']
        initial_direction = puzzle['initial_direction']
        apples = puzzle['apples']
        
        return self._find_solution(rows, cols, initial_snake, initial_direction, apples)
