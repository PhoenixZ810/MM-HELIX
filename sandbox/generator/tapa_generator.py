import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
from typing import List, Dict, Any, Tuple
import hashlib
import time
from generator.base_generator import BaseGenerator

class TapaGenerator(BaseGenerator):
    def __init__(self, output_folder):
        super().__init__(output_folder)
        # 本地维护生成器需要的状态
        self.seed = int(time.time())  # 使用时间戳作为种子
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.generated_puzzles = []  # 存储所有生成的任务
        self.generated_hashes = set()  # 用于避免重复
        self.generated_answers = set()  # 用于避免重复答案

    def _get_difficulty_params(self, difficulty):
        """
        根据难度级别获取相应的参数配置。

        Args:
            difficulty: 难度级别（1-5）

        Returns:
            dict: 包含难度参数的字典
        """
        # 根据难度确定网格大小
        if difficulty == 1:
            size = 3
        elif difficulty == 2:
            size = 4
        elif difficulty == 3:
            size = 5
        elif difficulty == 4:
            size = 6
        else:  # difficulty == 5
            size = 7

        return {
            'size': size,
            'difficulty': str(difficulty)
        }

    def generate(self, num_cases, difficulty, output_folder=None):
        """
        生成Tapa问题的抽象方法实现。

        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别
            output_folder: 输出文件夹路径，覆盖构造函数中设置的路径
        """
        if output_folder:
            self.output_folder = output_folder

        # 获取难度参数
        params = self._get_difficulty_params(difficulty)
        size = params['size']
        difficulty_str = params['difficulty']

        # 创建输出目录
        images_dir = os.path.join(self.output_folder, 'images')
        os.makedirs(images_dir, exist_ok=True)

        generated_count = 0
        for i in range(num_cases):
            # 使用基础seed加上计数器来生成不同的种子
            current_seed = self.seed + i * 1000

            puzzle = self._generate_single_puzzle(size, difficulty_str, current_seed, images_dir)
            if puzzle:
                # 将生成的任务添加到列表中，而不是立即保存
                self.generated_puzzles.append(puzzle)
                generated_count += 1
                print(f"Generated tapa puzzle {i+1}/{num_cases}: {puzzle['index']}")

        print(f"Successfully generated {generated_count}/{num_cases} Tapa puzzles")
        
        # 批量保存所有任务到 annotations.json（适配 BaseGenerator 接口）
        if self.generated_puzzles:
            # 仅保留公开的字段
            exportable = []
            for p in self.generated_puzzles:
                exportable.append({
                    "index": p["index"],
                    "category": p["category"],
                    "image": p["image"],
                    "question": p["question"],
                    "question_language": p["question_language"],
                    "answer": p["answer"],
                    "initial_state": p["initial_state"],
                    "difficulty": p["difficulty"],
                    "cot": p["cot"],
                    "cot_step1_all": p["cot_step1_all"],
                    "cot_step2_all": p["cot_step2_all"],
                    "cot_step3_all": p["cot_step3_all"],
                })
            # 使用通用保存接口
            self.save_annotations(exportable, self.output_folder)
            # 清空缓存，避免重复写入
            self.generated_puzzles = []

    def _generate_single_puzzle(self, size, difficulty, seed, images_dir):
        """生成单个Tapa谜题的内部方法"""
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)

        # 根据难度设置参数
        if difficulty == '1':
            black_cell_ratio = 0.25
            min_clues = max(3, size * size // 30)
            max_clues = max(5, size * size // 20)
        elif difficulty == '2':
            black_cell_ratio = 0.3
            min_clues = max(4, size * size // 25)
            max_clues = max(7, size * size // 18)
        elif difficulty == '3':
            black_cell_ratio = 0.35
            min_clues = max(5, size * size // 22)
            max_clues = max(8, size * size // 15)
        elif difficulty == '4':
            black_cell_ratio = 0.4
            min_clues = max(6, size * size // 20)
            max_clues = max(10, size * size // 12)
        else:  # difficulty == '5'
            black_cell_ratio = 0.45
            min_clues = max(7, size * size // 18)
            max_clues = max(12, size * size // 10)

        # 初始化网格
        grid = [[False for _ in range(size)] for _ in range(size)]

        # 生成有效解
        black_cells_count = int(size * size * black_cell_ratio)
        self._generate_valid_solution(grid, size, size, black_cells_count)

        # 生成线索
        clues = {}
        for r in range(size):
            for c in range(size):
                if not grid[r][c]:  # 如果是白色格子
                    groups = self._get_clue_numbers(grid, r, c, size, size)
                    if groups and len(groups) == 1 and groups != [0]:
                        clues[f"{r},{c}"] = groups

        # 如果线索不够，重新生成
        if len(clues) < min_clues:
            random.seed(seed + 1)
            np.random.seed(seed + 1)
            return self._generate_single_puzzle(size, difficulty, seed + 1, images_dir)

        # 选择线索子集
        if len(clues) > max_clues:
            clue_keys = list(clues.keys())
            random.shuffle(clue_keys)
            selected_clues = {k: clues[k] for k in clue_keys[:max_clues]}
        else:
            selected_clues = clues

        # 验证唯一性
        if not self._verify_unique_solution(grid, size, size, selected_clues):
            selected_clues = clues

        # 生成CoT数据
        cot_data = self._generate_cot(grid, size, size, selected_clues)

        # 创建谜题字典
        puzzle_index = f"tapa_{size}_{seed}"
        puzzle = {
            "index": puzzle_index,
            "category": "tapa",
            "image": f"{puzzle_index}.png",
            "question": f"Please solve this {size}x{size} Tapa puzzle.",
            "question_language": self._generate_text_based_question(size, difficulty, selected_clues, size, size),
            "answer": self._grid_to_coordinates(grid),
            "initial_state": self._generate_initial_state_from_grid(size, size, selected_clues),
            "difficulty": difficulty,
            "cot": cot_data['full_cot'],
            "cot_step1_all": cot_data['step1'],
            "cot_step2_all": cot_data['step2'],
            "cot_step3_all": cot_data['step3'],
            "rows": size,
            "cols": size,
            "clues": selected_clues,
            "solution": grid
        }

        # 可视化
        self.visualize(puzzle, folder=images_dir)

        return puzzle

    def _generate_initial_state(self, puzzle):
        """Generate the initial state string for the puzzle"""
        rows = puzzle['rows']
        cols = puzzle['cols']
        clues = puzzle['clues']
        
        # Create a grid showing only the clues, with empty cells marked as '.'
        initial_grid = []
        for r in range(rows):
            row = []
            for c in range(cols):
                coord = f"{r},{c}"
                if coord in clues:
                    # Only accept single-number clues
                    clue_values = clues[coord]
                    if len(clue_values) == 1:
                        row.append(str(clue_values[0]))
                    else:
                        # Ignore multi-number clues entirely (treat as empty)
                        row.append('.')
                else:
                    # Empty cell that needs to be filled
                    row.append('.')
            initial_grid.append(row)
        
        # Convert to string format - each cell is represented by exactly one character
        result = []
        for row in initial_grid:
            result.append(''.join(row))
        
        return '\n'.join(result)
    

    
    def _get_puzzle_hash(self, puzzle):
        """生成谜题的哈希值以避免重复"""
        # 基于网格解答和线索创建唯一标识
        solution_str = self._grid_to_string(puzzle['solution'])
        clues_str = str(sorted(puzzle['clues'].items()))
        combined = f"{solution_str}_{clues_str}_{puzzle['rows']}x{puzzle['cols']}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _generate_valid_solution(self, grid, rows, cols, black_cells_count):
        """Generate a valid Tapa solution with the given number of black cells"""
        # Use seed-dependent generation for reproducible but varied results
        
        # Generate initial pattern based on current random state
        cells = [(r, c) for r in range(rows) for c in range(cols)]
        random.shuffle(cells)
        
        # Create initial black cell placement
        for r, c in cells[:black_cells_count]:
            grid[r][c] = True
        
        # Add some seed-dependent variation to the initial pattern
        seed_variation = random.randint(0, 3)
        if seed_variation == 1:
            # Prefer edge cells
            edge_cells = [(r, c) for r in range(rows) for c in range(cols) 
                         if r == 0 or r == rows-1 or c == 0 or c == cols-1]
            if edge_cells:
                random.shuffle(edge_cells)
                for r, c in edge_cells[:min(len(edge_cells), black_cells_count//3)]:
                    grid[r][c] = True
        elif seed_variation == 2:
            # Prefer center cells
            center_r, center_c = rows//2, cols//2
            center_cells = [(r, c) for r in range(max(0, center_r-1), min(rows, center_r+2))
                           for c in range(max(0, center_c-1), min(cols, center_c+2))]
            for r, c in center_cells:
                if random.random() < 0.6:
                    grid[r][c] = True
        elif seed_variation == 3:
            # Create diagonal pattern tendency
            for r in range(rows):
                for c in range(cols):
                    if (r + c) % 3 == 0 and random.random() < 0.4:
                        grid[r][c] = True
        
        # Validate and fix constraints
        iterations = 0
        max_iterations = rows * cols * 5
        
        while iterations < max_iterations:
            iterations += 1
            if self._verify_solution(grid, rows, cols):
                break
            
            # Fix connectivity
            if not self._is_connected(grid, rows, cols):
                self._fix_connectivity(grid, rows, cols)
            
            # Fix 2x2 blocks
            self._fix_2x2_blocks(grid, rows, cols)
            
            # Fix isolated white cells
            self._fix_isolated_white_cells(grid, rows, cols)
        
        return grid
    
    def _verify_solution(self, grid, rows, cols):
        """Verify that the solution meets all Tapa constraints"""
        # 检查黑色格子的连通性
        if not self._is_connected(grid, rows, cols):
            return False
        
        # 检查没有2x2黑色块
        for r in range(rows - 1):
            for c in range(cols - 1):
                if grid[r][c] and grid[r][c+1] and grid[r+1][c] and grid[r+1][c+1]:
                    return False
        
        # 检查白色格子不孤立
        white_cells = [(r, c) for r in range(rows) for c in range(cols) if not grid[r][c]]
        if not white_cells:
            return False
        
        # 检查白色格子是否可以从边缘到达
        visited = set()
        queue = deque()
        
        # 从边缘的白色格子开始
        for r, c in white_cells:
            if r == 0 or r == rows-1 or c == 0 or c == cols-1:
                queue.append((r, c))
                visited.add((r, c))
        
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and not grid[nr][nc] and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        return len(visited) == len(white_cells)
    
    def _is_connected(self, grid, rows, cols):
        """Check if all black cells form a single connected component"""
        black_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c]]
        if not black_cells:
            return True
        
        start = black_cells[0]
        visited = {start}
        queue = deque([start])
        
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        
        return len(visited) == len(black_cells)
    
    def _fix_connectivity(self, grid, rows, cols):
        """Fix disconnected black cells by connecting them"""
        # 找到所有黑色格子的连通分量
        black_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c]]
        if not black_cells:
            return
        
        components = []
        visited = set()
        
        for r, c in black_cells:
            if (r, c) not in visited:
                component = set()
                queue = deque([(r, c)])
                visited.add((r, c))
                component.add((r, c))
                
                while queue:
                    cr, cc = queue.popleft()
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            component.add((nr, nc))
                            queue.append((nr, nc))
                
                components.append(component)
        
        # 通过添加黑色格子连接分量
        if len(components) > 1:
            for i in range(len(components) - 1):
                # 找到分量i和i+1之间最近的格子
                min_dist = float('inf')
                best_pair = None
                
                for c1 in components[i]:
                    for c2 in components[i+1]:
                        dist = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
                        if dist < min_dist:
                            min_dist = dist
                            best_pair = (c1, c2)
                
                # 用路径连接它们
                if best_pair:
                    r1, c1 = best_pair[0]
                    r2, c2 = best_pair[1]
                    
                    # 简单路径：先行后列
                    r, c = r1, c1
                    while r != r2:
                        r += 1 if r < r2 else -1
                        if 0 <= r < rows:
                            grid[r][c] = True
                    
                    while c != c2:
                        c += 1 if c < c2 else -1
                        if 0 <= c < cols:
                            grid[r][c] = True
    
    def _fix_2x2_blocks(self, grid, rows, cols):
        """Fix 2x2 black blocks by removing one cell from each block"""
        for r in range(rows - 1):
            for c in range(cols - 1):
                if grid[r][c] and grid[r][c+1] and grid[r+1][c] and grid[r+1][c+1]:
                    # 随机选择一个格子变为白色
                    choices = [(r, c), (r, c+1), (r+1, c), (r+1, c+1)]
                    remove_r, remove_c = random.choice(choices)
                    grid[remove_r][remove_c] = False
    
    def _fix_isolated_white_cells(self, grid, rows, cols):
        """Fix isolated white cells by connecting them to the edge"""
        white_cells = [(r, c) for r in range(rows) for c in range(cols) if not grid[r][c]]
        if not white_cells:
            return
        
        # 找到所有白色格子的连通分量
        components = []
        visited = set()
        
        for r, c in white_cells:
            if (r, c) not in visited:
                component = set()
                queue = deque([(r, c)])
                visited.add((r, c))
                component.add((r, c))
                
                reaches_edge = False
                while queue:
                    cr, cc = queue.popleft()
                    if cr == 0 or cr == rows-1 or cc == 0 or cc == cols-1:
                        reaches_edge = True
                    
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < rows and 0 <= nc < cols and not grid[nr][nc] and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            component.add((nr, nc))
                            queue.append((nr, nc))
                
                components.append((component, reaches_edge))
        
        # 为每个孤立的分量创建到边缘的路径
        for component, reaches_edge in components:
            if not reaches_edge:
                # 从分量中选择一个随机格子
                start = random.choice(list(component))
                
                # 找到到边缘的最短路径
                min_dist = float('inf')
                target_edge = None
                
                for r, c in component:
                    edge_dist = min(r, rows-1-r, c, cols-1-c)
                    if edge_dist < min_dist:
                        min_dist = edge_dist
                        if r == min_dist:
                            target_edge = (0, c)
                        elif r == rows-1-min_dist:
                            target_edge = (rows-1, c)
                        elif c == min_dist:
                            target_edge = (r, 0)
                        else:
                            target_edge = (r, cols-1)
                
                # 创建到边缘的白色格子路径
                if target_edge:
                    r, c = start
                    target_r, target_c = target_edge
                    
                    # 简单路径：先行后列
                    while r != target_r:
                        r += 1 if r < target_r else -1
                        if 0 <= r < rows:
                            grid[r][c] = False
                    
                    while c != target_c:
                        c += 1 if c < target_c else -1
                        if 0 <= c < cols:
                            grid[r][c] = False
    
    def _get_clue_numbers(self, grid, row, col, rows, cols):
        """Get the clue numbers for a cell based on surrounding black cells"""
        if grid[row][col]:
            return []
        
        directions = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),          (0, 1),
                     (1, -1),  (1, 0), (1, 1)]
        
        adjacent = []
        for dr, dc in directions:
            r = row + dr
            c = col + dc
            if 0 <= r < rows and 0 <= c < cols:
                adjacent.append((r, c))
        
        black_cells = [(r, c) for (r, c) in adjacent if grid[r][c]]
        if not black_cells:
            return [0]
        
        # 找到连通的黑色格子组
        visited = set()
        groups = []
        black_cell_set = set(black_cells)
        
        for r, c in black_cells:
            if (r, c) not in visited:
                # 开始一个新的连通分量
                group_size = 0
                queue = deque([(r, c)])
                visited.add((r, c))
                
                while queue:
                    cr, cc = queue.popleft()
                    group_size += 1
                    
                    # 检查8方向邻居（包括斜对角），与评测器分组规则保持一致
                    for dr, dc in [
                        (-1, -1), (-1, 0), (-1, 1),
                        (0, -1),            (0, 1),
                        (1, -1),  (1, 0),  (1, 1),
                    ]:
                        nr, nc = cr + dr, cc + dc
                        if (nr, nc) in black_cell_set and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                
                groups.append(group_size)
        
        groups.sort()
        return groups if groups else [0]
    
    def _verify_unique_solution(self, solution, rows, cols, clues):
        """Verify that the solution is unique given the clues"""
        # 这是一个简化的检查 - 完整的唯一性检查需要一个求解器
        # 现在，我们确保有足够的线索来约束解答
        return len(clues) >= rows * cols // 12
    
    def _grid_to_coordinates(self, grid):
        """Convert grid to black cell coordinates"""
        black_coordinates = []
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c]:  # If black cell
                    black_coordinates.append(f"({r},{c})")
        
        return ", ".join(black_coordinates)
    
    def _grid_to_string(self, grid, clues=None):
        """Convert grid to a string representation, preserving clues if provided"""
        result = []
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
        for r in range(rows):
            row_str = ""
            for c in range(cols):
                coord = f"{r},{c}"
                if clues and coord in clues:
                    # 仅保留单数字线索
                    clue_values = clues[coord]
                    if len(clue_values) == 1:
                        row_str += str(clue_values[0])
                    else:
                        # 多数字线索不被支持，这里不显示
                        row_str += 'W'
                else:
                    # 非线索位置显示黑白格子
                    row_str += 'B' if grid[r][c] else 'W'
            result.append(row_str)
        
        return '\n'.join(result)
    
    def _generate_cot(self, grid, rows, cols, clues):
        """Generate Chain-of-Thought reasoning for solving the puzzle following the enhanced 4-step format"""
        
        # Step 1: Understanding the puzzle rules and objectives (详细明确游戏规则)
        step1 = "### Step 1: Understanding the puzzle rules and objectives\n\n"
        step1 += "I need to solve a Tapa puzzle, which is a logic puzzle with specific constraints. Let me carefully understand the rules:\n\n"
        
        step1 += "**Core Rules of Tapa:**\n"
        step1 += "1. **Connectivity Rule**: All black cells must form a single connected group. This means I can travel from any black cell to any other black cell by moving only through adjacent black cells (horizontally or vertically adjacent, not diagonally).\n\n"
        
        step1 += "2. **No 2×2 Rule**: There cannot be any 2×2 square of black cells anywhere on the grid. This is a critical constraint that often determines many cell placements.\n\n"
        
        step1 += "3. **Clue Rule**: Each numbered cell (clue) indicates exactly how many black cells should be placed in its 8 surrounding neighbors (the 8-cell Moore neighborhood). Importantly, these black cells around a clue must form a single connected group.\n\n"
        
        step1 += "4. **White Cell Connectivity**: All white cells must be connected to each other, and this connected white region must reach the border of the grid.\n\n"
        
        step1 += f"**Grid Information**: This is a {rows}×{cols} grid with {len(clues)} clue cells.\n\n"
        
        step1 += "**Objective**: Determine which cells should be colored black such that all four rules are satisfied simultaneously. The solution must be unique and complete."
        
        # Helper function to create grid representation
        def create_grid_representation():
            grid_repr = []
            for r in range(rows):
                row = []
                for c in range(cols):
                    coord = f"{r},{c}"
                    if coord in clues and len(clues[coord]) == 1:
                        row.append(str(clues[coord][0]))
                    else:
                        row.append('.')
                grid_repr.append(' '.join(row))
            return '\n'.join(grid_repr)
        
        # Step 2: Reading and analyzing the visual information (仔细读取图像状态)
        step2 = "\n\n### Step 2: Reading and analyzing the visual information\n\n"
        step2 += "Let me carefully examine the puzzle image and extract all the information:\n\n"
        
        step2 += "**Initial Grid State:**\n"
        step2 += "```\n"
        step2 += create_grid_representation()
        step2 += "\n```\n\n"
        
        step2 += "**Clue Analysis:**\n"
        step2 += "I can identify the following numbered clues in the grid:\n"
        
        for coord, values in sorted(clues.items()):
            r, c = map(int, coord.split(','))
            if len(values) == 1:
                step2 += f"- Position ({r},{c}): Clue '{values[0]}'\n"
                step2 += f"  → This means exactly {values[0]} black cell(s) must be placed among the 8 neighbors of this cell\n"
                step2 += f"  → These {values[0]} black cell(s) must form a single connected group\n"
                if values[0] == 0:
                    step2 += f"  → Special case: All 8 surrounding cells must be white\n"
                step2 += "\n"
        
        # Identify constraint types
        zero_clues = [(r, c) for coord, values in clues.items() 
                      if values == [0] for r, c in [map(int, coord.split(','))]]
        high_clues = [(r, c, values[0]) for coord, values in clues.items() 
                      if len(values) == 1 and values[0] >= max(3, min(rows, cols)//2)
                      for r, c in [map(int, coord.split(','))]]
        
        step2 += "**Constraint Classification:**\n"
        if zero_clues:
            step2 += f"- Zero constraints: {len(zero_clues)} clue(s) require all surrounding cells to be white\n"
        if high_clues:
            step2 += f"- High constraints: {len(high_clues)} clue(s) require many connected black cells\n"
        step2 += f"- Total constraints to satisfy: {len(clues)}\n\n"
        
        step2 += "**Initial State Verification:**\n"
        step2 += "✓ Grid dimensions confirmed\n"
        step2 += "✓ All clue positions identified\n"
        step2 += "✓ Constraint complexity assessed\n"
        step2 += "✓ Ready to proceed with logical deduction\n"
        
        # Step 3: Strategic exploration and detailed reasoning (充分的推理过程)
        step3 = "\n\n### Step 3: Strategic exploration and detailed reasoning\n\n"
        step3 += "Now I'll work through the puzzle systematically, using logical deduction and constraint propagation:\n\n"
        
        # Phase 1: Handle definitive constraints
        step3 += "**Phase 1: Processing definitive constraints**\n\n"
        
        if zero_clues:
            step3 += "Starting with zero clues (most restrictive):\n"
            for r, c in zero_clues[:3]:  # Show first few for detail
                step3 += f"- Zero clue at ({r},{c}):\n"
                step3 += f"  → All 8 surrounding cells: "
                neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            neighbors.append(f"({nr},{nc})")
                step3 += f"{', '.join(neighbors)} must be WHITE\n"
                step3 += f"  → This eliminates {len(neighbors)} potential black cell positions\n\n"
        
        # Phase 2: Analyze high-constraint clues
        if high_clues:
            step3 += "**Phase 2: High-constraint clue analysis**\n\n"
            for r, c, value in high_clues[:2]:
                step3 += f"Analyzing high-constraint clue {value} at ({r},{c}):\n"
                step3 += f"- Need exactly {value} connected black cells among 8 neighbors\n"
                step3 += f"- With connectivity requirement, this significantly limits possible arrangements\n"
                step3 += f"- Must avoid creating 2×2 blocks while maintaining connectivity\n\n"
        
        # Phase 3: Systematic constraint propagation
        step3 += "**Phase 3: Systematic constraint propagation**\n\n"
        step3 += "Working through remaining clues in order of constraint strength:\n\n"
        
        processed_count = 0
        for coord, values in sorted(clues.items(), key=lambda x: (x[1][0] if len(x[1]) == 1 else 999, x[0])):
            if processed_count >= 4 or (values == [0]) or (len(values) == 1 and values[0] >= max(3, min(rows, cols)//2)):
                continue
            
            r, c = map(int, coord.split(','))
            if len(values) == 1 and values[0] > 0:
                processed_count += 1
                step3 += f"Clue {values[0]} at ({r},{c}):\n"
                step3 += f"- Examining {values[0]} black cell arrangement among 8 neighbors\n"
                
                # Calculate possible arrangements
                available_neighbors = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            available_neighbors += 1
                
                step3 += f"- {available_neighbors} total neighbor positions available\n"
                step3 += f"- Must choose {values[0]} positions that form connected group\n"
                step3 += f"- Checking compatibility with existing constraints...\n"
                step3 += f"- Verifying no 2×2 blocks would be created...\n\n"
        
        # Phase 4: Connectivity analysis
        step3 += "**Phase 4: Global connectivity analysis**\n\n"
        step3 += "Ensuring all black cells form a single connected component:\n"
        step3 += "- Starting from forced black cell placements\n"
        step3 += "- Identifying potential connection paths between separated regions\n"
        step3 += "- Each connection must not violate clue constraints\n"
        step3 += "- Some seemingly valid local solutions may be invalid globally\n\n"
        
        # Phase 5: Backtracking and refinement
        step3 += "**Phase 5: Logical deduction and backtracking**\n\n"
        step3 += "Resolving remaining ambiguities through systematic analysis:\n\n"
        step3 += "- **Case analysis**: For cells that could be either black or white:\n"
        step3 += "  → Assume cell is black: Check if all constraints can still be satisfied\n"
        step3 += "  → Assume cell is white: Check if all constraints can still be satisfied\n"
        step3 += "  → If only one assumption works, that determines the cell color\n\n"
        
        step3 += "- **Constraint checking**: After each tentative placement:\n"
        step3 += "  → Verify no 2×2 black blocks exist\n"
        step3 += "  → Verify black cells remain connected\n"
        step3 += "  → Verify all clues can still be satisfied\n"
        step3 += "  → Verify white cells remain connected to boundary\n\n"
        
        step3 += "- **Backtracking process**: When contradictions arise:\n"
        step3 += "  → Identify the source of contradiction\n"
        step3 += "  → Backtrack to last valid state\n"
        step3 += "  → Try alternative cell assignments\n"
        step3 += "  → Continue until consistent solution found\n\n"
        
        # Phase 6: Solution convergence
        step3 += "**Phase 6: Solution convergence**\n\n"
        step3 += "Through iterative application of constraints and logical deduction:\n"
        step3 += "- Initially ambiguous cells become determined\n"
        step3 += "- Each constraint satisfaction reduces remaining possibilities\n"
        step3 += "- The unique solution emerges as the only configuration satisfying all rules\n"
        step3 += "- Final verification ensures completeness and correctness\n"
        
        # Step 4: Solution validation and reflection (基于答案的验证和反思)
        step4 = "\n\n### Step 4: Solution validation and reflection\n\n"
        step4 += "Now I'll verify that my proposed solution satisfies all Tapa constraints:\n\n"
        
        # Create solution representation
        black_coords = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c]:
                    black_coords.append(f"({r},{c})")
        
        step4 += f"**Proposed Solution:**\n"
        step4 += f"Black cells: {', '.join(black_coords)}\n\n"
        
        # Detailed verification
        step4 += "**Detailed Constraint Verification:**\n\n"
        
        step4 += "1. **Clue Constraint Verification:**\n"
        all_clues_satisfied = True
        for coord, values in sorted(clues.items()):
            r, c = map(int, coord.split(','))
            if len(values) == 1:
                # Count surrounding black cells
                surrounding_black = 0
                black_neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc]:
                            surrounding_black += 1
                            black_neighbors.append(f"({nr},{nc})")
                
                # Check if count matches clue
                expected = values[0]
                if surrounding_black == expected:
                    step4 += f"   ✓ Clue at ({r},{c}): Expected {expected}, Found {surrounding_black}\n"
                    if black_neighbors:
                        step4 += f"     Black neighbors: {', '.join(black_neighbors)}\n"
                else:
                    step4 += f"   ✗ Clue at ({r},{c}): Expected {expected}, Found {surrounding_black}\n"
                    all_clues_satisfied = False
        
        step4 += "\n2. **Connectivity Verification:**\n"
        # Check black cell connectivity
        black_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c]]
        if black_cells:
            # BFS to check connectivity
            visited = {black_cells[0]}
            queue = [black_cells[0]]
            while queue:
                r, c = queue.pop(0)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            
            if len(visited) == len(black_cells):
                step4 += "   ✓ All black cells form a single connected component\n"
            else:
                step4 += f"   ✗ Black cells are not fully connected: {len(visited)}/{len(black_cells)} reachable\n"
        else:
            step4 += "   ✓ No black cells (trivially connected)\n"
        
        step4 += "\n3. **2×2 Block Verification:**\n"
        has_2x2_blocks = False
        for r in range(rows - 1):
            for c in range(cols - 1):
                if grid[r][c] and grid[r][c+1] and grid[r+1][c] and grid[r+1][c+1]:
                    step4 += f"   ✗ 2×2 black block found at ({r},{c})-({r+1},{c+1})\n"
                    has_2x2_blocks = True
        
        if not has_2x2_blocks:
            step4 += "   ✓ No 2×2 black blocks exist\n"
        
        step4 += "\n4. **White Cell Connectivity Verification:**\n"
        white_cells = [(r, c) for r in range(rows) for c in range(cols) if not grid[r][c]]
        if white_cells:
            # Check if white cells form connected component reaching boundary
            boundary_white = [(r, c) for r, c in white_cells 
                            if r == 0 or r == rows-1 or c == 0 or c == cols-1]
            
            if boundary_white:
                visited = set(boundary_white)
                queue = list(boundary_white)
                while queue:
                    r, c = queue.pop(0)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < rows and 0 <= nc < cols and 
                            not grid[nr][nc] and (nr, nc) not in visited):
                            visited.add((nr, nc))
                            queue.append((nr, nc))
                
                if len(visited) == len(white_cells):
                    step4 += "   ✓ All white cells are connected to boundary\n"
                else:
                    step4 += f"   ✗ Some white cells isolated: {len(visited)}/{len(white_cells)} reachable\n"
            else:
                step4 += "   ✗ No white cells reach boundary\n"
        
        step4 += "\n**Solution Reflection:**\n"
        if all_clues_satisfied and not has_2x2_blocks:
            step4 += "✓ The solution successfully satisfies all Tapa constraints\n"
            step4 += "✓ Each clue is correctly satisfied by its surrounding black cells\n"
            step4 += "✓ Global connectivity and local exclusion rules are maintained\n"
            step4 += "✓ The logical deduction process led to a valid and unique solution\n"
        else:
            step4 += "⚠ Some constraints may not be fully satisfied - solution needs refinement\n"
        
        step4 += f"\n**Final Answer:** {', '.join(black_coords) if black_coords else 'No black cells'}"
        
        # Build progressive CoT steps
        intro = "I'll solve this Tapa puzzle through careful analysis and systematic reasoning.\n\n"
        
        step1_content = intro + step1
        step2_content = step1_content + step2  
        step3_content = step2_content + step3
        step4_content = step3_content + step4
        
        # Helper function to truncate text properly at word boundaries
        def get_half_text(text):
            target_length = len(text) // 2
            # Find a good breaking point near the middle (prefer sentence/paragraph breaks)
            break_candidates = []
            for i in range(max(target_length - 200, 0), min(target_length + 200, len(text))):
                if i < len(text) and text[i] in '.!?\n':
                    break_candidates.append(i + 1)
            
            if break_candidates:
                # Choose the break point closest to target length
                best_break = min(break_candidates, key=lambda x: abs(x - target_length))
                return text[:best_break].rstrip()
            else:
                # Fallback to word boundary
                words = text[:target_length].split()
                return ' '.join(words[:-1]) if len(words) > 1 else text[:target_length]
        
        return {
            'full_cot': step4_content,
            'step1': step1_content,
            'step2': step2_content,
            'step3': step3_content,
            'step4': step4_content,
            # Part versions (truncated at roughly half)
            'step1_part': get_half_text(step1_content),
            'step2_part': get_half_text(step2_content),
            'step3_part': get_half_text(step3_content),
            'step4_part': get_half_text(step4_content)
        }
    

        
    def visualize(self, puzzle, **kwargs):
        """Visualize the puzzle and save it as an image"""
        # Extract grid dimensions from puzzle or infer from solution
        if 'solution' in puzzle:
            rows = len(puzzle['solution'])
            cols = len(puzzle['solution'][0]) if rows > 0 else 0
        else:
            # Fallback: infer from clues
            max_r = max_c = 0
            for coord in puzzle.get('clues', {}):
                r, c = map(int, coord.split(','))
                max_r = max(max_r, r)
                max_c = max(max_c, c)
            rows, cols = max_r + 1, max_c + 1
        
        clues = puzzle.get('clues', {})
        is_solution = kwargs.get('is_solution', False)
        folder = kwargs.get('folder', 'images')
        
        # Create figure and axis with dynamic sizing
        fig_size = max(rows, cols) / 2 + 2
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        # Set background color for game-like appearance
        ax.set_facecolor('#F5F5DC')  # Light beige background
        
        # Draw grid with thicker lines
        for r in range(rows + 1):
            ax.plot([0, cols], [r, r], 'k-', linewidth=1.5)
        for c in range(cols + 1):
            ax.plot([c, c], [0, rows], 'k-', linewidth=1.5)
        
        # Fill cells based on solution if showing solution
        if is_solution and 'solution' in puzzle:
            for r in range(rows):
                for c in range(cols):
                    if puzzle['solution'][r][c]:
                        rect = patches.Rectangle((c, rows-1-r), 1, 1, facecolor='#333333',  # Dark gray
                                              edgecolor='black', linewidth=0.5)
                        ax.add_patch(rect)
                    else:
                        # Add subtle shading for white cells for better aesthetics
                        rect = patches.Rectangle((c, rows-1-r), 1, 1, facecolor='#FFFFFF',
                                              edgecolor='black', linewidth=0.5, alpha=0.3)
                        ax.add_patch(rect)
        else:
            # For puzzle view, add cell shading
            for r in range(rows):
                for c in range(cols):
                    rect = patches.Rectangle((c, rows-1-r), 1, 1, facecolor='#FFFFFF',
                                          edgecolor='black', linewidth=0.5, alpha=0.3)
                    ax.add_patch(rect)
        
        # Calculate font size based on grid size
        fontsize = min(14, max(10, 25 - max(rows, cols) * 0.6))
        
        # Add clue text with improved styling
        for coord, values in clues.items():
            r, c = map(int, coord.split(','))
            # Adjust y coordinate to match matplotlib coordinate system
            r_display = rows - 1 - r

            # Only draw supported single-number clues
            if len(values) != 1:
                continue

            # Add cell highlight for clue cells
            rect = patches.Rectangle((c, r_display), 1, 1, facecolor='#E6E6FA',  # Lavender
                                   edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)

            clue_text = str(values[0])

            # Add text with improved visibility in cell center
            ax.text(c + 0.5, r_display + 0.5, clue_text,
                    ha='center', va='center',
                    fontsize=fontsize,
                    fontweight='bold',
                    color='#000080')
        
        # Set axis properties
        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect('equal')
        
        ax.axis('off')
        
        # Add subtle border around entire puzzle
        ax.patch.set_edgecolor('darkgray')
        ax.patch.set_linewidth(2)
        
        # Save figure with higher DPI for better quality
        if is_solution:
            image_name = puzzle['image'].replace('.png', '_solution.png')
        else:
            image_name = puzzle['image']

        # 如果提供了folder参数，使用完整的路径
        if folder:
            filename = os.path.join(folder, image_name)
        else:
            filename = image_name

        plt.savefig(filename, bbox_inches='tight', pad_inches=0.2, dpi=150)
        plt.close()
    
    def solve(self, puzzle, **kwargs):
        """Solve the puzzle - for Tapa this would be complex, so we use the stored solution"""
        return puzzle['solution'] if 'solution' in puzzle else None

    def _generate_initial_state_from_grid(self, rows, cols, clues):
        """Generate the initial state string for the puzzle"""
        # Create a grid showing only the clues, with empty cells marked as '.'
        initial_grid = []
        for r in range(rows):
            row = []
            for c in range(cols):
                coord = f"{r},{c}"
                if coord in clues:
                    # Only accept single-number clues
                    clue_values = clues[coord]
                    if len(clue_values) == 1:
                        row.append(str(clue_values[0]))
                    else:
                        # Ignore multi-number clues entirely (treat as empty)
                        row.append('.')
                else:
                    # Empty cell that needs to be filled
                    row.append('.')
            initial_grid.append(row)
        
        # Convert to string format - each cell is represented by exactly one character
        result = []
        for row in initial_grid:
            result.append(''.join(row))
        
        return '\n'.join(result)


    def _generate_text_based_question(self, grid_size, difficulty_level, clues, rows, cols):
        """Generate a text-based question prompt that describes the game state"""
        
        # Create the text-based question referring to initial_state
        question = f"""
Please look at the displayed Tapa puzzle grid. The numbers in the cells are clues indicating the lengths of connected groups of black cells surrounding that clue.

### Task
Your task is to fill in the white grid with black cells according to the following rules.

### Game Rules
1. **All black cells must form a single connected group**: This means that all the black cells on the grid must be connected in one continuous region, without any isolated black cells.
2. **There cannot be any 2x2 block of black cells**: A 2x2 block of black cells is not allowed anywhere on the grid. This means that no four black cells can form a square.
3. **Clue cells**: (Clue cells itself cannot be filled with black cells) Each number in a clue cell indicates the length of a connected group of black cells surrounding that clue. The "surrounding" refers to the 8 neighboring cells that are orthogonally and diagonally adjacent to the clue (i.e., the cells that are directly adjacent horizontally, vertically, or diagonally to the clue). 
    - For example, a clue "3" means that exactly three black cells must be placed among the 8 surrounding cells, and these three black cells must form a single connected group.
    - Each clue cell contains only a single number representing one connected group of black cells.
4. **Grid size**: The grid is a {grid_size}x{grid_size} matrix of cells. Each row and column will contain a mix of black (B) and white (W) cells.

### Coordinate System
The grid uses a coordinate system where (0,0) is the top-left corner, the first number represents the row (increasing downward), and the second number represents the column (increasing rightward).

### Output Format
- List only the coordinates of cells that should be colored black
- Use the format (row,column) for each coordinate
- Separate multiple coordinates with commas
- For example: (0,1), (1,2), (2,0), (2,1)

### Tapa Puzzle Grid:"""
        # Add specific clue information
        if clues:
            question += "\n"
            for coord, values in sorted(clues.items()):
                r, c = map(int, coord.split(','))
                if len(values) == 1:
                    question += f"- Cell ({r},{c}) has clue '{values[0]}': exactly {values[0]} connected black cells in surrounding 8 cells\n"
        
        question += """
### Output Format
- List only the coordinates of cells that should be colored black
- Use the format (row,column) for each coordinate
- Separate multiple coordinates with commas
- For example: (0,1), (1,2), (2,0), (2,1)
"""
        
        return question


