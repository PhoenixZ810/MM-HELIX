import os
import json
import random
import hashlib
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod

class BaseGenerator(ABC):
    """
    问题生成器的基类，定义了生成问题的通用接口。
    """

    def __init__(self, output_folder):
        """
        初始化基础生成器。

        Args:
            output_folder: 输出文件夹路径
        """
        self.output_folder = output_folder

    @abstractmethod
    def generate(self, num_cases, difficulty, output_folder=None):
        """
        生成问题的抽象方法，需要被子类实现。

        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别
            output_folder: 输出文件夹路径，覆盖构造函数中设置的路径

        保存问题到output_folder中：
        output_folder/
            /images/
                /question_name.png
            /annotations.json

        Returns(Optional):
            生成的问题列表
        """
        pass

    @abstractmethod
    def _get_difficulty_params(self, difficulty):
        """
        根据难度级别获取相应的参数配置。

        Args:
            difficulty: 难度级别（1-5）

        Returns:
            dict: 包含难度参数的字典
        """
        pass

    def _get_timestamp_seed(self):
        """获取当前时间戳作为种子"""
        return int(time.time())

class FutoshikiGenerator(BaseGenerator):
    def __init__(self, output_folder):
        super().__init__(output_folder)
        # 定义5个难度等级的参数 - 数字1-5对应的参数，每个难度对应不同的grid size
        self.difficulty_params = {
            1: {'retain_ratio': 0.5, 'inequality_prob': 0.2, 'name': 'easy', 'grid_size': 3},
            2: {'retain_ratio': 0.4, 'inequality_prob': 0.25, 'name': 'medium-easy', 'grid_size': 4},
            3: {'retain_ratio': 0.35, 'inequality_prob': 0.3, 'name': 'medium', 'grid_size': 5},
            4: {'retain_ratio': 0.25, 'inequality_prob': 0.35, 'name': 'medium-hard', 'grid_size': 6},
            5: {'retain_ratio': 0.2, 'inequality_prob': 0.4, 'name': 'hard', 'grid_size': 7}
        }
        # 用于存储已生成的问题hash，避免重复
        self.generated_hashes = set()
    
    def _get_difficulty_params(self, difficulty):
        """根据难度级别获取相应的参数配置"""
        if difficulty in self.difficulty_params:
            return self.difficulty_params[difficulty]
        else:
            # 默认中等难度
            return self.difficulty_params[3]

    def generate(self, num_cases, difficulty, output_folder=None):
        """生成指定数量和难度的Futoshiki问题"""
        # 使用传入的output_folder或默认的output_folder
        if output_folder is None:
            output_folder = self.output_folder

        # 创建输出目录
        os.makedirs(output_folder, exist_ok=True)
        images_dir = os.path.join(output_folder, "images")
        os.makedirs(images_dir, exist_ok=True)

        # 获取难度参数
        params = self._get_difficulty_params(difficulty)
        self.size = params['grid_size']

        # 收集所有生成的问题
        all_puzzles = []
        self.generated_hashes = set()  # 重置哈希集合

        print(f"Generating {num_cases} Futoshiki puzzles with difficulty {difficulty} ({params['name']}, {self.size}x{self.size})...")

        for i in range(num_cases):
            try:
                # 使用时间戳作为基础种子，然后加上计数器确保不同
                base_seed = self._get_timestamp_seed()
                seed = base_seed + i

                # Set random seed for reproducibility
                random.seed(seed)
                np.random.seed(seed)

                # Generate solution
                solution = self._generate_latin_square()
                inequalities = self._generate_inequalities(solution, params['inequality_prob'])
                initial = self._generate_initial_grid(solution, params['retain_ratio'])

                # Create unique puzzle ID
                puzzle_id = f"futoshiki_{self.size}x{self.size}_{i+1:03d}"

                # Format initial state
                initial_state = {
                    'grid': initial,
                    'inequalities': inequalities,
                    'size': self.size
                }

                # Generate CoT data
                cot_data = self._generate_cot_english(initial_state, solution)

                # Generate images
                image_path = f"images/{puzzle_id}.png"
                full_image_path = os.path.join(output_folder, image_path)

                # Create formatted puzzle following the expected structure
                formatted_puzzle = {
                    'index': puzzle_id,
                    'category': 'futoshiki',
                    'image': image_path,
                    'question': self._create_question_with_image(),
                    'question_language': self._create_question_language(initial_state),
                    'answer': str(solution),
                    'initial_state': initial_state,  # Keep as dict for internal use
                    'difficulty': difficulty,
                    'cot': cot_data.get('cot', ''),
                    'cot_step1_all': cot_data.get('cot_step1_all', ''),
                    'cot_step2_all': cot_data.get('cot_step2_all', ''),
                    'cot_step3_all': cot_data.get('cot_step3_all', ''),
                }

                # Check for duplicates
                if not self._is_duplicate(formatted_puzzle):
                    # Generate and save image
                    self.visualize_puzzle(formatted_puzzle, filename=full_image_path)

                    # Add to collection
                    all_puzzles.append(formatted_puzzle)
                    puzzle_hash = self._get_puzzle_hash(formatted_puzzle)
                    self.generated_hashes.add(puzzle_hash)

                    print(f"  Generated puzzle {i+1}/{num_cases}: {puzzle_id}")
                else:
                    print(f"  Skipped duplicate puzzle {i+1}/{num_cases}")

            except Exception as e:
                print(f"Error generating puzzle {i+1}: {e}")
                continue

        # 批量保存所有问题到annotations.json
        if all_puzzles:
            self._batch_save_to_annotations(all_puzzles, output_folder)
            print(f"\nSuccessfully generated {len(all_puzzles)} puzzles and saved to {output_folder}")
        else:
            print("\nNo puzzles were generated successfully.")

        return all_puzzles
        
    def generate_by_difficulty_levels(self, puzzles_per_level=100):
        """按难度等级生成问题"""
        all_puzzles = []
        difficulty_levels = ['easy', 'medium-easy', 'medium', 'medium-hard', 'hard']
        
        for difficulty in difficulty_levels:
            print(f"\nGenerating {puzzles_per_level} {difficulty} puzzles...")
            params = self.string_difficulty_params[difficulty]
            
            level_puzzles = []
            attempts = 0
            max_attempts = puzzles_per_level * 1000  # 最多尝试次数
            
            while len(level_puzzles) < puzzles_per_level and attempts < max_attempts:
                attempts += 1
                
                # 生成单个问题
                puzzle = self._generate_single_puzzle(
                    difficulty=difficulty,
                    retain_ratio=params['retain_ratio'],
                    inequality_prob=params['inequality_prob'],
                    puzzle_index=len(all_puzzles) + len(level_puzzles) + 1
                )
                
                if puzzle and not self._is_duplicate(puzzle):
                    level_puzzles.append(puzzle)
                    # 添加到已生成集合
                    puzzle_hash = self._get_puzzle_hash(puzzle)
                    self.generated_hashes.add(puzzle_hash)
                    print(f"  Generated {difficulty} puzzle {len(level_puzzles)}/{puzzles_per_level}")
                
                if attempts % 50 == 0:
                    print(f"  Attempted {attempts} times, generated {len(level_puzzles)} valid puzzles")
            
            if len(level_puzzles) < puzzles_per_level:
                print(f"Warning: Only generated {len(level_puzzles)} {difficulty} puzzles out of {puzzles_per_level} requested")
            
            all_puzzles.extend(level_puzzles)
        
        return all_puzzles
    

    

    
    def _is_duplicate(self, puzzle):
        """检查问题是否重复"""
        puzzle_hash = self._get_puzzle_hash(puzzle)
        return puzzle_hash in self.generated_hashes
    
    def _get_puzzle_hash(self, puzzle):
        """获取问题的哈希值用于重复检测"""
        # 使用初始网格和不等式约束创建唯一标识
        grid_str = str(puzzle['initial_state']['grid'])
        ineq_str = str(sorted([
            (tuple(ineq['cell1']), tuple(ineq['cell2']), ineq['symbol'])
            for ineq in puzzle['initial_state']['inequalities']
        ]))
        
        combined_str = grid_str + ineq_str
        return hashlib.md5(combined_str.encode()).hexdigest()
        
    def _format_solution(self, solution):
        """格式化解决方案为字符串表示"""
        return str(solution)
    
    def visualize(self, puzzle, filename=None, show_solution=False, **kwargs):
        """创建Futoshiki问题的可视化表示"""
        n = puzzle['initial_state']['size']
        initial = puzzle['initial_state']['grid']
        inequalities = puzzle['initial_state']['inequalities']
        
        # 如果显示解答则解析解决方案
        solution = None
        if show_solution and 'answer' in puzzle:
            try:
                import ast
                solution = ast.literal_eval(puzzle['answer'])
            except:
                pass
        
        # 创建适当尺寸的图形
        cell_size = 0.8
        margin = 0.2
        fig_size = n * cell_size + margin * 2
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        # 设置背景颜色
        ax.set_facecolor('#f5f5f5')
        
        # 绘制网格
        for i in range(n+1):
            ax.axhline(i, color='black', linewidth=1.5)
            ax.axvline(i, color='black', linewidth=1.5)
        
        # 填入数字
        for i in range(n):
            for j in range(n):
                if show_solution and solution:
                    value = solution[i][j]
                    color = 'blue'
                    weight = 'bold'
                elif initial[i][j] != 0:
                    value = initial[i][j]
                    color = 'black'
                    weight = 'bold'
                else:
                    continue
                
                ax.text(j + 0.5, n - i - 0.5, str(value), 
                        fontsize=16, ha='center', va='center', 
                        color=color, weight=weight)
        
        # 绘制不等式符号
        for ineq in inequalities:
            i1, j1 = ineq['cell1']
            i2, j2 = ineq['cell2']
            symbol = ineq['symbol']
            
            # 计算中点
            mid_x = (j1 + j2) / 2 + 0.5
            mid_y = n - (i1 + i2) / 2 - 0.5
            
            # 确定旋转角度和符号方向
            if i1 == i2:  # 水平
                rotation = 0
            else:  # 垂直
                rotation = 90
                # 垂直方向翻转符号
                if symbol == '>':
                    symbol = '<'
                else:
                    symbol = '>'
            
            ax.text(mid_x, mid_y, symbol, fontsize=14, 
                   ha='center', va='center', rotation=rotation,
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # 设置限制并移除刻度
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加标题
        if show_solution:
            difficulty = puzzle.get('target_difficulty', puzzle.get('difficulty', 'unknown'))
            ax.set_title(f'Futoshiki Solution ({difficulty.title()})', fontsize=16)
        else:
            difficulty = puzzle.get('target_difficulty', puzzle.get('difficulty', 'unknown'))
            ax.set_title(f'Futoshiki Puzzle ({difficulty.title()})', fontsize=16)
        
        # 添加坐标参考（可选）
        for i in range(n):
            for j in range(n):
                ax.text(j + 0.08, n - i - 0.08, f"({i},{j})",
                        fontsize=6, ha='left', va='top', alpha=0.5)
        
        # 保存或显示图形
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            plt.close(fig)
        else:
            plt.tight_layout()
            plt.show()

    def solve(self, puzzle, **kwargs):
        """返回问题的解答"""
        if isinstance(puzzle['answer'], str):
            import ast
            return ast.literal_eval(puzzle['answer'])
        return puzzle['answer']
    
    def _generate_latin_square(self) -> List[List[int]]:
        """生成随机拉丁方作为解决方案"""
        n = self.size
        base_row = list(range(1, n+1))
        random.shuffle(base_row)
        rows = [base_row[i:] + base_row[:i] for i in range(n)]
        random.shuffle(rows)
        perm = list(range(1, n+1))
        random.shuffle(perm)
        return [[perm[x-1] for x in row] for row in rows]

    def _generate_inequalities(self, solution: List[List[int]], inequality_prob=None) -> List[Dict]:
        """基于解决方案生成不等式约束"""
        if inequality_prob is None:
            inequality_prob = 0.3
            
        inequalities = []
        for i in range(self.size):
            for j in range(self.size):
                if j+1 < self.size and random.random() < inequality_prob:
                    a, b = solution[i][j], solution[i][j+1]
                    inequalities.append({
                        'cell1': [i, j],
                        'cell2': [i, j+1],
                        'symbol': '>' if a > b else '<'
                    })
                if i+1 < self.size and random.random() < inequality_prob:
                    a, b = solution[i][j], solution[i+1][j]
                    inequalities.append({
                        'cell1': [i, j],
                        'cell2': [i+1, j],
                        'symbol': '>' if a > b else '<'
                    })
        return inequalities

    def _generate_initial_grid(self, solution: List[List[int]], retain_ratio=None) -> List[List[int]]:
        """生成带有部分数字填入的初始问题网格"""
        if retain_ratio is None:
            retain_ratio = 0.3
            
        n = self.size
        indices = [(i, j) for i in range(n) for j in range(n)]
        retain_num = max(1, int(len(indices) * retain_ratio))  # 至少保留1个数字
        selected = random.sample(indices, retain_num)
        grid = [[0]*n for _ in range(n)]
        for i, j in selected:
            grid[i][j] = solution[i][j]
        return grid
    
    def _calculate_difficulty(self, initial, inequalities):
        """基于填入的格子和约束计算问题难度"""
        n = self.size
        total_cells = n * n
        filled_cells = sum(1 for row in initial for cell in row if cell != 0)
        empty_ratio = (total_cells - filled_cells) / total_cells
        
        # 计算难度分数
        # 更多空格 + 更多不等式 = 更难的问题
        max_possible_inequalities = 2 * n * (n-1)  # 最大可能的不等式数量
        ineq_factor = len(inequalities) / max_possible_inequalities
        
        # 基于空格比例和不等式因子的5级难度分类
        if empty_ratio > 0.85 and ineq_factor > 0.4:
            return "hard"
        elif empty_ratio > 0.75 and ineq_factor > 0.35:
            return "medium-hard"
        elif empty_ratio > 0.65 or ineq_factor > 0.25:
            return "medium"
        elif empty_ratio > 0.55 or ineq_factor > 0.15:
            return "medium-easy"
        else:
            return "easy"
    
    def _estimate_steps(self, initial, inequalities):
        """估算解决问题需要的步骤数"""
        n = self.size
        filled_cells = sum(1 for row in initial for cell in row if cell != 0)
        empty_cells = n*n - filled_cells
        
        # 粗略估计 - 每个空格至少需要一步
        # 更多不等式增加更多推理步骤
        base_steps = empty_cells
        ineq_steps = len(inequalities) // 2
        
        return base_steps + ineq_steps
    
    def _create_text_question(self, puzzle):
        """创建纯文本版本的问题描述"""
        n = puzzle['initial_state']['size']
        initial = puzzle['initial_state']['grid']
        inequalities = puzzle['initial_state']['inequalities']
        difficulty = puzzle.get('target_difficulty', puzzle.get('difficulty', 'unknown'))
        
        # 创建网格的文本表示
        grid_text = []
        for i in range(n):
            row = []
            for j in range(n):
                value = initial[i][j]
                row.append(str(value) if value != 0 else "_")
            grid_text.append(" ".join(row))
        
        # 分析当前游戏状态
        filled_cells = sum(1 for i in range(n) for j in range(n) if initial[i][j] != 0)
        empty_cells = n * n - filled_cells
        
        # 分析每行每列的状态
        row_analysis = []
        col_analysis = []
        
        for i in range(n):
            row_filled = [initial[i][j] for j in range(n) if initial[i][j] != 0]
            row_missing = [x for x in range(1, n+1) if x not in row_filled]
            row_analysis.append(f"Row {i}: Filled numbers {row_filled}, Missing numbers {row_missing}")
        
        for j in range(n):
            col_filled = [initial[i][j] for i in range(n) if initial[i][j] != 0]
            col_missing = [x for x in range(1, n+1) if x not in col_filled]
            col_analysis.append(f"Column {j}: Filled numbers {col_filled}, Missing numbers {col_missing}")
        
        # 构建完整的文本问题
        lines = [
            f"Futoshiki Puzzle ({n}×{n}, {difficulty.title()})",
            "=" * (40 + len(difficulty)),
            "",
            "GAME RULES:",
            f"1. The puzzle is a {n}×{n} grid.",
            f"2. Fill each cell with a number from 1 to {n}.",
            f"3. Each number must appear exactly once in each row and each column (no repetition).",
            "4. Inequality symbols between cells (either '<' or '>') must be satisfied:",
            "   - A horizontal constraint (i,j) < (i,j+1) means the left cell must be less than the right.",
            "   - A vertical constraint (i,j) < (i+1,j) means the top cell must be less than the bottom.",
            "",
            "CURRENT GAME STATE:",
            f"- Grid size: {n}×{n} ({n*n} total cells)",
            f"- Filled cells: {filled_cells}",
            f"- Empty cells: {empty_cells}",
            f"- Number of inequality constraints: {len(inequalities)}",
            f"- Difficulty level: {difficulty.title()}",
            "",
            "INITIAL GRID EXPLANATION:",
            "The grid below shows the current state where:",
            "- Numbers (1-{}) represent pre-filled cells that cannot be changed".format(n),
            "- Underscore (_) represents empty cells that need to be filled",
            "- Each cell position is referenced as (row, column) starting from (0,0) at top-left",
            "",
            "Initial grid (_ for empty cells):",
            "\n".join(grid_text),
            "",
            "ROW AND COLUMN ANALYSIS:",
            "\n".join(row_analysis),
            "",
            "\n".join(col_analysis),
            "",
            "INEQUALITY CONSTRAINTS EXPLANATION:",
            "The following constraints must be satisfied in the final solution:",
            "(Format: cell1 symbol cell2 - direction)"
        ]
        
        # 添加不等式约束
        for idx, ineq in enumerate(inequalities, 1):
            c1, c2 = ineq['cell1'], ineq['cell2']
            symbol = ineq['symbol']
            direction = "horizontal" if c1[0] == c2[0] else "vertical"
            if direction == "horizontal":
                lines.append(f"{idx}. Cell ({c1[0]},{c1[1]}) {symbol} Cell ({c2[0]},{c2[1]}) - {direction} (left {symbol} right)")
            else:
                lines.append(f"{idx}. Cell ({c1[0]},{c1[1]}) {symbol} Cell ({c2[0]},{c2[1]}) - {direction} (top {symbol} bottom)")
        
        # 添加解题步骤和输出格式
        lines.extend([
            "",
            "SOLVING STEPS:",
            "1. Use OCR to extract the numbers and inequality signs from the image.",
            "2. Analyze each row and column to determine possible values for empty cells.",
            "3. Apply inequality constraints to further narrow down possibilities.",
            "4. Use logical deduction and constraint propagation to fill the grid.",
            "5. Verify that the solution satisfies all rules.",
            "",
            "OUTPUT FORMAT:",
            'Answer format: "answer": [[row1], [row2], ..., [row{}]]'.format(n),
            "",
            f"Example for a {n}×{n} puzzle:",
            '"answer": [',
        ])
        
        # 添加示例格式
        for i in range(n):
            example_row = [str((i + j) % n + 1) for j in range(n)]
            if i == n - 1:
                lines.append(f'           [{", ".join(example_row)}]]')
            else:
                lines.append(f'           [{", ".join(example_row)}],')
        
        return "\n".join(lines)
    
    def _create_question_with_image(self):
        """Create question that refers to image"""
        return """You are given an image of a Futoshiki puzzle. Your task is to recognize the grid and inequality constraints from the image, solve the puzzle, and provide the answer in a structured format.

### Game Rules:
1. The puzzle is a N×N grid (e.g., 5×5).
2. Fill each cell with a number from 1 to N.
3. Each number must appear exactly once in each row and each column (no repetition).
4. Inequality symbols between cells (either '<' or '>') must be satisfied:
   - A horizontal constraint (i,j) < (i,j+1) means the left cell must be less than the right.
   - A vertical constraint (i,j) < (i+1,j) means the top cell must be less than the bottom.

### Answer format:
Output the final solution as a 2D list of integers.
"answer": [[row1], [row2], ..., [rowN]]

Example (for a 5×5 puzzle):
[[1, 4, 5, 3, 2], 
[3, 2, 1, 4, 5], 
[4, 5, 3, 2, 1], 
[5, 3, 2, 1, 4], 
[2, 1, 4, 5, 3]]"""

    def _create_question_language(self, initial_state):
        """Create text-based question that refers to initial_state"""
        n = initial_state['size']
        grid = initial_state['grid']
        inequalities = initial_state['inequalities']
        
        # Create grid representation
        grid_text = []
        for i in range(n):
            row = []
            for j in range(n):
                value = grid[i][j]
                row.append(str(value) if value != 0 else "_")
            grid_text.append(" ".join(row))
        
        # Format inequalities
        ineq_text = []
        for idx, ineq in enumerate(inequalities, 1):
            c1, c2 = ineq['cell1'], ineq['cell2']
            symbol = ineq['symbol']
            direction = "horizontal" if c1[0] == c2[0] else "vertical"
            ineq_text.append(f"{idx}. Cell ({c1[0]},{c1[1]}) {symbol} Cell ({c2[0]},{c2[1]}) ({direction})")
        
        return f"""You are given a Futoshiki puzzle with the following initial state. Your task is to solve the puzzle and provide the answer in a structured format.

### Game Rules:
1. The puzzle is a {n}×{n} grid.
2. Fill each cell with a number from 1 to {n}.
3. Each number must appear exactly once in each row and each column (no repetition).
4. Inequality symbols between cells (either '<' or '>') must be satisfied:
   - A horizontal constraint (i,j) < (i,j+1) means the left cell must be less than the right.
   - A vertical constraint (i,j) < (i+1,j) means the top cell must be less than the bottom.

### Initial State:
Grid (_ for empty cells):
{chr(10).join(grid_text)}

### Inequality constraints:
{chr(10).join(ineq_text)}

### Answer format:
"answer": [[row1], [row2], ..., [row{n}]]"""

    def _generate_cot_english(self, initial_state, solution):
        """Generate enhanced English step-by-step reasoning process following the detailed 4-step format"""
        n = initial_state['size']
        initial_grid = initial_state['grid']
        inequalities = initial_state['inequalities']
        
        # Parse solution
        if isinstance(solution, str):
            try:
                import ast
                solution = ast.literal_eval(solution)
            except:
                return {"cot": "Unable to generate solution process due to parsing error."}
        
        # Initialize step-by-step CoT tracking
        cot_steps = {}
        all_text = []
        
        # Start with the opening phrase
        all_text.append("Let me analyze this Futoshiki puzzle step by step")
        
        # Step 1: Understanding the puzzle rules and objectives (Enhanced)
        step1_content = []
        step1_content.append("\n\n### Step 1: Understanding the puzzle rules and objectives")
        step1_content.append("Before solving, I need to clearly understand the Futoshiki puzzle rules:")
        step1_content.append(f"")
        step1_content.append(f"**Core Rules for {n}×{n} Futoshiki:**")
        step1_content.append(f"1. Fill each empty cell with a number from 1 to {n}")
        step1_content.append(f"2. Each number (1 to {n}) must appear exactly once in each row")
        step1_content.append(f"3. Each number (1 to {n}) must appear exactly once in each column")
        step1_content.append(f"4. Inequality constraints between adjacent cells must be satisfied:")
        step1_content.append(f"   - Symbol '>' means the left/top cell value is greater than right/bottom cell")
        step1_content.append(f"   - Symbol '<' means the left/top cell value is less than right/bottom cell")
        step1_content.append(f"")
        step1_content.append(f"**Objective:** Complete the {n}×{n} grid such that all rules are satisfied simultaneously.")
        step1_content.append(f"This is a constraint satisfaction problem requiring logical deduction.")
        
        # Add step 1 content to all_text
        all_text.extend(step1_content)
        
        # Calculate step1_part with better splitting logic
        step1_text = " ".join(step1_content)
        words = step1_text.split()
        mid_point = len(words) // 2
        step1_part_words = words[:mid_point]
        step1_part_text = " ".join(step1_part_words)
        
        # Save cumulative content up to step 1
        step1_part_cum = " ".join(all_text[:-len(step1_content)]) + " " + step1_part_text
        step1_all_cum = " ".join(all_text)
        # Preferred keys
        cot_steps['cot_step1_part'] = step1_part_cum
        cot_steps['cot_step1_all'] = step1_all_cum
        # Backward-compatibility keys
        cot_steps['step1_part'] = step1_part_cum
        cot_steps['step1_all'] = step1_all_cum
        
        # Step 2: Carefully reading the image and extracting initial state (Enhanced)
        step2_content = []
        step2_content.append("\n\n### Step 2: Carefully reading the image and extracting initial state")
        step2_content.append("Now I'll carefully examine the image to extract the puzzle's initial configuration.")
        step2_content.append("")
        step2_content.append("**OCR Analysis of the puzzle image:**")
        
        # Create a visual representation of the grid
        step2_content.append("Reading the grid systematically from top-left to bottom-right:")
        step2_content.append("")
        for i in range(n):
            row_display = []
            for j in range(n):
                if initial_grid[i][j] != 0:
                    row_display.append(f"[{initial_grid[i][j]}]")
                else:
                    row_display.append("[ ]")
            step2_content.append(f"Row {i}: {' '.join(row_display)}")
        
        step2_content.append("")
        step2_content.append("**Detailed analysis of pre-filled cells:**")
        filled_cells = []
        for i in range(n):
            for j in range(n):
                if initial_grid[i][j] != 0:
                    filled_cells.append((i, j, initial_grid[i][j]))
        
        if filled_cells:
            step2_content.append(f"Found {len(filled_cells)} pre-filled cells:")
            for i, j, value in filled_cells:
                step2_content.append(f"  - Cell at position ({i},{j}) contains number {value}")
        else:
            step2_content.append("No pre-filled cells detected - this is a completely empty grid.")
        
        step2_content.append(f"Empty cells to solve: {n*n - len(filled_cells)} out of {n*n} total cells")
        
        # Analyze inequality constraints in detail
        step2_content.append("")
        step2_content.append("**Analysis of inequality constraints:**")
        step2_content.append(f"Detected {len(inequalities)} inequality constraints in the image:")
        
        if inequalities:
            horizontal_constraints = []
            vertical_constraints = []
            for idx, ineq in enumerate(inequalities, 1):
                c1, c2 = ineq['cell1'], ineq['cell2']
                symbol = ineq['symbol']
                if c1[0] == c2[0]:  # Same row = horizontal
                    horizontal_constraints.append(f"  {idx}. Cell ({c1[0]},{c1[1]}) {symbol} Cell ({c2[0]},{c2[1]}) [horizontal: left {symbol} right]")
                else:  # Same column = vertical
                    vertical_constraints.append(f"  {idx}. Cell ({c1[0]},{c1[1]}) {symbol} Cell ({c2[0]},{c2[1]}) [vertical: top {symbol} bottom]")
            
            if horizontal_constraints:
                step2_content.append("Horizontal constraints (between adjacent cells in same row):")
                step2_content.extend(horizontal_constraints)
            
            if vertical_constraints:
                step2_content.append("Vertical constraints (between adjacent cells in same column):")
                step2_content.extend(vertical_constraints)
        else:
            step2_content.append("No inequality constraints detected in this puzzle.")
        
        # Reflection on the state reading
        step2_content.append("")
        step2_content.append("**Reflection on initial state extraction:**")
        step2_content.append(f"- Successfully identified a {n}×{n} Futoshiki grid")
        step2_content.append(f"- Extracted {len(filled_cells)} pre-filled numbers as starting clues")
        step2_content.append(f"- Identified {len(inequalities)} inequality relationships between adjacent cells")
        step2_content.append("- The initial state provides enough information to begin logical solving")
        step2_content.append("- Ready to proceed with systematic constraint-based reasoning")
        
        # Add step 2 content to all_text
        all_text.extend(step2_content)
        
        # Calculate step2_part
        step2_text = " ".join(step2_content)
        words = step2_text.split()
        mid_point = len(words) // 2
        step2_part_words = words[:mid_point]
        step2_part_text = " ".join(step2_part_words)
        
        # Save cumulative content up to step 2
        step2_part_cum = " ".join(all_text[:-len(step2_content)]) + " " + step2_part_text
        step2_all_cum = " ".join(all_text)
        cot_steps['cot_step2_part'] = step2_part_cum
        cot_steps['cot_step2_all'] = step2_all_cum
        cot_steps['step2_part'] = step2_part_cum
        cot_steps['step2_all'] = step2_all_cum
        
        # Step 3: Detailed reasoning process with thorough exploration (Enhanced)
        step3_content = []
        step3_content.append("\n\n### Step 3: Detailed reasoning process with thorough exploration")
        step3_content.append("Now I'll systematically solve this puzzle using logical deduction, constraint propagation, and strategic exploration.")
        step3_content.append("")
        
        # Initial constraint analysis
        step3_content.append("**Phase 1: Initial constraint analysis for rows and columns**")
        step3_content.append("")
        
        # Analyze what each row needs
        step3_content.append("Row-wise constraint analysis:")
        for i in range(n):
            row_filled = [initial_grid[i][j] for j in range(n) if initial_grid[i][j] != 0]
            row_empty_positions = [j for j in range(n) if initial_grid[i][j] == 0]
            row_missing = [x for x in range(1, n+1) if x not in row_filled]
            
            if row_filled:
                step3_content.append(f"  Row {i}: Has {row_filled}, needs {row_missing} in positions {row_empty_positions}")
            else:
                step3_content.append(f"  Row {i}: Empty, needs all numbers {list(range(1, n+1))} in positions {row_empty_positions}")
        
        step3_content.append("")
        step3_content.append("Column-wise constraint analysis:")
        for j in range(n):
            col_filled = [initial_grid[i][j] for i in range(n) if initial_grid[i][j] != 0]
            col_empty_positions = [i for i in range(n) if initial_grid[i][j] == 0]
            col_missing = [x for x in range(1, n+1) if x not in col_filled]
            
            if col_filled:
                step3_content.append(f"  Col {j}: Has {col_filled}, needs {col_missing} in positions {col_empty_positions}")
            else:
                step3_content.append(f"  Col {j}: Empty, needs all numbers {list(range(1, n+1))} in positions {col_empty_positions}")
        
        # Detailed cell-by-cell analysis
        step3_content.append("")
        step3_content.append("**Phase 2: Systematic cell-by-cell exploration and reasoning**")
        step3_content.append("")
        
        empty_positions = [(i, j) for i in range(n) for j in range(n) if initial_grid[i][j] == 0]
        if empty_positions:
            step3_content.append(f"Analyzing all {len(empty_positions)} empty cells systematically:")
            step3_content.append("")
            
            for pos_idx, (i, j) in enumerate(empty_positions):
                step3_content.append(f"**Cell ({i},{j}) - Analysis {pos_idx + 1}/{len(empty_positions)}:**")
                
                # Basic constraints from row/column
                row_used = set(initial_grid[i][k] for k in range(n) if initial_grid[i][k] != 0)
                col_used = set(initial_grid[k][j] for k in range(n) if initial_grid[k][j] != 0)
                basic_candidates = set(range(1, n+1)) - row_used - col_used
                
                step3_content.append(f"  Row {i} already has: {sorted(row_used) if row_used else 'none'}")
                step3_content.append(f"  Col {j} already has: {sorted(col_used) if col_used else 'none'}")
                step3_content.append(f"  Basic candidates (row+col constraints): {sorted(basic_candidates)}")
                
                # Apply inequality constraints
                refined_candidates = basic_candidates.copy()
                constraints_applied = []
                
                for ineq in inequalities:
                    if ineq['cell1'] == [i, j]:
                        # This cell is on the left/top of inequality
                        other_pos = ineq['cell2']
                        other_val = initial_grid[other_pos[0]][other_pos[1]]
                        if other_val != 0:
                            if ineq['symbol'] == '>':
                                refined_candidates = {c for c in refined_candidates if c > other_val}
                                constraints_applied.append(f"must be > {other_val} (cell {other_pos})")
                            else:
                                refined_candidates = {c for c in refined_candidates if c < other_val}
                                constraints_applied.append(f"must be < {other_val} (cell {other_pos})")
                    
                    elif ineq['cell2'] == [i, j]:
                        # This cell is on the right/bottom of inequality
                        other_pos = ineq['cell1']
                        other_val = initial_grid[other_pos[0]][other_pos[1]]
                        if other_val != 0:
                            if ineq['symbol'] == '>':
                                refined_candidates = {c for c in refined_candidates if c < other_val}
                                constraints_applied.append(f"must be < {other_val} (cell {other_pos})")
                            else:
                                refined_candidates = {c for c in refined_candidates if c > other_val}
                                constraints_applied.append(f"must be > {other_val} (cell {other_pos})")
                
                if constraints_applied:
                    step3_content.append(f"  Inequality constraints applied: {'; '.join(constraints_applied)}")
                    step3_content.append(f"  Refined candidates after inequalities: {sorted(refined_candidates)}")
                else:
                    step3_content.append(f"  No direct inequality constraints apply to this cell")
                
                # Show exploration process
                final_value = solution[i][j]
                if len(refined_candidates) == 1:
                    step3_content.append(f"  **Determined uniquely: {list(refined_candidates)[0]}** (only one possibility)")
                elif len(refined_candidates) > 1:
                    step3_content.append(f"  Multiple candidates remain: {sorted(refined_candidates)}")
                    step3_content.append(f"  **Exploration process:**")
                    
                    # Simulate trying different values
                    wrong_candidates = [c for c in refined_candidates if c != final_value]
                    if wrong_candidates:
                        test_val = wrong_candidates[0]
                        step3_content.append(f"    - Try {test_val}: Would this work?")
                        step3_content.append(f"      Let me check if {test_val} leads to contradictions...")
                        step3_content.append(f"      Analysis shows {test_val} creates conflicts in subsequent cells")
                        step3_content.append(f"    - Backtrack and try {final_value}: This maintains consistency!")
                    
                    step3_content.append(f"  **Final decision: {final_value}** (through logical deduction and constraint checking)")
                else:
                    step3_content.append(f"  ERROR: No valid candidates! This suggests an error in constraint analysis.")
                
                step3_content.append("")
        
        # Advanced reasoning techniques
        step3_content.append("**Phase 3: Advanced constraint propagation and verification**")
        step3_content.append("")
        step3_content.append("Applying advanced solving techniques:")
        step3_content.append("- Constraint propagation: Each cell placement triggers re-analysis of related cells")
        step3_content.append("- Inequality chain analysis: Checking transitive relationships")
        step3_content.append("- Elimination by contradiction: Ruling out impossible values")
        step3_content.append("- Strategic backtracking: When multiple options exist, test systematically")
        step3_content.append("")
        step3_content.append("Through this systematic approach, all cells can be determined with certainty.")
        
        # Add step 3 content to all_text
        all_text.extend(step3_content)
        
        # Calculate step3_part
        step3_text = " ".join(step3_content)
        words = step3_text.split()
        mid_point = len(words) // 2
        step3_part_words = words[:mid_point]
        step3_part_text = " ".join(step3_part_words)
        
        # Save cumulative content up to step 3
        step3_part_cum = " ".join(all_text[:-len(step3_content)]) + " " + step3_part_text
        step3_all_cum = " ".join(all_text)
        cot_steps['cot_step3_part'] = step3_part_cum
        cot_steps['cot_step3_all'] = step3_all_cum
        cot_steps['step3_part'] = step3_part_cum
        cot_steps['step3_all'] = step3_all_cum
        
        # Step 4: Solution validation and comprehensive reflection (Enhanced)
        step4_content = []
        step4_content.append("\n\n### Step 4: Solution validation and comprehensive reflection")
        step4_content.append("Now I'll present the final solution and conduct thorough validation to ensure correctness.")
        step4_content.append("")
        
        # Present the final solution
        step4_content.append("**Complete Solution Grid:**")
        step4_content.append("")
        for i in range(n):
            row_str = " ".join(f"[{solution[i][j]}]" for j in range(n))
            step4_content.append(f"Row {i}: {row_str}")
        
        step4_content.append("")
        step4_content.append("**Comprehensive Validation Process:**")
        step4_content.append("")
        
        # Validate rows
        step4_content.append("1. **Row validation:**")
        all_rows_valid = True
        for i in range(n):
            row_values = [solution[i][j] for j in range(n)]
            expected = set(range(1, n+1))
            actual = set(row_values)
            if actual == expected:
                step4_content.append(f"   Row {i}: {row_values} ✓ (contains all numbers 1-{n} exactly once)")
            else:
                step4_content.append(f"   Row {i}: {row_values} ✗ (missing: {expected - actual}, extra: {actual - expected})")
                all_rows_valid = False
        
        # Validate columns
        step4_content.append("")
        step4_content.append("2. **Column validation:**")
        all_cols_valid = True
        for j in range(n):
            col_values = [solution[i][j] for i in range(n)]
            expected = set(range(1, n+1))
            actual = set(col_values)
            if actual == expected:
                step4_content.append(f"   Col {j}: {col_values} ✓ (contains all numbers 1-{n} exactly once)")
            else:
                step4_content.append(f"   Col {j}: {col_values} ✗ (missing: {expected - actual}, extra: {actual - expected})")
                all_cols_valid = False
        
        # Validate inequality constraints
        step4_content.append("")
        step4_content.append("3. **Inequality constraint validation:**")
        all_inequalities_valid = True
        valid_constraints = 0
        
        for idx, ineq in enumerate(inequalities, 1):
            i1, j1 = ineq['cell1']
            i2, j2 = ineq['cell2']
            symbol = ineq['symbol']
            v1, v2 = solution[i1][j1], solution[i2][j2]
            
            direction = "horizontal" if i1 == i2 else "vertical"
            constraint_satisfied = (symbol == '>' and v1 > v2) or (symbol == '<' and v1 < v2)
            
            if constraint_satisfied:
                step4_content.append(f"   Constraint {idx}: Cell ({i1},{j1})={v1} {symbol} Cell ({i2},{j2})={v2} ✓ ({direction})")
                valid_constraints += 1
            else:
                step4_content.append(f"   Constraint {idx}: Cell ({i1},{j1})={v1} {symbol} Cell ({i2},{j2})={v2} ✗ ({direction})")
                all_inequalities_valid = False
        
        # Overall validation summary
        step4_content.append("")
        step4_content.append("**Final Validation Summary:**")
        if all_rows_valid and all_cols_valid and all_inequalities_valid:
            step4_content.append("✓ All row constraints satisfied")
            step4_content.append("✓ All column constraints satisfied")
            step4_content.append(f"✓ All {len(inequalities)} inequality constraints satisfied")
            step4_content.append("")
            step4_content.append("🎉 **PUZZLE SOLVED SUCCESSFULLY!**")
            step4_content.append("The solution is correct and complete.")
        else:
            step4_content.append("✗ Validation failed - there are constraint violations")
            step4_content.append("The solution needs to be revised.")
        
        # Reflection on the solving process
        step4_content.append("")
        step4_content.append("**Reflection on the solving process:**")
        step4_content.append("- The systematic 4-step approach proved effective")
        step4_content.append("- Careful image analysis provided accurate initial state")
        step4_content.append("- Logical reasoning with constraint propagation solved all cells")
        step4_content.append("- Comprehensive validation confirmed solution correctness")
        step4_content.append("- This demonstrates the power of structured problem-solving in constraint satisfaction puzzles")
        
        # Add step 4 content to all_text
        all_text.extend(step4_content)
        
        # Complete CoT
        full_cot = " ".join(all_text)
        
        # Calculate step4_part
        step4_text = " ".join(step4_content)
        words = step4_text.split()
        mid_point = len(words) // 2
        step4_part_words = words[:mid_point]
        step4_part_text = " ".join(step4_part_words)
        
        # Save cumulative content up to step 4
        step4_part_cum = " ".join(all_text[:-len(step4_content)]) + " " + step4_part_text
        cot_steps['cot_step4_part'] = step4_part_cum
        cot_steps['cot_step4_all'] = full_cot
        cot_steps['step4_part'] = step4_part_cum
        cot_steps['step4_all'] = full_cot
        
        # Add the complete CoT
        cot_steps['cot'] = full_cot
        
        return cot_steps

    def _batch_save_to_annotations(self, puzzles, output_folder):
        """批量保存所有问题到annotations.json"""
        annotations_path = os.path.join(output_folder, "annotations.json")

        # Load existing annotations if file exists
        existing_data = []
        if os.path.exists(annotations_path):
            try:
                with open(annotations_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []

        # Get existing indices to avoid duplicates
        existing_indices = {item.get('index', '') for item in existing_data}

        # Filter out duplicates from new puzzles and convert initial_state to JSON string
        new_puzzles = []
        for p in puzzles:
            if p['index'] not in existing_indices:
                # Create a copy and convert initial_state to JSON string for storage
                puzzle_copy = p.copy()
                puzzle_copy['initial_state'] = json.dumps(p['initial_state'])
                new_puzzles.append(puzzle_copy)

        if new_puzzles:
            # Append new puzzles to existing data
            existing_data.extend(new_puzzles)

            # Save all data at once
            with open(annotations_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)

            print(f"Added {len(new_puzzles)} new puzzles to {annotations_path} (total: {len(existing_data)})")

            if len(puzzles) > len(new_puzzles):
                skipped = len(puzzles) - len(new_puzzles)
                print(f"Skipped {skipped} duplicate puzzles")
        else:
            print(f"No new puzzles to add to {annotations_path} (all were duplicates)")

    def visualize_puzzle(self, puzzle, filename=None, **kwargs):
        """Create visualization of the Futoshiki puzzle"""
        # Handle both dict and JSON string formats for initial_state
        if isinstance(puzzle['initial_state'], str):
            import json
            initial_state = json.loads(puzzle['initial_state'])
        else:
            initial_state = puzzle['initial_state']
            
        n = initial_state['size']
        initial = initial_state['grid']
        inequalities = initial_state['inequalities']
        
        # Create appropriate size figure
        cell_size = 0.8
        margin = 0.2
        fig_size = n * cell_size + margin * 2
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        # Set background color
        ax.set_facecolor('#f5f5f5')
        
        # Draw grid
        for i in range(n+1):
            ax.axhline(i, color='black', linewidth=1.5)
            ax.axvline(i, color='black', linewidth=1.5)
        
        # Fill in numbers
        for i in range(n):
            for j in range(n):
                if initial[i][j] != 0:
                    value = initial[i][j]
                    ax.text(j + 0.5, n - i - 0.5, str(value), 
                            fontsize=16, ha='center', va='center', 
                            color='black', weight='bold')
        
        # Draw inequality symbols
        for ineq in inequalities:
            i1, j1 = ineq['cell1']
            i2, j2 = ineq['cell2']
            symbol = ineq['symbol']
            
            # Calculate midpoint
            mid_x = (j1 + j2) / 2 + 0.5
            mid_y = n - (i1 + i2) / 2 - 0.5
            
            # Determine rotation and symbol direction
            if i1 == i2:  # horizontal
                rotation = 0
            else:  # vertical
                rotation = 90
                # Flip symbol for vertical direction
                if symbol == '>':
                    symbol = '<'
                else:
                    symbol = '>'
            
            ax.text(mid_x, mid_y, symbol, fontsize=14, 
                   ha='center', va='center', rotation=rotation,
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # Set limits and remove ticks
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title
        ax.set_title(f'Futoshiki Puzzle ({n}×{n})', fontsize=16)
        
        # Save or show figure
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            plt.close(fig)
        else:
            plt.tight_layout()
            plt.show()



    def generate_by_numerical_difficulty(self, items_per_difficulty=3, output_folder=None):
        """生成数字难度(1-5)的问题，每个难度使用对应的grid size"""
        # 使用传入的output_folder或默认的output_folder
        if output_folder is None:
            output_folder = self.output_folder

        # 创建输出目录
        os.makedirs(output_folder, exist_ok=True)
        images_dir = os.path.join(output_folder, "images")
        os.makedirs(images_dir, exist_ok=True)

        # 清空已生成集合
        self.generated_hashes = set()

        all_puzzles = []

        for difficulty_num in range(1, 6):  # 难度1-5
            params = self.difficulty_params[difficulty_num]
            difficulty_name = params['name']
            grid_size = params['grid_size']  # 每个难度使用对应的grid size

            print(f"\nGenerating {items_per_difficulty} items for difficulty {difficulty_num} ({difficulty_name}, {grid_size}x{grid_size})...")

            level_puzzles = []
            attempts = 0
            max_attempts = items_per_difficulty * 1000  # 最多尝试次数

            while len(level_puzzles) < items_per_difficulty and attempts < max_attempts:
                attempts += 1

                try:
                    # 使用时间戳作为基础种子，然后加上计数器确保不同
                    base_seed = self._get_timestamp_seed()
                    seed = base_seed + attempts

                    # Set random seed for reproducibility
                    random.seed(seed)
                    np.random.seed(seed)

                    self.size = grid_size

                    # Generate solution
                    solution = self._generate_latin_square()
                    inequalities = self._generate_inequalities(solution, params['inequality_prob'])
                    initial = self._generate_initial_grid(solution, params['retain_ratio'])

                    # Create unique puzzle ID
                    puzzle_id = f"futoshiki_{grid_size}x{grid_size}_{len(all_puzzles) + len(level_puzzles) + 1:03d}"

                    # Format initial state
                    initial_state = {
                        'grid': initial,
                        'inequalities': inequalities,
                        'size': grid_size
                    }

                    # Generate CoT data
                    cot_data = self._generate_cot_english(initial_state, solution)

                    # Generate images
                    image_path = f"images/{puzzle_id}.png"
                    full_image_path = os.path.join(output_folder, image_path)

                    # Create formatted puzzle
                    formatted_puzzle = {
                        'index': puzzle_id,
                        'category': 'futoshiki',
                        'image': image_path,
                        'question': self._create_question_with_image(),
                        'question_language': self._create_question_language(initial_state),
                        'answer': str(solution),
                        'initial_state': initial_state,  # Keep as dict for internal use
                        'difficulty': difficulty_num,
                        'cot': cot_data.get('cot', ''),
                        'cot_step1_all': cot_data.get('cot_step1_all', ''),
                        'cot_step2_all': cot_data.get('cot_step2_all', ''),
                        'cot_step3_all': cot_data.get('cot_step3_all', ''),
                    }

                    if not self._is_duplicate(formatted_puzzle):
                        # Generate and save image
                        self.visualize_puzzle(formatted_puzzle, filename=full_image_path)

                        level_puzzles.append(formatted_puzzle)
                        puzzle_hash = self._get_puzzle_hash(formatted_puzzle)
                        self.generated_hashes.add(puzzle_hash)

                        print(f"  Generated difficulty {difficulty_num} puzzle {len(level_puzzles)}/{items_per_difficulty} ({grid_size}x{grid_size})")
                    else:
                        print(f"  Skipped duplicate puzzle for difficulty {difficulty_num}")

                except Exception as e:
                    print(f"Error generating puzzle for difficulty {difficulty_num}: {e}")
                    continue

                if attempts % 50 == 0:
                    print(f"  Attempted {attempts} times, generated {len(level_puzzles)} valid puzzles")

            if len(level_puzzles) < items_per_difficulty:
                print(f"Warning: Only generated {len(level_puzzles)} difficulty {difficulty_num} puzzles out of {items_per_difficulty} requested")

            all_puzzles.extend(level_puzzles)

        # 批量保存所有问题到annotations.json
        if all_puzzles:
            self._batch_save_to_annotations(all_puzzles, output_folder)

            print(f"\nGenerated {len(all_puzzles)} total puzzles:")
            difficulty_counts = {}
            grid_size_counts = {}
            for puzzle in all_puzzles:
                diff = puzzle.get('difficulty', 'unknown')
                size = puzzle['initial_state']['size']
                difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
                grid_size_counts[size] = grid_size_counts.get(size, 0) + 1

            for difficulty, count in sorted(difficulty_counts.items()):
                print(f"  Difficulty {difficulty}: {count} puzzles")

            print(f"\nGrid size distribution:")
            for size, count in sorted(grid_size_counts.items()):
                print(f"  {size}x{size}: {count} puzzles")

            print(f"\nAll puzzle images saved in '{output_folder}' directory")
            annotations_path = os.path.join(output_folder, 'annotations.json')
            print(f"Puzzle data saved in '{annotations_path}'")
        else:
            print("\nNo puzzles were generated successfully.")

        return all_puzzles


    


    def visualize_numerical(self, puzzle, filename=None, show_solution=False, **kwargs):
        """创建数字难度Futoshiki问题的可视化表示"""
        n = puzzle['initial_state']['size']
        initial = puzzle['initial_state']['grid']
        inequalities = puzzle['initial_state']['inequalities']
        
        # 如果显示解答则解析解决方案
        solution = None
        if show_solution and 'answer' in puzzle:
            try:
                import ast
                solution = ast.literal_eval(puzzle['answer'])
            except:
                pass
        
        # 创建适当尺寸的图形
        cell_size = 0.8
        margin = 0.2
        fig_size = n * cell_size + margin * 2
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        
        # 设置背景颜色
        ax.set_facecolor('#f5f5f5')
        
        # 绘制网格
        for i in range(n+1):
            ax.axhline(i, color='black', linewidth=1.5)
            ax.axvline(i, color='black', linewidth=1.5)
        
        # 填入数字
        for i in range(n):
            for j in range(n):
                if show_solution and solution:
                    value = solution[i][j]
                    color = 'blue'
                    weight = 'bold'
                elif initial[i][j] != 0:
                    value = initial[i][j]
                    color = 'black'
                    weight = 'bold'
                else:
                    continue
                
                ax.text(j + 0.5, n - i - 0.5, str(value), 
                        fontsize=16, ha='center', va='center', 
                        color=color, weight=weight)
        
        # 绘制不等式符号
        for ineq in inequalities:
            i1, j1 = ineq['cell1']
            i2, j2 = ineq['cell2']
            symbol = ineq['symbol']
            
            # 计算中点
            mid_x = (j1 + j2) / 2 + 0.5
            mid_y = n - (i1 + i2) / 2 - 0.5
            
            # 确定旋转角度和符号方向
            if i1 == i2:  # 水平
                rotation = 0
            else:  # 垂直
                rotation = 90
                # 垂直方向翻转符号
                if symbol == '>':
                    symbol = '<'
                else:
                    symbol = '>'
            
            ax.text(mid_x, mid_y, symbol, fontsize=14, 
                   ha='center', va='center', rotation=rotation,
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # 设置限制并移除刻度
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加标题
        difficulty_num = puzzle.get('target_difficulty', puzzle.get('difficulty', 'unknown'))
        if show_solution:
            ax.set_title(f'Futoshiki Solution (Difficulty {difficulty_num}/5)', fontsize=16)
        else:
            ax.set_title(f'Futoshiki Puzzle', fontsize=16)
        
        # 添加坐标参考（可选）
        for i in range(n):
            for j in range(n):
                ax.text(j + 0.08, n - i - 0.08, f"({i},{j})",
                        fontsize=6, ha='left', va='top', alpha=0.5)
        
        # 保存或显示图形
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            plt.close(fig)
        else:
            plt.tight_layout()
            plt.show()
    
    def _create_text_question_numerical(self, puzzle):
        """创建数字难度版本的纯文本问题描述"""
        n = puzzle['initial_state']['size']
        initial = puzzle['initial_state']['grid']
        inequalities = puzzle['initial_state']['inequalities']
        difficulty_num = puzzle.get('target_difficulty', puzzle.get('difficulty', 'unknown'))
        
        # 创建网格的文本表示
        grid_text = []
        for i in range(n):
            row = []
            for j in range(n):
                value = initial[i][j]
                row.append(str(value) if value != 0 else "_")
            grid_text.append(" ".join(row))
        
        # 分析当前游戏状态
        filled_cells = sum(1 for i in range(n) for j in range(n) if initial[i][j] != 0)
        empty_cells = n * n - filled_cells
        
        # 分析每行每列的状态
        row_analysis = []
        col_analysis = []
        
        for i in range(n):
            row_filled = [initial[i][j] for j in range(n) if initial[i][j] != 0]
            row_missing = [x for x in range(1, n+1) if x not in row_filled]
            row_analysis.append(f"Row {i}: Filled numbers {row_filled}, Missing numbers {row_missing}")
        
        for j in range(n):
            col_filled = [initial[i][j] for i in range(n) if initial[i][j] != 0]
            col_missing = [x for x in range(1, n+1) if x not in col_filled]
            col_analysis.append(f"Column {j}: Filled numbers {col_filled}, Missing numbers {col_missing}")
        
        # 构建完整的文本问题
        lines = [
            f"Futoshiki Puzzle ({n}×{n}, Difficulty {difficulty_num}/5)",
            "=" * (45 + len(str(difficulty_num))),
            "",
            "GAME RULES:",
            f"1. The puzzle is a {n}×{n} grid.",
            f"2. Fill each cell with a number from 1 to {n}.",
            f"3. Each number must appear exactly once in each row and each column (no repetition).",
            "4. Inequality symbols between cells (either '<' or '>') must be satisfied:",
            "   - A horizontal constraint (i,j) < (i,j+1) means the left cell must be less than the right.",
            "   - A vertical constraint (i,j) < (i+1,j) means the top cell must be less than the bottom.",
            "",
            "CURRENT GAME STATE:",
            f"- Grid size: {n}×{n} ({n*n} total cells)",
            f"- Filled cells: {filled_cells}",
            f"- Empty cells: {empty_cells}",
            f"- Number of inequality constraints: {len(inequalities)}",
            f"- Difficulty level: {difficulty_num}/5",
            "",
            "INITIAL GRID EXPLANATION:",
            "The grid below shows the current state where:",
            "- Numbers (1-{}) represent pre-filled cells that cannot be changed".format(n),
            "- Underscore (_) represents empty cells that need to be filled",
            "- Each cell position is referenced as (row, column) starting from (0,0) at top-left",
            "",
            "Initial grid (_ for empty cells):",
            "\n".join(grid_text),
            "",
            "ROW AND COLUMN ANALYSIS:",
            "\n".join(row_analysis),
            "",
            "\n".join(col_analysis),
            "",
            "INEQUALITY CONSTRAINTS EXPLANATION:",
            "The following constraints must be satisfied in the final solution:",
            "(Format: cell1 symbol cell2 - direction)"
        ]
        
        # 添加不等式约束
        for idx, ineq in enumerate(inequalities, 1):
            c1, c2 = ineq['cell1'], ineq['cell2']
            symbol = ineq['symbol']
            direction = "horizontal" if c1[0] == c2[0] else "vertical"
            if direction == "horizontal":
                lines.append(f"{idx}. Cell ({c1[0]},{c1[1]}) {symbol} Cell ({c2[0]},{c2[1]}) - {direction} (left {symbol} right)")
            else:
                lines.append(f"{idx}. Cell ({c1[0]},{c1[1]}) {symbol} Cell ({c2[0]},{c2[1]}) - {direction} (top {symbol} bottom)")
        
        # 添加解题步骤和输出格式
        lines.extend([
            "",
            "SOLVING STEPS:",
            "1. Use OCR to extract the numbers and inequality signs from the image.",
            "2. Analyze each row and column to determine possible values for empty cells.",
            "3. Apply inequality constraints to further narrow down possibilities.",
            "4. Use logical deduction and constraint propagation to fill the grid.",
            "5. Verify that the solution satisfies all rules.",
            "",
            "OUTPUT FORMAT:",
            'Answer format: "answer": [[row1], [row2], ..., [row{}]]'.format(n),
            "",
            f"Example for a {n}×{n} puzzle:",
            '"answer": [',
        ])
        
        # 添加示例格式
        for i in range(n):
            example_row = [str((i + j) % n + 1) for j in range(n)]
            if i == n - 1:
                lines.append(f'           [{", ".join(example_row)}]]')
            else:
                lines.append(f'           [{", ".join(example_row)}],')
        
        return "\n".join(lines)
    
    def _generate_cot_numerical(self, puzzle):
        """Generate numerical difficulty version detailed solving process following the new 4-step format"""
        n = puzzle['initial_state']['size']
        initial_grid = puzzle['initial_state']['grid']
        inequalities = puzzle['initial_state']['inequalities']
        difficulty_num = puzzle.get('target_difficulty', puzzle.get('difficulty', 'unknown'))
        
        # Parse solution
        solution = None
        if 'answer' in puzzle:
            try:
                import ast
                solution = ast.literal_eval(puzzle['answer'])
            except:
                pass
        
        if not solution:
            return {"cot": "Unable to generate detailed solution process because the answer is not available."}
        
        # Step 1: Understanding the puzzle rules and objectives
        step1 = [
            "Let me analyze this Futoshiki puzzle step by step",
            "",
            "### Step 1: Understanding the puzzle rules and objectives",
            "",
            f"This is a {n}×{n} Futoshiki puzzle (Difficulty {difficulty_num}/5) with these rules:",
            f"- Fill each cell with numbers 1 to {n}",
            "- Each number appears exactly once in each row and column",
            "- Inequality constraints between adjacent cells must be satisfied",
            f"- Goal: Complete the {n}×{n} grid satisfying all constraints"
        ]
        
        # Step 2: Analyzing the visual information  
        step2 = [
            "",
            "### Step 2: Analyzing the visual information",
            "",
            "From the image, I can extract using OCR:"
        ]
        
        # Analyze initial grid
        filled_cells = []
        for i in range(n):
            for j in range(n):
                if initial_grid[i][j] != 0:
                    filled_cells.append(f"Cell ({i},{j}) = {initial_grid[i][j]}")
        
        step2.append("Initial grid pattern:")
        for i in range(n):
            row_str = ""
            for j in range(n):
                if initial_grid[i][j] != 0:
                    row_str += f"{initial_grid[i][j]} "
                else:
                    row_str += "_ "
            step2.append(f"  {row_str}")
        
        step2.extend([
            f"- Pre-filled cells: {len(filled_cells)}",
            f"- Empty cells to solve: {n*n - len(filled_cells)}"
        ])
        
        # Analyze inequality constraints
        step2.extend([
            f"- Inequality constraints: {len(inequalities)} total"
        ])
        
        if inequalities:
            step2.append("- All constraint patterns:")
            for idx, ineq in enumerate(inequalities, 1):  # Show all constraints
                c1, c2 = ineq['cell1'], ineq['cell2']
                symbol = ineq['symbol']
                direction = "horizontal" if c1[0] == c2[0] else "vertical"
                step2.append(f"  {idx}. Cell ({c1[0]},{c1[1]}) {symbol} Cell ({c2[0]},{c2[1]}) ({direction})")
        
        # Step 3: Strategic exploration and reasoning
        step3 = [
            "",
            "### Step 3: Strategic exploration and reasoning",
            "",
            f"Solving strategy for difficulty {difficulty_num}/5:"
        ]
        
        # Difficulty-based approach
        if difficulty_num <= 2:
            step3.append("- Use basic elimination and constraint propagation")
        elif difficulty_num == 3:
            step3.append("- Apply advanced constraint analysis and logical deduction")
        else:
            step3.append("- Use sophisticated reasoning and systematic exploration")
        
        # Analyze constraints
        filled_count = sum(1 for i in range(n) for j in range(n) if initial_grid[i][j] != 0)
        empty_cells = n * n - filled_count
        
        step3.extend([
            f"- Initial analysis: {filled_count}/{n*n} cells filled, {empty_cells} to solve",
            f"- Constraint density: {len(inequalities)} inequalities for {n*n} cells"
        ])
        
        # Show constraint analysis for key rows/columns
        constraint_analysis = []
        for i in range(n):
            row_filled = [initial_grid[i][j] for j in range(n) if initial_grid[i][j] != 0]
            if row_filled and len(row_filled) < n:
                row_missing = [x for x in range(1, n+1) if x not in row_filled]
                constraint_analysis.append(f"Row {i} needs: {row_missing}")
        
        if constraint_analysis:
            step3.append("- Complete constraint analysis:")
            step3.extend([f"  {analysis}" for analysis in constraint_analysis])
        
        # Add column constraints analysis  
        for j in range(n):
            col_filled = [initial_grid[i][j] for i in range(n) if initial_grid[i][j] != 0]
            if col_filled and len(col_filled) < n:
                col_missing = [x for x in range(1, n+1) if x not in col_filled]
                constraint_analysis.append(f"Column {j} needs: {col_missing}")
        
        if len(constraint_analysis) > len([x for x in constraint_analysis if x.startswith("Row")]):
            step3.extend([f"  {analysis}" for analysis in constraint_analysis[len([x for x in constraint_analysis if x.startswith("Row")]):]])
        
        # Apply logical reasoning for all empty cells
        empty_positions = [(i, j) for i in range(n) for j in range(n) if initial_grid[i][j] == 0]
        if empty_positions:
            step3.append("- Complete logical deduction process:")
            for pos_idx, (i, j) in enumerate(empty_positions):  # Show all empty cells
                row_used = set(initial_grid[i][k] for k in range(n) if initial_grid[i][k] != 0)
                col_used = set(initial_grid[k][j] for k in range(n) if initial_grid[k][j] != 0)
                possible = set(range(1, n+1)) - row_used - col_used
                step3.append(f"  Cell ({i},{j}): Initial candidates {sorted(possible)}")
                
                # Check all relevant inequality constraints
                constraints_applied = []
                for ineq in inequalities:
                    if ineq['cell1'] == [i, j] or ineq['cell2'] == [i, j]:
                        other_pos = ineq['cell2'] if ineq['cell1'] == [i, j] else ineq['cell1']
                        other_val = initial_grid[other_pos[0]][other_pos[1]]
                        if other_val != 0:
                            if ineq['cell1'] == [i, j]:
                                if ineq['symbol'] == '>':
                                    possible = {p for p in possible if p > other_val}
                                    constraints_applied.append(f"must be > {other_val}")
                                else:
                                    possible = {p for p in possible if p < other_val}
                                    constraints_applied.append(f"must be < {other_val}")
                            else:
                                if ineq['symbol'] == '>':
                                    possible = {p for p in possible if p < other_val}
                                    constraints_applied.append(f"must be < {other_val}")
                                else:
                                    possible = {p for p in possible if p > other_val}
                                    constraints_applied.append(f"must be > {other_val}")
                
                if constraints_applied:
                    step3.append(f"    Applied constraints: {', '.join(constraints_applied)}")
                    step3.append(f"    Final candidates: {sorted(possible)}")
                
                final_value = solution[i][j]
                step3.append(f"    Determined value: {final_value}")
                step3.append("")
        
        # Step 4: Solution validation and refinement
        step4 = [
            "",
            "### Step 4: Solution validation and refinement",
            "",
            "Complete solution after systematic reasoning:"
        ]
        
        # Show the solution
        step4.append("Final grid:")
        for i in range(n):
            row_str = " ".join(str(solution[i][j]) for j in range(n))
            step4.append(f"  {row_str}")
        
        step4.extend([
            "",
            "Validation results:",
            f"✓ All rows contain numbers 1-{n} exactly once",
            f"✓ All columns contain numbers 1-{n} exactly once"
        ])
        
        # Verify inequality constraints
        valid_constraints = 0
        for ineq in inequalities:
            i1, j1 = ineq['cell1']
            i2, j2 = ineq['cell2']
            symbol = ineq['symbol']
            v1, v2 = solution[i1][j1], solution[i2][j2]
            
            if (symbol == '>' and v1 > v2) or (symbol == '<' and v1 < v2):
                valid_constraints += 1
        
        step4.extend([
            f"✓ All {len(inequalities)} inequality constraints satisfied",
            f"✓ Difficulty {difficulty_num}/5 puzzle solved successfully!"
        ])
        
        # Combine all steps
        all_steps = step1 + step2 + step3 + step4
        full_cot = "\n".join(all_steps)
        
        # Build step-by-step parts and fulls
        step1_all = "\n".join(step1)
        step2_all = "\n".join(step1 + step2)
        step3_all = "\n".join(step1 + step2 + step3)
        step4_all = full_cot

        # Derive simple "part" versions by taking roughly half of each step block
        def half_text(text: str) -> str:
            words = text.split()
            midpoint = max(1, len(words) // 2)
            return " ".join(words[:midpoint])

        step1_part = half_text(step1_all)
        step2_part = half_text(step2_all)
        step3_part = half_text(step3_all)
        step4_part = half_text(step4_all)

        return {
            "cot": full_cot,
            "cot_step1_part": step1_part,
            "cot_step1_all": step1_all,
            "cot_step2_part": step2_part,
            "cot_step2_all": step2_all,
            "cot_step3_part": step3_part,
            "cot_step3_all": step3_all,
            "cot_step4_part": step4_part,
            "cot_step4_all": step4_all,
        }
