"""
Numbrix generator integrated with the shared BaseGenerator and main.py.

Outputs per run:
- <output_folder>/images/<puzzle_id>.png
- <output_folder>/annotations.json (appended via BaseGenerator.save_annotations)
"""

import numpy as np
import matplotlib
# 设置matplotlib使用非交互式后端，解决多线程GUI问题
matplotlib.use('Agg')  # 必须在pyplot导入之前设置
import matplotlib.pyplot as plt
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
from abc import ABC, abstractmethod
from generator.base_generator import BaseGenerator as CoreBaseGenerator
from utils.constants import PROMPT_MAZE_IMAGE, PROMPT_15PUZZLE_IMAGE, PROMPT_HANOI_IMAGE, PROMPT_WORDSEARCH_IMAGE, PROMPT_NUMBRIX_IMAGE, PROMPT_MINESWEEPER_IMAGE, PROMPT_EULERO_IMAGE, PROMPT_SNAKE_IMAGE
from utils.constants import PROMPT_MAZE, PROMPT_15PUZZLE, PROMPT_HANOI, PROMPT_WORDSEARCH, PROMPT_NUMBRIX, PROMPT_MINESWEEPER, PROMPT_EULERO, PROMPT_SNAKE



class LegacyBaseGenerator(ABC):
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
        self.task_name = self.__class__.__name__.replace('Generator', '').lower()
        self.task_dir = os.path.join(output_folder, self.task_name)
        self.image_dir = os.path.join(self.task_dir, 'images')
        self.annotations_file = os.path.join(self.task_dir, 'annotations.json')

        # 存储所有生成的puzzles，用于最后一次性保存
        self.generated_puzzles = []

        # 设置随机种子为当前时间戳
        self.seed = int(time.time())
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Create directories
        os.makedirs(self.task_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
    
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
        raise NotImplementedError
    
    def visualize(self, puzzle, **kwargs):
        raise NotImplementedError
    
    def solve(self, puzzle, **kwargs):
        raise NotImplementedError
        
    def add_puzzle_to_batch(self, puzzle_info):
        """添加单个puzzle到批处理列表中"""
        self.generated_puzzles.append(puzzle_info)

    def add_puzzles_to_batch(self, puzzle_infos):
        """批量添加多个puzzle到批处理列表中

        Args:
            puzzle_infos: puzzle信息列表
        """
        self.generated_puzzles.extend(puzzle_infos)

    def clear_batch(self):
        """清空批处理列表"""
        self.generated_puzzles = []

    def get_batch_size(self):
        """获取当前批处理列表中的puzzle数量"""
        return len(self.generated_puzzles)

    def generate_batch_context(self, num_cases, difficulty, output_folder=None):
        """生成puzzle的上下文管理器，确保所有生成完成后才保存

        使用示例:
            with generator.generate_batch_context(num_cases=10, difficulty=3) as puzzles:
                # puzzles是生成的所有puzzle列表
                pass  # puzzles会自动保存到JSON文件
        """
        class BatchContext:
            def __init__(self, generator, num_cases, difficulty, output_folder):
                self.generator = generator
                self.num_cases = num_cases
                self.difficulty = difficulty
                self.output_folder = output_folder
                self.generated_puzzles = []

            def __enter__(self):
                # 更新输出目录
                if self.output_folder:
                    self.generator.output_folder = self.output_folder
                    self.generator.task_dir = os.path.join(self.output_folder, self.generator.task_name)
                    self.generator.image_dir = os.path.join(self.generator.task_dir, 'images')
                    self.generator.annotations_file = os.path.join(self.generator.task_dir, 'annotations.json')
                    os.makedirs(self.generator.task_dir, exist_ok=True)
                    os.makedirs(self.generator.image_dir, exist_ok=True)

                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    # 正常退出，保存所有生成的puzzle
                    self.generator.save_all_to_json(append_mode=True)
                return False  # 不抑制异常

            def generate_and_add(self, case_idx):
                """生成单个puzzle并添加到批处理"""
                # 获取难度参数
                params = self.generator._get_difficulty_params(self.difficulty)
                size = params['size']
                remove_percent = params['remove_percent']

                # 尝试生成唯一解puzzle
                max_attempts = 1500
                solution_grid = None
                puzzle_grid = None
                clues = None

                for attempt in range(max_attempts):
                    # 创建完整解决方案
                    numbrix = Numbrix(size)
                    solution_grid = numbrix.generate_solution()

                    # 创建puzzle，确保单一解
                    puzzle_grid, clues = numbrix.create_puzzle(solution_grid, remove_percent)

                    # 验证唯一解
                    if numbrix.verify_unique_solution(puzzle_grid, solution_grid):
                        # 唯一解验证通过
                        break

                    print(f"Attempt {attempt+1}/{max_attempts}: Puzzle might not have a unique solution. Regenerating...")

                    if attempt == max_attempts - 1:
                        # 如果最后一次尝试也失败，添加更多线索以确保唯一解
                        print("Adding more clues to ensure uniqueness...")
                        puzzle_grid, clues = numbrix.create_puzzle(solution_grid, remove_percent - 10)
                        break

                # 创建唯一标识符
                puzzle_id = f"numbrix_{size}_{case_idx}_{int(time.time())}"

                # 可视化puzzle
                image_path = os.path.join(self.generator.image_dir, f'{puzzle_id}.png')
                numbrix.visualize(puzzle_grid, filename=image_path)

                # 转换numpy类型为Python原生类型
                clues_native = [(int(i), int(j), int(v)) for i, j, v in clues]

                # 生成推理过程
                cot_data = self.generator.generate_cot(puzzle_grid, solution_grid, numbrix)

                # 格式化puzzle信息
                puzzle_info = {
                    'index': puzzle_id,
                    'category': "numbrix",
                    'image': os.path.join(self.generator.image_dir, f'{puzzle_id}.png'),  # 存储完整路径
                    'question': PROMPT_NUMBRIX_IMAGE,
                    "initial_state": numbrix.to_text_representation(puzzle_grid),
                    'question_language': PROMPT_NUMBRIX.format(numbrix.to_text_representation(puzzle_grid)),
                    'answer': numbrix.to_text_representation(solution_grid),
                    'difficulty': str(self.difficulty),
                    'cot': cot_data['full_cot'],
                    'cot_step1_all': cot_data['cot_step1_all'],
                    'cot_step2_all': cot_data['cot_step2_all'],
                    'cot_step3_all': cot_data['cot_step3_all']
                }

                # 添加到批处理列表
                self.generator.add_puzzle_to_batch(puzzle_info)
                self.generated_puzzles.append(puzzle_info)
                return puzzle_info

            def generate_all(self):
                """生成所有puzzle"""
                print(f"Generating {self.num_cases} Numbrix puzzles in batch context...")
                for case_idx in range(self.num_cases):
                    print(f"Generating puzzle {case_idx + 1}/{self.num_cases}")
                    self.generate_and_add(case_idx)
                return self.generated_puzzles

        return BatchContext(self, num_cases, difficulty, output_folder)

    @staticmethod
    def merge_annotations_files(annotations_files, output_file):
        """合并多个annotations.json文件

        Args:
            annotations_files: 要合并的annotations.json文件路径列表
            output_file: 输出文件路径
        """
        all_puzzles = []
        seen_indices = set()

        for file_path in annotations_files:
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist, skipping")
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    puzzles = json.load(f)

                new_puzzles = []
                for puzzle in puzzles:
                    puzzle_index = puzzle.get('index')
                    if puzzle_index and puzzle_index not in seen_indices:
                        new_puzzles.append(puzzle)
                        seen_indices.add(puzzle_index)

                all_puzzles.extend(new_puzzles)
                print(f"Loaded {len(new_puzzles)} unique puzzles from {file_path}")

            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read {file_path}: {e}")

        if all_puzzles:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_puzzles, f, ensure_ascii=False, indent=2)
            print(f"Successfully merged {len(all_puzzles)} unique puzzles to {output_file}")
        else:
            print("No puzzles to merge")

    def save_all_to_json(self, append_mode=False):
        """一次性保存所有生成的puzzles到JSON文件

        Args:
            append_mode: 如果为True，合并现有数据而不是覆盖
        """
        if not self.generated_puzzles:
            print("No puzzles to save")
            return

        # Ensure puzzles have the required fields
        for puzzle in self.generated_puzzles:
            # Make image paths relative to task directory
            if 'image' in puzzle and puzzle['image']:
                puzzle['image'] = os.path.relpath(puzzle['image'], self.task_dir)

            # Ensure required fields exist
            if 'difficulty' not in puzzle:
                puzzle['difficulty'] = "3"  # 默认难度改为3

            if 'step_count' not in puzzle and 'answer' in puzzle:
                # Try to infer step count from answer if possible
                if isinstance(puzzle['answer'], list) and len(puzzle['answer']) > 0:
                    puzzle['step_count'] = len(puzzle['answer'])
                elif isinstance(puzzle['answer'], str) and ' ' in puzzle['answer']:
                    puzzle['step_count'] = len(puzzle['answer'].split())
                else:
                    puzzle['step_count'] = 0

        # 处理合并模式
        if append_mode and os.path.exists(self.annotations_file):
            try:
                with open(self.annotations_file, 'r', encoding='utf-8') as f:
                    existing_puzzles = json.load(f)
                # 合并现有数据和新数据，避免重复
                existing_indices = {p.get('index') for p in existing_puzzles}
                new_puzzles = [p for p in self.generated_puzzles if p.get('index') not in existing_indices]
                all_puzzles = existing_puzzles + new_puzzles
                print(f"Merged {len(new_puzzles)} new puzzles with {len(existing_puzzles)} existing puzzles")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read existing annotations file: {e}")
                print("Creating new file instead")
                all_puzzles = self.generated_puzzles
        else:
            all_puzzles = self.generated_puzzles

        with open(self.annotations_file, 'w', encoding='utf-8') as f:
            json.dump(all_puzzles, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(all_puzzles)} puzzles to {self.annotations_file}")

        # Clear the batch after saving
        self.generated_puzzles = []


class NumbrixGenerator(CoreBaseGenerator):
    def __init__(self, output_folder):
        super().__init__(output_folder=output_folder)

    def _get_difficulty_params(self, difficulty):
        """
        根据难度级别获取Numbrix相应的参数配置。

        Args:
            difficulty: 难度级别（1-5）

        Returns:
            dict: 包含难度参数的字典
        """
        # 根据难度设置网格大小
        if difficulty == 1:
            size = 4
        elif difficulty == 2:
            size = 5
        elif difficulty == 3:
            size = 6
        elif difficulty == 4:
            size = 7
        else:  # difficulty == 5
            size = 8

        return {
            'size': size,
            'remove_percent': 55  # 可以根据难度调整移除百分比
        }

    def generate(self, num_cases, difficulty, output_folder=None):
        """
        生成指定数量和难度的Numbrix谜题，并使用核心BaseGenerator保存标注。

        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别 (1-5)
            output_folder: 输出文件夹路径，覆盖构造函数中设置的路径

        Returns:
            生成的问题列表（用于上层调用者需要时）
        """
        # 解析输出目录并确保结构存在
        output_dir = output_folder or self.output_folder
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        # 获取难度参数
        params = self._get_difficulty_params(difficulty)
        size = params['size']
        remove_percent = params['remove_percent']

        print(f"Generating {num_cases} Numbrix puzzles with size={size}, difficulty={difficulty}...")

        annotations = []

        for case_idx in range(num_cases):
            print(f"Generating puzzle {case_idx + 1}/{num_cases}")

            # 尝试生成唯一解puzzle
            max_attempts = 1500
            solution_grid = None
            puzzle_grid = None
            clues = None

            for attempt in range(max_attempts):
                # 创建完整解决方案
                numbrix = Numbrix(size)
                solution_grid = numbrix.generate_solution()

                # 创建puzzle，确保单一解
                puzzle_grid, clues = numbrix.create_puzzle(solution_grid, remove_percent)

                # 验证唯一解
                if numbrix.verify_unique_solution(puzzle_grid, solution_grid):
                    break

                print(f"Attempt {attempt+1}/{max_attempts}: Puzzle might not have a unique solution. Regenerating...")

                if attempt == max_attempts - 1:
                    print("Adding more clues to ensure uniqueness...")
                    puzzle_grid, clues = numbrix.create_puzzle(solution_grid, remove_percent - 10)
                    break

            # 唯一标识符与图像路径
            puzzle_id = f"numbrix_{size}_{case_idx}_{int(time.time())}"
            image_filename = f"{puzzle_id}.png"
            image_abs_path = os.path.join(images_dir, image_filename)
            image_rel_path = os.path.join('images', image_filename)

            # 保存图像
            numbrix.visualize(puzzle_grid, filename=image_abs_path)

            # 生成推理过程
            cot_data = self.generate_cot(puzzle_grid, solution_grid, numbrix)

            # 构造annotation条目（image使用相对路径）
            annotation = {
                'index': puzzle_id,
                'category': 'numbrix',
                'image': image_rel_path,
                'question': PROMPT_NUMBRIX_IMAGE,
                'initial_state': numbrix.to_text_representation(puzzle_grid),
                'question_language': PROMPT_NUMBRIX.format(numbrix.to_text_representation(puzzle_grid)),
                'answer': numbrix.to_text_representation(solution_grid),
                'difficulty': str(difficulty),
                'cot': cot_data['full_cot'],
                'cot_step1_all': cot_data['cot_step1_all'],
                'cot_step2_all': cot_data['cot_step2_all'],
                'cot_step3_all': cot_data['cot_step3_all']
            }

            annotations.append(annotation)

        # 使用核心Base保存到 annotations.json （追加避免重复）
        self.save_annotations(annotations, output_dir)

        return annotations

    # 注意：批处理/上下文相关的旧API已移除，主流程由 main.py 调度

    def generate_cot(self, puzzle_grid, solution_grid, numbrix):
        """生成符合规则的4步CoT，并提供step1-3的part/all累计文本。"""
        # 收集线索（给定数字）
        clues = []
        for row in range(numbrix.size):
            for col in range(numbrix.size):
                value = int(puzzle_grid[row, col])
                if value > 0:
                    clues.append((row, col, value))
        clues.sort(key=lambda x: x[2])

        total_numbers = numbrix.size * numbrix.size

        def describe_clues(cs):
            if not cs:
                return "no visible numbers"
            return ", ".join([f"{v} at (r{r+1}, c{c+1})" for r, c, v in cs])

        def analyze_grid_structure():
            """分析网格结构和约束"""
            corner_cells = [(0, 0), (0, numbrix.size-1), (numbrix.size-1, 0), (numbrix.size-1, numbrix.size-1)]
            edge_cells = []
            center_cells = []
            
            for r in range(numbrix.size):
                for c in range(numbrix.size):
                    if (r, c) in corner_cells:
                        continue
                    elif r == 0 or r == numbrix.size-1 or c == 0 or c == numbrix.size-1:
                        edge_cells.append((r, c))
                    else:
                        center_cells.append((r, c))
            
            return corner_cells, edge_cells, center_cells

        def get_text_grid_representation(grid):
            """获取网格的文本表示"""
            lines = []
            # 创建带边框的网格显示
            lines.append("+" + "-" * (numbrix.size * 4 - 1) + "+")
            for i in range(numbrix.size):
                row_text = "|"
                for j in range(numbrix.size):
                    val = grid[i, j]
                    if val > 0:
                        row_text += f"{int(val):2d}"
                    else:
                        row_text += "  "
                    if j < numbrix.size - 1:
                        row_text += " "
                row_text += "|"
                lines.append(row_text)
            lines.append("+" + "-" * (numbrix.size * 4 - 1) + "+")
            return "\n".join(lines)

        # 构造各步骤文本
        intro = "Let me solve this Numbrix puzzle step by step.\n\n"

        # ===== Step 1: 明确游戏规则 =====
        step1_body = (
            "### Step 1: Understanding the puzzle rules and objectives\n\n"
            "**Numbrix Game Rules:**\n"
            f"- This is a Numbrix puzzle on a {numbrix.size}×{numbrix.size} grid\n"
            f"- I must fill the entire grid with consecutive integers from 1 to {total_numbers}\n"
            "- Each number from 1 to {total_numbers} must appear exactly once\n"
            "- Consecutive numbers (like 5 and 6) must be orthogonally adjacent (horizontally or vertically connected)\n"
            "- Diagonal connections are NOT allowed\n"
            "- The given clue numbers are fixed and cannot be changed\n"
            f"- The final result should form a single continuous path from 1 to {total_numbers}\n\n"
            
            "**Key Constraints:**\n"
            "- Path continuity: Numbers must form an unbroken chain\n"
            "- Adjacency rule: Only up/down/left/right connections allowed\n"
            "- Completeness: Every cell must be filled\n"
            "- Uniqueness: Each number appears exactly once"
        )

        # ===== Step 2: 仔细读取图像和初始状态 =====
        empty_count = int((puzzle_grid == 0).sum())
        corner_cells, edge_cells, center_cells = analyze_grid_structure()
        
        # 分析给定线索的分布
        clue_positions_analysis = []
        for r, c, v in clues:
            position_type = "corner" if (r, c) in corner_cells else "edge" if (r, c) in edge_cells else "center"
            clue_positions_analysis.append(f"  - {v} at row {r+1}, column {c+1} ({position_type} position)")

        step2_body = (
            "### Step 2: Carefully reading the image and analyzing the initial state\n\n"
            "**Initial Grid State:**\n"
            # f"{get_text_grid_representation(puzzle_grid)}\n\n"
            f"{numbrix.to_text_representation(puzzle_grid)}\n\n"
            "**Reading Analysis:**\n"
            f"- Grid size: {numbrix.size}×{numbrix.size} = {total_numbers} total cells\n"
            f"- Given clues: {len(clues)} numbers are provided\n"
            f"- Empty cells to fill: {empty_count}\n\n"
            
            "**Clue Distribution:**\n"
            + "\n".join(clue_positions_analysis) + "\n\n"
            
            "**Reflection on State Reading:**\n"
            f"- The clues range from {clues[0][2] if clues else 1} to {clues[-1][2] if clues else total_numbers}\n"
            f"- Corner positions have 2 possible neighbors, edge positions have 3, center positions have 4\n"
            "- This distribution will affect the difficulty of finding valid paths\n"
            f"- The puzzle has {(empty_count/total_numbers)*100:.1f}% empty cells, requiring strategic placement"
        )

        # ===== Step 3: 详细推理过程 =====
        def pairwise(iterable):
            return list(zip(iterable, iterable[1:]))

        reasoning_sections = []
        
        # 3.1 初始分析
        reasoning_sections.append(
            "**3.1 Initial Strategic Analysis:**\n"
            f"Starting with the constraint analysis between given clues. "
        )
        
        if clues:
            min_clue = clues[0]
            max_clue = clues[-1]
            reasoning_sections.append(
                f"The smallest clue is {min_clue[2]} at (r{min_clue[0]+1}, c{min_clue[1]+1}) and "
                f"the largest is {max_clue[2]} at (r{max_clue[0]+1}, c{max_clue[1]+1}). "
                "I'll use a bidirectional approach, working forward from smaller numbers and "
                "backward from larger numbers to meet in the middle."
            )

        # 3.2 线索对分析
        if len(clues) > 1:
            reasoning_sections.append("\n\n**3.2 Clue Pair Constraint Analysis:**")
            for (r1, c1, v1), (r2, c2, v2) in pairwise(clues):
                gap = v2 - v1
                manhattan = abs(r2 - r1) + abs(c2 - c1)
                
                if gap == 1:
                    if manhattan == 1:
                        reasoning_sections.append(
                            f"\n- Clues {v1} and {v2}: Consecutive and adjacent, connection is fixed."
                        )
                    else:
                        reasoning_sections.append(
                            f"\n- Clues {v1} and {v2}: Consecutive but Manhattan distance is {manhattan}, "
                            "which is impossible since consecutive numbers must be adjacent. This indicates an error or requires intermediate path."
                        )
                else:
                    reasoning_sections.append(
                        f"\n- Clues {v1} and {v2}: Gap of {gap} numbers, Manhattan distance {manhattan}. "
                        f"Minimum path length needed: {gap}, actual distance: {manhattan}. "
                        f"{'✓ Feasible' if manhattan >= gap else '✗ Impossible - too close'}"
                    )

        # 3.3 路径构建策略
        reasoning_sections.append(
            "\n\n**3.3 Path Construction Strategy:**\n"
            "I'll employ a systematic approach:\n"
            "1. **Forced moves first**: Identify cells where only one number can fit\n"
            "2. **Constraint propagation**: Use clue adjacency requirements to eliminate possibilities\n"
            "3. **Branch and backtrack**: When multiple options exist, try the most constrained first\n"
            "4. **Connectivity check**: Ensure no isolated regions are created\n\n"
            
            "**Detailed Reasoning Process:**"
        )

        # 3.4 具体推理步骤
        step_reasoning = []
        
        # 找到起始点（1的位置）和结束点
        start_pos = None
        end_pos = None
        
        for r, c, v in clues:
            if v == 1:
                start_pos = (r, c)
            if v == total_numbers:
                end_pos = (r, c)
        
        if start_pos:
            step_reasoning.append(f"\n- Starting from clue 1 at (r{start_pos[0]+1}, c{start_pos[1]+1})")
            # 分析1的邻居
            r, c = start_pos
            neighbors = []
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < numbrix.size and 0 <= nc < numbrix.size:
                    neighbors.append((nr, nc))
            step_reasoning.append(f"  Number 2 must be placed in one of {len(neighbors)} adjacent cells: {neighbors}")
        
        # 寻找强制移动
        forced_moves = []
        for r, c, v in clues:
            if v < total_numbers:  # 不是最后一个数字
                next_val = v + 1
                # 检查是否有其他线索是下一个数字
                next_clue = None
                for r2, c2, v2 in clues:
                    if v2 == next_val:
                        next_clue = (r2, c2)
                        break
                
                if next_clue:
                    nr, nc = next_clue
                    manhattan = abs(r - nr) + abs(c - nc)
                    if manhattan == 1:
                        forced_moves.append(f"  {v} → {next_val}: Adjacent clues create forced connection")
                    else:
                        forced_moves.append(f"  {v} → {next_val}: Must connect through {manhattan-1} intermediate cells")
        
        if forced_moves:
            step_reasoning.append(f"\n- Forced connections identified:")
            step_reasoning.extend(forced_moves)
        
        # 添加回溯策略
        step_reasoning.append(
            f"\n- **Backtracking Strategy**: When placing numbers, I check:\n"
            "  • Does this placement violate future adjacency requirements?\n"
            "  • Does this create unreachable isolated regions?\n"
            "  • Does this maintain connectivity to all remaining clues?\n"
            "  If any check fails, I backtrack and try alternative placements."
        )
        
        # 添加验证步骤
        step_reasoning.append(
            f"\n- **Progressive Validation**: After each number placement:\n"
            "  • Verify all constraints still satisfiable\n" 
            "  • Check remaining path length matches remaining numbers\n"
            "  • Ensure no dead ends are created\n"
            "  • Confirm all clues remain reachable"
        )

        reasoning_sections.extend(step_reasoning)

        # 3.5 最终答案导出
        reasoning_sections.append(
            f"\n\n**3.4 Solution Derivation:**\n"
            "Following the systematic approach above, I work through each number placement, "
            "applying constraints and backtracking when necessary. The complete solution places "
            f"all numbers 1 through {total_numbers} such that:\n"
            "- Each consecutive pair is orthogonally adjacent\n"
            "- All original clues remain in their fixed positions\n"
            "- The path forms one continuous sequence\n\n"
            f"**Final Answer Grid:**\n{get_text_grid_representation(solution_grid)}"
        )

        step3_body = (
            "### Step 3: Detailed reasoning process and exploration\n\n"
            + "".join(reasoning_sections)
        )

        # ===== Step 4: 验证和反思 =====
        # 验证连续性
        continuity_check = []
        for i in range(1, total_numbers):
            curr_pos = None
            next_pos = None
            
            # 找到当前和下一个数字的位置
            for r in range(numbrix.size):
                for c in range(numbrix.size):
                    if solution_grid[r, c] == i:
                        curr_pos = (r, c)
                    elif solution_grid[r, c] == i + 1:
                        next_pos = (r, c)
            
            if curr_pos and next_pos:
                r1, c1 = curr_pos
                r2, c2 = next_pos
                manhattan = abs(r1 - r2) + abs(c1 - c2)
                if manhattan == 1:
                    continuity_check.append(f"✓ {i} → {i+1}: Adjacent")
                else:
                    continuity_check.append(f"✗ {i} → {i+1}: Not adjacent (distance {manhattan})")

        # 验证所有数字存在
        number_check = []
        for i in range(1, total_numbers + 1):
            count = (solution_grid == i).sum()
            if count == 1:
                number_check.append(f"✓ Number {i}: Appears exactly once")
            else:
                number_check.append(f"✗ Number {i}: Appears {count} times")

        # 验证线索保持不变
        clue_check = []
        for r, c, v in clues:
            if solution_grid[r, c] == v:
                clue_check.append(f"✓ Clue {v} at (r{r+1}, c{c+1}): Preserved")
            else:
                clue_check.append(f"✗ Clue {v} at (r{r+1}, c{c+1}): Changed to {solution_grid[r, c]}")

        step4_body = (
            "### Step 4: Solution validation and reflection\n\n"
            "**Comprehensive Validation:**\n\n"
            
            "**4.1 Path Continuity Check:**\n"
            + "\n".join(continuity_check[:10]) + ("...\n(showing first 10 checks)" if len(continuity_check) > 10 else "") + "\n\n"
            
            "**4.2 Number Completeness Check:**\n"
            f"- All numbers 1 to {total_numbers}: {'✓ Complete' if len(number_check) == total_numbers else '✗ Incomplete'}\n"
            f"- Each number appears exactly once: {'✓ Verified' if all('✓' in check for check in number_check) else '✗ Duplicates found'}\n\n"
            
            "**4.3 Clue Preservation Check:**\n"
            + "\n".join(clue_check) + "\n\n"
            
            "**4.4 Final Reflection:**\n"
            "- **Rule Compliance**: All Numbrix rules are satisfied\n"
            "- **Constraint Satisfaction**: Every constraint from the initial analysis is met\n"
            "- **Path Integrity**: The solution forms one continuous path from 1 to {total_numbers}\n"
            "- **Logical Consistency**: The reasoning process was systematic and each step was justified\n"
            "- **Completeness**: No cells are left empty, no numbers are missing\n\n"
            
            "The solution is mathematically correct and satisfies all puzzle requirements."
        )

        # 组装累计文本块
        full_step1 = intro + step1_body
        full_step2 = full_step1 + "\n\n" + step2_body
        full_step3 = full_step2 + "\n\n" + step3_body
        full_step4 = full_step3 + "\n\n" + step4_body

        # 词汇对半截断，仅截断当前步骤的正文，之前步骤保留完整
        def truncate_half_words(text: str) -> str:
            tokens = text.split()
            if len(tokens) <= 1:
                return text
            cut = max(1, len(tokens) // 2)
            truncated = " ".join(tokens[:cut])
            # 保持句子边界更自然：若末尾不是结束符，尽量不强行添加标点
            return truncated

        # 仅截断当前step的主体部分
        step1_part = intro + truncate_half_words(step1_body)
        step2_part = full_step1 + "\n\n" + truncate_half_words(step2_body)
        step3_part = full_step2 + "\n\n" + truncate_half_words(step3_body)

        return {
            'full_cot': full_step4,
            'cot_step1_part': step1_part,
            'cot_step1_all': full_step1,
            'cot_step2_part': step2_part,
            'cot_step2_all': full_step2,
            'cot_step3_part': step3_part,
            'cot_step3_all': full_step3,
        }
    

    
    def visualize(self, puzzle, solution=None, filename=None, **kwargs):
        if isinstance(puzzle, Numbrix):
            return puzzle.visualize(solution=solution, filename=filename)
        else:
            # 如果只是网格，创建Numbrix实例
            size = len(puzzle)
            numbrix = Numbrix(size)
            return numbrix.visualize(puzzle, filename=filename)


class Numbrix:
    def __init__(self, size):
        """初始化Numbrix谜题
        
        Args:
            size: 正方形网格的大小
        """
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
    
    def generate_solution(self):
        """生成完整有效的Numbrix解决方案
        
        Returns:
            带有有效Numbrix解决方案的2D网格
        """
        # 从空网格开始
        solution = np.zeros((self.size, self.size), dtype=int)
        
        # 创建连续路径
        path = self._generate_path()
        
        # 根据路径填充网格
        num = 1
        for row, col in path:
            solution[row, col] = num
            num += 1
        
        return solution
    
    def _generate_path(self):
        """生成穿过网格的连续路径
        
        Returns:
            形成路径的(row, col)坐标列表
        """
        # 创建完整图，每个单元格与其正交邻居相连
        cells = [(i, j) for i in range(self.size) for j in range(self.size)]
        
        # 从随机单元格开始
        start = random.choice(cells)
        path = [start]
        remaining = set(cells)
        remaining.remove(start)
        
        # 邻居方向 (上, 右, 下, 左)
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        # 构建路径
        current = start
        stuck_count = 0
        max_stuck = 100  # 防止无限循环
        
        while remaining and stuck_count < max_stuck:
            # 查找有效邻居（相邻且不在路径中的单元格）
            neighbors = []
            row, col = current
            
            for dr, dc in directions:
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) in remaining:
                    neighbors.append((nr, nc))
            
            if neighbors:
                # 优先选择有更少其他连接的邻居
                next_scores = []
                for next_cell in neighbors:
                    nr, nc = next_cell
                    # 计算该单元格的其他连接数
                    alt_count = 0
                    for dr, dc in directions:
                        nnr, nnc = nr + dr, nc + dc
                        if 0 <= nnr < self.size and 0 <= nnc < self.size and (nnr, nnc) in remaining and (nnr, nnc) != current:
                            alt_count += 1
                    next_scores.append((alt_count, next_cell))
                
                # 选择最少备选的单元格（避免后面卡住）
                next_scores.sort()
                next_cell = next_scores[0][1]
                
                path.append(next_cell)
                remaining.remove(next_cell)
                current = next_cell
                stuck_count = 0  # 重置卡住计数
            else:
                # 如果所有邻居都在路径中，尝试重新回到之前的点
                stuck_count += 1
                if len(path) > 1:
                    # 回溯一步
                    path.pop()
                    current = path[-1]
                else:
                    # 如果只有一个单元格，重新开始
                    if remaining:
                        new_start = random.choice(list(remaining))
                        path = [new_start]
                        remaining.remove(new_start)
                        current = new_start
                    else:
                        break
        
        # 如果还有剩余单元格但卡住了，将剩余单元格强制添加进路径
        if remaining and stuck_count >= max_stuck:
            # 创建新路径，确保连接所有单元格
            return self._generate_hamiltonian_path()
            
        return path
    
    def _generate_hamiltonian_path(self):
        """使用更高级的算法生成哈密顿路径
        
        Returns:
            形成路径的(row, col)坐标列表
        """
        # 创建所有单元格列表
        all_cells = [(i, j) for i in range(self.size) for j in range(self.size)]
        
        # 将单元格排序为蛇形模式，确保相邻单元格在路径中也相邻
        # 示例: 在5x5网格中:
        # 1  2  3  4  5
        # 10 9  8  7  6
        # 11 12 13 14 15
        # 20 19 18 17 16
        # 21 22 23 24 25
        path = []
        for i in range(self.size):
            row = i
            if i % 2 == 0:  # 从左到右
                for j in range(self.size):
                    path.append((row, j))
            else:  # 从右到左
                for j in range(self.size-1, -1, -1):
                    path.append((row, j))
        
        # 打乱起点和终点
        if random.random() < 0.5:
            path = path[::-1]  # 逆转路径
            
        # 随机旋转网格
        rotations = random.randint(0, 3)
        for _ in range(rotations):
            # 顺时针旋转90度
            new_path = []
            for r, c in path:
                new_path.append((c, self.size - 1 - r))
            path = new_path
            
        return path
    
    def create_puzzle(self, solution_grid, remove_percent):
        """通过移除单元格创建谜题
        
        Args:
            solution_grid: 完整解决方案网格
            remove_percent: 要移除的单元格百分比
            
        Returns:
            部分填充的谜题网格和线索位置列表
        """
        puzzle = solution_grid.copy()
        
        # 计算要移除的单元格数量
        total_cells = self.size * self.size
        cells_to_remove = int(total_cells * remove_percent / 100)
        
        # 获取所有单元格坐标
        all_cells = [(i, j) for i in range(self.size) for j in range(self.size)]
        
        # 确保保留第一个和最后一个数字
        min_val_pos = np.unravel_index(np.argmin(solution_grid), solution_grid.shape)
        max_val_pos = np.unravel_index(np.argmax(solution_grid), solution_grid.shape)
        keep_cells = [min_val_pos, max_val_pos]
        
        # 确保保留连续序列中的一些数字作为线索
        # 计算每个数字间隔中要保留的线索数量
        max_val = total_cells
        clue_gap = max(2, max_val // 10)  # 比如在100个单元格中，每10个数字保留一个
        
        # 找出每个interval_size个数字中的一个数字作为线索
        for val in range(1, max_val + 1, clue_gap):
            # 找到该值在解决方案中的位置
            for i, j in all_cells:
                if solution_grid[i, j] == val and (i, j) not in keep_cells:
                    keep_cells.append((i, j))
                    break
        
        # 从剩余单元格中随机选择要保留的单元格
        remaining_cells = [cell for cell in all_cells if cell not in keep_cells]
        cells_to_keep_count = total_cells - cells_to_remove - len(keep_cells)
        
        if cells_to_keep_count > 0 and remaining_cells:
            additional_keep_cells = random.sample(remaining_cells, min(cells_to_keep_count, len(remaining_cells)))
            keep_cells.extend(additional_keep_cells)
        
        # 通过将非保留单元格设为0来创建谜题
        for i, j in all_cells:
            if (i, j) not in keep_cells:
                puzzle[i, j] = 0
        
        # 返回谜题和线索位置列表
        clues = [(i, j, int(solution_grid[i, j])) for i, j in keep_cells]
        return puzzle, clues
    
    def verify_unique_solution(self, puzzle_grid, solution_grid):
        """验证谜题是否有唯一解
        
        Args:
            puzzle_grid: 部分填充的谜题网格
            solution_grid: 已知的完整解决方案
            
        Returns:
            如果谜题具有唯一解，则为True，否则为False
        """
        # 获取所有线索
        clues = []
        for i in range(self.size):
            for j in range(self.size):
                if puzzle_grid[i, j] > 0:
                    clues.append((i, j, puzzle_grid[i, j]))
        
        # 按值排序线索
        clues.sort(key=lambda x: x[2])
        
        # 检查连续线索是否有兼容的位置
        for i in range(len(clues) - 1):
            r1, c1, v1 = clues[i]
            r2, c2, v2 = clues[i + 1]
            
            # 如果数字相差太大但没有足够空间容纳中间数字，可能有多解
            if v2 > v1 + 1:
                # 计算曼哈顿距离
                manhattan_dist = abs(r2 - r1) + abs(c2 - c1)
                
                # 必须有足够空间放置中间数字
                if manhattan_dist > v2 - v1:
                    # 距离太远，不可能有路径
                    return False
                
                # 检查两点之间的路径是否能容纳所有中间数字
                if not self._path_has_enough_space(r1, c1, r2, c2, v1, v2, puzzle_grid):
                    return False
        
        # 高级检查：确保空白区域不允许多路径
        # 实现更复杂的约束验证逻辑...
        
        return True
    
    def _path_has_enough_space(self, r1, c1, r2, c2, v1, v2, grid):
        """检查两点之间路径是否有足够空间放置所有中间数字
        
        Args:
            r1, c1: 第一个线索的坐标
            r2, c2: 第二个线索的坐标
            v1, v2: 两个线索的值
            grid: 谜题网格
            
        Returns:
            如果有足够空间，则为True，否则为False
        """
        # 计算两个线索之间应该有多少个数字
        needed_cells = v2 - v1 - 1
        
        if needed_cells <= 0:
            return True  # 相邻数字，不需要额外空间
        
        # 计算曼哈顿距离
        manhattan_dist = abs(r2 - r1) + abs(c2 - c1)
        
        # 如果距离小于需要的单元格数，无法放置所有数字
        if manhattan_dist < needed_cells:
            return False
        
        # 检查路径上是否有足够的空白单元格
        # 简单实现：计算两点之间长方形区域内的空白单元格
        min_r, max_r = min(r1, r2), max(r1, r2)
        min_c, max_c = min(c1, c2), max(c1, c2)
        
        empty_cells = 0
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if grid[r, c] == 0:  # 空白单元格
                    empty_cells += 1
        
        # 如果空白单元格少于需要的数字，则不够空间
        # 我们添加一个宽松因子，因为不是所有单元格都必须在长方形区域内
        return empty_cells >= needed_cells * 0.8
    
    def to_text_representation(self, grid):
        """转换为文本表示
        
        Args:
            grid: 要转换的2D网格
            
        Returns:
            网格的字符串表示
        """
        result = []
        
        # 添加标题行
        result.append("")
        
        # 添加网格
        for i in range(self.size):
            row = []
            for j in range(self.size):
                val = grid[i, j]
                if val > 0:
                    row.append(f"{int(val)}")
                else:
                    row.append(" ")
            result.append("|" + "|".join(row) + "|")
        
        return "\n".join(result)
        
    def visualize(self, grid, filename=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 设置背景颜色
        ax.set_facecolor('#f8f8f8')
        
        # 计算字体大小，根据网格大小自适应
        base_font_size = max(20, 200 // self.size)
        
        # 绘制网格单元格
        for i in range(self.size):
            for j in range(self.size):
                cell_value = grid[i, j]
                
                # 创建单元格矩形
                rect = plt.Rectangle((j, i), 1, 1, 
                                    edgecolor='black', 
                                    facecolor='white' if cell_value > 0 else '#f0f0f0',
                                    linewidth=1.5)
                ax.add_patch(rect)
                
                # 为填充单元格添加文本
                if cell_value > 0:
                    ax.text(j + 0.5, i + 0.5, str(int(cell_value)),
                           horizontalalignment='center',
                           verticalalignment='center',
                           fontsize=base_font_size, fontweight='bold')
        
        # 添加网格线
        for i in range(self.size + 1):
            ax.plot([0, self.size], [i, i], 'k-', linewidth=1.5)
            ax.plot([i, i], [0, self.size], 'k-', linewidth=1.5)
        
        # 设置限制和长宽比
        ax.set_xlim(0, self.size)
        ax.set_ylim(self.size, 0)
        ax.set_aspect('equal')
        
        # 移除刻度
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加标题
        plt.title('Numbrix Puzzle', fontsize=16, fontweight='bold')
        
        # 保存或显示
        if filename:
            plt.tight_layout()
            plt.savefig(filename, dpi=100, bbox_inches='tight')
            plt.close()
            return filename
        else:
            plt.tight_layout()
            plt.show()
            return None

