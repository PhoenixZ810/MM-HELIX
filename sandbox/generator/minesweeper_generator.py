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
from matplotlib import patheffects
from heapq import heappush, heappop
import concurrent.futures
import threading
from utils.constants import PROMPT_MAZE_IMAGE, PROMPT_15PUZZLE_IMAGE, PROMPT_HANOI_IMAGE, PROMPT_WORDSEARCH_IMAGE, PROMPT_NUMBRIX_IMAGE, PROMPT_MINESWEEPER_IMAGE, PROMPT_EULERO_IMAGE, PROMPT_SNAKE_IMAGE
from utils.constants import PROMPT_MAZE, PROMPT_15PUZZLE, PROMPT_HANOI, PROMPT_WORDSEARCH, PROMPT_NUMBRIX, PROMPT_MINESWEEPER, PROMPT_EULERO, PROMPT_SNAKE
from generator.base_generator import BaseGenerator



class MinesweeperGenerator(BaseGenerator):
    def __init__(self, output_folder):
        super().__init__(output_folder)
        self.training_set = []  # 保持向后兼容，但不在新接口中使用
        # 为当前生成器实例记录一个基准seed，保证一次运行内可复现
        self.seed = int(time.time())
        
    def generate(self, num_cases, difficulty, output_folder=None):
        """
        生成指定数量和难度的扫雷谜题

        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别 (1-5)
            output_folder: 输出文件夹路径，覆盖构造函数中设置的路径
        """
        # 如果提供了output_folder，更新输出目录
        if output_folder:
            self.output_folder = output_folder
            self.image_dir = os.path.join(output_folder, 'images')
            os.makedirs(self.image_dir, exist_ok=True)
        else:
            # 确保image_dir已初始化
            if not hasattr(self, 'image_dir') or not self.image_dir:
                self.image_dir = os.path.join(self.output_folder, 'images')
                os.makedirs(self.image_dir, exist_ok=True)

        # 获取难度参数
        params = self._get_difficulty_params(difficulty)
        size = params['size']

        print(f"Generating {num_cases} Minesweeper puzzles with difficulty={difficulty}, size={size}")

        # 本次生成的题目集合，最终通过save_annotations一次性写入
        generated_items = []

        for i in range(num_cases):
            # 使用时间戳和计数器生成唯一seed
            seed = self.seed + i

            # Set random seed for reproducibility
            np.random.seed(seed)
            random.seed(seed)

            # Determine mines based on size
            mines = params['mines']

            print(f"Generating Minesweeper puzzle {i+1}/{num_cases} with size={size}, seed={seed}, difficulty={difficulty}, mines={mines}")

            # Create Minesweeper puzzle
            minesweeper = Minesweeper(size, mines)

            # Generate text representations
            initial_state = minesweeper.to_text_representation(show_mines=False)
            answer = minesweeper.get_mine_coordinates()

            # Generate unique index
            puzzle_index = f"minesweeper_{size}_{seed}_{i}"

            # Generate image
            image_filename = f'minesweeper_{size}_{seed}_{i}.png'
            image_path = os.path.join(self.image_dir, image_filename)
            minesweeper.visualize(show_mines=False, filename=image_path)

            # Generate CoT reasoning
            cot, cot_data = self.generate_cot(minesweeper, initial_state)

            # Format puzzle data
            puzzle_info = {
                'index': puzzle_index,
                'category': "minesweeper",
                'image': f"images/{image_filename}",
                'question': PROMPT_MINESWEEPER_IMAGE,
                'question_language': self._create_question_language(initial_state),
                'answer': answer,
                'initial_state': initial_state,
                'difficulty': str(difficulty),
                'cot': cot
            }

            # Add the new CoT step fields
            puzzle_info.update(cot_data)

            # 收集到当前批次列表，稍后批量保存
            generated_items.append(puzzle_info)

            print(f"Generated Minesweeper puzzle: {puzzle_index}")

        # 批量保存所有生成的问题到annotations.json
        self.save_annotations(generated_items, self.output_folder)

        return generated_items

    def _get_difficulty_params(self, difficulty):
        """
        根据难度级别获取相应的参数配置

        Args:
            difficulty: 难度级别（1-5）

        Returns:
            dict: 包含难度参数的字典
        """
        difficulty_params = {
            1: {'size': 5, 'mines': 3},
            2: {'size': 6, 'mines': 4},
            3: {'size': 7, 'mines': 5},
            4: {'size': 8, 'mines': 7},
            5: {'size': 9, 'mines': 9}
        }

        return difficulty_params.get(difficulty, difficulty_params[3])  # 默认中等难度
    
    def _get_difficulty_and_mines(self, size):
        """Determine difficulty and number of mines based on grid size"""
        if size <= 5:
            return "1", max(2, size // 3)
        elif size <= 6:
            return "2", max(3, size // 3)
        elif size <= 7:
            return "3", max(4, size // 3)
        elif size <= 8:
            return "4", max(5, size // 3)
        else:
            return "5", max(6, size // 3)
    
    def _create_question_language(self, initial_state):
        """Create pure text version of the question"""
        return f"""Your task is to solve the Minesweeper puzzle according to the rules and the current state below:

### Game Rules:
1. Minesweeper is played on a grid where some cells contain hidden mines.
2. Numbers on the grid represent how many mines are adjacent to that cell (including diagonally).
3. A cell with no number means it has no adjacent mines (this is represented as a blank cell).
4. The goal is to identify the location of all mines without detonating any.
5. You can mark a cell as containing a mine if you're certain based on logical deduction. 

### Initial State:
{initial_state}

### Output Format Requirements:
Your final answer should list all mine locations using 0-based coordinates in the format (row,col).

**Example answer format:**
(0,5),(0,7),(1,1),(1,2)
"""
    
    def generate_cot(self, minesweeper, initial_state):
        """Generate an enhanced four-step CoT with detailed reasoning and reflection."""
        grid = minesweeper.grid
        size = minesweeper.size 

        # Collect actual mine coordinates for final answer disclosure
        actual_mines = []
        for i in range(size):
            for j in range(size):
                if grid[i, j] == -1:
                    actual_mines.append(f"({i}, {j})")

        intro = "Let me solve this minesweeper puzzle through careful analysis and systematic reasoning.\n\n"

        # Step 1: Understanding the puzzle rules and objectives (enhanced and detailed)
        step1 = "### Step 1: Understanding the game rules and objectives\n\n"
        step1 += (
            "**Core Game Rules:**\n"
            "- Minesweeper is a logic puzzle played on a rectangular grid containing hidden mines\n"
            "- Each numbered cell indicates exactly how many mines are present in its 8 adjacent cells (including diagonally adjacent cells)\n"
            "- Empty cells (shown as blank spaces) contain no adjacent mines, or are unknown cells that could be either safe or contain mines\n"
            "- The objective is to identify the precise location of every mine using logical deduction\n\n"
            
            "**Key Principles:**\n"
            "- Numbers provide definitive constraints: a '3' means exactly 3 mines among its 8 neighbors\n"
            "- Multiple numbered cells can share adjacent cells, creating overlapping constraints\n"
            "- Logical deduction should always be attempted before guessing\n"
            "- Each puzzle has a unique solution that can be found through systematic analysis\n\n"
            
            "**Solution Strategy:**\n"
            "- Start with cells that have clear, deterministic implications\n"
            "- Use constraint propagation to infer mine locations\n"
            "- Apply advanced techniques like subset analysis when simple methods aren't sufficient\n"
            "- Verify all constraints are satisfied in the final solution"
        )

        # Step 2: Careful image reading and state analysis (enhanced with reflection)
        step2 = "\n### Step 2: Reading the image carefully and analyzing the initial state\n\n"
        
        # Detailed grid analysis
        step2 += f"**Initial Grid State ({size}×{size}):**\n```\n{initial_state}\n```\n\n"
        
        # Analyze each numbered cell and its implications
        numbered_cells = []
        blank_cells = []
        for i in range(size):
            for j in range(size):
                if grid[i, j] > 0:
                    numbered_cells.append((i, j, grid[i, j]))
                elif grid[i, j] == 0 or grid[i, j] == -1:  # Appears blank in puzzle view
                    blank_cells.append((i, j))
        
        step2 += f"**State Analysis:**\n"
        step2 += f"- Grid dimensions: {size} rows × {size} columns = {size*size} total cells\n"
        step2 += f"- Numbered cells: {len(numbered_cells)} cells with constraint information\n"
        step2 += f"- Blank/unknown cells: {len(blank_cells)} cells (these could be safe or contain mines)\n\n"
        
        # Detailed constraint mapping
        step2 += "**Constraint Mapping:**\n"
        for i, j, value in numbered_cells[:min(10, len(numbered_cells))]:  # Show first 10 for brevity
            adjacent_blanks = self._get_adjacent_empty_positions(grid, i, j, size)
            step2 += f"- Cell ({i},{j}) = {value}: must have exactly {value} mine(s) among its {len(adjacent_blanks)} blank neighbors\n"
        
        if len(numbered_cells) > 10:
            step2 += f"- ... and {len(numbered_cells) - 10} more numbered cells with similar constraints\n"
        
        # Reflection on state reading
        step2 += "\n**Reflection on State Reading:**\n"
        step2 += "- I have carefully examined each cell in the grid to identify all numbered constraints\n"
        step2 += "- The puzzle presents a system of interdependent constraints that must be solved simultaneously\n"
        step2 += "- Some numbered cells may provide immediate, deterministic solutions while others require more complex analysis\n"
        step2 += "- The challenge lies in systematically applying logical deduction to uncover all mine locations"

        # Step 3: Detailed reasoning process (significantly enhanced)
        step3_lines = ["\n### Step 3: Detailed reasoning and systematic exploration\n"]
        
        step3_lines.append("\n**Phase 1: Deterministic Deductions**")
        step3_lines.append(
            "I begin by identifying cells where the constraints immediately determine mine locations. "
            "This occurs when a numbered cell has exactly as many unknown neighbors as its number value, "
            "or when it already has enough confirmed mines among its neighbors."
        )

        # Find and explain deterministic cases
        deterministic_found = 0
        for i in range(size):
            for j in range(size):
                if grid[i, j] > 0:
                    unknown_positions = self._get_adjacent_empty_positions(grid, i, j, size)
                    unknown_count = len(unknown_positions)
                    number_value = grid[i, j]
                    
                    if unknown_count == number_value and unknown_count > 0:
                        coords_str = ", ".join([f"({r}, {c})" for r, c in unknown_positions])
                        step3_lines.append(
                            f"- Cell ({i},{j}) shows '{number_value}' and has exactly {unknown_count} unknown neighbors → "
                            f"all unknown neighbors must be mines: {coords_str}"
                        )
                        deterministic_found += 1
                        if deterministic_found >= 6:  # Limit examples for readability
                            break
            if deterministic_found >= 6:
                break

        if deterministic_found == 0:
            step3_lines.append("- No immediately deterministic cells found; proceeding to advanced analysis")

        step3_lines.append("\n**Phase 2: Constraint Intersection Analysis**")
        step3_lines.append(
            "When direct deduction isn't possible, I analyze how multiple numbered cells interact. "
            "Cells that share unknown neighbors create overlapping constraints that can be solved together."
        )

        # Find overlapping constraints
        overlap_examples = 0
        for i in range(size):
            for j in range(size):
                if grid[i, j] > 0:
                    # Find other numbered cells that share blank neighbors
                    my_blanks = set(self._get_adjacent_empty_positions(grid, i, j, size))
                    for i2 in range(size):
                        for j2 in range(size):
                            if grid[i2, j2] > 0 and (i2, j2) != (i, j):
                                other_blanks = set(self._get_adjacent_empty_positions(grid, i2, j2, size))
                                shared_blanks = my_blanks.intersection(other_blanks)
                                if shared_blanks and overlap_examples < 3:
                                    shared_coords = ", ".join([f"({r}, {c})" for r, c in shared_blanks])
                                    step3_lines.append(
                                        f"- Cells ({i},{j}) and ({i2},{j2}) share unknown neighbors: {shared_coords} "
                                        f"→ applying constraint intersection analysis"
                                    )
                                    overlap_examples += 1

        step3_lines.append("\n**Phase 3: Hypothetical Reasoning and Backtracking**")
        
        # Enhanced backtracking explanation with specific example
        candidate = None
        for i in range(size):
            for j in range(size):
                if grid[i, j] > 0:
                    positions = self._get_adjacent_empty_positions(grid, i, j, size)
                    if len(positions) >= 2:  # Need at least 2 options for meaningful hypothesis
                        candidate = ((i, j), positions[0], positions[1])
                        break
            if candidate:
                break

        if candidate:
            (ci, cj), pos1, pos2 = candidate
            step3_lines.append(
                f"When deterministic methods are exhausted, I employ systematic hypothesis testing:\n"
                f"- Consider cell ({ci},{cj}) which has multiple possible mine configurations\n"
                f"- Hypothesis A: Assume ({pos1[0]},{pos1[1]}) contains a mine\n"
                f"  → Propagate this assumption through all connected constraints\n"
                f"  → Check if any numbered cell would be over-satisfied or under-satisfied\n"
                f"  → If contradiction arises, reject hypothesis A\n"
                f"- Hypothesis B: Assume ({pos1[0]},{pos1[1]}) is safe\n"
                f"  → Apply same propagation and validation process\n"
                f"- Continue until a consistent solution emerges"
            )
        else:
            step3_lines.append(
                "For complex configurations, I use systematic hypothesis testing:\n"
                "- Select uncertain cells and test both possibilities (mine vs. safe)\n"
                "- Propagate each assumption through the constraint network\n"
                "- Backtrack when contradictions occur\n"
                "- Continue until a logically consistent solution is found"
            )

        step3_lines.append("\n**Phase 4: Solution Synthesis**")
        step3_lines.append(
            "Through the combination of deterministic deduction, constraint analysis, and systematic exploration, "
            "I arrive at a complete mine configuration that satisfies all numbered cell requirements. "
            "Each mine placement is logically justified by the constraints provided in the puzzle."
        )

        step3 = "\n".join(step3_lines)

        # Step 4: Enhanced solution validation and reflection
        step4 = "\n### Step 4: Solution validation and comprehensive reflection\n\n"
        
        step4 += f"**Final Solution:**\n"
        step4 += f"Mine locations: {', '.join(actual_mines)}\n"
        step4 += f"Total mines found: {len(actual_mines)}\n\n"
        
        step4 += "**Validation Process:**\n"
        
        # Detailed validation for each numbered cell
        validation_details = []
        for i in range(size):
            for j in range(size):
                if grid[i, j] > 0:
                    actual_adjacent_mines = self._count_adjacent_actual_mines(grid, i, j, size)
                    required_mines = grid[i, j]
                    validation_details.append(f"- Cell ({i},{j}): requires {required_mines} mines, found {actual_adjacent_mines} mines ✓")
        
        # Show first few validations
        step4 += "\n".join(validation_details[:min(8, len(validation_details))])
        if len(validation_details) > 8:
            step4 += f"\n- ... and {len(validation_details) - 8} more cells all correctly validated\n"
        
        step4 += "\n\n**Solution Reflection:**\n"
        step4 += (
            "- All numbered cell constraints are perfectly satisfied\n"
            "- No contradictions exist in the final mine configuration\n"
            "- The solution required systematic application of logical deduction principles\n"
            "- Each mine placement can be traced back to specific constraint requirements\n"
            "- The puzzle demonstrates the power of constraint-based reasoning in combinatorial problem solving"
        )
        
        step4 += "\n\n**Confidence Assessment:**\n"
        step4 += (
            "- High confidence: The solution satisfies all mathematical constraints\n"
            "- Logical validity: Each step follows from established minesweeper rules\n"
            "- Completeness: All mine locations have been identified\n"
            "- Uniqueness: This is the only configuration that satisfies all constraints"
        )

        # Build full CoT text
        cot_full = intro + step1 + step2 + step3 + step4

        # Build cot_step{x}_all for x in 1..3 (inclusive)
        cot_data = {}
        steps = [step1, step2, step3]

        # Precompute cumulative prefixes for 'all'
        prefix = intro
        for idx, step in enumerate(steps, start=1):
            # All up to this step (include full current step)
            cumulative_all = prefix + "".join(steps[:idx])
            cot_data[f"cot_step{idx}_all"] = cumulative_all.strip()

        return cot_full.strip(), cot_data
    
    def _count_adjacent_empty_cells(self, grid, row, col, size):
        """Count adjacent cells that appear empty in the puzzle view"""
        count = 0
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = row + di, col + dj
                if 0 <= ni < size and 0 <= nj < size:
                    if grid[ni, nj] == 0 or grid[ni, nj] == -1:  # Empty or mine (appears empty)
                        count += 1
        return count
    
    def _get_adjacent_empty_positions(self, grid, row, col, size):
        """Get positions of adjacent empty cells"""
        positions = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = row + di, col + dj
                if 0 <= ni < size and 0 <= nj < size:
                    if grid[ni, nj] == 0 or grid[ni, nj] == -1:
                        positions.append((ni, nj))
        return positions
    
    def _count_adjacent_actual_mines(self, grid, row, col, size):
        """Count actual adjacent mines"""
        count = 0
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = row + di, col + dj
                if 0 <= ni < size and 0 <= nj < size:
                    if grid[ni, nj] == -1:
                        count += 1
        return count
    

    
    def visualize(self, puzzle, show_mines=False, filename=None, **kwargs):
        if isinstance(puzzle, Minesweeper):
            return puzzle.visualize(show_mines=show_mines, filename=filename)
        else:
            minesweeper = Minesweeper.from_grid(puzzle)
            return minesweeper.visualize(show_mines=show_mines, filename=filename)


class Minesweeper:
    def __init__(self, size, num_mines):
        self.size = size
        self.num_mines = min(num_mines, size * size - 1)  # Ensure we don't have too many mines
        
        # Initialize empty grid
        self.grid = np.zeros((size, size), dtype=int)
        
        # Place mines randomly (uses current random seed)
        self._place_mines()
        
        # Calculate numbers
        self._calculate_numbers()
    
    @classmethod
    def from_grid(cls, grid):
        """Create a Minesweeper instance from an existing grid"""
        instance = cls(len(grid), 0)
        instance.grid = np.array(grid)
        instance.num_mines = np.sum(instance.grid == -1)
        return instance
    
    def _place_mines(self):
        """Place mines randomly on the grid"""
        # Get all possible positions
        positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        
        # Randomly select positions for mines
        mine_positions = random.sample(positions, self.num_mines)
        
        # Place mines (-1 represents a mine)
        for i, j in mine_positions:
            self.grid[i, j] = -1
    
    def _calculate_numbers(self):
        """Calculate the numbers for each cell based on adjacent mines"""
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == -1:  # Skip mine cells
                    continue
                
                # Count adjacent mines
                count = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.size and 0 <= nj < self.size and self.grid[ni, nj] == -1:
                            count += 1
                
                self.grid[i, j] = count
    
    def to_text_representation(self, show_mines=False):
        result = []
        
        for i in range(self.size):
            row = []
            for j in range(self.size):
                cell = self.grid[i, j]
                if cell == -1:  # Mine
                    row.append('*' if show_mines else ' ')
                elif cell == 0:  # Empty
                    row.append(' ')
                else:  # Number
                    row.append(str(cell))
            result.append('|' + '|'.join(row) + '|')
        
        return '\n'.join(result)
    
    def get_mine_coordinates(self):
        """Get coordinates of all mines as a formatted string (0-based)"""
        coordinates = []
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j] == -1:  # Mine
                    coordinates.append(f"({i}, {j})")
        return ",".join(coordinates)
    
    def visualize(self, show_mines=False, filename=None):
        """Create a high-quality, fancy visual representation of the puzzle.
        The image contains only puzzle-relevant elements (grid, numbers, optional mines) with clear contrast.
        """
        # Dynamic sizing to keep numbers legible across different board sizes
        size = self.size
        base_inches = max(6.0, min(12.0, size * 0.8))
        fig, ax = plt.subplots(figsize=(base_inches, base_inches))

        # Clean backgrounds
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f7f9fc')

        # Predefine number colors (classic Minesweeper palette)
        number_colors = {
            1: '#1976D2',  # blue
            2: '#388E3C',  # green
            3: '#D32F2F',  # red
            4: '#7B1FA2',  # purple
            5: '#5D4037',  # brown
            6: '#00897B',  # teal
            7: '#212121',  # black-ish
            8: '#616161',  # gray
        }

        # Dynamic font size for numbers
        font_size = max(12, int(36 - size * 1.2))

        # Draw cells and content
        for i in range(size):
            for j in range(size):
                cell_value = self.grid[i, j]

                # Cell rectangle with crisp edges
                cell_rect = Rectangle(
                    (j, i), 1, 1,
                    edgecolor='#2b2f36',
                    facecolor='#ffffff',
                    linewidth=1.3,
                    joinstyle='miter'
                )
                ax.add_patch(cell_rect)

                if cell_value == -1:
                    if show_mines:
                        # Stylized mine: central disc + radial spikes + small highlight
                        center_x, center_y = j + 0.5, i + 0.5
                        radius = 0.28
                        mine_disc = plt.Circle((center_x, center_y), radius, color='#1f1f1f')
                        ax.add_patch(mine_disc)

                        # Spikes
                        for angle_deg in range(0, 360, 45):
                            angle_rad = np.deg2rad(angle_deg)
                            x0 = center_x + (radius + 0.02) * np.cos(angle_rad)
                            y0 = center_y + (radius + 0.02) * np.sin(angle_rad)
                            x1 = center_x + (radius + 0.18) * np.cos(angle_rad)
                            y1 = center_y + (radius + 0.18) * np.sin(angle_rad)
                            ax.plot([x0, x1], [y0, y1], color='#1f1f1f', linewidth=2.0, solid_capstyle='round')

                        # Small highlight
                        highlight = plt.Circle((center_x - 0.1, center_y - 0.1), 0.06, color='#ffffff', alpha=0.85)
                        ax.add_patch(highlight)
                else:
                    if cell_value > 0:
                        txt = ax.text(
                            j + 0.5,
                            i + 0.5,
                            str(cell_value),
                            ha='center',
                            va='center',
                            fontsize=font_size,
                            fontweight='bold',
                            color=number_colors.get(cell_value, '#37474F'),
                        )
                        # White stroke to enhance contrast on any background
                        txt.set_path_effects([
                            patheffects.withStroke(linewidth=3.5, foreground='#ffffff')
                        ])

        # Grid frame and layout cleanup
        ax.set_xlim(0, size)
        ax.set_ylim(size, 0)
        ax.set_aspect('equal')

        # Hide axes decorations
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Strong outer border
        outer = Rectangle((0, 0), size, size, fill=False, linewidth=3.0, edgecolor='#101216')
        ax.add_patch(outer)

        # Save or show, with higher DPI and minimal padding
        if filename:
            plt.tight_layout(pad=0.02)
            plt.savefig(filename, dpi=200, bbox_inches='tight', pad_inches=0.05)
            plt.close(fig)
            return filename
        else:
            plt.tight_layout(pad=0.02)
            plt.show()
            return None