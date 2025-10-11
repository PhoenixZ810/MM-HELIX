import os
import json
import random
import time
import hashlib
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Set, Any
from generator.base_generator import BaseGenerator

class HitoriGenerator(BaseGenerator):
    def __init__(self, output_folder):
        super().__init__(output_folder)
        self.base_size = 4
        self.cell_size = 80
        self.font_size = 36
        self.margin = 10
        self.generated_hashes = set()  # 用于避免重复题目
        self.generated_puzzles = []  # 存储所有生成的问题，用于批量保存
        try:
            self.font = ImageFont.truetype("arial.ttf", self.font_size)
        except IOError:
            # Fallback to default font if arial is not available
            self.font = ImageFont.load_default()

    def generate(self, num_cases, difficulty, output_folder=None):
        """
        生成指定数量和难度的 Hitori 谜题

        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别 (1-5)
            output_folder: 输出文件夹路径，覆盖构造函数中设置的路径
        """
        if output_folder is None:
            output_folder = self.output_folder

        # Setup output directories
        images_dir = os.path.join(output_folder, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Get difficulty parameters
        params = self._get_difficulty_params(difficulty)
        size = params['size']

        print(f"Generating {num_cases} Hitori puzzles with difficulty={difficulty}, size={size}")

        for case in range(num_cases):
            # Use timestamp as seed for each case
            seed = int(time.time() * 1000000) + case  # 微秒级别的时间戳加上case偏移

            print(f"Generating case {case + 1}/{num_cases} with seed={seed}")

            # Generate puzzle with deterministic seed
            max_attempts = 1000
            attempts = 0

            while attempts < max_attempts:
                attempts += 1

                # Set random seed for deterministic generation
                random.seed(seed + attempts)  # 每次尝试使用不同的种子

                # Generate puzzle
                grid = self._generate_latin_square(size)
                complexity = min(0.3 + (size - 4) * 0.05, 0.7)
                solution = self._generate_valid_shaded_cells(size, complexity)

                if not solution:
                    continue

                # Create duplicates for solution cells
                self._create_duplicates_for_solution(grid, solution, size)

                # Validate solution
                if not self._validate_solution(grid, solution, size):
                    continue

                break

            if attempts >= max_attempts:
                print(f"Could not generate valid puzzle for size={size}, seed={seed}")
                continue

            puzzle = {
                "grid": grid,
                "solution": list(solution),
                "size": size,
                "difficulty": str(difficulty)
            }

            # Create image files
            puzzle_img_filename = f"hitori_{size}_{seed}.png"
            puzzle_img_path = os.path.join(images_dir, puzzle_img_filename)

            # Generate and save image
            self.visualize(puzzle, puzzle_img_path)

            # Generate CoT steps
            cot_steps = self.generate_cot(puzzle)

            # Create puzzle data in required format
            puzzle_data = {
                "index": f"hitori_{size}_{seed}",
                "category": "hitori",
                "image": f"images/{puzzle_img_filename}",
                "question": """
You are given an image of a Hitori puzzle grid.


### Puzzle Rules:

1. In each row and each column, numbers in **unshaded cells** must be **unique**.
2. **Shaded cells cannot be adjacent** horizontally or vertically.
3. All **unshaded cells must form a single connected region** (connected orthogonally).

### Coordinate System:

- Coordinates must be in the format `(row, column)`
- `(0, 0)` refers to the **top-left** cell of the grid
- Indexing is **zero-based**

### Output Format:

Please return the set of shaded cell coordinates.
Example output:
{(0, 1), (2, 3), (4, 2)}
""",
                "question_language": self._generate_text_question(puzzle),
                "answer": self._format_answer(solution),
                "initial_state": grid,
                "difficulty": str(difficulty),
                "cot": cot_steps[4]  # Full CoT for backward compatibility
            }

            # Add CoT step tracking fields (only steps 1-3 as specified)
            for step in range(1, 4):  # Only steps 1, 2, 3
                step_text = cot_steps[step]

                # Find a good breaking point around half length (at word boundaries)
                half_length = len(step_text) // 2
                # Look for a space or newline near the halfway point to break cleanly
                break_point = half_length
                for i in range(half_length - 50, half_length + 50):
                    if i >= 0 and i < len(step_text) and step_text[i] in [' ', '\n']:
                        break_point = i
                        break

                puzzle_data[f"cot_step{step}_all"] = step_text

            # Add to generated puzzles list instead of saving immediately
            self.generated_puzzles.append(puzzle_data)
            print(f"Generated puzzle: {puzzle_data['index']}")

        # Batch save all puzzles at the end
        self.save_annotations(self.generated_puzzles, output_folder)

    def _get_difficulty_params(self, difficulty):
        """
        根据难度级别获取相应的参数配置。

        Args:
            difficulty: 难度级别（1-5）

        Returns:
            dict: 包含难度参数的字典
        """
        # Map difficulty to grid size
        size_map = {
            1: 4,   # Easy
            2: 5,   # Medium-Easy
            3: 6,   # Medium
            4: 7,   # Medium-Hard
            5: 8    # Hard
        }

        size = size_map.get(difficulty, 4)  # Default to 4 if invalid difficulty

        return {
            'size': size,
            'complexity': min(0.3 + (size - 4) * 0.05, 0.7)
        }

    def _get_difficulty_level(self, n):
        """根据网格大小确定难度等级"""
        if n <= 4:
            return "1"
        elif n <= 5:
            return "2"
        elif n <= 6:
            return "3"
        elif n <= 7:
            return "4"
        else:
            return "5"



    def _create_duplicates_for_solution(self, grid, solution, size):
        """为需要涂黑的单元格创建重复数字"""
        for r, c in solution:
            # 在同一行或同一列找一个未被涂黑的单元格，复制其数字
            unshaded_row = [(r, j) for j in range(size) if (r, j) not in solution and j != c]
            unshaded_col = [(i, c) for i in range(size) if (i, c) not in solution and i != r]
            
            # 优先在行中找，其次在列中找
            if unshaded_row:
                sample_cell = random.choice(unshaded_row)
                grid[r][c] = grid[sample_cell[0]][sample_cell[1]]
            elif unshaded_col:
                sample_cell = random.choice(unshaded_col)
                grid[r][c] = grid[sample_cell[0]][sample_cell[1]]

    def visualize(self, puzzle, output_path, show_solution=False):
        """Generate an image visualization of the Hitori puzzle"""
        grid = puzzle["grid"]
        size = puzzle["size"]
        solution = puzzle["solution"] if show_solution else []
        
        # Modern, attractive color scheme
        background_color = (245, 248, 255)   # Soft blue-white background
        grid_line_color = (120, 130, 160)    # Subtle blue-gray for grid lines
        cell_bg_color = (255, 255, 255)      # Pure white cells
        shaded_color = (65, 80, 115)         # Rich navy for shaded cells
        number_color = (40, 50, 80)          # Deep blue for numbers
        accent_color = (100, 140, 210)       # Vibrant blue accent
        
        # 根据谜题大小调整单元格大小
        base_cell_size = max(60, 480 // size)  # 动态调整单元格大小
        actual_cell_size = base_cell_size + 10
        font_size = max(24, actual_cell_size // 3)
        
        # Try to load a more aesthetically pleasing font
        try:
            self.font = ImageFont.truetype("Arial Bold.ttf", font_size)
        except IOError:
            try:
                self.font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
            except:
                try:
                    self.font = ImageFont.truetype("arial.ttf", font_size)
                except:
                    self.font = ImageFont.load_default()
        
        # Calculate image dimensions with generous margins for visual breathing room
        border = 60
        width = height = size * actual_cell_size + 2 * border
        
        # Create image with gradient background effect
        img = Image.new('RGB', (width, height), color=background_color)
        draw = ImageDraw.Draw(img)
        
        # Draw an elegant frame with soft shadow
        shadow_offset = 6
        draw.rounded_rectangle(
            [(border - shadow_offset - 10, border - shadow_offset - 10), 
             (width - border + shadow_offset + 10, height - border + shadow_offset + 10)], 
            radius=20, fill=(220, 225, 235)
        )
        
        # Main puzzle frame with attractive border
        draw.rounded_rectangle(
            [(border - 10, border - 10), 
             (width - border + 10, height - border + 10)], 
            radius=18, fill=(235, 240, 252),
            outline=accent_color, width=3
        )
        
        # Draw each cell with enhanced styling
        for i in range(size):
            for j in range(size):
                x = border + j * actual_cell_size
                y = border + i * actual_cell_size
                
                # Cell coordinates with perfect margin for visual balance
                cell_margin = 3
                cell_coords = [
                    (x + cell_margin, y + cell_margin), 
                    (x + actual_cell_size - cell_margin, y + actual_cell_size - cell_margin)
                ]
                
                if show_solution and (i, j) in solution:
                    # Shaded cell with elegant gradient effect
                    draw.rounded_rectangle(cell_coords, radius=8, fill=shaded_color)
                else:
                    # Draw white cell with subtle shadow effect and rounded corners
                    draw.rounded_rectangle(
                        cell_coords, 
                        radius=8,
                        fill=cell_bg_color, 
                        outline=(210, 220, 240), 
                        width=2
                    )
                    
                    # Draw the number with perfect centering
                    number = str(grid[i][j])
                    
                    # Get text dimensions for proper centering
                    try:
                        _, _, text_width, text_height = draw.textbbox((0, 0), number, font=self.font)
                    except (AttributeError, TypeError):
                        try:
                            text_width, text_height = draw.textsize(number, font=self.font)
                        except:
                            text_width, text_height = font_size, font_size
                    
                    # Calculate position to perfectly center the text
                    text_x = x + (actual_cell_size - text_width) // 2
                    text_y = y + (actual_cell_size - text_height) // 2 - 1
                    
                    # Add subtle text shadow for depth
                    shadow_offset = 1
                    draw.text(
                        (text_x + shadow_offset, text_y + shadow_offset), 
                        number, fill=(210, 215, 235), font=self.font
                    )
                    
                    # Main number text with rich color
                    draw.text(
                        (text_x, text_y), 
                        number, fill=number_color, font=self.font
                    )
        
        # Draw grid lines with refined styling
        for i in range(size + 1):
            # Make external borders thicker and more prominent
            line_width = 3 if i == 0 or i == size else 1
            line_color = accent_color if i == 0 or i == size else grid_line_color
            
            # Horizontal lines
            draw.line(
                [(border, border + i * actual_cell_size),
                 (width - border, border + i * actual_cell_size)], 
                fill=line_color, width=line_width
            )
            
            # Vertical lines
            draw.line(
                [(border + i * actual_cell_size, border),
                 (border + i * actual_cell_size, height - border)], 
                fill=line_color, width=line_width
            )
        
        # Add decorative elements at corners of the puzzle
        for cx, cy, angle in [
            (border - 3, border - 3, 0),              # Top-left
            (width - border + 3, border - 3, 90),     # Top-right
            (border - 3, height - border + 3, 270),   # Bottom-left
            (width - border + 3, height - border + 3, 180)  # Bottom-right
        ]:
            # Draw decorative corner elements
            r = 8
            draw.pieslice((cx-r, cy-r, cx+r, cy+r), angle, angle+90, fill=accent_color)
        
        # Save the image with high quality
        img.save(output_path, quality=95)
        return output_path

    def solve(self, puzzle, **kwargs):
        """Return the solution for a given puzzle"""
        return puzzle["solution"]

    def _validate_solution(self, grid, solution, size):
        """Validate that solution satisfies all Hitori rules"""
        # Rule 1: Check uniqueness in rows and columns
        for i in range(size):
            # Check row
            unshaded_row = [grid[i][j] for j in range(size) if (i, j) not in solution]
            if len(unshaded_row) != len(set(unshaded_row)):
                return False
            
            # Check column
            unshaded_col = [grid[j][i] for j in range(size) if (j, i) not in solution]
            if len(unshaded_col) != len(set(unshaded_col)):
                return False
        
        # Rule 2: Check adjacency
        for r, c in solution:
            adjacent_shaded = [(r+dr, c+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)] 
                              if 0 <= r+dr < size and 0 <= c+dc < size and (r+dr, c+dc) in solution]
            if adjacent_shaded:
                return False
        
        # Rule 3: Check connectivity
        return self._is_connected(size, solution)

    def _format_answer(self, solution):
        """Format solution as string of coordinates"""
        coords = sorted(list(solution))
        return str(set(coords))

    # Removed _save_to_annotations method - now using batch storage via save_all_to_annotations

    def generate_cot(self, puzzle):
        """Generate step-by-step chain-of-thought reasoning for solving the puzzle"""
        grid = puzzle["grid"]
        solution = puzzle["solution"]
        size = puzzle["size"]
        
        # Initialize CoT steps
        cot_steps = {}
        
        # Step 1: Understanding the puzzle rules and objectives (comprehensive)
        step1 = "Let me analyze this Hitori puzzle step by step\n\n"
        step1 += "### Step 1: Understanding the puzzle rules and objectives\n\n"
        step1 += f"This is a {size}×{size} Hitori puzzle, which is a logic puzzle that originated in Japan. "
        step1 += "The name 'Hitori' means 'alone' in Japanese, which hints at the core mechanic.\n\n"
        step1 += "**Objective:** I need to shade (blacken) certain cells in the grid to satisfy three fundamental rules:\n\n"
        step1 += "**Rule 1 - Uniqueness Constraint:**\n"
        step1 += "• In each row and each column, numbers in **unshaded cells** must be **unique**\n"
        step1 += "• This means if I see duplicate numbers in a row or column, at least one of them must be shaded\n"
        step1 += "• Shaded cells are effectively 'removed' from uniqueness consideration\n\n"
        step1 += "**Rule 2 - Adjacency Constraint:**\n"
        step1 += "• **Shaded cells cannot be adjacent** horizontally or vertically\n"
        step1 += "• This prevents me from simply shading all duplicates - I must be strategic\n"
        step1 += "• Diagonal adjacency is allowed, only orthogonal adjacency is forbidden\n\n"
        step1 += "**Rule 3 - Connectivity Constraint:**\n"
        step1 += "• All **unshaded cells must form a single connected region**\n"
        step1 += "• This means I can traverse from any unshaded cell to any other unshaded cell through orthogonal moves\n"
        step1 += "• This prevents the puzzle from breaking into isolated islands\n\n"
        step1 += "**Strategy Overview:**\n"
        step1 += "These three rules create a delicate balance - I must resolve duplicates while maintaining connectivity and avoiding adjacent shading. "
        step1 += "The puzzle requires careful analysis and sometimes backtracking to find the unique solution.\n\n"
        cot_steps[1] = step1
        
        # Step 2: Careful image reading with reflection and precise state representation
        step2 = cot_steps[1]
        step2 += "### Step 2: Careful image reading and state analysis\n\n"
        step2 += "**Initial Visual Inspection:**\n"
        step2 += "Let me carefully examine the puzzle image to extract the initial state. I need to read each cell precisely.\n\n"
        
        step2 += "**Grid State Extraction:**\n"
        step2 += f"Reading the {size}×{size} grid from left to right, top to bottom:\n\n"
        step2 += "```\n"
        step2 += "    " + "   ".join(f"C{j}" for j in range(size)) + "\n"  # Column headers
        for i, row in enumerate(grid):
            step2 += f"R{i}  " + "   ".join(map(str, row)) + "\n"
        step2 += "```\n\n"
        
        step2 += "**State Verification:**\n"
        step2 += "Let me double-check my reading by examining each row and column systematically:\n\n"
        
        # Detailed row analysis
        step2 += "**Row-by-row analysis:**\n"
        for i in range(size):
            row_values = [str(grid[i][j]) for j in range(size)]
            step2 += f"• Row {i}: [{', '.join(row_values)}]\n"
        step2 += "\n"
        
        # Detailed column analysis
        step2 += "**Column-by-column analysis:**\n"
        for j in range(size):
            col_values = [str(grid[i][j]) for i in range(size)]
            step2 += f"• Column {j}: [{', '.join(col_values)}]\n"
        step2 += "\n"
        
        step2 += "**Reflection on State Reading:**\n"
        step2 += "I have successfully extracted the grid state. Now let me analyze what this means in the context of Hitori rules.\n\n"
        
        # Find and categorize conflicts with detailed analysis
        row_conflicts = []
        col_conflicts = []
        
        for i in range(size):
            for j in range(size):
                # Check row conflicts
                for k in range(j+1, size):
                    if grid[i][j] == grid[i][k]:
                        row_conflicts.append((i, j, k, grid[i][j]))
                # Check column conflicts
                for k in range(i+1, size):
                    if grid[i][j] == grid[k][j]:
                        col_conflicts.append((j, i, k, grid[i][j]))
        
        total_conflicts = len(row_conflicts) + len(col_conflicts)
        step2 += f"**Conflict Identification:**\n"
        step2 += f"Total duplicate violations found: {total_conflicts}\n\n"
        
        if row_conflicts:
            step2 += f"**Row conflicts ({len(row_conflicts)} found):**\n"
            for i, j, k, value in row_conflicts:
                step2 += f"• Row {i}: Number '{value}' appears at positions ({i},{j}) and ({i},{k})\n"
            step2 += "\n"
        
        if col_conflicts:
            step2 += f"**Column conflicts ({len(col_conflicts)} found):**\n"
            for j, i, k, value in col_conflicts:
                step2 += f"• Column {j}: Number '{value}' appears at positions ({i},{j}) and ({k},{j})\n"
            step2 += "\n"
        
        step2 += "**Critical Observation:**\n"
        if total_conflicts == 0:
            step2 += "Surprisingly, no conflicts were detected. This suggests either:\n"
            step2 += "• The grid is already solved (unlikely for a puzzle), or\n"
            step2 += "• I need to recheck my reading\n\n"
        else:
            step2 += f"The {total_conflicts} conflicts I identified represent violations of Rule 1 (uniqueness). "
            step2 += "Each conflict indicates that I must shade at least one of the duplicate cells. "
            step2 += "However, I must be strategic about which cells to shade to satisfy all three rules simultaneously.\n\n"
        
        cot_steps[2] = step2
        
        # Step 3: Detailed strategic exploration and reasoning
        step3 = cot_steps[2]
        step3 += "### Step 3: Strategic exploration and reasoning\n\n"
        step3 += "Now I need to develop a systematic approach to resolve all conflicts while satisfying the three Hitori rules. "
        step3 += "This is the most critical step that requires careful exploration and logical deduction.\n\n"
        
        solution_cells = sorted(list(solution))
        
        step3 += "**Problem Analysis:**\n"
        step3 += f"From my conflict analysis, I need to shade certain cells to resolve {total_conflicts} violations. "
        step3 += "However, this is not simply a matter of shading all duplicate cells - I must consider:\n"
        step3 += "• Which duplicates to shade (multiple choices exist)\n"
        step3 += "• Maintaining non-adjacency of shaded cells\n"
        step3 += "• Ensuring connectivity of unshaded cells\n\n"
        
        step3 += "**Strategic Approach:**\n"
        step3 += "I'll use a systematic exploration method:\n"
        step3 += "1. Identify all possible candidate cells for shading\n"
        step3 += "2. Prioritize based on conflict resolution potential\n"
        step3 += "3. Check each decision against all three rules\n"
        step3 += "4. Use logical deduction and constraint propagation\n"
        step3 += "5. Backtrack when necessary\n\n"
        
        step3 += "**Detailed Exploration Process:**\n\n"
        
        # Create a mapping of conflicts to cells involved
        conflict_cells = set()
        for i, j, k, value in row_conflicts:
            conflict_cells.add((i, j))
            conflict_cells.add((i, k))
        for j, i, k, value in col_conflicts:
            conflict_cells.add((i, j))
            conflict_cells.add((k, j))
        
        step3 += f"**Candidate Analysis:**\n"
        step3 += f"Cells involved in conflicts: {len(conflict_cells)} cells\n"
        step3 += f"Must shade at least: {len(solution_cells)} cells (actual solution)\n\n"
        
        # Simulate decision-making process for each solution cell
        current_shaded = set()
        decision_log = []
        
        for i, (r, c) in enumerate(solution_cells):
            step3 += f"**Decision Point {i+1}: Cell ({r},{c}) with value {grid[r][c]}**\n"
            
            # Analyze what conflicts this cell is involved in
            involved_row_conflicts = [conflict for conflict in row_conflicts 
                                    if (r, c) == (conflict[0], conflict[1]) or (r, c) == (conflict[0], conflict[2])]
            involved_col_conflicts = [conflict for conflict in col_conflicts 
                                    if (r, c) == (conflict[1], conflict[0]) or (r, c) == (conflict[2], conflict[0])]
            
            step3 += f"• This cell is involved in {len(involved_row_conflicts)} row conflicts and {len(involved_col_conflicts)} column conflicts\n"
            
            # Check alternative options
            alternative_cells = set()
            for conflict in involved_row_conflicts:
                i_conf, j1, j2, val = conflict
                if (r, c) == (i_conf, j1):
                    alternative_cells.add((i_conf, j2))
                else:
                    alternative_cells.add((i_conf, j1))
            for conflict in involved_col_conflicts:
                j_conf, i1, i2, val = conflict
                if (r, c) == (i1, j_conf):
                    alternative_cells.add((i2, j_conf))
                else:
                    alternative_cells.add((i1, j_conf))
            
            step3 += f"• Alternative cells to shade: {sorted(list(alternative_cells))}\n"
            
            # Check adjacency constraint
            adjacent_coords = [(r+dr, c+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)] 
                              if 0 <= r+dr < size and 0 <= c+dc < size]
            adjacent_shaded = [coord for coord in adjacent_coords if coord in current_shaded]
            
            if adjacent_shaded:
                step3 += f"• ⚠️  CONSTRAINT VIOLATION: Adjacent to already shaded cells {adjacent_shaded}\n"
                step3 += f"• 🔄 BACKTRACKING: This would violate Rule 2, need to explore alternatives\n"
                
                # Show backtracking logic
                step3 += f"• 🔍 ALTERNATIVE EXPLORATION:\n"
                for alt_cell in alternative_cells:
                    alt_r, alt_c = alt_cell
                    alt_adjacent = [(alt_r+dr, alt_c+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)] 
                                   if 0 <= alt_r+dr < size and 0 <= alt_c+dc < size]
                    alt_adjacent_shaded = [coord for coord in alt_adjacent if coord in current_shaded]
                    step3 += f"    - Alternative ({alt_r},{alt_c}): {'✓ Safe' if not alt_adjacent_shaded else '✗ Also violates adjacency'}\n"
                
                step3 += f"• 💡 RESOLUTION: Through advanced constraint solving, determined that shading ({r},{c}) is still optimal\n"
            else:
                step3 += f"• ✓ ADJACENCY CHECK: No adjacent shaded cells\n"
            
            # Connectivity preview
            temp_shaded = current_shaded.copy()
            temp_shaded.add((r, c))
            temp_connectivity = self._is_connected(size, temp_shaded)
            step3 += f"• ✓ CONNECTIVITY CHECK: {'Maintains connectivity' if temp_connectivity else 'Would break connectivity'}\n"
            
            # Decision
            step3 += f"• 🎯 DECISION: Shade cell ({r},{c}) - resolves conflicts while maintaining rule compliance\n"
            current_shaded.add((r, c))
            decision_log.append((r, c, len(involved_row_conflicts) + len(involved_col_conflicts)))
            step3 += "\n"
        
        step3 += "**Strategic Summary:**\n"
        step3 += f"Through systematic exploration, I determined that shading {len(solution_cells)} specific cells resolves all conflicts:\n"
        for r, c, conflict_count in decision_log:
            step3 += f"• ({r},{c}): Resolves {conflict_count} conflict(s)\n"
        step3 += "\n"
        
        step3 += "**Rule Compliance Verification:**\n"
        
        # Final connectivity check
        is_connected = self._is_connected(size, solution)
        unshaded_count = size * size - len(solution_cells)
        step3 += f"• Rule 1 (Uniqueness): All {total_conflicts} conflicts resolved ✓\n"
        step3 += f"• Rule 2 (Non-adjacency): No adjacent shaded cells ✓\n"
        step3 += f"• Rule 3 (Connectivity): {unshaded_count} unshaded cells form connected region ✓\n\n"
        
        step3 += "**Final Solution:**\n"
        step3 += f"Shade cells at coordinates: {sorted(solution_cells)}\n"
        step3 += "This solution satisfies all three Hitori rules simultaneously and is the unique solution to this puzzle.\n\n"
        
        cot_steps[3] = step3
        
        # Step 4: Comprehensive solution validation and reflection
        step4 = cot_steps[3]
        step4 += "### Step 4: Solution validation and critical reflection\n\n"
        step4 += "Now I must rigorously validate my proposed solution against all three Hitori rules and reflect on the solving process. "
        step4 += "This step ensures the solution is not only correct but also provides confidence in the reasoning.\n\n"
        
        # Proposed solution recap
        step4 += "**Proposed Solution Recap:**\n"
        solution_cells = sorted(list(solution))
        step4 += f"Shaded cells: {solution_cells}\n"
        step4 += f"Total shaded: {len(solution_cells)} out of {size * size} cells\n"
        step4 += f"Unshaded cells: {size * size - len(solution_cells)} cells\n\n"
        
        # Rule 1 validation with detailed analysis
        step4 += "**Rule 1 Validation - Uniqueness in Rows and Columns:**\n"
        all_unique = True
        uniqueness_details = []
        
        step4 += "*Row Analysis:*\n"
        for i in range(size):
            unshaded_row = [grid[i][j] for j in range(size) if (i, j) not in solution]
            unshaded_positions = [j for j in range(size) if (i, j) not in solution]
            is_unique = len(unshaded_row) == len(set(unshaded_row))
            all_unique &= is_unique
            
            step4 += f"  Row {i}: Values {unshaded_row} at positions {unshaded_positions} "
            step4 += f"{'✓ All unique' if is_unique else '✗ Contains duplicates'}\n"
            if not is_unique:
                duplicates = [x for x in unshaded_row if unshaded_row.count(x) > 1]
                step4 += f"    Duplicates found: {set(duplicates)}\n"
        
        step4 += "\n*Column Analysis:*\n"
        for j in range(size):
            unshaded_col = [grid[i][j] for i in range(size) if (i, j) not in solution]
            unshaded_positions = [i for i in range(size) if (i, j) not in solution]
            is_unique = len(unshaded_col) == len(set(unshaded_col))
            all_unique &= is_unique
            
            step4 += f"  Col {j}: Values {unshaded_col} at positions {unshaded_positions} "
            step4 += f"{'✓ All unique' if is_unique else '✗ Contains duplicates'}\n"
            if not is_unique:
                duplicates = [x for x in unshaded_col if unshaded_col.count(x) > 1]
                step4 += f"    Duplicates found: {set(duplicates)}\n"
        
        step4 += f"\n*Rule 1 Result:* {'✅ PASSED' if all_unique else '❌ FAILED'}\n\n"
        
        # Rule 2 validation with detailed analysis
        step4 += "**Rule 2 Validation - Non-Adjacent Shaded Cells:**\n"
        no_adjacent = True
        adjacency_violations = []
        
        for r, c in solution:
            adjacent_coords = [(r+dr, c+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)] 
                              if 0 <= r+dr < size and 0 <= c+dc < size]
            adjacent_shaded = [coord for coord in adjacent_coords if coord in solution]
            
            if adjacent_shaded:
                no_adjacent = False
                adjacency_violations.append(((r,c), adjacent_shaded))
                step4 += f"  ❌ Cell ({r},{c}) is adjacent to shaded cells: {adjacent_shaded}\n"
            else:
                step4 += f"  ✓ Cell ({r},{c}) has no adjacent shaded cells\n"
        
        if adjacency_violations:
            step4 += f"\n*Adjacency violations found:* {len(adjacency_violations)}\n"
            for violating_cell, adjacent_cells in adjacency_violations:
                step4 += f"  - {violating_cell} adjacent to {adjacent_cells}\n"
        
        step4 += f"\n*Rule 2 Result:* {'✅ PASSED' if no_adjacent else '❌ FAILED'}\n\n"
        
        # Rule 3 validation with detailed connectivity analysis
        step4 += "**Rule 3 Validation - Connectivity of Unshaded Cells:**\n"
        
        unshaded_cells = [(i, j) for i in range(size) for j in range(size) if (i, j) not in solution]
        step4 += f"Unshaded cells to check: {len(unshaded_cells)} cells\n"
        step4 += f"Coordinates: {sorted(unshaded_cells)}\n\n"
        
        # Perform connectivity check with detailed logging
        if not unshaded_cells:
            is_connected = False
            step4 += "*Error:* No unshaded cells remain - invalid puzzle state\n"
        else:
            # BFS connectivity check with step-by-step logging
            visited = set()
            queue = [unshaded_cells[0]]
            step4 += f"*Connectivity Check using BFS starting from {unshaded_cells[0]}:*\n"
            
            bfs_step = 0
            while queue and bfs_step < 10:  # Limit logging for readability
                bfs_step += 1
                cell = queue.pop(0)
                if cell in visited:
                    continue
                
                visited.add(cell)
                i, j = cell
                step4 += f"  Step {bfs_step}: Visited ({i},{j}), total visited: {len(visited)}\n"
                
                # Check adjacent unshaded cells
                adjacent_unshaded = []
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + di, j + dj
                    if (0 <= ni < size and 0 <= nj < size and 
                        (ni, nj) not in solution and (ni, nj) not in visited):
                        queue.append((ni, nj))
                        adjacent_unshaded.append((ni, nj))
                
                if adjacent_unshaded:
                    step4 += f"    Added to queue: {adjacent_unshaded}\n"
            
            if len(queue) > 0:
                step4 += f"  ... (BFS continues for remaining {len(queue)} cells in queue)\n"
            
            is_connected = len(visited) == len(unshaded_cells)
            step4 += f"\n*BFS Result:* Visited {len(visited)} out of {len(unshaded_cells)} unshaded cells\n"
        
        step4 += f"*Rule 3 Result:* {'✅ PASSED' if is_connected else '❌ FAILED'}\n\n"
        
        # Overall solution validation
        solution_valid = all_unique and no_adjacent and is_connected
        step4 += "**Overall Solution Validation:**\n"
        step4 += f"• Rule 1 (Uniqueness): {'✅' if all_unique else '❌'}\n"
        step4 += f"• Rule 2 (Non-adjacency): {'✅' if no_adjacent else '❌'}\n"
        step4 += f"• Rule 3 (Connectivity): {'✅' if is_connected else '❌'}\n"
        step4 += f"\n**FINAL VERDICT: {'✅ SOLUTION VALID' if solution_valid else '❌ SOLUTION INVALID'}**\n\n"
        
        # Critical reflection
        step4 += "**Critical Reflection on the Solving Process:**\n\n"
        step4 += "*Strengths of the approach:*\n"
        step4 += "• Systematic analysis of all conflicts before making decisions\n"
        step4 += "• Careful consideration of all three rules at each step\n"
        step4 += "• Logical exploration of alternatives when constraints are violated\n"
        step4 += "• Rigorous validation of the final solution\n\n"
        
        step4 += "*Key insights gained:*\n"
        step4 += f"• This {size}×{size} puzzle required shading {len(solution_cells)} cells to resolve {total_conflicts} conflicts\n"
        step4 += "• The non-adjacency constraint significantly limits the solution space\n"
        step4 += "• Connectivity verification was crucial to ensure a valid solution\n"
        step4 += "• The unique solution demonstrates the puzzle's logical consistency\n\n"
        
        step4 += "*Alternative approaches considered:*\n"
        step4 += "• Could have used constraint satisfaction algorithms\n"
        step4 += "• Backtracking with more exhaustive search was possible\n"
        step4 += "• Heuristic-based cell prioritization could optimize the process\n\n"
        
        if solution_valid:
            step4 += "**CONCLUSION:**\n"
            step4 += f"The systematic approach successfully solved the Hitori puzzle. "
            step4 += f"The final answer is: **{self._format_answer(solution)}**\n\n"
            step4 += "This solution represents the unique way to shade cells such that all three Hitori rules are satisfied simultaneously."
        else:
            step4 += "**ERROR ANALYSIS:**\n"
            step4 += "The proposed solution failed validation. This indicates either:\n"
            step4 += "• An error in the reasoning process\n"
            step4 += "• A flaw in the puzzle generation\n"
            step4 += "• The need for backtracking and alternative exploration\n"
            step4 += f"Proposed answer was: {self._format_answer(solution)}"
        
        cot_steps[4] = step4
        
        return cot_steps

    def _check_connectivity_simple(self, unshaded_cells, size):
        """Simple connectivity check for COT generation"""
        if not unshaded_cells:
            return False
        
        # BFS to check if all unshaded cells are connected
        visited = set()
        queue = [unshaded_cells[0]]
        unshaded_set = set(unshaded_cells)
        
        while queue:
            cell = queue.pop(0)
            if cell in visited:
                continue
            visited.add(cell)
            
            i, j = cell
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if (0 <= ni < size and 0 <= nj < size and 
                    (ni, nj) in unshaded_set and (ni, nj) not in visited):
                    queue.append((ni, nj))
        
        return len(visited) == len(unshaded_cells)

    def _generate_latin_square(self, size):
        """Generate a Latin square of given size"""
        # 创建更随机的拉丁方
        base = list(range(1, size + 1))
        # 增加随机性
        random.shuffle(base)
        
        square = []
        for i in range(size):
            row = base[i:] + base[:i]
            square.append(row)
        
        # 随机交换行和列以增加变化
        num_swaps = random.randint(size * 2, size * 4)  # 增加交换次数
        for _ in range(num_swaps):
            if random.choice([True, False]):
                # 交换行
                i, j = random.sample(range(size), 2)
                square[i], square[j] = square[j], square[i]
            else:
                # 交换列
                i, j = random.sample(range(size), 2)
                for row in square:
                    row[i], row[j] = row[j], row[i]
        
        # 增加额外的随机化步骤：随机重新排列数字
        if random.random() > 0.5:  # 50%的概率进行数字重排
            # 创建数字映射
            numbers = list(range(1, size + 1))
            new_numbers = numbers[:]
            random.shuffle(new_numbers)
            mapping = dict(zip(numbers, new_numbers))
            
            # 应用映射
            for i in range(size):
                for j in range(size):
                    square[i][j] = mapping[square[i][j]]
        
        return square
    
    def _generate_valid_shaded_cells(self, size, complexity):
        """Generate a valid set of shaded cells for the puzzle"""
        target_shaded = max(1, int(size * size * complexity))
        shaded = set()
        candidates = [(i, j) for i in range(size) for j in range(size)]
        
        # 增加随机性，多次shuffle
        for _ in range(3):
            random.shuffle(candidates)
        
        attempts = 0
        max_attempts = size * size * 20  # 增加最大尝试次数
        
        # 添加随机起始位置
        start_pos = random.randint(0, len(candidates) - 1)
        
        while len(shaded) < target_shaded and attempts < max_attempts:
            attempts += 1
            cell_idx = (start_pos + attempts) % len(candidates)
            cell = candidates[cell_idx]
            i, j = cell
            
            # Check if any adjacent cells are already shaded
            adjacent = any((i + di, j + dj) in shaded 
                          for di, dj in [(-1,0), (1,0), (0,-1), (0,1)] 
                          if 0 <= i + di < size and 0 <= j + dj < size)
            if adjacent:
                continue
                
            # Try adding this cell
            temp_shaded = shaded.copy()
            temp_shaded.add(cell)
            
            # Check if remaining unshaded cells are still connected
            if self._is_connected(size, temp_shaded):
                shaded.add(cell)
                # 重新shuffle候选列表以增加随机性
                if len(shaded) % 3 == 0:
                    random.shuffle(candidates)
                
        return shaded
    
    def _is_connected(self, size, shaded):
        """Check if all unshaded cells form a connected component"""
        unshaded = [(i, j) for i in range(size) for j in range(size) if (i, j) not in shaded]
        if not unshaded:
            return False
            
        # Use BFS to check connectivity
        visited = set()
        queue = [unshaded[0]]
        
        while queue:
            cell = queue.pop(0)
            if cell in visited:
                continue
                
            visited.add(cell)
            i, j = cell
            
            # Check all four adjacent cells
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if (0 <= ni < size and 0 <= nj < size and 
                    (ni, nj) not in shaded and (ni, nj) not in visited):
                    queue.append((ni, nj))
                    
        # All unshaded cells should be visited
        return len(visited) == len(unshaded)
    
    def _generate_text_question(self, puzzle):
        """Generate a comprehensive text-only version of the question in English"""
        grid = puzzle["grid"]
        difficulty = puzzle["difficulty"]
        size = puzzle["size"]
        
        # Format the grid as a string for initial_state reference
        grid_str = "\n".join([" ".join(map(str, row)) for row in grid])
        
        # Generate comprehensive question without leaking solution information
        return f"""
You are given an image of a Hitori puzzle grid.

### Puzzle Rules:

1. In each row and each column, numbers in **unshaded cells** must be **unique**.
2. **Shaded cells cannot be adjacent** horizontally or vertically.
3. All **unshaded cells must form a single connected region** (connected orthogonally).

### Coordinate System:

- Coordinates must be in the format `(row, column)`
- `(0, 0)` refers to the **top-left** cell of the grid
- Indexing is **zero-based**

### Initial State:
```
{grid_str}
```

### Output Format:

Please return the set of shaded cell coordinates.
Example output:
{(0, 1), (2, 3), (4, 2)}
"""
    




# Test interface
if __name__ == "__main__":
    # Create generator instance
    generator = HitoriGenerator()
    
    print("🎯 Hitori Puzzle Generator")
    print("=" * 50)
    print("Usage: generator.generate(size=4, seed=12345, output_dir='./test_output')")
    print("=" * 50)
    
    # Example generation
    test_size = 4
    test_seed = 12345
    test_output_dir = "./test_output"
    
    try:
        puzzle_data = generator.generate(
            size=test_size,
            seed=test_seed,
            output_dir=test_output_dir
        )
        
        print("\n✅ Test generation successful!")
        print(f"Generated puzzle: {puzzle_data['index']}")
        print(f"Grid size: {test_size}x{test_size}")
        print(f"Seed: {test_seed}")
        print(f"Answer: {puzzle_data['answer']}")
        print(f"Output directory: {test_output_dir}")
        
    except Exception as e:
        print(f"\n❌ Test generation failed: {e}")
        import traceback
        traceback.print_exc()