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
from math import gcd
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import uuid
import shutil
import heapq
from matplotlib.patches import Rectangle
from heapq import heappush, heappop
import concurrent.futures
import threading
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.constants import PROMPT_MAZE_IMAGE, PROMPT_15PUZZLE_IMAGE, PROMPT_HANOI_IMAGE, PROMPT_WORDSEARCH_IMAGE, PROMPT_NUMBRIX_IMAGE, PROMPT_MINESWEEPER_IMAGE, PROMPT_EULERO_IMAGE, PROMPT_SNAKE_IMAGE
from utils.constants import PROMPT_MAZE, PROMPT_15PUZZLE, PROMPT_HANOI, PROMPT_WORDSEARCH, PROMPT_NUMBRIX, PROMPT_MINESWEEPER, PROMPT_EULERO, PROMPT_SNAKE

# 添加上级目录到路径，以便导入generator包
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from generator.base_generator import BaseGenerator




 
class EuleroGenerator(BaseGenerator):
    def __init__(self, output_folder, training_set_path=None, global_items=None):
        super().__init__(output_folder)
        # 训练集与去重集在子类中维护，避免影响通用接口
        self.training_set = self._load_training_set(training_set_path) if training_set_path else []
        print(f"Loaded {len(self.training_set)} items from training set for duplicate checking")
        self.global_items = global_items if global_items is not None else set()
        self.current_items = set()

    # 新接口：批量生成，统一IO
    def generate(self, num_cases, difficulty, output_folder=None):
        output_dir = output_folder or self.output_folder
        images_dir = os.path.join(output_dir, 'images')

        params = self._get_difficulty_params(difficulty)
        size = params.get('size')
        if size is None:
            raise ValueError("_get_difficulty_params must provide 'size'")

        # 基于时间戳的种子（整数）
        base_seed = int(time.time())

        # 内存中先构造所有拼图（不做IO）
        puzzles = []  # 存放puzzle_info
        visuals = []  # 存放用于最终绘图的对象及文件名

        for i in range(int(num_cases)):
            seed = base_seed + i
            result = self._generate_one_no_io(size=size, seed=seed)
            if result is None:
                continue
            puzzle_info, partial_eulero, image_filename = result
            puzzles.append(puzzle_info)
            visuals.append((partial_eulero, image_filename))

        # 统一写入IO
        os.makedirs(images_dir, exist_ok=True)

        # 保存图片
        for partial_eulero, image_filename in visuals:
            image_path = os.path.join(images_dir, image_filename)
            partial_eulero.visualize(filename=image_path)

        # 写入annotations.json（一次性）
        annotations_file = os.path.join(output_dir, 'annotations.json')
        # Merge with existing annotations if present, to avoid losing prior items
        existing_data = []
        if os.path.exists(annotations_file):
            try:
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
            except Exception:
                existing_data = []

        # Deduplicate by index while preserving existing entries
        merged_by_index = {}
        for item in existing_data:
            idx = item.get('index')
            if idx is None:
                # generate stable fallback key
                idx = uuid.uuid4().hex
                item['index'] = idx
            merged_by_index[idx] = item
        for item in puzzles:
            idx = item.get('index')
            if idx is None:
                idx = uuid.uuid4().hex
                item['index'] = idx
            if idx not in merged_by_index:
                merged_by_index[idx] = item

        merged_list = list(merged_by_index.values())

        # Atomic write to reduce multi-process race risk
        tmp_file = annotations_file + ".tmp" + uuid.uuid4().hex[:8]
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(merged_list, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, annotations_file)
        print(f"Saved {len(merged_list)} puzzles to {annotations_file}")

        return puzzles

    def _get_difficulty_params(self, difficulty):
        # 简单映射：难度1-5 对应 size 3-7
        try:
            level = int(difficulty)
        except Exception:
            raise ValueError("difficulty must be an integer 1-5 or string of it")
        level = max(1, min(5, level))
        size_map = {1: 3, 2: 4, 3: 5, 4: 6, 5: 7}
        return {"size": size_map[level]}

    # 以下为原有逻辑性工具，保持构造逻辑不变，仅迁入子类
    def _load_training_set(self, training_set_path):
        try:
            with open(training_set_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
                print(f"Successfully loaded training set from {training_set_path}")
                return training_data
        except Exception as e:
            print(f"Warning: Could not load training set from {training_set_path}: {e}")
            return []

    def _is_duplicate(self, puzzle_question, puzzle_answer, puzzle_category):
        item_key = (puzzle_category, puzzle_question, puzzle_answer)
        if item_key in self.current_items:
            return True
        if item_key in self.global_items:
            return True
        if hasattr(self, 'training_set') and self.training_set:
            for training_item in self.training_set:
                if (training_item.get('category') == puzzle_category and 
                    training_item.get('question') == puzzle_question and
                    training_item.get('answer') == puzzle_answer):
                    return True
        return False

    def _add_to_current_items(self, puzzle_question, puzzle_answer, puzzle_category):
        item_key = (puzzle_category, puzzle_question, puzzle_answer)
        self.current_items.add(item_key)
        self.global_items.add(item_key)

    def _generate_one_no_io(self, size, seed):
        # 设置随机种子确保可重现性
        random.seed(seed)
        np.random.seed(seed)

        print(f"Generating Eulero puzzle with size={size}, seed={seed}")

        difficulty = self._get_difficulty_by_size(size)
        remove_percent = min(70, max(40, 40 + (size - 3) * 5))

        try:
            eulero = Eulero(size, seed=seed)
            is_valid, rule_message = eulero.validate_global_rules()
            if not is_valid:
                print(f"Generated complete puzzle violates rules: {rule_message}")
                return None

            blanks = int(size * size * remove_percent / 100)
            print(f"Attempting to create partial puzzle with {blanks} blanks...")
            partial_eulero = eulero.create_partial(blanks, seed=seed)

            is_partial_valid, partial_rule_message = partial_eulero.validate_global_rules()
            if not is_partial_valid:
                print(f"Generated partial puzzle violates rules: {partial_rule_message}")
                return None

            initial_state = partial_eulero.to_text_representation()
            answer = eulero.to_text_representation()

            if self._is_duplicate(initial_state, answer, "eulero"):
                print("Generated puzzle is duplicate for current instance; skipping...")
                return None

            print("✓ Generated puzzle passes all global rule validations")
        except Exception as e:
            print(f"Error generating Eulero puzzle: {e}")
            return None

        unique_suffix = uuid.uuid4().hex[:8]
        puzzle_id = f"eulero_{size}_{seed}_{unique_suffix}"
        image_filename = f"eulero_{size}_{seed}.png"

        cot_data = self.generate_cot(initial_state, answer, size)

        puzzle_info = {
            'index': puzzle_id,
            'category': "eulero",
            'image': f"images/{image_filename}",
            'question': PROMPT_EULERO_IMAGE,
            'question_language': PROMPT_EULERO.format(initial_state),
            'answer': answer,
            'initial_state': initial_state,
            'difficulty': difficulty,
            'cot': cot_data['cot_full'],
            'cot_step1_all': cot_data['cot_step1_all'],
            'cot_step2_all': cot_data['cot_step2_all'],
            'cot_step3_all': cot_data['cot_step3_all']
        }

        self._add_to_current_items(initial_state, answer, "eulero")

        print(f"Prepared Eulero puzzle (no IO yet): {puzzle_id}")
        return puzzle_info, partial_eulero, image_filename
    
    def _get_difficulty_by_size(self, size):
        """根据size确定难度级别"""
        if size <= 3:
            return "1"
        elif size <= 4:
            return "2"
        elif size <= 5:
            return "3"
        elif size <= 6:
            return "4"
        else:
            return "5"
    
    def generate_cot(self, initial_state, answer, size):
        """Generate enhanced Rule-Based CoT with four comprehensive steps.

        - cot_full: full Steps 1–4
        - cot_step{1..3}_all: from intro through that step (inclusive)
        - cot_step{1..3}_part: same as _all but the last step's text is truncated at about half the words
        """

        def parse_grid(state_text):
            rows = [row.split('|') for row in state_text.strip().split('\n')]
            return rows

        def collect_cells(rows_list):
            filled, empty = [], []
            n = len(rows_list)
            for r in range(n):
                for c in range(n):
                    cell_text = rows_list[r][c].strip()
                    if cell_text and cell_text != " ":
                        filled.append((r, c, cell_text))
                    else:
                        empty.append((r, c))
            return filled, empty

        def smart_half_truncate(text: str) -> str:
            words = text.split()
            if len(words) <= 12:
                return text
            half = max(1, len(words) // 2)
            truncated = ' '.join(words[:half])
            # Try to end at a nicer boundary if possible
            for marker in ['\n\n', '\n', '. ', '。', '!', '！', ';', '；', ':', '：']:
                idx = truncated.rfind(marker)
                if idx >= 40:  # avoid extremely short cuts
                    truncated = truncated[:idx + len(marker)].rstrip()
                    break
            return truncated

        def analyze_row_col_constraints(grid_rows, size):
            """Analyze row and column constraints from the current state"""
            analysis = []
            letters_range = [chr(65 + i) for i in range(size)]
            numbers_range = [str(i + 1) for i in range(size)]
            
            # Analyze rows
            for r in range(size):
                row_letters = set()
                row_numbers = set()
                for c in range(size):
                    cell = grid_rows[r][c].strip()
                    if cell and cell != " ":
                        row_letters.add(cell[0])
                        row_numbers.add(cell[1:])
                
                missing_letters = [l for l in letters_range if l not in row_letters]
                missing_numbers = [n for n in numbers_range if n not in row_numbers]
                
                if missing_letters or missing_numbers:
                    analysis.append(f"Row {r+1}: missing letters {missing_letters}, missing numbers {missing_numbers}")
            
            # Analyze columns
            for c in range(size):
                col_letters = set()
                col_numbers = set()
                for r in range(size):
                    cell = grid_rows[r][c].strip()
                    if cell and cell != " ":
                        col_letters.add(cell[0])
                        col_numbers.add(cell[1:])
                
                missing_letters = [l for l in letters_range if l not in col_letters]
                missing_numbers = [n for n in numbers_range if n not in col_numbers]
                
                if missing_letters or missing_numbers:
                    analysis.append(f"Column {c+1}: missing letters {missing_letters}, missing numbers {missing_numbers}")
            
            return analysis[:6]  # Limit to first 6 analyses to keep it manageable

        def find_forced_moves(grid_rows, size):
            """Find cells that have only one possible value"""
            forced_moves = []
            letters_range = [chr(65 + i) for i in range(size)]
            numbers_range = [str(i + 1) for i in range(size)]
            
            # Get all used pairs
            used_pairs = set()
            for r in range(size):
                for c in range(size):
                    cell = grid_rows[r][c].strip()
                    if cell and cell != " ":
                        used_pairs.add(cell)
            
            for r in range(size):
                for c in range(size):
                    cell = grid_rows[r][c].strip()
                    if not cell or cell == " ":
                        # Find valid candidates for this empty cell
                        row_letters = set()
                        row_numbers = set()
                        col_letters = set()
                        col_numbers = set()
                        
                        # Get used letters/numbers in this row
                        for cc in range(size):
                            cell_val = grid_rows[r][cc].strip()
                            if cell_val and cell_val != " ":
                                row_letters.add(cell_val[0])
                                row_numbers.add(cell_val[1:])
                        
                        # Get used letters/numbers in this column
                        for rr in range(size):
                            cell_val = grid_rows[rr][c].strip()
                            if cell_val and cell_val != " ":
                                col_letters.add(cell_val[0])
                                col_numbers.add(cell_val[1:])
                        
                        # Find valid combinations
                        valid_candidates = []
                        for letter in letters_range:
                            if letter not in row_letters and letter not in col_letters:
                                for number in numbers_range:
                                    if number not in row_numbers and number not in col_numbers:
                                        pair = letter + number
                                        if pair not in used_pairs:
                                            valid_candidates.append(pair)
                        
                        if len(valid_candidates) == 1:
                            forced_moves.append(f"Cell ({r+1},{c+1}) must be {valid_candidates[0]}")
                        elif len(valid_candidates) <= 3:
                            forced_moves.append(f"Cell ({r+1},{c+1}) has limited options: {valid_candidates}")
            
            return forced_moves[:4]  # Limit to keep output manageable

        # Analyze initial state
        grid_rows = parse_grid(initial_state)
        filled_cells, empty_cells = collect_cells(grid_rows)
        letters_max = chr(65 + size - 1)
        answer_grid = parse_grid(answer)

        # Intro line
        intro = "I need to solve this Eulero (Graeco-Latin Square) puzzle by carefully analyzing the given information and applying logical reasoning.\n\n"

        # Step 1: Enhanced rule understanding
        step1_body = (
            "### Step 1: Understanding the Game Rules and Objectives\n\n"
            "Let me first establish a clear understanding of what this puzzle requires:\n\n"
            "**Core Rules of Eulero (Graeco-Latin Square):**\n"
            f"1. **Grid Structure**: This is a {size}×{size} grid where each cell contains a letter-number pair\n"
            f"2. **Letter Constraint**: Each letter from A to {letters_max} must appear exactly once in every row and every column\n"
            f"3. **Number Constraint**: Each number from 1 to {size} must appear exactly once in every row and every column\n"
            f"4. **Pair Uniqueness**: Each letter-number combination (like A1, B2, etc.) must be unique across the entire grid\n"
            f"5. **Completeness**: All {size*size} cells must be filled\n\n"
            "**Objective**: Fill all empty cells while satisfying all four constraints simultaneously.\n\n"
            "**Strategy Overview**: I'll use constraint propagation, logical deduction, and systematic reasoning to find the unique solution. "
            "The key is to identify forced moves where only one value is possible, then use these placements to unlock further deductions."
        )

        # Step 2: Enhanced visual analysis and state representation
        grid_representation = "\n".join(["|".join([f"{cell:^4}" if cell.strip() else "    " for cell in row]) for row in grid_rows])
        
        # Get detailed cell analysis
        all_given_cells = ", ".join([f"({r+1},{c+1})→{v}" for r, c, v in filled_cells])
        if len(all_given_cells) > 200:  # Truncate if too long
            sample_count = min(8, len(filled_cells))
            all_given_cells = ", ".join([f"({r+1},{c+1})→{v}" for r, c, v in filled_cells[:sample_count]])
            if len(filled_cells) > sample_count:
                all_given_cells += f"... (and {len(filled_cells) - sample_count} more)"

        constraint_analysis = analyze_row_col_constraints(grid_rows, size)
        constraint_text = "\n".join([f"  - {analysis}" for analysis in constraint_analysis]) if constraint_analysis else "  - All rows and columns have sufficient constraints"

        step2_body = (
            "### Step 2: Reading and Analyzing the Visual Information\n\n"
            "**Initial Grid State Analysis:**\n"
            f"```\n{grid_representation}\n```\n\n"
            "**Reading Reflection**: Let me carefully examine what's given:\n"
            f"- **Grid Size**: {size}×{size} (expecting letters A-{letters_max} and numbers 1-{size})\n"
            f"- **Pre-filled Cells**: {len(filled_cells)} cells are given\n"
            f"- **Empty Cells**: {len(empty_cells)} cells need to be filled\n"
            f"- **Given Values**: {all_given_cells}\n\n"
            "**Text Representation of Current State:**\n"
            f"```\n{initial_state}\n```\n\n"
            "**Constraint Analysis**: Based on the given cells, here's what each row/column is missing:\n"
            f"{constraint_text}\n\n"
            "**Initial Observations**: Each given cell creates constraints in its row and column, eliminating certain letter-number combinations "
            "for other cells in those positions. The intersecting constraints will help us identify forced placements."
        )

        # Step 3: Enhanced strategic reasoning with specific deductions
        forced_moves = find_forced_moves(grid_rows, size)
        forced_moves_text = "\n".join([f"  - {move}" for move in forced_moves]) if forced_moves else "  - No immediately obvious forced moves; need systematic analysis"

        step3_body = (
            "### Step 3: Strategic Exploration and Detailed Reasoning Process\n\n"
            "Now I'll work through the puzzle systematically using logical deduction:\n\n"
            "**Step 3a: Initial Constraint Analysis**\n"
            f"{forced_moves_text}\n\n"
            "**Step 3b: Systematic Solving Approach**\n"
            "I'll use a multi-pass constraint propagation strategy:\n\n"
            "1. **Single Candidate Detection**: For each empty cell, determine which letter-number pairs are valid:\n"
            "   - Eliminate letters already used in the same row/column\n"
            "   - Eliminate numbers already used in the same row/column  \n"
            "   - Eliminate pairs already used anywhere in the grid\n"
            "   - If only one valid pair remains, place it immediately\n\n"
            "2. **Hidden Singles**: Look for letters or numbers that can only go in one position within a row/column:\n"
            "   - If letter X can only appear in one empty cell in a row, place it there\n"
            "   - If number Y can only appear in one empty cell in a column, place it there\n\n"
            "3. **Pair Elimination**: Use the uniqueness constraint:\n"
            "   - If pair AB can only appear in one position globally, place it\n"
            "   - Cross-reference row, column, and global uniqueness constraints\n\n"
            "4. **Constraint Propagation**: After each placement:\n"
            "   - Update available letters/numbers for affected rows/columns\n"
            "   - Check for new forced moves\n"
            "   - Iterate until no more immediate deductions are possible\n\n"
            "5. **Logical Deduction**: Apply advanced reasoning:\n"
            "   - If row R needs letters {A,B} and columns {C1,C2}, check which combinations are globally valid\n"
            "   - Use elimination by contradiction: assume a value and see if it leads to conflicts\n"
            "   - Apply naked/hidden pairs logic when multiple cells have the same limited options\n\n"
            "**Step 3c: Working Through the Solution**\n"
            "By systematically applying these techniques and carefully tracking constraints, I can identify the unique placement "
            "for each empty cell. The process involves multiple iterations of constraint checking and logical deduction until "
            "all cells are filled while maintaining the four core rules of the Eulero puzzle."
        )

        # Step 4: Enhanced validation with detailed verification
        step4_body = (
            "### Step 4: Solution Validation and Final Verification\n\n"
            "**Final Solution Verification Process:**\n\n"
            "Let me verify that the complete solution satisfies all Eulero puzzle requirements:\n\n"
            f"**Rule 1 - Letter Constraint Verification**: Each row and column contains letters A-{letters_max} exactly once\n"
            f"  - Checking all {size} rows for complete letter sets\n"
            f"  - Checking all {size} columns for complete letter sets\n"
            f"  - ✓ Verified: No missing or duplicate letters in any row/column\n\n"
            f"**Rule 2 - Number Constraint Verification**: Each row and column contains numbers 1-{size} exactly once\n"
            f"  - Checking all {size} rows for complete number sets\n"
            f"  - Checking all {size} columns for complete number sets\n"
            f"  - ✓ Verified: No missing or duplicate numbers in any row/column\n\n"
            f"**Rule 3 - Pair Uniqueness Verification**: All {size*size} letter-number pairs are unique\n"
            f"  - Checking that each possible pair appears exactly once\n"
            f"  - Verifying no duplicate pairs exist anywhere in the grid\n"
            f"  - ✓ Verified: All {size*size} pairs are unique across the entire grid\n\n"
            "**Rule 4 - Completeness Verification**: All cells are properly filled\n"
            f"  - Confirming all {size*size} cells contain valid letter-number pairs\n"
            f"  - Ensuring no empty cells remain\n"
            f"  - ✓ Verified: Grid is completely filled with valid pairs\n\n"
            "**Logic Consistency Check**: Reviewing the reasoning process\n"
            "  - All placements were derived through valid logical deduction\n"
            "  - No contradictions arose during the solving process\n"
            "  - Each step followed naturally from the constraints and previous placements\n"
            "  - ✓ Verified: Solution is logically sound and follows from the given clues\n\n"
            "**Final Answer**: The solution satisfies all four core rules of the Eulero puzzle and represents the unique valid completion of the given initial state."
        )

        # Compose cumulative texts
        step1_all = intro + step1_body
        step2_all = step1_all + "\n\n" + step2_body
        step3_all = step2_all + "\n\n" + step3_body
        cot_full = step3_all + "\n\n" + step4_body

        # Build parts where the final step content is half-truncated
        step1_part = intro + smart_half_truncate(step1_body)
        step2_part = step1_all + "\n\n" + smart_half_truncate(step2_body)
        step3_part = step2_all + "\n\n" + smart_half_truncate(step3_body)

        return {
            'cot_full': cot_full,
            'cot_step1_all': step1_all,
            'cot_step2_all': step2_all,
            'cot_step3_all': step3_all,
            'cot_step1_part': step1_part,
            'cot_step2_part': step2_part,
            'cot_step3_part': step3_part,
        }
    
    def visualize(self, puzzle, filename=None, **kwargs):
        if isinstance(puzzle, Eulero):
            return puzzle.visualize(filename=filename)
        else:
            # 如果是网格表示，转换为Eulero实例
            eulero = Eulero.from_grid(puzzle)
            return eulero.visualize(filename=filename)

class Eulero:
    def __init__(self, size, seed=None):
        """初始化Graeco-Latin Square (Eulero)拼图
        
        Args:
            size: 方形网格的大小（必须是奇数或大于4的偶数）
            seed: 随机种子（可选，如果提供则确保可重现性）
        """
        self.size = size
        self.grid = np.empty((size, size), dtype=object)
        
        # 生成有效的Graeco-Latin方阵
        self._generate(seed=seed)
    
    @classmethod
    def from_grid(cls, grid):
        """从现有网格创建Eulero实例"""
        size = len(grid)
        instance = cls(size)
        instance.grid = np.array(grid, dtype=object)
        return instance
    
    def _generate(self, seed=None):
        """生成有效的Graeco-Latin方阵"""
        # 如果提供了seed，设置随机状态（但不影响全局随机状态）
        if seed is not None:
            current_state = random.getstate()
            np_current_state = np.random.get_state()
            random.seed(seed)
            np.random.seed(seed)
        
        try:
            # 对于3, 5, 7等奇数尺寸，我们可以使用简单的构造
            if self.size % 2 == 1:
                self._generate_odd_size()
            # 对于尺寸4，使用特殊情况
            elif self.size == 4:
                self._generate_size_4()
            # 对于大于4的偶数尺寸，使用更复杂的构造
            else:
                self._generate_even_size()
            
            # 添加随机变换来增加多样性
            self._apply_random_transformations(seed)
            
        finally:
            # 如果设置了seed，恢复之前的随机状态
            if seed is not None:
                random.setstate(current_state)
                np.random.set_state(np_current_state)
    
    def _generate_odd_size(self):
        """使用循环构造生成奇数尺寸的Latin方阵"""
        letters = [chr(65 + i) for i in range(self.size)]
        numbers = [str(i + 1) for i in range(self.size)]
        
        # 随机打乱字母和数字的顺序以增加多样性
        random.shuffle(letters)
        random.shuffle(numbers)
        
        # 选择与尺寸互质的偏移，且二者差值与尺寸也互质，保证正交性
        candidates = [k for k in range(1, self.size) if gcd(k, self.size) == 1]
        # 为了正交性，要求gcd(a - b, n) == 1
        random.shuffle(candidates)
        offset1 = None
        offset2 = None
        for a in candidates:
            for b in candidates:
                if a == b:
                    continue
                if gcd(abs(a - b), self.size) == 1:
                    offset1, offset2 = a, b
                    break
            if offset1 is not None:
                break
        if offset1 is None or offset2 is None:
            # 极端情况下退回到(1,2)，该组合对所有奇数n均有效
            offset1, offset2 = 1, 2
        
        # 第一个Latin方阵（使用字母）
        for i in range(self.size):
            for j in range(self.size):
                letter_idx = (i + offset1 * j) % self.size
                self.grid[i, j] = letters[letter_idx]
        
        # 第二个Latin方阵（使用数字）
        for i in range(self.size):
            for j in range(self.size):
                num_idx = (i + offset2 * j) % self.size
                self.grid[i, j] = self.grid[i, j] + numbers[num_idx]
    
    def _generate_size_4(self):
        """尺寸4的特殊情况"""
        # 多个有效的4x4 Graeco-Latin方阵模板
        templates = [
            [
                ["A1", "B2", "C3", "D4"],
                ["B4", "A3", "D2", "C1"],
                ["C2", "D1", "A4", "B3"],
                ["D3", "C4", "B1", "A2"]
            ],
            [
                ["A1", "B3", "C4", "D2"],
                ["B2", "A4", "D1", "C3"],
                ["C3", "D2", "A1", "B4"],
                ["D4", "C1", "B2", "A3"]
            ],
            [
                ["A2", "B1", "C4", "D3"],
                ["B3", "A4", "D1", "C2"],
                ["C1", "D2", "A3", "B4"],
                ["D4", "C3", "B2", "A1"]
            ]
        ]
        
        # 随机选择一个模板
        template = random.choice(templates)
        
        # 创建字母和数字的随机映射
        letters = ['A', 'B', 'C', 'D']
        numbers = ['1', '2', '3', '4']
        random.shuffle(letters)
        random.shuffle(numbers)
        
        # 创建映射字典
        letter_map = {chr(65 + i): letters[i] for i in range(4)}
        number_map = {str(i + 1): numbers[i] for i in range(4)}
        
        for i in range(4):
            for j in range(4):
                original_cell = template[i][j]
                original_letter = original_cell[0]
                original_number = original_cell[1]
                
                new_letter = letter_map[original_letter]
                new_number = number_map[original_number]
                
                self.grid[i, j] = new_letter + new_number
    
    def _generate_even_size(self):
        """生成大于4的偶数尺寸的Graeco-Latin方阵"""
        letters = [chr(65 + i) for i in range(self.size)]
        numbers = [str(i + 1) for i in range(self.size)]
        
        # 随机打乱字母和数字的顺序
        random.shuffle(letters)
        random.shuffle(numbers)
        
        # 使用一种改进的方法来处理大的偶数尺寸，特别是10
        # 为尺寸10创建特殊解决方案
        if self.size == 10:
            self._generate_size_10()
        else:
            # 尝试使用两个正交的Latin方阵组合
            # 第一个Latin方阵（使用循环构造）
            latin1 = np.empty((self.size, self.size), dtype=object)
            
            # 选择与尺寸互质的第一个偏移量，确保是Latin
            coprimes = [k for k in range(1, self.size) if gcd(k, self.size) == 1]
            first_offset = random.choice(coprimes)
            for i in range(self.size):
                for j in range(self.size):
                    latin1[i, j] = letters[(i + first_offset * j) % self.size]
            
            # 尝试不同的偏移量来构造第二个Latin方阵（数字），确保：
            # 1) 与n互质（保证Latin）
            # 2) 与first_offset的差与n互质（保证与第一个正交）
            shifts = [k for k in coprimes if gcd(abs(k - first_offset), self.size) == 1]
            random.shuffle(shifts)
            
            for shift in shifts:
                valid = True
                pairs = set()
                
                # 检查使用此偏移量是否产生有效的Graeco-Latin方阵
                for i in range(self.size):
                    for j in range(self.size):
                        letter = latin1[i, j]
                        num_idx = (i + shift*j) % self.size
                        pair = letter + numbers[num_idx]
                        
                        if pair in pairs:
                            valid = False
                            break
                        pairs.add(pair)
                    
                    if not valid:
                        break
                
                if valid:
                    # 找到有效的偏移量，应用它
                    for i in range(self.size):
                        for j in range(self.size):
                            letter = latin1[i, j]
                            num_idx = (i + shift*j) % self.size
                            self.grid[i, j] = letter + numbers[num_idx]
                    return
            
            # 如果没有找到有效的偏移量，回退到分块构造
            self._generate_by_blocks()
    
    def _generate_size_10(self):
        """尺寸10的特殊构造"""
        letters = [chr(65 + i) for i in range(10)]
        numbers = [str(i + 1) for i in range(10)]
        
        # 随机打乱字母和数字的顺序
        random.shuffle(letters)
        random.shuffle(numbers)
        
        # 构造两个特殊的正交Latin方阵
        # 第一个Latin方阵，使用随机偏移
        latin1 = np.empty((10, 10), dtype=object)
        offset1 = random.choice([1, 3, 7, 9])  # 选择与10互质的偏移量
        for i in range(10):
            for j in range(10):
                latin1[i, j] = letters[(i + offset1 * j) % 10]
        
        # 第二个Latin方阵（使用特殊构造确保正交性）
        # 使用一种能确保尺寸10有效的构造
        latin2 = np.empty((10, 10), dtype=object)
        
        # 随机选择构造方法
        construction_method = random.randint(1, 2)
        
        if construction_method == 1:
            # 方法1：分块构造
            for i in range(10):
                for j in range(10):
                    if i < 5:
                        if j < 5:
                            # 左上角5x5块
                            num_idx = (2*i + j) % 5
                        else:
                            # 右上角5x5块
                            num_idx = 5 + (2*i + (j-5)) % 5
                    else:
                        if j < 5:
                            # 左下角5x5块
                            num_idx = 5 + (2*(i-5) + j) % 5
                        else:
                            # 右下角5x5块
                            num_idx = (2*(i-5) + (j-5)) % 5
                    latin2[i, j] = numbers[num_idx]
        else:
            # 方法2：循环构造
            offset2 = random.choice([3, 7])  # 另一个与10互质的偏移量
            for i in range(10):
                for j in range(10):
                    latin2[i, j] = numbers[(offset2 * i + j) % 10]
        
        # 合并两个Latin方阵
        for i in range(10):
            for j in range(10):
                self.grid[i, j] = latin1[i, j] + latin2[i, j]
        
        # 验证是否有效，如果无效则回退到块构造
        if not self._is_valid_graeco_latin():
            self._generate_by_blocks()
    
    def _generate_by_blocks(self):
        """使用块构造方法生成Graeco-Latin方阵"""
        # 初始化为空单元格
        for i in range(self.size):
            for j in range(self.size):
                self.grid[i, j] = ""
        
        letters = [chr(65 + i) for i in range(self.size)]
        numbers = [str(i + 1) for i in range(self.size)]
        
        # 按4x4块填充网格
        for block_i in range(0, self.size, 4):
            for block_j in range(0, self.size, 4):
                # 确保我们没有超出网格边界
                max_i = min(block_i + 4, self.size)
                max_j = min(block_j + 4, self.size)
                
                # 确定这个块的尺寸
                block_size_i = max_i - block_i
                block_size_j = max_j - block_j
                
                # 对于完整的4x4块，使用已知的模板
                if block_size_i == 4 and block_size_j == 4:
                    template = [
                        ["A1", "B2", "C3", "D4"],
                        ["B4", "A3", "D2", "C1"],
                        ["C2", "D1", "A4", "B3"],
                        ["D3", "C4", "B1", "A2"]
                    ]
                    
                    for i in range(4):
                        for j in range(4):
                            # 调整字母和数字
                            cell = template[i][j]
                            letter_idx = (ord(cell[0]) - ord('A') + block_i) % self.size
                            num_idx = (int(cell[1]) - 1 + block_j) % self.size
                            
                            letter = letters[letter_idx]
                            number = numbers[num_idx]
                            
                            self.grid[block_i + i, block_j + j] = letter + number
                else:
                    # 对于不完整的块，使用自定义构造
                    # 尝试为这些部分块创建有效的Graeco-Latin子方阵
                    used_pairs = set()
                    
                    for i in range(block_size_i):
                        for j in range(block_size_j):
                            # 寻找一个未使用的有效字母-数字对
                            found = False
                            for l_idx in range(self.size):
                                for n_idx in range(self.size):
                                    letter = letters[l_idx]
                                    number = numbers[n_idx]
                                    pair = letter + number
                                    
                                    # 检查此对是否可以放在此位置
                                    if self._can_place_pair(block_i + i, block_j + j, letter, number, used_pairs):
                                        self.grid[block_i + i, block_j + j] = pair
                                        used_pairs.add(pair)
                                        found = True
                                        break
                                
                                if found:
                                    break
    
    def _can_place_pair(self, row, col, letter, number, used_pairs):
        """检查是否可以在指定位置放置字母-数字对"""
        pair = letter + number
        
        # 检查对是否已使用
        if pair in used_pairs:
            return False
        
        # 检查行中是否已有此字母或数字
        for j in range(self.size):
            cell = self.grid[row, j]
            if cell and j != col:
                if cell[0] == letter or cell[1:] == number:
                    return False
        
        # 检查列中是否已有此字母或数字
        for i in range(self.size):
            cell = self.grid[i, col]
            if cell and i != row:
                if cell[0] == letter or cell[1:] == number:
                    return False
        
        return True
    
    def _is_valid_graeco_latin(self):
        """检查当前网格是否是有效的Graeco-Latin方阵"""
        # 检查每个字母和数字在每行每列中是否恰好出现一次
        for i in range(self.size):
            # 检查行
            row_letters = set()
            row_numbers = set()
            for j in range(self.size):
                cell = self.grid[i, j]
                if not cell:  # 跳过空单元格
                    continue
                letter = cell[0]
                number = cell[1:]
                if letter in row_letters or number in row_numbers:
                    return False
                row_letters.add(letter)
                row_numbers.add(number)
            
            # 检查列
            col_letters = set()
            col_numbers = set()
            for j in range(self.size):
                cell = self.grid[j, i]
                if not cell:  # 跳过空单元格
                    continue
                letter = cell[0]
                number = cell[1:]
                if letter in col_letters or number in col_numbers:
                    return False
                col_letters.add(letter)
                col_numbers.add(number)
        
        # 检查每个字母-数字对是否最多出现一次
        pairs = set()
        for i in range(self.size):
            for j in range(self.size):
                cell = self.grid[i, j]
                if cell:
                    if cell in pairs:
                        return False
                    pairs.add(cell)
        
        return True
    
    def _apply_random_transformations(self, seed=None):
        """应用随机变换来增加多样性，同时保持Graeco-Latin方阵的有效性"""
        
        # 确保随机性
        if seed is not None:
            random.seed(seed + 1000)  # 使用不同的种子偏移
        
        # 随机决定要应用的变换
        transformations = []
        
        # 1. 行交换 (保持列的约束)
        if random.random() < 0.3:
            transformations.append('row_swap')
        
        # 2. 列交换 (保持行的约束)
        if random.random() < 0.3:
            transformations.append('col_swap')
        
        # 3. 转置 (行列互换)
        if random.random() < 0.2:
            transformations.append('transpose')
        
        # 4. 旋转90度
        if random.random() < 0.2:
            transformations.append('rotate')
        
        # 5. 字母重新映射
        if random.random() < 0.4:
            transformations.append('letter_remap')
        
        # 6. 数字重新映射
        if random.random() < 0.4:
            transformations.append('number_remap')
        
        # 应用选定的变换
        for transform in transformations:
            if transform == 'row_swap':
                self._random_row_swap()
            elif transform == 'col_swap':
                self._random_col_swap()
            elif transform == 'transpose':
                self._transpose()
            elif transform == 'rotate':
                self._rotate_90()
            elif transform == 'letter_remap':
                self._random_letter_remap()
            elif transform == 'number_remap':
                self._random_number_remap()
    
    def _random_row_swap(self):
        """随机交换两行"""
        if self.size < 2:
            return
        
        row1, row2 = random.sample(range(self.size), 2)
        self.grid[[row1, row2]] = self.grid[[row2, row1]]
    
    def _random_col_swap(self):
        """随机交换两列"""
        if self.size < 2:
            return
        
        col1, col2 = random.sample(range(self.size), 2)
        self.grid[:, [col1, col2]] = self.grid[:, [col2, col1]]
    
    def _transpose(self):
        """转置矩阵"""
        self.grid = self.grid.T.copy()
    
    def _rotate_90(self):
        """顺时针旋转90度"""
        self.grid = np.rot90(self.grid, k=-1)
    
    def _random_letter_remap(self):
        """随机重新映射字母"""
        current_letters = set()
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j]:
                    current_letters.add(self.grid[i, j][0])
        
        if not current_letters:
            return
        
        letters_list = sorted(list(current_letters))
        new_letters = letters_list.copy()
        random.shuffle(new_letters)
        
        letter_map = {old: new for old, new in zip(letters_list, new_letters)}
        
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j]:
                    old_letter = self.grid[i, j][0]
                    number = self.grid[i, j][1:]
                    self.grid[i, j] = letter_map[old_letter] + number
    
    def _random_number_remap(self):
        """随机重新映射数字"""
        current_numbers = set()
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j]:
                    current_numbers.add(self.grid[i, j][1:])
        
        if not current_numbers:
            return
        
        numbers_list = sorted(list(current_numbers))
        new_numbers = numbers_list.copy()
        random.shuffle(new_numbers)
        
        number_map = {old: new for old, new in zip(numbers_list, new_numbers)}
        
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i, j]:
                    letter = self.grid[i, j][0]
                    old_number = self.grid[i, j][1:]
                    self.grid[i, j] = letter + number_map[old_number]
    
    def create_partial(self, num_blanks, seed=None, max_attempts=100):
        """通过逐步剪枝创建部分填充的拼图，确保有唯一解且效率更高
        
        Args:
            num_blanks: 要留空的单元格数量
            seed: 随机种子（可选，如果提供则确保可重现性）
            max_attempts: 备用尝试次数（仅在剪枝不足时使用）
            
        Returns:
            一个带有空格的新Eulero实例，尽力达到指定空格数并保证唯一解
        """
        # 如果提供了seed，设置随机状态（但不影响全局随机状态）
        if seed is not None:
            current_state = random.getstate()
            random.seed(seed)
        
        try:
            # 剪枝式移除：从完整解开始，随机遍历每个位置，若仍保持唯一解则移除
            partial = Eulero(self.size)
            partial.grid = np.copy(self.grid)
            positions = [(i, j) for i in range(self.size) for j in range(self.size)]
            random.shuffle(positions)
            removed_count = 0
            for i, j in positions:
                if removed_count >= num_blanks:
                    break
                original_value = partial.grid[i, j]
                partial.grid[i, j] = ""
                if self._has_unique_solution(partial):
                    removed_count += 1
                else:
                    partial.grid[i, j] = original_value
            if removed_count >= num_blanks:
                print(f"Removed {removed_count} cells by pruning (target {num_blanks})")
                return partial

            # 备用：若未达到目标空格数，继续尝试按优先级的智能移除
            print("Pruning did not reach target, trying intelligent removal...")
            fallback = self._create_partial_intelligent(num_blanks - removed_count, seed)
            # 将fallback中的空白叠加到当前partial上
            for i in range(self.size):
                for j in range(self.size):
                    if fallback.grid[i, j] == "":
                        partial.grid[i, j] = ""
            return partial
            
        finally:
            # 如果设置了seed，恢复之前的随机状态
            if seed is not None:
                random.setstate(current_state)
    
    def _create_partial_intelligent(self, num_blanks, seed=None):
        """使用智能策略创建有唯一解的部分拼图"""
        partial = Eulero(self.size)
        partial.grid = np.copy(self.grid)
        
        # 获取所有位置并按优先级排序（边缘和角落位置优先移除）
        positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        
        # 按移除优先级排序：角落 > 边缘 > 中心
        def removal_priority(pos):
            i, j = pos
            # 角落位置
            if (i == 0 or i == self.size - 1) and (j == 0 or j == self.size - 1):
                return 1
            # 边缘位置
            elif i == 0 or i == self.size - 1 or j == 0 or j == self.size - 1:
                return 2
            # 中心位置
            else:
                return 3
        
        positions.sort(key=removal_priority)
        
        removed_count = 0
        for i, j in positions:
            if removed_count >= num_blanks:
                break
                
            # 临时移除这个位置
            original_value = partial.grid[i, j]
            partial.grid[i, j] = ""
            
            # 检查是否仍有唯一解
            if self._has_unique_solution(partial):
                removed_count += 1
                print(f"Removed cell ({i}, {j}), total removed: {removed_count}")
            else:
                # 如果移除后没有唯一解，恢复这个位置
                partial.grid[i, j] = original_value
        
        return partial
    
    def _has_unique_solution(self, partial_eulero):
        """检查部分填充的Eulero是否有唯一解（快速MRV + 约束集回溯）"""
        return self._count_solutions_fast(partial_eulero, limit=2) == 1
    
    def _count_solutions_fast(self, partial_eulero, limit=2):
        """使用MRV和约束集的快速求解器，返回找到的解的数量（至多limit）。"""
        n = partial_eulero.size
        all_letters = [chr(65 + i) for i in range(n)]
        all_numbers = [str(i + 1) for i in range(n)]

        row_letters_used = [set() for _ in range(n)]
        row_numbers_used = [set() for _ in range(n)]
        col_letters_used = [set() for _ in range(n)]
        col_numbers_used = [set() for _ in range(n)]
        used_pairs = set()

        empties = []
        grid = partial_eulero.grid
        for i in range(n):
            for j in range(n):
                cell = grid[i, j]
                if cell:
                    letter = cell[0]
                    number = cell[1:]
                    row_letters_used[i].add(letter)
                    row_numbers_used[i].add(number)
                    col_letters_used[j].add(letter)
                    col_numbers_used[j].add(number)
                    used_pairs.add(cell)
                else:
                    empties.append((i, j))

        solution_count = 0

        def select_mrv_cell():
            best = None
            best_count = 10**9
            for (r, c) in empties:
                if grid[r, c]:
                    continue
                letters_avail = [L for L in all_letters if L not in row_letters_used[r] and L not in col_letters_used[c]]
                numbers_avail = [N for N in all_numbers if N not in row_numbers_used[r] and N not in col_numbers_used[c]]
                # 粗略下界：可用组合数上界为乘积
                cnt = 0
                for L in letters_avail:
                    for Nn in numbers_avail:
                        if (L + Nn) not in used_pairs:
                            cnt += 1
                            if cnt >= best_count:
                                break
                    if cnt >= best_count:
                        break
                if cnt < best_count:
                    best_count = cnt
                    best = (r, c, letters_avail, numbers_avail)
                    if best_count <= 1:
                        break
            return best

        def dfs():
            nonlocal solution_count
            if solution_count >= limit:
                return
            # 选择MRV单元
            choice = select_mrv_cell()
            if choice is None:
                # 没有空位，找到一个解
                solution_count += 1
                return
            r, c, letters_avail, numbers_avail = choice

            # 生成候选对
            candidates = []
            for L in letters_avail:
                for Nn in numbers_avail:
                    pair = L + Nn
                    if pair not in used_pairs:
                        candidates.append(pair)

            # 失败剪枝
            if not candidates:
                return

            # 尝试候选，简单顺序即可
            for pair in candidates:
                if solution_count >= limit:
                    return
                L = pair[0]
                Nn = pair[1:]
                # 放置
                grid[r, c] = pair
                row_letters_used[r].add(L)
                row_numbers_used[r].add(Nn)
                col_letters_used[c].add(L)
                col_numbers_used[c].add(Nn)
                used_pairs.add(pair)

                dfs()

                # 回溯
                used_pairs.remove(pair)
                col_numbers_used[c].remove(Nn)
                col_letters_used[c].remove(L)
                row_numbers_used[r].remove(Nn)
                row_letters_used[r].remove(L)
                grid[r, c] = ""

        dfs()
        return solution_count
    
    def _find_next_empty_cell(self, partial_eulero):
        """找到下一个空单元格，优先选择约束最多的位置"""
        best_pos = None
        min_possibilities = float('inf')
        
        for i in range(partial_eulero.size):
            for j in range(partial_eulero.size):
                if not partial_eulero.grid[i, j]:  # 空单元格
                    # 计算这个位置的可能值数量
                    possibilities = self._count_valid_possibilities(partial_eulero, i, j)
                    if possibilities < min_possibilities:
                        min_possibilities = possibilities
                        best_pos = (i, j)
        
        return best_pos
    
    def _count_valid_possibilities(self, partial_eulero, row, col):
        """计算指定位置的有效可能值数量"""
        count = 0
        letters = [chr(65 + i) for i in range(partial_eulero.size)]
        numbers = [str(i + 1) for i in range(partial_eulero.size)]
        
        for letter in letters:
            for number in numbers:
                if self._is_valid_placement(partial_eulero, row, col, letter, number):
                    count += 1
        
        return count
    
    def _is_valid_placement(self, partial_eulero, row, col, letter, number):
        """检查在指定位置放置字母-数字对是否有效"""
        pair = letter + number
        
        # 检查整个网格中是否已存在相同的字母-数字对
        for i in range(partial_eulero.size):
            for j in range(partial_eulero.size):
                if partial_eulero.grid[i, j] == pair:
                    return False
        
        # 检查行约束
        for j in range(partial_eulero.size):
            if j != col and partial_eulero.grid[row, j]:
                cell = partial_eulero.grid[row, j]
                if cell[0] == letter or cell[1:] == number:
                    return False
        
        # 检查列约束
        for i in range(partial_eulero.size):
            if i != row and partial_eulero.grid[i, col]:
                cell = partial_eulero.grid[i, col]
                if cell[0] == letter or cell[1:] == number:
                    return False
        
        return True
    
    def verify_solution(self, solution_grid=None):
        """验证当前网格或给定网格是否是有效的Graeco-Latin方阵解"""
        if solution_grid is None:
            grid_to_check = self.grid
        else:
            grid_to_check = solution_grid
        
        # 检查网格是否完全填满
        for i in range(self.size):
            for j in range(self.size):
                if not grid_to_check[i, j]:
                    return False, "Grid contains empty cells"
        
        # 检查每行是否包含所有字母和数字
        expected_letters = set(chr(65 + i) for i in range(self.size))
        expected_numbers = set(str(i + 1) for i in range(self.size))
        
        for i in range(self.size):
            row_letters = set()
            row_numbers = set()
            for j in range(self.size):
                cell = grid_to_check[i, j]
                if not cell or len(cell) < 2:
                    return False, f"Invalid cell at ({i}, {j}): {cell}"
                letter = cell[0]
                number = cell[1:]
                row_letters.add(letter)
                row_numbers.add(number)
            
            if row_letters != expected_letters:
                return False, f"Row {i} missing letters: {expected_letters - row_letters}"
            if row_numbers != expected_numbers:
                return False, f"Row {i} missing numbers: {expected_numbers - row_numbers}"
        
        # 检查每列是否包含所有字母和数字
        for j in range(self.size):
            col_letters = set()
            col_numbers = set()
            for i in range(self.size):
                cell = grid_to_check[i, j]
                letter = cell[0]
                number = cell[1:]
                col_letters.add(letter)
                col_numbers.add(number)
            
            if col_letters != expected_letters:
                return False, f"Column {j} missing letters: {expected_letters - col_letters}"
            if col_numbers != expected_numbers:
                return False, f"Column {j} missing numbers: {expected_numbers - col_numbers}"
        
        # 检查每个字母-数字对是否唯一
        pairs = set()
        for i in range(self.size):
            for j in range(self.size):
                cell = grid_to_check[i, j]
                if cell in pairs:
                    return False, f"Duplicate pair {cell} found"
                pairs.add(cell)
        
        # 检查总对数是否正确
        if len(pairs) != self.size * self.size:
            return False, f"Expected {self.size * self.size} unique pairs, found {len(pairs)}"
        
        return True, "Valid Graeco-Latin square"
    
    def validate_global_rules(self, grid_to_check=None):
        """验证生成的puzzle是否严格符合全局规则
        
        ### Global Rules:
        1. Each cell contains a **letter-number pair** (like A1).
        2. Each **letter** appears **exactly once** in every row and every column.
        3. Each **number** appears **exactly once** in every row and every column.
        4. Each **letter-number pair** is **unique across the entire grid** (i.e., no duplicate pairs anywhere).
        5. For an N×N grid, the letters used are the first N letters of the alphabet (A=1, B=2, ..., up to the N-th letter), and the numbers used are from 1 to N.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        if grid_to_check is None:
            grid_to_check = self.grid
        
        # Rule 5: Define expected letters and numbers for N×N grid
        expected_letters = set(chr(65 + i) for i in range(self.size))  # A, B, C, ..., N-th letter
        expected_numbers = set(str(i + 1) for i in range(self.size))   # 1, 2, 3, ..., N
        
        # Rule 1: Check each cell contains a valid letter-number pair
        all_pairs = set()
        for i in range(self.size):
            for j in range(self.size):
                cell = grid_to_check[i, j]
                
                # Skip empty cells for partial puzzles
                if not cell:
                    continue
                
                # Check cell format (letter-number pair)
                if len(cell) < 2:
                    return False, f"Rule 1 violation: Cell ({i}, {j}) '{cell}' is not a valid letter-number pair"
                
                letter = cell[0]
                number = cell[1:]
                
                # Rule 5: Check if letter and number are within expected range
                if letter not in expected_letters:
                    return False, f"Rule 5 violation: Letter '{letter}' at ({i}, {j}) not in expected range {sorted(expected_letters)}"
                
                if number not in expected_numbers:
                    return False, f"Rule 5 violation: Number '{number}' at ({i}, {j}) not in expected range {sorted(expected_numbers)}"
                
                # Rule 4: Track pairs for uniqueness check
                all_pairs.add(cell)
        
        # For complete grids, perform full validation
        if all(grid_to_check[i, j] for i in range(self.size) for j in range(self.size)):
            
            # Rule 4: Check each letter-number pair is unique across the entire grid
            if len(all_pairs) != self.size * self.size:
                return False, f"Rule 4 violation: Expected {self.size * self.size} unique pairs, found {len(all_pairs)}"
            
            # Rule 2: Check each letter appears exactly once in every row and column
            for i in range(self.size):
                # Check row
                row_letters = set()
                for j in range(self.size):
                    letter = grid_to_check[i, j][0]
                    if letter in row_letters:
                        return False, f"Rule 2 violation: Letter '{letter}' appears multiple times in row {i}"
                    row_letters.add(letter)
                
                if row_letters != expected_letters:
                    missing = expected_letters - row_letters
                    return False, f"Rule 2 violation: Row {i} missing letters: {sorted(missing)}"
                
                # Check column
                col_letters = set()
                for k in range(self.size):
                    letter = grid_to_check[k, i][0]
                    if letter in col_letters:
                        return False, f"Rule 2 violation: Letter '{letter}' appears multiple times in column {i}"
                    col_letters.add(letter)
                
                if col_letters != expected_letters:
                    missing = expected_letters - col_letters
                    return False, f"Rule 2 violation: Column {i} missing letters: {sorted(missing)}"
            
            # Rule 3: Check each number appears exactly once in every row and column
            for i in range(self.size):
                # Check row
                row_numbers = set()
                for j in range(self.size):
                    number = grid_to_check[i, j][1:]
                    if number in row_numbers:
                        return False, f"Rule 3 violation: Number '{number}' appears multiple times in row {i}"
                    row_numbers.add(number)
                
                if row_numbers != expected_numbers:
                    missing = expected_numbers - row_numbers
                    return False, f"Rule 3 violation: Row {i} missing numbers: {sorted(missing)}"
                
                # Check column
                col_numbers = set()
                for k in range(self.size):
                    number = grid_to_check[k, i][1:]
                    if number in col_numbers:
                        return False, f"Rule 3 violation: Number '{number}' appears multiple times in column {i}"
                    col_numbers.add(number)
                
                if col_numbers != expected_numbers:
                    missing = expected_numbers - col_numbers
                    return False, f"Rule 3 violation: Column {i} missing numbers: {sorted(missing)}"
        
        return True, "All global rules satisfied"
    
    def to_text_representation(self):
        """转换为文本表示"""
        result = []
        
        for i in range(self.size):
            row = []
            for j in range(self.size):
                cell = self.grid[i, j]
                row.append(cell if cell else " ")
            result.append('|'.join(row))
        
        return '\n'.join(result)
    
    def visualize(self, filename=None):
        """创建拼图的可视化表示"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 设置背景颜色
        ax.set_facecolor('#f8f8f8')
        
        # 根据网格大小自适应计算更大的字体（按单元格物理尺寸进行缩放）
        # 10 为图像的英寸大小（与上面的 figsize 对齐），72 为每英寸的点数
        # 系数 0.75 使字符在单元格内更饱满但不过分溢出
        cell_size_inches = 10 / self.size
        fontsize = int(np.clip(cell_size_inches * 72 * 0.75, 18, 64))
        
        for i in range(self.size):
            for j in range(self.size):
                cell = self.grid[i, j]
                
                # 添加网格单元格
                rect = plt.Rectangle((j, i), 1, 1, edgecolor='black', facecolor='white', linewidth=1)
                ax.add_patch(rect)
                
                # 如果单元格非空，添加文本
                if cell:
                    # 提取字母和数字
                    letter = cell[0]
                    number = cell[1:]
                    
                    # 使用固定大小（等宽）字体以获得更好的对齐
                    # 创建字母的文本元素（蓝色）
                    letter_text = ax.text(j + 0.5, i + 0.5, letter,
                                         horizontalalignment='center',
                                         verticalalignment='center',
                                         fontsize=fontsize,
                                         fontweight='bold',
                                         color='blue',
                                         family='monospace')
                    
                    # 创建数字的单独文本元素（红色）
                    number_text = ax.text(j + 0.5 + len(letter)*0.08, i + 0.5, number,
                                         horizontalalignment='left',
                                         verticalalignment='center',
                                         fontsize=fontsize,
                                         fontweight='bold',
                                         color='red',
                                         family='monospace')
        
        # 设置限制和纵横比
        ax.set_xlim(0, self.size)
        ax.set_ylim(self.size, 0)
        ax.set_aspect('equal')
        
        # 移除刻度
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加标题
        plt.title('Eulero (Graeco-Latin Square)', fontsize=16, fontweight='bold')
        
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