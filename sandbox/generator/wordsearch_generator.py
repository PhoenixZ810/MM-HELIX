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
from utils.constants import PROMPT_MAZE_IMAGE, PROMPT_15PUZZLE_IMAGE, PROMPT_HANOI_IMAGE, PROMPT_WORDSEARCH_IMAGE, PROMPT_NUMBRIX_IMAGE, PROMPT_MINESWEEPER_IMAGE, PROMPT_EULERO_IMAGE, PROMPT_SNAKE_IMAGE
from utils.constants import PROMPT_MAZE, PROMPT_15PUZZLE, PROMPT_HANOI, PROMPT_WORDSEARCH, PROMPT_NUMBRIX, PROMPT_MINESWEEPER, PROMPT_EULERO, PROMPT_SNAKE


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
            seed: 随机种子
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



class BaseGeneratorImpl(BaseGenerator):
    def __init__(self, output_folder, training_set_path=None, global_items=None, **kwargs):
        super().__init__(output_folder)
        self.base_output_dir = output_folder
        self.task_name = self.__class__.__name__.replace('Generator', '').lower()
        self.task_dir = os.path.join(self.base_output_dir, self.task_name)

        # # Create task directory
        # os.makedirs(self.task_dir, exist_ok=True)

        # # Define image directory
        # self.image_dir = os.path.join(self.task_dir, 'images')
        # os.makedirs(self.image_dir, exist_ok=True)

        # # Define annotations file path
        # self.annotations_file = os.path.join(self.task_dir, 'annotations.json')
        
        # 加载训练集用于重复检查
        self.training_set = self._load_training_set(training_set_path) if training_set_path else []
        print(f"Loaded {len(self.training_set)} items from training set for duplicate checking")
        
        # 全局已生成的items，用于确保benchmark和training set不重复
        self.global_items = global_items if global_items is not None else set()
        
        # 当前生成的items，用于内部重复检查
        self.current_items = set()
    
    def _load_training_set(self, training_set_path):
        """加载训练集JSON文件"""
        try:
            with open(training_set_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
                print(f"Successfully loaded training set from {training_set_path}")
                return training_data
        except Exception as e:
            print(f"Warning: Could not load training set from {training_set_path}: {e}")
            return []
    
    def _is_duplicate(self, puzzle_question, puzzle_answer, puzzle_category):
        """检查是否与训练集、全局items或当前生成的items重复"""
        # 创建唯一标识符
        item_key = (puzzle_category, puzzle_question, puzzle_answer)
        
        # 检查当前生成过程中的重复
        if item_key in self.current_items:
            return True
            
        # 检查全局重复
        if item_key in self.global_items:
            return True
            
        # 检查与训练集重复
        if hasattr(self, 'training_set') and self.training_set:
            for training_item in self.training_set:
                if (training_item.get('category') == puzzle_category and 
                    training_item.get('question') == puzzle_question and
                    training_item.get('answer') == puzzle_answer):
                    return True
        return False
    
    def _add_to_current_items(self, puzzle_question, puzzle_answer, puzzle_category):
        """添加到当前生成的items集合"""
        item_key = (puzzle_category, puzzle_question, puzzle_answer)
        self.current_items.add(item_key)
        self.global_items.add(item_key)
    
    def _get_difficulty_by_step_count(self, step_count, difficulty_ranges):
        """根据step_count严格划分难度"""
        for difficulty, (min_steps, max_steps) in difficulty_ranges.items():
            if min_steps <= step_count <= max_steps:
                return difficulty
        # 如果不在任何范围内，返回中等难度
        return '3'
    
    def generate(self, num_cases, difficulty, output_folder=None):
        """
        生成问题的抽象方法，需要被子类实现。

        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别
            output_folder: 输出文件夹路径，覆盖构造函数中设置的路径

        Returns:
            生成的问题列表
        """
        raise NotImplementedError

    def _get_difficulty_params(self, difficulty):
        """
        根据难度级别获取相应的参数配置。

        Args:
            difficulty: 难度级别（1-5）

        Returns:
            dict: 包含难度参数的字典
        """
        # 默认实现，子类可以重写
        return {"difficulty": str(difficulty)}

    def generate_batch(self, num_cases, difficulty, output_folder=None):
        """
        批量生成问题并一次性存储

        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别
            output_folder: 输出文件夹路径，覆盖构造函数中设置的路径

        Returns:
            生成的问题列表
        """
        # Set seed based on timestamp
        timestamp_seed = int(time.time())
        random.seed(timestamp_seed)
        print(f"Using timestamp seed: {timestamp_seed}")

        # Use provided output folder or default
        if output_folder is None:
            output_folder = self.output_folder

        # Create directories
        images_dir = os.path.join(output_folder, 'images')
        os.makedirs(images_dir, exist_ok=True)
        annotations_file = os.path.join(output_folder, 'annotations.json')

        # Collect all puzzles
        all_puzzles = []

        # Generate puzzles
        for i in range(num_cases):
            try:
                # Generate single puzzle with unique seed
                case_seed = timestamp_seed + i
                puzzles = self.generate(num_cases=1, difficulty=difficulty, seed=case_seed, output_folder=output_folder)
                if puzzles:
                    all_puzzles.extend(puzzles)
            except Exception as e:
                print(f"Error generating puzzle {i+1}: {e}")
                continue

        # Save all puzzles at once
        if all_puzzles:
            self.save_to_json_batch(all_puzzles, annotations_file)
            print(f"Successfully generated and saved {len(all_puzzles)} puzzles to {output_folder}")

        return all_puzzles

    def save_to_json_batch(self, puzzles, filename):
        """
        批量保存所有puzzle数据到JSON文件（一次性写入）。

        Args:
            puzzles: 要保存的puzzle列表
            filename: JSON文件路径
        """
        # Normalize input to list
        if isinstance(puzzles, dict):
            puzzles = [puzzles]

        # Ensure puzzles have the required fields
        annotations_dir = os.path.dirname(os.path.abspath(filename))
        for puzzle in puzzles:
            # Make image paths relative to annotations file directory
            if 'image' in puzzle and puzzle['image']:
                try:
                    puzzle['image'] = os.path.relpath(os.path.abspath(os.path.join(annotations_dir, puzzle['image'])), annotations_dir)
                except Exception:
                    # Fallback: keep original
                    pass

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

        # Save all puzzles at once
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(puzzles, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(puzzles)} puzzles to {filename}")

    def visualize(self, puzzle, **kwargs):
        raise NotImplementedError

    def solve(self, puzzle, **kwargs):
        raise NotImplementedError
        
    def save_to_json(self, puzzles, filename=None):
        """Save puzzle data to JSON file incrementally (single save point).

        - Appends new items to existing JSON if present.
        - Normalizes fields and paths relative to the annotations file directory.
        """
        if filename is None:
            filename = self.annotations_file

        # Normalize input to list
        if isinstance(puzzles, dict):
            puzzles = [puzzles]

        # Ensure puzzles have the required fields
        annotations_dir = os.path.dirname(os.path.abspath(filename))
        for puzzle in puzzles:
            # Make image paths relative to annotations file directory
            if 'image' in puzzle and puzzle['image']:
                try:
                    puzzle['image'] = os.path.relpath(os.path.abspath(os.path.join(annotations_dir, puzzle['image'])), annotations_dir)
                except Exception:
                    # Fallback: keep original
                    pass

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

        # Load existing data (if any)
        existing_data = []
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        existing_data = loaded
            except Exception:
                existing_data = []

        # Append and save back
        existing_data.extend(puzzles)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(puzzles)} puzzles to {filename} (total {len(existing_data)})")

class WordSearchGenerator(BaseGeneratorImpl):
    def __init__(self, output_folder, training_set_path=None, **kwargs):
        super().__init__(output_folder, training_set_path=training_set_path, **kwargs)
        self.word_search_available = False
        try:
            from word_search_generator import WordSearch
            self.word_search_available = True
        except ImportError:
            print("Warning: word_search_generator package not installed. WordSearchGenerator will use mock data.")

    def generate(self, num_cases, difficulty, output_folder=None, seed=None, **kwargs):
        """
        生成WordSearch问题并保存到指定目录的annotations.json与images/下。

        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别（1-5）
            output_folder: 输出文件夹路径；若为None则使用构造函数的输出目录
            seed: 可选的随机种子；未提供时使用时间戳种子

        Returns:
            list: 生成的问题条目列表
        """
        # 获取参数
        params = self._get_difficulty_params(difficulty)
        size = params.get('size', 10)

        # 准备输出目录
        if output_folder is None:
            output_folder = self.output_folder
        images_dir = os.path.join(output_folder, 'images')
        os.makedirs(images_dir, exist_ok=True)
        annotations_file = os.path.join(output_folder, 'annotations.json')

        # 设定基础随机种子（用于批量内可复现的多样化）
        timestamp_seed = int(time.time()) if seed is None else int(seed)
        random.seed(timestamp_seed)

        # 词表
        word_list = [
            "dog", "cat", "pig", "horse", "donkey", "turtle", "goat", "sheep",
            "cow", "lion", "tiger", "elephant", "giraffe", "zebra", "monkey", "bird",
            "fish", "bear", "deer", "wolf", "fox", "rabbit", "mouse", "duck", "bee"
        ]

        # 过滤适配尺寸的词
        suitable_words = [w for w in word_list if len(w) <= size] or ["cat"]

        all_entries = []

        for i in range(num_cases):
            case_seed = timestamp_seed + i

            # 选择词并生成网格
            selected_word = suitable_words[case_seed % len(suitable_words)]
            grid = self._generate_word_search_grid(size, selected_word, case_seed)

            # initial_state
            initial_state = [row[:] for row in grid]

            # 问题文本
            question_text = f"Find the hidden word in this {size}x{size} grid.\n" + str(initial_state)

            # 生成图片
            puzzle_id = f"wordsearch_{size}_{case_seed}"
            image_filename = f"images/{puzzle_id}.png"
            image_path = os.path.join(output_folder, image_filename)
            self._generate_puzzle_image(grid, [selected_word], image_path, size)

            # 答案位置与CoT
            word_position = self._find_word_position(grid, selected_word)
            cot_data = self.generate_cot(initial_state, [selected_word], word_position)

            entry = {
                "index": puzzle_id,
                "category": "wordsearch",
                "image": image_filename,
                "question": PROMPT_WORDSEARCH_IMAGE,
                "question_language": PROMPT_WORDSEARCH.format(question_text),
                "answer": f"{selected_word.upper()} {word_position} @",
                "initial_state": initial_state,
                "difficulty": str(difficulty),
                "cot": cot_data['full_cot'],
                "cot_step1_all": cot_data['cot_step1_all'],
                "cot_step2_all": cot_data['cot_step2_all'],
                "cot_step3_all": cot_data['cot_step3_all']
            }

            print(f"Generated word search puzzle {puzzle_id} with word '{selected_word}' at {word_position}")
            all_entries.append(entry)

        # 批量保存到annotations.json
        if all_entries:
            self.save_to_json_batch(all_entries, annotations_file)

        return all_entries

    def _get_difficulty_params(self, difficulty):
        """
        根据难度级别获取WordSearch的相应参数配置。

        Args:
            difficulty: 难度级别（1-5）

        Returns:
            dict: 包含难度参数的字典
        """
        difficulty_config = {
            1: {"size": 5},   # Very easy
            2: {"size": 7},   # Easy
            3: {"size": 10},  # Medium
            4: {"size": 12},  # Hard
            5: {"size": 15}   # Very hard
        }
        return difficulty_config.get(difficulty, {"size": 10})
    
    def _generate_word_search_grid(self, size, word, seed):
        """Generate a word search grid with the given word placed randomly"""
        # Create random state for this specific grid
        local_random = random.Random(seed)
        
        # Initialize grid with random letters
        grid = []
        for i in range(size):
            row = []
            for j in range(size):
                row.append(local_random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
            grid.append(row)
        
        word = word.upper()
        
        # Possible directions: horizontal, vertical, diagonal
        directions = [
            (0, 1),   # horizontal right
            (1, 0),   # vertical down
            (1, 1),   # diagonal down-right
            (-1, 1),  # diagonal up-right
        ]
        
        # Try to place the word
        placed = False
        attempts = 0
        max_attempts = 100
        
        while not placed and attempts < max_attempts:
            direction = local_random.choice(directions)
            dr, dc = direction
            
            # Calculate valid starting positions
            # For a word of length L, it occupies positions from start to start + (L-1)*direction
            word_length = len(word)
            
            if dr >= 0:
                start_row_min = 0
                start_row_max = size - 1 - (word_length - 1) * dr
            else:
                start_row_min = (word_length - 1) * abs(dr)
                start_row_max = size - 1
                
            if dc >= 0:
                start_col_min = 0
                start_col_max = size - 1 - (word_length - 1) * dc
            else:
                start_col_min = (word_length - 1) * abs(dc)
                start_col_max = size - 1
            
            if start_row_max >= start_row_min and start_col_max >= start_col_min:
                start_row = local_random.randint(start_row_min, start_row_max)
                start_col = local_random.randint(start_col_min, start_col_max)
                
                # Place the word
                for i, char in enumerate(word):
                    row = start_row + i * dr
                    col = start_col + i * dc
                    grid[row][col] = char
                
                placed = True
            
            attempts += 1
        
        return grid
    
    def _find_word_position(self, grid, word):
        """Find the position and direction of a word in the grid"""
        word = word.upper()
        size = len(grid)
        
        # Possible directions
        directions = [
            (0, 1, 'E'),   # horizontal right
            (1, 0, 'S'),   # vertical down
            (1, 1, 'SE'),  # diagonal down-right
            (-1, 1, 'NE'), # diagonal up-right
        ]
        
        for start_row in range(size):
            for start_col in range(size):
                for dr, dc, dir_name in directions:
                    # Check if word fits in this direction
                    valid = True
                    end_row = start_row + (len(word) - 1) * dr
                    end_col = start_col + (len(word) - 1) * dc
                    
                    if (end_row < 0 or end_row >= size or 
                        end_col < 0 or end_col >= size):
                        continue
                    
                    # Check if word matches
                    match = True
                    for i, char in enumerate(word):
                        row = start_row + i * dr
                        col = start_col + i * dc
                        if grid[row][col] != char:
                            match = False
                            break
                    
                    if match:
                        return f"{dir_name} ({start_col + 1}, {start_row + 1})"
        
        return "Not found"
    
    def generate_cot(self, grid, target_words, word_position):
        """Generate enhanced chain-of-thought reasoning for word search puzzle with comprehensive 4-step approach."""
        size = len(grid)
        word = target_words[0].upper()

        # Helper to truncate a step body at a word boundary around half length
        def truncate_body_half_words(text: str) -> str:
            words = text.split()
            if len(words) <= 1:
                return text
            cut = max(1, len(words) // 2)
            return " ".join(words[:cut]) + " ..."

        # Introduction
        intro = "I will systematically solve this word search puzzle through careful analysis and step-by-step reasoning.\n\n"

        # ================================
        # STEP 1: Understanding Game Rules
        # ================================
        step1_body = "### Step 1: Understanding the Game Rules and Objectives\n\n"
        
        step1_body += "**Core Game Mechanics:**\n"
        step1_body += f"- This is a {size}×{size} word search puzzle\n"
        step1_body += f"- Objective: Find the hidden word '{word}' within the letter grid\n"
        step1_body += "- The target word can be placed in any of 8 possible directions\n\n"
        
        step1_body += "**Valid Search Directions:**\n"
        step1_body += "- Horizontal: Left-to-right (E) or Right-to-left (W)\n"
        step1_body += "- Vertical: Top-to-bottom (S) or Bottom-to-top (N)\n"
        step1_body += "- Diagonal: NE, NW, SE, SW (4 diagonal directions)\n\n"
        
        step1_body += "**Key Constraints:**\n"
        step1_body += "- Letters must form a continuous straight line\n"
        step1_body += "- No gaps or breaks allowed between letters\n"
        step1_body += "- The word must fit entirely within the grid boundaries\n"
        step1_body += "- Each letter position must match exactly with the target word\n\n"
        
        step1_body += f"**Target Analysis:**\n"
        step1_body += f"- Word to find: '{word}'\n"
        step1_body += f"- Length: {len(word)} letters\n"
        step1_body += f"- Letter sequence: {' → '.join(list(word))}\n"
        step1_body += f"- First letter to look for: '{word[0]}'\n"

        # ================================================
        # STEP 2: Reading Image and Extracting Game State
        # ================================================
        step2_body = "### Step 2: Carefully Reading the Image and Extracting the Initial Game State\n\n"
        
        step2_body += "**Grid State Extraction:**\n"
        step2_body += f"I will now carefully read the {size}×{size} grid from the image, scanning each row systematically:\n\n"
        
        # Read grid row by row with detailed description
        for i, row in enumerate(grid):
            step2_body += f"Row {i+1}: {' '.join(row)}\n"
        
        step2_body += "\n**Initial State Verification:**\n"
        step2_body += "Let me double-check my reading by examining the grid structure:\n"
        step2_body += f"- Grid dimensions: {size} rows × {size} columns ✓\n"
        step2_body += f"- Total cells: {size * size}\n"
        step2_body += "- All cells contain single uppercase letters ✓\n\n"
        
        # Letter frequency analysis
        letter_count = {}
        for row in grid:
            for letter in row:
                letter_count[letter] = letter_count.get(letter, 0) + 1
        
        step2_body += "**Letter Frequency Analysis:**\n"
        sorted_letters = sorted(letter_count.items(), key=lambda x: x[1], reverse=True)
        for letter, count in sorted_letters:
            step2_body += f"- '{letter}': appears {count} time{'s' if count != 1 else ''}\n"
        
        # Identify potential starting positions
        first_letter_positions = [(r, c) for r in range(size) for c in range(size) if grid[r][c] == word[0]]
        step2_body += f"\n**Target Word Analysis:**\n"
        step2_body += f"- Searching for: '{word}' ({len(word)} letters)\n"
        step2_body += f"- First letter '{word[0]}' appears at positions: {[(c+1, r+1) for r, c in first_letter_positions]}\n"
        step2_body += f"- These will be my starting anchor points for systematic exploration\n\n"
        
        step2_body += "**State Reading Reflection:**\n"
        step2_body += "I have successfully extracted the complete grid state from the image. "
        step2_body += f"The grid is clearly readable with {len(first_letter_positions)} potential starting positions for '{word[0]}'. "
        step2_body += "Now I can proceed with systematic word searching."

        # ======================================
        # STEP 3: Detailed Reasoning Process
        # ======================================
        step3_body = "### Step 3: Systematic Exploration and Detailed Reasoning Process\n\n"
        
        step3_body += "**Search Strategy:**\n"
        step3_body += f"I will systematically examine each occurrence of '{word[0]}' as a potential starting point, "
        step3_body += "testing all 8 directions from each position. For each direction, I'll:\n"
        step3_body += "1. Check if the word would fit within grid boundaries\n"
        step3_body += "2. Verify each subsequent letter matches the target sequence\n"
        step3_body += "3. Stop immediately upon finding a mismatch (backtrack)\n"
        step3_body += "4. Confirm complete word match if all letters align\n\n"
        
        directions = [
            (-1, -1, 'NW'), (-1, 0, 'N'), (-1, 1, 'NE'),
            (0, -1, 'W'),                 (0, 1, 'E'),
            (1, -1, 'SW'),  (1, 0, 'S'),  (1, 1, 'SE')
        ]
        
        step3_body += "**Detailed Exploration Process:**\n\n"
        
        found_solution = False
        solution_details = ""
        
        # Systematic exploration of each anchor point
        for anchor_index, (start_row, start_col) in enumerate(first_letter_positions):
            step3_body += f"**Anchor Point {anchor_index + 1}: Position ({start_col+1}, {start_row+1})**\n"
            step3_body += f"Found '{word[0]}' at grid position ({start_col+1}, {start_row+1}). Testing all directions:\n\n"
            
            exploration_results = []
            
            for direction_index, (dr, dc, dir_name) in enumerate(directions):
                # Check boundary constraints first
                end_row = start_row + (len(word) - 1) * dr
                end_col = start_col + (len(word) - 1) * dc
                
                if end_row < 0 or end_row >= size or end_col < 0 or end_col >= size:
                    exploration_results.append(f"  {dir_name}: Cannot fit - would extend outside grid boundaries")
                    continue
                
                # Attempt to trace the word in this direction
                step3_body += f"  Direction {dir_name}: "
                formed_letters = []
                letter_positions = []
                is_valid_path = True
                mismatch_position = -1
                
                for i in range(len(word)):
                    r = start_row + i * dr
                    c = start_col + i * dc
                    found_letter = grid[r][c]
                    expected_letter = word[i]
                    
                    formed_letters.append(found_letter)
                    letter_positions.append(f"({c+1},{r+1})")
                    
                    if found_letter != expected_letter:
                        is_valid_path = False
                        mismatch_position = i
                        break
                
                if is_valid_path:
                    # Found the complete word!
                    step3_body += f"✓ SUCCESS! Found '{word}'\n"
                    step3_body += f"    Path: {' → '.join([f'{formed_letters[i]}@{letter_positions[i]}' for i in range(len(word))])}\n"
                    step3_body += f"    Complete word: {''.join(formed_letters)}\n"
                    found_solution = True
                    solution_details = f"'{word}' found starting at ({start_col+1}, {start_row+1}) going {dir_name}"
                    break
                else:
                    step3_body += f"✗ Mismatch at position {mismatch_position + 1}\n"
                    step3_body += f"    Expected: {word[mismatch_position]}, Found: {formed_letters[mismatch_position]}\n"
                    step3_body += f"    Partial sequence: {''.join(formed_letters[:mismatch_position + 1])}\n"
            
            if found_solution:
                step3_body += f"\n**Solution Found!** {solution_details}\n"
                step3_body += "Terminating search as the target word has been successfully located.\n"
                break
            else:
                step3_body += f"No valid word found from anchor point ({start_col+1}, {start_row+1})\n\n"
        
        if not found_solution:
            step3_body += "\n**Search Completion:**\n"
            step3_body += f"Exhausted all {len(first_letter_positions)} anchor points. "
            step3_body += "Re-checking grid reading and target word specification...\n"

        # =======================================
        # STEP 4: Validation and Verification
        # =======================================
        step4_body = "### Step 4: Solution Validation and Final Verification\n\n"
        
        step4_body += "**Solution Verification Process:**\n"
        step4_body += f"I have identified the word '{word}' at position {word_position}. "
        step4_body += "Let me thoroughly verify this solution:\n\n"
        
        # Parse the word_position to extract details
        if "@" in word_position:
            position_parts = word_position.split("@")
            if len(position_parts) == 2:
                direction_info = position_parts[0].strip()
                step4_body += f"**Position Analysis:**\n"
                step4_body += f"- Word: {word}\n"
                step4_body += f"- Location: {direction_info}\n\n"
        
        step4_body += "**Constraint Verification:**\n"
        step4_body += "1. **Boundary Check:** ✓ The word fits entirely within the grid boundaries\n"
        step4_body += "2. **Continuity Check:** ✓ All letters form a continuous straight line\n"
        step4_body += "3. **Letter Matching:** ✓ Each position contains the exact expected letter\n"
        step4_body += "4. **Direction Validity:** ✓ The word follows one of the 8 valid compass directions\n\n"
        
        step4_body += "**Cross-Verification:**\n"
        step4_body += "I re-examined the identified path letter by letter to ensure:\n"
        step4_body += "- No calculation errors in position coordinates\n"
        step4_body += "- Correct directional vector application\n"
        step4_body += "- Accurate letter-to-letter matching\n"
        step4_body += "- Complete word formation without gaps\n\n"
        
        step4_body += "**Final Validation:**\n"
        step4_body += f"The solution has been rigorously verified. The word '{word}' is correctly located "
        step4_body += f"at {word_position}. All game rules and constraints are satisfied.\n\n"
        
        step4_body += f"**Final Answer:** {word} {word_position} @"

        # Build cumulative ALL strings
        step1_all = intro + step1_body
        step2_all = step1_all + "\n\n" + step2_body
        step3_all = step2_all + "\n\n" + step3_body
        step4_all = step3_all + "\n\n" + step4_body

        # Build PART strings (truncate only the last step body for steps 1..3)
        step1_part = intro + truncate_body_half_words(step1_body)
        step2_part = step1_all + "\n\n" + truncate_body_half_words(step2_body)
        step3_part = step2_all + "\n\n" + truncate_body_half_words(step3_body)

        cot_data = {
            'full_cot': step4_all,
            'cot_step1_all': step1_all,
            'cot_step1_part': step1_part,
            'cot_step2_all': step2_all,
            'cot_step2_part': step2_part,
            'cot_step3_all': step3_all,
            'cot_step3_part': step3_part,
        }

        return cot_data
    
    def _generate_puzzle_image(self, grid, target_words, image_path, grid_size):
        """Generate puzzle image using the existing image generation logic"""
        return self._generate_mock_image(grid, target_words, image_path, grid_size)
    
    def _generate_mock_image(self, grid, target_words, filename, grid_size=5, width=1200, height=1200):
        """生成字母查找图：网格与字母，并在顶部添加“Find these words: ...”提示。"""
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        # 计算单元格尺寸与边距（保留少量边缘空白）
        cell_size = min((width - 200) // grid_size, (height - 200) // grid_size)
        margin_x = (width - grid_size * cell_size) // 2
        margin_y = (height - grid_size * cell_size) // 2

        # 字体
        main_font_size = max(28, int(cell_size * 0.8))
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", main_font_size)
        except IOError:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", main_font_size)
            except IOError:
                font = ImageFont.load_default()

        # 顶部提示文字字体（相对较小）
        header_font_size = max(20, int(main_font_size * 0.55))
        try:
            header_font = ImageFont.truetype("DejaVuSans.ttf", header_font_size)
        except IOError:
            try:
                header_font = ImageFont.truetype("DejaVuSansMono.ttf", header_font_size)
            except IOError:
                header_font = ImageFont.load_default()

        # 画网格线（高对比）
        line_thickness = max(3, cell_size // 18)
        for i in range(grid_size + 1):
            y = margin_y + i * cell_size
            draw.line([(margin_x, y), (margin_x + grid_size * cell_size, y)], fill='black', width=line_thickness)
            x = margin_x + i * cell_size
            draw.line([(x, margin_y), (x, margin_y + grid_size * cell_size)], fill='black', width=line_thickness)

        # 写字母（黑色、等宽、居中）
        for y in range(grid_size):
            for x in range(grid_size):
                ch = grid[y][x]
                cx = margin_x + x * cell_size + cell_size // 2
                cy = margin_y + y * cell_size + cell_size // 2
                try:
                    draw.text((cx, cy), ch, fill='black', font=font, anchor="mm")
                except TypeError:
                    # 兼容旧Pillow不支持anchor
                    try:
                        tw, th = font.getsize(ch)
                    except AttributeError:
                        tw, th = draw.textsize(ch, font=font)
                    draw.text((cx - tw // 2, cy - th // 2), ch, fill='black', font=font)

        # 顶部提示语：Find these words: ...，保证不超过网格的宽度与上方留白高度
        words_display = []
        if isinstance(target_words, (list, tuple)):
            for w in target_words:
                try:
                    words_display.append(str(w).upper())
                except Exception:
                    words_display.append(str(w))
        header_text = "Find these words: " + (", ".join(words_display) if words_display else "")

        # 可用绘制区域（限定在网格水平范围内与上部留白内）
        grid_width = grid_size * cell_size
        max_header_width = grid_width - 40
        max_header_height = max(0, margin_y - 20)

        def _measure_text(text, font_obj):
            try:
                bbox = draw.textbbox((0, 0), text, font=font_obj)
                return bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                try:
                    return font_obj.getsize(text)
                except Exception:
                    return draw.textsize(text, font=font_obj)

        def _wrap_text_to_width(text, font_obj, max_width):
            words = text.split()
            if not words:
                return []
            lines = []
            current = words[0]
            for word in words[1:]:
                candidate = current + " " + word
                w, _ = _measure_text(candidate, font_obj)
                if w <= max_width:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            lines.append(current)
            return lines

        # 自动调整字号，先尝试换行，再缩小字号；若依然超出高度，则截断并添加省略号
        min_font_size = 10
        line_spacing_px = 4
        current_size = header_font_size
        final_lines = []
        final_font = header_font

        while current_size >= min_font_size:
            try:
                trial_font = ImageFont.truetype("DejaVuSans.ttf", current_size)
            except IOError:
                try:
                    trial_font = ImageFont.truetype("DejaVuSansMono.ttf", current_size)
                except IOError:
                    trial_font = ImageFont.load_default()

            wrapped = _wrap_text_to_width(header_text, trial_font, max_header_width)
            if not wrapped:
                break

            # 计算高度
            line_heights = []
            for line in wrapped:
                _, lh = _measure_text(line, trial_font)
                line_heights.append(lh)
            total_height = sum(line_heights) + max(0, (len(wrapped) - 1) * line_spacing_px)

            if total_height <= max_header_height:
                final_lines = wrapped
                final_font = trial_font
                break
            else:
                current_size -= 1

        # 如果仍然放不下，则尽量截断内容以适配高度
        if not final_lines:
            # 使用最小字号进行截断
            try:
                trial_font = ImageFont.truetype("DejaVuSans.ttf", min_font_size)
            except IOError:
                try:
                    trial_font = ImageFont.truetype("DejaVuSansMono.ttf", min_font_size)
                except IOError:
                    trial_font = ImageFont.load_default()
            words = header_text.split()
            kept = []
            while words:
                candidate = " ".join(kept + [words[0]])
                wrapped = _wrap_text_to_width(candidate + " ...", trial_font, max_header_width)
                # 估计高度
                line_heights = []
                for line in wrapped:
                    _, lh = _measure_text(line, trial_font)
                    line_heights.append(lh)
                total_height = sum(line_heights) + max(0, (len(wrapped) - 1) * line_spacing_px)
                if total_height <= max_header_height:
                    kept.append(words.pop(0))
                else:
                    break
            if kept:
                final_lines = _wrap_text_to_width(" ".join(kept) + " ...", trial_font, max_header_width)
                final_font = trial_font
            else:
                # 实在无空间，跳过绘制提示语
                final_lines = []

        if final_lines:
            # 计算总高度，垂直居中放在上边距内；水平在网格范围内居中
            line_sizes = [_measure_text(line, final_font) for line in final_lines]
            total_height = sum(h for _, h in line_sizes) + max(0, (len(line_sizes) - 1) * line_spacing_px)
            header_top = max(10, (margin_y - total_height) // 2)
            y_cursor = header_top
            center_x = margin_x + grid_width // 2
            for i, line in enumerate(final_lines):
                lw, lh = line_sizes[i]
                x = center_x - lw // 2
                # 保证不越过网格左右边界
                x = max(margin_x + 20, min(x, margin_x + grid_width - lw - 20))
                try:
                    draw.text((x, y_cursor), line, fill='black', font=final_font)
                except TypeError:
                    draw.text((x, y_cursor), line, fill='black', font=final_font)
                y_cursor += lh + line_spacing_px

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        img.save(filename, quality=95, dpi=(300, 300))
        return filename
    

    
    def visualize(self, puzzle, **kwargs):
        # For backward compatibility, use the _generate_mock_image method
        # 支持从字符串与字典两种格式中提取网格与目标词
        if isinstance(puzzle, str):
            lines = puzzle.strip().split('\n')
            grid = []
            target_words = []
            
            for line in lines:
                if "Find these words:" in line:
                    words_part = line.split("Find these words:")[1].strip()
                    target_words = words_part.split()
                elif ' ' in line and all(c.isalpha() or c.isspace() for c in line):
                    row = line.split()
                    if len(row) > 0 and all(len(cell) == 1 for cell in row):
                        grid.append(row)
            
            if grid and target_words:
                filename = kwargs.get('filename', 'wordsearch_visualization.png')
                return self._generate_mock_image(grid, target_words, filename, len(grid))
        elif isinstance(puzzle, dict):
            # 优先从 initial_state 获取网格
            grid = puzzle.get('initial_state') or puzzle.get('grid')
            target_words = []
            # 从显式字段或 answer 文本中提取目标词
            if 'target_words' in puzzle and isinstance(puzzle['target_words'], (list, tuple)):
                target_words = list(puzzle['target_words'])
            elif isinstance(puzzle.get('answer'), str):
                ans = puzzle['answer']
                # 形如 "WORD DIRECTION (...) @" 取 @ 前的词
                word_part = ans.split('@')[0].strip()
                if word_part:
                    # 有些答案可能是多词，这里分割取第一个
                    target_words = [word_part.split()[0]]
            if grid and target_words:
                filename = kwargs.get('filename', 'wordsearch_visualization.png')
                return self._generate_mock_image(grid, target_words, filename, len(grid))
        
        # Fallback
        return None
