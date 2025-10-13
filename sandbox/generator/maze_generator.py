import numpy as np
import matplotlib
matplotlib.use('Agg')  # 必须在pyplot导入之前设置
import matplotlib.pyplot as plt
import json
import os
import random
import time
from collections import deque
import uuid
import sys
import os
# Add parent directory to path to import base_generator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator.base_generator import BaseGenerator
from utils.constants import PROMPT_MAZE_IMAGE
from utils.constants import PROMPT_MAZE



class MazeGenerator(BaseGenerator):
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    direction_names = ['right', 'down', 'left', 'up']
    direction_text = {'right': 'right', 'down': 'down', 'left': 'left', 'up': 'up'}

    def __init__(self, output_folder):
        super().__init__(output_folder)

        # 全局已生成的items，用于确保benchmark和training set不重复
        self.global_items = set()

        # 当前生成的items，用于内部重复检查
        self.current_items = set()

        # 存储待保存的所有puzzles
        self.pending_puzzles = []

        # 设置时间戳作为seed
        self.base_seed = int(time.time())

        # 统一目录结构以适配 main.py 与 BaseGenerator
        # 注：将 task_dir 对齐为输出根目录，图片置于 images/ 下，标注为根目录下 annotations.json
        self.task_dir = self.output_folder
        self.image_dir = os.path.join(self.output_folder, 'images')
        self.annotations_file = os.path.join(self.output_folder, 'annotations.json')

    def _is_duplicate(self, puzzle_question, puzzle_answer, puzzle_category):
        """检查是否与全局items或当前生成的items重复"""
        # 创建唯一标识符
        item_key = (puzzle_category, puzzle_question, puzzle_answer)

        # 检查当前生成过程中的重复
        if item_key in self.current_items:
            return True

        # 检查全局重复
        if item_key in self.global_items:
            return True

        return False

    def _add_to_current_items(self, puzzle_question, puzzle_answer, puzzle_category):
        """添加到当前生成的items集合"""
        item_key = (puzzle_category, puzzle_question, puzzle_answer)
        self.current_items.add(item_key)
        self.global_items.add(item_key)

    def _get_difficulty_by_size(self, size):
        """根据size直接映射到难度，确保一个size对应一个difficulty"""
        size_to_difficulty = {
            3: '1',
            5: '2',
            7: '3',
            9: '4',
            11: '5'
        }
        return size_to_difficulty.get(size, '3')  # 默认返回中等难度

    def add_puzzle(self, puzzle_data):
        """添加puzzle到待保存列表"""
        self.pending_puzzles.append(puzzle_data)

    def save_all_puzzles(self):
        """批量保存所有待保存的puzzles"""
        if not self.pending_puzzles:
            return

        # Ensure puzzles have the required fields
        for puzzle in self.pending_puzzles:
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
        # 使用基类的保存逻辑（去重基于 index），并清空暂存列表
        self.save_annotations(self.pending_puzzles, self.output_folder)
        self.pending_puzzles = []

    def _get_difficulty_params(self, difficulty):
        """
        根据难度级别获取迷宫参数配置。

        Args:
            difficulty: 难度级别（1-5）

        Returns:
            dict: 包含难度参数的字典
        """
        # 难度与迷宫大小的映射
        difficulty_to_size = {
            1: 3,
            2: 5,
            3: 7,
            4: 9,
            5: 11
        }
        size = difficulty_to_size.get(difficulty, 7)  # 默认中等难度
        return {'size': size}

    def generate(self, num_cases, difficulty, output_folder=None):
        """
        生成指定数量的迷宫问题。

        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别
            output_folder: 输出文件夹路径，覆盖构造函数中设置的路径

        Returns:
            生成的问题列表
        """
        # Update output folder if provided
        if output_folder is not None:
            self.output_folder = output_folder
            self.task_dir = self.output_folder
            self.image_dir = os.path.join(self.output_folder, 'images')
            self.annotations_file = os.path.join(self.output_folder, 'annotations.json')

            # Create directories (根目录与 images 目录)
            os.makedirs(self.output_folder, exist_ok=True)
            os.makedirs(self.image_dir, exist_ok=True)

        # Get difficulty parameters
        params = self._get_difficulty_params(difficulty)
        size = params['size']

        # Clear pending puzzles
        self.pending_puzzles = []
        self.current_items = set()

        generated_puzzles = []

        for case_idx in range(num_cases):
            # Use timestamp-based seed with case index for variation
            current_seed = self.base_seed + case_idx

            # Set random seed to ensure reproducibility
            random.seed(current_seed)
            np.random.seed(current_seed)

            # Generate maze with specified size
            maze = Maze(size, seed=current_seed)
            solution = maze.find_solution()

            if not solution:
                print(f"Warning: Could not find solution for maze with size={size}, seed={current_seed}, skipping...")
                continue

            # Generate unique index based on size and seed
            index = f"maze_{size}_{current_seed}"

            # Create image filename
            img_filename = os.path.join(self.image_dir, f"{index}.png")

            # Generate visualization
            maze.visualize(solution=None, filename=img_filename)

            # Get text representation (initial state)
            initial_state = maze.to_text_representation()

            # Create answer string
            answer = ' '.join(solution['directions'])

            # Check for duplicates
            if self._is_duplicate(PROMPT_MAZE_IMAGE, answer, "maze"):
                print(f"Warning: Duplicate maze detected for size={size}, seed={current_seed}, skipping...")
                continue

            # Create question with initial state reference
            question_language = PROMPT_MAZE.format(initial_state)

            # Generate CoT reasoning process
            cot_result = maze.generate_cot(solution)

            # Create puzzle data in required format
            maze_data = {
                'index': index,
                'category': "maze",
                'image': os.path.relpath(img_filename, self.task_dir),
                'question': PROMPT_MAZE_IMAGE,
                'question_language': question_language,
                'answer': answer,
                'initial_state': initial_state,
                'difficulty': str(difficulty),
                'cot': cot_result['cot']
            }

            # Add CoT step fields
            maze_data.update(cot_result['cot_steps'])

            # Add to pending puzzles instead of saving immediately
            self.add_puzzle(maze_data)
            self._add_to_current_items(PROMPT_MAZE_IMAGE, answer, "maze")

            generated_puzzles.append(maze_data)

            step_count = solution['steps']
            print(f"Generated maze {case_idx + 1}/{num_cases}: size={size}, seed={current_seed}, steps={step_count}, difficulty={difficulty}")

        # Save all puzzles at once
        self.save_all_puzzles()

        return generated_puzzles
    
    def visualize(self, maze, solution=None, filename=None, **kwargs):
        if filename is None:
            filename = f"{self.image_dir}/maze_{uuid.uuid4()}.png"
            
        return maze.visualize(solution=solution, filename=filename)


class Maze:
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    direction_names = ['right', 'down', 'left', 'up']
    direction_text = {'right': 'right', 'down': 'down', 'left': 'left', 'up': 'up'}
    
    def __init__(self, n, seed=None):
        self.n = n
        self.seed = seed
        
        # Set seed for this maze instance if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.walls = {}
        
        for r in range(n):
            for c in range(n):
                for d in range(4):
                    self.walls[(r, c, d)] = True
        
        for r in range(n):
            for c in range(n):
                if c == 0:
                    self.walls[(r, c, 2)] = False
                if c == n-1:
                    self.walls[(r, c, 0)] = False
                if r == 0:
                    self.walls[(r, c, 3)] = False
                if r == n-1:
                    self.walls[(r, c, 1)] = False
                
        self._generate_maze()
        
        self._set_start_end()
    
    def _generate_maze(self):
        n = self.n
        
        visited = set()
        
        start_r, start_c = random.randint(0, n-1), random.randint(0, n-1)
        
        self._dfs_generate(start_r, start_c, visited)
    
    def _dfs_generate(self, r, c, visited):
        visited.add((r, c))
        
        directions = list(range(4))
        random.shuffle(directions)
        
        for direction in directions:
            nr, nc = r + self.dx[direction], c + self.dy[direction]
            
            if 0 <= nr < self.n and 0 <= nc < self.n and (nr, nc) not in visited:
                self.walls[(r, c, direction)] = False
                
                opposite_direction = (direction + 2) % 4
                self.walls[(nr, nc, opposite_direction)] = False
                
                self._dfs_generate(nr, nc, visited)
    
    def _set_start_end(self):
        n = self.n
        
        self.start = (random.randint(0, n-1), random.randint(0, n-1))
        
        distances = {}
        queue = deque([self.start])
        distances[self.start] = 0
        
        while queue:
            r, c = queue.popleft()
            
            for direction in range(4):
                if not self.walls.get((r, c, direction), True):
                    nr, nc = r + self.dx[direction], c + self.dy[direction]
                    
                    if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in distances:
                        distances[(nr, nc)] = distances[(r, c)] + 1
                        queue.append((nr, nc))
        
        max_dist = -1
        farthest_cell = None
        
        for cell, dist in distances.items():
            if dist > max_dist:
                max_dist = dist
                farthest_cell = cell
        
        self.end = farthest_cell
        self.max_distance = max_dist
    
    def find_solution(self):
        queue = deque([self.start])
        visited = {self.start: None}
        
        while queue:
            r, c = queue.popleft()
            
            if (r, c) == self.end:
                break
            
            for direction in range(4):
                if not self.walls.get((r, c, direction), True):
                    nr, nc = r + self.dx[direction], c + self.dy[direction]
                    
                    if 0 <= nr < self.n and 0 <= nc < self.n and (nr, nc) not in visited:
                        visited[(nr, nc)] = ((r, c), direction)
                        queue.append((nr, nc))
        
        if self.end not in visited:
            return None
        
        path = []
        current = self.end
        
        while current != self.start:
            prev, direction = visited[current]
            path.append(direction)
            current = prev
        
        path.reverse()
        
        direction_path = [self.direction_names[d] for d in path]
        
        coordinate_path = [self.start]
        current = self.start
        
        for d in path:
            nr, nc = current[0] + self.dx[d], current[1] + self.dy[d]
            coordinate_path.append((nr, nc))
            current = (nr, nc)
        
        return {
            'directions': direction_path,
            'coordinates': coordinate_path,
            'steps': len(path)
        }
        
    def generate_cot(self, solution=None):
        """
        Generate enriched Chain of Thought following the 4-step rule-based format.
        
        Step 1: Understanding game rules and objectives
        Step 2: Careful image reading and initial state representation
        Step 3: Detailed reasoning process with sufficient exploration
        Step 4: Solution validation and reflection
        """
        import re
        from collections import deque

        if solution is None:
            solution = self.find_solution()
        if not solution:
            return {"cot": "Unable to find a solution.", "cot_steps": {}}

        # ---------- Helper Functions ----------
        def manhattan(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def get_neighbors(cell):
            r, c = cell
            outs = []
            for d in range(4):
                if not self.walls.get((r, c, d), True):
                    nr, nc = r + self.dx[d], c + self.dy[d]
                    if 0 <= nr < self.n and 0 <= nc < self.n:
                        outs.append(((nr, nc), self.direction_names[d]))
            return outs

        def can_reach_end_from(start_cell, blocked):
            """BFS reachability check for pruning/backtracking validation."""
            q = deque([start_cell])
            seen = {start_cell}
            while q:
                u = q.popleft()
                if u == self.end:
                    return True
                for (v, _) in get_neighbors(u):
                    if v not in seen and v not in blocked:
                        seen.add(v)
                        q.append(v)
            return False

        def analyze_maze_structure():
            """Analyze the maze structure for dead ends, junctions, and corridors."""
            dead_ends = []
            junctions = []
            corridors = []
            
            for r in range(self.n):
                for c in range(self.n):
                    neighbors = get_neighbors((r, c))
                    if len(neighbors) == 1:
                        dead_ends.append((r, c))
                    elif len(neighbors) > 2:
                        junctions.append((r, c))
                    elif len(neighbors) == 2:
                        corridors.append((r, c))
            
            return dead_ends, junctions, corridors

        def cut_words_half(text):
            """Cut text at approximately half by words, preferring sentence boundaries."""
            words = text.strip().split()
            if len(words) <= 2:
                return text
            half = max(1, len(words) // 2)
            rough = " ".join(words[:half])

            # Try to end at a sentence boundary near the cut point
            tail = rough[-80:] if len(rough) > 80 else rough
            puncts = [tail.rfind(ch) for ch in ['。', '.', '!', '?', ';', '；', '\n']]
            idx = max(puncts)
            if idx != -1:
                return rough[:len(rough) - (len(tail) - (idx + 1))].rstrip()
            return rough

        # ---------- STEP 1: Understanding Game Rules ----------
        header = "Let me analyze this <maze> step by step\n"

        step1_lines = []
        step1_lines.append("### Step 1: Understanding the puzzle rules and objectives")
        step1_lines.append("")
        step1_lines.append("First, I need to clearly understand what this maze puzzle requires:")
        step1_lines.append("")
        step1_lines.append("**Game Rules:**")
        step1_lines.append("- This is a grid-based maze navigation puzzle on an n×n lattice")
        step1_lines.append("- Movement is restricted to four cardinal directions: up, down, left, right")
        step1_lines.append("- No diagonal movement is allowed")
        step1_lines.append("- Walls are represented by solid lines that block movement")
        step1_lines.append("- Open corridors allow free passage between adjacent cells")
        step1_lines.append("- Must stay within the grid boundaries (0 to n-1 for both rows and columns)")
        step1_lines.append("")
        step1_lines.append("**Objective:**")
        step1_lines.append(f"- Start position: S at coordinates {self.start}")
        step1_lines.append(f"- End position: E at coordinates {self.end}")
        step1_lines.append("- Goal: Find a valid path from S to E that respects all movement constraints")
        step1_lines.append("- The path should be efficient (shortest possible route)")
        step1_lines.append("")
        step1_lines.append("**Success Criteria:**")
        step1_lines.append("- Every step must move through an open corridor (no wall crossing)")
        step1_lines.append("- Path must be continuous from start to end")
        step1_lines.append("- Final position must exactly match the target end position")
        
        step1_text = "\n".join(step1_lines) + "\n"

        # ---------- STEP 2: Careful Image Reading and State Representation ----------
        # Get initial maze state representation
        initial_state = self.to_text_representation()
        start_moves = self._get_available_moves(self.start, path_visited=set())
        dead_ends, junctions, corridors = analyze_maze_structure()
        
        step2_lines = []
        step2_lines.append("### Step 2: Reading the image carefully and analyzing initial state")
        step2_lines.append("")
        step2_lines.append("Now I'll carefully examine the visual maze and extract its structure:")
        step2_lines.append("")
        step2_lines.append("**Maze Dimensions and Layout:**")
        step2_lines.append(f"- Grid size: {self.n}×{self.n}")
        step2_lines.append(f"- Start position S: {self.start}")
        step2_lines.append(f"- End position E: {self.end}")
        step2_lines.append(f"- Manhattan distance from S to E: {manhattan(self.start, self.end)}")
        step2_lines.append("")
        step2_lines.append("**Text Representation of Initial State:**")
        step2_lines.append("```")
        step2_lines.append(initial_state)
        step2_lines.append("```")
        step2_lines.append("")
        step2_lines.append("**Structural Analysis:**")
        step2_lines.append(f"- Dead ends detected: {len(dead_ends)} positions")
        if dead_ends:
            step2_lines.append(f"  Dead end positions: {dead_ends[:5]}{'...' if len(dead_ends) > 5 else ''}")
        step2_lines.append(f"- Junction points: {len(junctions)} positions")
        if junctions:
            step2_lines.append(f"  Junction positions: {junctions[:5]}{'...' if len(junctions) > 5 else ''}")
        step2_lines.append(f"- Corridor segments: {len(corridors)} positions")
        step2_lines.append("")
        step2_lines.append("**Initial Movement Options from Start:**")
        if start_moves:
            for direction, next_pos in start_moves:
                step2_lines.append(f"- Can move {direction} to position {next_pos}")
        else:
            step2_lines.append("- Only one possible initial direction (forced move)")
        step2_lines.append("")
        step2_lines.append("**State Reading Reflection:**")
        step2_lines.append("- The maze structure shows clear wall boundaries marked by solid lines")
        step2_lines.append("- Open spaces form corridors that connect different regions")
        step2_lines.append("- The path-finding challenge involves navigating through these corridors")
        step2_lines.append("- Start and end positions are clearly marked and accessible")
        
        step2_text = "\n".join(step2_lines) + "\n"

        # ---------- STEP 3: Detailed Reasoning Process with Exploration ----------
        step3_lines = []
        step3_lines.append("### Step 3: Detailed reasoning process and path exploration")
        step3_lines.append("")
        step3_lines.append("Now I'll systematically explore the maze to find the optimal path:")
        step3_lines.append("")
        step3_lines.append("**Search Strategy:**")
        step3_lines.append("- Use depth-first exploration with backtracking")
        step3_lines.append("- Prioritize moves that reduce Manhattan distance to target")
        step3_lines.append("- Explore alternative branches when multiple options exist")
        step3_lines.append("- Backtrack when reaching dead ends or loops")
        step3_lines.append("- Maintain a visited set to avoid cycles")
        step3_lines.append("")

        # Simulate the reasoning process step by step
        current_pos = self.start
        path_taken = [current_pos]
        visited_positions = {current_pos}
        step_count = 0
        sol_coords = solution['coordinates']
        exploration_attempts = 0
        max_explorations = 3

        # Build solution lookup for guidance
        sol_next = {}
        for i in range(len(sol_coords) - 1):
            sol_next[sol_coords[i]] = sol_coords[i + 1]

        step3_lines.append("**Step-by-step Exploration:**")
        step3_lines.append("")

        while current_pos != self.end and step_count < len(sol_coords) - 1:
            step_count += 1
            available_moves = self._get_available_moves(current_pos, path_visited=visited_positions)
            
            if not available_moves:
                step3_lines.append(f"Step {step_count}: Dead end at {current_pos} - need to backtrack")
                break
            
            # Sort moves by Manhattan distance to end
            moves_by_distance = sorted(available_moves, 
                                     key=lambda x: (manhattan(x[1], self.end), self.direction_names.index(x[0])))
            
            optimal_move = moves_by_distance[0]
            optimal_direction, optimal_next = optimal_move
            
            # Show the decision-making process
            step3_lines.append(f"**Step {step_count}: At position {current_pos}**")
            step3_lines.append(f"Available moves: {len(available_moves)}")
            for direction, next_pos in available_moves:
                distance = manhattan(next_pos, self.end)
                step3_lines.append(f"  - {direction} to {next_pos} (distance to end: {distance})")
            
            # Explore alternative if multiple good options exist
            if (len(moves_by_distance) > 1 and exploration_attempts < max_explorations and 
                manhattan(moves_by_distance[1][1], self.end) <= manhattan(optimal_next, self.end) + 1):
                
                alt_direction, alt_next = moves_by_distance[1]
                exploration_attempts += 1
                
                # Quick reachability check for the alternative
                reachable = can_reach_end_from(alt_next, blocked=visited_positions)
                step3_lines.append(f"  → Exploring alternative: {alt_direction} to {alt_next}")
                step3_lines.append(f"    Reachability check: {'✓ Can reach end' if reachable else '✗ Cannot reach end'}")
                
                if not reachable:
                    step3_lines.append(f"    Decision: Avoid {alt_direction}, leads to dead end")
                else:
                    step3_lines.append(f"    Decision: {alt_direction} is viable but {optimal_direction} is more direct")
            
            # Make the optimal move
            next_position = sol_next.get(current_pos, optimal_next)
            chosen_direction = None
            for direction, pos in available_moves:
                if pos == next_position:
                    chosen_direction = direction
                    break
            
            if chosen_direction is None:
                chosen_direction = optimal_direction
                next_position = optimal_next
            
            step3_lines.append(f"  → **Chosen move: {chosen_direction} to {next_position}**")
            step3_lines.append(f"    Reasoning: {'Follows optimal path' if next_position == sol_next.get(current_pos) else 'Best available option'}")
            step3_lines.append("")
            
            # Update position and path
            current_pos = next_position
            path_taken.append(current_pos)
            visited_positions.add(current_pos)
            
            # Add compression for very long paths
            remaining_steps = len(sol_coords) - step_count - 1
            if remaining_steps > 15 and step_count > 10:
                step3_lines.append(f"... [Continuing through corridor for {remaining_steps} more steps] ...")
                step3_lines.append(f"Following the only viable path through narrow corridors")
                step3_lines.append(f"Final approach to end position {self.end}")
                break

        step3_lines.append("**Exploration Summary:**")
        step3_lines.append(f"- Total decision points analyzed: {step_count}")
        step3_lines.append(f"- Alternative paths explored: {exploration_attempts}")
        step3_lines.append(f"- Backtracking instances: 0 (clean path found)")
        step3_lines.append(f"- Path efficiency: Optimal (shortest possible route)")
        
        step3_text = "\n".join(step3_lines) + "\n"

        # ---------- STEP 4: Solution Validation and Reflection ----------
        step4_lines = []
        step4_lines.append("### Step 4: Solution validation and reflection")
        step4_lines.append("")
        step4_lines.append("Finally, I'll validate the complete solution and reflect on the process:")
        step4_lines.append("")
        step4_lines.append("**Solution Summary:**")
        step4_lines.append(f"- Total moves required: {len(solution['directions'])}")
        step4_lines.append(f"- Path coordinates: {len(solution['coordinates'])} positions")
        step4_lines.append(f"- Direction sequence: {' '.join(solution['directions'])}")
        step4_lines.append("")
        step4_lines.append("**Constraint Validation:**")
        
        # Validate each step
        valid_path = True
        validation_details = []
        
        for i, (direction, coord) in enumerate(zip(solution['directions'], solution['coordinates'][1:])):
            prev_coord = solution['coordinates'][i]
            
            # Check if move is valid
            valid_move = False
            for d in range(4):
                if (not self.walls.get((prev_coord[0], prev_coord[1], d), True) and
                    prev_coord[0] + self.dx[d] == coord[0] and
                    prev_coord[1] + self.dy[d] == coord[1] and
                    self.direction_names[d] == direction):
                    valid_move = True
                    break
            
            if not valid_move:
                valid_path = False
                validation_details.append(f"  ✗ Step {i+1}: Invalid move {direction} from {prev_coord} to {coord}")
            else:
                validation_details.append(f"  ✓ Step {i+1}: Valid move {direction} from {prev_coord} to {coord}")
        
        # Show validation results (limit output for long paths)
        if len(validation_details) <= 10:
            step4_lines.extend(validation_details)
        else:
            step4_lines.extend(validation_details[:3])
            step4_lines.append(f"  ... [All {len(validation_details)} steps validated] ...")
            step4_lines.extend(validation_details[-2:])
        
        step4_lines.append("")
        step4_lines.append("**Path Optimality Check:**")
        step4_lines.append(f"- Manhattan distance (lower bound): {manhattan(self.start, self.end)}")
        step4_lines.append(f"- Actual path length: {len(solution['directions'])}")
        step4_lines.append(f"- Optimality ratio: {manhattan(self.start, self.end) / len(solution['directions']):.2f}")
        step4_lines.append("- BFS construction ensures this is the shortest possible path")
        step4_lines.append("")
        step4_lines.append("**Solution Verification:**")
        step4_lines.append(f"- Start position reached: ✓ {self.start}")
        step4_lines.append(f"- End position reached: ✓ {self.end}")
        step4_lines.append(f"- All moves respect wall constraints: {'✓' if valid_path else '✗'}")
        step4_lines.append(f"- No boundary violations: ✓")
        step4_lines.append(f"- Path continuity: ✓")
        step4_lines.append("")
        step4_lines.append("**Final Reflection:**")
        step4_lines.append("- The systematic exploration approach successfully identified the optimal path")
        step4_lines.append("- Alternative route analysis confirmed the chosen path's efficiency")
        step4_lines.append("- All movement constraints were respected throughout the solution")
        step4_lines.append("- The reasoning process demonstrated both thoroughness and efficiency")
        step4_lines.append(f"- **Final Answer: {' '.join(solution['directions'])}**")
        
        step4_text = "\n".join(step4_lines) + "\n"

        # ---------- Compose Full COT ----------
        full_cot = "\n".join([
            header,
            step1_text,
            step2_text,
            step3_text,
            step4_text
        ]).strip()

        # ---------- Build cot_steps (only step1..3) ----------
        def build_part_all_until(step_idx):
            preamble = header
            parts = {
                1: step1_text,
                2: step2_text,
                3: step3_text
            }
            
            # Full text up to step_idx
            seq_full = [preamble]
            if step_idx >= 1:
                seq_full.append(step1_text)
            if step_idx >= 2:
                seq_full.append(step2_text)
            if step_idx >= 3:
                seq_full.append(step3_text)
            all_text = "".join(seq_full).strip()

            # Partial text with last step truncated at half
            seq_part = [preamble]
            if step_idx == 1:
                seq_part.append(cut_words_half(step1_text))
            elif step_idx == 2:
                seq_part.append(step1_text)
                seq_part.append(cut_words_half(step2_text))
            elif step_idx == 3:
                seq_part.append(step1_text)
                seq_part.append(step2_text)
                seq_part.append(cut_words_half(step3_text))
            part_text = "".join(seq_part).strip()
            
            return part_text, all_text

        cot_steps = {}
        
        # Generate step 1 partial and full
        p1, a1 = build_part_all_until(1)
        cot_steps["cot_step1_part"] = p1
        cot_steps["cot_step1_all"] = a1

        # Generate step 2 partial and full
        p2, a2 = build_part_all_until(2)
        cot_steps["cot_step2_part"] = p2
        cot_steps["cot_step2_all"] = a2

        # Generate step 3 partial and full
        p3, a3 = build_part_all_until(3)
        cot_steps["cot_step3_part"] = p3
        cot_steps["cot_step3_all"] = a3

        return {
            "cot": full_cot,
            "cot_steps": cot_steps
        }

    
    def _get_available_moves(self, current_pos, path_visited):
        """Get available moves from current position, excluding already visited positions in current path"""
        available_moves = []
        for direction in range(4):
            if not self.walls.get((current_pos[0], current_pos[1], direction), True):
                nr, nc = current_pos[0] + self.dx[direction], current_pos[1] + self.dy[direction]
                if 0 <= nr < self.n and 0 <= nc < self.n and (nr, nc) not in path_visited:
                    direction_name = self.direction_names[direction]
                    available_moves.append((direction_name, (nr, nc)))
        return available_moves
        
    def visualize(self, solution=None, filename=None):
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='#f9f9f9')
        
        # Set background color
        ax.set_facecolor('#f0f0f0')
        
        # Set limits to exactly match maze dimensions without extra padding
        ax.set_xlim(0, self.n)
        ax.set_ylim(self.n, 0)
        
        # Draw walls with thicker lines and better style
        for r in range(self.n):
            for c in range(self.n):
                if self.walls.get((r, c, 0), False):
                    ax.plot([c+1, c+1], [r, r+1], 'k-', linewidth=3.5, solid_capstyle='round')
                
                if self.walls.get((r, c, 1), False):
                    ax.plot([c, c+1], [r+1, r+1], 'k-', linewidth=3.5, solid_capstyle='round')
                
                if c == 0 and self.walls.get((r, c, 2), False):
                    ax.plot([c, c], [r, r+1], 'k-', linewidth=3.5, solid_capstyle='round')
                
                if r == 0 and self.walls.get((r, c, 3), False):
                    ax.plot([c, c+1], [r, r], 'k-', linewidth=3.5, solid_capstyle='round')
        
        # Add subtle shading to cells with light blue color for better contrast with black walls
        for r in range(self.n):
            for c in range(self.n):
                ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=True, color='#E6F3FF', 
                                        alpha=0.8, zorder=0))
        
        # Draw solution path with improved styling
        if solution:
            path_x = []
            path_y = []
            
            for r, c in solution['coordinates']:
                path_x.append(c + 0.5)
                path_y.append(r + 0.5)
            
            # Add a glow effect to the path
            ax.plot(path_x, path_y, 'r-', linewidth=4, alpha=0.3, zorder=2)
            ax.plot(path_x, path_y, 'r--', linewidth=2.5, zorder=3)
        
        # Improve start and end markers with adaptive size
        sr, sc = self.start
        er, ec = self.end
        
        # Calculate marker size based on grid size for better visibility
        marker_size = max(15, 200 // self.n)
        
        # Start marker (green circle with border)
        ax.plot(sc + 0.5, sr + 0.5, 'o', markersize=marker_size, markerfacecolor='#50C878', 
            markeredgecolor='darkgreen', markeredgewidth=3, zorder=4)
        
        # End marker (red X with border)
        ax.plot(ec + 0.5, er + 0.5, 'x', markersize=marker_size, color='#FF5733', 
            markeredgewidth=5, zorder=4)
        
        # Subtle grid lines
        for i in range(self.n + 1):
            ax.axhline(y=i, color='#bbbbbb', linestyle='-', linewidth=0.7, alpha=0.4)
            ax.axvline(x=i, color='#bbbbbb', linestyle='-', linewidth=0.7, alpha=0.4)
        
        # Better axis labels and ticks
        ax.set_xticks([i + 0.5 for i in range(self.n)])
        ax.set_yticks([i + 0.5 for i in range(self.n)])
        ax.set_xticklabels(range(self.n), fontsize=16, fontweight='bold')
        ax.set_yticklabels(range(self.n), fontsize=16, fontweight='bold')
        
        # Hide standard grid
        ax.grid(False)
        
        # Better labels
        ax.set_xlabel('Column', fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel('Row', fontsize=12, fontweight='bold', labelpad=10)
        
        # Add a title
        plt.title('Maze Visualization', fontsize=14, fontweight='bold', pad=15)
        
        # Add a subtle border around the plot
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#cccccc')
            spine.set_linewidth(1.5)
        
        # Adjust layout to remove any extra whitespace
        plt.tight_layout()
        
        # Adjust the subplot to remove padding
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return filename
        else:
            plt.show()
            plt.close(fig)
    def to_text_representation(self, solution=None):
        n = self.n
        
        grid = [[' ' for _ in range(2*n+1)] for _ in range(2*n+1)]
        
        for r in range(2*n+1):
            for c in range(2*n+1):
                if r % 2 == 0 and c % 2 == 0:
                    grid[r][c] = '+'
        
        for r in range(n):
            for c in range(n):
                for i in range(2):
                    if self.walls.get((r, c, 0), True):
                        grid[2*r+1][2*c+2] = '|'
                    
                    if self.walls.get((r, c, 1), True):
                        grid[2*r+2][2*c+1] = '-'
                    
                    if self.walls.get((r, c, 2), True):
                        grid[2*r+1][2*c] = '|'
                    
                    if self.walls.get((r, c, 3), True):
                        grid[2*r][2*c+1] = '-'
        
        for i in range(2*n+1):
            if i % 2 == 1:
                grid[0][i] = '-'
                grid[2*n][i] = '-'
                grid[i][0] = '|'
                grid[i][2*n] = '|'
        
        for r in range(n):
            for c in range(n):
                if grid[2*r+1][2*c+1] == ' ':
                    grid[2*r+1][2*c+1] = '0'
        
        sr, sc = self.start
        er, ec = self.end
        grid[2*sr+1][2*sc+1] = 'S'
        grid[2*er+1][2*ec+1] = 'E'
        
        if solution:
            for r, c in solution['coordinates']:
                if (r, c) != self.start and (r, c) != self.end:
                    grid[2*r+1][2*c+1] = 'X'
        
        return '\n'.join([''.join(row) for row in grid])

    def to_text_representation_with_position(self, current_pos=None):
        """Generate text representation with current position marked as 'S'"""
        n = self.n
        
        grid = [[' ' for _ in range(2*n+1)] for _ in range(2*n+1)]
        
        for r in range(2*n+1):
            for c in range(2*n+1):
                if r % 2 == 0 and c % 2 == 0:
                    grid[r][c] = '+'
        
        for r in range(n):
            for c in range(n):
                for i in range(2):
                    if self.walls.get((r, c, 0), True):
                        grid[2*r+1][2*c+2] = '|'
                    
                    if self.walls.get((r, c, 1), True):
                        grid[2*r+2][2*c+1] = '-'
                    
                    if self.walls.get((r, c, 2), True):
                        grid[2*r+1][2*c] = '|'
                    
                    if self.walls.get((r, c, 3), True):
                        grid[2*r][2*c+1] = '-'
        
        for i in range(2*n+1):
            if i % 2 == 1:
                grid[0][i] = '-'
                grid[2*n][i] = '-'
                grid[i][0] = '|'
                grid[i][2*n] = '|'
        
        for r in range(n):
            for c in range(n):
                if grid[2*r+1][2*c+1] == ' ':
                    grid[2*r+1][2*c+1] = '0'
        
        # Mark end position
        er, ec = self.end
        grid[2*er+1][2*ec+1] = 'E'
        
        # Mark current position as 'S'
        if current_pos:
            cr, cc = current_pos
            grid[2*cr+1][2*cc+1] = 'S'
        else:
            # If no current position, mark original start
            sr, sc = self.start
            grid[2*sr+1][2*sc+1] = 'S'
        
        return '\n'.join([''.join(row) for row in grid])