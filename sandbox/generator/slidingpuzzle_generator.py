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
from utils.constants import PROMPT_MAZE_IMAGE, PROMPT_15PUZZLE_IMAGE, PROMPT_HANOI_IMAGE, PROMPT_WORDSEARCH_IMAGE, PROMPT_NUMBRIX_IMAGE, PROMPT_MINESWEEPER_IMAGE, PROMPT_EULERO_IMAGE, PROMPT_SNAKE_IMAGE
from utils.constants import PROMPT_MAZE, PROMPT_15PUZZLE, PROMPT_HANOI, PROMPT_WORDSEARCH, PROMPT_NUMBRIX, PROMPT_MINESWEEPER, PROMPT_EULERO, PROMPT_SNAKE
from generator.base_generator import BaseGenerator





class SlidingPuzzleGenerator(BaseGenerator):
    def __init__(self, output_folder, size=4, training_set_path=None, **kwargs):
        # 初始化BaseGenerator
        super().__init__(output_folder)

        # 保持原有的内部逻辑
        self.size = size
        self.goal_state = np.array(list(range(1, size*size)) + [0]).reshape(size, size)

        # 加载训练集用于重复检查
        self.training_set = self._load_training_set(training_set_path) if training_set_path else []
        print(f"Loaded {len(self.training_set)} items from training set for duplicate checking")

        # 全局已生成的items，用于确保benchmark和training set不重复
        self.global_items = set()

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

    def is_solvable(self, state):
        flattened = state.flatten()
        
        inversions = 0
        for i in range(len(flattened) - 1):
            if flattened[i] == 0:
                continue
            for j in range(i+1, len(flattened)):
                if flattened[j] != 0 and flattened[i] > flattened[j]:
                    inversions += 1
        
        empty_row = 0
        for i, val in enumerate(flattened):
            if val == 0:
                empty_row = i // self.size
                break
        
        if self.size % 2 == 0:
            blank_row_from_bottom = self.size - 1 - empty_row
            return (blank_row_from_bottom % 2 == 0 and inversions % 2 == 1) or \
                   (blank_row_from_bottom % 2 == 1 and inversions % 2 == 0)
        else:
            return inversions % 2 == 0
    
    def generate_random_moves(self, num_moves=20, rng: random.Random = None):
        state = self.goal_state.copy()
        r, c = self.size - 1, self.size - 1

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        local_rng = rng if rng is not None else random

        for _ in range(num_moves):
            moves = []
            for dr, dc in directions:
                new_r, new_c = r + dr, c + dc
                if 0 <= new_r < self.size and 0 <= new_c < self.size:
                    moves.append((new_r, new_c, dr, dc))

            if not moves:
                continue

            new_r, new_c, dr, dc = local_rng.choice(moves)

            state[r, c], state[new_r, new_c] = state[new_r, new_c], state[r, c]

            r, c = new_r, new_c

        return state
    
    def generate_solvable_state(self, max_attempts=100, use_random_moves=False, difficulty="medium"):
        if use_random_moves:
            if difficulty == "easy":
                moves = random.randint(5, 15)
            elif difficulty == "medium":
                moves = random.randint(16, 30)
            elif difficulty == "hard":
                moves = random.randint(31, 50)
            else:
                moves = random.randint(10, 40)
                
            return self.generate_random_moves(moves)
        else:
            for _ in range(max_attempts):
                numbers = list(range(0, self.size*self.size))
                random.shuffle(numbers)
                state = np.array(numbers).reshape(self.size, self.size)
                
                if self.is_solvable(state):
                    return state
            
            print("Failed to generate random solvable state, using random moves method instead")
            return self.generate_random_moves(20)
    
    def find_empty(self, state):
        for r in range(self.size):
            for c in range(self.size):
                if state[r, c] == 0:
                    return r, c
        raise ValueError("Cannot find empty space in the state")
    
    def get_neighbors(self, state):
        neighbors = []
        moves = []
        moved_tiles = []
        
        r, c = self.find_empty(state)
        
        directions = [(-1, 0, 'down'), (1, 0, 'up'), (0, -1, 'right'), (0, 1, 'left')]
        
        for dr, dc, move_name in directions:
            new_r, new_c = r + dr, c + dc
            
            if 0 <= new_r < self.size and 0 <= new_c < self.size:
                new_state = state.copy()
                moved_tile = new_state[new_r, new_c]
                new_state[r, c], new_state[new_r, new_c] = new_state[new_r, new_c], new_state[r, c]
                neighbors.append(new_state)
                moves.append(move_name)
                moved_tiles.append(moved_tile)
                
        return neighbors, moves, moved_tiles
    
    def manhattan_distance(self, state):
        distance = 0
        for r in range(self.size):
            for c in range(self.size):
                value = state[r, c]
                if value != 0:
                    goal_r, goal_c = (value - 1) // self.size, (value - 1) % self.size
                    distance += abs(r - goal_r) + abs(c - goal_c)
        return distance
    
    def linear_conflict(self, state):
        conflicts = 0
        
        for r in range(self.size):
            for c1 in range(self.size):
                val1 = state[r, c1]
                if val1 == 0 or (val1 - 1) // self.size != r:
                    continue
                
                for c2 in range(c1 + 1, self.size):
                    val2 = state[r, c2]
                    if val2 == 0 or (val2 - 1) // self.size != r:
                        continue
                    
                    if val1 > val2:
                        conflicts += 1
        
        for c in range(self.size):
            for r1 in range(self.size):
                val1 = state[r1, c]
                if val1 == 0 or (val1 - 1) % self.size != c:
                    continue
                
                for r2 in range(r1 + 1, self.size):
                    val2 = state[r2, c]
                    if val2 == 0 or (val2 - 1) % self.size != c:
                        continue
                    
                    if val1 > val2:
                        conflicts += 1
        
        return conflicts * 2
    
    def solve(self, initial_state, time_limit=15):
        start_time = time.time()
        
        initial_tuple = tuple(map(tuple, initial_state))
        goal_tuple = tuple(map(tuple, self.goal_state))
        
        if initial_tuple == goal_tuple:
            return [initial_state], [], []
        
        open_set = []
        closed_set = set()
        came_from = {}
        move_to = {}
        moved_tile = {}
        g_score = {initial_tuple: 0}
        
        h_score = self.manhattan_distance(initial_state) + self.linear_conflict(initial_state)
        f_score = g_score[initial_tuple] + h_score
        
        import heapq
        heapq.heappush(open_set, (f_score, 0, initial_tuple))
        
        counter = 1
        nodes_explored = 0
        
        while open_set:
            nodes_explored += 1
            
            # Check time limit every 1000 iterations
            if nodes_explored % 1000 == 0 and time.time() - start_time >= time_limit:
                print(f"Time limit ({time_limit}s) reached, explored {nodes_explored} nodes")
                return None, None, None
            
            _, _, current_tuple = heapq.heappop(open_set)
            
            if current_tuple in closed_set:
                continue
                
            closed_set.add(current_tuple)
            
            if current_tuple == goal_tuple:
                path = []
                moves = []
                tiles = []
                
                while current_tuple in came_from:
                    current_state = np.array(current_tuple)
                    path.append(current_state)
                    moves.append(move_to[current_tuple])
                    tiles.append(moved_tile.get(current_tuple, None))
                    current_tuple = came_from[current_tuple]
                
                path.append(initial_state)
                
                path = list(reversed(path))
                moves = list(reversed(moves))
                tiles = list(reversed(tiles))
                
                print(f"Solution found, explored {nodes_explored} nodes, steps: {len(moves)}")
                return path, moves, tiles
            
            current_state = np.array(current_tuple)
            
            neighbors, neighbor_moves, neighbor_tiles = self.get_neighbors(current_state)
            
            for neighbor, move, tile in zip(neighbors, neighbor_moves, neighbor_tiles):
                neighbor_tuple = tuple(map(tuple, neighbor))
                
                if neighbor_tuple in closed_set:
                    continue
                
                tentative_g = g_score[current_tuple] + 1
                
                if neighbor_tuple not in g_score or tentative_g < g_score[neighbor_tuple]:
                    came_from[neighbor_tuple] = current_tuple
                    move_to[neighbor_tuple] = move
                    moved_tile[neighbor_tuple] = tile
                    g_score[neighbor_tuple] = tentative_g
                    
                    h_score = self.manhattan_distance(neighbor) + self.linear_conflict(neighbor)
                    f_score = tentative_g + h_score
                    
                    heapq.heappush(open_set, (f_score, counter, neighbor_tuple))
                    counter += 1
        
        print("No solution found")
        return None, None, None
    

    

    def create_image(self, state, filename=None):
        if filename is None:
            state_str = '_'.join(map(str, state.flatten()))
            filename = f"{self.image_dir}/puzzle_{hash(state_str) & 0xffffffff}.png"
        
        tile_size = 100
        margin = 5
        img_size = self.size * tile_size + (self.size + 1) * margin
        img = Image.new('RGB', (img_size, img_size), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("Arial.ttf", 36)
        except IOError:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 36)
            except IOError:
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
                except IOError:
                    font = ImageFont.load_default()
        
        for r in range(self.size):
            for c in range(self.size):
                value = state[r, c]
                x = c * (tile_size + margin) + margin
                y = r * (tile_size + margin) + margin
                
                if value != 0:
                    draw.rectangle([x, y, x + tile_size, y + tile_size], fill='lightblue', outline='black')
                    text_x = x + tile_size // 2
                    text_y = y + tile_size // 2
                    text = str(value)
                    text_w, text_h = 20, 20
                    
                    try:
                        try:
                            text_bbox = draw.textbbox((0, 0), text, font=font)
                            text_w = text_bbox[2] - text_bbox[0]
                            text_h = text_bbox[3] - text_bbox[1]
                        except AttributeError:
                            text_w, text_h = draw.textsize(text, font=font)
                    except:
                        pass
                    
                    try:
                        draw.text((text_x - text_w // 2, text_y - text_h // 2), text, fill='black', font=font)
                    except:
                        draw.text((text_x - 10, text_y - 10), text, fill='black')
                else:
                    draw.rectangle([x, y, x + tile_size, y + tile_size], fill='white', outline='lightgray')
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        img.save(filename)
        
        return filename
    
    def visualize(self, state, filename=None, **kwargs):
        return self.create_image(state, filename)
    
    def generate_state_from_seed(self, seed, size=4):
        """根据seed生成确定的游戏状态"""
        # 使用seed设置随机状态
        rng = np.random.RandomState(seed)
        
        # 使用随机移动方法确保生成可解状态
        state = self.goal_state.copy()
        r, c = self.size - 1, self.size - 1  # 空格初始位置
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # 根据seed确定移动次数（15-45步）
        num_moves = 15 + (seed % 31)
        
        for i in range(num_moves):
            moves = []
            for dr, dc in directions:
                new_r, new_c = r + dr, c + dc
                if 0 <= new_r < self.size and 0 <= new_c < self.size:
                    moves.append((new_r, new_c, dr, dc))
            
            if not moves:
                continue
                
            # 使用seed + i确保每次移动都是确定的
            move_idx = (seed + i) % len(moves)
            new_r, new_c, dr, dc = moves[move_idx]
            
            state[r, c], state[new_r, new_c] = state[new_r, new_c], state[r, c]
            r, c = new_r, new_c
        
        return state

    def generate_cot(self, initial_state, solution_moves):
        """生成基于规则的推理过程（Chain of Thought）
        按照四步模板生成，并提供 step1..3 的 part/all 版本。
        """

        def heuristic(state: np.ndarray) -> int:
            return self.manhattan_distance(state) + self.linear_conflict(state)

        def smart_truncate_words(text: str) -> str:
            """按词汇对半截断，尽量在合适的边界（句号/换行）处截断。"""
            words = text.split()
            if len(words) <= 2:
                return text
            half = max(1, len(words) // 2)
            # 优先向前找到句末或换行边界
            def is_sentence_end(w: str) -> bool:
                return any(w.endswith(p) for p in ['.', '!', '?', '。', '！', '？'])

            # 找到半数附近更合适的切点
            cut_idx = half
            # 向后找最多 20 个词
            for j in range(half, min(len(words), half + 20)):
                if is_sentence_end(words[j]) or words[j] == '-' or '\n' in words[j]:
                    cut_idx = j + 1
                    break
            else:
                # 向前找最多 20 个词
                for j in range(half - 1, max(0, half - 20), -1):
                    if is_sentence_end(words[j]) or words[j] == '-' or '\n' in words[j]:
                        cut_idx = j + 1
                        break
            return " ".join(words[:cut_idx]).strip()

        # 标题
        title = "Let me analyze this 15-puzzle step by step"

        # Step 1: 明确游戏规则，确保理解游戏规则
        step1_header = "\n\n### Step 1: Understanding the puzzle rules and objectives\n\n"
        step1_body = (
            "This is a 15-puzzle, also known as a sliding puzzle. Let me understand the game rules clearly:\n\n"
            "**Basic Rules:**\n"
            "- The puzzle consists of a 4×4 grid with 15 numbered tiles (1-15) and one empty space.\n"
            "- Only tiles that are adjacent to the empty space can slide into it.\n"
            "- Valid moves are: up, down, left, right (tiles moving into the empty space).\n"
            "- No diagonal moves are allowed.\n"
            "- Tiles cannot jump over other tiles.\n\n"
            "**Objective:**\n"
            "- Arrange all numbered tiles in ascending order from left to right, top to bottom.\n"
            "- The final goal state should be:\n"
            "  [1, 2, 3, 4]\n"
            "  [5, 6, 7, 8]\n"
            "  [9, 10, 11, 12]\n"
            "  [13, 14, 15, 0] (where 0 represents the empty space)\n\n"
            "**Strategy Principles:**\n"
            "- Work systematically, typically row by row or column by column.\n"
            "- Protect already solved sections while working on new areas.\n"
            "- Sometimes temporary moves that seem to worsen the position are necessary for progress.\n"
            "- Use heuristics like Manhattan distance to evaluate position quality."
        )

        # Step 2: read the image carefully，然后精确读取游戏初始状态
        empty_pos = self.find_empty(initial_state)
        grid_text = initial_state.tolist()
        target_text = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]]
        h0 = heuristic(initial_state)
        
        step2_header = "\n\n### Step 2: Carefully reading and analyzing the initial state\n\n"
        
        # 生成详细的状态描述
        state_description = "Let me carefully examine the current puzzle configuration:\n\n"
        state_description += "**Visual Grid Analysis:**\n"
        for i, row in enumerate(initial_state):
            row_str = " | ".join([f"{int(val):2d}" if val != 0 else "  " for val in row])
            state_description += f"Row {i+1}: | {row_str} |\n"
        
        state_description += f"\n**Numerical Representation:**\n"
        state_description += f"Current state: {grid_text}\n"
        state_description += f"Target state:  {target_text}\n"
        
        state_description += f"\n**Key Observations:**\n"
        state_description += f"- Empty space is located at position: row {empty_pos[0]+1}, column {empty_pos[1]+1} (1-indexed)\n"
        state_description += f"- Empty space coordinates: ({empty_pos[0]}, {empty_pos[1]}) (0-indexed)\n"
        
        # 分析哪些数字在正确位置
        correct_positions = []
        misplaced_tiles = []
        for r in range(self.size):
            for c in range(self.size):
                current_val = initial_state[r, c]
                if current_val == 0:
                    expected_val = 0
                    expected_pos = (self.size-1, self.size-1)
                else:
                    expected_val = r * self.size + c + 1
                    expected_pos = ((current_val-1) // self.size, (current_val-1) % self.size)
                
                if current_val == expected_val:
                    if current_val != 0:
                        correct_positions.append(f"tile {current_val}")
                else:
                    if current_val != 0:
                        misplaced_tiles.append(f"tile {current_val} (currently at ({r},{c}), should be at {expected_pos})")
        
        if correct_positions:
            state_description += f"- Correctly positioned tiles: {', '.join(correct_positions)}\n"
        else:
            state_description += f"- No tiles are in their correct positions\n"
            
        state_description += f"- Misplaced tiles: {', '.join(misplaced_tiles[:5])}" # 限制显示数量
        if len(misplaced_tiles) > 5:
            state_description += f" (and {len(misplaced_tiles)-5} more)"
        state_description += "\n"
        
        # 可移动的tiles
        neighbors, moves, tiles = self.get_neighbors(initial_state)
        movable_tiles = [(move, tile) for move, tile in zip(moves, tiles)]
        state_description += f"- Currently movable tiles: {', '.join([f'{tile} ({move})' for move, tile in movable_tiles])}\n"
        
        state_description += f"\n**State Quality Assessment:**\n"
        manhattan_dist = self.manhattan_distance(initial_state)
        linear_conflicts = self.linear_conflict(initial_state)
        state_description += f"- Manhattan distance: {manhattan_dist}\n"
        state_description += f"- Linear conflicts: {linear_conflicts}\n"
        state_description += f"- Combined heuristic value: {h0}\n"
        
        state_description += f"\n**Reflection on State Reading:**\n"
        state_description += f"I have carefully examined the puzzle grid and identified all tile positions. "
        state_description += f"The initial assessment shows this puzzle requires strategic planning to solve efficiently. "
        state_description += f"The heuristic value of {h0} gives us a baseline estimate of the difficulty."
        
        step2_body = state_description

        # Step 3: 详细的推理过程，足够充分的探索并输出最终的答案
        step3_header = "\n\n### Step 3: Strategic exploration and detailed reasoning process\n\n"
        
        reasoning_text = "Now I'll work through the solution systematically, exploring different possibilities and explaining my decision-making process:\n\n"
        reasoning_text += "**Solution Strategy:**\n"
        reasoning_text += "I will solve this puzzle using a combination of strategic planning and heuristic evaluation. "
        reasoning_text += "My approach will be to:\n"
        reasoning_text += "1. Evaluate all possible moves at each step\n"
        reasoning_text += "2. Consider both immediate and long-term consequences\n"
        reasoning_text += "3. Use Manhattan distance and linear conflict heuristics to guide decisions\n"
        reasoning_text += "4. Allow for tactical sacrifices when they enable strategic progress\n\n"
        
        reasoning_text += "**Step-by-step Solution Process:**\n\n"
        
        current_state = initial_state.copy()
        current_h = heuristic(current_state)
        
        # 详细分析前几步和关键步骤
        max_detailed_steps = min(8, len(solution_moves))
        key_decision_points = []
        
        for i in range(max_detailed_steps):
            planned_move = solution_moves[i]
            neighbors, moves, move_tiles = self.get_neighbors(current_state)
            
            # 分析所有候选移动
            candidates = []
            for m, s, t in zip(moves, neighbors, move_tiles):
                h_val = heuristic(s)
                candidates.append((m, h_val, t, s))
            
            # 排序候选项
            candidates.sort(key=lambda x: x[1])
            
            reasoning_text += f"**Move {i+1}:**\n"
            reasoning_text += f"Current position analysis:\n"
            
            # 显示当前状态的简化版本
            current_empty = self.find_empty(current_state)
            reasoning_text += f"- Empty space at: ({current_empty[0]}, {current_empty[1]})\n"
            reasoning_text += f"- Current heuristic: {current_h}\n"
            
            reasoning_text += f"- Available moves: {len(candidates)}\n"
            for j, (move, h_val, tile, _) in enumerate(candidates):
                delta = h_val - current_h
                delta_str = f"(Δh: {delta:+d})" if delta != 0 else "(Δh: 0)"
                reasoning_text += f"  • {move}: move tile {tile} {delta_str}\n"
            
            # 解释选择的原因
            chosen_move = planned_move
            chosen_data = next((c for c in candidates if c[0] == chosen_move), None)
            
            if chosen_data:
                chosen_move, chosen_h, chosen_tile, chosen_state = chosen_data
                delta = chosen_h - current_h
                
                reasoning_text += f"\n**Decision: Choose '{chosen_move}' (move tile {chosen_tile})**\n"
                
                if delta < 0:
                    reasoning_text += f"This move improves the heuristic by {abs(delta)}, bringing us closer to the solution.\n"
                elif delta > 0:
                    reasoning_text += f"Although this increases the heuristic by {delta}, it's strategically necessary because:\n"
                    # 分析为什么这步是必要的
                    if i < len(solution_moves) - 3:  # 不是最后几步
                        reasoning_text += f"- It positions tiles for more efficient future moves\n"
                        reasoning_text += f"- It avoids potential deadlock situations\n"
                        reasoning_text += f"- It follows the optimal solution path found by the solver\n"
                else:
                    reasoning_text += f"This is a neutral move that maintains position quality while enabling progress.\n"
                
                # 添加一些战术考虑
                if i == 2:  # 在第3步展示探索其他选项
                    alternatives = [c for c in candidates if c[0] != chosen_move]
                    if alternatives:
                        alt_move, alt_h, alt_tile, _ = alternatives[0]
                        reasoning_text += f"\nAlternative consideration: I could try '{alt_move}' (move tile {alt_tile}, h={alt_h}), "
                        if alt_h > chosen_h:
                            reasoning_text += f"but this would be worse (h={alt_h} vs {chosen_h}).\n"
                        else:
                            reasoning_text += f"but the optimal path suggests '{chosen_move}' leads to faster solution.\n"
                
                current_state = chosen_state.copy()
                current_h = chosen_h
                
            reasoning_text += "\n"
        
        # 处理剩余步骤的总结
        remaining_moves = len(solution_moves) - max_detailed_steps
        if remaining_moves > 0:
            reasoning_text += f"**Remaining {remaining_moves} moves:**\n"
            reasoning_text += f"Following the same strategic principles, I continue with the remaining moves: "
            remaining_moves_list = solution_moves[max_detailed_steps:]
            reasoning_text += " → ".join(remaining_moves_list)
            reasoning_text += "\n\nEach of these moves follows the same decision-making process:\n"
            reasoning_text += "- Evaluate all legal moves\n"
            reasoning_text += "- Consider heuristic changes\n"
            reasoning_text += "- Choose moves that lead toward the goal state\n"
            reasoning_text += "- Accept temporary heuristic increases when they serve long-term strategy\n\n"
        
        # 最终答案
        solution_sequence = " ".join(solution_moves)
        reasoning_text += f"**Final Solution Sequence:**\n"
        reasoning_text += f"The complete solution is: {solution_sequence}\n"
        reasoning_text += f"Total moves required: {len(solution_moves)}\n\n"
        
        reasoning_text += f"**Solution Quality Analysis:**\n"
        reasoning_text += f"This solution efficiently guides all tiles to their target positions while minimizing the total number of moves. "
        reasoning_text += f"Each move serves a purpose in the overall strategy, whether for immediate improvement or strategic positioning."
        
        step3_body = reasoning_text

        # Step 4: 基于最终的答案进行验证和反思
        step4_header = "\n\n### Step 4: Solution validation and reflection\n\n"
        
        # 验证解的正确性
        replay_state = initial_state.copy()
        validation_steps = []
        valid_solution = True
        
        validation_text = "Let me thoroughly validate the proposed solution:\n\n"
        validation_text += "**Step-by-step Verification:**\n"
        
        for i, move in enumerate(solution_moves):
            neighbors, moves, tiles = self.get_neighbors(replay_state)
            if move in moves:
                move_idx = moves.index(move)
                moved_tile = tiles[move_idx]
                old_pos = self.find_empty(replay_state)
                replay_state = neighbors[move_idx]
                new_pos = self.find_empty(replay_state)
                validation_steps.append(f"Move {i+1}: {move} (tile {moved_tile}) - empty moves from {old_pos} to {new_pos} ✓")
            else:
                validation_steps.append(f"Move {i+1}: {move} - INVALID MOVE ✗")
                valid_solution = False
                break
        
        # 显示前5步和后5步的验证
        if len(validation_steps) <= 10:
            for step in validation_steps:
                validation_text += f"{step}\n"
        else:
            for step in validation_steps[:5]:
                validation_text += f"{step}\n"
            validation_text += f"... (verified {len(validation_steps)-10} intermediate steps) ...\n"
            for step in validation_steps[-5:]:
                validation_text += f"{step}\n"
        
        # 检查最终状态
        final_state_correct = np.array_equal(replay_state, self.goal_state)
        validation_text += f"\n**Final State Verification:**\n"
        if final_state_correct:
            validation_text += f"✓ Final state matches the target configuration perfectly\n"
            validation_text += f"✓ All tiles are in their correct positions\n"
            validation_text += f"✓ Empty space is at bottom-right corner (3,3)\n"
        else:
            validation_text += f"✗ Final state does not match the target configuration\n"
            validation_text += f"Final state: {replay_state.tolist()}\n"
            validation_text += f"Target state: {self.goal_state.tolist()}\n"
        
        validation_text += f"\n**Solution Properties:**\n"
        validation_text += f"- Total moves: {len(solution_moves)}\n"
        validation_text += f"- All moves valid: {'Yes' if valid_solution else 'No'}\n"
        validation_text += f"- Reaches goal state: {'Yes' if final_state_correct else 'No'}\n"
        validation_text += f"- Move sequence: {' '.join(solution_moves)}\n"
        
        validation_text += f"\n**Reflection and Analysis:**\n"
        if valid_solution and final_state_correct:
            validation_text += f"The solution is verified to be correct and optimal. "
            validation_text += f"Each move follows the puzzle rules, and the sequence successfully transforms "
            validation_text += f"the initial state into the goal state. "
            
            # 分析解的质量
            if len(solution_moves) <= 10:
                validation_text += f"The solution is quite efficient with only {len(solution_moves)} moves.\n"
            elif len(solution_moves) <= 20:
                validation_text += f"The solution uses {len(solution_moves)} moves, which is reasonable for this configuration.\n"
            else:
                validation_text += f"The solution requires {len(solution_moves)} moves, indicating a complex initial configuration.\n"
                
            validation_text += f"\n**Key Insights:**\n"
            validation_text += f"- The heuristic-guided approach successfully found an optimal path\n"
            validation_text += f"- Strategic thinking (accepting temporary setbacks for long-term gains) was crucial\n"
            validation_text += f"- Each move contributed meaningfully to reaching the solution\n"
            validation_text += f"- The systematic verification confirms the robustness of the solution process\n"
        else:
            validation_text += f"There appears to be an error in the solution. This requires further investigation "
            validation_text += f"to identify where the solving process failed.\n"
        
        step4_body = validation_text

        # 组装完整 CoT
        step1_full = title + step1_header + step1_body
        step2_full = step1_full + step2_header + step2_body
        step3_full = step2_full + step3_header + step3_body
        step4_full = step3_full + step4_header + step4_body

        # 生成截断版本
        step1_part = title + step1_header + smart_truncate_words(step1_body)
        step2_part = step1_full + step2_header + smart_truncate_words(step2_body)
        step3_part = step2_full + step3_header + smart_truncate_words(step3_body)

        return {
            'cot': step4_full,
            'cot_step1_all': step1_full,
            'cot_step2_all': step2_full,
            'cot_step3_all': step3_full,
        }

    def generate_solvable_state(self, max_attempts=100, use_random_moves=False, difficulty="3"):
        # 根据难度调整moves参数
        difficulty_moves = {
            '1': (1, 10),
            '2': (11, 20), 
            '3': (21, 30),
            '4': (31, 40),
            '5': (41, 100)
        }
        
        if use_random_moves:
            min_moves, max_moves = difficulty_moves.get(difficulty, (10, 30))
            moves = random.randint(min_moves, max_moves)
            return self.generate_random_moves(moves)
        else:
            for _ in range(max_attempts):
                numbers = list(range(0, self.size*self.size))
                random.shuffle(numbers)
                state = np.array(numbers).reshape(self.size, self.size)
                
                if self.is_solvable(state):
                    return state
            
            print("Failed to generate random solvable state, using random moves method instead")
            min_moves, max_moves = difficulty_moves.get(difficulty, (10, 30))
            return self.generate_random_moves(random.randint(min_moves, max_moves))

    def generate(self, num_cases=1, difficulty=3, output_folder=None, **kwargs):
        """
        生成多个滑动拼图问题

        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别（1-5）
            output_folder: 输出文件夹路径，覆盖构造函数中设置的路径

        Returns:
            生成的问题列表
        """
        if output_folder is not None:
            self.output_folder = output_folder

        # 创建目录结构（与 BaseGenerator/main.py 约定对齐）
        self.image_dir = os.path.join(self.output_folder, 'images')
        os.makedirs(self.image_dir, exist_ok=True)

        print(f"Generating {num_cases} sliding puzzles with difficulty {difficulty}")

        # 获取难度参数
        difficulty_params = self._get_difficulty_params(difficulty)

        generated_items = []
        # 使用时间作为基础种子以确保批次内唯一性
        base_seed = int(time.time() * 1000) & 0xFFFFFFFF

        for i in range(num_cases):
            # 为每个案例生成唯一的seed
            case_seed = base_seed + i

            # 生成单个puzzle
            puzzle_info = self._generate_single_puzzle(case_seed, difficulty, difficulty_params)
            if puzzle_info:
                # 收集生成结果
                generated_items.append(puzzle_info)

        # 持久化保存到 annotations.json（与 BaseGenerator 一致）
        if generated_items:
            self.save_annotations(generated_items, self.output_folder)
        print(f"Generated {len(generated_items)} sliding puzzles successfully")
        return generated_items

    def _generate_single_puzzle(self, seed, target_difficulty, difficulty_params):
        """生成单个滑动拼图"""
        time_limit = 30

        print(f"Generating sliding puzzle with seed={seed}, target_difficulty={target_difficulty}")

        # 根据seed生成初始状态，尝试匹配目标难度
        initial_state = None
        difficulty_moves = difficulty_params.get('moves_range', (13, 20))
        min_mv, max_mv = difficulty_moves

        # 尝试若干次以匹配目标难度
        for k in range(8):
            local_seed = seed + k
            rng = random.Random(local_seed)
            mv = min_mv + (local_seed % max(1, (max_mv - min_mv + 1)))
            candidate = self.generate_random_moves(mv, rng)

            # 避免初始即终局
            if np.array_equal(candidate, self.goal_state):
                candidate = self.generate_random_moves(max(1, mv), rng)

            path, moves, tiles = self.solve(candidate, time_limit)
            if path is None:
                continue

            step_count = len(moves)
            if step_count == 0:
                continue

            # 根据步数确定实际难度
            if step_count <= 3:
                actual_difficulty = "1"
            elif step_count <= 5:
                actual_difficulty = "2"
            elif step_count <= 7:
                actual_difficulty = "3"
            elif step_count <= 9:
                actual_difficulty = "4"
            else:
                actual_difficulty = "5"

            # 如果匹配目标难度，使用这个状态
            if actual_difficulty == str(target_difficulty):
                initial_state = candidate
                break

        # 如果没有找到匹配的状态，使用默认方法
        if initial_state is None:
            initial_state = self.generate_state_from_seed(seed, self.size)

        # 额外保护：若初始状态已是目标态，执行一次最少步随机移动
        if np.array_equal(initial_state, self.goal_state):
            rng = random.Random(seed ^ 0x9E3779B97F4A7C15)
            initial_state = self.generate_random_moves(1, rng)

        # 解决puzzle
        start_time = time.time()
        path, moves, tiles = self.solve(initial_state, time_limit)
        solve_time = time.time() - start_time

        if path is None:
            print(f"Failed to solve puzzle with seed {seed} within {time_limit} seconds")
            return None

        step_count = len(moves)
        if step_count == 0:
            # 避免生成空答案：对状态做一步随机移动并重新求解
            rng = random.Random(seed ^ 0xBF58476D1CE4E5B97F4A7C15)
            initial_state = self.generate_random_moves(1, rng)
            path, moves, tiles = self.solve(initial_state, time_limit)
            if path is None or len(moves) == 0:
                # 兜底：再做几步
                initial_state = self.generate_random_moves(2, rng)
                path, moves, tiles = self.solve(initial_state, time_limit)
                if path is None or len(moves) == 0:
                    print("Could not avoid zero-step puzzle generation")
                    return None
            step_count = len(moves)

        print(f"Solved puzzle in {step_count} steps, solving time: {solve_time:.2f}s")

        # 根据步数确定难度
        if step_count <= 3:
            actual_difficulty = "1"
        elif step_count <= 5:
            actual_difficulty = "2"
        elif step_count <= 7:
            actual_difficulty = "3"
        elif step_count <= 9:
            actual_difficulty = "4"
        else:
            actual_difficulty = "5"

        # 生成图片
        image_filename = f"slidingpuzzle_{self.size}_{seed}.png"
        image_path = os.path.join(self.image_dir, image_filename)
        self.create_image(initial_state, image_path)

        # 生成推理过程
        cot_result = self.generate_cot(initial_state, moves)

        # 创建puzzle数据
        index = f"slidingpuzzle_{seed}"
        initial_state_list = initial_state.tolist()
        solution_details = " ".join(moves)

        puzzle_info = {
            "index": index,
            "category": "slidingpuzzle",
            "image": f"images/{image_filename}",
            "question": PROMPT_15PUZZLE_IMAGE,
            "question_language": PROMPT_15PUZZLE.format(str(initial_state_list)),
            "answer": solution_details,
            "initial_state": str(initial_state_list),
            "difficulty": actual_difficulty,
            "cot": cot_result['cot'],
            "cot_step1_all": cot_result['cot_step1_all'],
            "cot_step2_all": cot_result['cot_step2_all'],
            "cot_step3_all": cot_result['cot_step3_all']
        }

        return puzzle_info

    def _get_difficulty_params(self, difficulty):
        """根据难度级别获取相应的参数配置"""
        difficulty_config = {
            1: {'moves_range': (1, 6), 'description': 'Easy'},
            2: {'moves_range': (7, 12), 'description': 'Medium-Easy'},
            3: {'moves_range': (13, 20), 'description': 'Medium'},
            4: {'moves_range': (21, 30), 'description': 'Medium-Hard'},
            5: {'moves_range': (31, 45), 'description': 'Hard'}
        }
        return difficulty_config.get(difficulty, difficulty_config[3])
    
    def _solve_bidirectional(self, initial_state, time_limit=15):
        start_time = time.time()
        
        initial_tuple = tuple(map(tuple, initial_state))
        goal_tuple = tuple(map(tuple, self.goal_state))
        
        if initial_tuple == goal_tuple:
            return [initial_state], [], []
        
        forward_queue = deque([(initial_tuple, [])])
        forward_visited = {initial_tuple: (None, None, None)}
        
        backward_queue = deque([(goal_tuple, [])])
        backward_visited = {goal_tuple: (None, None, None)}
        
        nodes_explored = 0
        
        while forward_queue and backward_queue and time.time() - start_time < time_limit:
            if forward_queue:
                nodes_explored += 1
                current_tuple, _ = forward_queue.popleft()
                current_state = np.array(current_tuple)
                
                neighbors, moves, tiles = self.get_neighbors(current_state)
                
                for neighbor, move, tile in zip(neighbors, moves, tiles):
                    neighbor_tuple = tuple(map(tuple, neighbor))
                    
                    if neighbor_tuple not in forward_visited:
                        forward_visited[neighbor_tuple] = (current_tuple, move, tile)
                        forward_queue.append((neighbor_tuple, []))
                        
                        if neighbor_tuple in backward_visited:
                            return self._reconstruct_bidirectional_path(
                                neighbor_tuple, forward_visited, backward_visited, initial_state)
            
            if backward_queue:
                nodes_explored += 1
                current_tuple, _ = backward_queue.popleft()
                current_state = np.array(current_tuple)
                
                neighbors, moves, tiles = self.get_neighbors(current_state)
                
                for neighbor, move, tile in zip(neighbors, moves, tiles):
                    neighbor_tuple = tuple(map(tuple, neighbor))
                    
                    reverse_move = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}[move]
                    
                    if neighbor_tuple not in backward_visited:
                        backward_visited[neighbor_tuple] = (current_tuple, reverse_move, tile)
                        backward_queue.append((neighbor_tuple, []))
                        
                        if neighbor_tuple in forward_visited:
                            return self._reconstruct_bidirectional_path(
                                neighbor_tuple, forward_visited, backward_visited, initial_state)
            
            if nodes_explored % 1000 == 0 and time.time() - start_time >= time_limit:
                print(f"Time limit ({time_limit}s) reached, explored {nodes_explored} nodes")
                return None, None, None
        
        print("Could not find solution within time limit")
        return None, None, None
    
    def _reconstruct_bidirectional_path(self, meeting_point, forward_visited, backward_visited, initial_state):
        forward_path = []
        forward_moves = []
        forward_tiles = []
        
        current = meeting_point
        while current in forward_visited and forward_visited[current][0] is not None:
            prev, move, tile = forward_visited[current]
            current_state = np.array(current)
            forward_path.append(current_state)
            forward_moves.append(move)
            forward_tiles.append(tile)
            current = prev
        
        backward_path = []
        backward_moves = []
        backward_tiles = []
        
        current = meeting_point
        while current in backward_visited and backward_visited[current][0] is not None:
            prev, move, tile = backward_visited[current]
            current = prev
            current_state = np.array(current)
            backward_path.append(current_state)
            backward_moves.append(move)
            backward_tiles.append(tile)
        
        path = [initial_state] + list(reversed(forward_path)) + backward_path
        moves = list(reversed(forward_moves)) + backward_moves
        tiles = list(reversed(forward_tiles)) + backward_tiles
        
        return path, moves, tiles
