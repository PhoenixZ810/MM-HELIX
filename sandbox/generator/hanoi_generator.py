import json
import os
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, Any, List, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np
from utils.constants import PROMPT_HANOI_IMAGE, PROMPT_HANOI

# Import the base generator from the correct location
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from generator.base_generator import BaseGenerator
class TowerOfHanoiGenerator(BaseGenerator):
    """Generator adapted for generate.py manager.

    - Ignores the provided size; always generates 3-disk, 3-peg Hanoi.
    - Returns a single puzzle dict per call (manager persists it).
    - Difficulty is bucketed by minimal step count.
    - Includes chain-of-thought fields expected by the pipeline.
    """

    def __init__(self, output_folder):
        super().__init__(output_folder)
        self.task_name = 'hanoi'
        # self.task_dir = os.path.join(self.output_folder, self.task_name)
        # os.makedirs(self.task_dir, exist_ok=True)
        # self.image_dir = os.path.join(self.task_dir, 'images')
        # os.makedirs(self.image_dir, exist_ok=True)

    def generate(self, num_cases, difficulty, output_folder=None):
        """
        生成 Tower of Hanoi 问题的抽象方法。

        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别 (1-5)
            output_folder: 输出文件夹路径，覆盖构造函数中设置的路径

        Returns:
            生成的问题列表
        """
        if output_folder is None:
            output_folder = self.output_folder

        # 确保输出目录存在
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'images'), exist_ok=True)

        # 获取难度参数
        difficulty_params = self._get_difficulty_params(difficulty)

        puzzles = []
        timestamp_seed = int(time.time())  # 使用时间戳作为种子

        # 第一遍：生成所有puzzles但不存储文件，只收集数据和生成图片
        for i in range(num_cases):
            # 使用时间戳和循环索引生成唯一种子
            seed = timestamp_seed + i
            random.seed(seed)

            puzzle = self._generate_single_puzzle(seed, output_folder, difficulty_params)
            if puzzle:
                puzzles.append(puzzle)

        # 第二遍：批量保存所有puzzles到annotations.json
        if puzzles:
            self.save_puzzles(puzzles, output_folder)

        return puzzles

    def _generate_single_puzzle(self, seed, output_folder, difficulty_params):
        """生成单个 puzzle 的内部方法"""
        # Always use 3 disks
        num_disks = 3

        # Generate a valid random initial state that isn't already solved,
        # and ensure the solution has at least one move (avoid empty answer)
        max_resample_attempts = 50
        attempt = 0
        solution_moves = None
        initial_state = None
        while attempt < max_resample_attempts:
            candidate_state = self._generate_random_state(num_disks)
            # Skip already-solved states to avoid empty solution sequences
            if TowerOfHanoi(num_disks, [peg.copy() for peg in candidate_state]).is_goal_state():
                attempt += 1
                continue
            candidate_solution = self._solve_bfs(candidate_state, num_disks)
            if candidate_solution and len(candidate_solution) > 0:
                initial_state = candidate_state
                solution_moves = candidate_solution
                break
            attempt += 1

        if not solution_moves or len(solution_moves) == 0:
            # Safe fallback to a known non-goal, solvable state
            initial_state = [[], [3, 2, 1], []]
            solution_moves = self._solve_bfs(initial_state, num_disks) or []

        # Build answer string
        answer = ' '.join([f"({disk},{to_peg})" for (disk, _from, to_peg) in solution_moves])
        step_count = len(solution_moves)

        # Map steps to difficulty bucket
        difficulty = self._difficulty_from_steps(step_count)

        # Render image within the output_dir/images folder
        puzzle_id = f"hanoi_{num_disks}_{seed}"
        image_rel = f"images/{puzzle_id}.png"
        image_abs = os.path.join(output_folder, image_rel)
        TowerOfHanoi(num_disks, [peg.copy() for peg in initial_state]).visualize(image_abs)

        # Build CoT text fields
        cot_data = self.generate_cot(initial_state, solution_moves)

        # Compose the puzzle entry
        entry: Dict[str, Any] = {
            'index': puzzle_id,
            'category': 'hanoi',
            'image': image_rel,
            'question': PROMPT_HANOI_IMAGE,
            'question_language': PROMPT_HANOI.format(str(initial_state)),
            'answer': answer,
            'initial_state': str(initial_state),
            'difficulty': difficulty,
            'step_count': step_count,
            'cot': cot_data['cot_full'],
            'cot_step1_all': cot_data['cot_step1_all'],
            'cot_step2_all': cot_data['cot_step2_all'],
            'cot_step3_all': cot_data['cot_step3_all'],
        }

        return entry

    def _get_difficulty_params(self, difficulty):
        """
        根据难度级别获取相应的参数配置。

        Args:
            difficulty: 难度级别（1-5）

        Returns:
            dict: 包含难度参数的字典
        """
        # 对于 Tower of Hanoi，我们主要根据步数来确定难度
        # 这里可以根据需要调整参数
        params = {
            1: {'min_steps': 1, 'max_steps': 2},
            2: {'min_steps': 2, 'max_steps': 3},
            3: {'min_steps': 3, 'max_steps': 5},
            4: {'min_steps': 5, 'max_steps': 6},
            5: {'min_steps': 6, 'max_steps': 15}
        }
        return params.get(difficulty, params[3])  # 默认中等难度

    def save_puzzles(self, puzzles, output_folder):
        """Save puzzles to annotations.json using the base class method"""
        # Ensure puzzles have the required fields (same logic as original save_to_json)
        for puzzle in puzzles:
            # Make image paths relative to task directory
            if 'image' in puzzle and puzzle['image']:
                puzzle['image'] = os.path.relpath(puzzle['image'], output_folder)

            # Ensure required fields exist
            if 'difficulty' not in puzzle:
                puzzle['difficulty'] = "medium"

            if 'step_count' not in puzzle and 'answer' in puzzle:
                # Try to infer step count from answer if possible
                if isinstance(puzzle['answer'], list) and len(puzzle['answer']) > 0:
                    puzzle['step_count'] = len(puzzle['answer'])
                elif isinstance(puzzle['answer'], str) and ' ' in puzzle['answer']:
                    puzzle['step_count'] = len(puzzle['answer'].split())
                else:
                    puzzle['step_count'] = 0

        # Use the base class method to save annotations
        self.save_annotations(puzzles, output_folder)


    def _generate_random_state(self, num_disks: int) -> List[List[int]]:
        state: List[List[int]] = [[] for _ in range(3)]
        for disk in range(num_disks, 0, -1):
            peg = random.randint(0, 2)
            state[peg].append(disk)
        for peg in state:
            peg.sort(reverse=True)
        return state

    def _solve_bfs(self, initial_state: List[List[int]], num_disks: int, timeout_seconds: float = 5.0) -> List[Tuple[int, int, int]]:
        tower = TowerOfHanoi(num_disks, [peg.copy() for peg in initial_state])
        queue: deque = deque([(tower, [])])
        visited = {tower.to_tuple()}
        start = time.time()

        while queue and (time.time() - start) < timeout_seconds:
            current_tower, path = queue.popleft()
            if current_tower.is_goal_state():
                return path
            for move in current_tower.get_possible_moves():
                next_tower = current_tower.copy()
                try:
                    next_tower.apply_move(move)
                except ValueError:
                    continue
                tup = next_tower.to_tuple()
                if tup not in visited:
                    visited.add(tup)
                    queue.append((next_tower, path + [move]))
        return None

    def _difficulty_from_steps(self, steps: int) -> str:
        if steps <= 2:
            return '1'
        if steps <= 3:
            return '2'
        if steps <= 5:
            return '3'
        if steps <= 6:
            return '4'
        else:
            return '5'

    def generate_cot(self, initial_state: List[List[int]], solution_moves: List[Tuple[int, int, int]]) -> Dict[str, str]:
        """Compose enhanced chain-of-thought fields with detailed reasoning.

        Returns:
            dict with keys: cot_full, cot_step1_all, cot_step2_all, cot_step3_all
        """

        def format_state(state: List[List[int]]) -> str:
            """Format state as readable text representation"""
            peg_descriptions = []
            for i, peg in enumerate(state):
                if not peg:
                    peg_descriptions.append(f"Peg {i+1}: empty")
                else:
                    disk_order = " on top of ".join([f"disk {d}" for d in reversed(peg)])
                    peg_descriptions.append(f"Peg {i+1}: {disk_order} (bottom to top)")
            return "; ".join(peg_descriptions)

        def identify_movable_disks(state: List[List[int]]) -> List[str]:
            """Identify which disks can currently be moved"""
            movable = []
            for i, peg in enumerate(state):
                if peg:
                    movable.append(f"disk {peg[-1]} from peg {i+1}")
            return movable

        def analyze_constraints(state: List[List[int]]) -> List[str]:
            """Analyze movement constraints for each disk"""
            constraints = []
            for i, peg in enumerate(state):
                if peg:
                    top_disk = peg[-1]
                    valid_destinations = []
                    for j, target_peg in enumerate(state):
                        if i != j and (not target_peg or target_peg[-1] > top_disk):
                            valid_destinations.append(str(j+1))
                    if valid_destinations:
                        constraints.append(f"disk {top_disk} can move to peg(s): {', '.join(valid_destinations)}")
                    else:
                        constraints.append(f"disk {top_disk} has no valid moves (blocked)")
            return constraints

        def simulate_moves_with_states(initial: List[List[int]], moves: List[Tuple[int, int, int]]) -> List[str]:
            """Simulate each move and describe the resulting state"""
            current_state = [peg.copy() for peg in initial]
            move_descriptions = []
            
            for i, (disk, from_peg, to_peg) in enumerate(moves):
                # Apply the move
                current_state[from_peg-1].pop()
                current_state[to_peg-1].append(disk)
                
                # Describe the move and resulting state
                step_desc = f"Move {i+1}: Move disk {disk} from peg {from_peg} to peg {to_peg}\n"
                step_desc += f"   Result: {format_state(current_state)}"
                move_descriptions.append(step_desc)
            
            return move_descriptions

        intro = "I need to solve this Tower of Hanoi puzzle step by step. Let me carefully analyze the situation and develop a solution strategy.\n\n"

        # Step 1: Enhanced rule understanding
        step1 = (
            "### Step 1: Understanding the game rules and objectives\n\n"
            "First, let me clearly establish the rules and goals of this Tower of Hanoi puzzle:\n\n"
            "**Game Rules:**\n"
            "- There are 3 pegs labeled 1, 2, and 3 from left to right\n"
            "- There are 3 disks of different sizes: disk 1 (smallest), disk 2 (medium), disk 3 (largest)\n"
            "- Only the topmost disk on any peg can be moved at a time\n"
            "- A larger disk can never be placed on top of a smaller disk\n"
            "- Disks must be moved one at a time from one peg to another\n\n"
            "**Objective:**\n"
            "- Move all disks to peg 3 (rightmost peg)\n"
            "- Final arrangement should have disk 3 at the bottom, disk 2 in the middle, and disk 1 on top\n"
            "- This creates a tower in descending order of size from bottom to top\n\n"
            "**Key Strategic Insight:**\n"
            "- To move a large disk, all smaller disks above it must first be moved out of the way\n"
            "- The puzzle requires careful planning to avoid creating situations where disks become trapped"
        )

        # Step 2: Enhanced visual analysis
        movable_disks = identify_movable_disks(initial_state)
        constraints = analyze_constraints(initial_state)
        
        step2 = (
            "\n\n### Step 2: Careful image analysis and state reading\n\n"
            "Now I'll examine the image carefully to understand the current state of the puzzle:\n\n"
            "**Initial State Reading:**\n"
            f"Looking at the image, I can see the current arrangement: {format_state(initial_state)}\n\n"
            "**Movable Disks Analysis:**\n"
            f"Currently, I can move: {', '.join(movable_disks) if movable_disks else 'no disks (all pegs empty)'}\n\n"
            "**Movement Constraints:**\n"
        )
        
        for constraint in constraints:
            step2 += f"- {constraint}\n"
            
        step2 += (
            "\n**State Verification:**\n"
            "Let me double-check my reading of the image:\n"
        )
        
        for i, peg in enumerate(initial_state):
            if peg:
                step2 += f"- Peg {i+1}: Contains disks {peg} (from bottom to top)\n"
            else:
                step2 += f"- Peg {i+1}: Empty\n"
                
        step2 += (
            "\n**Reflection on Current State:**\n"
            "This reading appears correct based on the visual information. I can see the relative sizes of the disks "
            "and their positions clearly. Now I need to plan the optimal sequence of moves."
        )

        # Step 3: Enhanced strategic reasoning with detailed move-by-move analysis
        move_descriptions = simulate_moves_with_states(initial_state, solution_moves)
        
        step3 = (
            "\n\n### Step 3: Detailed strategic reasoning and problem-solving\n\n"
            "Now I'll work through the solution systematically, considering various strategies and exploring the optimal path:\n\n"
            "**Strategic Approach:**\n"
            "For a 3-disk Tower of Hanoi, the general strategy is:\n"
            "1. Move the top 2 disks (1 and 2) to an auxiliary peg\n"
            "2. Move the largest disk (3) to the destination peg\n"
            "3. Move the 2 disks from the auxiliary peg to the destination peg\n\n"
            "**Detailed Move Analysis:**\n"
            "Let me work through each move carefully:\n\n"
        )
        
        for move_desc in move_descriptions:
            step3 += move_desc + "\n\n"
            
        step3 += (
            "**Strategic Considerations:**\n"
            "Throughout this sequence, I need to ensure that:\n"
            "- Each move is legal (only moving top disks, never placing large on small)\n"
            "- I'm making progress toward the goal without creating deadlock situations\n"
            "- I'm using the minimum number of moves possible for efficiency\n\n"
            "**Alternative Paths Considered:**\n"
            "I considered other possible move sequences, but this path is optimal because it follows the "
            "recursive nature of the Tower of Hanoi problem, systematically clearing the way for larger disks "
            "while maintaining the ability to rebuild the tower at the destination."
        )

        # Step 4: Enhanced validation and reflection
        final_state = [peg.copy() for peg in initial_state]
        for (disk, from_peg, to_peg) in solution_moves:
            final_state[from_peg-1].pop()
            final_state[to_peg-1].append(disk)
            
        step4 = (
            "\n\n### Step 4: Solution validation and comprehensive reflection\n\n"
            "Let me verify that my solution is correct and complete:\n\n"
            "**Move-by-Move Validation:**\n"
        )
        
        # Validate each move
        test_state = [peg.copy() for peg in initial_state]
        for i, (disk, from_peg, to_peg) in enumerate(solution_moves):
            # Check if move is legal
            if not test_state[from_peg-1] or test_state[from_peg-1][-1] != disk:
                step4 += f"❌ Move {i+1}: INVALID - Disk {disk} is not on top of peg {from_peg}\n"
            elif test_state[to_peg-1] and test_state[to_peg-1][-1] < disk:
                step4 += f"❌ Move {i+1}: INVALID - Cannot place disk {disk} on smaller disk {test_state[to_peg-1][-1]}\n"
            else:
                step4 += f"✅ Move {i+1}: VALID - Move disk {disk} from peg {from_peg} to peg {to_peg}\n"
                test_state[from_peg-1].pop()
                test_state[to_peg-1].append(disk)
        
        step4 += (
            f"\n**Final State Check:**\n"
            f"After all moves, the final state is: {format_state(final_state)}\n"
        )
        
        # Check if goal is achieved
        goal_achieved = len(final_state[2]) == 3 and final_state[2] == [3, 2, 1]
        if goal_achieved:
            step4 += "✅ SUCCESS: All disks are on peg 3 in correct order (3, 2, 1 from bottom to top)\n"
        else:
            step4 += "❌ FAILURE: Goal state not achieved\n"
            
        step4 += (
            f"\n**Solution Metrics:**\n"
            f"- Total moves required: {len(solution_moves)}\n"
            f"- Theoretical minimum for 3 disks: {2**3 - 1} moves (when starting from peg 1 with all disks)\n"
            f"- Solution efficiency: {'Optimal' if len(solution_moves) <= 7 else 'Sub-optimal'}\n\n"
            "**Reflection on Problem-Solving Process:**\n"
            "This solution demonstrates the recursive nature of the Tower of Hanoi problem. Each step "
            "was carefully planned to ensure progress toward the goal while maintaining valid game states. "
            "The key insight was recognizing when to use auxiliary pegs to temporarily store disks "
            "while clearing paths for larger disks to reach their destination.\n\n"
            "**Verification Complete:**\n"
            "The solution is mathematically sound, follows all game rules, and successfully achieves "
            "the objective of moving all disks to peg 3 in the correct order."
        )

        s1_all = f"{intro}{step1}"
        s2_all = f"{s1_all}{step2}"
        s3_all = f"{s2_all}{step3}"
        cot_full = f"{s3_all}{step4}"

        return {
            'cot_full': cot_full,
            'cot_step1_all': s1_all,
            'cot_step2_all': s2_all,
            'cot_step3_all': s3_all,
        }


class TowerOfHanoi:
    def __init__(self, num_disks, initial_state=None):
        self.num_disks = num_disks
        if initial_state is None:
            self.state = [list(range(num_disks, 0, -1)), [], []]
        else:
            self.state = initial_state
    
    def is_valid_state(self):
        disks_seen = set()
        for peg in self.state:
            for i in range(len(peg) - 1):
                if peg[i] < peg[i + 1]:
                    return False
            disks_seen.update(peg)
        return disks_seen == set(range(1, self.num_disks + 1))
    
    def is_goal_state(self):
        return len(self.state[2]) == self.num_disks and sorted(self.state[2], reverse=True) == self.state[2]
    
    def get_possible_moves(self):
        moves = []
        for from_peg in range(3):
            if not self.state[from_peg]:
                continue
            disk = self.state[from_peg][-1]
            for to_peg in range(3):
                if from_peg == to_peg:
                    continue
                if not self.state[to_peg] or self.state[to_peg][-1] > disk:
                    moves.append((disk, from_peg + 1, to_peg + 1))
        return moves
    
    def apply_move(self, move):
        disk, from_peg, to_peg = move
        from_peg -= 1
        to_peg -= 1
        
        if not self.state[from_peg] or self.state[from_peg][-1] != disk:
            raise ValueError(f"Disk {disk} is not on top of peg {from_peg+1}")
        
        if not self.state[from_peg]:
            raise ValueError(f"Cannot move from empty peg {from_peg+1}")
        
        if self.state[to_peg] and self.state[to_peg][-1] < disk:
            raise ValueError(f"Cannot place disk {disk} on smaller disk {self.state[to_peg][-1]}")
        
        self.state[from_peg].pop()
        self.state[to_peg].append(disk)
        return disk
    
    def copy(self):
        return TowerOfHanoi(self.num_disks, [peg.copy() for peg in self.state])
    
    def to_tuple(self):
        return tuple(tuple(peg) for peg in self.state)
    
    def to_numerical(self):
        result = []
        for disk in range(1, self.num_disks + 1):
            for peg_idx, peg in enumerate(self.state):
                if disk in peg:
                    result.append(peg_idx)
                    break
        return result
    
    def visualize(self, filename):
        disk_height = 0.5
        
        max_peg_height = max(len(peg) for peg in self.state) * disk_height
        img_height = max(max_peg_height + 1, self.num_disks * disk_height + 1)
        
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('#F5F5F5')
        ax.set_facecolor('#F8F8F8')
        
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.3, img_height)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['Peg 1', 'Peg 2', 'Peg 3'], fontsize=12, fontweight='bold')
        ax.set_yticks([])
        
        base_height = 0.25
        ax.add_patch(plt.Rectangle((-0.5, -base_height), 3, base_height, 
                                facecolor='#8B4513', edgecolor='#654321', 
                                linewidth=2, alpha=0.9, zorder=1))
        ax.add_patch(plt.Rectangle((-0.5, -base_height-0.05), 3, 0.05, 
                                facecolor='#654321', alpha=0.6, zorder=0))
        
        for i in range(3):
            ax.plot([i, i], [0, img_height - 0.5], color='#654321', 
                linewidth=6, solid_capstyle='round', zorder=2)
        
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, self.num_disks))
        
        # Import FancyBboxPatch for rounded rectangles
        from matplotlib.patches import FancyBboxPatch
        
        for peg_idx, peg in enumerate(self.state):
            for height_idx, disk in enumerate(peg):
                y_position = height_idx * disk_height
                width = disk / self.num_disks * 0.8
                
                disk_color = colors[disk-1]
                
                # Create disk with rounded corners
                rounding_size = 0.15 * disk_height  # Control roundness
                disk_patch = FancyBboxPatch(
                    (peg_idx - width / 2, y_position),  # lower left corner
                    width, disk_height,  # width, height
                    boxstyle=f"round,pad=0,rounding_size={rounding_size}",
                    edgecolor='#333333',
                    facecolor=disk_color,
                    linewidth=1.5,
                    alpha=0.9,
                    zorder=10
                )
                ax.add_patch(disk_patch)
                
                # Add subtle highlight with adjusted width for rounded corners
                highlight_width = width * 0.9  # Slightly narrower
                highlight_y_pos = y_position + disk_height - 0.1
                highlight_rounding = min(rounding_size * 0.7, 0.05)
                
                highlight_patch = FancyBboxPatch(
                    (peg_idx - highlight_width / 2, highlight_y_pos),
                    highlight_width, 0.1,
                    boxstyle=f"round,pad=0,rounding_size={highlight_rounding}",
                    facecolor=self.lighten_color(disk_color),
                    linewidth=0,
                    alpha=0.6,
                    zorder=11
                )
                ax.add_patch(highlight_patch)
                
                # Center hole and text
                circle_radius = min(0.15, width/4)
                circle = plt.Circle((peg_idx, y_position + disk_height/2), circle_radius, 
                                facecolor='white', edgecolor='#333333', linewidth=1, zorder=15)
                ax.add_patch(circle)
                
                ax.text(peg_idx, y_position + disk_height/2, str(disk), 
                        horizontalalignment='center', 
                        verticalalignment='center',
                        fontsize=10, 
                        fontweight='bold',
                        color='black',
                        zorder=16)
        
        plt.title(f'Tower of Hanoi', 
                fontsize=14, fontweight='bold', pad=20)
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#BBBBBB')
            spine.set_linewidth(1.5)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        return filename
    
    @staticmethod
    def lighten_color(color, amount=0.5):
        import matplotlib.colors as mc
        import colorsys
        try:
            c = mc.to_rgb(color)
        except:
            c = color
        c = colorsys.rgb_to_hls(*c)
        return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


class HanoiGenerator(BaseGenerator):
    """Backward-compatible alias retained if referenced elsewhere."""
    def __init__(self, output_folder):
        super().__init__(output_folder)
        self._impl = TowerOfHanoiGenerator(output_folder)

    def generate(self, num_cases, difficulty, output_folder=None):
        return self._impl.generate(num_cases, difficulty, output_folder)

    def _get_difficulty_params(self, difficulty):
        return self._impl._get_difficulty_params(difficulty)