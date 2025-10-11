import os
import json
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Set
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


class BridgesGenerator(BaseGenerator):
    """Generator for Bridges (Hashiwokakero) puzzles with image visualization."""
    
    def __init__(self, output_folder):
        """
        Initialize the Bridges puzzle generator.
        """
        super().__init__(output_folder)
        # Seed initialization switched to program runtime timestamp (int)
        self.run_seed = int(time.time())
        random.seed(self.run_seed)
        np.random.seed(self.run_seed)
        # Updated to support 5 difficulty levels (1-5) with optimized parameters for uniqueness
        self.difficulty_settings = {
            1: {"min_islands": 3, "max_islands": 4, "grid_size": 5, "name": "1"},  # Increased for more variety
            2: {"min_islands": 4, "max_islands": 5, "grid_size": 6, "name": "2"},  # Increased for more variety
            3: {"min_islands": 5, "max_islands": 6, "grid_size": 7, "name": "3"},  # Increased for more variety
            4: {"min_islands": 6, "max_islands": 7, "grid_size": 8, "name": "4"},  # Increased for more variety
            5: {"min_islands": 7, "max_islands": 10, "grid_size": 9, "name": "5"}, # Increased for more variety

            "easy": {"min_islands": 2, "max_islands": 4, "grid_size": 5, "name": "easy"},
            "medium": {"min_islands": 4, "max_islands": 6, "grid_size": 7, "name": "medium"},
            "hard": {"min_islands": 6, "max_islands": 8, "grid_size": 8, "name": "hard"}
        }

    def _bridges_cross(self, from_a: Tuple[int, int], to_a: Tuple[int, int], from_b: Tuple[int, int], to_b: Tuple[int, int]) -> bool:
        """Return True if bridge A crosses bridge B at a non-endpoint intersection.

        Crossing is only possible when one is vertical and the other is horizontal, and
        their segments intersect strictly inside both segments (not at endpoints).
        """
        x1a, y1a = from_a
        x1b, y1b = to_a
        x2a, y2a = from_b
        x2b, y2b = to_b

        # Normalize orientation checks based on evaluator's logic
        if x1a == x1b and y2a == y2b:  # A vertical, B horizontal
            return (min(y1a, y1b) < y2a < max(y1a, y1b) and
                    min(x2a, x2b) < x1a < max(x2a, x2b))
        if y1a == y1b and x2a == x2b:  # A horizontal, B vertical
            return (min(x1a, x1b) < x2a < max(x1a, x1b) and
                    min(y2a, y2b) < y1a < max(y2a, y2b))
        return False

    def _would_cross_existing(self, new_from: Tuple[int, int], new_to: Tuple[int, int], bridges: List[Dict]) -> bool:
        """Return True if a candidate bridge would cross any existing bridge in list."""
        for b in bridges:
            b_from = tuple(b['from'])
            b_to = tuple(b['to'])
            # Same segment (possibly double bridge) is fine
            if {new_from, new_to} == {b_from, b_to}:
                continue
            if self._bridges_cross(new_from, new_to, b_from, b_to):
                return True
        return False

    def generate_simple_puzzle(self, n: int) -> Dict:
        """
        Generate a simple Bridges puzzle with two islands.
        
        Args:
            n (int): Grid size (n x n)
            
        Returns:
            Dict: A puzzle dictionary with islands and bridges
        """
        direction = random.choice(['horizontal', 'vertical'])
        islands = []
        bridges = []

        if direction == 'horizontal':
            x = random.randint(0, n-1)
            y1 = random.randint(0, n-3)
            y2 = y1 + 2
            islands = [
                {'x': x, 'y': y1, 'num': 2},
                {'x': x, 'y': y2, 'num': 2},
            ]
            bridges = [{'from': (x, y1), 'to': (x, y2), 'count': 2}]
        else:
            y = random.randint(0, n-1)
            x1 = random.randint(0, n-3)
            x2 = x1 + 2
            islands = [
                {'x': x1, 'y': y, 'num': 2},
                {'x': x2, 'y': y, 'num': 2},
            ]
            bridges = [{'from': (x1, y), 'to': (x2, y), 'count': 2}]

        return {
            'islands': islands,
            'bridges': bridges
        }
    
    def generate_all_difficulties(self, items_per_difficulty: int = 300) -> List[Dict]:
        """
        Generate puzzles for all 5 difficulty levels with duplicate checking.
        
        Args:
            items_per_difficulty (int): Number of puzzles to generate per difficulty level
            
        Returns:
            List[Dict]: List of all puzzle data
        """
        # Batch in-memory generation first, IO is performed at the end
        all_puzzles = []
        image_jobs = []  # (puzzle, img_path, grid_size, show_solution)
        generated_puzzles = set()  # To track unique puzzles and avoid duplicates
        
        for difficulty_level in range(1, 6):  # 1-5 difficulty levels
            print(f"正在生成难度级别 {difficulty_level} 的谜题...")
            
            settings = self.difficulty_settings[difficulty_level]
            grid_size = settings["grid_size"]
            
            generated_count = 0
            attempt_count = 0
            max_attempts = items_per_difficulty * 10  # Allow up to 10x attempts to avoid infinite loops
            
            while generated_count < items_per_difficulty and attempt_count < max_attempts:
                attempt_count += 1
                
                # Generate puzzle
                puzzle = self.generate_complex_puzzle(grid_size, difficulty_level)
                
                # Create a unique identifier for the puzzle based on its content
                puzzle_signature = self.create_puzzle_signature(puzzle)
                
                # Check if this puzzle is unique
                if puzzle_signature in generated_puzzles:
                    print(f"  发现重复谜题，重新生成... (尝试 {attempt_count})")
                    continue
                
                # Add to generated puzzles set
                generated_puzzles.add(puzzle_signature)
                generated_count += 1
                
                # Generate unique ID for the puzzle
                puzzle_id = f"bridges_difficulty_{difficulty_level}_item_{generated_count}"
                
                # Queue image saving jobs to end to avoid progressive IO
                img_path = f"tasks/images/{puzzle_id}.png"
                solution_img_path = f"tasks/images/{puzzle_id}_solution.png"
                image_jobs.append((puzzle, img_path, grid_size, False))
                image_jobs.append((puzzle, solution_img_path, grid_size, True))
                
                # Generate step-by-step solution
                solution_steps = self.solve(puzzle)
                
                # Generate rule-based CoT with all step parts
                cot_data = self.generate_cot(puzzle, grid_size)
                
                # Format the answer string
                answer = self.format_answer(puzzle)
                
                # Create the puzzle data structure
                puzzle_data = {
                    "index": puzzle_id,
                    "category": "bridges",
                    "difficulty_level": difficulty_level,
                    "item_number": generated_count,
                    "grid_size": f"{grid_size}x{grid_size}",
                    "image": img_path,
                    "solution_image": solution_img_path,
                    "question": self.format_question_with_image(),
                    "question_language": self.format_question_text(puzzle),
                    "answer": answer,
                    "initial_state": json.dumps(puzzle),
                    "difficulty": f"{difficulty_level}",
                    "cot": cot_data["full_cot"],
                    "cot_step1_part": cot_data["cot_step1_part"],
                    "cot_step1_all": cot_data["cot_step1_all"],
                    "cot_step2_all": cot_data["cot_step2_all"],
                    "cot_step3_all": cot_data["cot_step3_all"],
                    "step_count": len(puzzle['bridges']),
                    "island_count": len(puzzle['islands'])
                }
                
                all_puzzles.append(puzzle_data)
                
                # Print progress every 50 puzzles
                if generated_count % 50 == 0:
                    print(f"  已生成 {generated_count}/{items_per_difficulty} 个谜题")
            
            if generated_count < items_per_difficulty:
                print(f"  警告：难度级别 {difficulty_level} 只生成了 {generated_count}/{items_per_difficulty} 个唯一谜题")
            else:
                print(f"  难度级别 {difficulty_level} 成功生成 {generated_count} 个唯一谜题")

        # Perform batched IO at the end
        os.makedirs('tasks/images', exist_ok=True)
        for puzzle, path, grid_size, show_solution in image_jobs:
            self.visualize(puzzle, save_path=path, grid_size=grid_size, show_solution=show_solution)

        self.save_annotations(all_puzzles, 'tasks')

        return all_puzzles

    def create_puzzle_signature(self, puzzle: Dict) -> str:
        """
        Create a unique signature for a puzzle to detect duplicates.
        
        Args:
            puzzle (Dict): The puzzle to create signature for
            
        Returns:
            str: A unique signature string for the puzzle
        """
        # Sort islands by position for consistent comparison
        sorted_islands = sorted(puzzle['islands'], key=lambda x: (x['x'], x['y']))
        
        # Sort bridges by position for consistent comparison
        sorted_bridges = sorted(puzzle['bridges'], key=lambda x: (x['from'], x['to']))
        
        # Create signature based on sorted islands and bridges
        islands_sig = tuple((island['x'], island['y'], island['num']) for island in sorted_islands)
        bridges_sig = tuple((bridge['from'], bridge['to'], bridge['count']) for bridge in sorted_bridges)
        
        # Combine both signatures
        signature = (islands_sig, bridges_sig)
        
        # Return as string for easier handling
        return str(signature)

    def generate_complex_puzzle(self, n: int, difficulty=3) -> Dict:
        """
        Generate a more complex Bridges puzzle based on difficulty level.
        
        Args:
            n (int): Grid size (n x n)
            difficulty (int or str): Difficulty level (1-5 or legacy string)
            
        Returns:
            Dict: A puzzle dictionary with islands and bridges
        """
        # Handle both numeric and string difficulty levels
        if isinstance(difficulty, (int, float)):
            if difficulty in self.difficulty_settings:
                settings = self.difficulty_settings[difficulty]
            else:
                settings = self.difficulty_settings[3]  # Default to level 3
        else:
            settings = self.difficulty_settings.get(difficulty, self.difficulty_settings["medium"])
        
        num_islands = random.randint(settings["min_islands"], settings["max_islands"])
        
        # Use the provided grid size
        width, height = n, n
        grid = [[False for _ in range(width)] for _ in range(height)]
        
        # Place islands with more variety for better uniqueness
        islands = []
        placement_attempts = 0
        max_placement_attempts = 100
        
        while len(islands) < num_islands and placement_attempts < max_placement_attempts:
            placement_attempts += 1
            
            # Try different placement strategies based on difficulty
            if difficulty == 1:
                # For easiest puzzles, try to place islands in different patterns
                if len(islands) == 0:
                    # First island can be anywhere
                    x = random.randint(0, width-1)
                    y = random.randint(0, height-1)
                elif len(islands) == 1:
                    # Second island should be at a reasonable distance
                    first_island = islands[0]
                    attempts = 0
                    while attempts < 20:
                        x = random.randint(0, width-1)
                        y = random.randint(0, height-1)
                        # Ensure minimum distance of 2 for interesting puzzles
                        if abs(x - first_island['x']) + abs(y - first_island['y']) >= 2:
                            break
                        attempts += 1
                    if attempts >= 20:
                        continue
                else:
                    # Additional islands
                    x = random.randint(0, width-1)
                    y = random.randint(0, height-1)
            else:
                # For higher difficulties, more random placement
                x = random.randint(0, width-1)
                y = random.randint(0, height-1)
            
            if not grid[y][x]:
                grid[y][x] = True
                islands.append({'x': x, 'y': y, 'num': 0})  # Placeholder for number
        
        if len(islands) < 2:  # Need at least 2 islands
            return self.generate_complex_puzzle(n, difficulty)
        
        # Try to connect islands
        bridges = []
        island_connections = defaultdict(int)  # Track how many bridges for each island
        
        # Function to check if we can place a bridge between two islands
        def can_place_bridge(island1: Dict, island2: Dict) -> bool:
            x1, y1 = island1['x'], island1['y']
            x2, y2 = island2['x'], island2['y']
            
            # Must be in same row or column
            if not (x1 == x2 or y1 == y2):
                return False
                
            # Check for islands in between
            if x1 == x2:  # Vertical bridge
                for y in range(min(y1, y2) + 1, max(y1, y2)):
                    if grid[y][x1]:
                        return False
            else:  # Horizontal bridge
                for x in range(min(x1, x2) + 1, max(x1, x2)):
                    if grid[y1][x]:
                        return False
            # Check for crossings with already added bridges
            if self._would_cross_existing((x1, y1), (x2, y2), bridges):
                return False
            
            return True
        
        # Connect islands to form a connected network with more variation
        connected = set([0])  # Start with the first island
        remaining = set(range(1, len(islands)))
        
        # Randomize connection order for more variety
        connection_order = list(remaining)
        random.shuffle(connection_order)
        
        while remaining:
            added = False
            # Try to connect in randomized order
            for j in connection_order:
                if j not in remaining:
                    continue
                    
                # Find all possible connections from connected islands
                possible_connections = []
                for i in connected:
                    if can_place_bridge(islands[i], islands[j]):
                        max_bridges = min(3 - island_connections[i], 3 - island_connections[j], 2)
                        if max_bridges > 0:
                            possible_connections.append((i, j, max_bridges))
                
                if possible_connections:
                    # Choose a random connection
                    i, j, max_bridges = random.choice(possible_connections)
                    
                    # Add some randomness to bridge count selection
                    if difficulty == 1:
                        # For easy puzzles, prefer single bridges
                        bridge_count = 1 if max_bridges >= 1 else max_bridges
                    else:
                        # For harder puzzles, vary bridge counts
                        bridge_count = random.randint(1, max_bridges)
                    
                    from_pos = (islands[i]['x'], islands[i]['y'])
                    to_pos = (islands[j]['x'], islands[j]['y'])
                    
                    # Ensure consistent ordering
                    if from_pos > to_pos:
                        from_pos, to_pos = to_pos, from_pos
                    
                    bridges.append({'from': from_pos, 'to': to_pos, 'count': bridge_count})
                    
                    # Update island bridge counts
                    islands[i]['num'] += bridge_count
                    islands[j]['num'] += bridge_count
                    
                    # Update connection tracking
                    island_connections[i] += bridge_count
                    island_connections[j] += bridge_count
                    
                    connected.add(j)
                    remaining.remove(j)
                    added = True
                    break
            
            if not added and remaining:  # Could not add more connections
                # Try again with a new puzzle
                return self.generate_complex_puzzle(n, difficulty)
        
        # Add some additional random bridges for variety (if possible)
        if difficulty > 1:
            additional_bridges = random.randint(0, min(2, len(islands) - 1))
            for _ in range(additional_bridges):
                # Try to add random bridges between already connected islands
                available_pairs = []
                for i in range(len(islands)):
                    for j in range(i + 1, len(islands)):
                        if (can_place_bridge(islands[i], islands[j]) and
                            island_connections[i] < 3 and island_connections[j] < 3):
                            # Check if this bridge doesn't already exist
                            from_pos = (islands[i]['x'], islands[i]['y'])
                            to_pos = (islands[j]['x'], islands[j]['y'])
                            if from_pos > to_pos:
                                from_pos, to_pos = to_pos, from_pos
                            
                            bridge_exists = any(
                                bridge['from'] == from_pos and bridge['to'] == to_pos
                                for bridge in bridges
                            )
                            
                            # Also ensure no crossing with existing bridges
                            if not bridge_exists and not self._would_cross_existing(from_pos, to_pos, bridges):
                                available_pairs.append((i, j))
                
                if available_pairs:
                    i, j = random.choice(available_pairs)
                    max_bridges = min(3 - island_connections[i], 3 - island_connections[j], 2)
                    if max_bridges > 0:
                        bridge_count = random.randint(1, max_bridges)
                        
                        from_pos = (islands[i]['x'], islands[i]['y'])
                        to_pos = (islands[j]['x'], islands[j]['y'])
                        
                        if from_pos > to_pos:
                            from_pos, to_pos = to_pos, from_pos
                        
                        bridges.append({'from': from_pos, 'to': to_pos, 'count': bridge_count})
                        
                        islands[i]['num'] += bridge_count
                        islands[j]['num'] += bridge_count
                        
                        island_connections[i] += bridge_count
                        island_connections[j] += bridge_count
        
        return {
            'islands': islands,
            'bridges': bridges
        }

    def generate_seeded_puzzle(self, size: int, seed: int, difficulty: int) -> Dict:
        """
        Generate a puzzle with deterministic seeded randomization.
        
        Args:
            size (int): Grid size (size x size)
            seed (int): Random seed for reproducible generation
            difficulty (int): Difficulty level (1-5)
            
        Returns:
            Dict: A puzzle dictionary with islands and bridges
        """
        # Create a unique random state for this size-seed combination
        local_random = random.Random(seed + size * 10000)
        
        # Use difficulty settings but ensure reproducibility
        if difficulty in self.difficulty_settings:
            settings = self.difficulty_settings[difficulty]
        else:
            settings = self.difficulty_settings[3]  # Default to level 3
        
        # Generate number of islands deterministically
        num_islands = local_random.randint(settings["min_islands"], settings["max_islands"])
        
        # Use the provided grid size
        width, height = size, size
        grid = [[False for _ in range(width)] for _ in range(height)]
        
        # Place islands with deterministic seeded placement
        islands = []
        placement_attempts = 0
        max_placement_attempts = 100
        
        while len(islands) < num_islands and placement_attempts < max_placement_attempts:
            placement_attempts += 1
            
            # Generate positions deterministically based on seed
            x = local_random.randint(0, width-1)
            y = local_random.randint(0, height-1)
            
            # For first few islands, ensure they're not too close for interesting puzzles
            if len(islands) > 0 and difficulty == 1:
                min_distance = 2
                too_close = any(
                    abs(x - island['x']) + abs(y - island['y']) < min_distance
                    for island in islands
                )
                if too_close and placement_attempts < max_placement_attempts - 10:
                    continue
            
            if not grid[y][x]:
                grid[y][x] = True
                islands.append({'x': x, 'y': y, 'num': 0})  # Placeholder for number
        
        if len(islands) < 2:  # Need at least 2 islands
            # Retry with modified seed to avoid infinite recursion
            return self.generate_seeded_puzzle(size, seed + 1, difficulty)
        
        # Connect islands deterministically
        bridges = []
        island_connections = defaultdict(int)
        
        # Function to check if we can place a bridge between two islands
        def can_place_bridge(island1: Dict, island2: Dict) -> bool:
            x1, y1 = island1['x'], island1['y']
            x2, y2 = island2['x'], island2['y']
            
            # Must be in same row or column
            if not (x1 == x2 or y1 == y2):
                return False
                
            # Check for islands in between
            if x1 == x2:  # Vertical bridge
                for y in range(min(y1, y2) + 1, max(y1, y2)):
                    if grid[y][x1]:
                        return False
            else:  # Horizontal bridge
                for x in range(min(x1, x2) + 1, max(x1, x2)):
                    if grid[y1][x]:
                        return False
            # Check for crossings with already added bridges
            if self._would_cross_existing((x1, y1), (x2, y2), bridges):
                return False
            
            return True
        
        # Connect islands to form a connected network deterministically
        connected = set([0])  # Start with the first island
        remaining = set(range(1, len(islands)))
        
        # Create deterministic connection order based on seed
        connection_order = list(remaining)
        local_random.shuffle(connection_order)
        
        while remaining:
            added = False
            # Try to connect in deterministic order
            for j in connection_order:
                if j not in remaining:
                    continue
                    
                # Find all possible connections from connected islands
                possible_connections = []
                for i in connected:
                    if can_place_bridge(islands[i], islands[j]):
                        max_bridges = min(3 - island_connections[i], 3 - island_connections[j], 2)
                        if max_bridges > 0:
                            possible_connections.append((i, j, max_bridges))
                
                if possible_connections:
                    # Choose connection deterministically
                    i, j, max_bridges = possible_connections[0]  # Take first valid connection
                    
                    # Determine bridge count deterministically
                    if difficulty == 1:
                        bridge_count = 1 if max_bridges >= 1 else max_bridges
                    else:
                        bridge_count = local_random.randint(1, max_bridges)
                    
                    from_pos = (islands[i]['x'], islands[i]['y'])
                    to_pos = (islands[j]['x'], islands[j]['y'])
                    
                    # Ensure consistent ordering
                    if from_pos > to_pos:
                        from_pos, to_pos = to_pos, from_pos
                    
                    bridges.append({'from': from_pos, 'to': to_pos, 'count': bridge_count})
                    
                    # Update island bridge counts
                    islands[i]['num'] += bridge_count
                    islands[j]['num'] += bridge_count
                    
                    # Update connection tracking
                    island_connections[i] += bridge_count
                    island_connections[j] += bridge_count
                    
                    connected.add(j)
                    remaining.remove(j)
                    added = True
                    break
            
            if not added and remaining:  # Could not add more connections
                # Retry with modified seed
                return self.generate_seeded_puzzle(size, seed + 1, difficulty)
        
        return {
            'islands': islands,
            'bridges': bridges
        }

    def generate_cot(self, puzzle: Dict, grid_size: int) -> Dict:
        """
        Generate comprehensive rule-based Chain of Thought reasoning for solving the puzzle.
        
        Args:
            puzzle (Dict): The puzzle to generate reasoning for
            grid_size (int): Size of the grid
            
        Returns:
            Dict: Dictionary containing full CoT and step-by-step parts
        """
        islands = puzzle['islands']
        bridges = puzzle['bridges']
        
        # Step 1: Understanding the puzzle rules and objectives (详细明确)
        step1_lines = []
        step1_lines.append("Let me solve this Bridges puzzle (also known as Hashiwokakero) step by step using systematic reasoning.")
        step1_lines.append("")
        step1_lines.append("### Step 1: Understanding the Game Rules and Objectives")
        step1_lines.append("")
        step1_lines.append("**Game Type:** Bridges (Hashiwokakero) - a logic puzzle involving connecting numbered islands with bridges.")
        step1_lines.append("")
        step1_lines.append("**Core Objective:** Connect all numbered islands using bridges such that each island has exactly the number of bridges specified by its number.")
        step1_lines.append("")
        step1_lines.append("**Critical Rules I must follow:**")
        step1_lines.append("1. **Bridge Direction:** Bridges can only run horizontally or vertically (never diagonally)")
        step1_lines.append("2. **Island Numbers:** Each island shows how many bridges must connect to it (1, 2, 3, etc.)")
        step1_lines.append("3. **No Crossings:** Bridges cannot cross over other bridges or pass through islands")
        step1_lines.append("4. **Bridge Limits:** Maximum of 2 bridges can connect any pair of islands")
        step1_lines.append("5. **Single Network:** All islands must form one connected network (no isolated groups)")
        step1_lines.append("6. **Path Clarity:** Bridges must have clear, unobstructed paths between islands")
        step1_lines.append("")
        step1_lines.append("**Success Criteria:** Every island's bridge count matches its number AND all islands are interconnected.")
        
        step1_text = "\n".join(step1_lines)
        
        # Step 2: Careful image reading and state extraction (精确读取和反思)
        step2_lines = []
        step2_lines.append("### Step 2: Careful Image Analysis and State Extraction")
        step2_lines.append("")
        step2_lines.append("**Initial Visual Scan:**")
        step2_lines.append(f"Looking at the image, I can see a {grid_size}×{grid_size} grid with numbered islands positioned at specific coordinates.")
        step2_lines.append("")
        step2_lines.append("**Precise State Reading:**")
        step2_lines.append(f"Grid dimensions: {grid_size} rows × {grid_size} columns")
        step2_lines.append(f"Coordinate system: (0,0) at top-left, X increases rightward, Y increases downward")
        step2_lines.append(f"Total islands detected: {len(islands)}")
        step2_lines.append("")
        
        # Create a text-based grid representation for better visualization
        grid_matrix = [['.' for _ in range(grid_size)] for _ in range(grid_size)]
        for island in islands:
            grid_matrix[island['y']][island['x']] = str(island['num'])
        
        step2_lines.append("**Grid State Representation:**")
        step2_lines.append("```")
        # Add column headers
        header = "   " + " ".join(f"{i:2d}" for i in range(grid_size))
        step2_lines.append(header)
        # Add rows with row numbers
        for y in range(grid_size):
            row_str = f"{y:2d} " + "  ".join(grid_matrix[y])
            step2_lines.append(row_str)
        step2_lines.append("```")
        step2_lines.append("(Numbers = islands with bridge requirements, '.' = empty cells)")
        step2_lines.append("")
        
        # Sort islands by position for consistent analysis
        sorted_islands = sorted(islands, key=lambda x: (x['y'], x['x']))
        step2_lines.append("**Detailed Island Analysis:**")
        total_bridges_needed = 0
        for island in sorted_islands:
            step2_lines.append(f"- Island at position ({island['x']},{island['y']}) requires exactly {island['num']} bridge connections")
            total_bridges_needed += island['num']
        
        step2_lines.append(f"- Total bridge endpoints needed: {total_bridges_needed} (each bridge connects 2 islands)")
        step2_lines.append(f"- Expected number of bridges: {total_bridges_needed // 2}")
        step2_lines.append("")
        
        step2_lines.append("**Reflection on State Interpretation:**")
        step2_lines.append("Let me verify my reading is correct by checking constraints:")
        
        # Check for potential connection paths
        step2_lines.append("- Checking for horizontal/vertical alignment opportunities...")
        connection_possibilities = 0
        for i, island1 in enumerate(islands):
            for j, island2 in enumerate(islands[i+1:], i+1):
                if (island1['x'] == island2['x'] or island1['y'] == island2['y']):
                    # Check if path is clear
                    x1, y1, x2, y2 = island1['x'], island1['y'], island2['x'], island2['y']
                    path_clear = True
                    if x1 == x2:  # Vertical connection
                        for y in range(min(y1, y2) + 1, max(y1, y2)):
                            if any(island['x'] == x1 and island['y'] == y for island in islands):
                                path_clear = False
                                break
                    else:  # Horizontal connection
                        for x in range(min(x1, x2) + 1, max(x1, x2)):
                            if any(island['x'] == x and island['y'] == y1 for island in islands):
                                path_clear = False
                                break
                    if path_clear:
                        connection_possibilities += 1
        
        step2_lines.append(f"- Found {connection_possibilities} possible clear connection paths")
        step2_lines.append("- State reading appears consistent with puzzle constraints")
        step2_lines.append("- Ready to proceed with systematic solving approach")
        
        step2_text = "\n".join(step2_lines)
        
        # Step 3: Detailed reasoning and exploration (充分探索和详细推理)
        step3_lines = []
        step3_lines.append("### Step 3: Systematic Reasoning and Solution Exploration")
        step3_lines.append("")
        step3_lines.append("**Strategic Analysis Phase:**")
        step3_lines.append("")
        
        # Categorize islands by bridge requirements
        high_requirement_islands = [island for island in islands if island['num'] >= 3]
        medium_requirement_islands = [island for island in islands if island['num'] == 2]
        low_requirement_islands = [island for island in islands if island['num'] == 1]
        
        step3_lines.append("**Island Categorization by Constraint Level:**")
        if high_requirement_islands:
            step3_lines.append(f"- High-constraint islands (≥3 bridges): {len(high_requirement_islands)} found")
            for island in high_requirement_islands:
                step3_lines.append(f"  * Island ({island['x']},{island['y']}) needs {island['num']} bridges - heavily constrains solution space")
        
        if medium_requirement_islands:
            step3_lines.append(f"- Medium-constraint islands (2 bridges): {len(medium_requirement_islands)} found")
            for island in medium_requirement_islands:
                step3_lines.append(f"  * Island ({island['x']},{island['y']}) needs {island['num']} bridges - moderate flexibility")
        
        if low_requirement_islands:
            step3_lines.append(f"- Low-constraint islands (1 bridge): {len(low_requirement_islands)} found")
            for island in low_requirement_islands:
                step3_lines.append(f"  * Island ({island['x']},{island['y']}) needs {island['num']} bridge - maximum flexibility")
        
        step3_lines.append("")
        step3_lines.append("**Connection Possibility Analysis:**")
        
        # Analyze all possible connections systematically
        all_possible_connections = []
        for i, island1 in enumerate(islands):
            for j, island2 in enumerate(islands[i+1:], i+1):
                if (island1['x'] == island2['x'] or island1['y'] == island2['y']):
                    # Check if path is clear
                    x1, y1, x2, y2 = island1['x'], island1['y'], island2['x'], island2['y']
                    path_clear = True
                    blocking_islands = []
                    
                    if x1 == x2:  # Vertical connection
                        for y in range(min(y1, y2) + 1, max(y1, y2)):
                            blocking = [island for island in islands if island['x'] == x1 and island['y'] == y]
                            if blocking:
                                path_clear = False
                                blocking_islands.extend(blocking)
                    else:  # Horizontal connection
                        for x in range(min(x1, x2) + 1, max(x1, x2)):
                            blocking = [island for island in islands if island['x'] == x and island['y'] == y1]
                            if blocking:
                                path_clear = False
                                blocking_islands.extend(blocking)
                    
                    distance = abs(x1 - x2) + abs(y1 - y2)
                    direction = "vertical" if x1 == x2 else "horizontal"
                    
                    connection_info = {
                        'from': island1, 'to': island2, 'clear': path_clear, 
                        'direction': direction, 'distance': distance,
                        'blocking': blocking_islands
                    }
                    all_possible_connections.append(connection_info)
        
        step3_lines.append("Evaluating all possible island pairs:")
        for i, conn in enumerate(all_possible_connections):
            island1, island2 = conn['from'], conn['to']
            status = "✓ POSSIBLE" if conn['clear'] else "✗ BLOCKED"
            step3_lines.append(f"{i+1}. ({island1['x']},{island1['y']}) ↔ ({island2['x']},{island2['y']}) {conn['direction']} distance {conn['distance']}: {status}")
            
            if not conn['clear']:
                for blocker in conn['blocking']:
                    step3_lines.append(f"   - Blocked by island at ({blocker['x']},{blocker['y']})")
        
        step3_lines.append("")
        step3_lines.append("**Solution Construction Process:**")
        step3_lines.append("")
        
        # Sort bridges for logical presentation of the actual solution
        sorted_bridges = sorted(bridges, key=lambda b: (min(b['from'], b['to']), max(b['from'], b['to'])))
        
        step3_lines.append("Working through the solution systematically:")
        step3_lines.append("")
        
        # Track island satisfaction as we build the solution
        island_satisfaction = {(island['x'], island['y']): 0 for island in islands}
        
        # Show reasoning process for each bridge
        for i, bridge in enumerate(sorted_bridges):
            from_x, from_y = bridge['from']
            to_x, to_y = bridge['to']
            count = bridge['count']
            
            # Find the corresponding islands
            from_island = next(island for island in islands 
                              if island['x'] == from_x and island['y'] == from_y)
            to_island = next(island for island in islands 
                            if island['x'] == to_x and island['y'] == to_y)
            
            direction = "horizontal" if from_y == to_y else "vertical"
            bridge_desc = f"{count} bridge{'s' if count > 1 else ''}"
            distance = abs(from_x - to_x) + abs(from_y - to_y)
            
            step3_lines.append(f"**Connection {i+1}: ({from_x},{from_y}) ↔ ({to_x},{to_y})**")
            step3_lines.append(f"- Direction: {direction} connection across {distance} grid units")
            step3_lines.append(f"- Bridge count: {bridge_desc}")
            
            # Calculate current satisfaction before this bridge
            current_from = island_satisfaction[(from_x, from_y)]
            current_to = island_satisfaction[(to_x, to_y)]
            
            step3_lines.append(f"- Island ({from_x},{from_y}) status: {current_from}/{from_island['num']} → {current_from + count}/{from_island['num']}")
            step3_lines.append(f"- Island ({to_x},{to_y}) status: {current_to}/{to_island['num']} → {current_to + count}/{to_island['num']}")
            
            # Update satisfaction tracking
            island_satisfaction[(from_x, from_y)] += count
            island_satisfaction[(to_x, to_y)] += count
            
            # Reasoning about this choice
            step3_lines.append("- **Reasoning:**")
            if count == 2:
                step3_lines.append("  * Using double bridge maximizes efficiency for high-requirement islands")
                step3_lines.append("  * Double bridges help satisfy multiple bridge requirements quickly")
            else:
                step3_lines.append("  * Single bridge provides precise connection without over-constraining")
            
            # Check if this connection helps with constraint satisfaction
            high_constraint_help = (from_island['num'] >= 3 or to_island['num'] >= 3)
            if high_constraint_help:
                step3_lines.append("  * This connection helps satisfy high-constraint island requirements")
            
            # Check connectivity implications
            step3_lines.append("  * This connection contributes to overall network connectivity")
            
            remaining_from = from_island['num'] - island_satisfaction[(from_x, from_y)]
            remaining_to = to_island['num'] - island_satisfaction[(to_x, to_y)]
            
            if remaining_from == 0:
                step3_lines.append(f"  * Island ({from_x},{from_y}) is now fully satisfied")
            else:
                step3_lines.append(f"  * Island ({from_x},{from_y}) still needs {remaining_from} more bridge(s)")
                
            if remaining_to == 0:
                step3_lines.append(f"  * Island ({to_x},{to_y}) is now fully satisfied")
            else:
                step3_lines.append(f"  * Island ({to_x},{to_y}) still needs {remaining_to} more bridge(s)")
            
            step3_lines.append("")
        
        step3_lines.append("**Network Connectivity Verification:**")
        step3_lines.append("- Checking that all islands form a single connected network...")
        step3_lines.append("- Verifying no bridges cross each other...")
        step3_lines.append("- Confirming all paths are geometrically valid...")
        step3_lines.append("- All connectivity requirements satisfied ✓")
        
        step3_text = "\n".join(step3_lines)
        
        # Step 4: Comprehensive validation and reflection (基于最终答案的验证和反思)
        step4_lines = []
        step4_lines.append("### Step 4: Solution Validation and Reflection")
        step4_lines.append("")
        step4_lines.append("**Final Answer Summary:**")
        
        # Format the complete solution
        answer_lines = []
        for bridge in sorted_bridges:
            x1, y1 = bridge['from']
            x2, y2 = bridge['to']
            count = bridge['count']
            answer_lines.append(f"({x1},{y1})-({x2},{y2}):{count}")
        
        step4_lines.append("```")
        step4_lines.extend(answer_lines)
        step4_lines.append("```")
        step4_lines.append("")
        
        step4_lines.append("**Comprehensive Validation Checks:**")
        step4_lines.append("")
        
        step4_lines.append("1. **Individual Island Verification:**")
        all_satisfied = True
        for island in sorted_islands:
            actual_count = sum(bridge['count'] for bridge in bridges 
                             if (bridge['from'] == (island['x'], island['y']) or 
                                 bridge['to'] == (island['x'], island['y'])))
            required_count = island['num']
            status = "✓" if actual_count == required_count else "✗"
            step4_lines.append(f"   - Island ({island['x']},{island['y']}): {actual_count}/{required_count} bridges {status}")
            if actual_count != required_count:
                all_satisfied = False
        
        step4_lines.append("")
        step4_lines.append("2. **Network Connectivity Check:**")
        step4_lines.append(f"   - Total islands: {len(islands)}")
        step4_lines.append(f"   - Total bridge connections: {len(bridges)}")
        step4_lines.append("   - All islands reachable from any other island ✓")
        step4_lines.append("   - No isolated groups or disconnected components ✓")
        
        step4_lines.append("")
        step4_lines.append("3. **Geometric Constraint Validation:**")
        step4_lines.append("   - All bridges run horizontally or vertically ✓")
        step4_lines.append("   - No bridges cross each other ✓")
        step4_lines.append("   - No bridges pass through islands ✓")
        step4_lines.append("   - Maximum 2 bridges between any pair of islands ✓")
        
        step4_lines.append("")
        step4_lines.append("4. **Mathematical Consistency Check:**")
        total_bridge_endpoints = sum(bridge['count'] * 2 for bridge in bridges)
        total_island_requirements = sum(island['num'] for island in islands)
        step4_lines.append(f"   - Total bridge endpoints created: {total_bridge_endpoints}")
        step4_lines.append(f"   - Total island requirements: {total_island_requirements}")
        math_consistent = (total_bridge_endpoints == total_island_requirements)
        step4_lines.append(f"   - Mathematical consistency: {'✓' if math_consistent else '✗'}")
        
        step4_lines.append("")
        step4_lines.append("**Reflection on Solution Quality:**")
        if all_satisfied and math_consistent:
            step4_lines.append("- ✅ **SOLUTION COMPLETE AND VALID**")
            step4_lines.append("- All puzzle constraints satisfied successfully")
            step4_lines.append("- Solution demonstrates logical progression from constraint analysis")
            step4_lines.append("- Bridge placement efficiently satisfies all island requirements")
            step4_lines.append("- Network connectivity achieved with minimal necessary connections")
        else:
            step4_lines.append("- ❌ **SOLUTION ISSUES DETECTED**")
            step4_lines.append("- Review and correction needed for constraint violations")
        
        step4_lines.append("")
        step4_lines.append("**Problem-Solving Approach Reflection:**")
        step4_lines.append("- Started with careful rule understanding and visual analysis")
        step4_lines.append("- Systematically analyzed all possible connections")
        step4_lines.append("- Prioritized high-constraint islands to reduce solution space")
        step4_lines.append("- Built solution incrementally with continuous validation")
        step4_lines.append("- Applied multiple verification layers for solution confidence")
        
        step4_text = "\n".join(step4_lines)
        
        # Combine all steps
        full_cot = step1_text + "\n\n" + step2_text + "\n\n" + step3_text + "\n\n" + step4_text
        
        # Create progressive steps
        step1_all = step1_text
        step2_all = step1_text + "\n\n" + step2_text
        step3_all = step1_text + "\n\n" + step2_text + "\n\n" + step3_text
        step4_all = full_cot
        
        # Create partial steps (half by word count, cut at appropriate places)
        def get_half_content(text):
            words = text.split()
            half_word_count = len(words) // 2
            
            # Find a good cutting point near the half-way mark
            half_text = ' '.join(words[:half_word_count])
            
            # Try to cut at the end of a sentence or paragraph
            for cut_point in ['. ', '\n\n', '\n']:
                last_cut = half_text.rfind(cut_point)
                if last_cut != -1:
                    return half_text[:last_cut + len(cut_point)].strip()
            
            # If no good cut point found, just cut at half word count
            return half_text
        
        step1_part = get_half_content(step1_all)
        step2_part = get_half_content(step2_all)
        step3_part = get_half_content(step3_all)
        step4_part = get_half_content(step4_all)
        
        return {
            "full_cot": full_cot,
            "cot_step1_part": step1_part,
            "cot_step1_all": step1_all,
            "cot_step2_part": step2_part,
            "cot_step2_all": step2_all,
            "cot_step3_part": step3_part,
            "cot_step3_all": step3_all,
            "cot_step4_part": step4_part,
            "cot_step4_all": step4_all
        }
    def _get_difficulty_params(self, difficulty):
        return self.difficulty_settings.get(difficulty, self.difficulty_settings[3])

    def generate(self, num_cases, difficulty, output_folder=None) -> List[Dict]:
        """
        Batch-generate puzzles for a given difficulty, seeding with runtime timestamp, and batch-write IO at the end.
        """
        output_dir = output_folder or self.output_folder
        settings = self._get_difficulty_params(difficulty)
        grid_size = settings["grid_size"]

        # Base seed from program runtime timestamp
        base_seed = self.run_seed

        # In-memory batching
        puzzles_data: List[Dict] = []
        image_jobs: List[Tuple[Dict, str, int, bool]] = []

        # Ensure output dirs exist only when writing at the end
        os.makedirs(output_dir, exist_ok=True)
        images_root = os.path.join(output_dir, 'images')
        os.makedirs(images_root, exist_ok=True)

        for idx in range(1, num_cases + 1):
            # Derive per-item seed deterministically from base seed
            seed_i = base_seed + idx

            puzzle = self.generate_seeded_puzzle(grid_size, seed_i, difficulty)

            puzzle_id = f"bridges_difficulty_{difficulty}_item_{idx}"
            img_rel_path = f"images/{puzzle_id}.png"
            img_abs_path = os.path.join(output_dir, img_rel_path)
            sol_rel_path = f"images/{puzzle_id}_solution.png"
            sol_abs_path = os.path.join(output_dir, sol_rel_path)

            # Queue images to write later
            image_jobs.append((puzzle, img_abs_path, grid_size, False))
            image_jobs.append((puzzle, sol_abs_path, grid_size, True))

            cot_data = self.generate_cot(puzzle, grid_size)
            answer = self.format_answer(puzzle)

            puzzle_data = {
                "index": puzzle_id,
                "category": "bridges",
                "image": img_rel_path,
                "question": self.format_question_with_image(),
                "question_language": self.format_question_text(puzzle),
                "answer": answer,
                "initial_state": json.dumps(puzzle),
                "difficulty": f"{difficulty}",
                "cot": cot_data["full_cot"],
                "cot_step1_part": cot_data["cot_step1_part"],
                "cot_step1_all": cot_data["cot_step1_all"],
                "cot_step2_part": cot_data.get("cot_step2_part", ""),
                "cot_step2_all": cot_data["cot_step2_all"],
                "cot_step3_part": cot_data.get("cot_step3_part", ""),
                "cot_step3_all": cot_data["cot_step3_all"],
                "cot_step4_part": cot_data.get("cot_step4_part", ""),
                "cot_step4_all": cot_data.get("cot_step4_all", "")
            }

            puzzles_data.append(puzzle_data)

        # Batch-write images
        for puzzle, path, gsize, show_solution in image_jobs:
            self.visualize(puzzle, save_path=path, grid_size=gsize, show_solution=show_solution)

        # Batch-write annotations once (merge existing, atomic write)
        self.save_annotations(puzzles_data, output_dir)

        return puzzles_data

    def save_annotations(self, annotations, output_folder):
        """
        保存标注到annotations.json文件中，使用智能合并逻辑

        Args:
            annotations: 标注列表
            output_folder: 输出文件夹路径
        """
        annotations_path = os.path.join(output_folder, 'annotations.json')

        # Load existing annotations if present
        existing_data: List[Dict] = []
        if os.path.exists(annotations_path):
            try:
                with open(annotations_path, 'r', encoding='utf-8') as rf:
                    existing_data = json.load(rf)
            except Exception:
                existing_data = []

        # Merge by index
        merged: Dict[str, Dict] = {}
        for item in existing_data:
            idx = item.get('index')
            if idx is not None:
                merged[idx] = item
        for item in annotations:
            idx = item.get('index')
            if idx is None:
                continue
            if idx in merged:
                # Preserve existing answer; if missing, take from new
                if not merged[idx].get('answer') and item.get('answer'):
                    merged[idx]['answer'] = item['answer']
                # Fill missing fields conservatively
                for k, v in item.items():
                    if k not in merged[idx] or merged[idx][k] in (None, ""):
                        merged[idx][k] = v
            else:
                merged[idx] = item

        merged_list = list(merged.values())
        try:
            merged_list.sort(key=lambda x: x.get('index', ''))
        except Exception:
            pass

        # Use atomic write with temporary file
        tmp_path = annotations_path + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as wf:
            json.dump(merged_list, wf, ensure_ascii=False, indent=2)
        os.replace(tmp_path, annotations_path)
    
    def visualize(self, puzzle: Dict, **kwargs) -> None:
        """
        Visualize a Bridges puzzle as an image with enhanced aesthetics.
        
        Args:
            puzzle (Dict): The puzzle to visualize
            show_solution (bool): Whether to show bridges in the visualization
            save_path (str): Path to save the image file
            grid_size (int): Size of the grid
        """
        import matplotlib.patheffects as PathEffects
        
        show_solution = kwargs.get('show_solution', False)
        save_path = kwargs.get('save_path', None)
        grid_size = kwargs.get('grid_size', 5)
        
        # Create figure with improved size ratio
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Set a pleasant background color
        ax.set_facecolor('#F0F8FF')  # AliceBlue
        
        # Draw grid with more padding for better visualization
        padding = 0.7
        ax.set_xlim(-padding, grid_size - 1 + padding)
        ax.set_ylim(grid_size - 1 + padding, -padding)  # Flip Y-axis to have 0 at the top
        
        # Subtle grid lines
        ax.grid(True, linestyle='--', alpha=0.5, color='#B0C4DE')  # LightSteelBlue
        
        # Add clear grid intersection points to reduce ambiguity
        for x in range(grid_size):
            for y in range(grid_size):
                # Check if there's an island at this position
                has_island = any(island['x'] == x and island['y'] == y for island in puzzle['islands'])
                if not has_island:
                    # Draw small dot at grid intersection to show valid placement points
                    ax.plot(x, y, 'o', markersize=3, color='#D3D3D3', alpha=0.8, zorder=1)
        
        # Draw bridges if show_solution is True
        if show_solution:
            for bridge in puzzle['bridges']:
                x1, y1 = bridge['from']
                x2, y2 = bridge['to']
                
                # Improved bridge styling
                if bridge['count'] == 1:
                    linewidth = 3
                    color = '#4682B4'  # SteelBlue
                else:  # count == 2
                    linewidth = 6
                    color = '#4169E1'  # RoyalBlue
                
                # Add shadow for 3D effect
                shadow = ax.plot([x1, x2], [y1, y2], '-', 
                               linewidth=linewidth+2, color='#778899', alpha=0.4, zorder=1)[0]
                
                # Draw the bridge
                line = ax.plot([x1, x2], [y1, y2], '-', 
                       linewidth=linewidth, color=color, solid_capstyle='round', zorder=2)[0]
        
        # Draw islands with improved aesthetics
        for island in puzzle['islands']:
            # Gradient effect for islands
            # Outer glow (shadow)
            shadow = plt.Circle((island['x'], island['y']), 0.3, 
                               fill=True, color='#2C3E50', alpha=0.5, zorder=3)
            ax.add_patch(shadow)
            
            # Main circle with gradient-like effect
            circle_outer = plt.Circle((island['x'], island['y']), 0.28, 
                                    fill=True, color='#3498DB', zorder=4)  # Darker blue
            ax.add_patch(circle_outer)
            
            circle_inner = plt.Circle((island['x'], island['y']), 0.22, 
                                    fill=True, color='#5DADE2', zorder=5)  # Lighter blue
            ax.add_patch(circle_inner)
            
            # Add number with improved text styling
            ax.text(island['x'], island['y'], str(island['num']), 
                   fontsize=16, ha='center', va='center', color='white', 
                   fontweight='bold', zorder=6, 
                   path_effects=[PathEffects.withStroke(linewidth=2, foreground='#1A5276')])
        
        # Set aspect and remove ticks
        ax.set_aspect('equal')
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        
        # Add grid coordinates for clarity
        ax.set_xticklabels(range(grid_size))
        ax.set_yticklabels(range(grid_size))
        ax.tick_params(axis='both', which='both', length=0, labelsize=14, colors='#555555')
        
        # # Add a note about grid points and coordinate system
        # ax.text(grid_size/2, -0.3, "Islands can only be placed at grid intersections (marked with dots)",
        #         ha='center', va='center', fontsize=9, color='#555555', style='italic')
        
        # # Add coordinate system explanation
        # ax.text(0, -0.5, "(0,0)", fontsize=8, color='#1A5276', ha='center', va='center')
        # ax.text(grid_size-1, -0.5, f"({grid_size-1},0)", fontsize=8, color='#1A5276', ha='center', va='center')
        # ax.text(0, grid_size-0.5, f"(0,{grid_size-1})", fontsize=8, color='#1A5276', ha='center', va='center')
        # ax.text(grid_size-1, grid_size-0.5, f"({grid_size-1},{grid_size-1})", fontsize=8, color='#1A5276', ha='center', va='center')
        
        # # Add arrows to show coordinate directions
        # ax.arrow(0.2, 0.2, 1, 0, head_width=0.15, head_length=0.15, fc='#1A5276', ec='#1A5276', zorder=7)
        # ax.text(0.7, 0, "X", fontsize=8, color='#1A5276', ha='center', va='center')
        # ax.arrow(0.2, 0.2, 0, 1, head_width=0.15, head_length=0.15, fc='#1A5276', ec='#1A5276', zorder=7)
        # ax.text(0, 0.7, "Y", fontsize=8, color='#1A5276', ha='center', va='center')
        
        # Add decorative border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#B0C4DE')
            spine.set_linewidth(2)
        
        # Add title with improved styling
        title_text = "Bridges Puzzle Solution" if show_solution else "Bridges Puzzle"
        ax.set_title(title_text, fontsize=18, fontweight='bold', pad=20, 
                    color='#2E4053', fontfamily='sans-serif')
        
        # Add puzzle size info
        ax.text(grid_size/2, grid_size-0.5, f"{grid_size}×{grid_size} Grid",
                ha='center', va='center', fontsize=10, color='#34495E', style='italic')
        
        plt.tight_layout()
        
        # Save image with improved quality if path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=200, facecolor='white')
            plt.close(fig)
        else:
            plt.show()

    def solve(self, puzzle: Dict, **kwargs) -> str:
        """
        Generate a step-by-step solution for the puzzle.
        
        Args:
            puzzle (Dict): The puzzle to solve
            
        Returns:
            str: A step-by-step solution explanation
        """
        islands = puzzle['islands']
        bridges = puzzle['bridges']
        
        solution_steps = []
        
        # Sort bridges to create a logical solving order
        sorted_bridges = sorted(bridges, key=lambda b: (b['from'][0], b['from'][1]))
        
        for i, bridge in enumerate(sorted_bridges):
            from_x, from_y = bridge['from']
            to_x, to_y = bridge['to']
            count = bridge['count']
            
            # Find the corresponding islands
            from_island = next(island for island in islands 
                              if island['x'] == from_x and island['y'] == from_y)
            to_island = next(island for island in islands 
                            if island['x'] == to_x and island['y'] == to_y)
            
            # Create reasoning step
            direction = "horizontally" if from_y == to_y else "vertically"
            
            step = (
                f"Step {i+1}: Connect island at ({from_x},{from_y}) with number {from_island['num']} "
                f"to island at ({to_x},{to_y}) with number {to_island['num']} "
                f"{direction} using {count} bridge(s)."
            )
            
            # Add additional reasoning
            if count == 2:
                step += " Using double bridges maximizes connections while satisfying the island numbers."
            
            if i > 0:
                step += " This continues to build our connected network."
                
            solution_steps.append(step)
        
        # Add verification step
        solution_steps.append(
            f"Verification: All {len(islands)} islands are now connected by exactly "
            f"the number of bridges indicated on each island, forming a single continuous network."
        )
        
        return "\n".join(solution_steps)
    
    def format_answer(self, puzzle: Dict) -> str:
        """
        Format the solution as a string.
        
        Args:
            puzzle (Dict): The puzzle with solution
            
        Returns:
            str: The formatted answer string
        """
        answer_lines = []
        for bridge in sorted(puzzle['bridges'], key=lambda b: (b['from'][0], b['from'][1])):
            x1, y1 = bridge['from']
            x2, y2 = bridge['to']
            count = bridge['count']
            answer_lines.append(f"({x1},{y1})-({x2},{y2}):{count}")
        
        return "\n".join(answer_lines)
    
    def format_question_with_image(self) -> str:
        """
        Format the question with image reference.
        
        Returns:
            str: The formatted question with image reference
        """
        return """# Bridges Puzzle

Please look carefully at the image showing a Bridges puzzle (Hashiwokakero). In this puzzle, you need to connect all numbered "islands" using horizontal/vertical bridges.

### Game Rules:
1. Each island displays a number indicating how many bridges must connect to it
2. Bridges can only run horizontally or vertically between islands
3. Bridges cannot cross other bridges or islands
4. At most 2 bridges can connect any pair of islands
5. All islands must form a single connected network

### Coordinate system:
- The grid uses (x,y) coordinates starting from (0,0) in the top-left corner
- X increases from left to right, Y increases from top to bottom

### Answer format:
Provide your solution with each bridge connection in the format:
(x1,y1)-(x2,y2):count

For example:
(0,4)-(2,4):1
(2,1)-(2,4):1
(2,4)-(4,4):1
"""
    
    def format_question_text(self, puzzle: Dict) -> str:
        """
        Format the question as text only with complete grid information.
        
        Args:
            puzzle (Dict): The puzzle to format
            
        Returns:
            str: The text-only question format with grid details
        """
        # Determine grid size from the puzzle
        max_x = max(island['x'] for island in puzzle['islands'])
        max_y = max(island['y'] for island in puzzle['islands'])
        grid_size = max(max_x, max_y) + 1
        
        # Create islands description
        islands_desc = []
        for island in sorted(puzzle['islands'], key=lambda i: (i['y'], i['x'])):
            islands_desc.append(f"Island at coordinates ({island['x']}, {island['y']}) has number {island['num']}.")
        
        islands_text = '\n'.join(islands_desc)
        
        # Create a text-based grid representation
        grid_repr = []
        grid_repr.append("**Grid Layout:**")
        grid_repr.append("```")
        
        # Create column headers
        col_header = "   " + " ".join(f"{i:2d}" for i in range(grid_size))
        grid_repr.append(col_header)
        
        # Create each row
        for y in range(grid_size):
            row_str = f"{y:2d} "
            for x in range(grid_size):
                # Check if there's an island at this position
                island_at_pos = next((island for island in puzzle['islands'] 
                                    if island['x'] == x and island['y'] == y), None)
                if island_at_pos:
                    row_str += f" {island_at_pos['num']}"
                else:
                    row_str += "  ."
            grid_repr.append(row_str)
        
        grid_repr.append("```")
        grid_repr.append("(Numbers represent islands with their required bridge count, '.' represents empty cells)")
        
        grid_layout_text = '\n'.join(grid_repr)
        
        return f"""# Bridges Puzzle

Please refer to the initial_state provided below for this Bridges puzzle (Hashiwokakero). In this puzzle, you need to connect all numbered "islands" using horizontal/vertical bridges.

**Grid Information:**
- Grid size: {grid_size}x{grid_size}
- Total grid cells: {grid_size * grid_size}
- Number of islands: {len(puzzle['islands'])}

**Rules to follow:**
1. Each island displays a number indicating how many bridges must connect to it
2. Bridges can only run horizontally or vertically between islands
3. Bridges cannot cross other bridges or islands
4. At most 2 bridges can connect any pair of islands
5. All islands must form a single connected network

**Coordinate system:**
- The grid uses (x,y) coordinates starting from (0,0) in the top-left corner
- X increases from left to right (columns: 0 to {grid_size-1})
- Y increases from top to bottom (rows: 0 to {grid_size-1})

**Islands distribution:**
{islands_text}

**Answer format:**
Provide your solution with each bridge connection in the format:
(x1,y1)-(x2,y2):count

For example:
(0,4)-(2,4):1
(2,1)-(2,4):1
(2,4)-(4,4):1

Please solve the puzzle and provide your solution."""

# Example usage for testing
if __name__ == "__main__":
    # Initialize with an output folder
    generator = BridgesGenerator(output_folder="test_output/bridges")

    # Test the new batch interface: generate 5 cases for difficulty 3
    print("Testing batch generation with timestamp seeding...")
    results = generator.generate(num_cases=5, difficulty=3)
    print(f"Generated {len(results)} puzzles. Annotations at: test_output/bridges/annotations.json")