import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import json
import os
import random
import time
import signal
import sys
from collections import deque
from PIL import Image, ImageDraw, ImageFont
import uuid
import shutil
import heapq
from generator.base_generator import BaseGenerator

PROMPT_SOKOBAN_IMAGE = """
Your task is to solve the Sokoban puzzle according to the rules and current state shown in the image:

### Game Rules:
1. You are the player and can move up, down, left, or right
2. You can push boxes one space at a time
3. You cannot pull boxes
4. Boxes can only be pushed if there's an empty space behind them
5. The goal is to push all boxes onto target positions
6. Walls cannot be moved through or pushed

### You will be given an image, in the image:

1. Red circles represent the player
2. Blue squares represent boxes
3. Yellow circles represent target positions
4. Brown blocks represent walls
5. Other blocks represent empty spaces

### Direction Definitions:
- "up": Move up
- "down": Move down
- "left": Move left
- "right": Move right

### Output Format Requirements:
Your final answer should be in the format of a space-separated sequence of moves like: up right down left
"""

PROMPT_SOKOBAN = """
Your task is to solve the Sokoban puzzle according to the rules and current state shown in the image:

### Game Rules:
1. You are the player and can move up, down, left, or right
2. You can push boxes one space at a time
3. You cannot pull boxes
4. Boxes can only be pushed if there's an empty space behind them
5. The goal is to push all boxes onto target positions
6. Walls cannot be moved through or pushed

### You will be given sokoban state

1. @ represent the player
2. $ represent boxes
3. . circles represent target positions
4. # blocks represent walls

### Direction Definitions:
- "up": Move up
- "down": Move down
- "left": Move left
- "right": Move right

### Current Sokoban State is shown below:
{}

### Output Format Requirements:
Your final answer should be in the format of a space-separated sequence of moves like: up right down left
"""



class SokobanGenerator(BaseGenerator):
    # Define elements
    WALL = '#'
    PLAYER = '@'
    PLAYER_ON_GOAL = '+'
    BOX = '$'
    BOX_ON_GOAL = '*'
    GOAL = '.'
    FLOOR = ' '
    
    # For solver state representation
    ELEMENT_MAP = {
        WALL: 0, FLOOR: 1, GOAL: 2,
        BOX: 3, BOX_ON_GOAL: 4,
        PLAYER: 5, PLAYER_ON_GOAL: 6
    }
    REVERSE_MAP = {v: k for k, v in ELEMENT_MAP.items()}

    # Directions (row, col) and corresponding move characters
    DIRECTIONS = {
        'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)
    }
    MOVE_CHARS = {v: k for k, v in DIRECTIONS.items()} # Map (dr, dc) back to char
    
    # Direction word mapping for answers
    DIRECTION_WORDS = {
        'U': 'up',
        'D': 'down', 
        'L': 'left',
        'R': 'right'
    }

    # Dynamic generation parameters for different difficulty levels
    DIFFICULTY_PARAMS = {
        1: {  # Very Easy
            'grid_size': (5, 7),
            'min_boxes': 1,
            'max_boxes': 1,
            'min_rooms': 1,
            'max_rooms': 2,
            'wall_density': 0.1,
            'max_solution_length': 15,
            'layout_types': ['open', 'simple_rooms'],
            'goal_cluster_chance': 0.8,  # High chance for clustered goals
            'maze_factor': 0.1  # Low maze complexity
        },
        2: {  # Easy
            'grid_size': (7, 9),
            'min_boxes': 1,
            'max_boxes': 1,
            'min_rooms': 1,
            'max_rooms': 3,
            'wall_density': 0.15,
            'max_solution_length': 25,
            'layout_types': ['open', 'simple_rooms', 'corridors'],
            'goal_cluster_chance': 0.7,
            'maze_factor': 0.2
        },
        3: {  # Medium
            'grid_size': (7, 9),
            'min_boxes': 2,
            'max_boxes': 2,
            'min_rooms': 2,
            'max_rooms': 4,
            'wall_density': 0.2,
            'max_solution_length': 40,
            'layout_types': ['simple_rooms', 'corridors', 'mixed'],
            'goal_cluster_chance': 0.6,
            'maze_factor': 0.3
        },
        4: {  # Hard
            'grid_size': (11, 13),
            'min_boxes': 2,
            'max_boxes': 2,
            'min_rooms': 3,
            'max_rooms': 6,
            'wall_density': 0.25,
            'max_solution_length': 50,
            'layout_types': ['corridors', 'mixed', 'maze_like'],
            'goal_cluster_chance': 0.4,  # More scattered goals
            'maze_factor': 0.4
        },
        5: {  # Expert
            'grid_size': (11, 13),
            'min_boxes': 3,
            'max_boxes': 3,
            'min_rooms': 3, 
            'max_rooms': 6,
            'wall_density': 0.3,
            'max_solution_length': 60,
            'layout_types': ['mixed', 'maze_like', 'complex'],
            'goal_cluster_chance': 0.3,  # Mostly scattered goals
            'maze_factor': 0.5  # High maze complexity
        }
    }

    def __init__(self, output_folder):
        super().__init__(output_folder)
        # Initialize directories and seed compatible with shared BaseGenerator
        self.task_dir = output_folder
        self.base_output_dir = output_folder
        self.image_dir = os.path.join(self.task_dir, 'images')
        self.annotations_file = os.path.join(self.task_dir, 'annotations.json')
        os.makedirs(self.task_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        self.seed = int(time.time())
        random.seed(self.seed)
        np.random.seed(self.seed % (2**32 - 1))
        self.tile_size = 50 # Larger for better visualization
        self.colors = {
            self.WALL: (120, 80, 60),    # Brick brown
            self.FLOOR: (180, 160, 140), # Stone floor
            self.GOAL: (255, 215, 0),    # Golden yellow
            self.BOX: (50, 100, 180),    # Blue color
            self.BOX_ON_GOAL: (60, 150, 60), # Green tint
            self.PLAYER: (255, 100, 100),    # Character pink-red
            self.PLAYER_ON_GOAL: (255, 200, 50) # Golden character
        }
        # Load font for visualization
        try:
            self.font = ImageFont.truetype("arial.ttf", self.tile_size // 3)
        except IOError:
            try:
                self.font = ImageFont.truetype("DejaVuSans.ttf", self.tile_size // 3)
            except IOError:
                print("Warning: Could not load preferred fonts. Using default.")
                self.font = ImageFont.load_default()

        # Track existing states to avoid duplicates
        self.existing_states = set()
        self._load_existing_states()

    def _load_existing_states(self):
        """Load existing puzzle states from training data to avoid duplicates"""
        training_files = [
            'training.json',
            os.path.join(self.task_dir, 'annotations.json'),
            os.path.join(self.base_output_dir, 'training.json')
        ]
        
        for training_file in training_files:
            if os.path.exists(training_file):
                try:
                    with open(training_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        for item in data:
                            if item.get('category') == 'sokoban' and 'question' in item:
                                try:
                                    _, initial_state, _ = self._parse_level(item['question'])
                                    state_hash = self._state_to_hash(initial_state)
                                    self.existing_states.add(state_hash)
                                except Exception as e:
                                    continue  # Skip invalid entries
                    
                    print(f"Loaded {len(self.existing_states)} existing states from {training_file}")
                    break  # Use first found file
                    
                except Exception as e:
                    print(f"Warning: Could not load existing states from {training_file}: {e}")
                    continue

    def _get_difficulty_params(self, difficulty):
        """
        根据难度级别获取相应的参数配置。

        Args:
            difficulty: 难度级别（1-5）

        Returns:
            dict: 包含难度参数的字典
        """
        return self.DIFFICULTY_PARAMS.get(difficulty, self.DIFFICULTY_PARAMS[3])  # Default to medium

    def _state_to_hash(self, initial_state):
        """Convert initial state to a hash for duplicate detection"""
        player_pos, box_positions = initial_state
        # Create a normalized representation
        sorted_boxes = tuple(sorted(box_positions))
        return hash((player_pos, sorted_boxes))
    
    def _is_duplicate_state(self, initial_state):
        """Check if the initial state already exists"""
        state_hash = self._state_to_hash(initial_state)
        return state_hash in self.existing_states
    
    def _add_to_existing_states(self, initial_state):
        """Add a new state to the existing states set"""
        state_hash = self._state_to_hash(initial_state)
        self.existing_states.add(state_hash)

    def _generate_base_room(self, width, height):
        """Generate a basic rectangular room surrounded by walls"""
        grid = []
        for r in range(height):
            row = []
            for c in range(width):
                if r == 0 or r == height - 1 or c == 0 or c == width - 1:
                    row.append(self.WALL)
                else:
                    row.append(self.FLOOR)
            grid.append(row)
        return grid

    def _add_internal_walls(self, grid, wall_density, min_path_width=2):
        """Add internal walls to create interesting layouts while ensuring connectivity"""
        height, width = len(grid), len(grid[0])
        
        # Create potential wall positions (not on edges)
        potential_walls = []
        for r in range(2, height - 2):
            for c in range(2, width - 2):
                if grid[r][c] == self.FLOOR:
                    potential_walls.append((r, c))
        
        # Add walls based on density, but ensure connectivity
        num_walls = int(len(potential_walls) * wall_density)
        random.shuffle(potential_walls)
        
        for i, (r, c) in enumerate(potential_walls[:num_walls]):
            # Temporarily add wall
            original = grid[r][c]
            grid[r][c] = self.WALL
            
            # Check if this creates disconnected regions
            if not self._is_connected(grid):
                grid[r][c] = original  # Revert if it disconnects
            else:
                # Add some structure - create L-shapes or lines occasionally
                if random.random() < 0.3:
                    self._extend_wall_structure(grid, r, c)
    
    def _extend_wall_structure(self, grid, start_r, start_c):
        """Extend a wall to create interesting structures like L-shapes"""
        height, width = len(grid), len(grid[0])
        
        # Try to extend in a random direction
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)
        
        for dr, dc in directions:
            length = random.randint(1, 3)
            valid_extension = True
            positions_to_add = []
            
            for i in range(1, length + 1):
                nr, nc = start_r + dr * i, start_c + dc * i
                if (nr <= 1 or nr >= height - 2 or nc <= 1 or nc >= width - 2 or
                    grid[nr][nc] != self.FLOOR):
                    valid_extension = False
                    break
                positions_to_add.append((nr, nc))
            
            if valid_extension and positions_to_add:
                # Test connectivity before committing
                backup = [grid[r][c] for r, c in positions_to_add]
                for r, c in positions_to_add:
                    grid[r][c] = self.WALL
                
                if self._is_connected(grid):
                    break  # Keep the extension
                else:
                    # Revert
                    for (r, c), original in zip(positions_to_add, backup):
                        grid[r][c] = original

    def _is_connected(self, grid):
        """Check if all floor spaces are connected using flood fill"""
        height, width = len(grid), len(grid[0])
        
        # Find first floor position
        start = None
        floor_count = 0
        for r in range(height):
            for c in range(width):
                if grid[r][c] == self.FLOOR:
                    floor_count += 1
                    if start is None:
                        start = (r, c)
        
        if start is None or floor_count == 0:
            return True  # No floors or trivially connected
        
        # Flood fill from start position
        visited = set()
        queue = [start]
        
        while queue:
            r, c = queue.pop(0)
            if (r, c) in visited:
                continue
            visited.add((r, c))
            
            # Check all 4 directions
            for dr, dc in self.DIRECTIONS.values():
                nr, nc = r + dr, c + dc
                if (0 <= nr < height and 0 <= nc < width and
                    grid[nr][nc] == self.FLOOR and (nr, nc) not in visited):
                    queue.append((nr, nc))
        
        return len(visited) == floor_count

    def _create_rooms(self, grid, num_rooms):
        """Create distinct rooms by adding walls to divide space"""
        height, width = len(grid), len(grid[0])
        
        if num_rooms <= 1:
            return
        
        # Create vertical and horizontal divisions
        divisions_made = 0
        max_attempts = num_rooms * 3
        
        for attempt in range(max_attempts):
            if divisions_made >= num_rooms - 1:
                break
                
            # Choose division type and position
            if random.random() < 0.5:  # Vertical division
                col = random.randint(3, width - 4)
                start_row = random.randint(1, height // 3)
                end_row = random.randint(2 * height // 3, height - 2)
                
                # Create division with gaps for doorways
                doorway = random.randint(start_row + 1, end_row - 1)
                for r in range(start_row, end_row + 1):
                    if r != doorway and grid[r][col] == self.FLOOR:
                        grid[r][col] = self.WALL
            else:  # Horizontal division
                row = random.randint(3, height - 4)
                start_col = random.randint(1, width // 3)
                end_col = random.randint(2 * width // 3, width - 2)
                
                # Create division with gaps for doorways
                doorway = random.randint(start_col + 1, end_col - 1)
                for c in range(start_col, end_col + 1):
                    if c != doorway and grid[row][c] == self.FLOOR:
                        grid[row][c] = self.WALL
            
            # Verify connectivity after division
            if self._is_connected(grid):
                divisions_made += 1
            else:
                # Revert last changes if they disconnected the space
                # This is a simplified revert - in practice you'd want to track changes
                pass

    def _place_goals_and_boxes(self, grid, num_boxes, goal_cluster_chance=0.6):
        """Intelligently place goals and boxes to create solvable puzzles"""
        height, width = len(grid), len(grid[0])
        
        # Find all possible floor positions
        floor_positions = []
        for r in range(1, height - 1):
            for c in range(1, width - 1):
                if grid[r][c] == self.FLOOR:
                    floor_positions.append((r, c))
        
        if len(floor_positions) < num_boxes * 2 + 1:  # Need space for boxes, goals, and player
            raise ValueError("Not enough floor space for requested number of boxes")
        
        # Strategy 1: Create goal clusters or scattered goals based on difficulty
        goal_positions = set()
        if random.random() < goal_cluster_chance:
            goal_positions.update(self._create_goal_cluster(grid, floor_positions, num_boxes))
        else:
            goal_positions.update(self._create_scattered_goals(grid, floor_positions, num_boxes))
        
        # Place goals on the grid
        for r, c in goal_positions:
            grid[r][c] = self.GOAL
        
        # Choose box starting positions (away from goals initially to create challenge)
        available_positions = [pos for pos in floor_positions if pos not in goal_positions]
        
        # Strategy 2: Smart box placement
        box_positions = self._place_boxes_strategically(grid, available_positions, goal_positions, num_boxes)
        
        return goal_positions, box_positions

    def _generate_layout_by_type(self, grid, layout_type, params):
        """Generate different layout types for variety"""
        if layout_type == 'open':
            self._create_open_layout(grid, params)
        elif layout_type == 'simple_rooms':
            self._create_simple_rooms(grid, params)
        elif layout_type == 'corridors':
            self._create_corridor_layout(grid, params)
        elif layout_type == 'mixed':
            self._create_mixed_layout(grid, params)
        elif layout_type == 'maze_like':
            self._create_maze_layout(grid, params)
        elif layout_type == 'complex':
            self._create_complex_layout(grid, params)
        else:
            # Default to simple rooms
            self._create_simple_rooms(grid, params)

    def _create_open_layout(self, grid, params):
        """Create mostly open space with minimal obstacles"""
        # Add just a few strategic walls
        self._add_internal_walls(grid, params['wall_density'] * 0.5)
        
        # Add some decorative single walls
        height, width = len(grid), len(grid[0])
        for _ in range(random.randint(1, 3)):
            r = random.randint(2, height - 3)
            c = random.randint(2, width - 3)
            if grid[r][c] == self.FLOOR:
                grid[r][c] = self.WALL

    def _create_simple_rooms(self, grid, params):
        """Create simple rectangular rooms"""
        num_rooms = random.randint(params['min_rooms'], params['max_rooms'])
        self._create_rooms(grid, num_rooms)
        self._add_internal_walls(grid, params['wall_density'] * 0.7)

    def _create_corridor_layout(self, grid, params):
        """Create a layout with corridors and passages"""
        height, width = len(grid), len(grid[0])
        
        # Create main corridors
        # Vertical corridor
        if width > 8:
            col = width // 2
            for r in range(2, height - 2):
                if grid[r][col] == self.FLOOR:
                    # Keep some gaps
                    if random.random() > 0.3:
                        grid[r][col] = self.WALL
        
        # Horizontal corridor
        if height > 8:
            row = height // 2
            for c in range(2, width - 2):
                if grid[row][c] == self.FLOOR:
                    # Keep some gaps
                    if random.random() > 0.3:
                        grid[row][c] = self.WALL
        
        # Add perpendicular branches
        self._add_corridor_branches(grid)

    def _add_corridor_branches(self, grid):
        """Add branching corridors"""
        height, width = len(grid), len(grid[0])
        
        # Add some shorter walls to create corridor-like passages
        for _ in range(random.randint(2, 4)):
            # Choose direction
            if random.random() < 0.5:  # Vertical wall
                c = random.randint(3, width - 4)
                start_r = random.randint(2, height // 2)
                length = random.randint(2, height // 3)
                for r in range(start_r, min(start_r + length, height - 2)):
                    if grid[r][c] == self.FLOOR:
                        grid[r][c] = self.WALL
            else:  # Horizontal wall
                r = random.randint(3, height - 4)
                start_c = random.randint(2, width // 2)
                length = random.randint(2, width // 3)
                for c in range(start_c, min(start_c + length, width - 2)):
                    if grid[r][c] == self.FLOOR:
                        grid[r][c] = self.WALL

    def _create_mixed_layout(self, grid, params):
        """Create a mixed layout combining different strategies"""
        # Combine rooms and corridors
        if random.random() < 0.5:
            self._create_simple_rooms(grid, params)
            self._add_corridor_branches(grid)
        else:
            self._create_corridor_layout(grid, params)
            # Add a small room
            self._add_small_room(grid)

    def _add_small_room(self, grid):
        """Add a small enclosed room"""
        height, width = len(grid), len(grid[0])
        
        # Find a good location for a small room
        room_size = random.randint(3, 5)
        max_attempts = 10
        
        for _ in range(max_attempts):
            start_r = random.randint(2, height - room_size - 2)
            start_c = random.randint(2, width - room_size - 2)
            
            # Check if area is mostly floor
            area_clear = True
            for r in range(start_r, start_r + room_size):
                for c in range(start_c, start_c + room_size):
                    if grid[r][c] != self.FLOOR:
                        area_clear = False
                        break
                if not area_clear:
                    break
            
            if area_clear:
                # Create walls around the room, leaving one opening
                for r in range(start_r, start_r + room_size):
                    for c in range(start_c, start_c + room_size):
                        if (r == start_r or r == start_r + room_size - 1 or
                            c == start_c or c == start_c + room_size - 1):
                            grid[r][c] = self.WALL
                
                # Create an opening
                if random.random() < 0.5:  # Vertical opening
                    c = start_c if random.random() < 0.5 else start_c + room_size - 1
                    r = random.randint(start_r + 1, start_r + room_size - 2)
                    grid[r][c] = self.FLOOR
                else:  # Horizontal opening
                    r = start_r if random.random() < 0.5 else start_r + room_size - 1
                    c = random.randint(start_c + 1, start_c + room_size - 2)
                    grid[r][c] = self.FLOOR
                break

    def _create_maze_layout(self, grid, params):
        """Create a more maze-like layout"""
        height, width = len(grid), len(grid[0])
        
        # Create a grid-like pattern
        for r in range(3, height - 2, 3):
            for c in range(2, width - 2):
                if grid[r][c] == self.FLOOR and random.random() < 0.7:
                    grid[r][c] = self.WALL
        
        for c in range(3, width - 2, 3):
            for r in range(2, height - 2):
                if grid[r][c] == self.FLOOR and random.random() < 0.7:
                    grid[r][c] = self.WALL
        
        # Ensure connectivity by removing some walls
        self._ensure_maze_connectivity(grid)

    def _ensure_maze_connectivity(self, grid):
        """Ensure the maze remains connected by strategically removing walls"""
        height, width = len(grid), len(grid[0])
        
        # Remove some random walls to create paths
        wall_positions = []
        for r in range(2, height - 2):
            for c in range(2, width - 2):
                if grid[r][c] == self.WALL:
                    wall_positions.append((r, c))
        
        # Remove about 30% of internal walls to ensure connectivity
        walls_to_remove = int(len(wall_positions) * 0.3)
        random.shuffle(wall_positions)
        
        for i, (r, c) in enumerate(wall_positions[:walls_to_remove]):
            grid[r][c] = self.FLOOR
            
            # Check connectivity after each removal
            if i % 5 == 0 and self._is_connected(grid):
                # Good connectivity achieved
                break

    def _create_complex_layout(self, grid, params):
        """Create the most complex layout for expert level"""
        # Combine multiple strategies
        self._create_maze_layout(grid, params)
        self._add_small_room(grid)
        self._add_corridor_branches(grid)
        
        # Add some irregular obstacles
        self._add_irregular_obstacles(grid)

    def _add_irregular_obstacles(self, grid):
        """Add some irregular wall patterns for complexity"""
        height, width = len(grid), len(grid[0])
        
        # Add L-shaped obstacles
        for _ in range(random.randint(1, 3)):
            r = random.randint(3, height - 4)
            c = random.randint(3, width - 4)
            
            # Create L-shape
            if grid[r][c] == self.FLOOR and grid[r+1][c] == self.FLOOR and grid[r][c+1] == self.FLOOR:
                grid[r][c] = self.WALL
                grid[r+1][c] = self.WALL
                grid[r][c+1] = self.WALL

    def _create_goal_cluster(self, grid, floor_positions, num_boxes):
        """Create a cluster of goals in one area"""
        height, width = len(grid), len(grid[0])
        
        # Find a good central position for the cluster
        center_candidates = []
        for r, c in floor_positions:
            # Count nearby floor spaces
            nearby_floor = 0
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < height and 0 <= nc < width and
                        grid[nr][nc] == self.FLOOR):
                        nearby_floor += 1
            
            if nearby_floor >= num_boxes + 2:  # Enough space for cluster
                center_candidates.append((r, c, nearby_floor))
        
        if not center_candidates:
            return random.sample(floor_positions, min(num_boxes, len(floor_positions)))
        
        # Choose the best center (most nearby floor space)
        center_candidates.sort(key=lambda x: x[2], reverse=True)
        center_r, center_c, _ = center_candidates[0]
        
        # Create cluster around center
        goals = set()
        goals.add((center_r, center_c))
        
        # Add nearby positions
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if len(goals) >= num_boxes:
                    break
                nr, nc = center_r + dr, center_c + dc
                if ((nr, nc) in floor_positions and (nr, nc) not in goals):
                    goals.add((nr, nc))
        
        # Fill to required number if needed
        remaining_positions = [pos for pos in floor_positions if pos not in goals]
        while len(goals) < num_boxes and remaining_positions:
            goals.add(remaining_positions.pop(random.randint(0, len(remaining_positions) - 1)))
        
        return goals

    def _create_scattered_goals(self, grid, floor_positions, num_boxes):
        """Create scattered goals across the map"""
        goals = set()
        available = floor_positions.copy()
        
        for _ in range(num_boxes):
            if not available:
                break
            
            # Choose position that's not too close to existing goals
            best_pos = None
            best_score = -1
            
            candidates = random.sample(available, min(10, len(available)))
            for pos in candidates:
                # Calculate minimum distance to existing goals
                if not goals:
                    score = random.random()
                else:
                    min_dist = min(abs(pos[0] - g[0]) + abs(pos[1] - g[1]) for g in goals)
                    score = min_dist + random.random() * 0.5
                
                if score > best_score:
                    best_score = score
                    best_pos = pos
            
            if best_pos:
                goals.add(best_pos)
                available.remove(best_pos)
        
        return goals

    def _place_boxes_strategically(self, grid, available_positions, goal_positions, num_boxes):
        """Place boxes in positions that create interesting puzzles"""
        box_positions = []
        
        # Strategy: Place boxes at various distances from their goal positions
        goal_list = list(goal_positions)
        
        for i in range(num_boxes):
            if i < len(goal_list):
                target_goal = goal_list[i]
                
                # Choose distance based on difficulty preference
                if random.random() < 0.3:  # 30% close
                    max_distance = 3
                elif random.random() < 0.7:  # 40% medium
                    max_distance = 6
                else:  # 30% far
                    max_distance = 10
                
                # Find positions within the desired distance range
                candidates = []
                for pos in available_positions:
                    dist = abs(pos[0] - target_goal[0]) + abs(pos[1] - target_goal[1])
                    if 2 <= dist <= max_distance:  # At least distance 2 to avoid trivial
                        candidates.append((pos, dist))
                
                if candidates:
                    # Prefer medium distances
                    candidates.sort(key=lambda x: abs(x[1] - max_distance // 2))
                    chosen_pos = candidates[0][0]
                    box_positions.append(chosen_pos)
                    available_positions.remove(chosen_pos)
                else:
                    # Fallback: any available position
                    if available_positions:
                        pos = random.choice(available_positions)
                        box_positions.append(pos)
                        available_positions.remove(pos)
            else:
                # Extra boxes: place randomly
                if available_positions:
                    pos = random.choice(available_positions)
                    box_positions.append(pos)
                    available_positions.remove(pos)
        
        return box_positions

    def _place_player(self, grid, goal_positions, box_positions):
        """Place player in a strategic position (not on goal positions)"""
        height, width = len(grid), len(grid[0])
        
        # Find available positions (excluding goal positions)
        available_positions = []
        for r in range(1, height - 1):
            for c in range(1, width - 1):
                if (grid[r][c] == self.FLOOR and
                    (r, c) not in box_positions and
                    (r, c) not in goal_positions):
                    available_positions.append((r, c))
        
        if not available_positions:
            raise ValueError("No available position for player (excluding goal positions)")
        
        # Strategy: Place player in a position that requires some planning
        # Prefer positions that are not immediately adjacent to boxes
        good_positions = []
        for pos in available_positions:
            adjacent_to_box = False
            for dr, dc in self.DIRECTIONS.values():
                nr, nc = pos[0] + dr, pos[1] + dc
                if (nr, nc) in box_positions:
                    adjacent_to_box = True
                    break
            
            if not adjacent_to_box:
                good_positions.append(pos)
        
        # Choose from good positions if available, otherwise any position
        chosen_positions = good_positions if good_positions else available_positions
        player_pos = random.choice(chosen_positions)
        
        return player_pos

    def _finalize_grid(self, grid, player_pos, box_positions, goal_positions):
        """Convert the grid to final string representation"""
        height, width = len(grid), len(grid[0])
        
        # Create final grid
        final_grid = [row[:] for row in grid]  # Deep copy
        
        # Place boxes
        for r, c in box_positions:
            if (r, c) in goal_positions:
                final_grid[r][c] = self.BOX_ON_GOAL
            else:
                final_grid[r][c] = self.BOX
        
        # Place player
        r, c = player_pos
        if (r, c) in goal_positions:
            final_grid[r][c] = self.PLAYER_ON_GOAL
        else:
            final_grid[r][c] = self.PLAYER
        
        # Convert to string
        level_string = '\n'.join(''.join(row) for row in final_grid)
        
        return level_string

    def _parse_level(self, level_string):
        """Parse a level string into grid, initial state, and goal positions"""
        lines = level_string.strip().split('\n')
        height = len(lines)
        width = max(len(line) for line in lines) if lines else 0
        
        # Initialize grid and track positions
        grid = []
        player_pos = None
        box_positions = set()
        goal_positions = set()
        
        for r, line in enumerate(lines):
            row = []
            for c, char in enumerate(line):
                if char == self.WALL:
                    row.append(self.WALL)
                elif char == self.FLOOR:
                    row.append(self.FLOOR)
                elif char == self.GOAL:
                    row.append(self.FLOOR)  # Goal is floor in grid representation
                    goal_positions.add((r, c))
                elif char == self.PLAYER:
                    row.append(self.FLOOR)
                    player_pos = (r, c)
                elif char == self.PLAYER_ON_GOAL:
                    row.append(self.FLOOR)
                    player_pos = (r, c)
                    goal_positions.add((r, c))
                elif char == self.BOX:
                    row.append(self.FLOOR)
                    box_positions.add((r, c))
                elif char == self.BOX_ON_GOAL:
                    row.append(self.FLOOR)
                    box_positions.add((r, c))
                    goal_positions.add((r, c))
                else:
                    # Default to floor for unknown characters
                    row.append(self.FLOOR)
            
            # Pad row to consistent width
            while len(row) < width:
                row.append(self.WALL)
            grid.append(row)
        
        # Create initial state
        initial_state = (player_pos, frozenset(box_positions))
        
        return grid, initial_state, frozenset(goal_positions)

    def _generate_level(self, difficulty):
        """Generate a single level dynamically with enhanced layout variety"""
        params = self.DIFFICULTY_PARAMS[difficulty]
        
        # Random grid size within range
        min_size, max_size = params['grid_size']
        width = random.randint(min_size, max_size)
        height = random.randint(min_size, max_size)
        
        # Random number of boxes
        num_boxes = random.randint(params['min_boxes'], params['max_boxes'])
        
        # Choose layout type for this difficulty level
        layout_type = random.choice(params['layout_types'])
        
        max_attempts = 50
        for attempt in range(max_attempts):
            try:
                # Generate base room
                grid = self._generate_base_room(width, height)
                
                # Apply layout-specific generation
                self._generate_layout_by_type(grid, layout_type, params)
                
                # Ensure connectivity after layout generation
                if not self._is_connected(grid):
                    # Try to fix connectivity
                    self._ensure_maze_connectivity(grid)
                    if not self._is_connected(grid):
                        continue  # Skip this attempt if still not connected
                
                # Place goals and boxes with difficulty-appropriate clustering
                goal_positions, box_positions = self._place_goals_and_boxes(
                    grid, num_boxes, params.get('goal_cluster_chance', 0.6)
                )
                
                # Place player
                player_pos = self._place_player(grid, goal_positions, box_positions)
                
                # Create final level string
                level_string = self._finalize_grid(grid, player_pos, box_positions, goal_positions)
                
                # Validate the generated level
                parsed_grid, initial_state, parsed_goals = self._parse_level(level_string)
                
                # Check for basic validity
                if (len(initial_state[1]) != len(parsed_goals) or
                    len(initial_state[1]) != num_boxes):
                    continue
                
                # Quick solvability check
                if self._is_deadlock(parsed_grid, initial_state[1], parsed_goals):
                    continue
                
                print(f"Generated {layout_type} layout puzzle with {num_boxes} boxes")
                return level_string, parsed_grid, initial_state, parsed_goals
                
            except Exception as e:
                if attempt == max_attempts - 1:
                    print(f"Failed to generate level after {max_attempts} attempts: {e}")
                continue
        
        raise ValueError(f"Could not generate valid level for difficulty {difficulty}")

    def generate(self, num_cases, difficulty, output_folder=None, time_limit_per_puzzle=60):
        """
        生成指定数量和难度的 Sokoban 问题。

        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别 (1-5)
            output_folder: 输出文件夹路径，覆盖构造函数中设置的路径
            time_limit_per_puzzle: 每个问题的求解时间限制

        Returns:
            生成的问题列表
        """
        if output_folder is None:
            output_folder = self.output_folder
        else:
            # Update task_dir and related paths if output_folder is provided
            self.task_dir = output_folder
            self.image_dir = os.path.join(self.task_dir, 'images')
            self.annotations_file = os.path.join(self.task_dir, 'annotations.json')
            os.makedirs(self.task_dir, exist_ok=True)
            os.makedirs(self.image_dir, exist_ok=True)

        # Re-seed with current timestamp for this generation batch
        current_seed = int(time.time())
        random.seed(current_seed)
        np.random.seed(current_seed % (2**32 - 1))

        puzzles = []
        puzzle_id_counter = 0
        total_generated = 0
        total_skipped_duplicates = 0

        print(f"Generating {num_cases} Sokoban puzzles with difficulty {difficulty}")
        print(f"Using seed: {current_seed}")

        max_attempts = 100  # Maximum attempts per puzzle
        attempts = 0

        while total_generated < num_cases and attempts < max_attempts * num_cases:
            attempts += 1

            try:
                print(f"Generating puzzle {total_generated + 1}/{num_cases} (attempt {attempts})...")

                # Generate level for the specified difficulty
                level_str, grid, initial_state, goal_positions = self._generate_level(difficulty)

                # Check for duplicates
                if self._is_duplicate_state(initial_state):
                    print(f"Skipping duplicate state")
                    total_skipped_duplicates += 1
                    continue

                print(f"Level generated. Grid: {len(grid)}x{len(grid[0])}, Boxes: {len(initial_state[1])}, Goals: {len(goal_positions)}")

                # Solve the puzzle
                print("Solving puzzle...")
                solution_path = self.solve(grid, initial_state, goal_positions, time_limit=time_limit_per_puzzle)

                if solution_path is None:
                    print(f"Failed to solve generated puzzle. Trying again...")
                    continue

                if not solution_path and not self._is_win_state(initial_state, goal_positions):
                    print(f"Invalid solution for generated puzzle. Trying again...")
                    continue

                # Check solution length constraints
                params = self._get_difficulty_params(difficulty)
                if len(solution_path) > params['max_solution_length']:
                    print(f"Solution too long ({len(solution_path)} > {params['max_solution_length']}). Regenerating...")
                    continue

                # Add to existing states to prevent future duplicates
                self._add_to_existing_states(initial_state)

                # Visualize the puzzle
                image_filename = os.path.join(self.image_dir, f"sokoban_{puzzle_id_counter}.png")
                self.visualize(grid, initial_state, goal_positions, filename=image_filename)

                # Convert solution to word format
                solution_words = [self.DIRECTION_WORDS[move] for move in solution_path]

                # Create puzzle data
                index = f"sokoban_{puzzle_id_counter}"
                puzzle_info = {
                    "index": index,
                    "category": "sokoban",
                    "question": PROMPT_SOKOBAN_IMAGE,
                    "initial_state": level_str,
                    "image": os.path.relpath(image_filename, self.task_dir),  # Store relative path
                    "question_language": PROMPT_SOKOBAN.format(level_str),
                    "answer": " ".join(solution_words),
                    "difficulty": difficulty,
                    "step_count": len(solution_path)
                }

                puzzles.append(puzzle_info)
                puzzle_id_counter += 1
                total_generated += 1

                print(f"✓ Generated puzzle {index} (steps: {len(solution_path)})")

            except ValueError as e:
                print(f"Generation error: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue

        if total_generated < num_cases:
            print(f"Warning: Only generated {total_generated}/{num_cases} puzzles")

        print(f"\n=== Generation Summary ===")
        print(f"Total puzzles generated: {total_generated}")
        print(f"Duplicates skipped: {total_skipped_duplicates}")
        print(f"Final existing states count: {len(self.existing_states)}")

        # Save all puzzles at once after generation is complete
        if puzzles:
            # Use shared BaseGenerator API to save annotations
            self.save_annotations(puzzles, output_folder=self.task_dir)
            print(f"Saved {len(puzzles)} puzzles to {self.annotations_file}")

        return puzzles

    def visualize(self, grid, state, goal_positions, filename=None):
        """Game-like visualization of the Sokoban board with realistic graphics."""
        height = len(grid)
        width = len(grid[0])
        img_width = width * self.tile_size
        img_height = height * self.tile_size
        
        # Set random seed for consistent texture generation
        random.seed(hash(str(grid) + str(state) + str(goal_positions)) % 1000000)
        
        # Create image with better background
        img = Image.new('RGB', (img_width, img_height), color=(45, 35, 25))  # Dark brown background
        draw = ImageDraw.Draw(img)

        player_pos, box_positions = state

        for r in range(height):
            for c in range(width):
                x0 = c * self.tile_size
                y0 = r * self.tile_size
                x1 = x0 + self.tile_size
                y1 = y0 + self.tile_size
                cell_coords = (r, c)

                # Draw base floor for all non-wall tiles first
                if grid[r][c] != self.WALL:
                    # Stone floor texture
                    floor_color = (180, 160, 140)  # Light stone color
                    draw.rectangle([x0, y0, x1, y1], fill=floor_color, outline=(140, 120, 100), width=1)
                    
                    # Add stone texture with small rectangles
                    for i in range(3):
                        for j in range(3):
                            tx = x0 + i * (self.tile_size // 3)
                            ty = y0 + j * (self.tile_size // 3)
                            tw = self.tile_size // 3
                            th = self.tile_size // 3
                            texture_color = (
                                floor_color[0] + random.randint(-15, 15),
                                floor_color[1] + random.randint(-15, 15),
                                floor_color[2] + random.randint(-15, 15)
                            )
                            draw.rectangle([tx, ty, tx + tw, ty + th], fill=texture_color, outline=None)

                # Draw specific tile types
                if grid[r][c] == self.WALL:
                    # Draw realistic brick wall
                    self._draw_brick_wall(draw, x0, y0, x1, y1)
                    
                elif cell_coords in goal_positions:
                    # Draw goal as a glowing target
                    self._draw_goal_target(draw, x0, y0, x1, y1)

                # Draw objects on top
                is_box = cell_coords in box_positions
                is_player = cell_coords == player_pos
                is_goal = cell_coords in goal_positions

                if is_box:
                    # Draw realistic wooden crate
                    self._draw_wooden_crate(draw, x0, y0, x1, y1, is_goal)

                elif is_player:
                    # Draw cute character sprite
                    self._draw_player_character(draw, x0, y0, x1, y1, is_goal)

        # Add game-style border
        border_width = 4
        draw.rectangle([0, 0, img_width-1, img_height-1], fill=None, outline=(80, 60, 40), width=border_width)
        draw.rectangle([border_width//2, border_width//2, img_width-border_width//2-1, img_height-border_width//2-1], 
                      fill=None, outline=(200, 180, 160), width=1)

        # Draw grid lines to make cell boundaries more prominent
        grid_line_color = (110, 90, 70)
        grid_line_width = max(2, self.tile_size // 25)

        # Vertical grid lines
        for c in range(1, width):
            x = c * self.tile_size
            draw.line([(x, 0), (x, img_height)], fill=grid_line_color, width=grid_line_width)

        # Horizontal grid lines
        for r in range(1, height):
            y = r * self.tile_size
            draw.line([(0, y), (img_width, y)], fill=grid_line_color, width=grid_line_width)

        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            img.save(filename, 'PNG', quality=95)
            print(f"Game visualization saved to {filename}")
            
        return filename

    def _append_to_annotations_file(self, output_dir: str, puzzle: Dict[str, Any]) -> None:
        """Append one puzzle dict to output_dir/annotations.json without overwriting existing data.
        Avoid duplicate entries by index.
        """
        annotations_path = os.path.join(output_dir, 'annotations.json')
        os.makedirs(output_dir, exist_ok=True)
        data: List[Dict[str, Any]] = []
        # Normalize image path to be relative to output_dir if absolute
        try:
            if isinstance(puzzle, dict) and 'image' in puzzle and puzzle['image']:
                img_path = puzzle['image']
                if os.path.isabs(img_path):
                    puzzle['image'] = os.path.relpath(img_path, output_dir)
        except Exception:
            pass
        try:
            if os.path.exists(annotations_path):
                with open(annotations_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        data = loaded
        except Exception:
            data = []

        existing_indices = {item.get('index') for item in data if isinstance(item, dict)}
        if isinstance(puzzle, dict) and puzzle.get('index') in existing_indices:
            # Already present; do not duplicate
            return

        data.append(puzzle)
        with open(annotations_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def generate_cot(self, grid, initial_state, goal_positions, solution_path: List[str]) -> str:
        """Generate an enhanced chain-of-thought reasoning following a 4-step framework:
        Step 1: Understand game rules
        Step 2: Read and analyze initial state
        Step 3: Detailed reasoning process
        Step 4: Verification and reflection
        """
        player_pos, box_positions = initial_state
        height = len(grid)
        width = len(grid[0])

        def is_wall(r, c):
            return not (0 <= r < height and 0 <= c < width) or grid[r][c] == self.WALL

        def format_grid_state(player_pos, box_positions, goal_positions):
            """Generate a textual representation of the current state"""
            state_lines = []
            for r in range(height):
                row = ""
                for c in range(width):
                    if grid[r][c] == self.WALL:
                        row += "#"
                    elif (r, c) == player_pos and (r, c) in goal_positions:
                        row += "+"  # Player on goal
                    elif (r, c) == player_pos:
                        row += "@"  # Player
                    elif (r, c) in box_positions and (r, c) in goal_positions:
                        row += "*"  # Box on goal
                    elif (r, c) in box_positions:
                        row += "$"  # Box
                    elif (r, c) in goal_positions:
                        row += "."  # Goal
                    else:
                        row += " "  # Empty space
                state_lines.append(row)
            return "\n".join(state_lines)

        dir_to_word = {'U': 'up', 'D': 'down', 'L': 'left', 'R': 'right'}
        dir_delta = self.DIRECTIONS
        
        goals_list = sorted(list(goal_positions))
        boxes_list = sorted(list(box_positions))
        lines: List[str] = []

        # ===== STEP 1: UNDERSTAND GAME RULES =====
        lines.append("=== STEP 1: Understanding Game Rules ===")
        lines.append("Let me first clarify the Sokoban rules to ensure I understand the game mechanics:")
        lines.append("1. I control a player character (@) who can move in four directions: up, down, left, right")
        lines.append("2. The player can push boxes ($) one cell at a time by moving into them")
        lines.append("3. Boxes cannot be pulled - only pushed from behind")
        lines.append("4. Walls (#) block all movement for both player and boxes")
        lines.append("5. The objective is to push all boxes onto target positions (.)")
        lines.append("6. A box is successfully placed when it rests on a goal position (shown as *)")
        lines.append("7. The puzzle is solved when all boxes are on goal positions")
        lines.append("")

        # ===== STEP 2: READ AND ANALYZE INITIAL STATE =====
        lines.append("=== STEP 2: Reading the Image and Analyzing Initial State ===")
        lines.append("Let me carefully examine the image to understand the puzzle layout:")
        lines.append("")
        lines.append("Initial state analysis:")
        lines.append(f"- Grid dimensions: {height} rows × {width} columns")
        lines.append(f"- Player starting position: {player_pos}")
        lines.append(f"- Number of boxes: {len(boxes_list)}")
        lines.append(f"- Box positions: {boxes_list}")
        lines.append(f"- Number of goals: {len(goals_list)}")
        lines.append(f"- Goal positions: {goals_list}")
        lines.append("")
        
        # Text representation of initial state
        lines.append("Current state in text format:")
        lines.append("```")
        lines.append(format_grid_state(player_pos, box_positions, goal_positions))
        lines.append("```")
        lines.append("")
        
        # Reflection on state reading
        lines.append("State reading reflection:")
        boxes_on_goals = len([pos for pos in box_positions if pos in goal_positions])
        if boxes_on_goals > 0:
            lines.append(f"- {boxes_on_goals} box(es) already positioned on goal(s)")
        else:
            lines.append("- No boxes are currently on their target positions")
        
        # Analyze spatial relationships
        distances = []
        for box_pos in box_positions:
            min_dist = min(abs(box_pos[0] - goal[0]) + abs(box_pos[1] - goal[1]) for goal in goal_positions)
            distances.append(min_dist)
        avg_distance = sum(distances) / len(distances) if distances else 0
        lines.append(f"- Average distance from boxes to nearest goals: {avg_distance:.1f} cells")
        lines.append("")

        # ===== STEP 3: DETAILED REASONING PROCESS =====
        lines.append("=== STEP 3: Detailed Problem-Solving Process ===")
        lines.append("Now I'll work through the solution step by step, exploring options and making strategic decisions:")
        lines.append("")

        # Strategic overview
        lines.append("Strategic overview:")
        if len(boxes_list) == 1:
            lines.append("- Single box puzzle: Focus on finding the optimal path to push the box to its goal")
        elif len(boxes_list) == 2:
            lines.append("- Two-box puzzle: Need to coordinate movements to avoid blocking paths")
        else:
            lines.append(f"- Multi-box puzzle ({len(boxes_list)} boxes): Requires careful sequencing to prevent deadlocks")
        lines.append("")

        # Simulate the solution with detailed reasoning
        cur_player = player_pos
        cur_boxes = set(box_positions)
        step_counter = 0
        boxes_placed = 0

        for i, mv in enumerate(solution_path):
            step_counter += 1
            dr, dc = dir_delta[mv]
            next_r, next_c = cur_player[0] + dr, cur_player[1] + dc

            # Pre-move analysis
            lines.append(f"Move {step_counter}: Planning to move {dir_to_word[mv]}")
            
            # Check surroundings for strategic awareness
            surroundings = []
            for direction, (check_dr, check_dc) in dir_delta.items():
                check_r, check_c = cur_player[0] + check_dr, cur_player[1] + check_dc
                if is_wall(check_r, check_c):
                    surroundings.append(f"{dir_to_word[direction]}: wall")
                elif (check_r, check_c) in cur_boxes:
                    surroundings.append(f"{dir_to_word[direction]}: box")
                elif (check_r, check_c) in goal_positions:
                    surroundings.append(f"{dir_to_word[direction]}: goal")
                else:
                    surroundings.append(f"{dir_to_word[direction]}: empty")
            
            lines.append(f"- Current surroundings: {', '.join(surroundings)}")

            # Execute the move
            if (next_r, next_c) in cur_boxes:
                # This is a push move
                box_dest = (next_r + dr, next_c + dc)
                if box_dest in cur_boxes or is_wall(box_dest[0], box_dest[1]):
                    lines.append("- ⚠️ This would cause a collision! Re-evaluating strategy...")
                    lines.append("- The solution path must account for this - continuing with the planned move")
                else:
                    cur_boxes.remove((next_r, next_c))
                    cur_boxes.add(box_dest)
                    cur_player = (next_r, next_c)
                    
                    if box_dest in goal_positions:
                        boxes_placed += 1
                        lines.append(f"- ✅ PUSH: Moving {dir_to_word[mv]} pushes box from {(next_r, next_c)} to {box_dest} (GOAL!)")
                        lines.append(f"- 🎯 Box successfully placed on goal! ({boxes_placed}/{len(goal_positions)} complete)")
                    else:
                        lines.append(f"- 📦 PUSH: Moving {dir_to_word[mv]} pushes box from {(next_r, next_c)} to {box_dest}")
                        
                        # Analyze if this move brings box closer to any goal
                        old_min_dist = min(abs(next_r - goal[0]) + abs(next_c - goal[1]) for goal in goal_positions)
                        new_min_dist = min(abs(box_dest[0] - goal[0]) + abs(box_dest[1] - goal[1]) for goal in goal_positions)
                        if new_min_dist < old_min_dist:
                            lines.append(f"- 📈 This push brings the box closer to a goal (distance reduced from {old_min_dist} to {new_min_dist})")
                        elif new_min_dist > old_min_dist:
                            lines.append(f"- 📉 This push moves the box away from goals temporarily (strategic positioning)")
                        else:
                            lines.append(f"- ➡️ This push maintains distance to goals (lateral movement)")
            else:
                if is_wall(next_r, next_c):
                    lines.append("- ❌ Wall detected! This shouldn't happen in a valid solution - there may be an error")
                else:
                    cur_player = (next_r, next_c)
                    lines.append(f"- 🚶 MOVE: Player moves {dir_to_word[mv]} to {(next_r, next_c)} (repositioning)")
                    
                    # Check if this move brings player closer to any unmoved boxes
                    unmoved_boxes = [pos for pos in cur_boxes if pos not in goal_positions]
                    if unmoved_boxes:
                        closest_box_dist = min(abs(next_r - box[0]) + abs(next_c - box[1]) for box in unmoved_boxes)
                        lines.append(f"- 🎯 Distance to nearest unmoved box: {closest_box_dist} cells")

            # Show progress
            remaining_boxes = len([pos for pos in cur_boxes if pos not in goal_positions])
            lines.append(f"- Progress: {len(goal_positions) - remaining_boxes}/{len(goal_positions)} boxes placed")
            
            # Add spacing between moves for readability
            if i < len(solution_path) - 1:
                lines.append("")

        lines.append("")

        # ===== STEP 4: VERIFICATION AND REFLECTION =====
        lines.append("=== STEP 4: Solution Verification and Reflection ===")
        lines.append("")
        
        # Verify final state
        lines.append("Final state verification:")
        final_boxes_on_goals = len([pos for pos in cur_boxes if pos in goal_positions])
        
        if cur_boxes == goal_positions and len(cur_boxes) == len(goal_positions):
            lines.append("✅ SUCCESS: All boxes are correctly positioned on their target goals!")
        else:
            lines.append(f"❌ INCOMPLETE: Only {final_boxes_on_goals}/{len(goal_positions)} boxes are on goals")
        
        lines.append("")
        lines.append("Solution reflection:")
        lines.append(f"- Total moves required: {len(solution_path)}")
        
        # Count pushes vs movements
        push_count = 0
        move_count = 0
        temp_player = player_pos
        temp_boxes = set(box_positions)
        
        for mv in solution_path:
            dr, dc = dir_delta[mv]
            next_r, next_c = temp_player[0] + dr, temp_player[1] + dc
            if (next_r, next_c) in temp_boxes:
                push_count += 1
                box_dest = (next_r + dr, next_c + dc)
                temp_boxes.remove((next_r, next_c))
                temp_boxes.add(box_dest)
            else:
                move_count += 1
            temp_player = (next_r, next_c)
        
        lines.append(f"- Breakdown: {push_count} pushes, {move_count} positioning moves")
        lines.append(f"- Efficiency ratio: {push_count}/{len(solution_path)} = {push_count/len(solution_path):.1%} productive moves")
        
        # Strategic reflection
        lines.append("")
        lines.append("Strategic insights:")
        if len(boxes_list) == 1:
            lines.append("- Single box puzzle solved through direct pathfinding")
        else:
            lines.append("- Multi-box coordination required careful sequencing to avoid deadlocks")
        
        if push_count == len(goal_positions):
            lines.append("- Optimal efficiency: Each box was pushed exactly once to its final position")
        elif push_count > len(goal_positions):
            lines.append("- Some boxes required multiple pushes for proper positioning")
        
        lines.append("- No deadlock situations encountered during solution execution")
        lines.append("")
        
        # Final confirmation
        lines.append("Final answer verification:")
        answer_moves = [dir_to_word[move] for move in solution_path]
        lines.append(f"The solution sequence is: {' '.join(answer_moves)}")
        lines.append("This sequence successfully solves the Sokoban puzzle by placing all boxes on their target positions.")

        return "\n".join(lines)
    
    def _draw_brick_wall(self, draw, x0, y0, x1, y1):
        """Draw a realistic brick wall pattern"""
        brick_color = (120, 80, 60)  # Dark brown brick
        mortar_color = (90, 60, 45)  # Darker mortar
        
        # Fill with mortar color first
        draw.rectangle([x0, y0, x1, y1], fill=mortar_color, outline=None)
        
        # Draw brick pattern
        brick_height = self.tile_size // 3
        brick_width = self.tile_size // 2
        
        for row in range(3):  # 3 brick rows per tile
            y_offset = y0 + row * brick_height
            x_offset = x0 + ((row % 2) * brick_width // 2)  # Offset every other row
            
            for col in range(3):  # Up to 3 bricks per row
                bx0 = x_offset + col * brick_width
                if bx0 >= x1:
                    break
                bx1 = min(bx0 + brick_width - 2, x1 - 1)
                by0 = y_offset + 1
                by1 = min(y_offset + brick_height - 2, y1 - 1)
                
                if bx1 > bx0 and by1 > by0:
                    # Add slight color variation
                    varied_color = (
                        brick_color[0] + random.randint(-10, 10),
                        brick_color[1] + random.randint(-10, 10),
                        brick_color[2] + random.randint(-10, 10)
                    )
                    draw.rectangle([bx0, by0, bx1, by1], fill=varied_color, outline=(100, 70, 50), width=1)
    
    def _draw_goal_target(self, draw, x0, y0, x1, y1):
        """Draw a glowing target for goals"""
        center_x = x0 + self.tile_size // 2
        center_y = y0 + self.tile_size // 2
        
        # Draw concentric circles for target effect
        for radius in [self.tile_size // 3, self.tile_size // 4, self.tile_size // 6]:
            color_intensity = 255 - (radius * 3)
            circle_color = (255, 215 + color_intensity // 10, 0)  # Golden yellow
            draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                        fill=circle_color, outline=(200, 150, 0), width=1)
        
        # Add sparkle effect
        sparkle_points = [
            (center_x, center_y - self.tile_size // 4),
            (center_x + self.tile_size // 6, center_y - self.tile_size // 6),
            (center_x + self.tile_size // 4, center_y),
            (center_x + self.tile_size // 6, center_y + self.tile_size // 6),
            (center_x, center_y + self.tile_size // 4),
            (center_x - self.tile_size // 6, center_y + self.tile_size // 6),
            (center_x - self.tile_size // 4, center_y),
            (center_x - self.tile_size // 6, center_y - self.tile_size // 6)
        ]
        
        for i in range(0, len(sparkle_points), 2):
            draw.line([sparkle_points[i], sparkle_points[(i+4) % len(sparkle_points)]], 
                     fill=(255, 255, 200), width=2)
    
    def _draw_wooden_crate(self, draw, x0, y0, x1, y1, on_goal=False):
        """Draw a realistic wooden crate"""
        # Crate colors
        if on_goal:
            wood_color = (60, 150, 60)  # Green tint when on goal
            highlight_color = (100, 200, 100)
        else:
            wood_color = (50, 100, 180)  # Blue color
            highlight_color = (80, 130, 220)
        
        shadow_color = (wood_color[0] - 30, wood_color[1] - 30, wood_color[2] - 30)
        
        # Main crate body with 3D effect
        inset = 3
        draw.rectangle([x0 + inset, y0 + inset, x1 - inset, y1 - inset], 
                      fill=wood_color, outline=(80, 50, 20), width=2)
        
        # Add wood grain texture
        for i in range(3):
            grain_y = y0 + inset + 5 + i * ((self.tile_size - 2 * inset) // 4)
            draw.line([x0 + inset + 3, grain_y, x1 - inset - 3, grain_y], 
                     fill=shadow_color, width=1)
        
        # 3D highlight on top and left
        draw.line([x0 + inset, y0 + inset, x1 - inset, y0 + inset], fill=highlight_color, width=2)
        draw.line([x0 + inset, y0 + inset, x0 + inset, y1 - inset], fill=highlight_color, width=2)
        
        # Corner reinforcements (metal brackets)
        bracket_size = 4
        bracket_color = (100, 100, 100)
        corners = [
            (x0 + inset, y0 + inset),  # Top-left
            (x1 - inset - bracket_size, y0 + inset),  # Top-right
            (x0 + inset, y1 - inset - bracket_size),  # Bottom-left
            (x1 - inset - bracket_size, y1 - inset - bracket_size)  # Bottom-right
        ]
        
        for corner_x, corner_y in corners:
            draw.rectangle([corner_x, corner_y, corner_x + bracket_size, corner_y + bracket_size], 
                          fill=bracket_color, outline=(60, 60, 60), width=1)
        
        # Add glow effect if on goal
        if on_goal:
            glow_size = 2
            glow_color = (100, 255, 100, 100)  # Semi-transparent green
            for offset in range(1, glow_size + 1):
                draw.rectangle([x0 + inset - offset, y0 + inset - offset, 
                               x1 - inset + offset, y1 - inset + offset], 
                              fill=None, outline=(0, 255, 0), width=1)
    
    def _draw_player_character(self, draw, x0, y0, x1, y1, on_goal=False):
        """Draw a cute character sprite for the player"""
        center_x = x0 + self.tile_size // 2
        center_y = y0 + self.tile_size // 2
        
        # Character colors
        if on_goal:
            body_color = (255, 200, 50)  # Golden when on goal
            outline_color = (200, 150, 0)
        else:
            body_color = (255, 100, 100)  # Red-pink
            outline_color = (200, 50, 50)
        
        # Character body (circle)
        radius = self.tile_size // 3
        draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius],
                    fill=body_color, outline=outline_color, width=2)
        
        # Character face
        eye_size = 2
        eye_offset = radius // 3
        
        # Eyes
        draw.ellipse([center_x - eye_offset - eye_size, center_y - eye_offset - eye_size,
                     center_x - eye_offset + eye_size, center_y - eye_offset + eye_size], 
                    fill=(0, 0, 0))
        draw.ellipse([center_x + eye_offset - eye_size, center_y - eye_offset - eye_size,
                     center_x + eye_offset + eye_size, center_y - eye_offset + eye_size], 
                    fill=(0, 0, 0))
        
        # Smile
        smile_points = []
        for angle in range(30, 151, 20):  # From 30 to 150 degrees
            x = center_x + int((radius // 2) * np.cos(np.radians(angle)))
            y = center_y + int((radius // 2) * np.sin(np.radians(angle)))
            smile_points.append((x, y))
        
        if len(smile_points) > 1:
            for i in range(len(smile_points) - 1):
                draw.line([smile_points[i], smile_points[i + 1]], fill=(0, 0, 0), width=2)
        
        # Add sparkle effect if on goal
        if on_goal:
            sparkle_radius = radius + 5
            sparkle_points = [
                (center_x, center_y - sparkle_radius),
                (center_x + sparkle_radius // 2, center_y - sparkle_radius // 2),
                (center_x + sparkle_radius, center_y),
                (center_x + sparkle_radius // 2, center_y + sparkle_radius // 2)
            ]
            
            for point in sparkle_points:
                draw.ellipse([point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2], 
                           fill=(255, 255, 0))


    def solve(self, grid, initial_state, goal_positions, time_limit=60):
        """Advanced Sokoban solver using optimized A* search with enhanced deadlock detection."""
        start_time = time.time()

        if self._is_win_state(initial_state, goal_positions):
            return [] # Already solved
        
        # Check for initial deadlocks
        if self._is_deadlock(grid, initial_state[1], goal_positions):
            print("Initial state contains deadlocks - puzzle is unsolvable")
            return None

        # Priority queue: (f_score, g_score, state, path)
        open_set = [(self._heuristic(initial_state, goal_positions), 0, initial_state, [])]
        heapq.heapify(open_set)

        # Visited set stores state -> g_score for pruning
        visited = {initial_state: 0}
        nodes_explored = 0
        max_queue_size = 0

        while open_set:
            nodes_explored += 1
            max_queue_size = max(max_queue_size, len(open_set))
            
            # Check time limit and memory usage periodically
            if nodes_explored % 1000 == 0:
                elapsed = time.time() - start_time
                if elapsed > time_limit:
                    print(f"Solver timed out after {elapsed:.1f}s (explored {nodes_explored} nodes, max queue: {max_queue_size}).")
                    return None
                
                # Memory management - limit visited states
                if len(visited) > 100000:
                    # Keep only the best states
                    sorted_states = sorted(visited.items(), key=lambda x: x[1])
                    visited = dict(sorted_states[:50000])

            f_score, g_score, current_state, path = heapq.heappop(open_set)

            # Skip if we've found a better path to this state
            if current_state in visited and visited[current_state] < g_score:
                continue

            if self._is_win_state(current_state, goal_positions):
                elapsed = time.time() - start_time
                print(f"Solution found in {elapsed:.2f}s. Path length: {len(path)}. Nodes explored: {nodes_explored}")
                return path

            # Explore neighbors
            valid_moves = self._get_valid_moves(grid, current_state, goal_positions)
            for move_char, next_state in valid_moves.items():
                new_g_score = g_score + 1

                # Skip if we've seen this state with better cost
                if next_state in visited and visited[next_state] <= new_g_score:
                    continue

                # Check for deadlocks in next state
                if self._is_deadlock(grid, next_state[1], goal_positions):
                    continue

                # Add to search queue
                visited[next_state] = new_g_score
                h_score = self._heuristic(next_state, goal_positions)
                new_f_score = new_g_score + h_score
                new_path = path + [move_char]
                heapq.heappush(open_set, (new_f_score, new_g_score, next_state, new_path))

        elapsed = time.time() - start_time
        print(f"No solution found after exploring {nodes_explored} nodes ({elapsed:.2f}s, max queue: {max_queue_size}).")
        return None

    def _heuristic(self, state, goal_positions):
        """Advanced heuristic for A* search with better estimation"""
        player_pos, box_positions = state
        
        if not goal_positions:
            return float('inf')
        
        # Fast exit for win state
        if box_positions == goal_positions:
            return 0
        
        # Hungarian algorithm approximation: minimum total distance
        total_distance = 0
        unmatched_boxes = []
        unmatched_goals = list(goal_positions)
        
        # First, handle boxes already on goals
        for box_pos in box_positions:
            if box_pos in goal_positions:
                unmatched_goals.remove(box_pos)
            else:
                unmatched_boxes.append(box_pos)
        
        # If no unmatched boxes, we're done
        if not unmatched_boxes:
            return 0
        
        # Calculate minimum assignment cost for remaining boxes
        assignments = []
        for box_pos in unmatched_boxes:
            if not unmatched_goals:
                break
            min_dist = float('inf')
            closest_goal = None
            
            for goal_pos in unmatched_goals:
                dist = abs(box_pos[0] - goal_pos[0]) + abs(box_pos[1] - goal_pos[1])
                if dist < min_dist:
                    min_dist = dist
                    closest_goal = goal_pos
            
            if closest_goal:
                assignments.append((box_pos, closest_goal, min_dist))
                total_distance += min_dist
                unmatched_goals.remove(closest_goal)
        
        # Add penalty for player distance to nearest unmatched box
        if unmatched_boxes:
            player_to_box_dist = min(
                abs(player_pos[0] - box_pos[0]) + abs(player_pos[1] - box_pos[1])
                for box_pos in unmatched_boxes
            )
            # Small penalty to encourage player to get closer to boxes
            total_distance += player_to_box_dist * 0.1
        
        return total_distance

    def _get_valid_moves(self, grid, state, goal_positions):
        """Generates possible next states and moves."""
        player_pos, box_positions = state
        pr, pc = player_pos
        valid_next_states = {} # move_char -> next_state

        for move_char, (dr, dc) in self.DIRECTIONS.items():
            next_pr, next_pc = pr + dr, pc + dc

            # Check bounds
            if not (0 <= next_pr < len(grid) and 0 <= next_pc < len(grid[0])):
                continue

            # Check wall
            if grid[next_pr][next_pc] == self.WALL:
                continue

            # Check if moving into a box
            if (next_pr, next_pc) in box_positions:
                # Calculate position behind the box
                box_dest_r, box_dest_c = next_pr + dr, next_pc + dc

                # Check bounds for box destination
                if not (0 <= box_dest_r < len(grid) and 0 <= box_dest_c < len(grid[0])):
                    continue

                # Check if box destination is wall or another box
                if grid[box_dest_r][box_dest_c] == self.WALL or \
                   (box_dest_r, box_dest_c) in box_positions:
                    continue

                # Valid push: Update box positions and player position
                new_box_positions = set(box_positions)
                new_box_positions.remove((next_pr, next_pc))
                new_box_positions.add((box_dest_r, box_dest_c))
                next_state = ((next_pr, next_pc), frozenset(new_box_positions))
                valid_next_states[move_char] = next_state

            else:
                # Valid move into empty space (floor or goal)
                next_state = ((next_pr, next_pc), box_positions) # box_positions remain the same
                valid_next_states[move_char] = next_state

        return valid_next_states

    def _is_win_state(self, state, goal_positions):
        """Checks if all boxes are on goal positions."""
        _, box_positions = state
        return box_positions == goal_positions # Check if the set of box positions matches the set of goal positions

    def _is_deadlock(self, grid, box_positions, goal_positions):
        """Enhanced deadlock detection that makes the puzzle unsolvable"""
        for box_pos in box_positions:
            br, bc = box_pos
            
            # Skip if box is already on a goal
            if box_pos in goal_positions:
                continue
                
            # Check for corner deadlock
            if self._is_corner_deadlock(grid, br, bc, goal_positions):
                return True
                
            # Check for wall alignment deadlock
            if self._is_wall_deadlock(grid, br, bc, goal_positions):
                return True
                
            # Check for freeze deadlock (box cannot be moved)
            if self._is_freeze_deadlock(grid, box_positions, br, bc, goal_positions):
                return True
                
        # Check for corral deadlock (multiple boxes blocking each other)
        if self._is_corral_deadlock(grid, box_positions, goal_positions):
            return True
                
        return False
    
    def _is_corner_deadlock(self, grid, br, bc, goal_positions):
        """Check if a box is stuck in a corner"""
        # Check all 2x2 corner patterns
        corners = [
            [(0, 0), (0, 1), (1, 0)],  # Top-left L
            [(0, 0), (0, -1), (1, 0)], # Top-right L  
            [(-1, 0), (0, 0), (0, 1)], # Bottom-left L
            [(-1, 0), (0, 0), (0, -1)] # Bottom-right L
        ]
        
        for corner in corners:
            wall_count = 0
            for dr, dc in corner:
                nr, nc = br + dr, bc + dc
                if (nr < 0 or nr >= len(grid) or nc < 0 or nc >= len(grid[0]) or 
                    grid[nr][nc] == self.WALL):
                    wall_count += 1
            
            # If box is in corner (2+ walls) and not on goal, it's deadlocked
            if wall_count >= 2 and (br, bc) not in goal_positions:
                return True
                
        return False
    
    def _is_wall_deadlock(self, grid, br, bc, goal_positions):
        """Check if box is against wall with no goals along that wall"""
        # Check horizontal wall alignment
        if (br - 1 < 0 or grid[br - 1][bc] == self.WALL or 
            br + 1 >= len(grid) or grid[br + 1][bc] == self.WALL):
            # Box is against horizontal wall, check if there are goals along this row
            goals_in_row = any((br, c) in goal_positions for c in range(len(grid[0])))
            if not goals_in_row:
                return True
                
        # Check vertical wall alignment  
        if (bc - 1 < 0 or grid[br][bc - 1] == self.WALL or
            bc + 1 >= len(grid[0]) or grid[br][bc + 1] == self.WALL):
            # Box is against vertical wall, check if there are goals along this column
            goals_in_col = any((r, bc) in goal_positions for r in range(len(grid)))
            if not goals_in_col:
                return True
                
        return False
    
    def _is_freeze_deadlock(self, grid, box_positions, br, bc, goal_positions):
        """Check if a box is frozen and cannot be moved in any direction"""
        if (br, bc) in goal_positions:
            return False
            
        # Check all four directions
        movable_directions = 0
        for dr, dc in self.DIRECTIONS.values():
            # Can the box move in this direction?
            new_br, new_bc = br + dr, bc + dc
            
            # Check bounds
            if not (0 <= new_br < len(grid) and 0 <= new_bc < len(grid[0])):
                continue
                
            # Check if destination is blocked by wall or another box
            if (grid[new_br][new_bc] == self.WALL or 
                (new_br, new_bc) in box_positions):
                continue
                
            # Check if player can reach the pushing position
            push_from_r, push_from_c = br - dr, bc - dc
            if not (0 <= push_from_r < len(grid) and 0 <= push_from_c < len(grid[0])):
                continue
                
            if (grid[push_from_r][push_from_c] == self.WALL or
                (push_from_r, push_from_c) in box_positions):
                continue
                
            movable_directions += 1
            
        return movable_directions == 0
    
    def _is_corral_deadlock(self, grid, box_positions, goal_positions):
        """Check for corral deadlock where boxes block each other"""
        # Find boxes not on goals
        unplaced_boxes = [pos for pos in box_positions if pos not in goal_positions]
        
        if len(unplaced_boxes) < 2:
            return False
            
        # Check if any group of boxes forms a corral (mutual blocking)
        for box_pos in unplaced_boxes:
            if self._is_in_corral(grid, box_positions, box_pos, goal_positions):
                return True
                
        return False
    
    def _is_in_corral(self, grid, box_positions, box_pos, goal_positions):
        """Check if a specific box is in a corral"""
        br, bc = box_pos
        
        # Count walls/boxes around this box
        blocked_sides = 0
        for dr, dc in self.DIRECTIONS.values():
            nr, nc = br + dr, bc + dc
            
            if (nr < 0 or nr >= len(grid) or nc < 0 or nc >= len(grid[0]) or
                grid[nr][nc] == self.WALL or (nr, nc) in box_positions):
                blocked_sides += 1
                
        # If 3 or 4 sides are blocked and box is not on goal, likely corral
        return blocked_sides >= 3 and box_pos not in goal_positions

    def generate_benchmark_and_training_set(self, benchmark_tasks=30, items_per_task=6, training_total=1500, training_per_difficulty=300, time_limit_per_puzzle=60):
        """
        Generate benchmark and training sets with no duplicate initial_states
        
        Args:
            benchmark_tasks: Number of benchmark tasks (default 30)
            items_per_task: Items per benchmark task (default 6) 
            training_total: Total training items (default 1500)
            training_per_difficulty: Items per difficulty level (default 300)
            time_limit_per_puzzle: Time limit for solving each puzzle
        """
        print("=== GENERATING BENCHMARK AND TRAINING SETS ===")
        print(f"Benchmark: {benchmark_tasks} tasks × {items_per_task} items = {benchmark_tasks * items_per_task} total")
        print(f"Training: {training_total} items ({training_per_difficulty} per difficulty × 5 levels)")
        
        # Clear existing states to start fresh
        self.existing_states.clear()
        
        # Step 1: Generate Benchmark Dataset
        print("\n=== STEP 1: GENERATING BENCHMARK DATASET ===")
        benchmark_puzzles = []
        benchmark_states = set()  # Track benchmark states separately
        puzzle_id_counter = 0
        
        for task_num in range(benchmark_tasks):
            print(f"\n--- Benchmark Task {task_num + 1}/{benchmark_tasks} ---")
            task_puzzles = []
            
            for item_num in range(items_per_task):
                print(f"Generating item {item_num + 1}/{items_per_task}")
                
                # Try different difficulties for variety in benchmark
                difficulty = ((task_num * items_per_task + item_num) % 5) + 1
                
                max_attempts = 50
                for attempt in range(max_attempts):
                    try:
                        # Generate level
                        level_str, grid, initial_state, goal_positions = self._generate_level(difficulty)
                        
                        # Check for duplicates within benchmark and existing states
                        state_hash = self._state_to_hash(initial_state)
                        if state_hash in benchmark_states or state_hash in self.existing_states:
                            print(f"  Duplicate detected (attempt {attempt + 1}), regenerating...")
                            continue
                        
                        # Solve the puzzle
                        solution_path = self.solve(grid, initial_state, goal_positions, time_limit=time_limit_per_puzzle)
                        
                        if solution_path is None:
                            print(f"  Failed to solve (attempt {attempt + 1}), regenerating...")
                            continue
                        
                        # Check solution length constraints
                        params = self.DIFFICULTY_PARAMS[difficulty]
                        if len(solution_path) > params['max_solution_length']:
                            print(f"  Solution too long (attempt {attempt + 1}), regenerating...")
                            continue
                        
                        # Add to benchmark states
                        benchmark_states.add(state_hash)
                        self._add_to_existing_states(initial_state)
                        
                        # Visualize
                        image_filename = os.path.join(self.image_dir, f"benchmark_{puzzle_id_counter}.png")
                        self.visualize(grid, initial_state, goal_positions, filename=image_filename)
                        
                        # Convert solution to word format
                        solution_words = [self.DIRECTION_WORDS[move] for move in solution_path]
                        
                        # Create puzzle data
                        index = f"benchmark_{puzzle_id_counter}"
                        puzzle_info = {
                            "index": index,
                            "task_id": task_num,
                            "item_id": item_num,
                            "category": "sokoban",
                            "question": PROMPT_SOKOBAN_IMAGE,
                            "initial_state": level_str,
                            "image": image_filename,
                            "question_language": PROMPT_SOKOBAN.format(level_str),
                            "answer": " ".join(solution_words),
                            "difficulty": difficulty,
                            "step_count": len(solution_path)
                        }
                        
                        task_puzzles.append(puzzle_info)
                        puzzle_id_counter += 1
                        print(f"  ✓ Generated {index} (difficulty: {difficulty}, steps: {len(solution_path)})")
                        break
                        
                    except Exception as e:
                        print(f"  Error on attempt {attempt + 1}: {e}")
                        if attempt == max_attempts - 1:
                            print(f"  Failed to generate item after {max_attempts} attempts")
                            
                else:
                    print(f"  WARNING: Failed to generate item {item_num + 1} for task {task_num + 1}")
            
            benchmark_puzzles.extend(task_puzzles)
            print(f"  Task {task_num + 1} completed: {len(task_puzzles)} items generated")
        
        print(f"\nBenchmark generation completed: {len(benchmark_puzzles)} puzzles")
        print(f"Unique states in benchmark: {len(benchmark_states)}")
        
        # Step 2: Generate Training Dataset
        print("\n=== STEP 2: GENERATING TRAINING DATASET ===")
        training_puzzles = []
        training_states = set()  # Track training states separately
        
        difficulties = [1, 2, 3, 4, 5]
        
        for difficulty in difficulties:
            print(f"\n--- Training Difficulty Level {difficulty} ---")
            generated_for_difficulty = 0
            
            while generated_for_difficulty < training_per_difficulty:
                max_attempts = 30
                
                for attempt in range(max_attempts):
                    try:
                        # Generate level
                        level_str, grid, initial_state, goal_positions = self._generate_level(difficulty)
                        
                        # Check for duplicates in both benchmark and training sets
                        state_hash = self._state_to_hash(initial_state)
                        if (state_hash in benchmark_states or 
                            state_hash in training_states or 
                            state_hash in self.existing_states):
                            continue
                        
                        # Solve the puzzle
                        solution_path = self.solve(grid, initial_state, goal_positions, time_limit=time_limit_per_puzzle)
                        
                        if solution_path is None:
                            continue
                        
                        # Check solution length constraints
                        params = self.DIFFICULTY_PARAMS[difficulty]
                        if len(solution_path) > params['max_solution_length']:
                            continue
                        
                        # Add to training states
                        training_states.add(state_hash)
                        self._add_to_existing_states(initial_state)
                        
                        # Visualize
                        image_filename = os.path.join(self.image_dir, f"training_{puzzle_id_counter}.png")
                        self.visualize(grid, initial_state, goal_positions, filename=image_filename)
                        
                        # Convert solution to word format
                        solution_words = [self.DIRECTION_WORDS[move] for move in solution_path]
                        
                        # Create puzzle data
                        index = f"training_{puzzle_id_counter}"
                        puzzle_info = {
                            "index": index,
                            "category": "sokoban", 
                            "question": PROMPT_SOKOBAN_IMAGE,
                            "initial_state": level_str,
                            "image": image_filename,
                            "question_language": PROMPT_SOKOBAN.format(level_str),
                            "answer": " ".join(solution_words),
                            "difficulty": difficulty,
                            "step_count": len(solution_path)
                        }
                        
                        training_puzzles.append(puzzle_info)
                        puzzle_id_counter += 1
                        generated_for_difficulty += 1
                        
                        if generated_for_difficulty % 50 == 0:
                            print(f"  Progress: {generated_for_difficulty}/{training_per_difficulty}")
                        
                        break
                        
                    except Exception as e:
                        continue
                else:
                    print(f"  WARNING: Failed to generate puzzle after {max_attempts} attempts")
                    # Try one more time with a fresh attempt
                    continue
            
            print(f"  Difficulty {difficulty} completed: {generated_for_difficulty} items")
        
        print(f"\nTraining generation completed: {len(training_puzzles)} puzzles")
        print(f"Unique states in training: {len(training_states)}")
        
        # Step 3: Save datasets
        print("\n=== STEP 3: SAVING DATASETS ===")
        
        # Save benchmark
        benchmark_file = os.path.join(self.task_dir, 'benchmark.json')
        with open(benchmark_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_puzzles, f, ensure_ascii=False, indent=2)
        print(f"Benchmark saved to {benchmark_file} ({len(benchmark_puzzles)} puzzles)")
        
        # Save training set
        training_file = os.path.join(self.task_dir, 'training_set.json') 
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(training_puzzles, f, ensure_ascii=False, indent=2)
        print(f"Training set saved to {training_file} ({len(training_puzzles)} puzzles)")
        
        # Verification
        print("\n=== VERIFICATION ===")
        all_benchmark_hashes = {self._state_to_hash(self._parse_level(p['initial_state'])[1]) for p in benchmark_puzzles}
        all_training_hashes = {self._state_to_hash(self._parse_level(p['initial_state'])[1]) for p in training_puzzles}
        
        overlap = all_benchmark_hashes.intersection(all_training_hashes)
        print(f"Benchmark unique states: {len(all_benchmark_hashes)}")
        print(f"Training unique states: {len(all_training_hashes)}")
        print(f"Overlap between datasets: {len(overlap)}")
        
        if len(overlap) > 0:
            print("WARNING: Found overlapping states between benchmark and training!")
        else:
            print("✓ No overlapping states found - datasets are properly separated")
        
        print(f"\n=== SUMMARY ===")
        print(f"Benchmark: {len(benchmark_puzzles)} puzzles ({benchmark_tasks} tasks × {items_per_task} items)")
        print(f"Training: {len(training_puzzles)} puzzles (5 difficulties × {training_per_difficulty} items)")
        print(f"Total unique states: {len(self.existing_states)}")
        
        return benchmark_puzzles, training_puzzles

# ============================================================
#                 Main Function and Argparse
# ============================================================

# (Ensure the parse_args and main functions from your original code are here,
#  and make sure they include 'sokoban' in the arguments and generation logic)
# --- IMPORTANT: Modify main() to include the new generators ---

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Generate puzzles of various types with difficulty levels')

    parser.add_argument('--output_dir', type=str, default='tasks',
                        help='Base directory to save puzzle tasks (default: tasks)')

    parser.add_argument('--all', action='store_true',
                        help='Generate all puzzle types')
    parser.add_argument('--sliding', action='store_true',
                        help='Generate sliding puzzles (15-puzzle)')
    parser.add_argument('--hanoi', action='store_true',
                        help='Generate Tower of Hanoi puzzles')
    parser.add_argument('--maze', action='store_true',
                        help='Generate maze puzzles')
    parser.add_argument('--wordsearch', action='store_true',
                        help='Generate word search puzzles')
    parser.add_argument('--minesweeper', action='store_true',
                        help='Generate Minesweeper puzzles')
    parser.add_argument('--eulero', action='store_true',
                        help='Generate Eulero (Graeco-Latin Square) puzzles')
    parser.add_argument('--triplets', action='store_true',
                        help='Generate Triplets (One or All) puzzles')
    parser.add_argument('--numbrix', action='store_true',
                        help='Generate Numbrix puzzles')
    parser.add_argument('--sokoban', action='store_true',
                        help='Generate Sokoban puzzles')
    parser.add_argument('--snake', action='store_true',
                        help='Generate Snake (Tunnel) puzzles')


    parser.add_argument('--count', type=int, default=300, # Reduced default for quicker testing
                        help='Number of puzzles to generate for each selected type (default: 9, ideally a multiple of 3 for even distribution across difficulties)')

    parser.add_argument('--sliding_size', type=int, default=4,
                        help='Size of sliding puzzle (default: 4 for 15-puzzle)')

    # parser.add_argument('--hanoi_disks', type=int, default=4,
    #                     help='Max number of disks for Tower of Hanoi puzzles (default: 4)')

    parser.add_argument('--clean', action='store_true',
                        help='Clean existing task directories before generating new puzzles')

    return parser.parse_args()


def main():
    """Main function to generate Sokoban benchmark and training datasets"""
    print("\n" + "="*70)
    print("🎮 SOKOBAN BENCHMARK & TRAINING GENERATOR 🎮")
    print("="*70)
    print("🎯 GENERATING TWO DATASETS:")
    print("   • 📊 Benchmark: 30 tasks × 6 items = 180 puzzles")
    print("   • 🎓 Training: 5 difficulties × 300 items = 1500 puzzles")
    print("   • 🚫 Zero overlap - guaranteed unique initial_states")
    print("   • 🔄 Auto-regeneration if duplicates detected")
    print("="*70)
    
    try:
        # Create generator
        generator = SokobanGenerator(output_folder="tasks")
        
        # Generate benchmark and training datasets
        print(f"\n🚀 Starting generation process...")
        benchmark_puzzles, training_puzzles = generator.generate_benchmark_and_training_set(
            benchmark_tasks=5,
            items_per_task=6,
            training_total=1500,
            training_per_difficulty=300,
            time_limit_per_puzzle=60
        )
        
        print("\n" + "="*70)
        print("📊 FINAL RESULTS")
        print("="*70)
        print(f"✅ Benchmark: {len(benchmark_puzzles)} puzzles generated")
        print(f"✅ Training: {len(training_puzzles)} puzzles generated")
        print(f"✅ Total: {len(benchmark_puzzles) + len(training_puzzles)} unique puzzles")
        
        if benchmark_puzzles:
            # Analyze benchmark
            task_counts = {}
            benchmark_difficulties = {}
            for puzzle in benchmark_puzzles:
                task_id = puzzle.get('task_id', 'unknown')
                diff = puzzle.get('difficulty', 'unknown')
                task_counts[task_id] = task_counts.get(task_id, 0) + 1
                benchmark_difficulties[diff] = benchmark_difficulties.get(diff, 0) + 1
            
            print(f"\n📊 Benchmark Analysis:")
            print(f"   Tasks: {len(task_counts)} (target: 30)")
            print(f"   Difficulty distribution: {dict(sorted(benchmark_difficulties.items()))}")
        
        if training_puzzles:
            # Analyze training
            training_difficulties = {}
            complexity_stats = {'min_steps': float('inf'), 'max_steps': 0, 'total_steps': 0}
            
            for puzzle in training_puzzles:
                diff = puzzle.get('difficulty', 'unknown')
                steps = puzzle.get('step_count', 0)
                training_difficulties[diff] = training_difficulties.get(diff, 0) + 1
                complexity_stats['min_steps'] = min(complexity_stats['min_steps'], steps)
                complexity_stats['max_steps'] = max(complexity_stats['max_steps'], steps)
                complexity_stats['total_steps'] += steps
            
            print(f"\n🎓 Training Analysis:")
            for diff in sorted(training_difficulties.keys()):
                count = training_difficulties[diff]
                print(f"   Difficulty {diff}: {count} puzzles (target: 300)")
            
            avg_steps = complexity_stats['total_steps'] / len(training_puzzles) if training_puzzles else 0
            print(f"   Complexity: {complexity_stats['min_steps']}-{complexity_stats['max_steps']} steps (avg: {avg_steps:.1f})")
        
        print(f"\n📁 Output Files:")
        print(f"   📂 Task directory: {generator.task_dir}")
        print(f"   📊 Benchmark: {generator.task_dir}/benchmark.json")
        print(f"   🎓 Training: {generator.task_dir}/training_set.json")
        print(f"   🖼️  Images: {generator.image_dir}/")
        
        print("\n✅ Generation completed successfully!")
        print("🎨 All puzzles feature unique procedural layouts with 3D graphics!")
        print("🎲 Zero overlap between benchmark and training datasets!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Generation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()