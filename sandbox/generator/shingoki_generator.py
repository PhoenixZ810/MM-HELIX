import os
import json
import random
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import List, Dict, Tuple, Set, Any, Optional
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import hashlib
from generator.base_generator import BaseGenerator

class ShingokiGenerator(BaseGenerator):
    def __init__(self, output_folder):
        super().__init__(output_folder)
        # 使用程序运行时的时间戳作为seed（与基类保持独立，不污染全局）
        self.seed = int(time.time())
        random.seed(self.seed)
        np.random.seed(self.seed)
        # Default fonts - adjust paths as needed
        # Defer font size decisions to draw time; set lightweight defaults
        # Optional user-provided fonts for "正规"效果
        self._font_candidates = [
            # Bold first for clarity
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
            "Arial Bold.ttf",
            "arialbd.ttf",
            # Regular fallbacks
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "Arial.ttf",
            "arial.ttf",
        ]
        try:
            self.regular_font = ImageFont.truetype(self._font_candidates[0], 20)
            self.small_font = ImageFont.truetype(self._font_candidates[0], 14)
        except Exception:
            try:
                self.regular_font = ImageFont.truetype(self._font_candidates[2], 20)
                self.small_font = ImageFont.truetype(self._font_candidates[2], 14)
            except Exception:
                self.regular_font = ImageFont.load_default()
                self.small_font = ImageFont.load_default()

        # Grid visualization parameters
        self.cell_size = 60
        self.margin = 30
        self.dot_radius = 4
        # Slightly larger default circle to accommodate clear numbers
        self.circle_radius = 22
        self.line_width = 3
        self.grid_width = 2
        self.loop_width = 6
        # Rendering scale factors
        # export_scale controls the final output size (in pixels)
        # render_scale controls supersampling for crisp anti-aliased output
        self.export_scale = 3
        self.render_scale = 3
        
        # Colors
        self.bg_color = (240, 245, 255)
        self.bg_gradient_top = (240, 245, 255)
        self.bg_gradient_bottom = (220, 232, 255)
        # All accents blue family
        self.grid_color = (184, 200, 238)
        self.dot_color = (36, 64, 142)
        self.line_color = (36, 64, 142)
        self.loop_color = (36, 64, 142)  # Deep blue loop
        self.white_circle_color = (255, 255, 255)
        self.black_circle_color = (18, 20, 22)
        self.text_color = (22, 24, 28)
        self.white_text_color = (255, 255, 255)
        # Numeric label color (override to blue for clarity)
        self.number_text_color = (36, 64, 142)
        # Numeric label stroke color (deep blue) for "all-blue" theme
        self.number_stroke_color = (20, 40, 110)
        self.border_color = (60, 100, 200)
        self.border_width = 3
        self.border_radius = 16

        # Effects
        self.loop_shadow_color = (0, 0, 0, 90)
        self.loop_shadow_blur = 2
        self.loop_shadow_offset = (0, 1)
        self.circle_shadow_color = (0, 0, 0, 80)
        self.circle_shadow_blur = 2
        
        # Track generated puzzles to avoid duplicates
        self.generated_puzzle_hashes = set()

    def _generate_initial_state(self, rows: int, cols: int, circles: Dict) -> str:
        """
        Generate the initial state of the puzzle as a simple 2D grid
        """
        grid = [['.' for _ in range(cols)] for _ in range(rows)]
        for pos_str, circle_info in circles.items():
            r, c = map(int, pos_str.split(','))
            circle_type = circle_info["type"]
            value = circle_info["value"]
            grid[r][c] = f'W{value}' if circle_type == "white" else f'B{value}'
        return json.dumps(grid, ensure_ascii=False)

    def _generate_grid_description(self, grid: List[List[str]], rows: int, cols: int) -> str:
        description = [
            f"Grid size: {rows} rows × {cols} columns",
            "Grid representation (row by row):"
        ]
        col_header = "  " + "".join(f"{c:2}" for c in range(cols))
        description.append(col_header)
        for r in range(rows):
            row_str = f"{r} " + "".join(f"{grid[r][c]:>2}" for c in range(cols))
            description.append(row_str)
        description.append("\nCircle locations and values:")
        circles_found = []
        for r in range(rows):
            for c in range(cols):
                cell = grid[r][c]
                if cell != '.':
                    if cell.startswith('W'):
                        circles_found.append(f"White circle at ({r},{c}) with value {cell[1:]}")
                    elif cell.startswith('B'):
                        circles_found.append(f"Black circle at ({r},{c}) with value {cell[1:]}")
        if circles_found:
            description.extend(circles_found)
        else:
            description.append("No circles found")
        return "\n".join(description)

    def save_to_json(self, puzzles, filename="puzzle.json"):
        json_ready_puzzles = []
        for puzzle in puzzles:
            json_puzzle = puzzle.copy()
            if "circles" in json_puzzle:
                circles_dict = {}
                for pos, info in json_puzzle["circles"].items():
                    if isinstance(pos, tuple):
                        pos_str = f"{pos[0]},{pos[1]}"
                        circles_dict[pos_str] = info
                    else:
                        circles_dict[pos] = info
                json_puzzle["circles"] = circles_dict
            json_ready_puzzles.append(json_puzzle)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_ready_puzzles, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(puzzles)} puzzles to {filename}")

    def _load_font(self, size: int, prefer_bold: bool = True) -> ImageFont.FreeTypeFont:
        """Best-effort font loader at requested size with bold preference for clarity."""
        # Try bold candidates first if requested
        bold_candidates = [p for p in self._font_candidates if 'Bold' in p or p.lower().endswith('bd.ttf')]
        regular_candidates = [p for p in self._font_candidates if p not in bold_candidates]
        candidates = (bold_candidates + regular_candidates) if prefer_bold else (regular_candidates + bold_candidates)
        for path in candidates:
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
        # Final fallback
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()
    
    def generate(self, num_cases, difficulty, output_folder=None):
        """
        生成指定数量和难度的Shingoki puzzle

        Args:
            num_cases: 要生成的问题数量
            difficulty: 问题难度级别 (1-5)
            output_folder: 输出文件夹路径，覆盖构造函数中设置的路径

        Returns:
            生成的问题列表
        """
        # 使用传入的output_folder或默认的output_folder
        if output_folder is None:
            output_folder = self.output_folder

        # Set up output directories
        self.output_dir = output_folder
        images_dir = os.path.join(output_folder, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Determine grid size based on difficulty (difficulty 1-5 maps to size 3-7)
        grid_size = difficulty + 2  # 1->3, 2->4, 3->5, 4->6, 5->7

        all_puzzles = []
        generated_count = 0

        while generated_count < num_cases:
            try:
                # Generate puzzle with unique index
                puzzle_index = f"shingoki_{grid_size}_{self.seed}_{generated_count}"

                # Generate single puzzle
                puzzle = self._generate_single_puzzle(
                    index=puzzle_index,
                    rows=grid_size,
                    cols=grid_size,
                    difficulty=difficulty,
                    seed=self.seed + generated_count  # Use base seed + counter for variation
                )

                # Create visualization and get image paths
                puzzle_image_path, solution_image_path = self.visualize(puzzle, images_dir)

                # Update puzzle with relative image paths
                puzzle["image"] = os.path.relpath(puzzle_image_path, output_folder)

                # Format final puzzle according to specification
                formatted_puzzle = {
                    "index": puzzle_index,
                    "category": "shingoki",
                    "image": puzzle["image"],
                    "question": puzzle["question"],
                    "question_language": puzzle["question_language"],
                    "answer": puzzle["answer"],
                    "initial_state": puzzle["initial_state"],
                    "difficulty": str(difficulty),
                    "cot": puzzle["cot"]
                }

                # Add optional CoT fields
                if "cot_step1_all" in puzzle:
                    formatted_puzzle["cot_step1_all"] = puzzle["cot_step1_all"]
                if "cot_step2_all" in puzzle:
                    formatted_puzzle["cot_step2_all"] = puzzle["cot_step2_all"]
                if "cot_step3_all" in puzzle:
                    formatted_puzzle["cot_step3_all"] = puzzle["cot_step3_all"]

                all_puzzles.append(formatted_puzzle)
                generated_count += 1

                print(f"Generated Shingoki puzzle {generated_count}/{num_cases}: {puzzle_index}")

            except Exception as e:
                print(f"Error generating puzzle {generated_count + 1}: {e}")
                continue

        # 使用通用的保存方法追加写入 annotations.json
        if all_puzzles:
            self.save_annotations(all_puzzles, output_folder)

        return all_puzzles

    def _get_difficulty_params(self, difficulty):
        """
        根据难度级别获取相应的参数配置。

        Args:
            difficulty: 难度级别（1-5）

        Returns:
            dict: 包含难度参数的字典
        """
        grid_size = difficulty + 2  # 1->3, 2->4, 3->5, 4->6, 5->7
        base_points = grid_size * grid_size

        if difficulty == 1:  # Easy
            return {
                'circle_count': max(3, int(base_points * 0.08)),  # 8% of grid points
                'min_loop_length': grid_size + grid_size,
                'grid_complexity': 1.0,
                'difficulty_name': 'easy',
                'grid_size': grid_size
            }
        elif difficulty == 2:  # Medium-Easy
            return {
                'circle_count': max(4, int(base_points * 0.12)),  # 12% of grid points
                'min_loop_length': int((grid_size + grid_size) * 1.2),
                'grid_complexity': 1.2,
                'difficulty_name': 'medium-easy',
                'grid_size': grid_size
            }
        elif difficulty == 3:  # Medium
            return {
                'circle_count': max(4, int(base_points * 0.15)),  # 15% of grid points
                'min_loop_length': int((grid_size + grid_size) * 1.5),
                'grid_complexity': 1.5,
                'difficulty_name': 'medium',
                'grid_size': grid_size
            }
        elif difficulty == 4:  # Medium-Hard
            return {
                'circle_count': max(5, int(base_points * 0.18)),  # 18% of grid points
                'min_loop_length': int((grid_size + grid_size) * 1.8),
                'grid_complexity': 1.8,
                'difficulty_name': 'medium-hard',
                'grid_size': grid_size
            }
        else:  # difficulty == 5, Hard
            return {
                'circle_count': max(6, int(base_points * 0.22)),  # 22% of grid points
                'min_loop_length': int((grid_size + grid_size) * 2.2),
                'grid_complexity': 2.0,
                'difficulty_name': 'hard',
                'grid_size': grid_size
            }

    def _get_puzzle_hash(self, puzzle_data: Dict) -> str:
        """Generate a hash for puzzle uniqueness checking"""
        # Create a string representation of key puzzle features
        rows = puzzle_data["rows"]
        cols = puzzle_data["cols"]
        circles = puzzle_data["circles"]
        
        # Sort circles by position for consistent hashing
        sorted_circles = []
        for pos_str, info in sorted(circles.items()):
            sorted_circles.append((pos_str, info["type"], info["value"]))
        
        # Create hash input string
        hash_input = f"{rows}x{cols}_{sorted_circles}"
        
        # Generate hash
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_enhanced_puzzle_hash(self, puzzle_data: Dict) -> str:
        """Generate an enhanced hash for better puzzle uniqueness checking"""
        # Create a more comprehensive string representation
        rows = puzzle_data["rows"]
        cols = puzzle_data["cols"]
        circles = puzzle_data["circles"]
        loop_segments = puzzle_data.get("loop_segments", [])
        
        # Sort circles by position for consistent hashing
        sorted_circles = []
        for pos_str, info in sorted(circles.items()):
            sorted_circles.append((pos_str, info["type"], info["value"]))
        
        # Sort loop segments for consistent hashing
        sorted_segments = []
        for segment in loop_segments:
            # Normalize segment direction (smaller coordinate first)
            p1, p2 = segment
            if p1 < p2:
                sorted_segments.append((p1, p2))
            else:
                sorted_segments.append((p2, p1))
        sorted_segments.sort()
        
        # Create comprehensive hash input including grid layout, circles, and solution
        hash_input = f"{rows}x{cols}_{sorted_circles}_{sorted_segments}"
        
        # Use SHA-256 for better collision resistance with large dataset
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def _place_circles(self, rows: int, cols: int, circle_count: int, 
                      loop_segments: List[Tuple], loop_points: Set[Tuple], seed: int = None) -> Dict:
        """Place circles on the grid and calculate their values"""
        circles = {}
        
        # Use seed for reproducible circle placement
        if seed is not None:
            random.seed(seed + 100)  # Offset to ensure different randomness
        
        # Identify straight and turning points
        straight_points = set()
        turning_points = set()
        
        # Create a mapping of points to their connected segments
        point_segments = defaultdict(list)
        for segment in loop_segments:
            p1, p2 = segment
            point_segments[p1].append(segment)
            point_segments[p2].append(segment)
        
        # Determine straight and turning points
        for point in loop_points:
            segments = point_segments[point]
            if len(segments) != 2:  # Should always be 2 for a valid loop
                continue
                
            # Get the directions of the two segments
            p1, p2 = segments[0]
            p3, p4 = segments[1]
            
            # Ensure point is one end of each segment
            dir1 = None
            if point == p1:
                dir1 = (p2[0] - p1[0], p2[1] - p1[1])
            else:
                dir1 = (p1[0] - p2[0], p1[1] - p2[1])
                
            dir2 = None
            if point == p3:
                dir2 = (p4[0] - p3[0], p4[1] - p3[1])
            else:
                dir2 = (p3[0] - p4[0], p3[1] - p4[1])
            
            # Check if it's a straight or turning point
            if dir1[0] == -dir2[0] and dir1[1] == -dir2[1]:
                straight_points.add(point)
            else:
                turning_points.add(point)
        
        # Calculate line lengths for each point using the improved method
        line_lengths = {}
        for point in loop_points:
            if len(point_segments[point]) != 2:  # Skip if not part of exactly 2 segments
                continue
                
            # Calculate the lengths of the straight line segments
            length1, length2 = self._calculate_line_lengths(point, loop_segments, rows, cols)
            total_length = length1 + length2
            
            # Only keep points with valid lengths
            if total_length > 0:
                line_lengths[point] = total_length
        
        # Place circles based on calculated values
        available_points = list(loop_points)
        random.shuffle(available_points)
        
        placed = 0
        for point in available_points:
            if placed >= circle_count:
                break
                
            if point in line_lengths:
                circle_type = "white" if point in straight_points else "black"
                value = line_lengths[point]
                
                # Double-check that the values match the circle type
                length1, length2 = self._calculate_line_lengths(point, loop_segments, rows, cols)
                if circle_type == "white" and (length1 == 0 or length2 == 0):
                    # Skip white circles that don't have two valid straight segments
                    continue
                if circle_type == "black" and (length1 == 0 or length2 == 0):
                    # Skip black circles that don't have two valid turning segments
                    continue
                
                # Don't place circles with value 0
                if value > 0:
                    circles[point] = {"type": circle_type, "value": value}
                    placed += 1
        
        return circles
    

    
    def _generate_single_puzzle(self, index: str, rows: int, cols: int, difficulty: int, seed: int = None) -> Dict:
        """Generate a single Shingoki puzzle with numeric difficulty (1-5) without recursive retries"""
        # Prepare parameters based on difficulty
        params = self._get_difficulty_params(difficulty)
        circle_count = params['circle_count']
        min_loop_length = params['min_loop_length']
        difficulty_name = params['difficulty_name']

        # Establish a base seed and bounded retry loop to avoid deep recursion
        base_seed = seed if seed is not None else random.randrange(2**32)
        max_attempts = 2000

        for attempt in range(max_attempts):
            # Derive a deterministic attempt seed to vary generation while staying reproducible
            attempt_seed = (base_seed + attempt * 10007) % (2**32)

            # Seed RNGs for this attempt
            random.seed(attempt_seed)
            np.random.seed(attempt_seed)

            # Generate a valid loop and place circles using the attempt seed
            loop_segments, loop_points = self._generate_valid_loop(rows, cols, min_loop_length, attempt_seed)
            circles = self._place_circles(rows, cols, circle_count, loop_segments, loop_points, attempt_seed)

            # Convert tuple keys in circles to string representation
            circles_dict = {}
            for pos, info in circles.items():
                pos_str = f"{pos[0]},{pos[1]}"
                circles_dict[pos_str] = info

            # Generate initial state (puzzle setup without solution)
            initial_state = self._generate_initial_state(rows, cols, circles_dict)

            # Create the puzzle data with all fields (some will be filtered out in JSON)
            puzzle = {
                "index": index,
                "category": "shingoki",
                "rows": rows,
                "cols": cols,
                "difficulty": difficulty,
                "difficulty_name": difficulty_name,
                "circles": circles_dict,
                "loop_segments": loop_segments,
                "initial_state": initial_state,
                "step_count": len(loop_segments),
                "answer": self._format_answer(loop_segments)
            }

            # Check for uniqueness using enhanced hash; only accept unique puzzles
            puzzle_hash = self._get_enhanced_puzzle_hash(puzzle)
            if puzzle_hash in self.generated_puzzle_hashes:
                # Duplicate, try next attempt
                if attempt % 100 == 0:
                    print("Warning: Duplicate puzzle encountered; retrying with a new seed...")
                continue

            # Validate the puzzle before accepting it
            if not self.validate_puzzle(puzzle):
                if attempt % 100 == 0:
                    print("Warning: Generated an invalid puzzle; retrying with a new seed...")
                continue

            # Mark hash as used only after successful validation
            self.generated_puzzle_hashes.add(puzzle_hash)

            # Generate textual question and CoT
            puzzle["question"] = self._generate_question_with_image(puzzle)
            puzzle["question_language"] = self._generate_text_question(puzzle)
            puzzle["cot"] = self._generate_cot(puzzle)

            return puzzle

        # If we reach here, puzzle generation failed repeatedly
        raise RuntimeError(f"Failed to generate a unique, valid puzzle after {max_attempts} attempts (base_seed={base_seed})")

    def _generate_initial_state_deprecated(self, rows: int, cols: int, circles: Dict) -> str:
        """
        DEPRECATED: Generate the initial state of the puzzle (grid with circles but no solution lines)
        This method is kept for backward compatibility but not used in main generation.
        
        Args:
            rows: Number of rows in the grid
            cols: Number of columns in the grid  
            circles: Dictionary of circles with their positions and properties
            
        Returns:
            JSON string representing the initial puzzle state
        """
        # Create empty grid
        grid = [['.' for _ in range(cols)] for _ in range(rows)]
        
        # Place circles on the grid
        for pos_str, circle_info in circles.items():
            r, c = map(int, pos_str.split(','))
            circle_type = circle_info["type"]
            value = circle_info["value"]
            
            # Use different symbols for different circle types
            if circle_type == "white":
                grid[r][c] = f'W{value}'  # White circle with value
            else:  # black
                grid[r][c] = f'B{value}'  # Black circle with value
        
        # Create initial state dictionary
        initial_state = {
            "grid": grid,
            "rows": rows,
            "cols": cols,
            "circles": circles,
            "connected_segments": [],  # No segments connected initially
            "rules": {
                "single_loop": "Must form exactly one continuous loop",
                "white_circles": "Must be passed through in straight lines",
                "black_circles": "Must be turned upon (change direction)",
                "circle_values": "Sum of lengths of two line segments from each circle"
            }
        }
        
        return json.dumps(initial_state, ensure_ascii=False)
    
    def _generate_valid_loop(self, rows: int, cols: int, min_length: int, seed: int = None) -> Tuple[List[Tuple], Set[Tuple]]:
        """Generate a valid loop path on the grid"""
        # Use seed for reproducible loop generation
        if seed is not None:
            random.seed(seed + 50)  # Offset to ensure different randomness
        
        max_attempts = 100
        best_loop = None
        best_length = 0
        
        for attempt in range(max_attempts):
            # Start at a position determined by seed and attempt
            if seed is not None:
                start_r = (seed + attempt) % rows
                start_c = (seed + attempt * 7) % cols  # Use prime number for better distribution
            else:
                start_r = random.randint(0, rows - 1)
                start_c = random.randint(0, cols - 1)
            current = (start_r, start_c)
            
            # Keep track of visited points and the path
            visited = {current}
            path = []
            
            # Random walk
            stuck = False
            while not stuck:
                # Find valid neighbors
                neighbors = []
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = current[0] + dr, current[1] + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                        neighbors.append((nr, nc))
                
                if not neighbors:
                    # Try to close the loop
                    if len(path) >= min_length:
                        # Check if we can connect back to start
                        sr, sc = start_r, start_c
                        cr, cc = current
                        if abs(sr - cr) + abs(sc - cc) == 1:  # Adjacent to start
                            path.append(((cr, cc), (sr, sc)))
                            break  # Successfully closed the loop
                    
                    # Can't continue or close
                    stuck = True
                else:
                    # Move to a neighbor (with seed-based preference)
                    if seed is not None:
                        # Use seed to determine neighbor selection for reproducibility
                        next_idx = (seed + len(path)) % len(neighbors)
                        next_point = neighbors[next_idx]
                    else:
                        next_point = random.choice(neighbors)
                    path.append((current, next_point))
                    visited.add(next_point)
                    current = next_point
            
            # If we have a valid loop that meets our length requirements
            if not stuck and len(path) >= min_length:
                if len(path) > best_length:
                    best_loop = path
                    best_length = len(path)
        
        if best_loop is None:
            # Fallback: create a simple rectangular loop with seed-based variation
            best_loop = []
            if seed is not None and seed % 2 == 0:
                # Clockwise rectangle
                for c in range(cols - 1):
                    best_loop.append(((0, c), (0, c + 1)))
                for r in range(rows - 1):
                    best_loop.append(((r, cols - 1), (r + 1, cols - 1)))
                for c in range(cols - 1, 0, -1):
                    best_loop.append(((rows - 1, c), (rows - 1, c - 1)))
                for r in range(rows - 1, 0, -1):
                    best_loop.append(((r, 0), (r - 1, 0)))
            else:
                # Counter-clockwise rectangle or different pattern
                for r in range(rows - 1):
                    best_loop.append(((r, 0), (r + 1, 0)))
                for c in range(cols - 1):
                    best_loop.append(((rows - 1, c), (rows - 1, c + 1)))
                for r in range(rows - 1, 0, -1):
                    best_loop.append(((r, cols - 1), (r - 1, cols - 1)))
                for c in range(cols - 1, 0, -1):
                    best_loop.append(((0, c), (0, c - 1)))
        
        # Extract all points on the loop
        loop_points = set()
        for segment in best_loop:
            loop_points.add(segment[0])
            loop_points.add(segment[1])
        
        return best_loop, loop_points
    
    def _calculate_line_lengths(self, point, segments, rows, cols):
        """
        Calculate the lengths of straight line segments from a point
        
        Args:
            point: The (row, col) coordinates of the point
            segments: The list of all line segments in the loop
            rows: Grid row count
            cols: Grid column count
            
        Returns:
            A tuple (length1, length2) of the two line segment lengths
        """
        # Create a graph of connections
        graph = {}
        for p1, p2 in segments:
            if p1 not in graph:
                graph[p1] = []
            if p2 not in graph:
                graph[p2] = []
            graph[p1].append(p2)
            graph[p2].append(p1)
        
        # Ensure the point is in the graph and has exactly 2 connections
        if point not in graph or len(graph[point]) != 2:
            return (0, 0)
        
        # Get the two connected points
        neighbors = graph[point]
        
        # Calculate lengths in both directions
        lengths = []
        for neighbor in neighbors:
            # Start measuring from the neighbor
            curr = point
            next_point = neighbor
            length = 1  # Initial segment length (from point to neighbor)
            
            # Get the initial direction
            direction = (next_point[0] - curr[0], next_point[1] - curr[1])
            
            # Follow this direction until we hit a turn
            visited = {curr}
            while True:
                # Add current point to visited
                visited.add(next_point)
                
                # Look for next point in the same direction
                curr_r, curr_c = next_point
                next_r, next_c = curr_r + direction[0], curr_c + direction[1]
                next_candidate = (next_r, next_c)
                
                # Check if the next candidate is valid and connected
                if (0 <= next_r < rows and 0 <= next_c < cols and 
                    next_candidate in graph and 
                    next_candidate not in visited and
                    (next_point, next_candidate) in segments or (next_candidate, next_point) in segments):
                    # Continue in the same direction
                    curr = next_point
                    next_point = next_candidate
                    length += 1
                else:
                    # Hit a turn or dead end
                    break
            
            lengths.append(length)
        
        # If we somehow only got one length, add a zero
        if len(lengths) == 1:
            lengths.append(0)
            
        return tuple(lengths)
    
    def _format_answer(self, loop_segments: List[Tuple]) -> str:
        """Format the loop as a string answer"""
        answer_parts = []
        
        for segment in loop_segments:
            (r1, c1), (r2, c2) = segment
            answer_parts.append(f"({r1},{c1})-({r2},{c2})")
        
        return " ".join(answer_parts)
    
    def _generate_text_question(self, puzzle: Dict) -> str:
        """Generate pure text representation of the puzzle without any image references"""
        rows = puzzle["rows"]
        cols = puzzle["cols"]
        circles = puzzle["circles"]
        difficulty = puzzle["difficulty"]
        difficulty_name = puzzle["difficulty_name"]
        
        text = [
            "\nYou are given a Shingoki puzzle. This is a logic puzzle where you need to draw a single continuous loop on a grid.",
            "\nGame Rules:",
            "1. Draw exactly one continuous loop without crossings or branches",
            "2. The loop must eventually return to its starting point",
            "3. White circles must be passed through in a straight line (no turning at white circles)",
            "4. Black circles must be turned upon (the path must change direction at black circles)",
            "5. Each circle has a number that represents the sum of the lengths of the two straight line segments extending from that circle",
            "\nGrid Information:",
            f"- Grid size: {rows} rows × {cols} columns",
            "- (0,0) is the top-left corner",
            "- Row numbers increase downward, column numbers increase rightward",
            "- The loop connects adjacent grid points (no diagonal connections)",
        ]
        
        # # Sort circles by position for consistent presentation
        # sorted_circles = []
        # for pos_str, circle_info in circles.items():
        #     if isinstance(pos_str, tuple):
        #         r, c = pos_str
        #     else:
        #         r, c = map(int, pos_str.split(','))
        #     sorted_circles.append((r, c, circle_info))
        
        # sorted_circles.sort(key=lambda x: (x[0], x[1]))  # Sort by row, then column
        
        # for r, c, circle_info in sorted_circles:
        #     circle_type = circle_info["type"]
        #     value = circle_info["value"]
        #     constraint_desc = "must be passed through in a straight line" if circle_type == "white" else "must be turned upon (change direction)"
        #     text.append(f"- {circle_type.capitalize()} circle at position ({r},{c}) with value {value} - {constraint_desc}")
        
        text.extend([
            "\nObjective:",
            "Find the single continuous loop that:",
            "- Passes through all circles according to their type constraints",
            "- Satisfies all circle value constraints (sum of line segment lengths)",
            "- Forms a closed loop without crossings or branches",
            "\nOutput Format Requirements:",
            "- Represent your solution as a sequence of connected line segments",
            "- Each segment connects two adjacent grid points: (r1,c1)-(r2,c2)",
            "- Adjacent points differ by exactly 1 in either row or column (no diagonals)",
            "- List all segments separated by spaces in one continuous string",
            "- The segments should form a complete closed loop",
            "- Example format: (0,0)-(0,1) (0,1)-(1,1) (1,1)-(1,0) (1,0)-(0,0)",
        ])
        
        # Add a visual grid representation for better understanding
        text.append("\nGrid Layout:")
        grid_repr = self._generate_text_grid_representation(rows, cols, circles)
        text.append(grid_repr)
        
        return "\n".join(text)
    
    def _generate_text_grid_representation(self, rows: int, cols: int, circles: Dict) -> str:
        """Generate a text-based visual representation of the grid"""
        # Create a text grid
        grid = [['.' for _ in range(cols)] for _ in range(rows)]
        
        # Place circles on the grid
        for pos_str, circle_info in circles.items():
            if isinstance(pos_str, tuple):
                r, c = pos_str
            else:
                r, c = map(int, pos_str.split(','))
            
            circle_type = circle_info["type"]
            value = circle_info["value"]
            
            # Use W for white circles, B for black circles, followed by value
            symbol = f"W{value}" if circle_type == "white" else f"B{value}"
            if r < rows and c < cols:
                grid[r][c] = symbol
        
        # Build the text representation
        result = []
        
        # Add column headers
        header = "   " + "".join(f"{c:>4}" for c in range(cols))
        result.append(header)
        
        # Add rows with row numbers
        for r in range(rows):
            row_str = f"{r:>2} "
            for c in range(cols):
                cell = grid[r][c]
                row_str += f"{cell:>4}"
            result.append(row_str)
        
        result.append("\nLegend: W# = White circle with value #, B# = Black circle with value #, . = Empty cell")
        
        return "\n".join(result)
    
    def _generate_question_with_image(self, puzzle: Dict) -> str:
        """Generate question that refers to the image"""
        rows = puzzle["rows"]
        cols = puzzle["cols"]
        difficulty = puzzle["difficulty"]
        
        question = f"""
You are given a Shingoki puzzle. This is a logic puzzle where you need to draw a single continuous loop on a grid.\n\n### Game Rules:\n1. Draw exactly one continuous loop without crossings or branches\n2. The loop must eventually return to its starting point\n3. White circles must be passed through in a straight line (no turning at white circles)\n4. Black circles must be turned upon (the path must change direction at black circles)\n5. Each circle has a number that represents the sum of the lengths of the two straight line segments extending from that circle\n\n\n### Coordinate system: \n- (0,0) is the top-left corner\n- Row numbers increase downward, column numbers increase rightward\n- The loop connects adjacent grid points (no diagonal connections)\n\n### Objective:\nFind the single continuous loop that:\n- Passes through all circles according to their type constraints\n- Satisfies all circle value constraints (sum of line segment lengths)\n- Forms a closed loop without crossings or branches\n\n### Output Format:\n- Represent your solution as a sequence of connected line segments\n- Each segment connects two adjacent grid points: (r1,c1)-(r2,c2)\n- Adjacent points differ by exactly 1 in either row or column (no diagonals)\n- List all segments separated by spaces in one continuous string\n- The segments should form a complete closed loop\n- Example format: (0,0)-(0,1) (0,1)-(1,1) (1,1)-(1,0) (1,0)-(0,0)\n
"""
        
        return question
    
    def _generate_cot(self, puzzle: Dict) -> str:
        """Generate chain-of-thought reasoning process following the 4-step enhanced format"""
        rows = puzzle["rows"]
        cols = puzzle["cols"]
        circles = puzzle["circles"]
        loop_segments = puzzle["loop_segments"]
        answer = puzzle["answer"]
        
        # Step 1: Understanding the puzzle rules and objectives (详细且完整)
        step1_content = [
            "Let me solve this Shingoki puzzle using a systematic approach.",
            "",
            "### Step 1: Understanding the Game Rules and Mechanics",
            "",
            f"This is a {rows}×{cols} Shingoki puzzle, which is a logic puzzle involving loop construction on a grid.",
            "",
            "**Core Rules:**",
            "1. **Single Loop Constraint**: Draw exactly one continuous closed loop on the grid",
            "   - The loop must return to its starting point",
            "   - No crossings, branches, or multiple loops allowed",
            "   - Loop segments connect adjacent grid intersection points (no diagonals)",
            "",
            "2. **White Circle Rule**: White circles must be passed through in straight lines",
            "   - When the loop passes through a white circle, it cannot change direction",
            "   - The path must continue straight through the circle",
            "   - This creates a constraint on loop topology",
            "",
            "3. **Black Circle Rule**: Black circles must be turning points",
            "   - When the loop reaches a black circle, it must change direction (turn)",
            "   - The path cannot continue straight through a black circle",
            "   - This forces specific geometric configurations",
            "",
            "4. **Value Constraint**: Each circle's number represents a sum",
            "   - The number equals the total length of the two line segments extending from that circle",
            "   - For white circles: sum of the two straight line segments",
            "   - For black circles: sum of the two segments forming the turn",
            "   - Segment length is measured in grid units (number of edges)",
            "",
            "**Solving Strategy:**",
            "I need to find the unique loop configuration that simultaneously satisfies:",
            "- All topological constraints (single closed loop)",
            "- All circle type constraints (straight vs. turning)",
            "- All numerical value constraints (segment length sums)",
            "",
            "Understanding these rules is crucial for systematic exploration."
        ]
        
        # Step 2: Careful image reading and precise initial state extraction (详细且完整)
        step2_content = [
            "",
            "### Step 2: Careful Image Reading and Initial State Analysis",
            "",
            "**Image Examination Process:**",
            f"Looking at the provided image, I can see a {rows}×{cols} grid with intersection points where the loop can be drawn.",
            "",
            "**Grid Structure:**",
            f"- Grid dimensions: {rows} rows × {cols} columns",
            f"- Total intersection points: {rows*cols}",
            "- Coordinate system: (0,0) at top-left, rows increase downward, columns increase rightward",
            "- Each grid cell represents the space between four adjacent intersection points",
            "",
            "**Circle Detection and Analysis:**",
            "Scanning the grid systematically, I identify the following circles:"
        ]
        
        # Generate text grid representation for reference
        grid_repr = self._generate_text_grid_representation(rows, cols, circles)
        step2_content.extend([
            "",
            "**Initial State Representation:**",
            grid_repr,
            "",
            "**Detailed Circle Analysis:**"
        ])
        
        # Analyze circles with detailed constraints
        sorted_circles = []
        for pos_str, circle_info in circles.items():
            r, c = map(int, pos_str.split(','))
            sorted_circles.append((r, c, circle_info))
        sorted_circles.sort(key=lambda x: (x[0], x[1]))  # Sort by position for systematic analysis
        
        for r, c, circle_info in sorted_circles:
            circle_type = circle_info["type"]
            value = circle_info["value"]
            
            step2_content.append(f"\n**Circle at position ({r},{c}):**")
            step2_content.append(f"- Type: {circle_type.capitalize()} circle")
            step2_content.append(f"- Value: {value}")
            
            if circle_type == "white":
                step2_content.extend([
                    f"- Constraint: Must be passed through in a straight line",
                    f"- Value interpretation: The two straight line segments extending from this circle must sum to {value}",
                    f"- Possible orientations: Horizontal or vertical straight line through ({r},{c})"
                ])
                
                # Calculate maximum possible extensions
                max_left = c
                max_right = cols - 1 - c
                max_up = r
                max_down = rows - 1 - r
                step2_content.extend([
                    f"- Maximum extensions: Left={max_left}, Right={max_right}, Up={max_up}, Down={max_down}",
                    f"- This constrains possible segment combinations that sum to {value}"
                ])
                
            else:  # black circle
                step2_content.extend([
                    f"- Constraint: Must be a turning point (path changes direction)",
                    f"- Value interpretation: The two segments forming the L-shaped turn must sum to {value}",
                    f"- Possible turn directions: Up-Right, Right-Down, Down-Left, Left-Up"
                ])
                
                # Calculate maximum possible extensions for each turn direction
                step2_content.append(f"- Maximum L-turn combinations:")
                step2_content.append(f"  • Up-Right: max {r}+{cols-1-c} = {r + cols-1-c}")
                step2_content.append(f"  • Right-Down: max {cols-1-c}+{rows-1-r} = {cols-1-c + rows-1-r}")
                step2_content.append(f"  • Down-Left: max {rows-1-r}+{c} = {rows-1-r + c}")
                step2_content.append(f"  • Left-Up: max {c}+{r} = {c + r}")
        
        step2_content.extend([
            "",
            "**Initial State Verification:**",
            "- Grid properly parsed with all circle positions and values identified",
            "- Circle constraints understood and maximum possible segment lengths calculated",
            "- Ready to proceed with systematic exploration of valid loop configurations",
            "",
            "**State Reflection:**",
            f"The puzzle presents {len(circles)} circles on a {rows}×{cols} grid. Each circle significantly constrains",
            "the possible loop paths, and the combination of all constraints should lead to a unique solution."
        ])
        
        # Step 3: Detailed reasoning process with systematic exploration (丰富的探索推理过程)
        step3_content = [
            "",
            "### Step 3: Systematic Exploration and Reasoning Process",
            "",
            "**Strategic Approach:**",
            "I'll use a constraint-satisfaction approach, starting with the most constrained circles",
            "and propagating decisions to find the unique valid solution.",
            "",
            "**Phase 1: Constraint Analysis and Prioritization**"
        ]
        
        # Enhanced exploration with detailed reasoning
        if circles:
            # Find most constrained circles for strategic analysis
            circle_analysis = []
            for pos_str, circle_info in circles.items():
                r, c = map(int, pos_str.split(','))
                value = circle_info["value"]
                circle_type = circle_info["type"]
                # Calculate constraint strength based on position and value
                edge_distance = min(r, rows-r-1, c, cols-c-1)
                max_reach = edge_distance * 2 + 2
                constraint_ratio = value / max_reach if max_reach > 0 else 1
                circle_analysis.append((constraint_ratio, r, c, value, circle_type))
            
            # Sort by constraint level (most constrained first)
            circle_analysis.sort(reverse=True)
            
            step3_content.append("Ranking circles by constraint level (most constrained first):")
            for i, (ratio, r, c, value, circle_type) in enumerate(circle_analysis):
                constraint_level = "HIGH" if ratio > 0.7 else "MEDIUM" if ratio > 0.4 else "LOW"
                step3_content.append(f"{i+1}. {circle_type.capitalize()} circle at ({r},{c}), value {value} - {constraint_level} constraint")
            
            step3_content.append("\n**Phase 2: Detailed Circle-by-Circle Analysis**")
            
            # Detailed exploration of each circle
            for i, (ratio, r, c, value, circle_type) in enumerate(circle_analysis):
                step3_content.append(f"\n**Analyzing Circle {i+1}: {circle_type.capitalize()} at ({r},{c}) with value {value}**")
                
                if circle_type == "white":
                    step3_content.extend([
                        "This white circle must have a straight line passing through it.",
                        f"Need to find segment combinations that sum to {value}.",
                        "",
                        "**Horizontal Orientation Analysis:**"
                    ])
                    
                    # Detailed horizontal analysis
                    max_left = c
                    max_right = cols - 1 - c
                    step3_content.append(f"- Maximum leftward extension: {max_left} units")
                    step3_content.append(f"- Maximum rightward extension: {max_right} units")
                    
                    # Generate possible combinations
                    h_combinations = []
                    for left in range(min(max_left, value) + 1):
                        right = value - left
                        if right <= max_right:
                            h_combinations.append((left, right))
                    
                    if h_combinations:
                        step3_content.append("- Valid horizontal combinations (left + right = total):")
                        for left, right in h_combinations[:3]:  # Show first 3
                            step3_content.append(f"  • {left} + {right} = {value}")
                        if len(h_combinations) > 3:
                            step3_content.append(f"  • ... and {len(h_combinations)-3} more combinations")
                    else:
                        step3_content.append("- No valid horizontal combinations possible")
                    
                    step3_content.append("\n**Vertical Orientation Analysis:**")
                    max_up = r
                    max_down = rows - 1 - r
                    step3_content.append(f"- Maximum upward extension: {max_up} units")
                    step3_content.append(f"- Maximum downward extension: {max_down} units")
                    
                    # Generate possible combinations
                    v_combinations = []
                    for up in range(min(max_up, value) + 1):
                        down = value - up
                        if down <= max_down:
                            v_combinations.append((up, down))
                    
                    if v_combinations:
                        step3_content.append("- Valid vertical combinations (up + down = total):")
                        for up, down in v_combinations[:3]:  # Show first 3
                            step3_content.append(f"  • {up} + {down} = {value}")
                        if len(v_combinations) > 3:
                            step3_content.append(f"  • ... and {len(v_combinations)-3} more combinations")
                    else:
                        step3_content.append("- No valid vertical combinations possible")
                    
                    # Decision process
                    total_options = len(h_combinations) + len(v_combinations)
                    step3_content.append(f"\n**Decision Space:** {total_options} total valid orientations to explore")
                    
                else:  # black circle
                    step3_content.extend([
                        "This black circle must be a turning point (L-shaped path).",
                        f"Need to find two perpendicular segments that sum to {value}.",
                        "",
                        "**Turn Direction Analysis:**"
                    ])
                    
                    # Analyze each possible turn direction
                    turn_directions = [
                        ("Up-Right", r, cols-1-c, "upward", "rightward"),
                        ("Right-Down", cols-1-c, rows-1-r, "rightward", "downward"),
                        ("Down-Left", rows-1-r, c, "downward", "leftward"),
                        ("Left-Up", c, r, "leftward", "upward")
                    ]
                    
                    valid_turns = []
                    for turn_name, max1, max2, dir1, dir2 in turn_directions:
                        step3_content.append(f"\n- **{turn_name} Turn:**")
                        step3_content.append(f"  Maximum {dir1}: {max1} units")
                        step3_content.append(f"  Maximum {dir2}: {max2} units")
                        
                        # Find valid combinations for this turn
                        combinations = []
                        for seg1 in range(1, min(max1, value) + 1):
                            seg2 = value - seg1
                            if 1 <= seg2 <= max2:
                                combinations.append((seg1, seg2))
                        
                        if combinations:
                            step3_content.append(f"  Valid combinations ({dir1} + {dir2} = total):")
                            for seg1, seg2 in combinations[:2]:  # Show first 2
                                step3_content.append(f"    • {seg1} + {seg2} = {value}")
                            if len(combinations) > 2:
                                step3_content.append(f"    • ... and {len(combinations)-2} more")
                            valid_turns.extend([(turn_name, combo) for combo in combinations])
                        else:
                            step3_content.append("  No valid combinations possible")
                    
                    step3_content.append(f"\n**Decision Space:** {len(valid_turns)} total valid turn configurations to explore")
                
                # Add constraint propagation insight
                if i < len(circle_analysis) - 1:
                    step3_content.extend([
                        "",
                        "**Constraint Propagation Impact:**",
                        "Each choice here will constrain the remaining circles. Need to check compatibility."
                    ])
            
            # Phase 3: Integration and backtracking
            step3_content.extend([
                "",
                "**Phase 3: Solution Construction with Backtracking**",
                "",
                "Using the constraint analysis above, I'll systematically build the solution:",
                "",
                "**Backtracking Algorithm:**",
                "1. Start with the most constrained circle (highest priority)",
                "2. Try each valid configuration for that circle",
                "3. For each choice, propagate constraints to remaining circles",
                "4. If all remaining circles can be satisfied → potential solution found",
                "5. If conflict detected → backtrack and try next configuration",
                "6. Continue until unique solution is found",
                "",
                "**Step-by-Step Construction:**"
            ])
            
            # Simulate the construction process
            if len(circle_analysis) >= 1:
                first_circle = circle_analysis[0]
                r1, c1, val1, type1 = first_circle[1], first_circle[2], first_circle[3], first_circle[4]
                
                step3_content.extend([
                    f"Starting with {type1} circle at ({r1},{c1}) value {val1}:",
                    f"- Try first valid configuration for this circle",
                    f"- This locks in specific grid edges as part of the loop",
                    f"- Check if remaining circles can still be satisfied..."
                ])
                
                if len(circle_analysis) >= 2:
                    second_circle = circle_analysis[1]
                    r2, c2, val2, type2 = second_circle[1], second_circle[2], second_circle[3], second_circle[4]
                    
                    step3_content.extend([
                        f"",
                        f"Next, handle {type2} circle at ({r2},{c2}) value {val2}:",
                        f"- Check which configurations are still possible given first circle's choice",
                        f"- If compatible configuration found → continue",
                        f"- If no compatible configuration → backtrack to first circle"
                    ])
            
            step3_content.extend([
                "",
                "**Loop Closure Verification:**",
                "As the path is constructed, continuously verify:",
                "- All segments connect properly (no gaps)",
                "- Path forms exactly one closed loop",
                "- No grid edges are used twice",
                "- All circle constraints remain satisfied",
                "",
                "**Solution Discovery:**",
                "After systematic exploration with backtracking, the unique valid solution is found.",
                "This solution satisfies all constraints simultaneously:"
            ])
            
            # Add verification of actual solution
            step3_content.extend([
                "- All white circles are on straight line segments",
                "- All black circles are at turning points", 
                "- All circle values match the sum of their extending segments",
                "- The path forms exactly one continuous closed loop"
            ])
        
        # Add the final answer
        step3_content.extend([
            "",
            f"**Final Answer Found:**",
            f"The unique solution path is: {answer}"
        ])
        
        # Step 4: Solution validation and refinement (简练但完整)
        step4_content = [
            "",
            "### Step 4: Solution validation and refinement",
            "",
            "Validating the solution against all constraints:"
        ]
        
        # Verify each circle constraint
        for pos_str, circle_info in circles.items():
            r, c = map(int, pos_str.split(','))
            circle_type = circle_info["type"]
            value = circle_info["value"]
            
            # Calculate actual lengths for verification
            point = (r, c)
            length1, length2 = self._calculate_line_lengths(point, loop_segments, rows, cols)
            total_length = length1 + length2
            
            constraint_met = "✓" if total_length == value else "✗"
            step4_content.append(f"- Circle ({r},{c}): {circle_type} rule + value {value} → actual {total_length} {constraint_met}")
        
        step4_content.extend([
            "",
            f"Loop integrity: {len(loop_segments)} segments form one continuous closed path.",
            "All constraints satisfied. Solution verified.",
            "",
            f"Final answer: {answer}"
        ])
        
        # Combine all steps
        all_steps = step1_content + step2_content + step3_content + step4_content
        full_cot = "\n".join(all_steps)
        
        # Store incremental CoT steps in puzzle data
        self._store_cot_steps(puzzle, step1_content, step2_content, step3_content, step4_content)
        
        return full_cot
    
    def _store_cot_steps(self, puzzle: Dict, step1: List[str], step2: List[str], step3: List[str], step4: List[str]) -> None:
        """Store incremental CoT steps with part and all versions"""
        
        def split_text_smartly(text: str) -> str:
            """Split text at roughly half point, but at a word boundary"""
            words = text.split()
            if len(words) <= 2:
                return text
            
            half_words = len(words) // 2
            return " ".join(words[:half_words])
        
        # Step 1 (only step 1)
        step1_text = "\n".join(step1)
        puzzle["cot_step1_part"] = split_text_smartly(step1_text)
        puzzle["cot_step1_all"] = step1_text
        
        # Step 2 (cumulative: step 1 + step 2)
        step1_2_text = "\n".join(step1 + step2)
        puzzle["cot_step2_part"] = split_text_smartly(step1_2_text)
        puzzle["cot_step2_all"] = step1_2_text
        
        # Step 3 (cumulative: step 1 + step 2 + step 3)
        step1_3_text = "\n".join(step1 + step2 + step3)
        puzzle["cot_step3_part"] = split_text_smartly(step1_3_text)
        puzzle["cot_step3_all"] = step1_3_text
        
        # Note: 根据要求，只需要step1到step3的增量保存
    

    
    def _generate_solution_steps(self, puzzle: Dict) -> str:
        """Generate step-by-step solution with OCR process and exploration"""
        circles = puzzle["circles"]
        rows = puzzle["rows"]
        cols = puzzle["cols"]
        difficulty = puzzle["difficulty"]
        difficulty_name = puzzle["difficulty_name"]
        
        # Simulate OCR process
        ocr_steps = [
            "## OCR Process",
            f"I can see a {rows}x{cols} Shingoki puzzle grid (Difficulty Level: {difficulty} - {difficulty_name.title()}) with the following circles:"
        ]
        
        for pos_str, circle_info in circles.items():
            # Handle both tuple and string positions
            if isinstance(pos_str, tuple):
                r, c = pos_str
            else:
                r, c = map(int, pos_str.split(','))
                
            circle_type = circle_info["type"]
            value = circle_info["value"]
            ocr_steps.append(f"- {circle_type.capitalize()} circle at position ({r},{c}) with value {value}")
        
        # Simulated reasoning steps
        reasoning_steps = [
            "\n## Solution Exploration",
            f"Let me solve this difficulty level {difficulty} puzzle step by step, exploring different possibilities:",
        ]
        
        # Simulate starting with white circles (straight lines)
        white_circles = []
        for pos_str, info in circles.items():
            if info["type"] == "white":
                if isinstance(pos_str, tuple):
                    pos = pos_str
                else:
                    pos = tuple(map(int, pos_str.split(',')))
                white_circles.append((pos, info))
                
        if white_circles:
            reasoning_steps.append("\n### 1. Starting with white circles (straight lines)")
            for (r, c), info in white_circles[:2]:  # Just use first 2 for example
                value = info["value"]
                reasoning_steps.append(f"- For white circle at ({r},{c}) with value {value}:")
                reasoning_steps.append(f"  - This circle requires a straight line passing through it")
                reasoning_steps.append(f"  - The sum of the two line segments must be {value}")
                
                # Try horizontal option
                reasoning_steps.append(f"  - Option 1: Try horizontal line")
                max_left = min(c, value - 1)
                max_right = min(cols - c - 1, value - 1)
                reasoning_steps.append(f"    - Can extend at most {max_left} left and {max_right} right")
                
                # Try vertical option
                reasoning_steps.append(f"  - Option 2: Try vertical line")
                max_up = min(r, value - 1)
                max_down = min(rows - r - 1, value - 1)
                reasoning_steps.append(f"    - Can extend at most {max_up} up and {max_down} down")
        
        # Simulate exploring black circles (turns)
        black_circles = []
        for pos_str, info in circles.items():
            if info["type"] == "black":
                if isinstance(pos_str, tuple):
                    pos = pos_str
                else:
                    pos = tuple(map(int, pos_str.split(',')))
                black_circles.append((pos, info))
                
        if black_circles:
            reasoning_steps.append("\n### 2. Exploring black circles (turning points)")
            for (r, c), info in black_circles[:2]:  # Just use first 2 for example
                value = info["value"]
                reasoning_steps.append(f"- For black circle at ({r},{c}) with value {value}:")
                reasoning_steps.append(f"  - This circle requires a turn")
                reasoning_steps.append(f"  - The sum of the two straight line segments must be {value}")
                
                # Try different turn configurations
                reasoning_steps.append(f"  - Trying different turn possibilities:")
                reasoning_steps.append(f"    - Option 1: Turn from up to right would allow {min(r, value-1)} up and {min(cols-c-1, value-1)} right")
                reasoning_steps.append(f"    - Option 2: Turn from right to down would allow {min(cols-c-1, value-1)} right and {min(rows-r-1, value-1)} down")
                reasoning_steps.append(f"    - Option 3: Turn from down to left would allow {min(rows-r-1, value-1)} down and {min(c, value-1)} left")
                reasoning_steps.append(f"    - Option 4: Turn from left to up would allow {min(c, value-1)} left and {min(r, value-1)} up")
        
        # Simulate constraint propagation
        reasoning_steps.extend([
            "\n### 3. Constraint Propagation",
            "Now I need to ensure the loop is continuous and closed.",
            "Let me try connecting the segments I've identified so far:",
            "- If I connect white circle at (x,y) to black circle at (p,q), the path must be...",
            "- When trying this connection, I encounter a conflict with...",
            "- Let me backtrack and try a different approach..."
        ])
        
        # Add difficulty-specific reasoning
        if difficulty >= 4:  # Medium-hard and hard
            reasoning_steps.extend([
                f"\n### 4. Advanced Reasoning (for difficulty level {difficulty})",
                "This puzzle requires more sophisticated constraint propagation:",
                "- Multiple constraint conflicts need to be resolved simultaneously",
                "- The solution space is more constrained with additional circles",
                "- Need to consider global consistency, not just local constraints"
            ])
        
        # Simulate final solution found
        reasoning_steps.extend([
            "\n### Final Step: Solution Found",
            f"After exploring different possibilities and resolving all constraints, I've found a valid solution that satisfies all rules.",
            f"The solution forms a continuous loop passing through all the circles according to their rules.",
            f"The complete path is: {puzzle['answer']}"
        ])
        
        return "\n".join(ocr_steps + reasoning_steps)
    
    def visualize(self, puzzle: Dict, images_dir: str = None, **kwargs) -> Tuple[str, str]:
        """Create visual representations of the puzzle and its solution"""
        if images_dir is None:
            images_dir = self.output_dir
            
        rows = puzzle["rows"]
        cols = puzzle["cols"]
        circles = puzzle["circles"]
        loop_segments = puzzle["loop_segments"]
        
        # Create paths for the image files with unified naming
        puzzle_img_path = os.path.join(images_dir, f"{puzzle['index']}.png")
        solution_img_path = os.path.join(images_dir, f"{puzzle['index']}_solution.png")
        
        # Check if images already exist to avoid duplicate generation
        if os.path.exists(puzzle_img_path) and os.path.exists(solution_img_path):
            print(f"Images already exist for {puzzle['index']}, skipping generation...")
            return puzzle_img_path, solution_img_path
        
        # Create puzzle image (without solution)
        self._create_puzzle_image(puzzle, puzzle_img_path, include_solution=False)
        
        # Create solution image
        self._create_puzzle_image(puzzle, solution_img_path, include_solution=True)
        
        # Keep the original specific question from _generate_text_question instead of overriding with generic template
        # Only set a fallback question if none exists
        if "question" not in puzzle or not puzzle["question"]:
            puzzle["question"] = puzzle.get("question_language", "")
        
        return puzzle_img_path, solution_img_path
    
    def _create_puzzle_image(self, puzzle: Dict, filename: str, include_solution: bool = False) -> None:
        """Create a high-resolution, eye-catching puzzle image.

        Improvements:
        - High-res export and supersampled rendering for crisp lines
        - Rounded line caps/joins, layered strokes, subtle glow
        - Strong separation of elements; no unrelated text
        """
        rows = puzzle["rows"]
        cols = puzzle["cols"]
        circles = puzzle["circles"]

        # Compute base canvas size and scaled render size
        base_width = 2 * self.margin + (cols - 1) * self.cell_size
        base_height = 2 * self.margin + (rows - 1) * self.cell_size

        export_scale = max(1, int(self.export_scale))
        render_scale = max(1, int(self.render_scale))
        width = base_width * export_scale * render_scale
        height = base_height * export_scale * render_scale

        # Scale drawing metrics
        def s(value: int) -> int:
            return int(value * export_scale * render_scale)
        cell = s(self.cell_size)
        margin = s(self.margin)
        dot_r = max(2, s(self.dot_radius))
        circ_r = max(6, s(self.circle_radius))
        grid_w = max(1, s(self.grid_width))
        line_w = max(2, s(self.line_width))
        loop_w = max(4, s(self.loop_width))
        border_w = max(2, s(self.border_width))
        border_radius = max(4, s(self.border_radius))
        loop_shadow_blur = max(1, s(self.loop_shadow_blur))
        circle_shadow_blur = max(1, s(self.circle_shadow_blur))
        loop_shadow_offset = (max(0, s(self.loop_shadow_offset[0])), max(0, s(self.loop_shadow_offset[1])))

        # Helper: vertical gradient background
        def create_vertical_gradient(size: Tuple[int, int], top: Tuple[int, int, int], bottom: Tuple[int, int, int]) -> Image.Image:
            w, h = size
            gradient = Image.new("RGB", (w, h), top)
            top_r, top_g, top_b = top
            bot_r, bot_g, bot_b = bottom
            # Draw scanlines to avoid per-pixel loops
            draw_g = ImageDraw.Draw(gradient)
            for y in range(h):
                t = y / max(1, h - 1)
                r = int(top_r * (1 - t) + bot_r * t)
                g = int(top_g * (1 - t) + bot_g * t)
                b = int(top_b * (1 - t) + bot_b * t)
                draw_g.line([(0, y), (w, y)], fill=(r, g, b))
            return gradient

        # Base RGBA canvas with gradient
        base = create_vertical_gradient((width, height), self.bg_gradient_top, self.bg_gradient_bottom).convert("RGBA")
        draw = ImageDraw.Draw(base)

        # Grid rectangle bounds
        grid_left = margin
        grid_top = margin
        grid_right = margin + (cols - 1) * cell
        grid_bottom = margin + (rows - 1) * cell

        # Rounded border with subtle double stroke
        pad = s(22)
        border_rect = (grid_left - pad, grid_top - pad, grid_right + pad, grid_bottom + pad)
        try:
            draw.rounded_rectangle(border_rect, radius=border_radius, outline=self.border_color, width=border_w)
            # Inner lighter stroke for depth
            inner_border = (border_rect[0] + s(3), border_rect[1] + s(3), border_rect[2] - s(3), border_rect[3] - s(3))
            draw.rounded_rectangle(inner_border, radius=max(0, border_radius - s(4)), outline=(255, 255, 255, 90), width=max(1, s(1)))
        except AttributeError:
            draw.rectangle(border_rect, outline=self.border_color, width=border_w)

        # Draw grid lines
        for r in range(rows):
            y = grid_top + r * cell
            draw.line([(grid_left, y), (grid_right, y)], fill=self.grid_color, width=grid_w)
        for c in range(cols):
            x = grid_left + c * cell
            draw.line([(x, grid_top), (x, grid_bottom)], fill=self.grid_color, width=grid_w)

        # Dot glow layer
        dots_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        dots_draw = ImageDraw.Draw(dots_layer)
        for r in range(rows):
            for c in range(cols):
                x = grid_left + c * cell
                y = grid_top + r * cell
                # Soft glow
                glow_radius = dot_r + s(4)
                dots_draw.ellipse([(x - glow_radius, y - glow_radius), (x + glow_radius, y + glow_radius)], fill=(0, 0, 0, 45))
                # Solid dot
                dots_draw.ellipse([(x - dot_r, y - dot_r), (x + dot_r, y + dot_r)], fill=self.dot_color)
        dots_layer = dots_layer.filter(ImageFilter.GaussianBlur(max(0, int(0.5 * export_scale * render_scale))))
        base.alpha_composite(dots_layer)

        # Loop layer with shadow (only for solution view)
        if include_solution:
            loop_segments = puzzle["loop_segments"]
            if loop_segments:
                # Shadow pass
                loop_shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                ls_draw = ImageDraw.Draw(loop_shadow)
                off_x, off_y = loop_shadow_offset
                for (p1, p2) in loop_segments:
                    (r1, c1), (r2, c2) = (p1, p2)
                    x1 = grid_left + c1 * cell + off_x
                    y1 = grid_top + r1 * cell + off_y
                    x2 = grid_left + c2 * cell + off_x
                    y2 = grid_top + r2 * cell + off_y
                    ls_draw.line([(x1, y1), (x2, y2)], fill=self.loop_shadow_color, width=loop_w, joint="curve")
                loop_shadow = loop_shadow.filter(ImageFilter.GaussianBlur(loop_shadow_blur))
                base.alpha_composite(loop_shadow)

                # Main loop stroke
                loop_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                l_draw = ImageDraw.Draw(loop_layer)
                for (p1, p2) in loop_segments:
                    (r1, c1), (r2, c2) = (p1, p2)
                    x1 = grid_left + c1 * cell
                    y1 = grid_top + r1 * cell
                    x2 = grid_left + c2 * cell
                    y2 = grid_top + r2 * cell
                    # Underlay for halo
                    l_draw.line([(x1, y1), (x2, y2)], fill=(255, 255, 255, 180), width=int(loop_w * 1.6), joint="curve")
                    # Main colored stroke
                    l_draw.line([(x1, y1), (x2, y2)], fill=self.loop_color, width=loop_w, joint="curve")
                base.alpha_composite(loop_layer)

        # Circles layer with soft shadow (draw shapes at render resolution)
        circles_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        c_draw = ImageDraw.Draw(circles_layer)
        for pos_str, circle_info in circles.items():
            if isinstance(pos_str, tuple):
                r, c = pos_str
            else:
                r, c = map(int, pos_str.split(','))

            x = grid_left + c * cell
            y = grid_top + r * cell
            circle_type = circle_info["type"]
            value = circle_info["value"]

            fill_color = self.white_circle_color if circle_type == "white" else self.black_circle_color

            # Shadow
            shadow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            s_draw = ImageDraw.Draw(shadow_layer)
            shadow_ellipse = [
                (x - circ_r + s(1), y - circ_r + s(2)),
                (x + circ_r + s(1), y + circ_r + s(2)),
            ]
            s_draw.ellipse(shadow_ellipse, fill=self.circle_shadow_color)
            shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(circle_shadow_blur))
            base.alpha_composite(shadow_layer)

            # Circle body
            ellipse_box = [
                (x - circ_r, y - circ_r),
                (x + circ_r, y + circ_r),
            ]
            # Outer ring for contrast
            c_draw.ellipse(ellipse_box, outline=(255, 255, 255, 230), width=max(1, int(loop_w * 0.35)))
            # Main circle
            c_draw.ellipse(ellipse_box, fill=fill_color, outline=self.line_color, width=max(2, int(loop_w * 0.3)))

            # Defer text drawing to post-downsample stage for extra sharpness
            pass

        base.alpha_composite(circles_layer)

        # Downsample from supersampled render to export size for crisp output
        if render_scale > 1:
            target_size = (base_width * export_scale, base_height * export_scale)
            downsampled = base.resize(target_size, resample=Image.LANCZOS)
        else:
            downsampled = base

        # Draw numeric labels after downsampling for maximum clarity
        rows = puzzle["rows"]
        cols = puzzle["cols"]
        circles = puzzle["circles"]
        grid_left_ds = self.margin * export_scale
        grid_top_ds = self.margin * export_scale
        cell_ds = self.cell_size * export_scale
        circ_r_ds = max(6, self.circle_radius * export_scale)
        draw_ds = ImageDraw.Draw(downsampled)
        for pos_str, circle_info in circles.items():
            if isinstance(pos_str, tuple):
                r, c = pos_str
            else:
                r, c = map(int, pos_str.split(','))
            x = grid_left_ds + c * cell_ds
            y = grid_top_ds + r * cell_ds
            circle_type = circle_info["type"]
            value = circle_info["value"]
            # Use configured blue for numbers regardless of circle type
            text_color = self.number_text_color
            value_str = str(value)
            # Dynamically fit font to circle with padding, bold, centered
            # Try several sizes large -> small until fit
            max_size = int(circ_r_ds * 1.25)
            min_size = max(18, int(circ_r_ds * 0.7))
            best_font = None
            best_bbox = None
            padding = max(2, int(circ_r_ds * 0.12))
            max_w = max(1, circ_r_ds * 2 - padding * 2)
            max_h = max(1, circ_r_ds * 2 - padding * 2)
            for fs in range(max_size, min_size - 1, -1):
                f = self._load_font(fs, prefer_bold=True)
                bb = draw_ds.textbbox((0, 0), value_str, font=f)
                tw = bb[2] - bb[0]
                th = bb[3] - bb[1]
                if tw <= max_w and th <= max_h:
                    best_font = f
                    best_bbox = (tw, th)
                    break
            if best_font is None:
                best_font = self._load_font(min_size, prefer_bold=True)
                bb = draw_ds.textbbox((0, 0), value_str, font=best_font)
                best_bbox = (bb[2] - bb[0], bb[3] - bb[1])
            tw, th = best_bbox
            # Preferred: anchor-based exact centering
            try:
                # For black circles, no white outline: use blue fill only
                if circle_type == "black":
                    draw_ds.text((int(x), int(y)), value_str, font=best_font, fill=text_color, anchor="mm")
                else:
                    stroke_w = max(1, int(best_font.size * 0.13))
                    draw_ds.text((int(x), int(y)), value_str, font=best_font, fill=text_color, anchor="mm", stroke_width=stroke_w, stroke_fill=self.number_stroke_color)
            except TypeError:
                # Fallback: manual centering
                pos = (int(x - tw // 2), int(y - th // 2))
                if circle_type == "black":
                    draw_ds.text(pos, value_str, font=best_font, fill=text_color)
                else:
                    ox = [-1, 1, 0, 0]
                    oy = [0, 0, -1, 1]
                    for dx, dy in zip(ox, oy):
                        draw_ds.text((pos[0] + dx, pos[1] + dy), value_str, font=best_font, fill=self.number_stroke_color)
                    draw_ds.text(pos, value_str, font=best_font, fill=text_color)

        # Save as PNG at high quality
        downsampled.convert("RGB").save(filename, format="PNG", optimize=True)
    
    def solve(self, puzzle: Dict, **kwargs) -> List[Tuple]:
        """Solve the puzzle - for Shingoki, we already have the solution"""
        return puzzle["loop_segments"]
        
    def validate_puzzle(self, puzzle: Dict) -> bool:
        """
        Validate that a puzzle meets all Shingoki rules
        
        Args:
            puzzle: The puzzle dictionary
            
        Returns:
            True if valid, False otherwise
        """
        rows = puzzle["rows"]
        cols = puzzle["cols"]
        circles = puzzle["circles"]
        loop_segments = puzzle["loop_segments"]
        
        # 1. Validate circle values match actual line lengths
        for pos_str, circle_info in circles.items():
            # Handle both tuple and string positions
            if isinstance(pos_str, tuple):
                pos = pos_str
            else:
                pos = tuple(map(int, pos_str.split(',')))
                
            value = circle_info["value"]
            circle_type = circle_info["type"]
            
            # Calculate actual line lengths
            length1, length2 = self._calculate_line_lengths(pos, loop_segments, rows, cols)
            total_length = length1 + length2
            
            # Validate value matches total length
            if total_length != value:
                print(f"Validation error: Circle at {pos} has value {value} but actual length {total_length}")
                return False
            
            # Validate white circles are on straight lines
            if circle_type == "white":
                # For a straight line, one direction should be opposite of the other
                # So the directions should sum to (0,0)
                # Create a graph of connections
                graph = {}
                for p1, p2 in loop_segments:
                    if p1 not in graph:
                        graph[p1] = []
                    if p2 not in graph:
                        graph[p2] = []
                    graph[p1].append(p2)
                    graph[p2].append(p1)
                
                if pos not in graph or len(graph[pos]) != 2:
                    print(f"Validation error: Circle at {pos} is not properly connected")
                    return False
                    
                n1, n2 = graph[pos]
                dir1 = (n1[0] - pos[0], n1[1] - pos[1])
                dir2 = (n2[0] - pos[0], n2[1] - pos[1])
                
                if dir1[0] + dir2[0] != 0 or dir1[1] + dir2[1] != 0:
                    print(f"Validation error: White circle at {pos} is not on a straight line")
                    return False
            
            # Validate black circles are at turning points
            if circle_type == "black":
                # For a turn, directions should not be opposites
                # Create a graph of connections
                graph = {}
                for p1, p2 in loop_segments:
                    if p1 not in graph:
                        graph[p1] = []
                    if p2 not in graph:
                        graph[p2] = []
                    graph[p1].append(p2)
                    graph[p2].append(p1)
                
                if pos not in graph or len(graph[pos]) != 2:
                    print(f"Validation error: Circle at {pos} is not properly connected")
                    return False
                    
                n1, n2 = graph[pos]
                dir1 = (n1[0] - pos[0], n1[1] - pos[1])
                dir2 = (n2[0] - pos[0], n2[1] - pos[1])
                
                # If directions are opposite, it's a straight line, not a turn
                if dir1[0] + dir2[0] == 0 and dir1[1] + dir2[1] == 0:
                    print(f"Validation error: Black circle at {pos} is not at a turning point")
                    return False
        
        return True

    def generate_batch_deprecated(self, output_dir: str, puzzles_per_difficulty: int = 300, clean_existing: bool = False) -> Dict:
        """
        Generate a complete batch of puzzles with 5 difficulty levels
        Different difficulties use different grid sizes: 1=3x3, 2=4x4, 3=5x5, 4=6x6, 5=7x7
        
        Args:
            output_dir: Base output directory
            puzzles_per_difficulty: Number of puzzles per difficulty level (default: 300)
            clean_existing: Whether to clean existing images before generating new ones
            
        Returns:
            Dictionary containing all generated puzzles organized by difficulty
        """
        # Set up output directories
        self.output_dir = output_dir
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        # Prepare annotations path
        annotations_path = os.path.join(output_dir, "annotations.json")
        
        # Clean existing images if requested
        if clean_existing:
            import glob
            existing_images = glob.glob(os.path.join(images_dir, "*.png"))
            for img_path in existing_images:
                os.remove(img_path)
            print(f"Cleaned {len(existing_images)} existing images")
            # Reset the generated puzzle hashes to avoid conflicts
            self.generated_puzzle_hashes.clear()
            print("Reset puzzle hash cache")
            # Also reset annotations if requested
            try:
                if os.path.exists(annotations_path):
                    os.remove(annotations_path)
                    print("Cleared existing annotations.json")
            except Exception as _e:
                print(f"Warning: could not clear annotations.json: {_e}")
        
        all_puzzles = []
        difficulty_stats = {}
        
        # Define grid sizes for each difficulty level
        difficulty_grid_sizes = {
            1: 3,  # Easy: 3x3
            2: 4,  # Medium-Easy: 4x4
            3: 5,  # Medium: 5x5
            4: 6,  # Medium-Hard: 6x6
            5: 7   # Hard: 7x7
        }
        
        # Generate puzzles for each difficulty level (1-5)
        for difficulty in range(1, 6):
            grid_size = difficulty_grid_sizes[difficulty]
            print(f"\nGenerating difficulty {difficulty} puzzles ({grid_size}x{grid_size} grid)...")
            print(f"Target: {puzzles_per_difficulty} unique puzzles")
            difficulty_puzzles = []
            
            # Generate puzzles for this difficulty
            task_num = 0
            max_total_attempts = puzzles_per_difficulty * 10  # Maximum total attempts per difficulty
            total_attempts = 0
            
            while len(difficulty_puzzles) < puzzles_per_difficulty and total_attempts < max_total_attempts:
                task_num += 1
                total_attempts += 1
                
                # Generate puzzle with unique naming (zero-padded for better sorting)
                puzzle_id = f"shingoki_{grid_size}_{grid_size}_{task_num:03d}"
                
                try:
                    puzzle = self._generate_single_puzzle(
                        index=puzzle_id,
                        rows=grid_size,
                        cols=grid_size,
                        difficulty=difficulty
                    )
                    
                    # Check for uniqueness using enhanced hash
                    puzzle_hash = self._get_enhanced_puzzle_hash(puzzle)
                    if puzzle_hash in self.generated_puzzle_hashes:
                        print(f"  Skipping duplicate puzzle (attempt {total_attempts})")
                        continue  # Try again with different random seed
                    
                    # Add to unique set
                    self.generated_puzzle_hashes.add(puzzle_hash)
                    
                    # Create visualization and get image paths
                    puzzle_image_path, solution_image_path = self.visualize(puzzle, images_dir)
                    
                    # Update puzzle with relative image paths
                    puzzle["image"] = os.path.relpath(puzzle_image_path, output_dir)
                    puzzle["solution_image"] = os.path.relpath(solution_image_path, output_dir)
                    
                    difficulty_puzzles.append(puzzle)
                    all_puzzles.append(puzzle)
                    
                    # Save this item to annotations.json immediately
                    annotation = {
                        "index": puzzle["index"],
                        "category": puzzle["category"],
                        "image": puzzle["image"],
                        "question": puzzle["question"],
                        "question_language": puzzle["question_language"],
                        "answer": puzzle["answer"],
                        "initial_state": puzzle["initial_state"],
                        "difficulty": puzzle["difficulty"],
                        "cot": puzzle["cot"],
                    }
                    self._save_to_annotations(annotation, output_dir)
                    
                    # Progress reporting
                    if len(difficulty_puzzles) % 50 == 0:
                        print(f"  Generated {len(difficulty_puzzles)}/{puzzles_per_difficulty} puzzles ({grid_size}x{grid_size})")
                    
                except Exception as e:
                    print(f"  Error generating puzzle (attempt {total_attempts}): {e}")
                    continue
            
            difficulty_stats[difficulty] = len(difficulty_puzzles)
            print(f"Completed difficulty {difficulty}: {len(difficulty_puzzles)} puzzles ({grid_size}x{grid_size})")
            
            if len(difficulty_puzzles) < puzzles_per_difficulty:
                print(f"  Warning: Only generated {len(difficulty_puzzles)} out of {puzzles_per_difficulty} requested puzzles")
        
        # Already appended each item to annotations.json during generation
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total puzzles generated: {len(all_puzzles)}")
        print(f"Target was: {puzzles_per_difficulty * 5} puzzles")
        print(f"Output directory: {output_dir}")
        print(f"Images directory: {images_dir}")
        print(f"Annotations file: {annotations_path}")
        print("\nDifficulty breakdown:")
        for diff, count in difficulty_stats.items():
            grid_size = difficulty_grid_sizes[diff]
            print(f"  Difficulty {diff} ({grid_size}x{grid_size}): {count} puzzles")
        
        return {
            "puzzles": all_puzzles,
            "stats": difficulty_stats,
            "total_count": len(all_puzzles),
            "output_dir": output_dir,
            "images_dir": images_dir,
            "annotations_file": annotations_path,
            "difficulty_grid_sizes": difficulty_grid_sizes
        }
    
    def _save_annotations_deprecated(self, puzzles: List[Dict], filename: str, difficulty_grid_sizes: Dict) -> None:
        """Save all puzzles to annotations.json with simplified format"""
        annotations = []
        
        for puzzle in puzzles:
            # Create simplified annotation entry
            annotation = {
                "index": puzzle["index"],
                "category": puzzle["category"],
                "image": puzzle["image"],
                "question": puzzle["question"],
                "question_language": puzzle["question_language"],
                "answer": puzzle["answer"],
                "initial_state": puzzle["initial_state"],
                "difficulty": puzzle["difficulty"],
                "cot": puzzle["cot"]
            }
            
            annotations.append(annotation)
        
        # Save to file as simple array
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        
        print(f"Saved annotations for {len(puzzles)} puzzles to {filename}")


