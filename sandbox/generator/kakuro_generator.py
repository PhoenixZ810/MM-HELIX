import os
import json
import random
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Any, Set
import hashlib
from abc import ABC, abstractmethod
# Make BaseGenerator import robust whether running from project root or from inside MMRLSandBox
try:
    from generator.base_generator import BaseGenerator
except Exception:
    import sys
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from generator.base_generator import BaseGenerator

class KakuroGenerator(BaseGenerator):
    def __init__(self, output_folder, **kwargs):
        super().__init__(output_folder)
        self.font_path = kwargs.get('font_path', 'arial.ttf')
        self.cell_size = kwargs.get('cell_size', 60)
        # Enforce sizes consistent with manager's configuration by default
        self.allowed_sizes = kwargs.get('allowed_sizes', [3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        # 高清导出相关参数（可通过初始化传参覆盖）
        self.export_scale = kwargs.get('export_scale', 3.0)  # 默认 2x 放大绘制
        self.export_dpi = kwargs.get('export_dpi', 1200)      # 默认 300 DPI
        self.difficulty_levels = {
            1: {'min_sum': 3, 'max_sum': 16, 'max_length': 3, 'grid_size': 3},
            2: {'min_sum': 10, 'max_sum': 30, 'max_length': 5, 'grid_size': 4},
            3: {'min_sum': 20, 'max_sum': 45, 'max_length': 9, 'grid_size': 5},
            4: {'min_sum': 30, 'max_sum': 60, 'max_length': 12, 'grid_size': 6},
            5: {'min_sum': 40, 'max_sum': 80, 'max_length': 15, 'grid_size': 7},
        }
    def generate(self, num_cases, difficulty, output_folder=None):
        """Generate Kakuro puzzles with specified number of cases and difficulty."""
        if output_folder is None:
            output_folder = self.output_folder

        # Get difficulty parameters
        difficulty_params = self._get_difficulty_params(difficulty)
        grid_size = difficulty_params['grid_size']

        # Create output directories
        os.makedirs(output_folder, exist_ok=True)
        images_dir = os.path.join(output_folder, 'images')
        os.makedirs(images_dir, exist_ok=True)

        # Collect all puzzles data
        all_puzzles = []

        # Generate specified number of cases
        for case_num in range(num_cases):
            # Use timestamp as base seed for each case
            base_timestamp = int(time.time())
            seed = f"{base_timestamp}_{case_num}"

            # Set random seed for deterministic generation
            seed_str = f"{grid_size}-{seed}"
            combined_seed = int(hashlib.sha256(seed_str.encode('utf-8')).hexdigest(), 16) % (2**32)
            random.seed(combined_seed)
            np.random.seed(combined_seed)

            # Generate a single puzzle deterministically per call
            puzzle = self._generate_single_puzzle(grid_size, grid_size, difficulty)

            puzzle_index = f"kakuro_{grid_size}_{difficulty}_{case_num}_{seed}"

            # Generate image paths
            puzzle_image_name = f"{puzzle_index}.png"
            puzzle_image_path = os.path.join(images_dir, puzzle_image_name)

            # Visualize and save puzzle image
            self.visualize(puzzle, save_path=puzzle_image_path)

            # Generate CoT steps
            cot_steps = self._generate_cot(puzzle)

            # Prepare the puzzle data in the required format
            puzzle_data = {
                "index": puzzle_index,
                "category": "kakuro",
                "image": f"images/{puzzle_image_name}",
                "question": f"Your task is to solve the Kakuro puzzle from the given image by filling white cells with appropriate digits.\n\n### Game Rules:\n1. The puzzle is a grid where black cells contain clue numbers and white cells need to be filled with digits 1-9.\n2. In black cells, numbers below the diagonal are 'down' clues, and numbers above are 'right' clues.\n3. Each clue indicates the sum of consecutive white cells in that direction.\n4. Digits in each run cannot repeat.\n\n### Coordinate System:\n- The grid coordinates start at (0,0) in the top-left corner\n- Rows increase downward and columns increase to the right\n\n### Output Format:\nProvide your answer as a space-separated list of coordinate-value pairs in the format: (row,column):value\n\nExample: (0,2):5 (0,7):7 ...\n",
                "question_language": self._generate_text_prompt(puzzle),
                "answer": self._format_solution(puzzle['solution']),
                "initial_state": self._format_initial_state(puzzle['grid']),
                "difficulty": difficulty,
                "cot": cot_steps['step4'],  # Full CoT for backward compatibility

                # Add step-by-step CoT fields
                "cot_step1_all": cot_steps['step1'],
                "cot_step2_all": cot_steps['step2'],
                "cot_step3_all": cot_steps['step3'],
            }

            all_puzzles.append(puzzle_data)

        # Save annotations using base class method
        self.save_annotations(all_puzzles, output_folder)

        print(f"Generated {len(all_puzzles)} new Kakuro puzzles (difficulty {difficulty}) to {output_folder}")
        return all_puzzles

    def _get_difficulty_params(self, difficulty):
        """Get difficulty parameters for the specified difficulty level."""
        return self.difficulty_levels.get(difficulty, self.difficulty_levels[1])
    
    def _generate_single_puzzle(self, rows, cols, difficulty):
        """Generate a single Kakuro puzzle with specified dimensions and difficulty."""
        difficulty_params = self._get_difficulty_params(difficulty)
        
        # Create initial grid with all black cells
        grid = [[{'type': 'black'} for _ in range(cols)] for _ in range(rows)]
        
        # Create puzzle structure (determine white vs black cells)
        grid = self._create_puzzle_structure(grid, rows, cols, difficulty)
        
        # Generate clues and solution
        puzzle, solution = self._generate_clues_and_solution(grid, difficulty_params)
        
        return {
            'grid': puzzle,
            'solution': solution,
            'rows': rows,
            'cols': cols,
            'difficulty': difficulty
        }
    
    def _create_puzzle_structure(self, grid, rows, cols, difficulty):
        """Create the puzzle structure by determining which cells are white vs black."""
        # The top-left cell is always black
        grid[0][0]['type'] = 'black'
        
        # Determine complexity based on difficulty (1-5)
        white_cell_ratio = {
            1: 0.48,  # easiest: fewer white cells
            2: 0.56,
            3: 0.62,
            4: 0.68,
            5: 0.74   # hardest: more white cells
        }[difficulty]
        
        # First row and column typically have black cells for clues
        for i in range(1, cols):
            if random.random() < white_cell_ratio:
                grid[0][i]['type'] = 'white'
            
        for i in range(1, rows):
            if random.random() < white_cell_ratio:
                grid[i][0]['type'] = 'white'
        
        # For the rest of the grid
        for i in range(1, rows):
            for j in range(1, cols):
                if random.random() < white_cell_ratio:
                    grid[i][j]['type'] = 'white'
        
        # Ensure white cells are in valid runs (connected horizontally or vertically)
        self._ensure_valid_runs(grid, rows, cols)
        
        return grid
    
    def _ensure_valid_runs(self, grid, rows, cols):
        """Ensure all white cells are part of valid runs."""
        for i in range(rows):
            for j in range(cols):
                if grid[i][j]['type'] == 'white':
                    # Check if this white cell has a black cell above or to the left
                    has_top_clue = (i > 0 and grid[i-1][j]['type'] == 'black')
                    has_left_clue = (j > 0 and grid[i][j-1]['type'] == 'black')
                    
                    # If no clues, convert to black
                    if not (has_top_clue or has_left_clue):
                        grid[i][j]['type'] = 'black'
    
    def _create_minimal_runs(self, grid, rows, cols):
        """Create minimal valid runs when no runs exist."""
        # Create at least one horizontal run in the first row
        if cols >= 3:
            grid[0][0]['type'] = 'black'  # clue cell
            grid[0][1]['type'] = 'white'  # run cell 1
            grid[0][2]['type'] = 'white'  # run cell 2
        
        # Create at least one vertical run in the first column
        if rows >= 3:
            grid[0][0]['type'] = 'black'  # clue cell (already set)
            grid[1][0]['type'] = 'white'  # run cell 1
            grid[2][0]['type'] = 'white'  # run cell 2
        
        # For small grids (2x2), create a simple structure
        if rows == 2 and cols == 2:
            grid[0][0]['type'] = 'black'
            grid[0][1]['type'] = 'white'
            grid[1][0]['type'] = 'white'
            grid[1][1]['type'] = 'white'
    
    def _generate_clues_and_solution(self, grid, difficulty_params):
        """Generate clues and solution for the puzzle."""
        rows = len(grid)
        cols = len(grid[0])
        solution = {}
        
        # First pass: identify all runs (horizontal and vertical)
        horizontal_runs = []
        vertical_runs = []
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j]['type'] == 'black':
                    # Check for horizontal run to the right
                    run_cells = []
                    for k in range(j+1, cols):
                        if k < cols and grid[i][k]['type'] == 'white':
                            run_cells.append((i, k))
                        else:
                            break
                    
                    if run_cells:
                        horizontal_runs.append({
                            'clue_cell': (i, j),
                            'run_cells': run_cells,
                            'length': len(run_cells)
                        })
                    
                    # Check for vertical run below
                    run_cells = []
                    for k in range(i+1, rows):
                        if k < rows and grid[k][j]['type'] == 'white':
                            run_cells.append((k, j))
                        else:
                            break
                    
                    if run_cells:
                        vertical_runs.append({
                            'clue_cell': (i, j),
                            'run_cells': run_cells,
                            'length': len(run_cells)
                        })
        
        # Check if we have any valid runs, if not, create a minimal valid structure
        if not horizontal_runs and not vertical_runs:
            # No valid runs found, create a minimal valid puzzle structure
            # Force at least one horizontal and one vertical run
            self._create_minimal_runs(grid, rows, cols)
            # Re-run the analysis after creating minimal runs
            horizontal_runs = []
            vertical_runs = []
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j]['type'] == 'black':
                        # Check for horizontal run to the right
                        run_cells = []
                        for k in range(j+1, cols):
                            if k < cols and grid[i][k]['type'] == 'white':
                                run_cells.append((i, k))
                            else:
                                break
                        if run_cells:
                            horizontal_runs.append({
                                'clue_cell': (i, j),
                                'run_cells': run_cells,
                                'length': len(run_cells)
                            })
                        
                        # Check for vertical run below
                        run_cells = []
                        for k in range(i+1, rows):
                            if k < rows and grid[k][j]['type'] == 'white':
                                run_cells.append((k, j))
                            else:
                                break
                        if run_cells:
                            vertical_runs.append({
                                'clue_cell': (i, j),
                                'run_cells': run_cells,
                                'length': len(run_cells)
                            })
        
        # Second pass: assign values to white cells
        white_cells = [(i, j) for i in range(rows) for j in range(cols) if grid[i][j]['type'] == 'white']
        available_values = {cell: set(range(1, 10)) for cell in white_cells}
        
        # Assign values to cells satisfying constraints
        while white_cells:
            # Process cell with fewest available values first
            white_cells.sort(key=lambda cell: len(available_values[cell]))
            cell = white_cells.pop(0)
            
            if not available_values[cell]:
                # No viable values, need to backtrack or regenerate
                return self._generate_clues_and_solution(grid, difficulty_params)
            
            # Choose a random value from available ones
            value = random.choice(list(available_values[cell]))
            solution[cell] = value
            
            # Update available values for remaining cells in the same runs
            for run in horizontal_runs:
                if cell in run['run_cells']:
                    for other_cell in run['run_cells']:
                        if other_cell != cell and other_cell in white_cells:
                            # Remove the used value from other cells in this run
                            if value in available_values[other_cell]:
                                available_values[other_cell].remove(value)
            
            for run in vertical_runs:
                if cell in run['run_cells']:
                    for other_cell in run['run_cells']:
                        if other_cell != cell and other_cell in white_cells:
                            # Remove the used value from other cells in this run
                            if value in available_values[other_cell]:
                                available_values[other_cell].remove(value)
        
        # Third pass: calculate sums for each run and add clues to grid
        for run in horizontal_runs:
            clue_i, clue_j = run['clue_cell']
            run_sum = sum(solution[cell] for cell in run['run_cells'])
            grid[clue_i][clue_j]['right'] = (run_sum, len(run['run_cells']))
        
        for run in vertical_runs:
            clue_i, clue_j = run['clue_cell']
            run_sum = sum(solution[cell] for cell in run['run_cells'])
            grid[clue_i][clue_j]['down'] = (run_sum, len(run['run_cells']))
        
        return grid, solution
    
    def visualize(self, puzzle, **kwargs):
        """Visualize the Kakuro puzzle as a high-resolution image with clear fonts."""
        include_solution = kwargs.get('include_solution', False)
        save_path = kwargs.get('save_path', None)
        export_scale = float(kwargs.get('export_scale', self.export_scale))
        export_dpi = int(kwargs.get('export_dpi', self.export_dpi))

        grid = puzzle['grid']
        solution = puzzle['solution'] if include_solution else {}
        rows = len(grid)
        cols = len(grid[0])

        # Scale-aware dimensions for HD rendering
        scaled_cell_size = max(1, int(self.cell_size * export_scale))
        padding = max(2, int(10 * export_scale))  # Padding around the grid, scaled

        # Fixed ratios to control font sizes relative to cell size
        clue_text_ratio = float(kwargs.get('clue_text_ratio', 0.30))  # For triangle clues; conservative to prevent overflow
        white_text_ratio = float(kwargs.get('white_text_ratio', 0.58))  # For white-cell digits

        # Create image with padding at high pixel density
        width = cols * scaled_cell_size + 2 * padding
        height = rows * scaled_cell_size + 2 * padding
        image = Image.new('RGB', (width, height), '#F5F0E1')
        draw = ImageDraw.Draw(image)

        # Scaled stroke widths
        outline_wide = max(2, int(3 * export_scale))
        outline_thin = max(1, int(1 * export_scale))
        diagonal_width = max(2, int(3 * export_scale))
        inner_shadow_width = max(1, int(2 * export_scale))
        grid_border_width = max(2, int(2 * export_scale))
        shadow_offset = max(4, int(8 * export_scale))

        # Background shadow for subtle depth
        shadow_color = '#D0C8B0'
        draw.rectangle(
            [
                padding + shadow_offset,
                padding + shadow_offset,
                width - padding + shadow_offset,
                height - padding + shadow_offset,
            ],
            fill=shadow_color,
        )

        # Board background
        board_bg_color = '#E8DFCA'
        draw.rectangle(
            [padding, padding, width - padding, height - padding],
            fill=board_bg_color,
            outline='#7D6E83',
            width=grid_border_width,
        )

        # Fonts (scaled). Try specified font, then common fallbacks, else default.
        def load_font(path_candidates, size):
            for path in path_candidates:
                try:
                    return ImageFont.truetype(path, size)
                except (IOError, OSError):
                    continue
            return ImageFont.load_default()

        clue_font_size = max(8, int(scaled_cell_size * clue_text_ratio))
        solution_font_size = max(8, int(scaled_cell_size * white_text_ratio))
        title_font_size = max(10, int(min(width, height) * 0.08))

        font_candidates = [self.font_path, 'DejaVuSans.ttf', 'Arial.ttf', 'LiberationSans-Regular.ttf']
        clue_font = load_font(font_candidates, clue_font_size)
        solution_font = load_font(font_candidates, solution_font_size)

        # Cached loader for many font sizes
        _font_cache = {}
        def load_font_cached(size):
            key = int(size)
            if key not in _font_cache:
                _font_cache[key] = load_font(font_candidates, key)
            return _font_cache[key]

        # Measure text size
        def measure_text(font_obj, text):
            bbox = font_obj.getbbox(text)
            if bbox is None:
                return (0, 0)
            return (bbox[2] - bbox[0], bbox[3] - bbox[1])

        # Get a font that fits inside a given bounding box
        def get_fitting_font(text, max_w, max_h, preferred_size, min_size=6):
            size = int(preferred_size)
            size = max(min_size, size)
            while size >= min_size:
                f = load_font_cached(size)
                w, h = measure_text(f, text)
                if w <= max_w and h <= max_h:
                    return f, (w, h)
                size -= 1
            f = load_font_cached(min_size)
            return f, measure_text(f, text)

        # Draw grid cells
        for i in range(rows):
            for j in range(cols):
                cell = grid[i][j]
                x = j * scaled_cell_size + padding
                y = i * scaled_cell_size + padding

                if cell['type'] == 'black':
                    # Black cell
                    draw.rectangle(
                        [x, y, x + scaled_cell_size, y + scaled_cell_size],
                        fill='#393E46',
                        outline='#222831',
                        width=outline_thin,
                    )

                    # Diagonal line for clue cells
                    if 'right' in cell or 'down' in cell:
                        draw.line(
                            [(x, y), (x + scaled_cell_size, y + scaled_cell_size)],
                            fill='#EEEEEE',
                            width=diagonal_width,
                        )

                    # Right clue (upper-right triangle, centered at triangle centroid and fitted)
                    if 'right' in cell:
                        right_sum, _ = cell['right']
                        # Triangle vertices: (x,y), (x+S,y), (x+S,y+S)
                        # Centroid = (x + 2S/3, y + S/3)
                        center_x = x + scaled_cell_size * (2.0 / 3.0)
                        center_y = y + scaled_cell_size * (1.0 / 3.0)
                        # Conservative axis-aligned bounding box inside triangle
                        max_w = scaled_cell_size * 0.36
                        max_h = scaled_cell_size * 0.28
                        fit_font, (tw, th) = get_fitting_font(str(right_sum), max_w, max_h, clue_font_size)
                        draw.text((center_x - tw / 2 + 1, center_y - th / 2 + 1), str(right_sum), fill='#222831', font=fit_font)
                        draw.text((center_x - tw / 2, center_y - th / 2), str(right_sum), fill='#FFD369', font=fit_font)

                    # Down clue (lower-left triangle, centered at triangle centroid and fitted)
                    if 'down' in cell:
                        down_sum, _ = cell['down']
                        # Triangle vertices: (x,y), (x,y+S), (x+S,y+S)
                        # Centroid = (x + S/3, y + 2S/3)
                        center_x = x + scaled_cell_size * (1.0 / 3.0)
                        center_y = y + scaled_cell_size * (2.0 / 3.0)
                        max_w = scaled_cell_size * 0.36
                        max_h = scaled_cell_size * 0.28
                        fit_font, (tw, th) = get_fitting_font(str(down_sum), max_w, max_h, clue_font_size)
                        draw.text((center_x - tw / 2 + 1, center_y - th / 2 + 1), str(down_sum), fill='#222831', font=fit_font)
                        draw.text((center_x - tw / 2, center_y - th / 2), str(down_sum), fill='#FFD369', font=fit_font)

                elif cell['type'] == 'white':
                    # White cell
                    draw.rectangle(
                        [x, y, x + scaled_cell_size, y + scaled_cell_size],
                        fill='#EEEEEE',
                        outline='#7D6E83',
                        width=outline_thin,
                    )

                    # Inner shadow
                    draw.line(
                        [(x + 2, y + 2), (x + 2, y + scaled_cell_size - 2)],
                        fill='#FFFFFF',
                        width=inner_shadow_width,
                    )
                    draw.line(
                        [(x + 2, y + 2), (x + scaled_cell_size - 2, y + 2)],
                        fill='#FFFFFF',
                        width=inner_shadow_width,
                    )

                    # Optional solution value (centered and fitted inside the cell)
                    if include_solution and (i, j) in solution:
                        value = solution[(i, j)]
                        center_x = x + scaled_cell_size / 2
                        center_y = y + scaled_cell_size / 2
                        max_w = scaled_cell_size * 0.8
                        max_h = scaled_cell_size * 0.8
                        fit_font, (tw, th) = get_fitting_font(str(value), max_w, max_h, solution_font_size)
                        draw.text(
                            (center_x - tw / 2 + 1, center_y - th / 2 + 1),
                            str(value),
                            fill='#7D6E83',
                            font=fit_font,
                        )
                        draw.text(
                            (center_x - tw / 2, center_y - th / 2),
                            str(value),
                            fill='#222831',
                            font=fit_font,
                        )

        # Grid lines
        for i in range(rows + 1):
            line_y = i * scaled_cell_size + padding
            draw.line(
                [(padding, line_y), (width - padding, line_y)],
                fill='#7D6E83',
                width=grid_border_width if i == 0 or i == rows else outline_thin,
            )

        for j in range(cols + 1):
            line_x = j * scaled_cell_size + padding
            draw.line(
                [(line_x, padding), (line_x, height - padding)],
                fill='#7D6E83',
                width=grid_border_width if j == 0 or j == cols else outline_thin,
            )

        # Decorative title for non-solution image
        if not include_solution:
            try:
                title_font = load_font(font_candidates, title_font_size)
                # Slightly above the board

            except Exception:
                pass

        # Save or return the image with high DPI metadata
        if save_path:
            try:
                image.save(save_path, dpi=(export_dpi, export_dpi))
            except Exception:
                image.save(save_path)

        return image
    
    def solve(self, puzzle, **kwargs):
        """Solve a Kakuro puzzle."""
        # For this implementation, we'll return the pre-computed solution
        return puzzle['solution']
    
    def _format_solution(self, solution):
        """Format the solution as a string."""
        ordered_solution = sorted(solution.items())
        solution_str = " ".join([f"({i},{j}):{val}" for (i, j), val in ordered_solution])
        return solution_str
    
    def _format_grid(self, grid):
        """Format the grid as a JSON string."""
        return json.dumps(grid)
    
    def _format_initial_state(self, grid):
        """Format the grid as a simplified list representation for initial_state."""
        # Create a simplified representation showing only essential puzzle structure
        # Format: list of lists where each cell is either 'black' with clues or 'white' (empty)
        simplified_grid = []
        for row in grid:
            simplified_row = []
            for cell in row:
                if cell['type'] == 'black':
                    # For black cells, only include the clue information needed for solving
                    black_cell = {'type': 'black'}
                    if 'right' in cell:
                        black_cell['right'] = cell['right'][0]  # Only sum, not length
                    if 'down' in cell:
                        black_cell['down'] = cell['down'][0]  # Only sum, not length
                    simplified_row.append(black_cell)
                else:
                    # For white cells, just indicate they need to be filled
                    simplified_row.append({'type': 'white'})
            simplified_grid.append(simplified_row)
        
        return simplified_grid
    
    def _generate_text_prompt(self, puzzle):
        """Generate a comprehensive text description of the puzzle without revealing solution."""
        grid = puzzle['grid']
        initial_state = self._format_initial_state(grid)
        rows = len(grid)
        cols = len(grid[0])
        
        # Create ASCII representation of the grid
        ascii_grid = []
        for i in range(rows):
            row_str = ""
            for j in range(cols):
                cell = grid[i][j]
                if cell['type'] == 'black':
                    # Format black cell with clues if present
                    down_val = "  " if 'down' not in cell else f"{cell['down'][0]:2}"
                    right_val = "  " if 'right' not in cell else f"{cell['right'][0]:2}"
                    row_str += f"[{down_val}\\{right_val}]"
                else:  # White cell
                    row_str += "[    ]"
            ascii_grid.append(row_str)
        
        # Collect clues in a structured format - only show puzzle structure, not solution
        clues = []
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell['type'] == 'black':
                    parts = []
                    if 'right' in cell:
                        sum_r, len_r = cell['right']
                        parts.append(f"horizontal run with sum {sum_r}")
                    if 'down' in cell:
                        sum_d, len_d = cell['down']
                        parts.append(f"vertical run with sum {sum_d}")
                    if parts:
                        clues.append(f"Black cell at ({i},{j}): {', '.join(parts)}")
        
        # Collect white cell coordinates - only coordinates, no values
        white_cells = []
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell['type'] == 'white':
                    white_cells.append(f"({i},{j})")
        
        # Build the comprehensive prompt
        prompt = "# Kakuro Puzzle\n\n"
        
        # prompt += "## Initial State\n"
        # prompt += f"The initial state is provided as: {initial_state}\n\n"
        
        # Add ASCII representation of the grid
        prompt += "## Grid Representation\n"
        prompt += "Below is a text representation of the Kakuro grid:\n"
        prompt += "- Black cells show clues as [down\\right]\n"
        prompt += "- White cells are shown as [    ] and need to be filled\n\n"
        prompt += "## Initial State\n"
        for row in ascii_grid:
            prompt += row + "\n"

        
        # Add clues section - without revealing cell arrangements
        prompt += "## Clues\n"
        prompt += "\n".join(clues) + "\n\n"
        
        # Add white cells to fill
        prompt += "## White Cells to Fill\n"
        prompt += "The following cells need to be filled with digits 1-9:\n"
        prompt += ", ".join(white_cells) + "\n\n"
        
        # Add game rules
        prompt += "## Game Rules\n"
        prompt += "1. Each white cell must contain a digit from 1 to 9.\n"
        prompt += "2. Digits in the same row or column run cannot repeat.\n"
        prompt += "3. The sum of digits in each run must equal the clue number shown in the black cell.\n"
        prompt += "4. A clue number above the diagonal line refers to a vertical run (down).\n"
        prompt += "5. A clue number below the diagonal line refers to a horizontal run (right).\n\n"
        
        # Add output format instructions
        prompt += "## Output Format\n"
        prompt += "Provide your answer as a space-separated list of coordinate-value pairs in the format:\n"
        prompt += "'(row,column):value', e.g., '(0,2):5 (0,7):7 ...'\n"
        prompt += "- Coordinates start at (0,0) in the top-left corner\n"
        prompt += "- Rows increase downward, columns increase to the right\n"
        prompt += "- List all white cells in your answer\n"
        prompt += "## Example: \n(0,2):5 (0,7):7\n\n"
        
        
        return prompt
    
    def _generate_cot(self, puzzle):
        """Generate a structured 4-step CoT following the specified format with enhanced detail."""
        grid = puzzle['grid']
        solution = puzzle['solution']
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
        # Safety check: ensure we have at least some white cells
        white_cell_count = sum(1 for row in grid for cell in row if cell['type'] == 'white')
        if white_cell_count == 0:
            # Return a minimal CoT for degenerate puzzles
            return self._generate_minimal_cot(puzzle)
        
        # Extract runs information for analysis
        horizontal_runs = []
        vertical_runs = []
        
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if cell['type'] == 'black':
                    if 'right' in cell:
                        sum_r, len_r = cell['right']
                        run_cells = []
                        for k in range(j + 1, len(row)):
                            if k < len(row) and row[k]['type'] == 'white':
                                run_cells.append((i, k))
                            else:
                                break
                        if run_cells:
                            horizontal_runs.append({
                                'clue': (i, j),
                                'sum': sum_r,
                                'cells': run_cells
                            })
                    
                    if 'down' in cell:
                        sum_d, len_d = cell['down']
                        run_cells = []
                        for k in range(i + 1, len(grid)):
                            if k < len(grid) and grid[k][j]['type'] == 'white':
                                run_cells.append((k, j))
                            else:
                                break
                        if run_cells:
                            vertical_runs.append({
                                'clue': (i, j),
                                'sum': sum_d,
                                'cells': run_cells
                            })
        
        all_runs = horizontal_runs + vertical_runs
        white_cells = [(i, j) for i in range(rows) for j in range(cols) if grid[i][j]['type'] == 'white']
        
        # Generate ASCII representation for better visualization
        def generate_ascii_grid():
            ascii_lines = []
            for i in range(rows):
                row_str = ""
                for j in range(cols):
                    cell = grid[i][j]
                    if cell['type'] == 'black':
                        down_val = "  " if 'down' not in cell else f"{cell['down'][0]:2}"
                        right_val = "  " if 'right' not in cell else f"{cell['right'][0]:2}"
                        row_str += f"[{down_val}\\{right_val}]"
                    else:  # White cell
                        row_str += "[    ]"
                ascii_lines.append(row_str)
            return ascii_lines
        
        ascii_grid = generate_ascii_grid()
        
        # =============== STEP 1: Understanding Game Rules ===============
        step1 = "Let me solve this Kakuro puzzle step by step.\n\n"
        step1 += "### Step 1: Understanding the puzzle rules and objectives\n\n"
        step1 += "**Kakuro Rules:**\n"
        step1 += "1. **Grid Structure**: This is a crossword-style number puzzle with black cells (containing clues) and white cells (to be filled with digits).\n"
        step1 += "2. **Digit Constraints**: Each white cell must contain a digit from 1 to 9.\n"
        step1 += "3. **Clue System**: Black cells contain sum clues with a diagonal line:\n"
        step1 += "   - Number above diagonal = sum for vertical run (downward)\n"
        step1 += "   - Number below diagonal = sum for horizontal run (rightward)\n"
        step1 += "4. **Uniqueness Rule**: Within each run (horizontal or vertical), digits cannot repeat.\n"
        step1 += "5. **Sum Constraint**: All digits in a run must sum exactly to the clue number.\n\n"
        step1 += "**Objective**: Fill all white cells with digits 1-9 such that each run sums to its clue and contains no duplicate digits.\n\n"
        step1 += f"**Puzzle Overview**: This is a {rows}×{cols} grid with {len(white_cells)} white cells to fill, organized into {len(horizontal_runs)} horizontal runs and {len(vertical_runs)} vertical runs."
        
        # =============== STEP 2: Reading and Analyzing Visual Information ===============
        step2 = step1 + "\n\n### Step 2: Reading the image carefully and analyzing the initial state\n\n"
        step2 += "**Careful Image Analysis:**\n"
        step2 += "Let me examine the grid structure systematically from the image:\n\n"
        
        # ASCII representation
        step2 += "**Grid Layout (ASCII representation):**\n"
        step2 += "Format: [down\\right] for black cells, [    ] for white cells\n"
        step2 += "```\n"
        for line in ascii_grid:
            step2 += line + "\n"
        step2 += "```\n\n"
        
        # Detailed cell-by-cell analysis
        step2 += "**Detailed Initial State Reading:**\n"
        black_cells = []
        for i in range(rows):
            for j in range(cols):
                cell = grid[i][j]
                if cell['type'] == 'black':
                    parts = []
                    if 'down' in cell:
                        parts.append(f"down clue: {cell['down'][0]}")
                    if 'right' in cell:
                        parts.append(f"right clue: {cell['right'][0]}")
                    if parts:
                        black_cells.append(f"- Cell ({i},{j}): {', '.join(parts)}")
        
        step2 += "\n".join(black_cells[:8])  # Show first 8 for brevity
        if len(black_cells) > 8:
            step2 += f"\n... and {len(black_cells) - 8} more clue cells"
        step2 += "\n\n"
        
        # White cells identification
        step2 += f"**White Cells to Fill:** {len(white_cells)} cells\n"
        white_cell_coords = [f"({i},{j})" for i, j in white_cells[:12]]  # Show first 12
        step2 += f"Coordinates: {', '.join(white_cell_coords)}"
        if len(white_cells) > 12:
            step2 += f" ... (and {len(white_cells) - 12} more)"
        step2 += "\n\n"
        
        # Runs analysis
        step2 += "**Run Structure Analysis:**\n"
        step2 += f"- **Horizontal runs**: {len(horizontal_runs)} total\n"
        for i, run in enumerate(horizontal_runs[:5]):  # Show first 5
            clue_i, clue_j = run['clue']
            step2 += f"  • Run {i+1}: Clue at ({clue_i},{clue_j}) → sum {run['sum']}, cells {run['cells']}\n"
        if len(horizontal_runs) > 5:
            step2 += f"  ... and {len(horizontal_runs) - 5} more horizontal runs\n"
        
        step2 += f"- **Vertical runs**: {len(vertical_runs)} total\n"
        for i, run in enumerate(vertical_runs[:5]):  # Show first 5
            clue_i, clue_j = run['clue']
            step2 += f"  • Run {i+1}: Clue at ({clue_i},{clue_j}) → sum {run['sum']}, cells {run['cells']}\n"
        if len(vertical_runs) > 5:
            step2 += f"  ... and {len(vertical_runs) - 5} more vertical runs\n"
        
        # State reading reflection
        step2 += "\n**Reflection on State Reading:**\n"
        step2 += "The image analysis reveals a well-structured Kakuro puzzle. I can clearly identify:\n"
        step2 += f"- {len(all_runs)} distinct runs with varying difficulty levels\n"
        step2 += f"- Grid density: {len(white_cells)}/{rows*cols} = {len(white_cells)/(rows*cols):.1%} white cells\n"
        
        # Find constraint analysis
        run_lengths = [len(run['cells']) for run in all_runs]
        if run_lengths:
            min_length, max_length = min(run_lengths), max(run_lengths)
            step2 += f"- Run complexity: lengths range from {min_length} to {max_length} cells\n"
            step2 += f"- The most constrained runs (length {min_length}) will be my starting points for logical deduction.\n"
        else:
            step2 += f"- Warning: No valid runs found in the puzzle structure.\n"
            step2 += f"- This may indicate an issue with puzzle generation.\n"
        
        # =============== STEP 3: Detailed Reasoning Process ===============
        step3 = step2 + "\n\n### Step 3: Strategic exploration and detailed reasoning process\n\n"
        step3 += "**Solving Strategy:**\n"
        step3 += "I'll use constraint propagation and logical deduction, starting with the most constrained runs and working systematically through intersections.\n\n"
        
        # Find most constrained runs for detailed analysis
        sorted_runs = sorted(all_runs, key=lambda r: (len(r['cells']), r['sum']))
        
        # Detailed analysis of first few runs
        step3 += "**Phase 1: Analyzing Most Constrained Runs**\n\n"
        
        for phase_idx, run in enumerate(sorted_runs[:min(4, len(sorted_runs))]):
            clue_i, clue_j = run['clue']
            cells = run['cells']
            target_sum = run['sum']
            values = [solution[cell] for cell in cells]
            run_type = "horizontal" if run in horizontal_runs else "vertical"
            run_num = phase_idx + 1
            
            step3 += f"**Run {run_num} ({run_type} at ({clue_i},{clue_j})):**\n"
            step3 += f"- Target: {len(cells)} unique digits summing to {target_sum}\n"
            step3 += f"- Cells: {cells}\n"
            
            # Show constraint analysis
            possible_combinations = self._get_possible_combinations(target_sum, len(cells))
            step3 += f"- Mathematical constraint: {len(possible_combinations)} possible combinations for sum {target_sum} with {len(cells)} digits\n"
            
            if len(possible_combinations) <= 5:
                step3 += f"- Possible combinations: {possible_combinations}\n"
            elif len(possible_combinations) <= 20:
                step3 += f"- Examples: {possible_combinations[:3]}... ({len(possible_combinations)} total)\n"
            else:
                step3 += f"- High combinatorial space ({len(possible_combinations)} possibilities) - will need intersection constraints\n"
            
            # Show reasoning process
            if len(cells) == 2:
                step3 += f"- For 2-cell run: systematically testing pairs that sum to {target_sum}\n"
                step3 += f"- Solution found: {values[0]} + {values[1]} = {target_sum} ✓\n"
            elif len(cells) == 3:
                step3 += f"- For 3-cell run: considering digit uniqueness and sum constraint\n"
                step3 += f"- Testing combinations systematically...\n"
                step3 += f"- Solution: {' + '.join(map(str, values))} = {target_sum} ✓\n"
            else:
                step3 += f"- Complex run requiring careful analysis of {len(cells)} cells\n"
                step3 += f"- Using constraint propagation and backtracking...\n"
                step3 += f"- Final assignment: {' + '.join(map(str, values))} = {target_sum} ✓\n"
            
            step3 += f"- **Values placed**: " + ", ".join([f"({cell[0]},{cell[1]}):{solution[cell]}" for cell in cells]) + "\n\n"
        
        # Intersection analysis
        step3 += "**Phase 2: Intersection Analysis and Constraint Propagation**\n\n"
        
        # Find key intersections
        intersections = []
        for i, run1 in enumerate(all_runs):
            for j, run2 in enumerate(all_runs[i+1:], i+1):
                common_cells = set(run1['cells']) & set(run2['cells'])
                if common_cells:
                    intersections.append({
                        'run1': run1, 'run2': run2, 
                        'intersection': list(common_cells)[0],
                        'run1_type': 'H' if run1 in horizontal_runs else 'V',
                        'run2_type': 'H' if run2 in horizontal_runs else 'V'
                    })
        
        step3 += f"Found {len(intersections)} intersections between runs. Key examples:\n\n"
        
        for i, intersection in enumerate(intersections[:3]):  # Show first 3 intersections
            int_cell = intersection['intersection']
            run1, run2 = intersection['run1'], intersection['run2']
            step3 += f"**Intersection {i+1}** at {int_cell}:\n"
            step3 += f"- {intersection['run1_type']} run: sum {run1['sum']}, cells {run1['cells']}\n"
            step3 += f"- {intersection['run2_type']} run: sum {run2['sum']}, cells {run2['cells']}\n"
            step3 += f"- Shared value: {solution[int_cell]} (must satisfy both runs)\n"
            
            # Verify the intersection constraint
            run1_values = [solution[cell] for cell in run1['cells']]
            run2_values = [solution[cell] for cell in run2['cells']]
            step3 += f"- Verification: {intersection['run1_type']} sum = {' + '.join(map(str, run1_values))} = {sum(run1_values)} ✓\n"
            step3 += f"- Verification: {intersection['run2_type']} sum = {' + '.join(map(str, run2_values))} = {sum(run2_values)} ✓\n\n"
        
        # Solving progression
        step3 += "**Phase 3: Complete Solving Process**\n\n"
        step3 += "Using the constraint propagation approach:\n"
        step3 += "1. **Initial constraints**: Started with most constrained runs (shortest length, limited combinations)\n"
        step3 += "2. **Progressive solving**: Each solved cell reduces possibilities for intersecting runs\n"
        step3 += "3. **Intersection verification**: Ensured all shared cells satisfy multiple run constraints\n"
        step3 += "4. **Backtracking when needed**: Some attempted values violated uniqueness - systematically explored alternatives\n"
        step3 += "5. **Logical deduction**: Used elimination and constraint satisfaction to narrow possibilities\n"
        step3 += "6. **Complete coverage**: Ensured all white cells are assigned values satisfying all constraints\n\n"
        
        step3 += "The solving process required careful balance between mathematical constraints and logical deduction, "
        step3 += "with each step building upon previous assignments to maintain consistency across all runs."
        
        # =============== STEP 4: Solution Validation and Refinement ===============
        step4 = step3 + "\n\n### Step 4: Solution validation and reflection\n\n"
        step4 += "**Final Solution:**\n"
        
        # Present complete solution in organized format
        ordered_solution = sorted(solution.items())
        solution_by_row = {}
        for (row, col), value in ordered_solution:
            if row not in solution_by_row:
                solution_by_row[row] = []
            solution_by_row[row].append(f"({row},{col}):{value}")
        
        step4 += "```\n"
        for row in sorted(solution_by_row.keys()):
            step4 += f"Row {row}: {' '.join(solution_by_row[row])}\n"
        step4 += "```\n\n"
        
        step4 += "**Complete coordinate-value format:**\n"
        step4 += " ".join([f"({cell[0]},{cell[1]}):{value}" for cell, value in ordered_solution])
        step4 += "\n\n"
        
        # Comprehensive validation
        step4 += "**Validation Process:**\n\n"
        
        # Check all horizontal runs
        step4 += "**Horizontal Run Verification:**\n"
        all_horizontal_valid = True
        for i, run in enumerate(horizontal_runs):
            values = [solution[cell] for cell in run['cells']]
            clue_i, clue_j = run['clue']
            actual_sum = sum(values)
            expected_sum = run['sum']
            has_duplicates = len(values) != len(set(values))
            
            status = "✓" if actual_sum == expected_sum and not has_duplicates else "✗"
            if i < 5:  # Show first 5 in detail
                step4 += f"- Run {i+1} at ({clue_i},{clue_j}): {'+'.join(map(str, values))} = {actual_sum} (expect {expected_sum}) {status}\n"
            
            if actual_sum != expected_sum or has_duplicates:
                all_horizontal_valid = False
        
        if len(horizontal_runs) > 5:
            step4 += f"- ... and {len(horizontal_runs) - 5} more horizontal runs all verified ✓\n"
        
        # Check all vertical runs  
        step4 += "\n**Vertical Run Verification:**\n"
        all_vertical_valid = True
        for i, run in enumerate(vertical_runs):
            values = [solution[cell] for cell in run['cells']]
            clue_i, clue_j = run['clue']
            actual_sum = sum(values)
            expected_sum = run['sum']
            has_duplicates = len(values) != len(set(values))
            
            status = "✓" if actual_sum == expected_sum and not has_duplicates else "✗"
            if i < 5:  # Show first 5 in detail
                step4 += f"- Run {i+1} at ({clue_i},{clue_j}): {'+'.join(map(str, values))} = {actual_sum} (expect {expected_sum}) {status}\n"
            
            if actual_sum != expected_sum or has_duplicates:
                all_vertical_valid = False
        
        if len(vertical_runs) > 5:
            step4 += f"- ... and {len(vertical_runs) - 5} more vertical runs all verified ✓\n"
        
        # Overall validation summary
        step4 += "\n**Final Validation Summary:**\n"
        total_runs = len(all_runs)
        step4 += f"- **Total runs checked**: {total_runs} ({len(horizontal_runs)} horizontal + {len(vertical_runs)} vertical)\n"
        step4 += f"- **Sum constraints**: All {total_runs} runs sum correctly to their clue values ✓\n"
        step4 += f"- **Uniqueness constraints**: No duplicate digits within any run ✓\n"
        step4 += f"- **Coverage**: All {len(white_cells)} white cells filled with valid digits (1-9) ✓\n"
        step4 += f"- **Consistency**: All intersection points satisfy multiple run constraints ✓\n\n"
        
        # Reflection on solution quality
        step4 += "**Solution Reflection:**\n"
        step4 += "The solution demonstrates a successful application of constraint satisfaction principles. "
        step4 += f"Starting with the most constrained runs and using systematic constraint propagation, "
        step4 += f"I was able to determine unique values for all {len(white_cells)} cells. "
        step4 += f"The {len(intersections)} intersection points served as crucial verification checkpoints, "
        step4 += "ensuring the solution's mathematical correctness and logical consistency. "
        step4 += "This Kakuro puzzle required careful balance between combinatorial analysis and logical deduction."
        
        return {
            'step1': step1,
            'step2': step2, 
            'step3': step3,
            'step4': step4
        }
    
    def _count_solution_steps(self, puzzle):
        """Count the number of logical steps needed to solve the puzzle."""
        grid = puzzle['grid']
        run_count = 0
        
        for row in grid:
            for cell in row:
                if cell['type'] == 'black':
                    if 'right' in cell:
                        run_count += 1
                    if 'down' in cell:
                        run_count += 1
        
        # Estimate steps based on puzzle complexity
        white_cell_count = sum(1 for row in grid for cell in row if cell['type'] == 'white')
        
        # Greater of (1.5 * run count) or (white cell count)
        return max(int(run_count * 1.5), white_cell_count, 5)
    
    def _get_possible_combinations(self, target_sum, length, used_values=None):
        """Get all possible digit combinations for a run that sum to the target."""
        if used_values is None:
            used_values = set()
        
        def backtrack(remaining_sum, remaining_length, used, current):
            if remaining_length == 0:
                return [current.copy()] if remaining_sum == 0 else []
                
            if remaining_sum <= 0:
                return []
                
            result = []
            for i in range(1, 10):
                if i not in used and i <= remaining_sum:
                    current.append(i)
                    used.add(i)
                    result.extend(backtrack(remaining_sum - i, remaining_length - 1, used, current))
                    current.pop()
                    used.remove(i)
                    
            return result
            
        return backtrack(target_sum, length, used_values.copy(), [])
    
    def _generate_minimal_cot(self, puzzle):
        """Generate minimal CoT for degenerate puzzles with no white cells."""
        grid = puzzle['grid']
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        
        step1 = "Let me analyze this Kakuro puzzle.\n\n"
        step1 += "### Step 1: Understanding the puzzle rules and objectives\n\n"
        step1 += "This appears to be a degenerate Kakuro puzzle with no white cells to fill.\n"
        step1 += "In a standard Kakuro puzzle, white cells would need to be filled with digits 1-9."
        
        step2 = step1 + "\n\n### Step 2: Reading the image and analyzing the initial state\n\n"
        step2 += f"**Grid Analysis**: This is a {rows}×{cols} grid consisting entirely of black cells.\n"
        step2 += "No white cells are present that require digit placement."
        
        step3 = step2 + "\n\n### Step 3: Strategic exploration and reasoning process\n\n"
        step3 += "Since there are no white cells to fill, no solving strategy is required.\n"
        step3 += "This puzzle is already in its complete state."
        
        step4 = step3 + "\n\n### Step 4: Solution validation and reflection\n\n"
        step4 += "**Final Solution**: No cells to fill.\n"
        step4 += "**Validation**: The puzzle contains no white cells, so no solution is needed."
        
        return {
            'step1': step1,
            'step2': step2,
            'step3': step3,
            'step4': step4
        }




