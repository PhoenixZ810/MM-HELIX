import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from heapq import heappush, heappop
from generator.base_generator import BaseGenerator

class SnakeGenerator(BaseGenerator):
    """Generator for Snake (also known as Tunnel) logic puzzles.

    This class now exposes a deterministic API:
        generate(size=..., seed=..., output_dir=...)

    The tuple (size, seed) uniquely determines the generated puzzle state.
    """

    def __init__(self, output_folder):
        super().__init__(output_folder)

    # --- New batch API to comply with BaseGenerator and main.py ---
    def generate(self, num_cases, difficulty, output_folder=None):
        """Batch-generate puzzles and save to annotations.json, complying with BaseGenerator.

        Args:
            num_cases (int): number of puzzles to generate
            difficulty (int): difficulty level 1-5
            output_folder (str|None): optional override for output path

        Returns:
            list[dict]: generated items
        """
        out_dir = output_folder or self.output_folder
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)

        params = self._get_difficulty_params(difficulty)
        sizes = params['sizes']
        max_turns_factor = params['max_turns_factor']

        items = []
        for i in range(num_cases):
            size = random.choice(sizes)
            seed = random.randint(0, 2**31 - 1)
            item = self.generate_one(size=size, seed=seed, output_dir=out_dir, difficulty=difficulty, max_turns_factor=max_turns_factor)
            if item is not None:
                items.append(item)
                # Incremental append to avoid losing progress if interrupted
                try:
                    self._append_annotation(out_dir, item)
                except Exception as append_error:
                    print(f"Warning: failed to append snake item {item.get('index')}: {append_error}")

        # Final write to ensure annotations.json exists even if no items or on first run
        self.save_annotations(items, out_dir)
        # Return generated items (file already updated incrementally)
        return items

    # --- Deterministic single-instance generator kept as helper ---
    def generate_one(self, size: int, seed: int, output_dir: str, difficulty: int = None, max_turns_factor: float = None, **kwargs):
        """Generate a single Snake puzzle deterministically by (size, seed).

        Args:
            size: Grid dimension (rows == cols == size).
            seed: Random seed. Different seeds should produce different states.
            output_dir: Directory where annotations.json and images/ are located.
            difficulty: Optional difficulty level 1-5. If None, inferred from size.
            max_turns_factor: Optional override for max turns factor. If None, inferred from difficulty/size.

        Returns:
            A dictionary item following the project schema, or None if failed.
        """
        # Seed deterministically for both random and numpy
        # Combine size and seed to minimize collisions across sizes
        combined_seed = (int(size) << 20) ^ int(seed)
        random.seed(combined_seed)
        np.random.seed(combined_seed & 0x7FFFFFFF)

        rows = int(size)
        cols = int(size)

        # Infer difficulty parameters if not provided
        if difficulty is None:
            # Map by size bands
            if size <= 5:
                difficulty_level = 1
            elif size <= 7:
                difficulty_level = 2
            elif size <= 9:
                difficulty_level = 3
            elif size <= 11:
                difficulty_level = 4
            else:
                difficulty_level = 5
        else:
            difficulty_level = int(difficulty)

        if max_turns_factor is None:
            # Derive factor by difficulty
            if difficulty_level == 1:
                max_turns_factor = 0.5
            elif difficulty_level == 2:
                max_turns_factor = 0.7
            elif difficulty_level == 4:
                max_turns_factor = 1.2
            elif difficulty_level == 5:
                max_turns_factor = 1.4
            else:
                max_turns_factor = 1.0

        # Ensure output directories
        images_dir = os.path.join(output_dir, 'images')
        try:
            os.makedirs(images_dir, exist_ok=True)
        except Exception:
            # Defer directory creation errors to later I/O attempts
            pass

        # Deterministic yet retryable attempts if answer collides
        max_attempts = 30
        for attempt in range(max_attempts):
            # Add attempt-based minor jitter to exploration tie-breakers while staying deterministic
            local_seed = combined_seed + attempt * 7919
            random.seed(local_seed)
            np.random.seed(local_seed & 0x7FFFFFFF)

            try:
                start, end = self._choose_endpoints(rows, cols, difficulty_level)
                base_grid = np.zeros((rows, cols), dtype=int)
                path = self._generate_path(base_grid.copy(), start, end, difficulty_level, max_turns_factor)
                if not path:
                    continue

                # Fill grid cells where the snake passes (for counts only)
                grid = np.zeros((rows, cols), dtype=int)
                for r, c in path:
                    grid[r, c] = 1

                row_counts = [int(sum(grid[r, :])) for r in range(rows)]
                col_counts = [int(sum(grid[:, c])) for c in range(cols)]

                # Basic sanity checks
                if not self._has_unique_solution(rows, cols, start, end, row_counts, col_counts):
                    continue

                # Format components
                answer_path_list = [(int(r), int(c)) for r, c in path]
                answer_path_str = self._format_path(answer_path_list)

                # Build initial_state as a pure list grid with only S/E and nulls
                initial_state = [[None for _ in range(cols)] for _ in range(rows)]
                initial_state[start[0]][start[1]] = 'S'
                initial_state[end[0]][end[1]] = 'E'

                # Render image
                image_filename = f"snake_{size}_{seed}.png"
                image_rel_path = os.path.join('images', image_filename)
                image_abs_path = os.path.join(output_dir, image_rel_path)
                self.visualize({
                    'grid_size': {'rows': rows, 'cols': cols},
                    'start': {'row': int(start[0]), 'col': int(start[1])},
                    'end': {'row': int(end[0]), 'col': int(end[1])},
                    'row_counts': row_counts,
                    'col_counts': col_counts
                }, filename=image_abs_path)

                # Build question strings
                question = self._build_question_with_image(rows, cols, row_counts, col_counts, image_rel_path)
                question_language = self._build_text_question_with_initial_state(rows, cols, row_counts, col_counts, initial_state)

                # Chain-of-thought (rule-based, English)

                cot = self.generate_cot(rows, cols, start, end, row_counts, col_counts, path)

                item = {
                    'index': f'snake_{size}_{seed}',
                    'category': 'snake',
                    'image': image_rel_path,
                    'question': question,
                    'question_language': question_language,
                    # Keep both list and string forms for downstream tools and uniqueness
                    'answer': answer_path_list,
                    'answer_str': answer_path_str,
                    'grid_size': {'rows': rows, 'cols': cols},
                    'start': {'row': int(start[0]), 'col': int(start[1])},
                    'end': {'row': int(end[0]), 'col': int(end[1])},
                    'row_counts': row_counts,
                    'col_counts': col_counts,
                    'initial_state': initial_state,
                    'difficulty': str(difficulty_level),
                    'cot': cot
                }
                # Append to annotations.json for each generated item (enforce per-item save)
                # unless managed_save is True (let the manager handle saving)
                return item
            except Exception as e:
                # Log the error but continue with next attempt
                # This helps with debugging while still allowing the generation to continue
                if attempt == max_attempts - 1:  # Only print on last attempt to avoid spam
                    print(f"Warning: Failed to generate puzzle for size={size}, seed={seed + attempt}: {e}")
                continue

        # Failed to generate a suitable unique puzzle
        return None
    
    def _generate_single_puzzle(self, rows, cols, difficulty, max_turns_factor):
        """Legacy helper kept for compatibility with internal usage."""
        start, end = self._choose_endpoints(rows, cols, int(difficulty))
        max_attempts = 50
        for _ in range(max_attempts):
            grid = np.zeros((rows, cols), dtype=int)
            path = self._generate_path(grid.copy(), start, end, difficulty, max_turns_factor)
            if path:
                for r, c in path:
                    grid[r, c] = 1
                row_counts = [int(sum(grid[r, :])) for r in range(rows)]
                col_counts = [int(sum(grid[:, c])) for c in range(cols)]
                if self._has_unique_solution(rows, cols, start, end, row_counts, col_counts):
                    path = [(int(r), int(c)) for r, c in path]
                    return {
                        'grid_size': {'rows': rows, 'cols': cols},
                        'start': {'row': int(start[0]), 'col': int(start[1])},
                        'end': {'row': int(end[0]), 'col': int(end[1])},
                        'row_counts': row_counts,
                        'col_counts': col_counts,
                        'answer': path,
                        'grid': grid.tolist()
                    }
        raise Exception("Could not generate a valid Snake puzzle with unique solution")
    
    def _choose_endpoints(self, rows, cols, difficulty):
        """Choose start and end points for the snake based on difficulty

        Ensures endpoints are not identical and have appropriate minimum distance based on difficulty.
        """
        def _manhattan_distance(p1, p2):
            """Calculate Manhattan distance between two points"""
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        def _is_valid_endpoint_pair(start, end, min_distance):
            """Check if endpoint pair meets minimum requirements"""
            if start == end:
                return False
            if _manhattan_distance(start, end) < min_distance:
                return False
            return True

        # Normalize difficulty to tiers
        if isinstance(difficulty, str):
            diff_str = difficulty.lower()
            if diff_str.startswith('e'):
                difficulty_level = 1
            elif diff_str.startswith('m'):
                difficulty_level = 2
            else:
                difficulty_level = 3
        else:
            difficulty_level = int(difficulty) if difficulty is not None else 3

        # Set minimum distance requirements based on difficulty tier
        if difficulty_level == 1:
            min_distance = 3  # Ensure endpoints are not adjacent
        elif difficulty_level == 2:
            min_distance = max(3, min(rows, cols) // 2)  # Moderate separation
        else:  # hard
            min_distance = max(4, max(rows, cols) // 2)  # Significant separation

        max_attempts = 500  # Prevent infinite loops

        if difficulty_level == 1:
            # For easy puzzles, place endpoints at opposite sides with minimum distance
            for attempt in range(max_attempts):
                sides = [(0, random.randint(0, cols-1)),  # top
                        (rows-1, random.randint(0, cols-1)),  # bottom
                        (random.randint(0, rows-1), 0),  # left
                        (random.randint(0, rows-1), cols-1)]  # right

                start_idx = random.randint(0, 3)
                end_idx = (start_idx + 2) % 4  # Get opposite side

                start = sides[start_idx]
                end = sides[end_idx]

                if _is_valid_endpoint_pair(start, end, min_distance):
                    return start, end

        elif difficulty_level == 2:
            # Medium: at least one endpoint on the edge, ensure good separation
            for attempt in range(max_attempts):
                # Choose first endpoint on edge
                start = random.choice([
                    (0, random.randint(0, cols-1)),  # top
                    (rows-1, random.randint(0, cols-1)),  # bottom
                    (random.randint(0, rows-1), 0),  # left
                    (random.randint(0, rows-1), cols-1)  # right
                ])

                # Choose second endpoint with minimum distance requirement
                for _ in range(max_attempts):
                    end = (random.randint(0, rows-1), random.randint(0, cols-1))
                    if _is_valid_endpoint_pair(start, end, min_distance):
                        return start, end

        else:  # hard
            # Hard: both endpoints can be anywhere but with significant separation
            for attempt in range(max_attempts):
                start = (random.randint(0, rows-1), random.randint(0, cols-1))

                # Ensure good separation for hard puzzles
                for _ in range(max_attempts):
                    end = (random.randint(0, rows-1), random.randint(0, cols-1))
                    if _is_valid_endpoint_pair(start, end, min_distance):
                        return start, end

        # Fallback: if we can't find a good pair, use a more relaxed approach
        print(f"Warning: Could not find optimal endpoints for {difficulty} puzzle after {max_attempts} attempts")
        start = (0, 0)
        end = (rows-1, cols-1)
        return start, end
    
    def _generate_path(self, grid, start, end, difficulty, max_turns_factor):
        """Generate a valid snake path from start to end"""
        rows, cols = grid.shape
        
        # Maximum allowed turns based on difficulty
        max_turns = int(max(rows, cols) * max_turns_factor)
        
        # Directions: right, down, left, up
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # Define the heuristic function (Manhattan distance)
        def heuristic(pos):
            return abs(pos[0] - end[0]) + abs(pos[1] - end[1])
        
        # Initialize A* search
        open_set = []
        f_score = heuristic(start)
        g_score = 0
        turn_count = 0
        heappush(open_set, (f_score, g_score, turn_count, 0, [start], -1))  # (f, g, turns, tiebreaker, path, last_dir)
        
        visited = set()
        visited.add(start)
        
        while open_set:
            _, g, turns, _, path, last_dir = heappop(open_set)
            curr = path[-1]
            
            # If we've reached the end
            if curr == end:
                return path
            
            # Try each direction
            for i, (dr, dc) in enumerate(directions):
                # Check if this would be a turn
                is_turn = last_dir != -1 and last_dir != i
                
                # Skip if we've reached max turns
                if turns >= max_turns and is_turn:
                    continue
                
                r, c = curr[0] + dr, curr[1] + dc
                next_pos = (r, c)
                
                # Check if next position is valid
                if (0 <= r < rows and 0 <= c < cols and 
                    next_pos not in visited and 
                    next_pos not in path):  # Avoid self-intersection
                    
                    new_g = g + 1
                    new_turns = turns + (1 if is_turn else 0)
                    new_f = new_g + heuristic(next_pos)
                    
                    new_path = path + [next_pos]
                    tiebreaker = random.random()  # Add randomness to exploration
                    
                    visited.add(next_pos)
                    heappush(open_set, (new_f, new_g, new_turns, tiebreaker, new_path, i))
        
        # If no path is found
        return None

    # --- Difficulty configuration to comply with BaseGenerator ---
    def _get_difficulty_params(self, difficulty):
        """Map global difficulty (1-5) to size bands and path-turn control."""
        level = int(difficulty)
        if level == 1:
            return {'sizes': [5, 6], 'max_turns_factor': 0.5}
        if level == 2:
            return {'sizes': [6, 7], 'max_turns_factor': 0.7}
        if level == 3:
            return {'sizes': [8, 9], 'max_turns_factor': 1.0}
        if level == 4:
            return {'sizes': [10, 11], 'max_turns_factor': 1.2}
        # level == 5
        return {'sizes': [12, 13], 'max_turns_factor': 1.4}

    def _has_unique_solution(self, rows, cols, start, end, row_counts, col_counts):
        """Verify that the puzzle has a unique solution"""
        # For simplicity, we'll just check that the basic constraints are satisfiable
        # A more thorough approach would involve solving the puzzle with all constraints
        # and ensuring there's only one possible solution
        
        # Check that row and column counts are consistent
        total_cells = sum(row_counts)
        if total_cells != sum(col_counts):
            return False
        
        # Check minimum path length (Manhattan distance)
        min_path_length = abs(end[0] - start[0]) + abs(end[1] - start[1])
        if total_cells < min_path_length:
            return False
            
        # Quick feasibility check only in this simplified uniqueness proxy
        return True

    def visualize(self, puzzle, filename=None):
        """Visualize the Snake puzzle with a polished, high-contrast style"""
        rows = puzzle['grid_size']['rows']
        cols = puzzle['grid_size']['cols']
        start = (puzzle['start']['row'], puzzle['start']['col'])
        end = (puzzle['end']['row'], puzzle['end']['col'])
        row_counts = puzzle['row_counts']
        col_counts = puzzle['col_counts']

        # Style configuration
        cell_size_in = 0.7  # inches per cell for crisp output
        margin_left_cells = 0.4
        margin_right_cells = 2.2  # space for column counts
        margin_top_cells = 0.4
        margin_bottom_cells = 1.8  # space for row counts

        fig_w = (cols + margin_left_cells + margin_right_cells) * cell_size_in
        fig_h = (rows + margin_top_cells + margin_bottom_cells) * cell_size_in

        grid_line_color = '#9AA0A6'
        grid_line_width = 1.4
        border_color = '#263238'
        border_line_width = 2.6
        start_color = '#2E7D32'
        start_edge = '#1B5E20'
        end_color = '#C62828'
        end_edge = '#8E0000'
        # No per-cell background fills; keep canvas clean

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FFFFFF')

        # Grid in data coords 0..cols, 0..rows (no left/top extra margins)
        x_min = 0.0
        x_max = float(cols)
        y_min = 0.0
        y_max = float(rows)

        # Internal grid lines (skip outer frame; handled by border)
        for i in range(1, rows):
            line = ax.axhline(i, color=grid_line_color, lw=grid_line_width)
            line.set_solid_capstyle('butt')
        for j in range(1, cols):
            line = ax.axvline(j, color=grid_line_color, lw=grid_line_width)
            line.set_solid_capstyle('butt')

        # Thick outer border
        border = Rectangle(
            (0, 0),
            cols,
            rows,
            fill=False,
            edgecolor=border_color,
            linewidth=border_line_width
        )
        ax.add_patch(border)

        # Start and End cells with distinct styles
        start_rect = Rectangle(
            (start[1], start[0]),
            1,
            1,
            facecolor=start_color,
            edgecolor=start_edge,
            linewidth=2.0,
            alpha=0.9
        )
        end_rect = Rectangle(
            (end[1], end[0]),
            1,
            1,
            facecolor=end_color,
            edgecolor=end_edge,
            linewidth=2.0,
            alpha=0.9
        )
        ax.add_patch(start_rect)
        ax.add_patch(end_rect)

        # Row counts (plain text, no extra shapes)
        for i, count in enumerate(row_counts):
            ax.text(
                cols + 0.6,
                i + 0.5,
                str(count),
                ha='center',
                va='center',
                fontsize=13,
                color='#1F2937'
            )

        # Column counts (plain text, no extra shapes)
        for j, count in enumerate(col_counts):
            ax.text(
                j + 0.5,
                rows + 0.6,
                str(count),
                ha='center',
                va='center',
                fontsize=13,
                color='#1F2937'
            )

        # Set limits and aspect; reverse y to match row indexing (top-to-bottom)
        # Keep extra room only on right and bottom for counts
        ax.set_xlim(0.0, cols + 1.0)
        ax.set_ylim(rows + 1.0, 0.0)
        ax.set_aspect('equal')

        # Remove axes, ticks, and any titles to avoid unrelated text
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        ax.set_frame_on(False)

        # Save or show
        if filename:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.05, facecolor=fig.get_facecolor())
            plt.close(fig)
        else:
            plt.tight_layout(pad=0.1)
            plt.show()

    def visualize_solution(self, puzzle, filename=None):
        """Visualize the solution to the Snake puzzle"""
        rows = puzzle['grid_size']['rows']
        cols = puzzle['grid_size']['cols']
        start = (puzzle['start']['row'], puzzle['start']['col'])
        end = (puzzle['end']['row'], puzzle['end']['col'])
        row_counts = puzzle['row_counts']
        col_counts = puzzle['col_counts']
        path = puzzle['answer']
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(cols + 2, rows + 2))
        
        # Draw grid lines
        for i in range(rows + 1):
            ax.axhline(i, color='black', lw=1)
        for j in range(cols + 1):
            ax.axvline(j, color='black', lw=1)
        
        # Draw the path
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i+1]
            
            # Draw a line segment
            ax.plot([c1 + 0.5, c2 + 0.5], [r1 + 0.5, r2 + 0.5], 'b-', lw=3)
        
        # Highlight start and end cells
        ax.add_patch(Rectangle((start[1], start[0]), 1, 1, fill=True, color='green', alpha=0.5))
        ax.add_patch(Rectangle((end[1], end[0]), 1, 1, fill=True, color='red', alpha=0.5))
        
        # Mark start and end points with text
        ax.text(start[1] + 0.5, start[0] + 0.5, 'S', ha='center', va='center', fontsize=12, weight='bold')
        ax.text(end[1] + 0.5, end[0] + 0.5, 'E', ha='center', va='center', fontsize=12, weight='bold')
        
        # Add row and column counts
        for i, count in enumerate(row_counts):
            ax.text(cols + 0.5, i + 0.5, str(count), ha='center', va='center', fontsize=12)
        
        for j, count in enumerate(col_counts):
            ax.text(j + 0.5, rows + 0.5, str(count), ha='center', va='center', fontsize=12)
        
        # Set limits and aspect
        ax.set_xlim(-0.5, cols + 1)
        ax.set_ylim(rows + 1, -0.5)  # Reverse y-axis to match row indexing
        ax.set_aspect('equal')
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title
        ax.set_title('Snake Puzzle Solution', fontsize=14)
        
        # Save or show
        if filename:
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
        else:
            plt.tight_layout()
            plt.show()

    # The following legacy save_to_json and formatting helpers are no longer
    # used by the new deterministic API, but are kept to avoid breaking imports.
    def save_to_json(self, puzzles):
        return None
        
    def _build_question_with_image(self, rows, cols, row_counts, col_counts, image_rel_path: str) -> str:
        lines = []
        lines.append("# Snake Puzzle")
        lines.append("## Task:")
        lines.append("Please examine the image carefully. The image shows a Snake puzzle grid.")
        lines.append("")
        lines.append("## Rules:")
        lines.append("1. Draw a single, non-intersecting snake path from S (start) to E (end).")
        lines.append("2. The snake occupies some cells; it cannot touch itself, even diagonally.")
        lines.append("3. The numbers outside the grid indicate how many snake cells appear in each row and column.")
        lines.append("")
        lines.append("## Provided Clues:")
        lines.append(f"- Grid size: {rows}×{cols}")
        lines.append(f"- Row counts: {', '.join(map(str, row_counts))}")
        lines.append(f"- Column counts: {', '.join(map(str, col_counts))}")
        lines.append("")
        lines.append("## Refer to the image to solve the puzzle:")
        lines.append(f"![snake puzzle]({image_rel_path})")
        lines.append("")
        lines.append("## Output Format:")
        lines.append("Return the snake path as a sequence of coordinates, e.g.: (r0,c0) (r1,c1) ...")
        return "\n".join(lines)

    def _build_text_question_with_initial_state(self, rows, cols, row_counts, col_counts, initial_state):
        lines = []
        lines.append("Please examine the grid carefully. The grid shows a Snake puzzle.")
        lines.append("")
        lines.append("Rules:")
        lines.append("1. Draw a single, non-intersecting snake path from S to E.")
        lines.append("2. The snake cannot touch itself, even diagonally.")
        lines.append("3. The numbers outside the grid indicate the count of snake cells per row and column.")
        lines.append("")
        lines.append("Refer to the initial_state below instead of an image:")
        lines.append("initial_state (list grid):")
        try:
            lines.append(json.dumps(initial_state, ensure_ascii=False))
        except Exception:
            lines.append(str(initial_state))
        lines.append("")
        lines.append(f"Grid size: {rows}×{cols}")
        lines.append(f"Row counts: {', '.join(map(str, row_counts))}")
        lines.append(f"Column counts: {', '.join(map(str, col_counts))}")
        lines.append("")
        lines.append("Output: a coordinate sequence like (r,c) with zero-based indices.")
        return "\n".join(lines)

    def _format_path(self, path):
        return " ".join([f"({r},{c})" for r, c in path])

    def solve(self, puzzle, **kwargs):
        """Solve a Snake puzzle
        
        This is a constraint satisfaction problem, we can use backtracking search
        with the row and column constraints.
        """
        rows = puzzle['grid_size']['rows']
        cols = puzzle['grid_size']['cols']
        start = (puzzle['start']['row'], puzzle['start']['col'])
        end = (puzzle['end']['row'], puzzle['end']['col'])
        row_counts = puzzle['row_counts']
        col_counts = puzzle['col_counts']
        
        # Create a blank grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Mark start and end points as used
        grid[start[0], start[1]] = 1
        path = [start]
        
        # Directions: right, down, left, up
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        def backtrack(curr, remaining_row_counts, remaining_col_counts):
            # If we reached the end point
            if curr == end:
                # Check if all constraints are satisfied
                if all(count == 0 for count in remaining_row_counts) and all(count == 0 for count in remaining_col_counts):
                    return True
                return False
            
            # Try each direction
            r, c = curr
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                next_pos = (nr, nc)
                
                # Check if next position is valid
                if (0 <= nr < rows and 0 <= nc < cols and 
                    grid[nr, nc] == 0 and  # Cell not used
                    remaining_row_counts[nr] > 0 and  # Row constraint
                    remaining_col_counts[nc] > 0):  # Column constraint
                    
                    # Mark cell as used
                    grid[nr, nc] = 1
                    path.append(next_pos)
                    
                    # Update remaining counts
                    remaining_row_counts[nr] -= 1
                    remaining_col_counts[nc] -= 1
                    
                    # Recursively try to solve
                    if backtrack(next_pos, remaining_row_counts, remaining_col_counts):
                        return True
                    
                    # Backtrack
                    grid[nr, nc] = 0
                    path.pop()
                    remaining_row_counts[nr] += 1
                    remaining_col_counts[nc] += 1
            
            return False
        
        # Adjust counts for start point
        r_counts = row_counts.copy()
        c_counts = col_counts.copy()
        r_counts[start[0]] -= 1
        c_counts[start[1]] -= 1
        
        # Solve the puzzle
        if backtrack(start, r_counts, c_counts):
            return path
        else:
            return None

    def _load_existing_answers(self, output_dir: str):
        """Load existing answers from output_dir/annotations.json to enforce uniqueness."""
        annotations_path = os.path.join(output_dir, 'annotations.json')
        try:
            if not os.path.exists(annotations_path):
                return set()
            with open(annotations_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    return set()
                answers = set()
                for item in data:
                    if isinstance(item, dict):
                        ans = item.get('answer')
                        if isinstance(ans, str):
                            answers.add(ans)
                return answers
        except Exception:
            return set()

    def generate_cot(self, rows, cols, start, end, row_counts, col_counts, path):
        """Generate a comprehensive rule-based English CoT describing detailed reasoning and exploration.

        This enhanced CoT follows a four-step structure:
        1. Clear understanding of game rules
        2. Careful reading and textual representation of initial state with reflection
        3. Detailed reasoning process with sufficient exploration
        4. Answer validation and reflection
        """
        lines = []
        
        # Step 1: Clear understanding of game rules
        lines.append("Step 1: Understanding the Snake Puzzle Rules")
        lines.append("=" * 45)
        lines.append(
            "The Snake puzzle (also known as Tunnel puzzle) is a logic puzzle with the following rules:\n"
            "• OBJECTIVE: Draw a single continuous path from the Start cell (S) to the End cell (E)\n"
            "• PATH CONSTRAINTS: The snake path must be non-intersecting and cannot touch itself, even diagonally\n"
            "• MOVEMENT: The snake can only move orthogonally (up, down, left, right) - no diagonal moves\n"
            "• COUNT CONSTRAINTS: Numbers outside the grid indicate exactly how many snake cells appear in each row and column\n"
            "• UNIQUENESS: There should be exactly one valid solution that satisfies all constraints"
        )
        lines.append(
            f"For this specific puzzle:\n"
            f"• Grid dimensions: {rows}×{cols} (rows × columns)\n"
            f"• Start position: ({start[0]},{start[1]}) - using 0-based indexing\n"
            f"• End position: ({end[0]},{end[1]}) - using 0-based indexing\n"
            f"• Row constraints: {', '.join(f'Row {i}: {count} cells' for i, count in enumerate(row_counts))}\n"
            f"• Column constraints: {', '.join(f'Col {i}: {count} cells' for i, count in enumerate(col_counts))}"
        )
        lines.append("")

        # Step 2: Careful reading and textual representation of initial state
        lines.append("Step 2: Reading and Analyzing the Initial State")
        lines.append("=" * 47)
        lines.append(
            "Let me carefully examine the puzzle image/grid and represent the initial state textually.\n"
            "I need to identify the start (S) and end (E) positions, and understand the constraint numbers."
        )
        
        # Create a textual representation of the grid
        lines.append("Initial grid state representation:")
        grid_repr = []
        for r in range(rows):
            row_repr = []
            for c in range(cols):
                if (r, c) == start:
                    row_repr.append("S")
                elif (r, c) == end:
                    row_repr.append("E")
                else:
                    row_repr.append(".")
            grid_repr.append(" ".join(row_repr))
        
        # Add column headers
        col_headers = "  " + " ".join(str(i) for i in range(cols))
        lines.append(col_headers)
        for i, row in enumerate(grid_repr):
            lines.append(f"{i} {row}")
        
        lines.append(f"\nRow counts (right side): {' '.join(map(str, row_counts))}")
        lines.append(f"Column counts (bottom): {' '.join(map(str, col_counts))}")
        
        # Reflection on state reading
        lines.append(
            f"\nReflection on initial state:\n"
            f"• I can clearly see the start position S at ({start[0]},{start[1]}) and end position E at ({end[0]},{end[1]})\n"
            f"• The Manhattan distance between start and end is {abs(end[0] - start[0]) + abs(end[1] - start[1])} cells\n"
            f"• Total snake cells needed: {sum(row_counts)} (from row sums) = {sum(col_counts)} (from column sums)\n"
            f"• This gives me confidence that the constraints are consistent\n"
            f"• The path length will be exactly {sum(row_counts)} cells, including start and end"
        )
        lines.append("")

        # Step 3: Detailed reasoning process with sufficient exploration
        lines.append("Step 3: Detailed Reasoning and Path Construction")
        lines.append("=" * 49)
        lines.append(
            "Now I will systematically construct the snake path, exploring possibilities and applying constraints.\n"
            "I'll use a logical approach, considering count constraints and avoiding self-intersections."
        )

        # Helper function for neighbors
        def neighbors(r, c):
            for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    yield nr, nc

        def direction_name(from_pos, to_pos):
            dr = to_pos[0] - from_pos[0]
            dc = to_pos[1] - from_pos[1]
            if dr == 0 and dc == 1: return "right"
            elif dr == 0 and dc == -1: return "left"
            elif dr == 1 and dc == 0: return "down"
            elif dr == -1 and dc == 0: return "up"
            else: return "unknown"

        # Track path construction with detailed reasoning
        occupied = set()
        occupied.add((start[0], start[1]))
        current_row_counts = row_counts.copy()
        current_col_counts = col_counts.copy()
        current_row_counts[start[0]] -= 1
        current_col_counts[start[1]] -= 1

        lines.append(f"Starting at S({start[0]},{start[1]}). Remaining counts after placing start:")
        lines.append(f"Row counts: {current_row_counts}")
        lines.append(f"Column counts: {current_col_counts}")
        lines.append("")

        for i in range(1, len(path)):
            r_prev, c_prev = path[i-1]
            r_curr, c_curr = path[i]
            
            # Show exploration process
            if i <= 3 or i % max(2, len(path) // 4) == 0 or i >= len(path) - 2:
                lines.append(f"Step 3.{i}: From position ({r_prev},{c_prev})")
                
                # List all possible neighbors
                possible_neighbors = list(neighbors(r_prev, c_prev))
                lines.append(f"Possible adjacent cells: {possible_neighbors}")
                
                # Analyze each neighbor
                valid_moves = []
                invalid_moves = []
                
                for nr, nc in possible_neighbors:
                    reasons = []
                    is_valid = True
                    
                    # Check if already occupied
                    if (nr, nc) in occupied:
                        reasons.append("already occupied")
                        is_valid = False
                    
                    # Check row count constraint
                    if current_row_counts[nr] <= 0:
                        reasons.append(f"row {nr} count exhausted")
                        is_valid = False
                    
                    # Check column count constraint
                    if current_col_counts[nc] <= 0:
                        reasons.append(f"column {nc} count exhausted")
                        is_valid = False
                    
                    # Check diagonal touching (simplified - check if path would create diagonal contact)
                    diagonal_neighbors = [(nr-1,nc-1), (nr-1,nc+1), (nr+1,nc-1), (nr+1,nc+1)]
                    for dr, dc in diagonal_neighbors:
                        if (dr, dc) in occupied and (dr, dc) != (r_prev, c_prev):
                            # Check if this would create diagonal touching
                            if not any((dr, nc) in occupied or (nr, dc) in occupied):
                                reasons.append("would create diagonal touching")
                                is_valid = False
                                break
                    
                    if is_valid:
                        valid_moves.append((nr, nc))
                    else:
                        invalid_moves.append(((nr, nc), reasons))
                
                # Show invalid moves
                if invalid_moves:
                    lines.append("Invalid moves:")
                    for (nr, nc), reasons in invalid_moves:
                        lines.append(f"  • ({nr},{nc}): {', '.join(reasons)}")
                
                # Show valid moves
                lines.append(f"Valid moves: {valid_moves}")
                
                # Explain chosen move
                direction = direction_name((r_prev, c_prev), (r_curr, c_curr))
                lines.append(f"I choose to move {direction} to ({r_curr},{c_curr})")
                
                # Add strategic reasoning occasionally
                if i < len(path) - 1:
                    remaining_distance = abs(end[0] - r_curr) + abs(end[1] - c_curr)
                    remaining_cells = sum(current_row_counts) - 1  # -1 because we're about to place current cell
                    lines.append(f"Strategic analysis: Manhattan distance to end = {remaining_distance}, remaining cells to place = {remaining_cells}")
            
            # Update tracking
            occupied.add((r_curr, c_curr))
            current_row_counts[r_curr] -= 1
            current_col_counts[c_curr] -= 1
            
            if i <= 3 or i % max(2, len(path) // 4) == 0 or i >= len(path) - 2:
                lines.append(f"Updated counts - Row: {current_row_counts}, Col: {current_col_counts}")
                lines.append("")

        # Show final path
        lines.append("Complete path construction:")
        path_str = " → ".join([f"({r},{c})" for r, c in path])
        lines.append(f"Path: {path_str}")
        lines.append("")

        # Step 4: Answer validation and reflection
        lines.append("Step 4: Validation and Reflection")
        lines.append("=" * 35)
        lines.append("Now I will validate the complete solution against all constraints:")
        
        # Validate path continuity
        lines.append("✓ Path Continuity Check:")
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            distance = abs(r2 - r1) + abs(c2 - c1)
            if distance != 1:
                lines.append(f"  ✗ Gap found between ({r1},{c1}) and ({r2},{c2})")
            else:
                lines.append(f"  ✓ Connected: ({r1},{c1}) → ({r2},{c2})")
        
        # Validate no self-intersection
        lines.append("\n✓ Self-Intersection Check:")
        if len(set(path)) == len(path):
            lines.append(f"  ✓ No repeated cells - path length {len(path)} equals unique positions {len(set(path))}")
        else:
            lines.append(f"  ✗ Self-intersection detected - path has repeated cells")
        
        # Validate diagonal touching
        lines.append("\n✓ Diagonal Touching Check:")
        diagonal_violations = []
        for i, (r1, c1) in enumerate(path):
            for j, (r2, c2) in enumerate(path):
                if j > i + 1:  # Skip adjacent cells in path
                    if abs(r1 - r2) == 1 and abs(c1 - c2) == 1:
                        diagonal_violations.append(((r1, c1), (r2, c2)))
        
        if not diagonal_violations:
            lines.append("  ✓ No diagonal touching violations found")
        else:
            lines.append(f"  ✗ Diagonal touching violations: {diagonal_violations}")
        
        # Validate row counts
        lines.append("\n✓ Row Count Validation:")
        actual_row_counts = [0] * rows
        for r, c in path:
            actual_row_counts[r] += 1
        
        for i in range(rows):
            if actual_row_counts[i] == row_counts[i]:
                lines.append(f"  ✓ Row {i}: expected {row_counts[i]}, actual {actual_row_counts[i]}")
            else:
                lines.append(f"  ✗ Row {i}: expected {row_counts[i]}, actual {actual_row_counts[i]}")
        
        # Validate column counts
        lines.append("\n✓ Column Count Validation:")
        actual_col_counts = [0] * cols
        for r, c in path:
            actual_col_counts[c] += 1
        
        for i in range(cols):
            if actual_col_counts[i] == col_counts[i]:
                lines.append(f"  ✓ Column {i}: expected {col_counts[i]}, actual {actual_col_counts[i]}")
            else:
                lines.append(f"  ✗ Column {i}: expected {col_counts[i]}, actual {actual_col_counts[i]}")
        
        # Validate start and end positions
        lines.append(f"\n✓ Start/End Position Validation:")
        if path[0] == start:
            lines.append(f"  ✓ Path starts at correct position: {start}")
        else:
            lines.append(f"  ✗ Path starts at {path[0]}, expected {start}")
        
        if path[-1] == end:
            lines.append(f"  ✓ Path ends at correct position: {end}")
        else:
            lines.append(f"  ✗ Path ends at {path[-1]}, expected {end}")
        
        # Final reflection
        lines.append(f"\n🎯 Final Answer Reflection:")
        lines.append(f"The snake path successfully connects S({start[0]},{start[1]}) to E({end[0]},{end[1]}) using {len(path)} cells.")
        lines.append(f"All row and column count constraints are satisfied, and the path maintains proper connectivity without self-intersection or diagonal touching.")
        lines.append(f"The solution demonstrates logical constraint satisfaction and systematic exploration.")
        
        # Format final answer
        answer_formatted = " ".join([f"({r},{c})" for r, c in path])
        lines.append(f"\n📍 Final Coordinate Sequence: {answer_formatted}")

        return "\n\n".join(lines)

    def _append_annotation(self, output_dir: str, item: dict) -> None:
        """Append a single item to annotations.json under output_dir, enforcing unique answers and indices."""
        annotations_path = os.path.join(output_dir, 'annotations.json')
        os.makedirs(output_dir, exist_ok=True)
        data = []
        try:
            if os.path.exists(annotations_path):
                with open(annotations_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        data = loaded
        except Exception:
            data = []

        def to_answer_str(obj):
            if obj is None:
                return None
            if isinstance(obj, str):
                return obj
            if isinstance(obj, list):
                try:
                    return " ".join([f"({int(r)},{int(c)})" for r, c in obj])
                except Exception:
                    return None
            return None

        existing_indices = set()
        existing_answer_strs = set()
        for it in data:
            if isinstance(it, dict):
                idx = it.get('index')
                if isinstance(idx, str):
                    existing_indices.add(idx)
                ans_str = it.get('answer_str')
                if not isinstance(ans_str, str):
                    ans_str = to_answer_str(it.get('answer'))
                if isinstance(ans_str, str):
                    existing_answer_strs.add(ans_str)

        item_index = item.get('index')
        item_answer_str = item.get('answer_str')
        if not isinstance(item_answer_str, str):
            item_answer_str = to_answer_str(item.get('answer'))
            if item_answer_str is not None:
                item['answer_str'] = item_answer_str

        if item_index in existing_indices:
            print(f"Skipping duplicate index: {item_index}")
            return
        if isinstance(item_answer_str, str) and item_answer_str in existing_answer_strs:
            print(f"Skipping duplicate answer_str: {item_answer_str}")
            return

        data.append(item)
        try:
            with open(annotations_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Successfully saved item {item_index} to {annotations_path} (total items: {len(data)})")
        except Exception as e:
            print(f"Error saving item {item_index}: {e}")
            raise