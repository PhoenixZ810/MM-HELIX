import numpy as np
import matplotlib
matplotlib.use('Agg')  # 必须在pyplot导入之前设置
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import json
import os
import random
import time
from collections import deque
import uuid
import fcntl
import multiprocessing as mp

try:
    from tqdm import tqdm
except Exception:
    # Fallback no-op progress bar
    class _DummyTqdm:
        def __init__(self, total=None, desc=None, leave=True):
            self.total = total
        def update(self, n=1):
            pass
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()
    def tqdm(*args, **kwargs):
        return _DummyTqdm(total=kwargs.get('total'), desc=kwargs.get('desc'), leave=kwargs.get('leave', True))


# Cache serif font properties globally to avoid repeated and expensive font discovery
_GLOBAL_FONT_PROPS = None
_GLOBAL_FONT_FAMILY = None

def get_cached_serif_font_properties():
    global _GLOBAL_FONT_PROPS, _GLOBAL_FONT_FAMILY
    if _GLOBAL_FONT_PROPS is not None:
        return _GLOBAL_FONT_PROPS, _GLOBAL_FONT_FAMILY
    candidates = [
        'Times New Roman',
        'Times',
        'Nimbus Roman',
        'TimesNewRoman',
        'Liberation Serif',
        'DejaVu Serif',
        'Serif'
    ]
    for family in candidates:
        try:
            prop = fm.FontProperties(family=family)
            path = fm.findfont(prop, fallback_to_default=False)
            if path and os.path.exists(path):
                _GLOBAL_FONT_PROPS, _GLOBAL_FONT_FAMILY = prop, family
                return _GLOBAL_FONT_PROPS, _GLOBAL_FONT_FAMILY
        except Exception:
            continue
    # Final fallback
    _GLOBAL_FONT_PROPS, _GLOBAL_FONT_FAMILY = fm.FontProperties(family='DejaVu Serif'), 'DejaVu Serif'
    return _GLOBAL_FONT_PROPS, _GLOBAL_FONT_FAMILY


from generator.base_generator import BaseGenerator


def _save_puzzles_batch(puzzles, output_dir):
    """Batch save puzzles to annotations.json once, with cross-process safety.
    Mirrors the approach used in other generators (advisory lock + atomic write).
    """
    annotations_path = os.path.join(output_dir, "annotations.json")
    tmp_path = annotations_path + ".tmp"
    lock_path = annotations_path + ".lock"
    os.makedirs(output_dir, exist_ok=True)

    # Acquire exclusive lock
    with open(lock_path, 'w') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        # Reload existing annotations under the lock
        existing_data = []
        if os.path.exists(annotations_path):
            try:
                with open(annotations_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        existing_data = loaded
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []

        existing_indices = {item.get('index', '') for item in existing_data}
        merged = list(existing_data)
        new_count = 0
        for p in puzzles:
            if p.get('index') not in existing_indices:
                merged.append(p)
                existing_indices.add(p.get('index'))
                new_count += 1

        # Atomic write
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, annotations_path)

    print(f"Saved {new_count} new puzzles, total {len(merged)} to {annotations_path}")


class SudokuGenerator(BaseGenerator):
    def __init__(self, output_folder, training_set_path=None, **kwargs):
        super().__init__(output_folder)

    def _get_difficulty_params(self, difficulty):
        difficulty = int(difficulty)
        # Fixed 9x9 Sudoku; fewer clues for higher difficulty
        config = {
            1: {"size": 9, "target_clues": 40},
            2: {"size": 9, "target_clues": 36},
            3: {"size": 9, "target_clues": 32},
            4: {"size": 9, "target_clues": 28},
            5: {"size": 9, "target_clues": 24},
        }
        return config.get(difficulty, {"size": 9, "target_clues": 32})

    def generate(self, num_cases, difficulty, output_folder=None):
        output_dir = output_folder or self.output_folder
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        params = self._get_difficulty_params(difficulty)
        n = int(params.get("size", 9))
        target_clues = int(params.get("target_clues", 32))

        # Load existing annotations to avoid duplicates across runs
        existing_indices = set()
        existing_initial_states = set()
        annotations_path = os.path.join(output_dir, 'annotations.json')
        if os.path.exists(annotations_path):
            try:
                with open(annotations_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                if isinstance(existing, list):
                    for item in existing:
                        if item.get('category') == 'sudoku':
                            idx = item.get('index')
                            if idx:
                                existing_indices.add(idx)
                            init = item.get('initial_state')
                            if isinstance(init, str):
                                existing_initial_states.add(init)
            except Exception:
                pass

        # Base seed for batch; offset by difficulty to avoid cross-difficulty collisions
        base_seed = int(time.time()) + int(difficulty) * 10**6

        seen_indices = set()
        seen_initial_states = set()
        all_entries = []
        attempts = 0
        target = int(num_cases)
        max_attempts = max(1000, target * 200)

        while len(all_entries) < target and attempts < max_attempts:
            case_seed = base_seed + attempts
            attempts += 1
            entry = self._generate_one(n, case_seed, output_dir, images_dir, target_clues, difficulty)
            if entry is None:
                continue
            idx = entry.get('index')
            init = entry.get('initial_state')
            if idx in existing_indices or idx in seen_indices:
                continue
            if isinstance(init, str) and (init in existing_initial_states or init in seen_initial_states):
                continue
            seen_indices.add(idx)
            if isinstance(init, str):
                seen_initial_states.add(init)
            all_entries.append(entry)

        if len(all_entries) < target:
            print(f"Warning: Requested {target} puzzles but only generated {len(all_entries)} unique after {attempts} attempts.")

        if all_entries:
            _save_puzzles_batch(all_entries, output_dir)

        return all_entries

    def _generate_one(self, n, seed, output_dir, images_dir, target_clues, difficulty):
        try:
            random.seed(seed)
            np.random.seed(seed)
            index = f"sudoku_{n}_d{int(difficulty)}_{seed}"
            puzzle_img_path = os.path.join(images_dir, f"{index}_puzzle.png")
            solution_img_path = os.path.join(images_dir, f"{index}_solution.png")

            game = Sudoku(n=n, seed=seed)
            full = game.generate_complete()
            puzzle, solution = game.make_puzzle(full_grid=full, target_clues=target_clues)
            initial_state = game.to_text_representation(puzzle)

            # Render images
            game.visualize_board(puzzle, filename=puzzle_img_path)
            game.visualize_board(solution, filename=solution_img_path)

            # Build answer text row-major
            answer_text = game.grid_to_string(solution)
            givens = int(np.sum(np.array(puzzle) > 0))
            step_count = int(n * n - np.sum(np.array(puzzle) > 0))

            # Question text (image-based)
            sudoku_prompt = f"""Your task is to solve the {n}x{n} Sudoku puzzle shown in the image.\n\n""" \
                           "- Output the fully solved grid as {n*n} integers in row-major order, separated by single spaces\n" \
                           "- Use only digits 1–9 (no 0, '.', commas, or any other characters)\n" \
                           f"- Do not include any explanation, steps, or extra text\n"

            # Question language with explicit grid
            block_note = "Each 3x3 subgrid must contain 1–9 exactly once" if n == 9 else ""
            sudoku_prompt_text = f"""Solve the {n}x{n} Sudoku. Fill rows and columns with digits 1–{n} exactly once. {block_note}\n\n""" \
                                    "0 represents empty cells in the grid below:\n\n" + initial_state

            entry = {
                'index': index,
                'category': 'sudoku',
                'image': os.path.join('images', f'{index}_puzzle.png'),
                'solution_image': os.path.join('images', f'{index}_solution.png'),
                'question': sudoku_prompt,
                'question_language': sudoku_prompt_text,
                'answer': answer_text,
                'initial_state': initial_state,
                'difficulty': str(int(difficulty)),
                'size': n,
                'givens': givens,
                'step_count': step_count,
            }
            print(f"Generated sudoku: size={n}, seed={seed}, clues={givens}, difficulty={entry['difficulty']}")
            return entry
        except Exception as _:
            return None


class Sudoku:
    def __init__(self, n, seed=None):
        self.n = int(n)
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # Only 9x9 uses sub-grid regions; others are Latin squares (no regions)
        self.enable_blocks = (self.n == 9)
        # Determine sub-grid shape (rows x cols)
        self.block_rows, self.block_cols = self._compute_block_shape(self.n)
        # Track whether font fallback notice was printed
        self._font_notice_printed = False

    def _compute_block_shape(self, n):
        # Use 3x3 blocks only for standard 9x9 Sudoku; otherwise no regions
        if n == 9:
            return 3, 3
        # No regions: choose a degenerate block to avoid bold lines; logic ignores blocks when disabled
        return 1, n

    # ---------------- Grid utilities ----------------
    def generate_complete(self):
        """Generate a complete valid grid. For n==9 use Sudoku pattern; else Latin square."""
        n = self.n
        br, bc = self.block_rows, self.block_cols

        def pattern(r, c):
            if self.enable_blocks:
                # Standard Sudoku pattern respects 3x3 regions
                return (r * bc + r // br + c) % n
            # Latin square pattern without regions
            return (r + c) % n

        base = np.array([[pattern(r, c) for c in range(n)] for r in range(n)], dtype=int)

        # Shuffle rows and columns within bands/stacks and shuffle bands/stacks themselves
        def shuffled_groups(group_size):
            groups = list(range(group_size))
            random.shuffle(groups)
            return groups

        rows = []
        for rg in shuffled_groups(self.n // br):
            band = [rg * br + i for i in range(br)]
            random.shuffle(band)
            rows.extend(band)

        cols = []
        for cg in shuffled_groups(self.n // bc):
            stack = [cg * bc + i for i in range(bc)]
            random.shuffle(stack)
            cols.extend(stack)

        grid = base[rows][:, cols]

        # Random digit permutation 0..n-1 mapped to 1..n
        digits = list(range(n))
        random.shuffle(digits)
        mapping = {d: i + 1 for i, d in enumerate(digits)}
        grid = np.vectorize(mapping.get)(grid).astype(int)
        return grid.tolist()

    def make_puzzle(self, full_grid, target_clues):
        """Carve clues from a full grid ensuring uniqueness with a backtracking solver."""
        n = self.n
        puzzle = np.array(full_grid, dtype=int)

        # Positions list
        cells = [(r, c) for r in range(n) for c in range(n)]
        random.shuffle(cells)

        # Remove while still unique and over target
        for r, c in cells:
            if int(np.count_nonzero(puzzle)) <= target_clues:
                break
            backup = puzzle[r, c]
            puzzle[r, c] = 0
            # Check uniqueness
            count = self._count_solutions(puzzle.tolist(), limit=2)
            if count != 1:
                puzzle[r, c] = backup

        return puzzle.tolist(), full_grid

    # ---------------- Solver ----------------
    def _count_solutions(self, grid, limit=2):
        """Count number of solutions up to limit."""
        n = self.n
        br, bc = self.block_rows, self.block_cols

        rows = [set() for _ in range(n)]
        cols = [set() for _ in range(n)]
        if self.enable_blocks:
            blocks = [[set() for _ in range(n // bc)] for _ in range(n // br)]
        else:
            blocks = None

        empties = []
        for r in range(n):
            for c in range(n):
                v = grid[r][c]
                if v:
                    rows[r].add(v)
                    cols[c].add(v)
                    if self.enable_blocks:
                        blocks[r // br][c // bc].add(v)
                else:
                    empties.append((r, c))

        solution_count = 0

        def backtrack(idx):
            nonlocal solution_count
            if solution_count >= limit:
                return
            if idx == len(empties):
                solution_count += 1
                return
            r, c = empties[idx]
            for v in range(1, n + 1):
                block_ok = True
                if self.enable_blocks:
                    block_ok = v not in blocks[r // br][c // bc]
                if v not in rows[r] and v not in cols[c] and block_ok:
                    rows[r].add(v)
                    cols[c].add(v)
                    if self.enable_blocks:
                        blocks[r // br][c // bc].add(v)
                    grid[r][c] = v
                    backtrack(idx + 1)
                    grid[r][c] = 0
                    rows[r].discard(v)
                    cols[c].discard(v)
                    if self.enable_blocks:
                        blocks[r // br][c // bc].discard(v)

        # Heuristic: sort empties by least candidates
        def candidates_count(cell):
            r, c = cell
            used = rows[r] | cols[c]
            if self.enable_blocks:
                used = used | blocks[r // br][c // bc]
            return len(used)

        empties.sort(key=candidates_count, reverse=True)
        backtrack(0)
        return solution_count

    # ---------------- Rendering ----------------
    def visualize_board(self, grid, filename=None):
        n = self.n
        br, bc = self.block_rows, self.block_cols
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='#f9f9f9')
        ax.set_facecolor('#ffffff')

        # Resolve serif font using global cache
        font_props, chosen_family = get_cached_serif_font_properties()
        if chosen_family != 'Times New Roman' and not self._font_notice_printed:
            self._font_notice_printed = True

        # Draw grid lines
        for i in range(n + 1):
            if self.enable_blocks and (i % br == 0):
                lw = 2.5
            elif (not self.enable_blocks) and (i == 0 or i == n):
                lw = 2.5
            else:
                lw = 0.8
            ax.plot([0, n], [i, i], color='black', linewidth=lw, alpha=0.8)
        for j in range(n + 1):
            if self.enable_blocks and (j % bc == 0):
                lw = 2.5
            elif (not self.enable_blocks) and (j == 0 or j == n):
                lw = 2.5
            else:
                lw = 0.8
            ax.plot([j, j], [0, n], color='black', linewidth=lw, alpha=0.8)

        # Place numbers
        for r in range(n):
            for c in range(n):
                v = grid[r][c]
                if v:
                    fs = 20 if self.enable_blocks else 28
                    if self.enable_blocks:
                        ax.text(c + 0.5, r + 0.5, str(v), va='center', ha='center', fontsize=fs, fontproperties=font_props)
                    else:
                        ax.text(c + 0.5, r + 0.5, str(v), va='center', ha='center', fontsize=fs, fontweight='bold', fontproperties=font_props)

        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return filename
        else:
            plt.show()
            plt.close(fig)

    def to_text_representation(self, grid):
        n = self.n
        lines = []
        for r in range(n):
            row = []
            for c in range(n):
                v = grid[r][c]
                row.append('.' if v == 0 else str(v))
            lines.append(' '.join(row))
        return '\n'.join(lines)

    def grid_to_string(self, grid):
        """Serialize grid row-major as space-separated values."""
        n = self.n
        vals = []
        for r in range(n):
            for c in range(n):
                vals.append(str(grid[r][c]))
        return ' '.join(vals)

        
def _worker_generate_sudoku(args):
    # Not used in the new unified API; kept for potential future needs.
    try:
        n, attempt_seed, image_dir, task_dir, target_clues = args
        gen = SudokuGenerator(output_folder=task_dir)
        images_dir = os.path.join(task_dir, 'images')
        return gen._generate_one(n, attempt_seed, task_dir, images_dir, target_clues, difficulty=3)
    except Exception:
        return None


if __name__ == '__main__':
    # Simple manual test: generate a few puzzles in the project's output structure
    out = os.path.join('output', 'sudoku')
    os.makedirs(out, exist_ok=True)
    SudokuGenerator(output_folder=out).generate(num_cases=1, difficulty=3, output_folder=out)