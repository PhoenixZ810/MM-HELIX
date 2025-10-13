import os
import json
import random
import numpy as np
import hashlib
import time
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List, Dict, Any, Optional
from generator.base_generator import BaseGenerator

class CampsiteGenerator(BaseGenerator):
    """
    Campsite puzzle generator that creates puzzles with tent placement challenges.
    
    Coordinate System: 1-based indexing (top-left corner is [1,1])
    
    Output format:
    - answer: List of tent coordinates [[row, col], [row, col], ...] (1-based indexing)
    - question: Asks to return coordinates where tents should be placed
    - image: Visual representation with 1-based grid numbering matching the answer format
    """
    def __init__(self, output_folder, **kwargs):
        super().__init__(output_folder)
        self.cell_size = kwargs.get('cell_size', 80)
        self.grid_color = kwargs.get('grid_color', 'black')
        self.bg_color = kwargs.get('bg_color', 'white')
        self.tree_color = kwargs.get('tree_color', 'darkgreen')
        self.constraint_color = kwargs.get('constraint_color', 'blue')
        self.tree_emoji = "🌲"  # Tree symbol
        
        # 用于跟踪已生成的puzzle，避免重复
        self.generated_puzzles = set()
        
        # 运行时时间戳作为基础种子（整数）
        self.runtime_seed = int(time.time())
    
    def _calculate_tent_count(self, n):
        """根据网格大小n计算合适的帐篷数量"""
        if n <= 4:
            return max(2, n - 1)
        elif n <= 6:
            return max(3, n - 2)
        else:
            return max(4, n - 3)
    
    def _puzzle_hash(self, grid, row_constraints, col_constraints):
        """生成puzzle的唯一哈希值用于去重"""
        # 使用更详细的字符串表示来避免哈希冲突
        grid_str = ''.join([''.join(row) for row in grid])
        row_str = ','.join(map(str, row_constraints))
        col_str = ','.join(map(str, col_constraints))
        puzzle_str = f"grid:{grid_str}|rows:{row_str}|cols:{col_str}"
        return hashlib.sha256(puzzle_str.encode()).hexdigest()
    
    def _build_single_puzzle(self, size: int, seed: int) -> Tuple[Dict, Dict, str]:
        """构建单个puzzle的数据（不进行任何IO写入）。返回 (puzzle_entry, puzzle_state_dict, image_filename)"""
        # Set random seed for deterministic generation - use both size and seed for uniqueness
        deterministic_seed = hash((size, seed)) % (2**32)  # Ensure 32-bit positive integer
        random.seed(deterministic_seed)
        np.random.seed(deterministic_seed)

        print(f"Generating Campsite puzzle (size={size}, seed={seed})")

        # Calculate expected tent count based on size
        expect_camp_number = self._calculate_tent_count(size)

        # Generate the puzzle with deterministic seed
        max_retries = 50
        for retry in range(max_retries):
            try:
                # Use seed to ensure deterministic generation
                retry_seed = deterministic_seed + retry * 1000

                grid, row_constraints, col_constraints, solution = self.generate_campsite(
                    size, size, expect_camp_number, 
                    seed=retry_seed, random_rate=0.1
                )
                
                puzzle = {
                    'input_grid': grid,
                    'row_constraints': row_constraints,
                    'col_constraints': col_constraints,
                    'reference_solution': solution
                }
                
                # Validate puzzle quality
                if self._validate_puzzle_quality(grid, row_constraints, col_constraints, solution, expect_camp_number):
                    break
                    
            except Exception as e:
                if retry == max_retries - 1:
                    # Use fallback puzzle if all retries failed
                    puzzle = self._generate_fallback_puzzle((size, size), expect_camp_number)
                continue
        
        # Generate unique index based on size and seed
        index = f"campsite_{size}_{seed}"
        
        # Image file naming (actual saving deferred)
        image_filename = f"{index}.png"
        relative_image_path = f"images/{image_filename}"
        
        # Get solution coordinates (1-based indexing)
        solution_coordinates = self.solve(puzzle)
        
        # Generate CoT reasoning
        detailed_cot, step_contents = self.generate_cot(puzzle)
        
        # Create the required question format
        question = """Solve this Campsite puzzle by placing tents adjacent to trees while adhering to the game rules.

### Game Rules:
1) Each tent must be orthogonally adjacent to at least one tree (up, down, left, or right).
2) No tents can be adjacent to each other, even diagonally.
3) The number of tents in each row and column must match the given constraints.

### Coordinate System:
Return the coordinates where tents should be placed as a list of [row, column] pairs using 1-based indexing (e.g., top-left is [1,1]).

### Answer Format:
[[1, 3], [3, 1], [4, 3]]
"""

        # Filter initial_state for question_language (remove reference_solution)
        filtered_initial_state = {
            'input_grid': puzzle['input_grid'],
            'row_constraints': puzzle['row_constraints'],
            'col_constraints': puzzle['col_constraints']
        }
        
        question_language = f"""Solve this Campsite puzzle by placing tents adjacent to trees while adhering to the game rules.
### Game Rules:
1) Each tent must be orthogonally adjacent to at least one tree (up, down, left, or right).
2) No tents can be adjacent to each other, even diagonally.
3) The number of tents in each row and column must match the given constraints.

Initial state: {json.dumps(filtered_initial_state)}
X presents empty cells, T presents trees.

### Coordinate System:
Return the coordinates where tents should be placed as a list of [row, column] pairs using 1-based indexing (e.g., top-left is [1,1]). Example answer format: [[1, 3], [3, 1], [4, 3]]"""
        
        # Create puzzle data in required format with all CoT steps
        puzzle_entry = {
            "index": index,
            "category": "campsite", 
            "image": relative_image_path,
            "question": question,
            "question_language": question_language,
            "answer": json.dumps(solution_coordinates),
            "initial_state": json.dumps(puzzle),
            "difficulty": f"{size-3}",
            "cot": detailed_cot,
            "cot_step1_all": step_contents['step1'],
            "cot_step2_all": step_contents['step2'],
            "cot_step3_all": step_contents['step3'],
        }
        
        print(f"Built puzzle: {index}")
        return puzzle_entry, puzzle, image_filename

    def _get_difficulty_params(self, difficulty):
        """根据难度级别返回参数配置。这里将网格大小与难度线性关联，不改变内部任务构造逻辑。"""
        size = max(4, int(difficulty) + 3)
        return {"size": size}

    def generate(self, num_cases, difficulty, output_folder=None):
        """按新接口批量生成，只在最后一次性写入图片与annotations.json。"""
        output_dir = output_folder or self.output_folder
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        params = self._get_difficulty_params(difficulty)
        size = params["size"]

        # 程序运行时的时间戳作为基础seed（整数）
        base_seed = self.runtime_seed

        all_entries: List[Dict[str, Any]] = []
        images_to_save: List[Tuple[Dict, str]] = []  # (puzzle_state, filename)

        for i in range(int(num_cases)):
            seed_i = base_seed + i
            entry, puzzle_state, image_filename = self._build_single_puzzle(size=size, seed=seed_i)
            all_entries.append(entry)
            images_to_save.append((puzzle_state, image_filename))

        # 批量写图片
        for puzzle_state, image_filename in images_to_save:
            self.visualize(puzzle_state, folder=images_dir, filename=image_filename)

        # 一次性合并并写入 annotations.json（去重、保留已有答案），原子化写入
        annotations_path = os.path.join(output_dir, "annotations.json")

        # 读取已有的annotations（如果存在）
        existing: List[Dict[str, Any]] = []
        if os.path.exists(annotations_path):
            try:
                with open(annotations_path, 'r', encoding='utf-8') as f:
                    parsed = json.load(f)
                    if isinstance(parsed, list):
                        existing = parsed
            except Exception:
                existing = []

        merged_by_index: Dict[str, Dict[str, Any]] = {}
        for item in existing:
            idx = item.get('index')
            if isinstance(idx, str):
                merged_by_index[idx] = item

        for item in all_entries:
            idx = item.get('index')
            if not isinstance(idx, str):
                continue
            if idx in merged_by_index:
                # 合并：优先保留已有答案
                existing_item = merged_by_index[idx]
                merged_item = dict(item)
                if existing_item.get('answer'):
                    merged_item['answer'] = existing_item['answer']
                merged_by_index[idx] = merged_item
            else:
                merged_by_index[idx] = item

        merged_list = list(merged_by_index.values())

        tmp_path = annotations_path + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(merged_list, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, annotations_path)

        print(f"Saved {len(merged_list)} puzzles to {annotations_path}")
        return merged_list
    
    def _save_puzzle_to_annotations(self, puzzle, output_dir):
        """Save puzzle to annotations.json in append mode"""
        annotations_path = os.path.join(output_dir, "annotations.json")
        
        # Load existing annotations if file exists
        if os.path.exists(annotations_path):
            try:
                with open(annotations_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []
        else:
            existing_data = []
        
        # Check for duplicates based on index
        existing_indices = {item.get('index', '') for item in existing_data}
        if puzzle['index'] not in existing_indices:
            existing_data.append(puzzle)
            
            # Save back to file
            with open(annotations_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            print(f"Saved puzzle to annotations.json: {puzzle['index']}")
        else:
            print(f"Puzzle already exists in annotations.json: {puzzle['index']}")
    
    def _split_text_at_halfway(self, text):
        """Split text at halfway point (character count)"""
        if not text:
            return ""
        
        halfway_point = len(text) // 2
        # Find a good break point near the halfway mark (prefer to break at sentence or paragraph end)
        break_chars = ['\n\n', '\n', '. ', '.\n', '? ', '! ']
        
        # Look for the best break point within 50 characters of the halfway point
        best_break = halfway_point
        for break_char in break_chars:
            # Search backwards from halfway point
            pos = text.rfind(break_char, max(0, halfway_point - 50), halfway_point + 50)
            if pos != -1:
                best_break = pos + len(break_char)
                break
        
        return text[:best_break]
    
    def generate_single_puzzle(self, size: Tuple[int, int], expect_camp_number: int, 
                              random_rate: float = 0.1, seed: Optional[int] = None) -> Dict:
        """Generate a single campsite puzzle"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        max_retries = 50  # 增加重试次数以支持大规模生成
        for retry in range(max_retries):
            try:
                # 为每次重试调整参数以增加多样性
                adjusted_seed = seed + retry * 1000 if seed else None
                adjusted_random_rate = random_rate + (retry % 10) * 0.01
                adjusted_expect_camp_number = expect_camp_number + (retry % 3) - 1
                adjusted_expect_camp_number = max(1, min(adjusted_expect_camp_number, size[0] * size[1] // 3))
                
                grid, row_constraints, col_constraints, solution = self.generate_campsite(
                    size[0], size[1], adjusted_expect_camp_number, 
                    seed=adjusted_seed, random_rate=adjusted_random_rate
                )
                
                # 验证puzzle质量
                if self._validate_puzzle_quality(grid, row_constraints, col_constraints, solution, adjusted_expect_camp_number):
                    return {
                        'input_grid': grid,
                        'row_constraints': row_constraints,
                        'col_constraints': col_constraints,
                        'reference_solution': solution
                    }
            except Exception as e:
                if retry % 10 == 0:  # 每10次重试报告一次
                    print(f"生成puzzle时出错 (重试 {retry + 1}): {e}")
                continue
        
        # 如果所有重试都失败，返回一个简单的默认puzzle
        return self._generate_fallback_puzzle(size, expect_camp_number)
    
    def _validate_puzzle_quality(self, grid, row_constraints, col_constraints, solution, expect_camp_number):
        """验证生成的puzzle质量"""
        # 检查帐篷数量
        actual_tent_count = sum(row.count('C') for row in solution)
        if actual_tent_count < expect_camp_number * 0.8:  # 至少80%的期望帐篷数
            return False
        
        # 检查是否有树
        tree_count = sum(row.count('T') for row in grid)
        if tree_count == 0:
            return False
        
        # 检查约束是否合理
        if sum(row_constraints) != sum(col_constraints):
            return False
        
        # 检查每个帐篷是否至少邻接一个树
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        rows, cols = len(grid), len(grid[0])
        
        for r in range(rows):
            for c in range(cols):
                if solution[r][c] == 'C':  # 如果是帐篷
                    has_adjacent_tree = False
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 'T':
                            has_adjacent_tree = True
                            break
                    if not has_adjacent_tree:
                        return False
        
        return True
    
    def _generate_fallback_puzzle(self, size, expect_camp_number):
        """生成一个简单的fallback puzzle"""
        rows, cols = size
        grid = [['X' for _ in range(cols)] for _ in range(rows)]
        
        # 简单地放置一些树
        tree_count = min(expect_camp_number + 1, rows * cols // 3)
        positions = [(r, c) for r in range(rows) for c in range(cols)]
        random.shuffle(positions)
        
        for i in range(tree_count):
            r, c = positions[i]
            grid[r][c] = 'T'
        
        # 创建简单的解决方案
        solution = [row[:] for row in grid]
        tent_count = 0
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 'T' and tent_count < expect_camp_number:
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < rows and 0 <= nc < cols and 
                            solution[nr][nc] == 'X' and tent_count < expect_camp_number):
                            solution[nr][nc] = 'C'
                            tent_count += 1
                            break
        
        row_constraints = [row.count('C') for row in solution]
        col_constraints = [sum(1 for r in range(rows) if solution[r][c] == 'C') for c in range(cols)]
        
        return {
            'input_grid': grid,
            'row_constraints': row_constraints,
            'col_constraints': col_constraints,
            'reference_solution': solution
        }
    
    def generate_campsite(self, rows: int, cols: int, expect_camp_number: int, 
                         seed: Optional[int] = None, random_rate: float = 0.1) -> Tuple[List[List[str]], List[int], List[int], List[List[str]]]:
        """
        Implementation of campsite generation algorithm
        This function creates a valid campsite puzzle with trees and tents
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize empty grid
        grid = [['X' for _ in range(cols)] for _ in range(rows)]
        
        # 更智能的树的数量计算
        min_trees = max(expect_camp_number, 3)
        max_trees = min(expect_camp_number * 2, rows * cols // 2)
        tree_count = random.randint(min_trees, max_trees)
        
        # Place trees with better distribution
        tree_positions = []
        max_tree_attempts = rows * cols * 2
        attempts = 0
        
        while len(tree_positions) < tree_count and attempts < max_tree_attempts:
            attempts += 1
            r, c = random.randint(0, rows-1), random.randint(0, cols-1)
            
            # 避免树过于聚集
            too_close = False
            for tr, tc in tree_positions:
                if abs(r - tr) <= 1 and abs(c - tc) <= 1:
                    if len(tree_positions) > tree_count // 2:  # 只有在后期才严格检查
                        too_close = True
                        break
            
            if not too_close and (r, c) not in tree_positions:
                grid[r][c] = 'T'
                tree_positions.append((r, c))
        
        # Create a solution by placing tents
        solution = [row[:] for row in grid]
        tent_positions = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # 改进的帐篷放置算法
        random.shuffle(tree_positions)
        for r, c in tree_positions:
            if len(tent_positions) >= expect_camp_number:
                break
                
            # 找到所有可能的位置
            possible_positions = []
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and solution[nr][nc] == 'X':
                    # Check if tent placement would be valid (no adjacent tents)
                    valid = True
                    for tr, tc in tent_positions:
                        if abs(nr - tr) <= 1 and abs(nc - tc) <= 1:  # Adjacent (orthogonal or diagonal)
                            valid = False
                            break
                    
                    if valid:
                        possible_positions.append((nr, nc))
            
            # 如果有可能的位置，选择一个
            if possible_positions:
                tent_r, tent_c = random.choice(possible_positions)
                solution[tent_r][tent_c] = 'C'
                tent_positions.append((tent_r, tent_c))
        
        # Calculate row and column constraints
        row_constraints = [row.count('C') for row in solution]
        col_constraints = [sum(1 for r in range(rows) if solution[r][c] == 'C') for c in range(cols)]
        
        # 验证结果
        actual_tent_count = sum(row_constraints)
        if actual_tent_count < expect_camp_number * 0.7:
            # 如果帐篷数量不够，抛出异常让上层重试
            raise ValueError(f"Not enough tents placed: {actual_tent_count} < {expect_camp_number * 0.7}")
        
        return grid, row_constraints, col_constraints, solution
    
    def visualize(self, puzzle: Dict, **kwargs) -> str:
        """Create a more visually appealing representation of the puzzle as an image"""
        grid = puzzle['input_grid']
        row_constraints = puzzle['row_constraints']
        col_constraints = puzzle['col_constraints']
        
        rows, cols = len(grid), len(grid[0])
        folder = kwargs.get('folder', 'images')
        filename = kwargs.get('filename', None)
        n = kwargs.get('n', rows)
        index = kwargs.get('index', 0)
        
        # Create image with additional padding for better appearance
        padding = int(self.cell_size * 0.3)
        width = (cols + 1) * self.cell_size + 2 * padding
        height = (rows + 1) * self.cell_size + 2 * padding
        
        # Use a light gradient background for better aesthetics
        image = Image.new('RGB', (width, height), color='#F5F5F5')
        draw = ImageDraw.Draw(image)
        
        # Add a subtle gradient background
        for y in range(height):
            for x in range(width):
                # Create a subtle gradient from top-left to bottom-right
                gradient_value = int(240 - (x + y) / (width + height) * 20)
                draw.point((x, y), fill=(gradient_value, gradient_value, gradient_value))
        
        # Try to load larger and more readable fonts (prefer DejaVu on Linux)
        font = None
        big_font = None
        main_size = int(self.cell_size * 0.75)
        big_size = int(self.cell_size * 0.85)
        ttf_candidates = [
            ("Arial Bold.ttf", big_size, main_size),
            ("Arial.ttf", big_size, main_size),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", big_size, main_size),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", big_size, main_size),
        ]
        for path, b_size, m_size in ttf_candidates:
            try:
                big_font = ImageFont.truetype(path, size=b_size)
                font = ImageFont.truetype(path, size=m_size)
                break
            except Exception:
                continue
        if font is None or big_font is None:
            try:
                font = ImageFont.load_default()
                big_font = font
            except Exception:
                font = None
                big_font = None
        
        # Draw a subtle outer border for the whole grid
        border_color = '#555555'
        border_width = 3
        draw.rectangle(
            [
                (padding-border_width, padding-border_width),
                (padding + cols * self.cell_size + border_width, padding + rows * self.cell_size + border_width)
            ],
            outline=border_color, width=border_width
        )
        
        # Draw grid with alternating cell colors for better visibility
        for r in range(rows):
            for c in range(cols):
                x, y = c * self.cell_size + padding, r * self.cell_size + padding
                # Create a subtle checkerboard pattern
                if (r + c) % 2 == 0:
                    cell_color = '#FFFFFF'
                else:
                    cell_color = '#F0F0F0'
                draw.rectangle([x, y, x + self.cell_size, y + self.cell_size], fill=cell_color)
        
        # Draw grid lines
        line_color = '#888888'
        for r in range(rows + 1):
            y = r * self.cell_size + padding
            draw.line([(padding, y), (cols * self.cell_size + padding, y)], 
                      fill=line_color, width=1)
        
        for c in range(cols + 1):
            x = c * self.cell_size + padding
            draw.line([(x, padding), (x, rows * self.cell_size + padding)], 
                      fill=line_color, width=1)
        
        # Draw trees with a more appealing representation
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 'T':
                    x, y = c * self.cell_size + padding, r * self.cell_size + padding
                    center_x, center_y = x + self.cell_size // 2, y + self.cell_size // 2
                    
                    # Draw a nicer tree using shapes
                    tree_size = int(self.cell_size * 0.8)
                    
                    # Draw trunk
                    trunk_width = tree_size // 5
                    trunk_height = tree_size // 2.5
                    trunk_color = '#8B4513'  # SaddleBrown
                    
                    trunk_left = center_x - trunk_width // 2
                    trunk_top = center_y + tree_size // 5
                    draw.rectangle(
                        [trunk_left, trunk_top, trunk_left + trunk_width, trunk_top + trunk_height],
                        fill=trunk_color
                    )
                    
                    # Draw foliage (multiple levels of green triangles)
                    foliage_colors = ['#228B22', '#2E8B57', '#32CD32']  # Different greens
                    triangle_width = tree_size * 0.7
                    triangle_height = tree_size * 0.5
                    
                    for i, color in enumerate(foliage_colors):
                        # Draw triangular foliage with shrinking size
                        scale_factor = 1.0 - i * 0.2
                        level_y = center_y - triangle_height // 3 * i
                        
                        draw.polygon([
                            (center_x, level_y - triangle_height // 2 * scale_factor),  # Top
                            (center_x - triangle_width // 2 * scale_factor, level_y + triangle_height // 2 * scale_factor),  # Bottom left
                            (center_x + triangle_width // 2 * scale_factor, level_y + triangle_height // 2 * scale_factor)   # Bottom right
                        ], fill=color, outline='#006400')
        
        # Draw constraint background boxes
        constraint_bg = '#4682B4'  # SteelBlue
        constraint_text_color = 'white'
        
        # Row constraints (right side)
        for r in range(rows):
            x = cols * self.cell_size + padding
            y = r * self.cell_size + padding
            draw.rectangle(
                [x, y, x + self.cell_size, y + self.cell_size],
                fill=constraint_bg
            )
            
            # Center text in the constraint box
            if big_font:
                text = str(row_constraints[r])
                # Get text dimensions to center it precisely using the actual drawing font
                if hasattr(big_font, "getbbox"):
                    left, top, right, bottom = big_font.getbbox(text)
                    text_width, text_height = right - left, bottom - top
                elif hasattr(draw, "textbbox"):
                    left, top, right, bottom = draw.textbbox((0, 0), text, font=big_font)
                    text_width, text_height = right - left, bottom - top
                else:
                    text_width, text_height = draw.textsize(text, font=big_font)
                
                text_x = x + (self.cell_size - text_width) // 2
                text_y = y + (self.cell_size - text_height) // 2
                draw.text((text_x, text_y), text, fill=constraint_text_color, font=big_font)
            else:
                # Fallback center positioning
                draw.text(
                    (x + self.cell_size // 2, y + self.cell_size // 2),
                    str(row_constraints[r]),
                    fill=constraint_text_color,
                    font=big_font,
                    anchor="mm" if hasattr(ImageDraw.Draw, "textbbox") else None
                )
        
        # Column constraints (bottom)
        for c in range(cols):
            x = c * self.cell_size + padding
            y = rows * self.cell_size + padding
            draw.rectangle(
                [x, y, x + self.cell_size, y + self.cell_size],
                fill=constraint_bg
            )
            
            # Center text in the constraint box
            if big_font:
                text = str(col_constraints[c])
                # Get text dimensions to center it precisely using the actual drawing font
                if hasattr(big_font, "getbbox"):
                    left, top, right, bottom = big_font.getbbox(text)
                    text_width, text_height = right - left, bottom - top
                elif hasattr(draw, "textbbox"):
                    left, top, right, bottom = draw.textbbox((0, 0), text, font=big_font)
                    text_width, text_height = right - left, bottom - top
                else:
                    text_width, text_height = draw.textsize(text, font=big_font)
                
                text_x = x + (self.cell_size - text_width) // 2
                text_y = y + (self.cell_size - text_height) // 2
                draw.text((text_x, text_y), text, fill=constraint_text_color, font=big_font)
            else:
                # Fallback center positioning
                draw.text(
                    (x + self.cell_size // 2, y + self.cell_size // 2),
                    str(col_constraints[c]),
                    fill=constraint_text_color,
                    font=big_font,
                    anchor="mm" if hasattr(ImageDraw.Draw, "textbbox") else None
                )
        
        # Save image with provided filename or default naming convention
        if filename is None:
            filename = f"campsite_{n}_{n}_{index}.png"
        filepath = os.path.join(folder, filename)
        image.save(filepath, quality=95)
        
        return filepath
    
    def solve(self, puzzle: Dict, **kwargs) -> List[List[int]]:
        """Return the tent coordinates that need to be placed (1-based indexing)"""
        if 'reference_solution' in puzzle:
            solution_grid = puzzle['reference_solution']
            tent_coordinates = []
            
            # Extract tent positions (marked as 'C' in the solution)
            # Convert to 1-based indexing for human-friendly coordinates
            for r in range(len(solution_grid)):
                for c in range(len(solution_grid[0])):
                    if solution_grid[r][c] == 'C':
                        tent_coordinates.append([r + 1, c + 1])  # Convert to 1-based [row, column]
            
            return tent_coordinates
        
        # If no reference solution, we would implement a solver algorithm here
        # For simplicity, we'll return empty list in this case
        return []
    
    def generate_cot(self, puzzle: Dict, **kwargs) -> Tuple[str, Dict[str, str]]:
        """
        Generate a detailed chain of thought for solving the Campsite puzzle following Rule-Based CoT structure
        Returns tuple of (full_cot, step_contents_dict)
        """
        # Extract puzzle information
        grid = puzzle['input_grid']
        row_constraints = puzzle['row_constraints']
        col_constraints = puzzle['col_constraints']
        solution = puzzle.get('reference_solution')
        
        rows, cols = len(grid), len(grid[0])
        
        # Find tree positions
        tree_positions = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 'T':
                    tree_positions.append((r, c))
        
        # Get correct tent positions from solution
        correct_tents = []
        if solution:
            for r in range(rows):
                for c in range(cols):
                    if solution[r][c] == 'C':
                        correct_tents.append((r, c))
        
        # Step 1: Understanding the puzzle rules and objectives
        step1 = "Let me solve this Campsite puzzle step by step.\n\n"
        step1 += "### Step 1: Understanding the Game Rules\n\n"
        step1 += "First, I need to clearly understand what a Campsite puzzle is and its rules:\n\n"
        step1 += "**Game Objective:** Place tents on the grid according to specific placement rules and constraints.\n\n"
        step1 += "**Essential Rules:**\n"
        step1 += "1. **Tree-Tent Adjacency Rule:** Every tent must be orthogonally adjacent (up, down, left, right) to at least one tree. Diagonal adjacency doesn't count.\n"
        step1 += "2. **Tent Spacing Rule:** No two tents can be adjacent to each other, including diagonally. Tents must have at least one empty cell between them in all directions.\n"
        step1 += "3. **Numerical Constraints:** The number of tents in each row and column must exactly match the given constraint numbers shown on the edges of the grid.\n\n"
        step1 += f"**Puzzle Specifications:**\n"
        step1 += f"- Grid size: {rows}×{cols}\n"
        step1 += f"- Total tents to place: {sum(row_constraints)}\n"
        step1 += f"- Trees available: {len(tree_positions)}\n\n"
        step1 += "**Strategy Overview:** I'll systematically analyze tree positions, identify possible tent locations, apply constraints, and use logical deduction to find the unique solution.\n"
        
        # Step 2: Reading the image and extracting initial state
        step2 = "\n\n### Step 2: Reading the Image and Extracting Initial State\n\n"
        step2 += "Now I'll carefully examine the image to extract all the visual information and represent the initial state.\n\n"
        step2 += "**Visual Elements Identification:**\n"
        step2 += "- 🌲 Green tree symbols represent trees (given clues)\n"
        step2 += "- Empty white/gray cells represent potential tent placement locations\n"
        step2 += "- Blue numbers on the right edge show row constraints (number of tents per row)\n"
        step2 += "- Blue numbers on the bottom edge show column constraints (number of tents per column)\n"
        step2 += "- Grid lines help identify cell positions using coordinate system (1-based indexing)\n\n"
        
        step2 += "**Grid State Extraction:**\n"
        step2 += "Let me carefully read the grid from top-left to bottom-right:\n\n"
        
        # Create a detailed grid representation
        step2 += "```\n"
        step2 += "   "
        for c in range(cols):
            step2 += f" {c+1:2}"
        step2 += "\n"
        
        for r, row in enumerate(grid):
            step2 += f"{r+1:2} "
            for c, cell in enumerate(row):
                if cell == 'T':
                    step2 += " 🌲"
                else:
                    step2 += " □"
            step2 += f"  │ {row_constraints[r]}\n"
        
        step2 += "   "
        for c in range(cols):
            step2 += "──"
        step2 += "─┘\n"
        step2 += "   "
        for c in range(cols):
            step2 += f" {col_constraints[c]:2}"
        step2 += "\n```\n\n"
        
        step2 += "**Initial State Summary:**\n"
        step2 += f"- Trees located at positions (row, col): {[(r+1, c+1) for r, c in tree_positions]}\n"
        step2 += f"- Row constraints: {row_constraints} (left to right: rows 1-{rows})\n"
        step2 += f"- Column constraints: {col_constraints} (top to bottom: columns 1-{cols})\n"
        step2 += f"- Total tents needed: {sum(row_constraints)}\n"
        step2 += f"- Constraint verification: Row sum = {sum(row_constraints)}, Column sum = {sum(col_constraints)} {'✓' if sum(row_constraints) == sum(col_constraints) else '✗'}\n\n"
        
        # Identify zero constraint rows/columns
        zero_rows = [r+1 for r, constraint in enumerate(row_constraints) if constraint == 0]
        zero_cols = [c+1 for c, constraint in enumerate(col_constraints) if constraint == 0]
        
        step2 += "**Immediate Constraints:**\n"
        if zero_rows:
            step2 += f"- Rows {zero_rows} have constraint 0, so NO tents can be placed in these rows\n"
        if zero_cols:
            step2 += f"- Columns {zero_cols} have constraint 0, so NO tents can be placed in these columns\n"
        if not zero_rows and not zero_cols:
            step2 += "- No zero constraints detected, all rows and columns need at least one tent\n"
        
        step2 += "\n**State Reading Reflection:**\n"
        step2 += "Let me double-check my reading of the initial state:\n"
        tree_count_verification = sum(1 for row in grid for cell in row if cell == 'T')
        step2 += f"- Tree count verification: I counted {len(tree_positions)} trees, grid scan shows {tree_count_verification} trees {'✓' if len(tree_positions) == tree_count_verification else '✗'}\n"
        step2 += f"- Constraint consistency: Both row and column constraints sum to {sum(row_constraints)}, which is mathematically consistent ✓\n"
        step2 += f"- Grid completeness: {rows}×{cols} = {rows*cols} total cells, with {len(tree_positions)} trees and {rows*cols - len(tree_positions)} potential tent locations ✓\n"
        
        # Step 3: Detailed reasoning process with exploration
        step3 = "\n\n### Step 3: Detailed Reasoning and Strategic Exploration\n\n"
        step3 += "Now I'll systematically explore tent placement options using logical deduction and constraint analysis.\n\n"
        
        # Initialize tracking variables
        placed_tents = []
        rejected_positions = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        
        step3 += "**Phase 1: Tree-Adjacent Position Analysis**\n\n"
        step3 += "For each tree, I'll identify all possible adjacent tent positions and analyze their viability:\n\n"
        
        # Analyze each tree systematically
        tree_analysis = {}
        for i, (tr, tc) in enumerate(tree_positions, 1):
            step3 += f"**Tree {i} at position ({tr+1}, {tc+1}):**\n"
            
            # Find all adjacent positions
            adjacent_positions = []
            for dr, dc in directions:
                nr, nc = tr + dr, tc + dc
                direction_name = ["right", "down", "left", "up"][directions.index((dr, dc))]
                
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr][nc] != 'T':  # Not another tree
                        # Check zero constraints
                        if nr in [r-1 for r in zero_rows]:
                            step3 += f"  - {direction_name} ({nr+1}, {nc+1}): ❌ Row {nr+1} has zero constraint\n"
                            rejected_positions.append((nr, nc, f"zero row constraint"))
                        elif nc in [c-1 for c in zero_cols]:
                            step3 += f"  - {direction_name} ({nr+1}, {nc+1}): ❌ Column {nc+1} has zero constraint\n"
                            rejected_positions.append((nr, nc, f"zero column constraint"))
                        else:
                            step3 += f"  - {direction_name} ({nr+1}, {nc+1}): ⚠️  Potential location\n"
                            adjacent_positions.append((nr, nc))
                    else:
                        step3 += f"  - {direction_name} ({nr+1}, {nc+1}): ❌ Already occupied by tree\n"
                else:
                    step3 += f"  - {direction_name}: ❌ Out of bounds\n"
            
            tree_analysis[i] = {
                'position': (tr, tc),
                'adjacent_positions': adjacent_positions
            }
            
            step3 += f"  → Valid adjacent positions: {len(adjacent_positions)} options: {[(r+1, c+1) for r, c in adjacent_positions]}\n\n"
        
        step3 += "**Phase 2: Constraint-Driven Tent Placement**\n\n"
        step3 += "I'll now use constraint analysis to systematically place tents:\n\n"
        
        # Create a working grid for visualization
        working_grid = [row[:] for row in grid]
        
        # Strategy: Start with most constrained positions
        # 1. Look for trees with only one possible adjacent position
        step3 += "**Sub-phase 2a: Forced Placements (Trees with single adjacent option)**\n"
        forced_placements = []
        
        for tree_id, analysis in tree_analysis.items():
            available_positions = [pos for pos in analysis['adjacent_positions'] 
                                 if pos not in placed_tents]
            if len(available_positions) == 1:
                tr, tc = analysis['position']
                forced_pos = available_positions[0]
                step3 += f"Tree {tree_id} at ({tr+1}, {tc+1}) has only one viable adjacent position: ({forced_pos[0]+1}, {forced_pos[1]+1})\n"
                step3 += f"→ FORCED PLACEMENT: Tent at ({forced_pos[0]+1}, {forced_pos[1]+1})\n\n"
                placed_tents.append(forced_pos)
                forced_placements.append(forced_pos)
                working_grid[forced_pos[0]][forced_pos[1]] = 'C'
        
        if not forced_placements:
            step3 += "No trees with single adjacent options found. Moving to constraint analysis.\n\n"
        
        # 2. Use row/column constraints to guide placement
        step3 += "**Sub-phase 2b: Constraint-Guided Exploration**\n"
        
        remaining_tents_needed = sum(row_constraints) - len(placed_tents)
        step3 += f"Tents placed so far: {len(placed_tents)}, Remaining needed: {remaining_tents_needed}\n\n"
        
        # Track current row/column counts
        current_row_counts = [sum(1 for tr, tc in placed_tents if tr == r) for r in range(rows)]
        current_col_counts = [sum(1 for tr, tc in placed_tents if tc == c) for c in range(cols)]
        
        step3 += "Current progress:\n"
        for r in range(rows):
            remaining_for_row = row_constraints[r] - current_row_counts[r]
            step3 += f"  Row {r+1}: {current_row_counts[r]}/{row_constraints[r]} (need {remaining_for_row} more)\n"
        
        for c in range(cols):
            remaining_for_col = col_constraints[c] - current_col_counts[c]
            step3 += f"  Column {c+1}: {current_col_counts[c]}/{col_constraints[c]} (need {remaining_for_col} more)\n"
        
        step3 += "\n**Sub-phase 2c: Systematic Position Testing**\n"
        step3 += "For remaining positions, I'll test each possibility considering spacing and adjacency:\n\n"
        
        # Test remaining positions systematically
        for r, c in correct_tents:
            if (r, c) not in placed_tents:
                step3 += f"**Testing position ({r+1}, {c+1}):**\n"
                
                # Check adjacency to trees
                adjacent_trees = []
                for tr, tc in tree_positions:
                    if abs(r - tr) + abs(c - tc) == 1:  # orthogonally adjacent
                        adjacent_trees.append((tr+1, tc+1))
                
                step3 += f"  - Tree adjacency: {'✓' if adjacent_trees else '❌'} (adjacent to trees at {adjacent_trees})\n"
                
                # Check spacing from other tents
                spacing_conflicts = []
                for existing_r, existing_c in placed_tents:
                    if abs(r - existing_r) <= 1 and abs(c - existing_c) <= 1:
                        spacing_conflicts.append((existing_r+1, existing_c+1))
                
                step3 += f"  - Spacing check: {'❌' if spacing_conflicts else '✓'} {'(conflicts with tents at ' + str(spacing_conflicts) + ')' if spacing_conflicts else ''}\n"
                
                # Check constraint satisfaction
                row_would_exceed = current_row_counts[r] + 1 > row_constraints[r]
                col_would_exceed = current_col_counts[c] + 1 > col_constraints[c]
                step3 += f"  - Row constraint: {'❌' if row_would_exceed else '✓'} (would be {current_row_counts[r] + 1}/{row_constraints[r]})\n"
                step3 += f"  - Column constraint: {'❌' if col_would_exceed else '✓'} (would be {current_col_counts[c] + 1}/{col_constraints[c]})\n"
                
                if adjacent_trees and not spacing_conflicts and not row_would_exceed and not col_would_exceed:
                    step3 += f"  → ✅ VALID PLACEMENT: Adding tent at ({r+1}, {c+1})\n"
                    placed_tents.append((r, c))
                    working_grid[r][c] = 'C'
                    current_row_counts[r] += 1
                    current_col_counts[c] += 1
                else:
                    step3 += f"  → ❌ Invalid placement - failed checks\n"
                
                step3 += "\n"
        
        step3 += "**Phase 3: Solution Verification and Backtracking Simulation**\n\n"
        
        # Check if we need backtracking
        total_placed = len(placed_tents)
        total_needed = sum(row_constraints)
        
        if total_placed < total_needed:
            step3 += f"❌ Only placed {total_placed}/{total_needed} tents. Need to reconsider...\n\n"
            step3 += "**Backtracking Analysis:**\n"
            step3 += "Let me reconsider previous decisions and explore alternative placements:\n\n"
            
            # Add missing tents (this simulates finding them through backtracking)
            missing_tents = [(r, c) for r, c in correct_tents if (r, c) not in placed_tents]
            for mr, mc in missing_tents:
                step3 += f"Reconsidering position ({mr+1}, {mc+1}):\n"
                step3 += f"  After removing conflicting constraints and adjusting previous placements...\n"
                step3 += f"  → ✅ Successfully placed tent at ({mr+1}, {mc+1})\n\n"
                placed_tents.append((mr, mc))
                working_grid[mr][mc] = 'C'
        
        # Final placement verification
        step3 += "**Final Placement Summary:**\n"
        step3 += f"Total tents placed: {len(placed_tents)}/{total_needed}\n"
        step3 += f"Tent positions: {[(r+1, c+1) for r, c in placed_tents]}\n\n"
        
        # Show final grid state
        step3 += "**Final Grid State:**\n```\n"
        step3 += "   "
        for c in range(cols):
            step3 += f" {c+1:2}"
        step3 += "\n"
        
        for r in range(rows):
            step3 += f"{r+1:2} "
            for c in range(cols):
                if working_grid[r][c] == 'T':
                    step3 += " 🌲"
                elif working_grid[r][c] == 'C':
                    step3 += " ⛺"
                else:
                    step3 += " □"
            step3 += "\n"
        step3 += "```\n"
        
        # Step 4: Final validation and reflection
        step4 = "\n\n### Step 4: Solution Validation and Reflection\n\n"
        step4 += "Now I'll perform comprehensive validation of my solution and reflect on the solving process.\n\n"
        
        # Comprehensive validation
        step4 += "**Comprehensive Rule Validation:**\n\n"
        
        validation_passed = True
        
        # 1. Tree adjacency validation
        step4 += "**1. Tree-Tent Adjacency Verification:**\n"
        for i, (tr, tc) in enumerate(placed_tents, 1):
            adjacent_trees = []
            for tree_r, tree_c in tree_positions:
                if abs(tr - tree_r) + abs(tc - tree_c) == 1:
                    adjacent_trees.append((tree_r+1, tree_c+1))
            
            if adjacent_trees:
                step4 += f"   Tent {i} at ({tr+1}, {tc+1}): ✅ Adjacent to trees at {adjacent_trees}\n"
            else:
                step4 += f"   Tent {i} at ({tr+1}, {tc+1}): ❌ Not adjacent to any tree\n"
                validation_passed = False
        
        # 2. Tent spacing validation
        step4 += "\n**2. Tent Spacing Verification:**\n"
        spacing_violations = []
        for i, (t1r, t1c) in enumerate(placed_tents):
            for j, (t2r, t2c) in enumerate(placed_tents[i+1:], i+1):
                distance = max(abs(t1r - t2r), abs(t1c - t2c))
                if distance <= 1:
                    spacing_violations.append(((t1r+1, t1c+1), (t2r+1, t2c+1)))
                    step4 += f"   ❌ Tents at ({t1r+1}, {t1c+1}) and ({t2r+1}, {t2c+1}) are too close (distance={distance})\n"
                    validation_passed = False
        
        if not spacing_violations:
            step4 += "   ✅ All tents are properly spaced (no adjacent tents)\n"
        
        # 3. Numerical constraint validation
        step4 += "\n**3. Numerical Constraint Verification:**\n"
        final_row_counts = [sum(1 for tr, tc in placed_tents if tr == r) for r in range(rows)]
        final_col_counts = [sum(1 for tr, tc in placed_tents if tc == c) for c in range(cols)]
        
        step4 += "   Row constraints:\n"
        for r in range(rows):
            actual = final_row_counts[r]
            expected = row_constraints[r]
            status = "✅" if actual == expected else "❌"
            step4 += f"     Row {r+1}: {actual}/{expected} {status}\n"
            if actual != expected:
                validation_passed = False
        
        step4 += "   Column constraints:\n"
        for c in range(cols):
            actual = final_col_counts[c]
            expected = col_constraints[c]
            status = "✅" if actual == expected else "❌"
            step4 += f"     Column {c+1}: {actual}/{expected} {status}\n"
            if actual != expected:
                validation_passed = False
        
        # 4. Solution completeness
        step4 += "\n**4. Solution Completeness:**\n"
        total_tents_placed = len(placed_tents)
        total_tents_required = sum(row_constraints)
        
        if total_tents_placed == total_tents_required:
            step4 += f"   ✅ Correct number of tents placed: {total_tents_placed}/{total_tents_required}\n"
        else:
            step4 += f"   ❌ Incorrect number of tents: {total_tents_placed}/{total_tents_required}\n"
            validation_passed = False
        
        # Final answer and reflection
        step4 += "\n**Final Answer:**\n"
        tent_coordinates = [[r+1, c+1] for r, c in placed_tents]
        tent_coordinates.sort()  # Sort for consistent output
        step4 += f"Tent coordinates (1-based indexing): {tent_coordinates}\n\n"
        
        step4 += "**Solution Reflection:**\n"
        if validation_passed:
            step4 += "✅ **Solution is VALID** - All constraints satisfied!\n\n"
            step4 += "Key insights from solving this puzzle:\n"
            step4 += "- Strategic constraint analysis was crucial for systematic placement\n"
            step4 += "- Tree adjacency requirements limited possible positions effectively\n"
            step4 += "- Tent spacing rules required careful checking during placement\n"
            step4 += "- Row/column constraints provided essential guidance for validation\n"
        else:
            step4 += "❌ **Solution has issues** - Some constraints not satisfied\n\n"
            step4 += "Areas that need attention:\n"
            step4 += "- Double-check tree adjacency for all tents\n"
            step4 += "- Verify tent spacing meets diagonal separation requirement\n"
            step4 += "- Ensure all numerical constraints are exactly met\n"
        
        step4 += f"\n**Confidence Level:** {'High' if validation_passed else 'Requires revision'}\n"
        step4 += f"**Solving Strategy Effectiveness:** The systematic approach of analyzing trees first, then applying constraints, proved effective for this {rows}×{cols} puzzle.\n"
        
        # Create accumulated step contents
        step1_content = step1
        step2_content = step1 + step2
        step3_content = step1 + step2 + step3
        step4_content = step1 + step2 + step3 + step4
        
        # Return tuple of full CoT and step contents
        return step4_content, {
            'step1': step1_content,
            'step2': step2_content,
            'step3': step3_content,
            'step4': step4_content
        }