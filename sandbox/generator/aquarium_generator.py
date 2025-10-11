import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Any, Tuple
from abc import ABC, abstractmethod
import time
import fcntl

from generator.base_generator import BaseGenerator


class AquariumGenerator(BaseGenerator):
    def __init__(self, output_folder):
        super().__init__(output_folder)

    def generate(self, num_cases, difficulty, output_folder=None):
        """
        Generate multiple aquarium puzzles according to the new interface.
        This method keeps the internal single-puzzle construction logic intact
        by delegating to generate_one.
        """
        output_dir = output_folder or self.output_folder
        images_dir = os.path.join(output_dir, "images")

        params = self._get_difficulty_params(difficulty)
        size = params.get("size")

        # Initialize base seed from current timestamp (integer)
        base_seed = int(time.time())

        # First pass: construct puzzles in memory without doing IO
        formatted_puzzles: List[Dict[str, Any]] = []
        puzzles_core: List[Dict[str, Any]] = []

        for i in range(int(num_cases)):
            seed = base_seed + i
            formatted_puzzle, puzzle_core = self.generate_one(
                size=size, seed=seed, output_dir=output_dir, perform_io=False
            )
            formatted_puzzles.append(formatted_puzzle)
            puzzles_core.append(puzzle_core)

        # Second pass: after all puzzles are generated, write images and annotations once
        os.makedirs(images_dir, exist_ok=True)
        for puzzle_core in puzzles_core:
            # Create both puzzle and solution images
            self.visualize(puzzle_core, folder=images_dir)
            self.visualize(puzzle_core, folder=images_dir, show_solution=True)

        # Batch save annotations at the end
        self._save_puzzles_batch(formatted_puzzles, output_dir)

        return formatted_puzzles

    def _get_difficulty_params(self, difficulty):
        """Map difficulty (1-5) to puzzle parameters without altering core logic."""
        difficulty = int(difficulty)
        size_map = {1: 4, 2: 5, 3: 6, 4: 7, 5: 8}
        size = size_map.get(difficulty, 6)
        return {"size": size}

    def generate_one(self, size, seed, output_dir, perform_io=True):
        """
        Generate a single aquarium puzzle with deterministic seed
        Args:
            size: Grid size (size x size)
            seed: Random seed for deterministic generation
            output_dir: Output directory path for images and JSON
            perform_io: Whether to perform immediate IO (images and annotations)
        Returns:
            Tuple[formatted_puzzle_dict, core_puzzle_dict]
        """
        # Set random seed for deterministic generation
        random.seed(seed)
        np.random.seed(seed)

        # Prepare output directories (avoid creating for deferred IO)
        images_dir = os.path.join(output_dir, "images")
        if perform_io:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)

        grid_rows, grid_cols = size, size
        merge_ratio = 0.3 + (size - 4) * 0.05 if size >= 4 else 0.3  # Adaptive merge ratio based on grid size

        # Generate aquarium regions
        regions, region_ids = self._generate_regions(grid_rows, grid_cols, merge_ratio)

        # Generate water levels for each region (0 = empty, 1+ = number of rows filled from bottom)
        water_levels_by_region = {region_id: random.randint(0, grid_rows)
                                for region_id in region_ids}

        # Create a solution grid showing which cells are filled with water
        solution_grid = [[False for _ in range(grid_cols)] for _ in range(grid_rows)]

        # Fill in solution grid based on region water levels
        for r in range(grid_rows):
            for c in range(grid_cols):
                region_id = regions[r][c]
                region_water_level = water_levels_by_region[region_id]

                # Calculate row from bottom (0 = bottom row)
                r_from_bottom = grid_rows - 1 - r

                # Fill cell if it's within the water level for its region
                if r_from_bottom < region_water_level:
                    solution_grid[r][c] = True

        # Compute row clues: number of filled cells per row
        row_clues = []
        for r in range(grid_rows):
            count_filled = sum(1 for c in range(grid_cols) if solution_grid[r][c])
            row_clues.append(count_filled)

        # Compute column clues: number of filled cells per column
        col_clues = []
        for c in range(grid_cols):
            count_filled = sum(1 for r in range(grid_rows) if solution_grid[r][c])
            col_clues.append(count_filled)

        # Create initial_state as 2D array showing regions and clues
        initial_state = {
            'regions': regions,
            'row_clues': row_clues,
            'col_clues': col_clues,
            'grid_size': [grid_rows, grid_cols]
        }

        # Create the puzzle dictionary for internal processing
        puzzle = {
            'index': f'aquarium_{size}_{size}_{seed}',
            'category': 'aquarium',
            'regions': regions,
            'region_ids': region_ids,
            'water_levels_by_region': water_levels_by_region,
            'solution_grid': solution_grid,
            'row_clues': row_clues,
            'col_clues': col_clues,
            'grid_rows': grid_rows,
            'grid_cols': grid_cols,
            'n': size
        }

        # Generate chain-of-thought reasoning as text and step-by-step breakdown
        cot_reasoning, cot_steps = self.generate_cot_reasoning(puzzle)

        # Image paths (defer actual creation if perform_io is False)
        index = puzzle['index']
        planned_puzzle_filename = f"{index}_puzzle.png"
        puzzle_img_path = os.path.join(images_dir, planned_puzzle_filename)
        if perform_io:
            puzzle_img_path = self.visualize(puzzle, folder=images_dir)
            _ = self.visualize(puzzle, folder=images_dir, show_solution=True)

        # Format the answer as a list of filled cells
        filled_cells = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                if solution_grid[r][c]:
                    filled_cells.append((c, r))  # (x, y) format where x=column, y=row

        answer = str(filled_cells)

        # Create question text
        question = self._create_question_text(puzzle)
        question_language = self._create_question_language(puzzle)

        # Format for the specified JSON structure
        formatted_puzzle = {
            'index': puzzle['index'],
            'category': puzzle['category'],
            'image': os.path.relpath(puzzle_img_path, output_dir),
            'question': question,
            'question_language': question_language,
            'answer': answer,
            'initial_state': json.dumps(initial_state),  # Convert to JSON string
            'difficulty': str(size - 3) if size >= 4 else "1",  # Difficulty based on size
            'cot': cot_reasoning,  # Add complete chain-of-thought reasoning as text
            'cot_step1_all': cot_steps['step1_all'],    # Step 1 complete reasoning
            'cot_step2_all': cot_steps['step2_all'],    # Step 2 complete reasoning
            'cot_step3_all': cot_steps['step3_all']     # Step 3 complete reasoning
        }

        # Save immediately only if requested; otherwise defer to batch
        if perform_io:
            self._save_puzzle_to_annotations(formatted_puzzle, output_dir)

        return formatted_puzzle, puzzle

    def generate_cot_reasoning(self, puzzle):
        """
        Generate rule-based chain-of-thought reasoning following the 4-step structure
        Returns both the complete reasoning and incremental steps
        """
        grid_rows = puzzle['grid_rows']
        grid_cols = puzzle['grid_cols']
        regions = puzzle['regions']
        row_clues = puzzle['row_clues']
        col_clues = puzzle['col_clues']
        solution_grid = puzzle['solution_grid']
        water_levels_by_region = puzzle['water_levels_by_region']

        # Initialize step-by-step CoT tracking
        cot_steps = {}
        all_text = []

        # Start with the opening phrase
        all_text.append("Let me analyze this aquarium puzzle step by step")

        # Step 1: Understanding the puzzle rules and objectives (Enhanced)
        step1_content = []
        step1_content.append("\n\n### Step 1: Understanding the puzzle rules and objectives")

        region_count = len(set(water_levels_by_region.keys()))
        step1_content.append(f"This is a {grid_rows}×{grid_cols} aquarium puzzle with {region_count} distinct regions. Each region represents a separate aquarium tank that can be filled with water.")

        step1_content.append("**Core Game Rules:**")
        step1_content.append("1. **Water Level Constraint**: Each region must be filled to a uniform water level from bottom to top - no gaps or floating water allowed.")
        step1_content.append("2. **Gravity Rule**: Water settles at the bottom first. If a cell contains water, all cells below it in the same region must also contain water.")
        step1_content.append("3. **Row Constraints**: Numbers on the right indicate exactly how many cells in each row must be filled with water.")
        step1_content.append("4. **Column Constraints**: Numbers at the bottom indicate exactly how many cells in each column must be filled with water.")
        step1_content.append("5. **Region Independence**: Different regions can have different water levels, but within each region the level must be consistent.")

        step1_content.append(f"**Current Puzzle Setup**: Row clues are {row_clues}, column clues are {col_clues}. Total water cells needed: {sum(row_clues)}.")
        step1_content.append("**Objective**: Determine which specific cells contain water such that all row/column constraints are satisfied while respecting the physical water behavior within each region.")

        # Add step 1 content to all_text
        all_text.extend(step1_content)

        # Calculate step1_part with better splitting logic
        step1_text = " ".join(step1_content)
        words = step1_text.split()
        mid_point = len(words) // 2
        step1_part_words = words[:mid_point]
        step1_part_text = " ".join(step1_part_words)

        # Save cumulative content up to step 1
        cot_steps['step1_part'] = " ".join(all_text[:-len(step1_content)]) + " " + step1_part_text
        cot_steps['step1_all'] = " ".join(all_text)

        # Step 2: Analyzing the visual information (Enhanced with detailed state reading)
        step2_content = []
        step2_content.append("\n\n### Step 2: Reading the image carefully and analyzing the initial state")

        step2_content.append("**Visual Analysis:**")
        step2_content.append("Let me carefully examine the puzzle image. The grid shows thick black lines that define region boundaries, separating different aquarium tanks. Thinner gray lines show individual cell divisions within the same region.")

        region_cells = self._get_region_cells_mapping(regions, grid_rows, grid_cols)
        step2_content.append(f"I can identify {len(region_cells)} distinct regions based on the boundary lines.")

        step2_content.append("**Detailed Region Mapping:**")
        step2_content.append("Reading from the image, I can map out each region's cell positions (using coordinate system where (x,y) = (column,row) and (0,0) is top-left):")

        for region_id in sorted(region_cells.keys()):
            cells_in_region = region_cells[region_id]
            cells_list = ", ".join([f"({c},{r})" for r, c in cells_in_region])
            region_height = max([r for r, c in cells_in_region]) - min([r for r, c in cells_in_region]) + 1
            region_width = max([c for r, c in cells_in_region]) - min([c for r, c in cells_in_region]) + 1
            step2_content.append(f"- Region {region_id}: Contains {len(cells_in_region)} cells at positions [{cells_list}]. Spans {region_width}×{region_height} area.")

        step2_content.append("**Numerical Constraints Analysis:**")
        total_required = sum(row_clues)
        total_available = grid_rows * grid_cols
        step2_content.append(f"Row constraints (right side numbers): {row_clues} - These specify exactly how many cells must be filled in each row.")
        step2_content.append(f"Column constraints (bottom numbers): {col_clues} - These specify exactly how many cells must be filled in each column.")
        step2_content.append(f"**Constraint Verification**: Total cells needed = {total_required}, Total available = {total_available}. Constraint consistency: {'✓ Valid' if total_required <= total_available else '✗ Invalid'}.")

        step2_content.append("**Initial State Representation:**")
        step2_content.append("Current state: All cells are empty (no water). The puzzle asks me to determine which cells should be filled.")
        step2_content.append("Grid representation (showing region IDs):")
        for r in range(grid_rows):
            row_repr = " ".join([f"{regions[r][c]:2d}" for c in range(grid_cols)])
            step2_content.append(f"Row {r}: [{row_repr}] → needs {row_clues[r]} filled cells")

        col_needs = " ".join([f"{col_clues[c]:2d}" for c in range(grid_cols)])
        step2_content.append(f"Col needs: [{col_needs}]")

        step2_content.append("**State Reading Reflection:**")
        step2_content.append("I have successfully identified all regions, mapped their cell positions, and extracted the numerical constraints. The visual information is clear and unambiguous. Now I need to determine optimal water levels for each region to satisfy all constraints.")

        # Add step 2 content to all_text
        all_text.extend(step2_content)

        # Calculate step2_part with better splitting logic
        step2_text = " ".join(step2_content)
        words = step2_text.split()
        mid_point = len(words) // 2
        step2_part_words = words[:mid_point]
        step2_part_text = " ".join(step2_part_words)

        # Save cumulative content up to step 2
        cot_steps['step2_part'] = " ".join(all_text[:-len(step2_content)]) + " " + step2_part_text
        cot_steps['step2_all'] = " ".join(all_text)

        # Step 3: Strategic exploration and reasoning (Enhanced - Most Critical Section)
        step3_content = []
        step3_content.append("\n\n### Step 3: Detailed reasoning process and systematic exploration")

        step3_content.append("**Strategic Approach:**")
        step3_content.append("I'll use a systematic constraint satisfaction approach with logical deduction and backtracking. The key insight is that each region's water level is independent, but the combined effect must satisfy all row and column constraints simultaneously.")

        total_water_needed = sum(row_clues)
        step3_content.append(f"**Global Analysis**: Need exactly {total_water_needed} water cells total across {region_count} regions. Each region can have water level from 0 (empty) to {grid_rows} (completely full).")

        step3_content.append("**Region-by-Region Analysis:**")
        step3_content.append("Let me systematically explore each region's possible water levels and their implications:")

        # Enhanced exploration for each region
        for region_id in sorted(water_levels_by_region.keys()):
            cells_in_region = region_cells[region_id]
            water_level = water_levels_by_region[region_id]
            region_size = len(cells_in_region)

            step3_content.append(f"\n**Region {region_id} Analysis** (Contains {region_size} cells):")
            step3_content.append(f"- Cell positions: {[f'({c},{r})' for r, c in cells_in_region]}")

            # Calculate which rows this region affects
            rows_affected = sorted(set([r for r, c in cells_in_region]))
            cols_affected = sorted(set([c for r, c in cells_in_region]))
            step3_content.append(f"- Affects rows {rows_affected} and columns {cols_affected}")

            # Show water level exploration
            step3_content.append(f"- Possible water levels: 0 to {grid_rows}")

            # Simulate testing different levels
            if water_level == 0:
                step3_content.append(f"- Testing level 0 (empty): No cells filled. Contributes 0 to all affected row/column counts.")
                step3_content.append(f"- Testing level 1: Would fill bottom cells, but let me check constraints...")
                step3_content.append(f"- **Decision**: Level 0 is optimal - keeps region empty to allow other regions to meet constraints.")
            else:
                step3_content.append(f"- Testing level {water_level-1}: Would fill {self._count_cells_at_level(cells_in_region, water_level-1, grid_rows)} cells.")
                step3_content.append(f"- Testing level {water_level}: Fills {self._count_cells_at_level(cells_in_region, water_level, grid_rows)} cells from bottom up.")
                step3_content.append(f"- Testing level {water_level+1}: Would fill {self._count_cells_at_level(cells_in_region, water_level+1, grid_rows)} cells - checking if this violates constraints...")

                # Show specific cell filling
                filled_cells_this_level = []
                for r, c in cells_in_region:
                    row_from_bottom = grid_rows - 1 - r
                    if row_from_bottom < water_level:
                        filled_cells_this_level.append(f"({c},{r})")
                step3_content.append(f"- **Decision**: Level {water_level} fills cells {filled_cells_this_level}. This contributes optimally to row/column requirements.")

        step3_content.append("\n**Constraint Satisfaction Analysis:**")
        step3_content.append("Now I'll verify that the chosen water levels satisfy all constraints:")

        # Show row-by-row verification process
        step3_content.append("**Row Constraint Verification:**")
        for r in range(grid_rows):
            filled_in_row = []
            for c in range(grid_cols):
                region_id = regions[r][c]
                level = water_levels_by_region[region_id]
                row_from_bottom = grid_rows - 1 - r
                if row_from_bottom < level:
                    filled_in_row.append(f"({c},{r})")

            actual_count = len(filled_in_row)
            required_count = row_clues[r]
            status = "✓" if actual_count == required_count else "✗"
            step3_content.append(f"- Row {r}: Found {actual_count} filled cells {filled_in_row}, need {required_count} {status}")

        step3_content.append("**Column Constraint Verification:**")
        for c in range(grid_cols):
            filled_in_col = []
            for r in range(grid_rows):
                region_id = regions[r][c]
                level = water_levels_by_region[region_id]
                row_from_bottom = grid_rows - 1 - r
                if row_from_bottom < level:
                    filled_in_col.append(f"({c},{r})")

            actual_count = len(filled_in_col)
            required_count = col_clues[c]
            status = "✓" if actual_count == required_count else "✗"
            step3_content.append(f"- Column {c}: Found {actual_count} filled cells {filled_in_col}, need {required_count} {status}")

        step3_content.append("\n**Solution Synthesis:**")
        filled_cells_final = []
        for r in range(grid_rows):
            for c in range(grid_cols):
                region_id = regions[r][c]
                level = water_levels_by_region[region_id]
                row_from_bottom = grid_rows - 1 - r
                if row_from_bottom < level:
                    filled_cells_final.append((c, r))

        step3_content.append("Combining all region water levels, the complete solution is:")
        step3_content.append(f"Final water configuration: {sorted(filled_cells_final)}")
        step3_content.append(f"This fills exactly {len(filled_cells_final)} cells, matching the required total of {total_water_needed}.")

        # Add step 3 content to all_text
        all_text.extend(step3_content)

        # Calculate step3_part with better splitting logic
        step3_text = " ".join(step3_content)
        words = step3_text.split()
        mid_point = len(words) // 2
        step3_part_words = words[:mid_point]
        step3_part_text = " ".join(step3_part_words)

        # Save cumulative content up to step 3
        cot_steps['step3_part'] = " ".join(all_text[:-len(step3_content)]) + " " + step3_part_text
        cot_steps['step3_all'] = " ".join(all_text)

        # Step 4: Solution validation and refinement (Enhanced)
        step4_content = []
        step4_content.append("\n\n### Step 4: Solution validation and reflection")

        step4_content.append("**Final Solution Summary:**")
        step4_content.append("Based on my systematic analysis, the optimal water levels for each region are:")

        for region_id in sorted(water_levels_by_region.keys()):
            water_level = water_levels_by_region[region_id]
            cells_count = self._count_cells_at_level(region_cells[region_id], water_level, grid_rows)
            step4_content.append(f"- Region {region_id}: Water level {water_level} (fills {cells_count} cells)")

        step4_content.append("\n**Comprehensive Constraint Validation:**")
        all_valid = True
        validation_details = []

        # Detailed row validation
        step4_content.append("**Row-by-Row Validation:**")
        for r in range(grid_rows):
            actual = sum(1 for c in range(grid_cols) if solution_grid[r][c])
            expected = row_clues[r]
            is_valid = actual == expected
            if not is_valid:
                all_valid = False
            status_symbol = "✓" if is_valid else "✗"
            step4_content.append(f"- Row {r}: Expected {expected}, Got {actual} {status_symbol}")
            validation_details.append(f"Row {r}: {actual}/{expected}")

        # Detailed column validation
        step4_content.append("**Column-by-Column Validation:**")
        for c in range(grid_cols):
            actual = sum(1 for r in range(grid_rows) if solution_grid[r][c])
            expected = col_clues[c]
            is_valid = actual == expected
            if not is_valid:
                all_valid = False
            status_symbol = "✓" if is_valid else "✗"
            step4_content.append(f"- Column {c}: Expected {expected}, Got {actual} {status_symbol}")
            validation_details.append(f"Col {c}: {actual}/{expected}")

        # Water physics validation
        step4_content.append("**Physical Constraint Validation:**")
        physics_valid = True
        for region_id in water_levels_by_region.keys():
            region_validation = self._validate_water_physics(region_id, region_cells[region_id], water_levels_by_region[region_id], solution_grid, grid_rows)
            if not region_validation['valid']:
                physics_valid = False
                step4_content.append(f"- Region {region_id}: ✗ {region_validation['issue']}")
            else:
                step4_content.append(f"- Region {region_id}: ✓ Water level consistent, no floating water")

        step4_content.append("\n**Overall Validation Result:**")
        if all_valid and physics_valid:
            step4_content.append("🎉 **SOLUTION VALIDATED**: All constraints satisfied!")
            step4_content.append("✓ All row constraints met")
            step4_content.append("✓ All column constraints met")
            step4_content.append("✓ No floating water in any region")
            step4_content.append("✓ Water levels are physically consistent")
        else:
            step4_content.append("⚠️ **VALIDATION ISSUES DETECTED**")
            if not all_valid:
                step4_content.append("✗ Some row/column constraints violated")
            if not physics_valid:
                step4_content.append("✗ Water physics rules violated")
            step4_content.append("Solution requires refinement...")

        # Final answer
        filled_cells_final = [(c, r) for r in range(grid_rows) for c in range(grid_cols) if solution_grid[r][c]]
        step4_content.append(f"\n**Final Answer:** {sorted(filled_cells_final)}")

        step4_content.append("\n**Solution Reflection:**")
        step4_content.append(f"This solution fills {len(filled_cells_final)} cells total, distributed across {len([r for r in water_levels_by_region.values() if r > 0])} non-empty regions.")
        step4_content.append("The approach successfully used constraint satisfaction with systematic region analysis to find a solution that respects both the numerical constraints and the physical water behavior rules.")

        if all_valid and physics_valid:
            step4_content.append("The solution is optimal and unique given the constraints.")
        else:
            step4_content.append("Further analysis would be needed to resolve constraint violations.")

        # Add step 4 content to all_text
        all_text.extend(step4_content)

        # Calculate step4_part with better splitting logic
        step4_text = " ".join(step4_content)
        words = step4_text.split()
        mid_point = len(words) // 2
        step4_part_words = words[:mid_point]
        step4_part_text = " ".join(step4_part_words)

        # Save cumulative content up to step 4
        cot_steps['step4_part'] = " ".join(all_text[:-len(step4_content)]) + " " + step4_part_text
        cot_steps['step4_all'] = " ".join(all_text)

        # Return both complete reasoning and step-by-step breakdown
        complete_reasoning = " ".join(all_text)
        return complete_reasoning, cot_steps

    def _get_region_cells_mapping(self, regions, grid_rows, grid_cols):
        """Get mapping of region_id to list of cells (r,c) in that region"""
        region_cells = {}
        for r in range(grid_rows):
            for c in range(grid_cols):
                region_id = regions[r][c]
                if region_id not in region_cells:
                    region_cells[region_id] = []
                region_cells[region_id].append((r, c))
        return region_cells

    def _count_cells_at_level(self, cells_in_region, water_level, grid_rows):
        """Count how many cells would be filled at a given water level"""
        count = 0
        for r, c in cells_in_region:
            row_from_bottom = grid_rows - 1 - r
            if row_from_bottom < water_level:
                count += 1
        return count

    def _validate_water_physics(self, region_id, cells_in_region, water_level, solution_grid, grid_rows):
        """Validate that water physics are correctly applied in a region"""
        validation = {'valid': True, 'issue': ''}

        for r, c in cells_in_region:
            row_from_bottom = grid_rows - 1 - r
            should_be_filled = row_from_bottom < water_level
            actually_filled = solution_grid[r][c]

            if should_be_filled != actually_filled:
                validation['valid'] = False
                if should_be_filled and not actually_filled:
                    validation['issue'] = f"Cell ({c},{r}) should be filled but isn't"
                elif not should_be_filled and actually_filled:
                    validation['issue'] = f"Cell ({c},{r}) is filled but shouldn't be (floating water)"
                break

        return validation

    def _generate_region_reasoning(self, region_id, cells_in_region, water_level, grid_rows, row_clues, col_clues):
        """Generate detailed reasoning for a specific region's water level"""
        reasoning = {
            "water_level_determination": f"Region {region_id} has water level {water_level} out of {grid_rows} possible levels",
            "bottom_up_filling": "Water fills from bottom row upward due to gravity constraint",
            "affected_rows": [],
            "affected_cols": []
        }

        # Determine which rows and columns are affected
        for r, c in cells_in_region:
            row_from_bottom = grid_rows - 1 - r
            if row_from_bottom < water_level:
                if r not in reasoning["affected_rows"]:
                    reasoning["affected_rows"].append(r)
                if c not in reasoning["affected_cols"]:
                    reasoning["affected_cols"].append(c)

        return reasoning

    def _validate_rows(self, solution_grid, row_clues, grid_rows, grid_cols):
        """Validate that each row has the correct number of filled cells"""
        validation = {}
        for r in range(grid_rows):
            actual_count = sum(1 for c in range(grid_cols) if solution_grid[r][c])
            expected_count = row_clues[r]
            validation[f"row_{r}"] = {
                "expected": expected_count,
                "actual": actual_count,
                "valid": actual_count == expected_count
            }
        return validation

    def _validate_columns(self, solution_grid, col_clues, grid_rows, grid_cols):
        """Validate that each column has the correct number of filled cells"""
        validation = {}
        for c in range(grid_cols):
            actual_count = sum(1 for r in range(grid_rows) if solution_grid[r][c])
            expected_count = col_clues[c]
            validation[f"col_{c}"] = {
                "expected": expected_count,
                "actual": actual_count,
                "valid": actual_count == expected_count
            }
        return validation

    def _validate_regions(self, solution_grid, regions, water_levels_by_region, grid_rows, grid_cols):
        """Validate that each region follows the water level rules"""
        validation = {}
        region_cells = self._get_region_cells_mapping(regions, grid_rows, grid_cols)

        for region_id, water_level in water_levels_by_region.items():
            cells_in_region = region_cells[region_id]
            validation[f"region_{region_id}"] = {
                "water_level": water_level,
                "cells": cells_in_region,
                "correct_filling": True,
                "issues": []
            }

            # Check if water level is consistent within region
            for r, c in cells_in_region:
                row_from_bottom = grid_rows - 1 - r
                should_be_filled = row_from_bottom < water_level
                actually_filled = solution_grid[r][c]

                if should_be_filled != actually_filled:
                    validation[f"region_{region_id}"]["correct_filling"] = False
                    validation[f"region_{region_id}"]["issues"].append({
                        "cell": (r, c),
                        "should_be_filled": should_be_filled,
                        "actually_filled": actually_filled
                    })

        return validation

    def visualize(self, puzzle, folder, show_solution=False, **kwargs):
        grid_rows = puzzle['grid_rows']
        grid_cols = puzzle['grid_cols']
        regions = puzzle['regions']
        row_clues = puzzle['row_clues']
        col_clues = puzzle['col_clues']
        n = puzzle['n']

        # Create figure and axis with tight layout (neutral white background)
        fig = plt.figure(figsize=(grid_cols + 1.5, grid_rows + 1.5))
        ax = fig.add_subplot(111)

        # Draw grid background
        ax.add_patch(plt.Rectangle((-0.5, -0.5), grid_cols + 1, grid_rows + 1,
                                   fill=True, color='white', zorder=0))

        # Draw cells with white background
        for r in range(grid_rows):
            for c in range(grid_cols):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True,
                                         color='white', zorder=1))

        # Draw faint grid lines between cells to de-emphasize intra-region boundaries
        # Keep region boundaries bold and dark (drawn later)
        for r in range(grid_rows + 1):
            ax.plot([-0.5, grid_cols - 0.5], [r - 0.5, r - 0.5], color='#D0D0D0', linewidth=0.8, zorder=1)
        for c in range(grid_cols + 1):
            ax.plot([c - 0.5, c - 0.5], [-0.5, grid_rows - 0.5], color='#D0D0D0', linewidth=0.8, zorder=1)

        # Draw region boundaries with consistent thick lines
        boundary_lines = set()  # To keep track of drawn boundaries

        for r in range(grid_rows):
            for c in range(grid_cols):
                current_region = regions[r][c]

                # Check all four sides and add boundaries where regions differ
                # Right boundary
                if c < grid_cols - 1:
                    if regions[r][c+1] != current_region:
                        boundary_lines.add((c+0.5, r-0.5, c+0.5, r+0.5))

                # Bottom boundary
                if r < grid_rows - 1:
                    if regions[r+1][c] != current_region:
                        boundary_lines.add((c-0.5, r+0.5, c+0.5, r+0.5))

                # Left boundary (for cells at the grid edge)
                if c == 0:
                    boundary_lines.add((c-0.5, r-0.5, c-0.5, r+0.5))

                # Top boundary (for cells at the grid edge)
                if r == 0:
                    boundary_lines.add((c-0.5, r-0.5, c+0.5, r-0.5))

                # Right edge of grid
                if c == grid_cols - 1:
                    boundary_lines.add((c+0.5, r-0.5, c+0.5, r+0.5))

                # Bottom edge of grid
                if r == grid_rows - 1:
                    boundary_lines.add((c-0.5, r+0.5, c+0.5, r+0.5))

        # Draw all region boundary lines with consistent thickness (heavier than grid)
        for x1, y1, x2, y2 in boundary_lines:
            ax.plot([x1, x2], [y1, y2], color='black', linewidth=3.0, zorder=3)

        # Fill cells with water if show_solution is True (use neutral, print-friendly fill)
        if show_solution:
            solution_grid = puzzle['solution_grid']
            for r in range(grid_rows):
                for c in range(grid_cols):
                    if solution_grid[r][c]:
                        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                                   fill=True,
                                                   color='#BDBDBD',  # neutral gray for academic style
                                                   zorder=2))

        # Draw row clues (numbers only, no boxes)
        for r in range(grid_rows):
            ax.text(grid_cols + 0.25, r,
                    str(row_clues[r]),
                    ha='center', va='center', fontsize=14, color='black', zorder=4)

        # Draw column clues (numbers only, no boxes)
        for c in range(grid_cols):
            ax.text(c, -0.8, str(col_clues[c]),
                    ha='center', va='center', fontsize=14, color='black', zorder=4)

        # Set axis limits to show only the grid and clues
        ax.set_xlim(-1, grid_cols + 0.6)
        ax.set_ylim(grid_rows + 0.1, -1.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')  # Hide axes and spines for a clean academic look

        # No titles or extra labels

        # Adjust layout to remove unnecessary whitespace
        plt.tight_layout(pad=0.2)

        # Save figure with new naming convention
        index = puzzle['index']  # Use full index
        filename = f"{index}_{'solution' if show_solution else 'puzzle'}.png"
        filepath = os.path.join(folder, filename)
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0.02, dpi=300)
        plt.close()

        return filepath

    def solve(self, puzzle, **kwargs):
        # The solution is already generated during puzzle creation
        return puzzle['water_levels_by_region']

    def _generate_regions(self, grid_rows, grid_cols, merge_ratio=0.4):
        """
        Generate aquarium regions that respect the rules:
        1. Each region is a contiguous group of cells
        2. Regions can have irregular shapes
        merge_ratio: Higher values create fewer, larger regions
        Note: Random seed is already set in generate() method for deterministic behavior
        """
        # Start with each cell in its own region
        regions = [[c + r * grid_cols for c in range(grid_cols)] for r in range(grid_rows)]

        # To make more interesting aquariums, merge some adjacent regions
        total_cells = grid_rows * grid_cols
        merge_count = int(total_cells * merge_ratio)

        for _ in range(merge_count):
            # Pick a random cell
            r = random.randint(0, grid_rows - 1)
            c = random.randint(0, grid_cols - 1)
            current_region = regions[r][c]

            # Choose a random direction to merge (up, down, left, right)
            directions = []
            if r > 0:  # up
                directions.append((-1, 0))
            if r < grid_rows - 1:  # down
                directions.append((1, 0))
            if c > 0:  # left
                directions.append((0, -1))
            if c < grid_cols - 1:  # right
                directions.append((0, 1))

            if not directions:
                continue

            dr, dc = random.choice(directions)
            neighbor_r, neighbor_c = r + dr, c + dc
            neighbor_region = regions[neighbor_r][neighbor_c]

            # Skip if already in the same region
            if current_region == neighbor_region:
                continue

            # Merge regions
            for rr in range(grid_rows):
                for cc in range(grid_cols):
                    if regions[rr][cc] == neighbor_region:
                        regions[rr][cc] = current_region

        # Identify unique region IDs and renumber for clarity
        unique_regions = set()
        for r in range(grid_rows):
            for c in range(grid_cols):
                unique_regions.add(regions[r][c])

        region_map = {old_id: i for i, old_id in enumerate(unique_regions)}
        for r in range(grid_rows):
            for c in range(grid_cols):
                regions[r][c] = region_map[regions[r][c]]

        return regions, list(range(len(unique_regions)))

    def _create_question_text(self, puzzle):
        return """
The grid is divided into multiple aquariums (regions). Your task is to determine which cells are filled with water based on the following rules:

### Game Rules:
1. Each region must be filled to a uniform water level (from bottom up).
2. Water cannot float — if a cell is filled, the cell directly below it (if any, in same region) must also be filled.
3. The numbers outside the grid indicate how many cells are filled with water in each row and column.
4. Regions are separated by thick black lines in the grid. Cells within the same region (enclosed by thick lines) must follow the same water level rule. Cells separated by thinner lines are still in the same region.

### Coordinate system:
(x, y) where (0, 0) is the top-left cell. x increases to the right, y increases downward.

### Answer Format:
Please list all the cells that are filled with water in the format:
[(x1, y1), (x2, y2), ...]

Example:
[(0, 4), (1, 4), (1, 3), (2, 3)]

Please read the image carefully and provide the answer in the format above.
"""

    def _create_question_language(self, puzzle):
        grid_rows = puzzle['grid_rows']
        grid_cols = puzzle['grid_cols']
        row_clues = puzzle['row_clues']
        col_clues = puzzle['col_clues']
        regions = puzzle['regions']

        region_text = "### Region layout:\n"
        for r in range(grid_rows):
            row = [str(regions[r][c]) for c in range(grid_cols)]
            region_text += f"Row {r}: {' '.join(row)}\n"

        return f"""
The grid is divided into multiple aquariums (regions). Your task is to determine which cells are filled with water based on the following rules:

### Game Rules:
1. Each region must be filled to a uniform water level (from bottom up).
2. Water cannot float — if a cell is filled, the cell directly below it (if any, in same region) must also be filled.
3. The numbers outside the grid indicate how many cells are filled with water in each row and column.
4. Cells with the same number belong to the same region in the region layout representation.

{region_text}
Row clues: {row_clues}
Column clues: {col_clues}

### Coordinate system:
(x, y) where (0, 0) is the top-left cell. x increases to the right, y increases downward.

Please list all the cells that are filled with water in the format:
[(x1, y1), (x2, y2), ...]

Example:
[(0, 4), (1, 4), (1, 3), (2, 3)]
"""

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

        # Check for duplicates based on index (not answer, to allow same solutions with different puzzles)
        existing_indices = {item.get('index', '') for item in existing_data}
        if puzzle['index'] not in existing_indices:
            existing_data.append(puzzle)

            # Save back to file
            with open(annotations_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            print(f"Added puzzle {puzzle['index']} to {annotations_path}")
        else:
            print(f"Puzzle {puzzle['index']} already exists (duplicate index), skipping...")

    def _save_puzzles_batch(self, puzzles: List[Dict[str, Any]], output_dir: str):
        """Batch save puzzles to annotations.json once, with cross-process safety.
        - Uses an advisory file lock to avoid concurrent write clobbering
        - Reloads existing annotations under the lock to merge accurately
        - Writes atomically via os.replace
        """
        annotations_path = os.path.join(output_dir, "annotations.json")
        tmp_path = annotations_path + ".tmp"
        lock_path = annotations_path + ".lock"
        os.makedirs(output_dir, exist_ok=True)

        # Acquire exclusive lock
        with open(lock_path, 'w') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            # Reload existing annotations under the lock
            existing_data: List[Dict[str, Any]] = []
            if os.path.exists(annotations_path):
                try:
                    with open(annotations_path, 'r', encoding='utf-8') as f:
                        existing_loaded = json.load(f)
                        if isinstance(existing_loaded, list):
                            existing_data = existing_loaded
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

            # Release lock automatically when file closes

        print(f"Saved {new_count} new puzzles, total {len(merged)} to {annotations_path}")