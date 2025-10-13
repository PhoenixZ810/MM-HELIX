# MM-HELIX SandBox

## Usage

### Basic Usage

To generate all types of problems with all difficulty levels:

```bash
python main.py
```

By default, this will generate 10 cases for each problem type and difficulty level.

### Generate Specific Problem Type

To generate a specific problem type with a specific difficulty:

```bash
python main.py --generator wordladder --difficulty 3
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--generator` | Specific generator to use (e.g., `wordladder`, `twenty_four_points`) |
| `--num_cases` | Number of problems to generate per difficulty level (default: 10) |
| `--difficulty` | Difficulty level from 1-5 (required when using `--generator`) |
| `--output_folder` | Output folder for generated problems (default: "output") |
| `--rewrite` | If set, will overwrite existing output folder |
| `--processes` | Number of processes to use for parallel generation (default: CPU count) |
| `--no-multiprocessing` | Disable multiprocessing (use single process) |
<!-- | `--parallel-difficulties` | Process different difficulty levels in parallel for each generator | -->

### Examples

Generate 20 word ladder problems with difficulty level 4:
```bash
python main.py --generator wordladder --difficulty 4 --num_cases 20
```

Generate all problem types with all difficulties, using 8 processes:
```bash
python main.py --processes 8
```

Generate all problem types but disable multiprocessing:
```bash
python main.py --no-multiprocessing
```

## Output Format

Generated problems are saved in the specified output folder (default: "output") with the following structure:
```
output/
  ├── wordladder/
  │    ├── annotations.json
  │    └── images/
  │         ├── wordladder_1_1.png
  │         ├── wordladder_1_1_solution.png
  │         └── ...
  ├── twenty_four_points/
  │    └── ...
  └── ...
```

Each problem type has its own folder containing:
- `annotations.json`: Contains all the problem data, including questions, answers, and metadata
- `images/`: Contains the rendered problem visualizations and solutions

## Extending

To add a new problem generator, create a new class inheriting from `BaseGenerator` in the `generator` folder, and add it to `generator_type_map` in `generator_type.py`.

## TODO

- Seamless integration with vlmevalkit by adding TSV generation scripts
  - Convert generated annotations to TSV format compatible with vlmevalkit

- Combined with SERG pipeline to generate high-quality, reflective reasoning instruction-tuning samples

