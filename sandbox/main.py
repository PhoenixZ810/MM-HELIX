import argparse
import os
import multiprocessing
import time
from generator_type import generator_type_map

def generate_problems_for_difficulty(task_info):
    """处理单个生成器的单一难度级别"""
    generator_name, generator_class, output_base, rewrite, num_cases, difficulty = task_info
    
    if output_base is None:
        output_folder = f"output/{generator_name}"
    else:
        output_folder = f"{output_base}/{generator_name}"
    
    # 创建生成器实例
    generator = generator_class(output_folder=output_folder)
    
    # 处理指定难度
    start_time = time.time()
    print(f" {generator_name} - Difficulty {difficulty}: generating {num_cases} cases...")
    
    generator.generate(
        num_cases=num_cases,
        difficulty=difficulty,
        output_folder=output_folder
    )
    
    end_time = time.time()
    print(f" {generator_name} - Difficulty {difficulty}: completed in {end_time - start_time:.2f} seconds")
    
    return f"Completed {generator_name} - Difficulty {difficulty}"

def generate_problems_for_type(generator_info):
    """处理单个生成器类型的所有难度"""
    generator_name, generator_class, output_base, rewrite, num_cases, difficulty_levels, use_multiprocessing = generator_info
    
    print(f"\n=== Generating {generator_name} ===")
    
    if output_base is None:
        output_folder = f"output/{generator_name}"
    else:
        output_folder = f"{output_base}/{generator_name}"
    
    # 清理输出目录（如果需要）
    if rewrite:
        import shutil
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
    
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 准备难度级别
    if difficulty_levels is None:
        difficulty_levels = [1, 2, 3, 4, 5]
    else:
        difficulty_levels = [difficulty_levels]
    
    # 如果启用多进程处理单个生成器的多个难度
    if use_multiprocessing and len(difficulty_levels) > 1:
        difficulty_tasks = []
        for difficulty in difficulty_levels:
            difficulty_tasks.append((
                generator_name, 
                generator_class, 
                output_base, 
                False,  # 不要在每个子进程中重写目录
                num_cases, 
                difficulty
            ))
        
        # 使用进程池处理多个难度
        with multiprocessing.Pool(processes=min(len(difficulty_levels), multiprocessing.cpu_count())) as pool:
            results = pool.map(generate_problems_for_difficulty, difficulty_tasks)
        
        for result in results:
            print(result)
    else:
        # 顺序处理难度级别
        for difficulty in difficulty_levels:
            task = (generator_name, generator_class, output_base, False, num_cases, difficulty)
            result = generate_problems_for_difficulty(task)
            print(result)
    
    return f"Completed all difficulties for {generator_name}"

parser = argparse.ArgumentParser(description="Generate problems")
parser.add_argument("--generator", type=str, default=None, help="Generator to use (if not specified, generate all with all difficulties)")
parser.add_argument("--num_cases", type=int, default=10, help="Number of problems to generate per difficulty level")
parser.add_argument("--difficulty", type=int, default=None, help="Difficulty level (only used when generator is specified)")
parser.add_argument("--output_folder", type=str, default="output", help="Output folder")
parser.add_argument("--rewrite", action="store_true", help="Rewrite the output folder")
parser.add_argument("--processes", type=int, default=multiprocessing.cpu_count(), help="Number of processes to use for parallel generation")
parser.add_argument("--no-multiprocessing", action="store_true", help="Disable multiprocessing (use single process)")
parser.add_argument("--parallel-difficulties", action="store_true", help="Process different difficulty levels in parallel for each generator")
args = parser.parse_args()
print(args)

if __name__ == '__main__':
    # 使用 if __name__ == '__main__' 确保多进程在 Windows 上也能正常工作
    start_time_total = time.time()
    
    if args.generator is None:
        # 模式1：生成所有类型，所有难度
        print(f"Mode 1: Generating all types with all difficulty levels...")
        print(f"Will generate {len(generator_type_map)} types × 5 difficulties × {args.num_cases} cases = {len(generator_type_map) * 5 * args.num_cases} total problems")
        print(f"Using {args.processes} processes\n")
        
        # 准备所有生成器信息
        generator_tasks = []
        for generator_name, generator_class in generator_type_map.items():
            generator_tasks.append((
                generator_name, 
                generator_class, 
                args.output_folder, 
                args.rewrite, 
                args.num_cases, 
                None,  # 所有难度
                args.parallel_difficulties and not args.no_multiprocessing  # 是否对每个生成器的难度使用多进程
            ))
        
        # 使用进程池并行处理
        with multiprocessing.Pool(processes=args.processes) as pool:
            results = pool.map(generate_problems_for_type, generator_tasks)
        
        for result in results:
            print(result)
            
    else:
        # 模式2：生成特定类型和难度
        if args.generator not in generator_type_map:
            print(f"Error: Unknown generator '{args.generator}'")
            print(f"Available generators: {', '.join(generator_type_map.keys())}")
            exit(1)
        if args.difficulty is None:
            print(f"Error: When specifying a generator, you must also specify difficulty (1-5)")
            exit(1)
        if args.difficulty not in [1, 2, 3, 4, 5]:
            print(f"Error: Difficulty must be 1, 2, 3, 4, or 5 (got {args.difficulty})")
            exit(1)
            
        print(f"Mode 2: Generating specific type with specific difficulty")
        print(f"Generator: {args.generator}, Difficulty: {args.difficulty}, Cases: {args.num_cases}\n")
        
        # 直接调用处理函数处理单个生成器
        generator_info = (
            args.generator,
            generator_type_map[args.generator],
            args.output_folder,
            args.rewrite,
            args.num_cases,
            args.difficulty,
            False  # 不使用多进程，因为只处理单个难度
        )
        result = generate_problems_for_type(generator_info)
        print(result)
    
    end_time_total = time.time()
    print(f"\nTotal execution time: {end_time_total - start_time_total:.2f} seconds")