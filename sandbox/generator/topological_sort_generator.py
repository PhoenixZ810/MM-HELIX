import argparse
import json
import random
import os
import time
from .base_generator import BaseGenerator

try:
    import networkx as nx
    import matplotlib.pyplot as plt
except ImportError:
    print("请安装 'networkx' 和 'matplotlib' 以绘制和保存DAG图像。")
    nx = None
    plt = None




class TopologicalSortGenerator(BaseGenerator):
    """
    拓扑排序生成器类，用于生成、验证和处理有向无环图(DAG)的拓扑排序问题。
    """
    
    @staticmethod
    def verify_topological_order(adj_list, proposed_order):
        """
        验证提议的排序是否是DAG的有效拓扑排序。
        如果有效返回True，否则返回False。
        """
        all_nodes = set(adj_list.keys())
        # 1) 检查是否恰好包含所有节点一次
        if len(proposed_order) != len(all_nodes):
            return False
        if set(proposed_order) != all_nodes:
            return False
        
        # 2) 构建位置映射以比较顺序
        position = {node: idx for idx, node in enumerate(proposed_order)}
        
        # 3) 对于每条边u->v，确保pos[u] < pos[v]
        for u in adj_list:
            for v in adj_list[u]:
                if position[u] > position[v]:
                    return False
        
        return True
    
    @staticmethod
    def generate_single_topological_sort(adj_list):
        """
        生成一个有效的拓扑排序（使用Kahn算法）。
        返回单个拓扑排序列表。
        """
        # 计算入度
        in_degree = {node: 0 for node in adj_list}
        for u in adj_list:
            for v in adj_list[u]:
                in_degree[v] += 1
        
        # Kahn算法
        queue = [node for node in in_degree if in_degree[node] == 0]
        queue.sort()  # 确保结果一致性
        result = []
        
        while queue:
            # 选择入度为0的节点（按数字排序保证一致性）
            current = queue.pop(0)
            result.append(current)
            
            # 更新相邻节点的入度
            for neighbor in adj_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
            
            # 保持队列排序
            queue.sort()
        
        return result
    
    @staticmethod
    def draw_and_save_dag(adj_list, file_path):
        """
        使用networkx绘制DAG并保存为图像，使用网格布局。
        """
        if nx is None or plt is None:
            return
            
        G = nx.DiGraph()
        # 添加边
        for u in adj_list:
            G.add_node(u)
            for v in adj_list[u]:
                G.add_edge(u, v)
        
        num_nodes = len(adj_list)
        
        # 计算网格大小
        import math
        if num_nodes <= 4:
            grid_cols, grid_rows = 2, 2
        elif num_nodes <= 9:
            grid_cols, grid_rows = 3, 3
        elif num_nodes <= 16:
            grid_cols, grid_rows = 4, 4
        elif num_nodes <= 25:
            grid_cols, grid_rows = 5, 5
        else:
            grid_size = math.ceil(math.sqrt(num_nodes))
            grid_cols, grid_rows = grid_size, grid_size
        
        # 创建网格布局位置
        pos = {}
        nodes = sorted(adj_list.keys())
        for i, node in enumerate(nodes):
            row = i // grid_cols
            col = i % grid_cols
            # 使用更大的间距并居中布局
            pos[node] = (col * 2.0, (grid_rows - 1 - row) * 2.0)
        
        # 计算图形大小
        max_x = max(pos.values(), key=lambda p: p[0])[0] if pos else 0
        max_y = max(pos.values(), key=lambda p: p[1])[1] if pos else 0
        fig_width = max(10, max_x + 4)
        fig_height = max(8, max_y + 3)
        
        # 绘制图形
        plt.figure(figsize=(fig_width, fig_height))
        
        # 为有向边使用彩色
        import matplotlib.cm as cm
        import numpy as np
        
        edges = list(G.edges())
        num_edges = len(edges)
        if num_edges > 0:
            # 使用色谱生成不同的颜色并加深
            colors = cm.Set3(np.linspace(0, 1, num_edges))
            edge_colors = []
            for color in colors:
                if len(color) >= 3:
                    # 将颜色加深50%，让颜色更深
                    darker_color = [max(0, c * 0.5) for c in color[:3]] + [1.0]
                    edge_colors.append(darker_color)
                else:
                    edge_colors.append(color)
        else:
            edge_colors = ['darkblue']
        
        # 分别绘制直线边和曲线边
        straight_edges = []
        curved_edges = []
        straight_colors = []
        curved_colors = []
        
        def is_same_row_or_col_non_adjacent(u, v):
            """检查两个节点是否在同行或同列但不相邻"""
            u_row, u_col = u // grid_cols, u % grid_cols
            v_row, v_col = v // grid_cols, v % grid_cols
            
            # 检查是否在同行或同列
            same_row = (u_row == v_row)
            same_col = (u_col == v_col)
            
            if same_row or same_col:
                # 如果在同行或同列，检查是否相邻
                row_diff = abs(u_row - v_row)
                col_diff = abs(u_col - v_col)
                is_adjacent = (row_diff <= 1 and col_diff <= 1 and (row_diff + col_diff == 1))
                return not is_adjacent  # 同行/列但不相邻
            return False
        
        # 分类边
        for i, (u, v) in enumerate(edges):
            if is_same_row_or_col_non_adjacent(u, v):
                curved_edges.append((u, v))
                curved_colors.append(edge_colors[i])
            else:
                straight_edges.append((u, v))
                straight_colors.append(edge_colors[i])
        
        # 绘制直线边
        if straight_edges:
            nx.draw_networkx_edges(G, pos, width=3, alpha=0.9, edge_color=straight_colors,
                                  arrowstyle='->', arrowsize=30, edgelist=straight_edges)
        
        # 绘制曲线边（同行/列且不相邻）
        if curved_edges:
            nx.draw_networkx_edges(G, pos, width=3, alpha=0.9, edge_color=curved_colors,
                                  arrowstyle='->', arrowsize=30, edgelist=curved_edges,
                                  connectionstyle="arc3,rad=0.3")
        
        # 绘制节点（使用醒目的样式）
        nx.draw_networkx_nodes(G, pos, node_size=1200, node_color='lightblue', 
                              edgecolors='darkblue', linewidths=2)
        
        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, font_size=18, font_weight='bold', font_color='black')
        
        # 设置坐标轴范围，确保图形居中
        if pos:
            x_coords = [p[0] for p in pos.values()]
            y_coords = [p[1] for p in pos.values()]
            margin = 1.5
            plt.xlim(min(x_coords) - margin, max(x_coords) + margin)
            plt.ylim(min(y_coords) - margin, max(y_coords) + margin)
        
        # 保持长宽比并去除坐标轴
        plt.gca().set_aspect('equal')
        plt.axis("off")
        plt.tight_layout(pad=0.3)
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def generate_connected_dag(n=4, max_extra_edges=2, seed=None):
        """
        基于网格生成连通的DAG，避免视觉交叉。
        - 在网格中放置节点
        - 优先添加从上到下或从左到右的有向边，自然保证无环
        - 确保连通性和合理的拓扑结构
        """
        if seed is not None:
            random.seed(seed)
        
        import math
        
        # 计算网格大小
        if n <= 4:
            grid_cols, grid_rows = 2, 2
        elif n <= 9:
            grid_cols, grid_rows = 3, 3
        elif n <= 16:
            grid_cols, grid_rows = 4, 4
        elif n <= 25:
            grid_cols, grid_rows = 5, 5
        else:
            grid_size = math.ceil(math.sqrt(n))
            grid_cols, grid_rows = grid_size, grid_size
        
        # 创建节点位置映射
        node_positions = {}
        for i in range(n):
            row = i // grid_cols
            col = i % grid_cols
            node_positions[i] = (row, col)
        
        # 初始化邻接表
        adj_list = {i: [] for i in range(n)}
        
        def add_edge(u, v):
            """添加有向边u->v"""
            if v not in adj_list[u]:
                adj_list[u].append(v)
        
        def is_topologically_valid(u, v):
            """检查边u->v是否在拓扑上有效（从上到下或从左到右）"""
            u_row, u_col = node_positions[u]
            v_row, v_col = node_positions[v]
            
            # 允许的方向：向下、向右、或向右下对角
            return (v_row > u_row) or (v_row == u_row and v_col > u_col) or (v_row > u_row and v_col > u_col)
        
        def creates_cycle_if_add(u, v):
            """检查添加边u->v是否会形成环"""
            visited = set()
            stack = [v]
            while stack:
                curr = stack.pop()
                if curr == u:
                    return True
                if curr not in visited:
                    visited.add(curr)
                    for neighbor in adj_list[curr]:
                        if neighbor not in visited:
                            stack.append(neighbor)
            return False
        
        # 步骤1：基于网格系统化生成基础边结构（包含反向边）
        # 1.1 添加向右的边（同行相邻）+ 反向边（向左）
        for i in range(n):
            row, col = node_positions[i]
            
            # 查找同行右侧相邻节点
            for j in range(n):
                j_row, j_col = node_positions[j]
                if j_row == row and j_col == col + 1:
                    if random.random() < 0.5:  # 50%概率添加向右边
                        add_edge(i, j)
                    elif random.random() < 0.5 and not creates_cycle_if_add(j, i):  # 50%概率添加反向边（向左）
                        add_edge(j, i)
                    break
        
        # 1.2 添加向下的边（同列相邻）+ 反向边（向上）
        for i in range(n):
            row, col = node_positions[i]
            
            # 查找同列下方相邻节点
            for j in range(n):
                j_row, j_col = node_positions[j]
                if j_row == row + 1 and j_col == col:
                    if random.random() < 0.5:  # 50%概率添加向下边
                        add_edge(i, j)
                    elif random.random() < 0.5 and not creates_cycle_if_add(j, i):  # 50%概率添加反向边（向上）
                        add_edge(j, i)
                    break
        
        # 1.3 添加对角边（多方向）
        for i in range(n):
            row, col = node_positions[i]
            
            # 右下对角
            for j in range(n):
                j_row, j_col = node_positions[j]
                if j_row == row + 1 and j_col == col + 1:
                    if random.random() < 0.3:  # 30%概率添加右下对角边
                        add_edge(i, j)
                    elif random.random() < 0.3 and not creates_cycle_if_add(j, i):  # 30%概率添加左上对角边（反向）
                        add_edge(j, i)
                    break
            
            # 左下对角（反向边的一种）
            for j in range(n):
                j_row, j_col = node_positions[j]
                if j_row == row + 1 and j_col == col - 1:
                    if random.random() < 0.3 and not creates_cycle_if_add(i, j):  # 30%概率添加左下对角边
                        add_edge(i, j)
                    break
        
        # 步骤2：添加额外的边来增加复杂性，包括反向边
        extra_edges_added = 0
        attempts = 0
        max_attempts = n * 5
        
        # 2.1 优先添加一些反向边（避免简单升序解决）
        reverse_edges_target = min(max_extra_edges // 2, 3)  # 至少添加一半额外边作为反向边
        reverse_edges_added = 0
        
        while reverse_edges_added < reverse_edges_target and attempts < max_attempts:
            attempts += 1
            
            # 选择反向边：从较大编号指向较小编号
            u, v = random.sample(range(n), 2)
            if u < v:  # 确保是反向边
                u, v = v, u
            
            # 检查边是否已存在
            if v in adj_list[u]:
                continue
            
            # 检查是否会形成环
            if creates_cycle_if_add(u, v):
                continue
            
            # 添加反向边
            add_edge(u, v)
            reverse_edges_added += 1
            extra_edges_added += 1
        
        # 2.2 添加其他类型的额外边
        attempts = 0
        max_attempts = n * 3
        
        while extra_edges_added < max_extra_edges and attempts < max_attempts:
            attempts += 1
            
            # 随机选择两个不同的节点
            u, v = random.sample(range(n), 2)
            
            # 检查边是否已存在
            if v in adj_list[u]:
                continue
            
            # 检查拓扑有效性
            if not is_topologically_valid(u, v):
                continue
            
            # 检查是否会形成环
            if creates_cycle_if_add(u, v):
                continue
            
            # 添加边
            add_edge(u, v)
            extra_edges_added += 1
        
        # 步骤3：确保图是连通的（在无向意义下）
        # 使用DFS检查连通性，如果不连通则添加必要的边
        def is_connected():
            """检查图在无向意义下是否连通"""
            visited = set()
            
            def dfs(node):
                if node in visited:
                    return
                visited.add(node)
                
                # 遍历所有邻居（双向）
                for neighbor in adj_list[node]:
                    dfs(neighbor)
                
                for other_node in adj_list:
                    if node in adj_list[other_node]:
                        dfs(other_node)
            
            dfs(0)
            return len(visited) == n
        
        # 如果不连通，添加必要的连接
        if not is_connected():
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if not is_connected():
                        # 尝试添加一个有效的边
                        if is_topologically_valid(i, j) and not creates_cycle_if_add(i, j):
                            add_edge(i, j)
                        elif is_topologically_valid(j, i) and not creates_cycle_if_add(j, i):
                            add_edge(j, i)
        
        # 排序邻接表以获得一致输出
        for key in adj_list:
            adj_list[key].sort()
        
        return adj_list
    
    @staticmethod
    def generate_solution_steps(adj_list):
        """
        为拓扑排序问题生成解法过程的详细步骤说明。
        
        这个方法使用Kahn算法的变体来生成拓扑排序解法步骤，详细记录了：
        1. 初始化每个节点的入度
        2. 每一步选择入度为0的节点（按数字排序保证结果一致性）
        3. 移除所选节点并更新其相邻节点的入度
        4. 最终输出完整的拓扑排序结果
        
        Args:
            adj_list: 有向无环图的邻接表
            
        Returns:
            一个包含以下键的字典：
            - solution: 生成的拓扑排序结果
            - reasoning_steps: 英文解法步骤（文本形式）
        """
        # 计算每个节点的入度
        in_degree = {node: 0 for node in adj_list}
        for u in adj_list:
            for v in adj_list[u]:
                in_degree[v] += 1
        
        # 生成一个拓扑排序的解法
        solution = []
        in_degree_copy = in_degree.copy()
        remaining_nodes = set(adj_list.keys())
        
        # 使用文本形式记录步骤
        steps = "To find a valid topological ordering, I'll follow these steps:\n\n"
        
        # 添加解释步骤 - 初始化
        steps += "Step 1: Initialize the graph.\n"
        steps += "1.1: Calculate the in-degree of each node:\n"
        for node in sorted(adj_list.keys()):
            steps += f"  - Node {node}: in-degree = {in_degree[node]}\n"
        
        step_count = 2
        while remaining_nodes:
            # 找出所有入度为0的节点
            zero_in_nodes = [node for node in remaining_nodes if in_degree_copy[node] == 0]
            zero_in_nodes.sort()  # 确保结果一致性
            
            if not zero_in_nodes:
                # 如果没有入度为0的节点，但图中还有节点，说明有环
                steps += f"\nStep {step_count}: No valid topological ordering exists.\n"
                steps += f"{step_count}.1: There are no nodes with in-degree 0, but there are still unvisited nodes.\n"
                steps += f"{step_count}.2: This indicates a cycle in the graph.\n"
                steps += f"{step_count}.3: A DAG cannot have cycles, so no valid topological ordering exists.\n"
                break
            
            # 选择第一个入度为0的节点
            next_node = zero_in_nodes[0]
            solution.append(next_node)
            remaining_nodes.remove(next_node)
            
            # 更新相邻节点的入度
            neighbors = adj_list[next_node]
            
            steps += f"\nStep {step_count}: Select a node with in-degree 0.\n"
            steps += f"{step_count}.1: Current nodes with in-degree 0: {zero_in_nodes}\n"
            steps += f"{step_count}.2: Select node {next_node} and add it to the ordering.\n"
            steps += f"{step_count}.3: Current partial ordering: {solution}\n"
            step_count += 1
            
            if neighbors:
                update_info = []
                for neigh in neighbors:
                    if neigh in remaining_nodes:  # 只更新尚未访问的节点
                        in_degree_copy[neigh] -= 1
                        update_info.append(f"Node {neigh}: in-degree reduced from {in_degree_copy[neigh]+1} to {in_degree_copy[neigh]}")
                
                if update_info:
                    steps += f"\nStep {step_count}: Update in-degrees of adjacent nodes.\n"
                    steps += f"{step_count}.1: After removing node {next_node}, update in-degrees:\n"
                    for info in update_info:
                        steps += f"  - {info}\n"
                    step_count += 1
        
        # 添加最终结果
        if solution:
            steps += "\nFinal Result:\n"
            steps += f"1. The complete topological ordering is: {solution}\n"
            steps += "2. This ordering is valid because:\n"
            steps += "   - All nodes are included exactly once\n"
            steps += "   - For every edge u->v, u appears before v in the ordering\n"
        
        return {
            "solution": solution,
            "reasoning_steps": steps
        }
    
    def __init__(self, output_folder="output/topological_sort"):
        """
        初始化拓扑排序生成器。
        
        Args:
            output_folder: 输出文件夹路径
        """
        super().__init__(output_folder)
        
        # 在输出文件夹中创建图片目录
        self.image_dir = os.path.join(output_folder, "images")
        os.makedirs(self.image_dir, exist_ok=True)
    
    def generate_problem_set(self, num_cases=100, n=4, max_extra_edges=2, seed=time.time(), difficulty="hard"):
        """
        生成问题集。对每个DAG，产生两种类型的问题：
          - 带图片的
          - 纯文本的（无图片）
        每个问题包含所有可能的拓扑排序作为参考答案。
        添加了验证逻辑：如果生成的答案验证失败，则重新生成。
        """
        
        problems = []
        max_retries = 10  # 每个问题的最大重试次数
        
        for i in range(num_cases):
            retry_count = 0
            valid_problem = False
            
            while not valid_problem and retry_count < max_retries:
                current_seed = time.time() + retry_count * 1000  # 确保每次重试使用不同的种子
                
                # 生成一个连通DAG
                adj_list = self.generate_connected_dag(n=n, max_extra_edges=max_extra_edges, seed=current_seed)
                
                # 生成一个有效的拓扑排序
                single_answer = self.generate_single_topological_sort(adj_list)
                
                # 验证生成的答案是否正确
                if self.verify_topological_order(adj_list, single_answer):
                    valid_problem = True
                else:
                    retry_count += 1
            
            # 如果超过最大重试次数仍未成功，跳过此问题
            if not valid_problem:
                continue
            
            # 生成解法步骤
            solution_steps = self.generate_solution_steps(adj_list)
            
            # 构建邻接表文本
            adj_str = "\n".join([f"{u}: {adj_list[u]}" for u in sorted(adj_list.keys())])
            
            # (可选) 绘制并保存图像
            image_filename = f"dag_{difficulty}_{i}.png"
            image_path = os.path.join(self.image_dir, image_filename)
            self.draw_and_save_dag(adj_list, image_path)
            
            # 带图片问题
            problem_with_image = {
                "index": f"topological_sort_{difficulty}_{i}",
                "category": "topological_sort",
                "question": (
                    "Given the directed acyclic graph (DAG) shown in the image below, "
                    "please list ONE possible valid topological orders. Answer with a lists of numbers ONLY. For example: [0, 1, 2, 3]."
                ),
                "question_language": (
                    "Given the following adjacency list of a connected DAG, please list ONE possible valid topological orders. "
                    "Answer with a lists of numbers, for example: [0, 1, 2, 3]."
                    f"{adj_str}\n\n",
                ),
                "image": f"images/{image_filename}",  # 使用相对路径 
                "difficulty": difficulty,
                "answer": single_answer,
                "initial_state": adj_list,
                # "cot": solution_steps["reasoning_steps"],
            }
            problems.append(problem_with_image)
            
        return problems
    
    def add_verified_solution(self, problem_list, problem_index, user_solution):
        """
        验证用户解决方案与指定问题索引的邻接表。如果正确，将该解决方案添加到'user_solutions'字段。
        """
        # 定位问题项
        for p in problem_list:
            if p["index"] == problem_index:
                # 验证
                adj_list = p["initial_state"]
                if self.verify_topological_order(adj_list, user_solution):
                    # 如果没有'user_solutions'键，创建它
                    if "user_solutions" not in p:
                        p["user_solutions"] = []
                    p["user_solutions"].append(user_solution)
                    return True
                else:
                    return False
        return False
    
    def _get_difficulty_params(self, difficulty):
        """
        根据难度级别获取相应的参数配置。
        
        Args:
            difficulty: 难度级别（1-5）
            
        Returns:
            dict: 包含难度参数的字典
        """
        params = {}
        
        if difficulty == 5:
            params["n"] = 25
            params["max_extra_edges"] = 2
        elif difficulty == 4:
            params["n"] = 16
            params["max_extra_edges"] = 2
        elif difficulty == 3:
            params["n"] = 12
            params["max_extra_edges"] = 3
        elif difficulty == 2:
            params["n"] = 9
            params["max_extra_edges"] = 3
        else:  # difficulty == 1
            params["n"] = 6
            params["max_extra_edges"] = 3
            
        return params
    
    def generate_problems_by_difficulty(self, num_cases=6, difficulty=1, seed=None):
        """
        根据难度级别生成拓扑排序问题。
        
        Args:
            num_cases: 每个难度级别要生成的问题数量
            difficulty: 难度级别（1-5）
            seed: 随机种子
            
        Returns:
            生成的问题列表
        """
        # 获取难度参数
        params = self._get_difficulty_params(difficulty)
        n = params["n"]
        max_extra_edges = params["max_extra_edges"]
            
        return self.generate_problem_set(num_cases=num_cases, n=n, max_extra_edges=max_extra_edges, 
                                        seed=seed, difficulty=str(difficulty))
    
    def generate(self, num_cases=10, difficulty=1, output_folder=None):
        """
        实现BaseGenerator的generate方法，生成特定难度级别的拓扑排序问题。
        
        Args:
            num_cases: 要生成的问题数量
            difficulty: 难度级别（1-5）
            seed: 随机种子，默认为None
            output_folder: 输出文件夹路径，覆盖构造函数中设置的路径
            
        Returns:
            生成的问题列表
        """
        # 如果提供了新的输出文件夹，更新它
        if output_folder:
            self.output_folder = output_folder
            os.makedirs(output_folder, exist_ok=True)
            self.image_dir = os.path.join(output_folder, "images")
            os.makedirs(self.image_dir, exist_ok=True)

        problems = self.generate_problems_by_difficulty(num_cases=num_cases, difficulty=difficulty)
        self.save_annotations(problems, self.output_folder)
        return problems
        
    def generate_all_problems(self, num_cases=6, seed=None):
        """
        生成所有难度级别的问题集。
        
        Args:
            num_cases: 每个难度级别要生成的问题数量
            seed: 随机种子，默认为None
            
        Returns:
            所有生成的问题列表
        """
        problems = []
        
        for difficulty in [1, 2, 3, 4, 5]:
            problems.extend(self.generate(num_cases=num_cases, difficulty=difficulty))
        
        return problems


def main():
    parser = argparse.ArgumentParser(description="生成拓扑排序问题")
    parser.add_argument("--num_cases", type=int, default=6, help="要生成的每个难度级别的问题数量")
    parser.add_argument("--output_folder", type=str, default="topological_sort_problems", help="输出文件夹")
    parser.add_argument("--difficulty", type=int, default=0, help="问题难度级别（1-5），0表示生成所有难度级别")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    args = parser.parse_args()
    
    # 创建拓扑排序生成器
    generator = TopologicalSortGenerator(output_folder=args.output_folder)
    
    # 生成问题
    if args.difficulty == 0:
        # 生成所有难度级别的问题
        problems = generator.generate_all_problems(num_cases=args.num_cases, seed=args.seed)
    else:
        # 生成特定难度级别的问题
        problems = generator.generate(num_cases=args.num_cases, difficulty=args.difficulty, seed=args.seed)
    
    # 保存问题
    generator.save_problems(problems)


if __name__ == "__main__":
    main()