import argparse
import os
import json
import random
import time
from .base_generator import BaseGenerator

try:
    import networkx as nx
    import matplotlib.pyplot as plt
except ImportError:
    print("Please install 'networkx' and 'matplotlib' if you want to draw and save images.")
    nx = None
    plt = None


class HamiltonianPathGenerator(BaseGenerator):
    """
    哈密顿路径问题生成器，生成图和相关的哈密顿路径问题
    """
    
    def __init__(self, output_folder="output/hamiltonian_path"):
        """
        初始化生成器
        
        Args:
            output_folder: 输出文件夹路径
        """
        super().__init__(output_folder)
    
    def generate(self, num_cases=10, difficulty=1, seed=None, output_folder=None, max_degree=None, use_random_start=True):
        """
        生成哈密顿路径问题
        
        Args:
            num_cases: 每个难度级别生成的问题数量
            difficulty: 难度级别 (1-5)
            seed: 随机种子
            output_folder: 输出文件夹路径
            max_degree: 图中节点的最大度数
            use_random_start: 是否使用随机起点
            
        Returns:
            生成的问题列表
        """
        if output_folder is None:
            output_folder = self.output_folder
            
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取难度参数
        params = self._get_difficulty_params(difficulty)
        
        # 调用生成哈密顿路径问题的方法
        problems = HamiltonianPathGenerator.generate_hamiltonian_path_problems(
            num_cases=num_cases, 
            difficulty=difficulty,
            max_degree=max_degree if max_degree is not None else params["max_degree"],
            use_random_start=use_random_start,
            output_folder=output_folder
        )
        
        self.save_annotations(problems, output_folder)
            
        # 保存评分方法
        metrics_file = os.path.join(output_folder, "metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump({"score_function": "hamiltonian_path_evaluator"}, f, ensure_ascii=False, indent=4)
            
        return problems
    
    def _get_difficulty_params(self, difficulty):
        """
        根据难度级别返回合适的参数
        
        Args:
            difficulty: 难度级别 (1-5)
            
        Returns:
            dict: 包含图生成参数的字典
        """
        params = {
            "edge_prob": 0.2,  # 随机边的概率
            "max_degree": 3    # 节点的最大度数
        }
        
        # 根据难度调整节点数量
        if difficulty == 1:
            params["num_nodes"] = 4
        elif difficulty == 2:
            params["num_nodes"] = 6
        elif difficulty == 3:
            params["num_nodes"] = 8
        elif difficulty == 4:
            params["num_nodes"] = 12
        elif difficulty == 5:
            params["num_nodes"] = 16
        else:
            # 默认难度
            params["num_nodes"] = 4
            
        return params
    
    @staticmethod
    def generate_random_connected_undirected_graph(num_nodes, edge_prob=0.2, max_degree=None, seed=None):
        """
        基于网格结构生成连通的无向图，完全避免边的视觉交叉，并支持度数限制。
        策略：只允许网格中真正相邻的节点连接，确保所有边都不穿过其他节点。
        :param num_nodes: 节点数量
        :param edge_prob: 在相邻节点间添加边的概率
        :param max_degree: 节点的最大度数限制，None表示无限制
        :param seed: 随机种子（可选）
        :return: adjacency_list (dict), 例如 {0: [1,2], 1:[0], 2:[0], ...}
        """
        if seed is not None:
            random.seed(seed)
        
        import math
        
        adjacency_list = {i: [] for i in range(num_nodes)}
        
        # 计算合适的网格大小
        if num_nodes <= 4:
            grid_cols, grid_rows = 2, 2
        elif num_nodes <= 9:
            grid_cols, grid_rows = 3, 3
        elif num_nodes <= 16:
            grid_cols, grid_rows = 4, 4
        else:
            grid_size = math.ceil(math.sqrt(num_nodes))
            grid_cols, grid_rows = grid_size, grid_size
        
        # 将节点按顺序排列到网格位置（不随机打乱，保持一致性）
        node_positions = {}
        nodes = list(range(num_nodes))
        
        for i, node in enumerate(nodes):
            row = i // grid_cols
            col = i % grid_cols
            node_positions[node] = (row, col)
        
        # 创建位置到节点的反向映射
        pos_to_node = {pos: node for node, pos in node_positions.items()}
        
        def add_edge(u, v):
            """添加无向边，支持度数限制"""
            if v not in adjacency_list[u] and u != v:
                # 检查度数限制
                if max_degree is not None:
                    if len(adjacency_list[u]) >= max_degree or len(adjacency_list[v]) >= max_degree:
                        return False
                
                adjacency_list[u].append(v)
                adjacency_list[v].append(u)
                return True
            return False
        
        def get_adjacent_neighbors(row, col):
            """获取网格中某位置的直接相邻位置（只包括上下左右和对角线）"""
            neighbors = []
            # 8个方向：上下左右 + 4个对角线
            directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < grid_rows and 0 <= new_col < grid_cols and 
                    (new_row, new_col) in pos_to_node):
                    neighbors.append((new_row, new_col))
            return neighbors
        
        # 为每个节点收集所有可能的相邻连接
        possible_edges = []
        for node in nodes:
            row, col = node_positions[node]
            neighbors = get_adjacent_neighbors(row, col)
            
            for neighbor_pos in neighbors:
                neighbor_node = pos_to_node[neighbor_pos]
                if node < neighbor_node:  # 避免重复边
                    # 计算连接类型和权重
                    dr = abs(neighbor_pos[0] - row)
                    dc = abs(neighbor_pos[1] - col)
                    
                    if dr <= 1 and dc <= 1:  # 直接相邻（包括对角线）
                        if dr == 0 or dc == 0:  # 水平或垂直相邻
                            weight = 3
                        else:  # 对角线相邻
                            weight = 2
                        possible_edges.append((node, neighbor_node, weight))
        
        # 首先确保图的连通性：使用深度优先搜索构建最小连通子图
        visited = set()
        edge_added = set()
        
        def dfs_connect(node):
            visited.add(node)
            row, col = node_positions[node]
            neighbors = get_adjacent_neighbors(row, col)
            
            for neighbor_pos in neighbors:
                neighbor_node = pos_to_node[neighbor_pos]
                edge = tuple(sorted([node, neighbor_node]))
                
                if neighbor_node not in visited and edge not in edge_added:
                    if add_edge(node, neighbor_node):
                        edge_added.add(edge)
                        dfs_connect(neighbor_node)
        
        # 从第一个节点开始构建连通图
        dfs_connect(nodes[0])
        
        # 如果还有未连接的节点，连接它们
        for node in nodes:
            if node not in visited:
                # 找到最近的已连接节点
                row, col = node_positions[node]
                neighbors = get_adjacent_neighbors(row, col)
                
                for neighbor_pos in neighbors:
                    neighbor_node = pos_to_node[neighbor_pos]
                    if neighbor_node in visited:
                        if add_edge(node, neighbor_node):
                            dfs_connect(node)
                            break
        
        # 根据概率添加额外的相邻连接来增加图的复杂度
        for node, neighbor_node, weight in possible_edges:
            if neighbor_node not in adjacency_list[node]:
                # 根据连接类型调整概率
                if weight == 3:  # 水平/垂直相邻
                    connect_prob = edge_prob * 1.0
                elif weight == 2:  # 对角线相邻
                    connect_prob = edge_prob * 0.7
                else:
                    connect_prob = edge_prob * 0.3
                
                if random.random() < connect_prob:
                    add_edge(node, neighbor_node)
        
        # 添加特殊的非重合对角线连接来增加复杂度
        def is_restricted_connection(node_row, node_col, other_row, other_col):
            """
            检查两个节点之间的连接是否受限制
            受限制的连接类型：同行、同列或标准对角线(n±k, n±k)
            这些连接只允许相邻的（距离为1）
            """
            # 检查是否同行或同列
            if node_row == other_row or node_col == other_col:
                return True
            
            # 检查是否是标准对角线(n±k, n±k)
            row_diff = abs(other_row - node_row)
            col_diff = abs(other_col - node_col)
            if row_diff == col_diff:
                return True
                
            return False
        
        def add_extended_connections():
            """
            添加扩展连接：
            1. 对于同行/同列/标准对角线：只允许相邻连接
            2. 对于非重合对角线：不衰减概率的连接
            """
            for node in nodes:
                node_row, node_col = node_positions[node]
                
                for other_node in nodes:
                    if other_node <= node:  # 避免重复检查
                        continue
                        
                    other_row, other_col = node_positions[other_node]
                    
                    # 如果已经连接，跳过
                    if other_node in adjacency_list[node]:
                        continue
                    
                    row_diff = abs(other_row - node_row)
                    col_diff = abs(other_col - node_col)
                    max_distance = max(row_diff, col_diff)
                    
                    # 判断连接类型
                    is_restricted = is_restricted_connection(node_row, node_col, other_row, other_col)
                    
                    if is_restricted:
                        # 受限制的连接：只允许相邻的（距离为1）
                        if max_distance == 1:
                            # 这些应该已经在前面的相邻连接中处理了
                            # 这里可以添加一些额外的相邻连接机会
                            connect_prob = edge_prob * 0.3
                            if random.random() < connect_prob:
                                add_edge(node, other_node)
                    else:
                        # 非重合对角线连接：不同行且不同列且不是标准对角线
                        if (node_row != other_row and node_col != other_col and 
                            row_diff != col_diff):
                            
                            # 对角链接不衰减，使用固定概率
                            connect_prob = edge_prob * 2
                            if random.random() < connect_prob:
                                add_edge(node, other_node)
        
        # 应用扩展连接
        add_extended_connections()
        
        # 专门针对哈密顿路径：优化图结构
        def optimize_for_hamiltonian_path():
            """
            针对哈密顿路径问题的优化：
            1. 允许最多两个度数为1的节点（作为路径的起点和终点）
            2. 控制连接密度，确保有些图有解有些无解
            """
            # 计算度数为1的节点
            degree_one_nodes = [node for node in nodes if len(adjacency_list[node]) == 1]
            
            # 如果度数为1的节点超过2个，需要修复
            while len(degree_one_nodes) > 2:
                node = degree_one_nodes[0]  # 取第一个度数为1的节点
                node_row, node_col = node_positions[node]
                
                # 尝试为其添加连接
                neighbors = get_adjacent_neighbors(node_row, node_col)
                connected = False
                
                # 优先连接到相邻节点
                for neighbor_pos in neighbors:
                    neighbor_node = pos_to_node[neighbor_pos]
                    if neighbor_node not in adjacency_list[node]:
                        if add_edge(node, neighbor_node):
                            connected = True
                            break
                
                # 如果相邻节点都已连接或达到度数限制，尝试非重合对角线
                if not connected:
                    for other_node in nodes:
                        if (other_node != node and 
                            other_node not in adjacency_list[node]):
                            
                            other_row, other_col = node_positions[other_node]
                            
                            # 检查是否为非重合对角线连接
                            if (node_row != other_row and node_col != other_col and
                                abs(other_row - node_row) != abs(other_col - node_col)):
                                if add_edge(node, other_node):
                                    connected = True
                                    break
                
                # 重新计算度数为1的节点
                degree_one_nodes = [node for node in nodes if len(adjacency_list[node]) == 1]
                
                # 防止无限循环
                if not connected:
                    break
        
        # 应用哈密顿路径专用优化
        optimize_for_hamiltonian_path()
        
        # 排序一下每个节点的邻居列表，以便输出更整齐
        for k in adjacency_list:
            adjacency_list[k].sort()
        
        return adjacency_list
    
    @staticmethod
    def has_hamiltonian_path(adj_list, start_node=None):
        """
        判断无向图是否存在哈密顿路径；若存在，返回第一条找到的路径（list），否则返回 None。
        使用回溯法，在小规模图可行。
        :param adj_list: 图的邻接表
        :param start_node: 指定的起点，None表示尝试所有起点
        """
        n = len(adj_list)
        visited = [False] * n
        path = []
        
        def backtrack(node, depth):
            # 将当前节点加入路径
            path.append(node)
            visited[node] = True
            
            # 若路径已经包含所有节点，则找到了一条完整的哈密顿路径
            if depth == n:
                return True
            
            # 尝试继续访问相邻节点
            for neigh in adj_list[node]:
                if not visited[neigh]:
                    if backtrack(neigh, depth + 1):
                        return True
            
            # 回溯
            visited[node] = False
            path.pop()
            return False
        
        # 如果指定了起点，只从该起点开始尝试
        if start_node is not None:
            if backtrack(start_node, 1):
                return path[:]
            return None
        
        # 尝试以每个节点作为起点
        for start_node in range(n):
            if backtrack(start_node, 1):
                return path[:]  # 成功找到后，返回一条可行解
        return None
    
    @staticmethod
    def draw_and_save_graph(adj_list, file_path, start_node=None):
        """
        使用 networkx 绘制并保存无向图，使用与生成时相同的网格布局。需要安装 networkx 和 matplotlib。
        如果指定了start_node，会用特殊颜色标注起点。
        """
        if nx is None or plt is None:
            return
        
        G = nx.Graph()
        # 添加节点、边
        for u in adj_list:
            G.add_node(u)
            for v in adj_list[u]:
                if v > u:  # 避免重复添加 (u,v) 和 (v,u)
                    G.add_edge(u, v)
        
        import math
        num_nodes = len(G.nodes())
        
        # 使用与生成时相同的网格大小计算
        if num_nodes <= 4:
            grid_cols, grid_rows = 2, 2
        elif num_nodes <= 9:
            grid_cols, grid_rows = 3, 3
        elif num_nodes <= 16:
            grid_cols, grid_rows = 4, 4
        else:
            grid_size = math.ceil(math.sqrt(num_nodes))
            grid_cols, grid_rows = grid_size, grid_size
        
        # 使用简单的顺序布局（与生成时一致）
        nodes = sorted(G.nodes())
        pos = {}
        
        for i, node in enumerate(nodes):
            row = i // grid_cols
            col = i % grid_cols
            # 使用更大的间距，并将坐标翻转（y轴向上）
            pos[node] = (col * 2.0, (grid_rows - 1 - row) * 2.0)
        
        # 计算图形的适当尺寸
        max_x = max(pos.values(), key=lambda p: p[0])[0] if pos else 0
        max_y = max(pos.values(), key=lambda p: p[1])[1] if pos else 0
        fig_width = max(8, max_x + 3)
        fig_height = max(8, max_y + 3)
        
        # 绘制图形
        plt.figure(figsize=(fig_width, fig_height))
        
        # 绘制边（使用适中的线宽和颜色）
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.8, edge_color='gray')
        
        # 设置节点颜色和大小
        node_colors = []
        node_sizes = []
        for node in nodes:
            if start_node is not None and node == start_node:
                node_colors.append('red')  # 起点用红色标注
                node_sizes.append(1400)  # 起点更大
            else:
                node_colors.append('lightblue')  # 其他节点用浅蓝色
                node_sizes.append(1200)  # 其他节点正常大小
        
        # 绘制节点（使用醒目的样式）
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                              edgecolors='darkblue', linewidths=2)
        
        # 绘制标签
        nx.draw_networkx_labels(G, pos, font_size=18, font_weight='bold', font_color='black')
        
        # 设置坐标轴范围，确保图形居中
        if pos:
            x_coords = [p[0] for p in pos.values()]
            y_coords = [p[1] for p in pos.values()]
            margin = 1.0
            plt.xlim(min(x_coords) - margin, max(x_coords) + margin)
            plt.ylim(min(y_coords) - margin, max(y_coords) + margin)
        
        
        plt.axis("off")
        plt.gca().set_aspect('equal')  # 确保比例一致
        plt.tight_layout()
        plt.savefig(file_path, dpi=150, bbox_inches='tight', pad_inches=0.3)
        plt.close()
    
    @staticmethod
    def generate_solution_steps(adj_list, has_path, path=None, start_node=None):
        """
        生成哈密顿路径问题的详细解决步骤，真正模拟回溯算法的执行过程。
        
        Args:
            adj_list: 图的邻接表
            has_path: 是否存在哈密顿路径
            path: 如果存在路径，则提供具体的路径
            start_node: 指定的起点，None表示尝试所有起点
            
        Returns:
            dict: 包含解法步骤的字典
        """
        # 使用networkx计算图的基本属性
        G = nx.Graph()
        for u in adj_list:
            G.add_node(u)
            for v in adj_list[u]:
                if v > u:  # 避免重复添加
                    G.add_edge(u, v)
        
        # 初始化步骤列表
        steps = []
        
        # 创建一个段落式的解法描述
        if start_node is not None:
            paragraph_description = f"To find a Hamiltonian path starting from vertex {start_node} in the given graph, I'll use a backtracking algorithm that systematically explores all possible paths."
        else:
            paragraph_description = "To find a Hamiltonian path in the given graph, I'll use a backtracking algorithm that systematically explores all possible paths."
        
        # 步骤1：初始化
        steps.append("Step 1: Initialize the search.")
        n = len(adj_list)
        steps.append(f"1.1: The graph has {n} vertices, so we need to find a path of length {n}.")
        if start_node is not None:
            steps.append(f"1.2: Starting vertex is specified as {start_node}.")
            steps.append("1.3: We'll use a backtracking algorithm that:")
        else:
            steps.append("1.2: We'll use a backtracking algorithm that:")
        steps.append("  - Maintains a current path")
        steps.append("  - Keeps track of visited vertices")
        steps.append("  - Tries each possible next vertex")
        
        # 步骤2：回溯搜索过程
        steps.append("\nStep 2: Backtracking search process.")
        
        # 模拟回溯搜索过程
        visited = [False] * n
        current_path = []
        
        def simulate_backtrack(node, depth):
            """
            模拟回溯过程，记录每一步的状态
            """
            visited[node] = True
            current_path.append(node)
            
            # 记录当前状态
            steps.append(f"\n2.{depth}: Current state:")
            steps.append(f"  - Current path: {' -> '.join(map(str, current_path))}")
            steps.append(f"  - Visited vertices: {sorted([i for i in range(n) if visited[i]])}")
            steps.append(f"  - Current vertex: {node}")
            
            if depth == n:
                steps.append("\nFound a valid Hamiltonian path!")
                return True
            
            # 尝试所有可能的邻居
            for neigh in adj_list[node]:
                if not visited[neigh]:
                    steps.append(f"  - Trying next vertex: {neigh}")
                    steps.append(f"  - Valid move: {neigh in adj_list[node]}")
                    
                    if simulate_backtrack(neigh, depth + 1):
                        return True
                    
                    steps.append(f"  - Backtracking from {neigh}")
            
            visited[node] = False
            current_path.pop()
            return False
        
        # 根据是否指定起点来搜索
        found_path = False
        if start_node is not None:
            steps.append(f"\nStarting from specified vertex {start_node}")
            if simulate_backtrack(start_node, 1):
                found_path = True
        else:
            # 从每个可能的起点开始搜索
            for start in range(n):
                steps.append(f"\nTrying starting vertex {start}")
                if simulate_backtrack(start, 1):
                    found_path = True
                    break
        
        if found_path:
            steps.append("\nFinal Conclusion: A Hamiltonian path exists in the graph.")
            steps.append(f"  - Complete path: {' -> '.join(map(str, current_path))}")
            steps.append(f"  - Path as list format: {current_path}")
            steps.append(f"  - Length: {len(current_path)}")
            steps.append(f"  - All vertices visited: {set(current_path) == set(range(n))}")
            steps.append(f"  - No repeated vertices: {len(set(current_path)) == len(current_path)}")
            
            # 验证路径
            valid_edges = True
            for i in range(len(current_path)-1):
                if current_path[i+1] not in adj_list[current_path[i]]:
                    valid_edges = False
                    break
            steps.append(f"  - All edges exist in the graph: {valid_edges}")
            steps.append(f"\nAnswer: {current_path}")
        else:
            steps.append("\nAfter exploring all possibilities:")
            steps.append("  - No valid Hamiltonian path found")
            steps.append("  - All possible paths have been explored")
            if start_node is not None:
                steps.append(f"\nFinal Conclusion: No Hamiltonian path exists starting from vertex {start_node}.")
            else:
                steps.append("\nFinal Conclusion: No Hamiltonian path exists in the graph.")
            steps.append("\nAnswer: No")
        
        paragraph_description += "\n\n" + "\n".join(steps)
        return {"reasoning_steps": paragraph_description}
    
    @staticmethod
    def generate_hamiltonian_path_problems(num_cases=5, difficulty=1, max_degree=3, use_random_start=False, output_folder="hamiltonian_path_problems"):
        """
        生成哈密顿路径问题集
        
        Args:
            num_cases: 生成的问题数量
            difficulty: 难度级别 (1-5)
            max_degree: 节点的最大度数限制
            use_random_start: 是否使用随机起点
            output_folder: 输出文件夹路径
            
        Returns:
            list: 生成的问题列表
        """
        images_folder = os.path.join(output_folder, "images")
        os.makedirs(images_folder, exist_ok=True)
        
        # 创建临时实例获取参数
        temp_instance = HamiltonianPathGenerator()
        params = temp_instance._get_difficulty_params(difficulty)
        
        num_nodes = params["num_nodes"]
        
        # 分别存放两类数据：带图片引用 / 不带图片引用
        with_image_data = []
        
        answer_fields = [0, 1, 1, 1, 1]
        
        for i in range(num_cases):
            edge_prob = params["edge_prob"]
            adjacency_list = HamiltonianPathGenerator.generate_random_connected_undirected_graph(
                num_nodes, 
                edge_prob, 
                max_degree, 
                seed=time.time()
            )
            
            # 随机选择起点（如果启用了随机起点）
            if use_random_start:
                start_node = random.randint(0, num_nodes - 1)
            else:
                start_node = None
            
            # 判断是否存在哈密顿路径，如存在则找到一条
            path_result = HamiltonianPathGenerator.has_hamiltonian_path(adjacency_list, start_node)
            
            answer_field = random.choice(answer_fields)
            while isinstance(path_result, list) != answer_field:
                adjacency_list = HamiltonianPathGenerator.generate_random_connected_undirected_graph(
                    num_nodes, 
                    edge_prob, 
                    max_degree, 
                    seed=time.time()
                )
                # 每次重新生成图时也重新选择起点
                if use_random_start:
                    start_node = random.randint(0, num_nodes - 1)
                path_result = HamiltonianPathGenerator.has_hamiltonian_path(adjacency_list, start_node)
            
            # 生成详细的解法步骤
            solution_steps = HamiltonianPathGenerator.generate_solution_steps(
                adjacency_list, 
                path_result is not None, 
                path_result, 
                start_node
            )
            
            # 绘图保存
            image_name = f"graph_{difficulty}_{i}.png"
            image_path = os.path.join(images_folder, image_name)
            HamiltonianPathGenerator.draw_and_save_graph(adjacency_list, image_path, start_node)
            
            # 组装题目文本
            adj_str = "\n".join([f"{u}: {adjacency_list[u]}" for u in sorted(adjacency_list.keys())])
            
            # 根据是否指定起点来调整问题描述
            if start_node is not None:
                question_text_with_image = (
                    f"Given an undirected graph below, determine whether "
                    f"a Hamiltonian path starting from vertex {start_node} (marked in red) exists. "
                    f"If it exists, output the path as a list (e.g., [0,1,2,3]). If not, output 'No'."
                )
                question_text_no_image = (
                    f"Given an undirected graph with adjacency list below, determine whether "
                    f"a Hamiltonian path starting from vertex {start_node} exists. "
                    f"If it exists, output the path as a list (e.g., [0,1,2,3]). If not, output 'No'.\n"
                    f"{adj_str}\n\n"
                )
            else:
                question_text_with_image = (
                    "Given an undirected graph below, determine whether "
                    "a Hamiltonian path exists. "
                    "If it exists, output the path as a list (e.g., [0,1,2,3]). If not, output 'No'."
                )
                question_text_no_image = (
                    "Given an undirected graph with adjacency list below, determine whether "
                    "a Hamiltonian path exists. "
                    "If it exists, output the path as a list (e.g., [0,1,2,3]). If not, output 'No'.\n"
                    f"{adj_str}\n\n"
                )
            
            # 存储答案
            if path_result is not None:
                answer_text = str(path_result)  # 输出路径列表格式
            else:
                answer_text = "No"
            
            # 生成 JSON 数据
            problem_id = f"hamiltonian_path_with_image_{difficulty}_{i}"
            if max_degree is not None:
                problem_id += f"_maxdeg{max_degree}"
            if start_node is not None:
                problem_id += f"_start{start_node}"
                
            with_image_entry = {
                "index": problem_id,
                "category": "hamiltonian_path",
                "question": question_text_with_image,
                "question_language": question_text_no_image,
                "difficulty": difficulty,
                "image": f"images/{image_name}",
                "answer": answer_text,
                "initial_state": adjacency_list,
            }
            
            with_image_data.append(with_image_entry)
        
        return with_image_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Hamiltonian path problems")  
    parser.add_argument("--num_cases", type=int, default=6, help="Number of problems to generate")
    parser.add_argument("--output_folder", type=str, default="hamiltonian_path_problems", help="Output folder")
    parser.add_argument("--max_degree", type=int, default=None, help="Maximum degree for vertices (None for no limit)")
    parser.add_argument("--use_random_start", type=bool, default=True, help="Use random start node for each problem")
    args = parser.parse_args()
    
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    
    generator = HamiltonianPathGenerator(output_folder=output_folder)
    problems = generator.generate(
        num_cases=args.num_cases, 
        difficulty=1,  # 可以根据需要设置不同的难度
        output_folder=output_folder,
        max_degree=args.max_degree,
        use_random_start=args.use_random_start
    )
    
    print(f"带图片的问题集已保存到 {output_folder}/annotations.json。")
    if args.max_degree is not None:
        print(f"顶点最大度数限制: {args.max_degree}")
    if args.use_random_start:
        print("使用随机起点")