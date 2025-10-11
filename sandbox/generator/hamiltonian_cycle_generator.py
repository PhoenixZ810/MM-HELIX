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


class HamiltonianCycleGenerator(BaseGenerator):
    """
    哈密顿回路问题生成器，生成图和相关的哈密顿回路问题
    """
    
    def __init__(self, output_folder="output/hamiltonian_cycle"):
        """
        初始化生成器
        
        Args:
            output_folder: 输出文件夹路径
        """
        super().__init__(output_folder)
    
    def generate(self, num_cases=10, difficulty=1, seed=None, output_folder=None):
        """
        生成哈密顿回路问题
        
        Args:
            num_cases: 每个难度级别生成的问题数量
            difficulty: 难度级别 (1-5)
            seed: 随机种子
            output_folder: 输出文件夹路径
            
        Returns:
            生成的问题列表
        """
        if output_folder is None:
            output_folder = self.output_folder
            
        os.makedirs(output_folder, exist_ok=True)
        
        # 调用生成哈密顿回路问题的方法
        problems = HamiltonianCycleGenerator.generate_hamiltonian_cycle_problems(
            num_cases=num_cases, 
            difficulty=difficulty,
            output_folder=output_folder
        )
        
        self.save_annotations(problems, output_folder)
            
        # 保存评分方法
        metrics_file = os.path.join(output_folder, "metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump({"score_function": "hamiltonian_cycle_evaluator"}, f, ensure_ascii=False, indent=4)
            
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
            "edge_prob": 0.2  # 随机边的概率
        }
        
        # 根据难度调整节点数量
        if difficulty == 1:
            params["num_nodes"] = 6
        elif difficulty == 2:
            params["num_nodes"] = 9
        elif difficulty == 3:
            params["num_nodes"] = 12
        elif difficulty == 4:
            params["num_nodes"] = 14
        elif difficulty == 5:
            params["num_nodes"] = 16
        else:
            # 默认难度
            params["num_nodes"] = 5
            
        return params
    
    @staticmethod
    def generate_random_connected_undirected_graph(num_nodes, edge_prob=0.1, seed=None):
        """
        基于网格结构生成连通的无向图，完全避免边的视觉交叉。
        策略：只允许网格中真正相邻的节点连接，确保所有边都不穿过其他节点。
        :param num_nodes: 节点数量
        :param edge_prob: 在相邻节点间添加边的概率
        :param seed: 随机种子（可选）
        :return: adjacency_list (dict), 例如 {0: [1,2], 1: [0,2], 2: [0,1], ...}
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
        elif num_nodes <= 25:
            grid_cols, grid_rows = 5, 5
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
            """添加无向边"""
            if v not in adjacency_list[u] and u != v:
                adjacency_list[u].append(v)
                adjacency_list[v].append(u)
        
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
                    add_edge(node, neighbor_node)
                    edge_added.add(edge)
                    dfs_connect(neighbor_node)
        
        # 从第一个节点开始构建连通图
        dfs_connect(nodes[0])
        
        # 如果还有未连接的节点，连接它们
        for node in nodes:
            if node not in visited:
                # 找到最近的已连接节点
                min_dist = float('inf')
                best_connection = None
                
                row, col = node_positions[node]
                neighbors = get_adjacent_neighbors(row, col)
                
                for neighbor_pos in neighbors:
                    neighbor_node = pos_to_node[neighbor_pos]
                    if neighbor_node in visited:
                        best_connection = neighbor_node
                        break
                
                if best_connection:
                    add_edge(node, best_connection)
                    dfs_connect(node)
        
        # 根据概率添加额外的相邻连接来增加图的复杂度
        for node, neighbor_node, weight in possible_edges:
            if neighbor_node not in adjacency_list[node]:
                # 根据连接类型调整概率
                if weight == 3:  # 水平/垂直相邻
                    connect_prob = edge_prob * 1.0
                elif weight == 2:  # 对角线相邻
                    connect_prob = edge_prob * 0.7
                else:
                    connect_prob = edge_prob * 0.2
                
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
                            connect_prob = edge_prob
                            if random.random() < connect_prob:
                                add_edge(node, other_node)
        
        # 应用扩展连接
        add_extended_connections()
        
        # 专门针对哈密顿回路：处理度数为1的点和控制问题难度
        def optimize_for_hamiltonian():
            """
            优化图结构以适合哈密顿回路问题：
            1. 避免度数为1的点（无法形成回路）
            2. 控制连接密度，确保有些图无解
            """
            # 检查并修复度数为1的节点
            def fix_degree_one_nodes():
                max_iterations = 10
                iteration = 0
                
                while iteration < max_iterations:
                    degree_one_nodes = []
                    
                    # 找到所有度数为1的节点
                    for node in nodes:
                        if len(adjacency_list[node]) == 1:
                            degree_one_nodes.append(node)
                    
                    if not degree_one_nodes:
                        break  # 没有度数为1的节点，退出
                    
                    # 为度数为1的节点添加连接
                    for node in degree_one_nodes:
                        node_row, node_col = node_positions[node]
                        
                        # 首先尝试连接到相邻的节点
                        neighbors = get_adjacent_neighbors(node_row, node_col)
                        connected = False
                        
                        for neighbor_pos in neighbors:
                            neighbor_node = pos_to_node[neighbor_pos]
                            if neighbor_node not in adjacency_list[node]:
                                add_edge(node, neighbor_node)
                                connected = True
                                break
                        
                        # 如果相邻节点都已连接，尝试非重合对角线连接
                        if not connected:
                            for other_node in nodes:
                                if (other_node != node and 
                                    other_node not in adjacency_list[node]):
                                    
                                    other_row, other_col = node_positions[other_node]
                                    
                                    # 检查是否为非重合对角线连接
                                    if (node_row != other_row and node_col != other_col and
                                        abs(other_row - node_row) != abs(other_col - node_col)):
                                        add_edge(node, other_node)
                                        connected = True
                                        break
                    
                    iteration += 1
            
            # 控制图的连接密度，创造有解和无解的情况
            def balance_connectivity():
                """
                根据随机因子决定是否增加连接以提高哈密顿回路存在的可能性
                """
                # 计算当前平均度数
                total_degree = sum(len(adj) for adj in adjacency_list.values())
                avg_degree = total_degree / len(nodes) if nodes else 0
                
                # 如果平均度数较低，有概率增加一些连接
                if avg_degree < 2.5 and random.random() < 0.6:
                    # 随机选择一些节点对增加连接
                    connection_attempts = min(3, len(nodes) // 2)
                    
                    for _ in range(connection_attempts):
                        node1 = random.choice(nodes)
                        node2 = random.choice(nodes)
                        
                        if (node1 != node2 and node2 not in adjacency_list[node1]):
                            node1_row, node1_col = node_positions[node1]
                            node2_row, node2_col = node_positions[node2]
                            
                            # 检查连接规则
                            if not is_restricted_connection(node1_row, node1_col, node2_row, node2_col):
                                # 如果是非重合对角线，允许连接
                                if (node1_row != node2_row and node1_col != node2_col and
                                    abs(node2_row - node1_row) != abs(node2_col - node1_col)):
                                    # add_edge(node1, node2)
                                    pass
                            elif max(abs(node2_row - node1_row), abs(node2_col - node1_col)) == 1:
                                # 相邻节点也允许连接
                                add_edge(node1, node2)
            
            # 执行优化
            fix_degree_one_nodes()
            balance_connectivity()
        
        # 应用哈密顿回路专用优化
        optimize_for_hamiltonian()
        
        # 每个邻接表按升序排序，方便输出美观
        for k in adjacency_list:
            adjacency_list[k].sort()
        
        return adjacency_list
    
    @staticmethod
    def find_hamiltonian_cycle(adj_list):
        """
        判断无向图是否存在哈密顿回路；若存在，返回"一个"可行解(循环的节点序列)；否则返回 None。
        回溯法：在小规模图可用。
        注意：返回的 cycle 列表最后会包含回到起点的边。例如若是 0->1->2->0，则返回 [0, 1, 2, 0] 表示完整的环路。
        """
        n = len(adj_list)
        visited = [False] * n
        path = []
    
        def backtrack(node, depth):
            """
            :param node: 当前节点
            :param depth: 已访问的节点计数
            """
            if depth == n:
                # 当已经访问了 n 个节点，如果 path[-1] 与 path[0] 相连，就成环
                first_node = path[0]
                last_node = path[-1]
                if first_node in adj_list[last_node]:
                    return True
                return False
    
            for neigh in adj_list[node]:
                if not visited[neigh]:
                    visited[neigh] = True
                    path.append(neigh)
    
                    if backtrack(neigh, depth + 1):
                        return True
    
                    visited[neigh] = False
                    path.pop()
    
            return False
    
        # 可以从任意节点开始，这里固定从 0 开始
        # 如果图有节点 0，则从 0 出发；若 num_nodes=0, 这种特殊情况就不讨论了
        start_node = 0
        visited[start_node] = True
        path.append(start_node)
    
        if backtrack(start_node, 1):
            path.append(start_node)  # 添加起点以形成完整环路
            return path
        else:
            return None
    
    @staticmethod
    def draw_and_save_graph(adj_list, file_path):
        """
        使用 networkx 绘制并保存无向图，使用与生成时相同的网格布局。需安装 networkx 和 matplotlib。
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
        
        # 绘制节点（使用醒目的样式）
        nx.draw_networkx_nodes(G, pos, node_size=1200, node_color='lightblue', 
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
    def generate_solution_steps(adj_list):
        """
        生成哈密顿回路问题的详细解决步骤，真正模拟回溯算法的执行过程。
        
        Args:
            adj_list: 图的邻接表
            has_cycle: 是否存在哈密顿回路
            cycle: 如果存在回路，则提供具体的回路
            
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
        paragraph_description = "To find a Hamiltonian cycle in the given graph, I'll use a backtracking algorithm that systematically explores all possible paths and checks if they can form a cycle."
        
        # 步骤1：初始化
        steps.append("Step 1: Initialize the search.")
        n = len(adj_list)
        steps.append(f"1.1: The graph has {n} vertices, so we need to find a cycle of length {n}.")
        steps.append("1.2: We'll use a backtracking algorithm that:")
        steps.append("  - Maintains a current path")
        steps.append("  - Keeps track of visited vertices")
        steps.append("  - Tries each possible next vertex")
        steps.append("  - Checks if the last vertex can connect back to the start")
        
        # 步骤2：回溯搜索过程
        steps.append("\nStep 2: Backtracking search process.")
        
        # 模拟回溯搜索过程
        visited = [False] * n
        current_path = []
        start_node = 0
        
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
                # 检查是否能形成回路
                first_node = current_path[0]
                last_node = current_path[-1]
                can_form_cycle = first_node in adj_list[last_node]
                
                steps.append(f"  - Path length reached {n}, checking if can form cycle:")
                steps.append(f"    * First vertex: {first_node}")
                steps.append(f"    * Last vertex: {last_node}")
                steps.append(f"    * Can connect back to start: {can_form_cycle}")
                
                if can_form_cycle:
                    steps.append("\nFound a valid Hamiltonian cycle!")
                    return True
                else:
                    steps.append("  - Cannot form cycle, backtracking...")
                    return False
            
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
        
        # 从起点开始搜索
        steps.append(f"\nStarting search from vertex {start_node}")
        if simulate_backtrack(start_node, 1):
            steps.append("\nFinal Conclusion: A Hamiltonian cycle exists in the graph.")
            steps.append(f"  - Complete cycle: {' -> '.join(map(str, current_path))} -> {current_path[0]}")
            steps.append(f"  - Cycle as list format: {current_path}")
            steps.append(f"  - Length: {len(current_path)}")
            steps.append(f"  - All vertices visited: {set(current_path) == set(range(n))}")
            steps.append(f"  - No repeated vertices: {len(set(current_path)) == len(current_path)}")
            
            # 验证路径
            valid_edges = True
            for i in range(len(current_path)-1):
                if current_path[i+1] not in adj_list[current_path[i]]:
                    valid_edges = False
                    break
            if current_path[0] not in adj_list[current_path[-1]]:
                valid_edges = False
            steps.append(f"  - All edges exist in the graph: {valid_edges}")
            steps.append(f"\nAnswer: {current_path}")
        else:
            steps.append("\nAfter exploring all possibilities:")
            steps.append("  - No valid Hamiltonian cycle found")
            steps.append("  - All possible paths have been explored")
            steps.append("\nFinal Conclusion: No Hamiltonian cycle exists in the graph.")
            steps.append("\nAnswer: No")
        
        paragraph_description += "\n\n" + "\n".join(steps)
        return {"reasoning_steps": paragraph_description}
    
    @staticmethod
    def generate_hamiltonian_cycle_problems(num_cases=5, difficulty=1, output_folder="hamiltonian_cycle_problems"):
        """
        生成哈密顿回路问题集
        
        Args:
            num_cases: 生成的问题数量
            difficulty: 难度级别 (1-5)
            output_folder: 输出文件夹路径
            
        Returns:
            list: 生成的问题列表
        """
        images_folder = os.path.join(output_folder, "images")
        os.makedirs(images_folder, exist_ok=True)
        
        # 创建临时实例获取参数
        temp_instance = HamiltonianCycleGenerator()
        params = temp_instance._get_difficulty_params(difficulty)
        
        num_nodes = params["num_nodes"]
        edge_prob = params["edge_prob"]
        
        with_image_data = []
        answer_fields = [0, 1, 1, 1, 1]
        
        for i in range(num_cases):
            adjacency_list = HamiltonianCycleGenerator.generate_random_connected_undirected_graph(
                num_nodes, 
                edge_prob, 
                seed=time.time()
            )
            
            cycle_result = HamiltonianCycleGenerator.find_hamiltonian_cycle(adjacency_list)
            
            answer_field = random.choice(answer_fields)
            while isinstance(cycle_result, list) != answer_field:
                adjacency_list = HamiltonianCycleGenerator.generate_random_connected_undirected_graph(
                    num_nodes, 
                    edge_prob, 
                    seed=time.time()
                )
                cycle_result = HamiltonianCycleGenerator.find_hamiltonian_cycle(adjacency_list)
                
            solution_steps = HamiltonianCycleGenerator.generate_solution_steps(adjacency_list)
            
            image_name = f"graph_{difficulty}_{i}.png"
            image_path = os.path.join(images_folder, image_name)
            HamiltonianCycleGenerator.draw_and_save_graph(adjacency_list, image_path)
            
            adj_str = "\n".join([f"{u}: {adjacency_list[u]}" for u in sorted(adjacency_list.keys())])
            question_text = (
                "Given the undirected, connected graph below, "
                "determine if there is a Hamiltonian cycle. "
                "If it exists, output the cycle as a list (e.g., [0,1,2,3]). If not, output 'No'. "
            )
            question_text_no_image = (
                "Given the undirected, connected graph below in adjacency-list form, "
                "determine if there is a Hamiltonian cycle. "
                "If it exists, output the cycle as a list (e.g., [0,1,2,3]). If not, output 'No'. "
                f"{adj_str}\n"
            )
            
            if cycle_result is not None:
                answer_text = str(cycle_result)  # 输出回路列表格式
            else:
                answer_text = "No"
            
            with_image_entry = {
                "index": f"hamiltonian_cycle_with_image_{difficulty}_{i}",
                "category": "hamiltonian_cycle",
                "question": question_text,
                "question_language": question_text_no_image,
                "difficulty": difficulty,
                "image": f"images/{image_name}",
                "answer": answer_text,
                "initial_state": adjacency_list,
            }
            
            with_image_data.append(with_image_entry)
        
        return with_image_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Hamiltonian cycle problems")
    parser.add_argument("--num_cases", type=int, default=6, help="Number of problems to generate")
    parser.add_argument("--difficulty", type=int, default=1, help="Difficulty level (1-5)")
    parser.add_argument("--output_folder", type=str, default="hamiltonian_cycle_problems", help="Output folder")
    args = parser.parse_args()
    
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    
    generator = HamiltonianCycleGenerator(output_folder=output_folder)
    problems = generator.generate(
        num_cases=args.num_cases, 
        difficulty=args.difficulty,
        output_folder=output_folder
    )
    
    print(f"Done! Files are in the folder: {output_folder}")