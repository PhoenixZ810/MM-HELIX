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


class EulerianPathGenerator(BaseGenerator):
    """
    欧拉路径问题生成器，生成图和相关的欧拉路径问题
    """
    
    def __init__(self, output_folder="output/eulerian_path"):
        """
        初始化生成器
        
        Args:
            output_folder: 输出文件夹路径
        """
        super().__init__(output_folder)
    
    def generate(self, num_cases=10, difficulty=1, seed=None, output_folder=None):
        """
        生成欧拉路径问题
        
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
        
        # 调用生成欧拉路径问题的方法
        problems = EulerianPathGenerator.generate_eulerian_path_problems(
            num_cases=num_cases, 
            difficulty=difficulty,
            output_folder=output_folder
        )
        
        self.save_annotations(problems, output_folder)
        # 保存评分方法
        metrics_file = os.path.join(output_folder, "metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump({"score_function": "eulerian_path_evaluator"}, f, ensure_ascii=False, indent=4)
            
        return problems
    
    def _get_difficulty_params(self, difficulty):
        """
        根据难度级别返回合适的参数
        
        Args:
            difficulty: 难度级别 (1-5)
            
        Returns:
            dict: 包含图生成参数的字典
        """
        params = {}
        
        if difficulty == 5:
            params["num_nodes"] = 16
            params["edge_prob"] = 0.2
        elif difficulty == 4:
            params["num_nodes"] = 12
            params["edge_prob"] = 0.2
        elif difficulty == 3:
            params["num_nodes"] = 10
            params["edge_prob"] = 0.15
        elif difficulty == 2:
            params["num_nodes"] = 8
            params["edge_prob"] = 0.25
        elif difficulty == 1:
            params["num_nodes"] = 6
            params["edge_prob"] = 0.25
        else:
            # 默认难度
            params["num_nodes"] = 6
            params["edge_prob"] = 0.25
            
        return params
    
    @staticmethod
    def generate_random_connected_undirected_graph(num_nodes, edge_prob=0.3, seed=None):
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
        
        # 每个邻接表按升序排序，方便输出美观
        for k in adjacency_list:
            adjacency_list[k].sort()
        
        return adjacency_list
    
    @staticmethod
    def is_eulerian_path(adj_list):
        """
        判断一个无向连通图是否有欧拉路径：
          - 若所有顶点度数都为偶数，或恰好有两个顶点度数为奇数，则返回 True
          - 否则返回 False
        """
        odd_degree_count = 0
        for node in adj_list:
            if len(adj_list[node]) % 2 != 0:
                odd_degree_count += 1
        return odd_degree_count == 0 or odd_degree_count == 2
    
    @staticmethod
    def find_eulerian_path(adj_list):
        """
        如果图存在欧拉路径，使用 Hierholzer 算法构造并返回一条欧拉路径。
        否则返回 None。
        
        返回的路径是一个顶点序列，例如 [0, 2, 1, 3] 表示 0->2->1->3。
        同时返回算法执行步骤的记录。
        """
        if not EulerianPathGenerator.is_eulerian_path(adj_list):
            return None, []
        
        # 记录算法执行步骤
        algorithm_steps = []
        
        # 将图拷贝一份，因为构造欧拉路径会"消耗"边
        graph_copy = {node: adj_list[node][:] for node in adj_list}
        algorithm_steps.append("Step 1: Copy the adjacency list to avoid modifying the original graph")
        
        # 如果有奇数度的顶点，从其中一个开始
        # 否则从任意顶点开始
        start_node = None
        for node in graph_copy:
            if len(graph_copy[node]) % 2 == 1:
                start_node = node
                break
        if start_node is None:
            start_node = next(iter(graph_copy.keys()))
            algorithm_steps.append(f"Step 2: All vertices have even degree, choose vertex {start_node} as starting point")
        else:
            algorithm_steps.append(f"Step 2: Choose odd-degree vertex {start_node} as starting point")
        
        path = []
        trail = [start_node]
        algorithm_steps.append(f"Step 3: Initialize trail=[{start_node}], path=[]")
        
        step_count = 4
        while trail:
            current = trail[-1]
            if graph_copy[current]:
                # 取出一个邻居
                neighbor = graph_copy[current].pop()
                # 同时从 neighbor 那边也删去 current
                graph_copy[neighbor].remove(current)
                trail.append(neighbor)
                algorithm_steps.append(f"Step {step_count}: Move from vertex {current} to neighbor {neighbor}, update trail={trail}")
            else:
                # 当前节点已无可走的边，将它弹出添加到路径
                removed = trail.pop()
                path.append(removed)
                algorithm_steps.append(f"Step {step_count}: Vertex {removed} has no more edges, add to path, current trail={trail}")
            step_count += 1
        
        # path 逆序存放了路径，翻转一下
        path.reverse()
        algorithm_steps.append(f"Step {step_count}: Reverse the path to get final result: {path}")
        
        return path, algorithm_steps
    
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
                if v > u:  # 避免重复添加
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
    def generate_solution_steps(adj_list, has_path, path=None, algorithm_steps=[]):
        """
        生成欧拉路径问题的详细解决步骤，记录判断和构造过程。
        
        Args:
            adj_list: 图的邻接表
            has_path: 是否存在欧拉路径
            path: 如果存在路径，则提供具体的路径
            algorithm_steps: 算法执行步骤
            
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
        paragraph_description = "To determine if an Eulerian path exists in the given graph, I'll follow these steps:"
        
        # 步骤1：检查基本条件
        steps.append("Step 1: Check basic conditions for Eulerian path.")
        steps.append("1.1: For a graph to have an Eulerian path, it must have either:")
        steps.append("  - All vertices with even degrees, or")
        steps.append("  - Exactly two vertices with odd degrees (these will be the start and end vertices)")
        
        # 检查每个顶点的度数
        odd_vertices = []
        for node in adj_list:
            degree = len(adj_list[node])
            steps.append(f"  - Vertex {node}: degree = {degree} ({'odd' if degree % 2 != 0 else 'even'})")
            if degree % 2 != 0:
                odd_vertices.append(node)
        
        # 步骤2：判断是否存在欧拉路径
        odd_count = len(odd_vertices)
        steps.append(f"\n1.2: Number of vertices with odd degree: {odd_count}")
        
        if odd_count == 0:
            steps.append("  - All vertices have even degree")
            steps.append("  - This means an Eulerian path exists (in fact, an Eulerian cycle exists)")
            
            if has_path and path is not None:
                # 步骤3：显示构造的欧拉路径
                steps.append("\nStep 2: Eulerian path found using Hierholzer's algorithm.")
                steps.append(f"2.1: Complete path: {' -> '.join(map(str, path))}")
                steps.append(f"2.2: Path as list format: {path}")
                steps.append(f"2.3: Length of path: {len(path)}")
                
                # 添加算法执行步骤的简洁叙述
                if algorithm_steps:
                    steps.append("\n2.4: Algorithm execution steps:")
                    for step in algorithm_steps:
                        steps.append(f"  - {step}")
                
                # 验证路径
                valid_edges = True
                for i in range(len(path)-1):
                    if path[i+1] not in adj_list[path[i]]:
                        valid_edges = False
                        break
                steps.append(f"2.5: All edges exist in the graph: {valid_edges}")
                
                steps.append("\nFinal Conclusion: The graph has an Eulerian path.")
                steps.append(f"Answer: {path}")
            else:
                steps.append("\nFinal Conclusion: The graph has an Eulerian path.")
                steps.append("Answer: [path would be constructed here]")
                
        elif odd_count == 2:
            steps.append(f"  - Exactly two vertices have odd degree: {odd_vertices}")
            steps.append("  - These will be the start and end vertices of the Eulerian path")
            
            if has_path and path is not None:
                # 步骤3：显示构造的欧拉路径
                steps.append("\nStep 2: Eulerian path found using Hierholzer's algorithm.")
                steps.append(f"2.1: Complete path: {' -> '.join(map(str, path))}")
                steps.append(f"2.2: Path as list format: {path}")
                steps.append(f"2.3: Length of path: {len(path)}")
                steps.append(f"2.4: Start vertex: {path[0]} (degree: {len(adj_list[path[0]])})")
                steps.append(f"2.5: End vertex: {path[-1]} (degree: {len(adj_list[path[-1]])})")
                
                # 添加算法执行步骤的简洁叙述
                if algorithm_steps:
                    steps.append("\n2.6: Algorithm execution steps:")
                    for step in algorithm_steps:
                        steps.append(f"  - {step}")
                
                # 验证路径
                valid_edges = True
                for i in range(len(path)-1):
                    if path[i+1] not in adj_list[path[i]]:
                        valid_edges = False
                        break
                steps.append(f"2.7: All edges exist in the graph: {valid_edges}")
                
                steps.append("\nFinal Conclusion: The graph has an Eulerian path.")
                steps.append(f"Answer: {path}")
            else:
                steps.append("\nFinal Conclusion: The graph has an Eulerian path.")
                steps.append("Answer: [path would be constructed here]")
        else:
            steps.append(f"  - Found {odd_count} vertices with odd degree: {odd_vertices}")
            steps.append("\nFinal Conclusion: The graph does not have an Eulerian path because it has an invalid number of odd-degree vertices.")
            steps.append("Answer: No")
        
        paragraph_description += "\n\n" + "\n".join(steps)
        return {"reasoning_steps": paragraph_description, "algorithm_steps": algorithm_steps}
    
    @staticmethod
    def generate_eulerian_path_problems(num_cases=6, difficulty=1, output_folder="eulerian_path_problems"):
        """
        生成欧拉路径问题集
        
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
        temp_instance = EulerianPathGenerator()
        params = temp_instance._get_difficulty_params(difficulty)
        
        num_nodes = params["num_nodes"]
        edge_prob = params["edge_prob"]
        
        with_image_data = []
        answer_fields = [0, 1, 1, 1]
        
        for i in range(num_cases):
            # 1) 生成连通无向图
            adjacency_list = EulerianPathGenerator.generate_random_connected_undirected_graph(
                num_nodes, 
                edge_prob, 
                seed=time.time()
            )
            
            # 2) 查找欧拉路径
            path, algorithm_steps = EulerianPathGenerator.find_eulerian_path(adjacency_list)
            
            answer_field = random.choice(answer_fields)
            while isinstance(path, list) != answer_field:
                adjacency_list = EulerianPathGenerator.generate_random_connected_undirected_graph(
                    num_nodes, 
                    edge_prob, 
                    seed=time.time()
                )
                path, algorithm_steps = EulerianPathGenerator.find_eulerian_path(adjacency_list)
            
            # 3) 生成详细的解法步骤
            solution_steps = EulerianPathGenerator.generate_solution_steps(
                adjacency_list, 
                path is not None, 
                path, 
                algorithm_steps
            )
            
            # 4) 画图保存
            image_name = f"graph_{difficulty}_{i}.png"
            image_path = os.path.join(images_folder, image_name)
            EulerianPathGenerator.draw_and_save_graph(adjacency_list, image_path)
            
            # 5) 题目描述
            adj_str = "\n".join([f"{u}: {adjacency_list[u]}" for u in sorted(adjacency_list.keys())])
            question_text = (
                "Given the undirected, connected graph below, "
                "determine if there is an Eulerian path. "
                "If it exists, output the path as a list (e.g., [0,1,2,3]). If not, output 'No'."
            )
            question_text_no_image = (
                "Given the undirected, connected graph below in adjacency-list form, "
                "determine if there is an Eulerian path. "
                "If it exists, output the path as a list (e.g., [0,1,2,3]). If not, output 'No'."
                f"{adj_str}\n"
            )
            
            # 6) 答案
            if path is not None:
                answer_text = str(path)  # 输出路径列表格式
            else:
                answer_text = "No"
            
            # 7) 生成 JSON 条目
            with_image_entry = {
                "index": f"eulerian_path_with_image_{difficulty}_{i}",
                "category": "eulerian_path",
                "question_language": question_text_no_image,
                "difficulty": difficulty,
                "question": question_text,
                "image": f"images/{image_name}",
                "answer": answer_text,
                "initial_state": adjacency_list,
            }
            
            with_image_data.append(with_image_entry)
        
        return with_image_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate Eulerian path problems")
    parser.add_argument("--num_cases", type=int, default=6, help="Number of problems to generate")
    parser.add_argument("--difficulty", type=int, default=1, help="Difficulty level")
    parser.add_argument("--output_folder", type=str, default="eulerian_path_problems", help="Output folder")
    args = parser.parse_args()
    
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    
    generator = EulerianPathGenerator(output_folder=output_folder)
    problems = generator.generate(
        num_cases=args.num_cases, 
        difficulty=args.difficulty,
        output_folder=output_folder
    )
    
    print(f"带图片的问题集已保存到 {output_folder}/annotations.json。")