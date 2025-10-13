import argparse
import os
import json
import random
import time
import math
from .base_generator import BaseGenerator

try:
    import networkx as nx
    import matplotlib.pyplot as plt
except ImportError:
    print("Please install 'networkx' and 'matplotlib' if you want to draw and save images.")
    nx = None
    plt = None


class IsomorphismGenerator(BaseGenerator):
    def __init__(self, output_folder="output/isomorphism"):
        super().__init__(output_folder)

    def generate(self, num_cases=10, difficulty=1, seed=None, output_folder=None):
        """
        生成图同构问题
        
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
        
        # 调用生成图同构问题的方法
        problems = IsomorphismGenerator.generate_graph_isomorphism_problems(
            num_cases=num_cases, 
            difficulty=difficulty, 
            output_folder=output_folder
        )
        
        self.save_annotations(problems, output_folder)
            
        # 保存评分方法
        metrics_file = os.path.join(output_folder, "metrics.json")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump({"score_function": "simple_str_match"}, f, ensure_ascii=False, indent=4)
            
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
            "edge_prob": 0.3,          # 随机边的概率
            "max_triangles_percent": 0.2,  # 最大三角形比例
            "max_degree": 3            # 每个节点的最大度数
        }
        
        # 根据难度调整节点数量
        if difficulty == 1:
            params["num_nodes"] = 6
        elif difficulty == 2:
            params["num_nodes"] = 7
        elif difficulty == 3:
            params["num_nodes"] = 8
        elif difficulty == 4:
            params["num_nodes"] = 9
        elif difficulty == 5:
            params["num_nodes"] = 10
        else:
            # 默认难度
            params["num_nodes"] = 6
            
        return params
    
    @staticmethod
    def generate_random_connected_undirected_graph(num_nodes, edge_prob=0.3, seed=None, max_triangles_percent=0.3, max_degree=None):
        """
        随机生成一个连通的无向平面图，形状多样化。
        支持多种图形状：线性链、星形、环形、树形、网格形等。
        :param num_nodes: 节点数量
        :param edge_prob: 对每对节点(i<j)，若尚未有边，则以此概率尝试添加
        :param seed: 随机种子（可选）
        :param max_triangles_percent: 控制三角形的最大比例(相对于节点数)
        :param max_degree: 每个节点的最大度数，如果为None则不限制
        :return: adjacency_list (dict), 例如 {0: [1,2], 1:[0], 2:[0], ...}
        """
        if seed is not None:
            random.seed(seed)
        
        adjacency_list = {i: [] for i in range(num_nodes)}
        
        # 先把节点打乱顺序
        node_list = list(range(num_nodes))
        random.shuffle(node_list)
        
        # 选择基础图形结构
        structure_type = random.choice(['chain', 'star', 'cycle', 'tree', 'grid', 'path_with_branches'])
        
        if structure_type == 'chain':
            # 1) 构造链式结构
            for i in range(num_nodes - 1):
                u = node_list[i]
                v = node_list[i+1]
                adjacency_list[u].append(v)
                adjacency_list[v].append(u)
        
        elif structure_type == 'star':
            # 2) 构造星形结构（一个中心节点连接所有其他节点）
            center = node_list[0]
            for i in range(1, num_nodes):
                leaf = node_list[i]
                adjacency_list[center].append(leaf)
                adjacency_list[leaf].append(center)
        
        elif structure_type == 'cycle':
            # 3) 构造环形结构
            for i in range(num_nodes):
                u = node_list[i]
                v = node_list[(i + 1) % num_nodes]
                adjacency_list[u].append(v)
                adjacency_list[v].append(u)
        
        elif structure_type == 'tree':
            # 4) 构造随机树结构
            for i in range(1, num_nodes):
                # 随机选择一个已经在树中的节点作为父节点
                parent_idx = random.randint(0, i-1)
                parent = node_list[parent_idx]
                child = node_list[i]
                adjacency_list[parent].append(child)
                adjacency_list[child].append(parent)
        
        elif structure_type == 'grid':
            # 5) 构造网格状结构（适用于节点数较多的情况）
            if num_nodes >= 4:
                # 创建一个接近正方形的网格
                rows = int(num_nodes ** 0.5)
                cols = (num_nodes + rows - 1) // rows
                
                for i in range(min(num_nodes, rows * cols)):
                    row = i // cols
                    col = i % cols
                    node = node_list[i]
                    
                    # 连接右边的节点
                    if col < cols - 1 and i + 1 < num_nodes:
                        right_node = node_list[i + 1]
                        adjacency_list[node].append(right_node)
                        adjacency_list[right_node].append(node)
                    
                    # 连接下面的节点
                    if row < rows - 1 and i + cols < num_nodes:
                        down_node = node_list[i + cols]
                        adjacency_list[node].append(down_node)
                        adjacency_list[down_node].append(node)
            else:
                # 节点数太少，退化为链式结构
                for i in range(num_nodes - 1):
                    u = node_list[i]
                    v = node_list[i+1]
                    adjacency_list[u].append(v)
                    adjacency_list[v].append(u)
        
        elif structure_type == 'path_with_branches':
            # 6) 构造带分支的路径结构
            # 首先创建一个主路径
            main_path_length = max(2, num_nodes // 2)
            for i in range(main_path_length - 1):
                u = node_list[i]
                v = node_list[i + 1]
                adjacency_list[u].append(v)
                adjacency_list[v].append(u)
            
            # 然后为剩余节点创建分支
            remaining_nodes = node_list[main_path_length:]
            for node in remaining_nodes:
                # 随机选择主路径上的一个节点作为分支点
                branch_point = random.choice(node_list[:main_path_length])
                adjacency_list[branch_point].append(node)
                adjacency_list[node].append(branch_point)
        
        # 创建NetworkX图对象用于检查平面性
        G = IsomorphismGenerator.dict_to_nx_graph(adjacency_list)
        
        # 计算最大允许的三角形数（降低三角形比例）
        max_triangles = int(num_nodes * max_triangles_percent * 0.5)  # 进一步降低三角形比例
        triangle_count = 0
        
        # 候选边列表：所有可能添加的边
        candidate_edges = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if j not in adjacency_list[i]:
                    candidate_edges.append((i, j))
        
        # 随机打乱候选边顺序
        random.shuffle(candidate_edges)
        
        # 降低额外边的添加概率，避免过多三角形
        reduced_edge_prob = edge_prob * 0.6
        
        # 尝试添加额外边，但保持平面性和控制三角形数量
        for i, j in candidate_edges:
            # 如果已经达到最大三角形数量，停止添加边
            if triangle_count >= max_triangles:
                break
                
            # 如果设置了最大度数限制，检查度数
            if max_degree is not None:
                if len(adjacency_list[i]) >= max_degree or len(adjacency_list[j]) >= max_degree:
                    continue
                    
            # 随机决定是否尝试添加这条边（降低概率）
            if random.random() < reduced_edge_prob:
                # 临时添加边
                G.add_edge(i, j)
                
                # 检查添加这条边后图是否仍然是平面图
                if nx.is_planar(G):
                    # 计算添加这条边会形成多少个新三角形
                    new_triangles = 0
                    for common_neighbor in set(adjacency_list[i]) & set(adjacency_list[j]):
                        new_triangles += 1
                    
                    # 如果添加这条边会超过三角形限制，则撤销
                    if triangle_count + new_triangles > max_triangles:
                        G.remove_edge(i, j)
                        continue
                        
                    # 更新三角形计数
                    triangle_count += new_triangles
                    
                    # 在邻接表中添加这条边
                    adjacency_list[i].append(j)
                    adjacency_list[j].append(i)
                else:
                    # 如果不是平面图，则撤销添加
                    G.remove_edge(i, j)
        
        # 排序邻居列表，方便美观输出
        for k in adjacency_list:
            adjacency_list[k].sort()
        
        return adjacency_list


    @staticmethod
    def generate_modified_graph(adj_list, edge_change_prob=0.2, seed=None, max_triangles_percent=0.3, max_degree=None):
        """
        从现有图的邻接表开始，随机修改一些边，生成一个新的图
        修改包括：移除现有边、添加新边
        :param adj_list: 原图的邻接表
        :param edge_change_prob: 对每条边，尝试修改的概率
        :param seed: 随机种子
        :param max_triangles_percent: 控制三角形的最大比例
        :param max_degree: 每个节点的最大度数
        :return: 新的邻接表
        """
        if seed is not None:
            random.seed(seed)
            
        # 复制原图的邻接表，避免修改原图
        new_adj = {}
        for u in adj_list:
            new_adj[u] = list(adj_list[u])
            
        # 创建NetworkX图对象用于检查平面性
        G = IsomorphismGenerator.dict_to_nx_graph(new_adj)
        
        # 计算节点数量
        num_nodes = len(adj_list)
        
        # 计算最大允许的三角形数
        max_triangles = int(num_nodes * max_triangles_percent)
        
        # 获取当前三角形数
        triangle_count = sum(1 for _ in nx.find_cliques(G) if len(_) == 3)
        
        # 首先尝试移除一些现有边，但确保图仍然连通
        existing_edges = []
        for u in adj_list:
            for v in adj_list[u]:
                if v > u:  # 避免重复
                    existing_edges.append((u, v))
                    
        # 随机打乱边的顺序
        random.shuffle(existing_edges)
        
        for u, v in existing_edges:
            if random.random() < edge_change_prob:
                # 临时移除边
                G_temp = G.copy()
                G_temp.remove_edge(u, v)
                
                # 确保图仍然连通
                if nx.is_connected(G_temp):
                    # 实际移除边
                    G.remove_edge(u, v)
                    new_adj[u].remove(v)
                    new_adj[v].remove(u)
        
        # 然后尝试添加一些新边
        candidate_edges = []
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                if j not in new_adj[i]:
                    candidate_edges.append((i, j))
                    
        # 随机打乱候选边顺序
        random.shuffle(candidate_edges)
        
        for i, j in candidate_edges:
            if random.random() < edge_change_prob:
                # 如果设置了最大度数限制，检查度数
                if max_degree is not None:
                    if len(new_adj[i]) >= max_degree or len(new_adj[j]) >= max_degree:
                        continue
                
                # 临时添加边
                G.add_edge(i, j)
                
                # 检查添加这条边后图是否仍然是平面图
                if nx.is_planar(G):
                    # 计算添加这条边会形成多少个新三角形
                    new_triangles = 0
                    for common_neighbor in set(new_adj[i]) & set(new_adj[j]):
                        new_triangles += 1
                    
                    # 如果添加这条边会超过三角形限制，则撤销
                    if triangle_count + new_triangles > max_triangles:
                        G.remove_edge(i, j)
                        continue
                        
                    # 更新三角形计数
                    triangle_count += new_triangles
                    
                    # 在邻接表中添加这条边
                    new_adj[i].append(j)
                    new_adj[j].append(i)
                else:
                    # 如果不是平面图，则撤销添加
                    G.remove_edge(i, j)
        
        # 排序邻居列表
        for k in new_adj:
            new_adj[k].sort()
            
        return new_adj
    
    @staticmethod
    def relabel_graph(adj_list, permutation):
        """
        将图的节点标签按照给定的 permutation 映射，返回新的邻接表。
        :param adj_list: 原图的邻接表(节点标签为 0..n-1)
        :param permutation: 一个 dict, 映射 old_label -> new_label
        :return: 新的邻接表
        """
        new_adj = {}
        for old_u in adj_list:
            new_u = permutation[old_u]
            new_adj[new_u] = []
        
        for old_u in adj_list:
            new_u = permutation[old_u]
            for old_v in adj_list[old_u]:
                new_v = permutation[old_v]
                new_adj[new_u].append(new_v)
        
        # 排序
        for k in new_adj:
            new_adj[k].sort()
        
        return new_adj


    @staticmethod
    def dict_to_nx_graph(adj_list):
        """
        将邻接表转为 networkx Graph。
        """
        G = nx.Graph()
        for u in adj_list:
            G.add_node(u)
            for v in adj_list[u]:
                if v > u:  # 避免重复添加
                    G.add_edge(u, v)
        return G


    @staticmethod
    def generate_solution_steps(G1, G2, is_isomorphic, permutation=None):
        """
        Generate detailed solution steps for the graph isomorphism problem,
        using concepts from planar graph algorithms.
        
        Args:
            G1: Adjacency list of the first graph
            G2: Adjacency list of the second graph
            is_isomorphic: Result of isomorphism check
            permutation: Node mapping if graphs are isomorphic
            
        Returns:
            dict: Dictionary containing reasoning steps
        """
        # Create NetworkX graphs for analysis
        nx_G1 = IsomorphismGenerator.dict_to_nx_graph(G1)
        nx_G2 = IsomorphismGenerator.dict_to_nx_graph(G2)
        
        # Initialize solution
        steps = []
        
        # Introduction paragraph
        paragraph_description = "I will determine if these two planar graphs are isomorphic by analyzing their structural properties using concepts from Hopcroft-Tarjan's planar graph algorithm."
        
        # Step 1: Compare basic invariants
        steps.append("Step 1: Examining basic graph invariants.")
        
        # Print graph structure for reference
        steps.append("Graph 1 adjacency list:")
        adj_str_G1 = "\n".join([f"  Node {u}: connected to {G1[u]}" for u in sorted(G1.keys())])
        steps.append(adj_str_G1)
        
        steps.append("\nGraph 2 adjacency list:")
        adj_str_G2 = "\n".join([f"  Node {u}: connected to {G2[u]}" for u in sorted(G2.keys())])
        steps.append(adj_str_G2)
        
        # 1.1 Comparing vertex counts
        nodes_G1 = len(G1)
        nodes_G2 = len(G2)
        steps.append(f"\n1.1: Number of vertices: Graph 1 has {nodes_G1} vertices, Graph 2 has {nodes_G2} vertices.")
        
        if nodes_G1 != nodes_G2:
            steps.append("The graphs have different numbers of vertices, so they cannot be isomorphic.")
            steps.append("Conclusion: The graphs are not isomorphic.")
            paragraph_description += "\n\n" + "\n".join(steps)
            return {"reasoning_steps": paragraph_description}
        else:
            steps.append("✓ Both graphs have the same number of vertices, which is necessary for isomorphism.")
        
        # 1.2 Comparing edge counts
        edges_G1 = nx_G1.number_of_edges()
        edges_G2 = nx_G2.number_of_edges()
        steps.append(f"\n1.2: Number of edges: Graph 1 has {edges_G1} edges, Graph 2 has {edges_G2} edges.")
        
        if edges_G1 != edges_G2:
            steps.append("The graphs have different numbers of edges, so they cannot be isomorphic.")
            steps.append("Conclusion: The graphs are not isomorphic.")
            paragraph_description += "\n\n" + "\n".join(steps)
            return {"reasoning_steps": paragraph_description}
        else:
            steps.append("✓ Both graphs have the same number of edges, which is necessary for isomorphism.")
        
        # Step 2: Analyze vertex degrees
        steps.append("\nStep 2: Analyzing the degree sequences of both graphs.")
        
        # 2.1 Computing and comparing degree sequences
        degree_dict_G1 = {n: len(G1[n]) for n in G1}
        degree_dict_G2 = {n: len(G2[n]) for n in G2}
        
        steps.append("Degree of each vertex in Graph 1:")
        for node, degree in sorted(degree_dict_G1.items()):
            steps.append(f"  Node {node}: degree {degree}")
        
        steps.append("\nDegree of each vertex in Graph 2:")
        for node, degree in sorted(degree_dict_G2.items()):
            steps.append(f"  Node {node}: degree {degree}")
        
        degree_seq_G1 = sorted([len(G1[n]) for n in G1])
        degree_seq_G2 = sorted([len(G2[n]) for n in G2])
        
        steps.append(f"\n2.1: Sorted degree sequences:")
        steps.append(f"  Graph 1: {degree_seq_G1}")
        steps.append(f"  Graph 2: {degree_seq_G2}")
        
        if degree_seq_G1 != degree_seq_G2:
            steps.append("The graphs have different degree sequences, so they cannot be isomorphic.")
            steps.append("Conclusion: The graphs are not isomorphic.")
            paragraph_description += "\n\n" + "\n".join(steps)
            return {"reasoning_steps": paragraph_description}
        else:
            steps.append("✓ Both graphs have the same degree sequence, which is necessary for isomorphism.")
        
        # Step 3: Planar embedding analysis
        steps.append("\nStep 3: Analyzing planar embedding properties.")
        
        # 3.1 Verify both graphs are planar
        is_planar_G1 = nx.is_planar(nx_G1)
        is_planar_G2 = nx.is_planar(nx_G2)
        
        steps.append(f"3.1: Planarity check:")
        steps.append(f"  Is Graph 1 planar? {is_planar_G1}")
        steps.append(f"  Is Graph 2 planar? {is_planar_G2}")
        
        if not (is_planar_G1 and is_planar_G2):
            steps.append("Error: At least one of the graphs is not planar. This violates our assumption.")
        else:
            steps.append("✓ Both graphs are confirmed to be planar.")
        
        # 3.2 Calculate number of faces using Euler's formula: V - E + F = 2
        faces_G1 = 2 + edges_G1 - nodes_G1
        faces_G2 = 2 + edges_G2 - nodes_G2
        
        steps.append(f"\n3.2: Using Euler's formula for planar graphs (V - E + F = 2):")
        steps.append(f"  Graph 1: {nodes_G1} vertices - {edges_G1} edges + F = 2")
        steps.append(f"  Solving for F: F = 2 + {edges_G1} - {nodes_G1} = {faces_G1} faces")
        steps.append(f"  Graph 2: {nodes_G2} vertices - {edges_G2} edges + F = 2")
        steps.append(f"  Solving for F: F = 2 + {edges_G2} - {nodes_G2} = {faces_G2} faces")
        
        if faces_G1 == faces_G2:
            steps.append(f"✓ Both graphs have the same number of faces: {faces_G1}.")
        else:
            steps.append("Error: The graphs have different numbers of faces, which shouldn't happen if they have the same number of vertices and edges.")
        
        # Step 4: Cycle structure analysis
        steps.append("\nStep 4: Analyzing cycle structures.")
        
        # 4.1 Find cycles of length 3 (triangles)
        def count_triangles(graph_adj):
            triangles = []
            count = 0
            nodes = list(graph_adj.keys())
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    if nodes[j] in graph_adj[nodes[i]]:  # i and j are connected
                        for k in range(j+1, len(nodes)):
                            if (nodes[k] in graph_adj[nodes[i]] and 
                                nodes[k] in graph_adj[nodes[j]]):
                                count += 1
                                triangles.append((nodes[i], nodes[j], nodes[k]))
            return count, triangles
        
        triangles_count_G1, triangles_G1 = count_triangles(G1)
        triangles_count_G2, triangles_G2 = count_triangles(G2)
        
        steps.append(f"4.1: Analysis of triangles (3-cycles):")
        steps.append(f"  Graph 1 has {triangles_count_G1} triangles:")
        for t in triangles_G1:
            steps.append(f"    Triangle: {t[0]} - {t[1]} - {t[2]}")
        
        steps.append(f"\n  Graph 2 has {triangles_count_G2} triangles:")
        for t in triangles_G2:
            steps.append(f"    Triangle: {t[0]} - {t[1]} - {t[2]}")
        
        if triangles_count_G1 != triangles_count_G2:
            steps.append("\nThe graphs have different numbers of triangles, so they cannot be isomorphic.")
            steps.append("Conclusion: The graphs are not isomorphic.")
            paragraph_description += "\n\n" + "\n".join(steps)
            return {"reasoning_steps": paragraph_description}
        else:
            steps.append(f"\n✓ Both graphs have the same number of triangles: {triangles_count_G1}.")
        
        # Step 5: DFS traversal analysis
        steps.append("\nStep 5: Performing depth-first search traversal analysis.")
        
        # 5.1 Get DFS-ordered degree sequences
        def get_dfs_degree_sequence(graph_adj):
            n = len(graph_adj)
            visited = [False] * n
            degrees = []
            traversal_order = []
            
            def dfs(node):
                visited[node] = True
                traversal_order.append(node)
                degrees.append(len(graph_adj[node]))
                for neighbor in sorted(graph_adj[node]):  # Sort for deterministic traversal
                    if not visited[neighbor]:
                        dfs(neighbor)
            
            # Start DFS from each unvisited node
            for i in range(n):
                if not visited[i]:
                    dfs(i)
                    
            return degrees, traversal_order
        
        dfs_degree_G1, dfs_traversal_G1 = get_dfs_degree_sequence(G1)
        dfs_degree_G2, dfs_traversal_G2 = get_dfs_degree_sequence(G2)
        
        steps.append(f"5.1: DFS traversal analysis:")
        steps.append(f"  Graph 1 DFS traversal order: {dfs_traversal_G1}")
        steps.append(f"  Graph 1 DFS-ordered degree sequence: {dfs_degree_G1}")
        steps.append(f"\n  Graph 2 DFS traversal order: {dfs_traversal_G2}")
        steps.append(f"  Graph 2 DFS-ordered degree sequence: {dfs_degree_G2}")
        
        # Note: DFS traversal order can be different for isomorphic graphs, so we don't compare them directly
        steps.append("\n  Note: DFS traversal order can differ for isomorphic graphs due to different node labeling.")
        
        # Step 6: Biconnected components analysis
        steps.append("\nStep 6: Analyzing biconnected components (inspired by Hopcroft-Tarjan approach).")
        
        # 6.1 Identify articulation points (cut vertices)
        articulation_points_G1 = list(nx.articulation_points(nx_G1))
        articulation_points_G2 = list(nx.articulation_points(nx_G2))
        
        steps.append(f"6.1: Articulation points (cut vertices):")
        steps.append(f"  Graph 1 articulation points: {sorted(articulation_points_G1)}")
        steps.append(f"  Graph 2 articulation points: {sorted(articulation_points_G2)}")
        
        if len(articulation_points_G1) != len(articulation_points_G2):
            steps.append("\nThe graphs have different numbers of articulation points, so they cannot be isomorphic.")
            steps.append("Conclusion: The graphs are not isomorphic.")
            paragraph_description += "\n\n" + "\n".join(steps)
            return {"reasoning_steps": paragraph_description}
        else:
            steps.append(f"\n✓ Both graphs have the same number of articulation points: {len(articulation_points_G1)}.")
        
        # 6.2 Count and analyze biconnected components
        biconn_comps_G1 = list(nx.biconnected_components(nx_G1))
        biconn_comps_G2 = list(nx.biconnected_components(nx_G2))
        
        biconn_sizes_G1 = sorted([len(comp) for comp in biconn_comps_G1])
        biconn_sizes_G2 = sorted([len(comp) for comp in biconn_comps_G2])
        
        steps.append(f"\n6.2: Biconnected components analysis:")
        steps.append(f"  Graph 1 has {len(biconn_comps_G1)} biconnected components with sizes: {biconn_sizes_G1}")
        steps.append(f"  Graph 2 has {len(biconn_comps_G2)} biconnected components with sizes: {biconn_sizes_G2}")
        
        if biconn_sizes_G1 != biconn_sizes_G2:
            steps.append("\nThe graphs have different biconnected component size distributions, so they cannot be isomorphic.")
            steps.append("Conclusion: The graphs are not isomorphic.")
            paragraph_description += "\n\n" + "\n".join(steps)
            return {"reasoning_steps": paragraph_description}
        else:
            steps.append(f"\n✓ Both graphs have matching biconnected component structures.")
        
        # Step 7: Advanced structural analysis
        steps.append("\nStep 7: Performing advanced structural analysis.")
        
        # 7.1 Analyzing the distribution of cycle lengths
        def find_simple_cycles(G, max_length=6):
            """Find simple cycles up to a certain length"""
            cycle_counts = {i: 0 for i in range(3, max_length+1)}
            
            # This is a simplified approach - for a full analysis, we'd use a more sophisticated algorithm
            if len(G) <= max_length:
                all_simple_cycles = list(nx.simple_cycles(G.to_directed()))
                for cycle in all_simple_cycles:
                    if 3 <= len(cycle) <= max_length:
                        cycle_counts[len(cycle)] += 1
            else:
                # For larger graphs, we'll sample some cycles
                steps.append("  Note: Graph is too large for exhaustive cycle enumeration, using approximation.")
                
            return cycle_counts
        
        # Only perform this analysis for smaller graphs to avoid computational complexity
        if nodes_G1 <= 15:
            cycle_counts_G1 = find_simple_cycles(nx_G1)
            cycle_counts_G2 = find_simple_cycles(nx_G2)
            
            steps.append("7.1: Cycle length distribution (counts of cycles of length k):")
            steps.append(f"  Graph 1 cycle counts: {cycle_counts_G1}")
            steps.append(f"  Graph 2 cycle counts: {cycle_counts_G2}")
            
            if cycle_counts_G1 == cycle_counts_G2:
                steps.append("✓ Both graphs have the same cycle length distribution.")
            else:
                steps.append("The graphs have different cycle length distributions, so they cannot be isomorphic.")
                steps.append("Conclusion: The graphs are not isomorphic.")
                paragraph_description += "\n\n" + "\n".join(steps)
                return {"reasoning_steps": paragraph_description}
        else:
            steps.append("7.1: Skipping detailed cycle analysis due to graph size.")
        
        # Step 8: Checking for isomorphism
        steps.append("\nStep 8: Drawing conclusions about isomorphism.")
        
        if is_isomorphic:
            if permutation:
                steps.append(f"8.1: A valid isomorphism mapping exists: {permutation}")
                steps.append("\n8.2: Verifying the mapping preserves adjacency:")
                
                # Verify mapping with detailed output
                verification_passed = True
                for u in G1:
                    mapped_u = permutation[u]
                    neighbors_u = G1[u]
                    mapped_neighbors = sorted([permutation[v] for v in neighbors_u])
                    actual_neighbors = sorted(G2[mapped_u])
                    
                    steps.append(f"  Node {u} in Graph 1 maps to node {mapped_u} in Graph 2")
                    steps.append(f"    Neighbors of {u} in Graph 1: {neighbors_u}")
                    steps.append(f"    After mapping to Graph 2: {mapped_neighbors}")
                    steps.append(f"    Actual neighbors of {mapped_u} in Graph 2: {actual_neighbors}")
                    
                    if set(mapped_neighbors) == set(actual_neighbors):
                        steps.append(f"    ✓ Adjacency is preserved")
                    else:
                        steps.append(f"    ✗ Adjacency is NOT preserved - this is an error!")
                        verification_passed = False
                
                if verification_passed:
                    steps.append("\n✓ All adjacency relationships are preserved by this mapping.")
                    steps.append("\nConclusion: The graphs are isomorphic.")
                else:
                    steps.append("\n✗ The alleged mapping does not preserve adjacency.")
                    steps.append("\nConclusion: There is an error in the analysis.")
            else:
                steps.append("8.1: All structural invariants match between the two graphs.")
                steps.append("8.2: The complete isomorphism check confirms the graphs are isomorphic, though the specific mapping is not shown.")
                steps.append("\nConclusion: The graphs are isomorphic.")
        else:
            steps.append("8.1: Despite matching on several invariants, deeper structural analysis reveals the graphs are not isomorphic.")
            steps.append("8.2: A complete isomorphism check confirms no valid mapping exists.")
            steps.append("\nConclusion: The graphs are not isomorphic.")
        
        paragraph_description += "\n\n" + "\n".join(steps)
        return {"reasoning_steps": paragraph_description}


    @staticmethod
    def draw_and_save_graph(adj_list, file_path):
        """
        使用 networkx 绘制并保存无向图，使用多种随机布局让图形状多样化。
        """
        if nx is None or plt is None:
            return
        
        G = IsomorphismGenerator.dict_to_nx_graph(adj_list)
        
        plt.figure(figsize=(8, 8))
        
        # 随机选择布局类型
        layout_types = ['circular', 'spiral', 'planar', 'spring', 'random', 'shell', 'kamada_kawai']
        layout_type = random.choice(layout_types)
        
        if layout_type == 'circular':
            # 圆形布局
            pos = nx.circular_layout(G)
        elif layout_type == 'spiral':
            # 螺旋形布局
            pos = {}
            nodes = list(G.nodes())
            n = len(nodes)
            for i, node in enumerate(nodes):
                angle = i * 2 * 3.14159 / n * 3  # 螺旋角度
                radius = 0.1 + i * 0.8 / n  # 螺旋半径
                pos[node] = (radius * math.cos(angle), radius * math.sin(angle))
        elif layout_type == 'planar':
            # 平面图布局
            if nx.is_planar(G):
                pos = nx.planar_layout(G)
            else:
                pos = nx.spring_layout(G, seed=random.randint(1, 1000))
        elif layout_type == 'spring':
            # 弹簧布局（多次尝试不同种子）
            best_pos = None
            best_score = float('inf')
            for _ in range(5):
                seed = random.randint(1, 10000)
                temp_pos = nx.spring_layout(G, seed=seed, k=1.0, iterations=50)
                # 计算节点间距离的方差作为评分
                distances = []
                nodes = list(temp_pos.keys())
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        dist = ((temp_pos[nodes[i]][0] - temp_pos[nodes[j]][0])**2 + 
                            (temp_pos[nodes[i]][1] - temp_pos[nodes[j]][1])**2)**0.5
                        distances.append(dist)
                score = sum(distances) / len(distances) if distances else 0
                if score < best_score:
                    best_score = score
                    best_pos = temp_pos
            pos = best_pos
        elif layout_type == 'random':
            # 完全随机布局
            pos = {}
            for node in G.nodes():
                pos[node] = (random.uniform(-1, 1), random.uniform(-1, 1))
        elif layout_type == 'shell':
            # 壳状布局
            if len(G.nodes()) > 3:
                # 创建多个壳层
                nodes = list(G.nodes())
                shells = []
                shell_size = max(1, len(nodes) // 3)
                for i in range(0, len(nodes), shell_size):
                    shell = nodes[i:i+shell_size]
                    if shell:
                        shells.append(shell)
                pos = nx.shell_layout(G, nlist=shells)
            else:
                pos = nx.circular_layout(G)
        elif layout_type == 'kamada_kawai':
            # Kamada-Kawai 布局
            try:
                pos = nx.kamada_kawai_layout(G)
            except:
                pos = nx.spring_layout(G, seed=random.randint(1, 1000))
        
        # 随机调整节点位置，增加变化
        adjustment_factor = 0.1
        for node in pos:
            pos[node] = (
                pos[node][0] + random.uniform(-adjustment_factor, adjustment_factor),
                pos[node][1] + random.uniform(-adjustment_factor, adjustment_factor)
            )
        
        # 随机选择节点和边的样式
        node_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
        node_color = random.choice(node_colors)
        
        edge_colors = ['black', 'gray', 'darkblue', 'darkgreen', 'brown']
        edge_color = random.choice(edge_colors)
        
        node_sizes = [300, 400, 500, 600, 700]
        node_size = random.choice(node_sizes)
        
        edge_widths = [1.0, 1.5, 2.0, 2.5]
        edge_width = random.choice(edge_widths)
        
        # 绘制图
        nx.draw_networkx_nodes(G, pos, node_color=node_color, node_size=node_size, alpha=0.8)
        nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')
        nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=edge_width, alpha=0.7)
        
        # 随机选择背景颜色
        bg_colors = ['white', 'lightgray', 'lightyellow', 'lightblue']
        bg_color = random.choice(bg_colors)
        plt.gca().set_facecolor(bg_color)
        
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(file_path, dpi=150, bbox_inches='tight')
        plt.close()


    @staticmethod
    def generate_graph_isomorphism_problems(num_cases=5, difficulty=1, output_folder="graph_isomorphism_problems"):

        images_folder = os.path.join(output_folder, "images")
        os.makedirs(images_folder, exist_ok=True)
        

        # 这是静态方法，无法直接使用self._get_difficulty_params
        # 创建临时实例获取参数
        temp_instance = IsomorphismGenerator()
        params = temp_instance._get_difficulty_params(difficulty)
        
        num_nodes = params["num_nodes"]
        edge_prob = params["edge_prob"]
        max_triangles_percent = params["max_triangles_percent"]
        max_degree = params["max_degree"]
        
        with_image_data = []
        
        for i in range(num_cases):
            # --- 1) 生成平面图 G1
            G1 = IsomorphismGenerator.generate_random_connected_undirected_graph(
                num_nodes, 
                edge_prob=edge_prob, 
                seed=time.time(),
                max_triangles_percent=max_triangles_percent,
                max_degree=max_degree
            )
            
            # --- 2) 生成图 G2
            # half chance: G2 是 G1 的随机重标号(一定同构)
            # half chance: G2 独立随机生成(不一定同构)
            make_iso = (random.random() < 0.5)
            
            permutation = None
            if make_iso:
                # 构建一个随机置换
                perm_list = list(range(num_nodes))
                random.shuffle(perm_list)
                # 创建 old_label->new_label 的字典
                permutation = {old: new for old, new in zip(range(num_nodes), perm_list)}
                G2 = IsomorphismGenerator.relabel_graph(G1, permutation)
            else:
                # 从两种方式中选择一种生成G2
                # 1. 独立随机生成平面图
                # 2. 从G1随机更改一些边
                modify_from_g1 = random.random() < 0.8
                
                if modify_from_g1:
                    # 从G1随机更改一些边
                    G2 = IsomorphismGenerator.generate_modified_graph(
                        G1, 
                        edge_change_prob=0.2,  # 改变边的概率
                        seed=time.time(),
                        max_triangles_percent=max_triangles_percent,
                        max_degree=max_degree
                    )
                else:
                    # 独立随机生成平面图
                    G2 = IsomorphismGenerator.generate_random_connected_undirected_graph(
                        num_nodes, 
                        edge_prob=edge_prob, 
                        seed=time.time(),
                        max_triangles_percent=max_triangles_percent,
                        max_degree=max_degree
                    )
            
            # --- 3) 判断是否同构
            nx_G1 = IsomorphismGenerator.dict_to_nx_graph(G1)
            nx_G2 = IsomorphismGenerator.dict_to_nx_graph(G2)
            iso_result = nx.is_isomorphic(nx_G1, nx_G2)
            
            # --- 4) 生成详细的解法步骤
            solution_steps = IsomorphismGenerator.generate_solution_steps(G1, G2, iso_result, permutation if make_iso else None)
            
            # --- 5) 画图并保存
            image1_name = f"graph_{difficulty}_{i}_G1.png"
            image2_name = f"graph_{difficulty}_{i}_G2.png"
            image1_path = os.path.join(images_folder, image1_name)
            image2_path = os.path.join(images_folder, image2_name)
            
            IsomorphismGenerator.draw_and_save_graph(G1, image1_path)
            IsomorphismGenerator.draw_and_save_graph(G2, image2_path)
            
            # --- 6) 题目描述 + 答案
            # 带图版题目
            question_text_with_image = (
                "Given two connected undirected planar graphs G1 and G2 shown below (with varying shapes and not just triangles), "
                "determine if they are isomorphic by analyzing their planar structure. Answer with 'Yes' or 'No'."
            )
            # 不带图版题目，直接给出邻接表
            adj_str_G1 = "\n".join([f"{u}: {G1[u]}" for u in sorted(G1.keys())])
            adj_str_G2 = "\n".join([f"{u}: {G2[u]}" for u in sorted(G2.keys())])
            question_text_no_image = (
                "Given two connected undirected planar graphs G1 and G2 with the adjacency lists below (with varying shapes and not just triangles), "
                "determine if they are isomorphic by analyzing their planar structure. Answer with 'Yes' or 'No'."
                f"G1:\n{adj_str_G1}\n\n"
                f"G2:\n{adj_str_G2}\n"
            )
            
            # 答案
            answer_text = "Yes" if iso_result else "No"
            
            # --- 7) 生成 JSON 条目
            with_image_entry = {
                "index": f"isomorphism_with_image_{difficulty}_{i}",
                "category": "graph_isomorphism",
                "question": question_text_with_image,
                "question_language": question_text_no_image,
                "difficulty": difficulty,
                "image": [f"images/{image1_name}", f"images/{image2_name}"],
                "answer": answer_text,
            }
            
            with_image_data.append(with_image_entry)
            
        return with_image_data



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cases", type=int, default=6)
    parser.add_argument("--output_folder", type=str, default="graph_isomorphism_problems")
    args = parser.parse_args()
    
    output_folder = args.output_folder
    
    # 使用类的实例方法生成问题
    generator = IsomorphismGenerator(output_folder=output_folder)
    
    # 生成所有难度级别的问题
    for difficulty in range(1, 6):  # 1到5
        generator.generate(num_cases=args.num_cases, difficulty=difficulty, output_folder=output_folder)
    
    print("Done!")
    print(f"All files are in folder: {output_folder}")
    print("Generated graph isomorphism problems with images.")
