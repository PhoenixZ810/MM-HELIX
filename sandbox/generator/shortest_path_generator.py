import argparse
import os
import json
import random
import heapq
import time
from .base_generator import BaseGenerator

try:
    import networkx as nx
    import matplotlib.pyplot as plt
except ImportError:
    print("请安装 'networkx' 和 'matplotlib' 以绘制和保存图像。")
    nx = None
    plt = None


class ShortestPathGenerator(BaseGenerator):
    """
    最短路径问题生成器，用于生成带权无向图的最短路径问题。
    """
    
    @staticmethod
    def generate_random_connected_weighted_graph(num_nodes, edge_prob=0.3, min_weight=1, max_weight=10, seed=None):
        """
        基于网格结构生成连通的加权无向图，完全避免边的视觉交叉。
        策略：只允许网格中真正相邻的节点连接，确保所有边都不穿过其他节点。
        :param num_nodes: 节点数量
        :param edge_prob: 在相邻节点间添加边的概率
        :param min_weight: 最小边权重
        :param max_weight: 最大边权重
        :param seed: 随机种子（可选）
        :return: adjacency_list (dict), 例如 {0: [(1,5), (2,3)], 1:[(0,5)], 2:[(0,3)], ...}
                其中每个元组(neighbor, weight)表示邻居节点和边权重
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
        
        def add_weighted_edge(u, v, weight):
            """添加加权无向边"""
            if not any(neighbor == v for neighbor, _ in adjacency_list[u]) and u != v:
                adjacency_list[u].append((v, weight))
                adjacency_list[v].append((u, weight))
        
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
                    weight = random.randint(min_weight, max_weight)
                    add_weighted_edge(node, neighbor_node, weight)
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
                        weight = random.randint(min_weight, max_weight)
                        add_weighted_edge(node, neighbor_node, weight)
                        dfs_connect(node)
                        break
        
        # 根据概率添加额外的相邻连接来增加图的复杂度
        for node, neighbor_node, weight_type in possible_edges:
            if not any(neighbor == neighbor_node for neighbor, _ in adjacency_list[node]):
                # 根据连接类型调整概率
                if weight_type == 3:  # 水平/垂直相邻
                    connect_prob = edge_prob * 10.0
                elif weight_type == 2:  # 对角线相邻
                    connect_prob = edge_prob
                else:
                    connect_prob = edge_prob * 0.3
                
                if random.random() < connect_prob:
                    weight = random.randint(min_weight, max_weight)
                    add_weighted_edge(node, neighbor_node, weight)
        
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
                    if any(neighbor == other_node for neighbor, _ in adjacency_list[node]):
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
                                weight = random.randint(min_weight, max_weight)
                                add_weighted_edge(node, other_node, weight)
                    else:
                        # 非重合对角线连接：不同行且不同列且不是标准对角线
                        if (node_row != other_row and node_col != other_col and 
                            row_diff != col_diff):
                            
                            # 对角链接不衰减，使用固定概率
                            connect_prob = edge_prob * 2
                            if random.random() < connect_prob:
                                weight = random.randint(min_weight, max_weight)
                                # add_weighted_edge(node, other_node, weight)
        
        # 应用扩展连接
        add_extended_connections()
        
        # 排序邻居列表，方便美观输出
        for k in adjacency_list:
            adjacency_list[k].sort()
        
        return adjacency_list
    
    @staticmethod
    def dijkstra_shortest_distance(adj_list, start, goal):
        """
        使用 Dijkstra 算法在加权无向图上计算从 start 到 goal 的最短距离。
        如果找不到则返回 None。
        """
        # 初始化距离字典，所有节点距离为无穷大，起点为0
        distance = {node: float('inf') for node in adj_list}
        distance[start] = 0
        
        # 优先队列：(距离, 节点)
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node in visited:
                continue
                
            visited.add(current_node)
            
            if current_node == goal:
                return distance[goal]
            
            # 检查所有邻居
            for neighbor, weight in adj_list[current_node]:
                if neighbor not in visited:
                    new_dist = current_dist + weight
                    if new_dist < distance[neighbor]:
                        distance[neighbor] = new_dist
                        heapq.heappush(pq, (new_dist, neighbor))
        
        return None  # 对于连通图不会发生
    
    @staticmethod
    def generate_solution_steps(adj_list, start, goal):
        """
        使用Dijkstra算法寻找从start到goal的最短路径，并生成详细的解法步骤。
        
        Args:
            adj_list: 图的邻接表 (加权)
            start: 起始节点
            goal: 目标节点
            
        Returns:
            dict: 包含解法步骤的字典
        """
        # 初始化距离字典和前驱字典
        distance = {node: float('inf') for node in adj_list}
        distance[start] = 0
        predecessor = {start: None}
        
        # 优先队列：(距离, 节点)
        pq = [(0, start)]
        visited = set()
        
        # 创建一个段落式的解法描述
        adj_str = ", ".join([f"{u}: {[(neighbor, weight) for neighbor, weight in adj_list[u]]}" for u in sorted(adj_list.keys())])
        paragraph_description = f"To find the shortest path from node {start} to node {goal} in this weighted undirected graph with adjacency list {{{adj_str}}}, I'll use Dijkstra's algorithm. "
        
        # 步骤列表    
        steps = []
        
        # 初始化步骤
        steps.append(f"Step 1: Initialize Dijkstra's algorithm with starting node {start}. Set distance[{start}] = 0, all other distances to infinity. Priority queue = [(0, {start})].")
        
        step_counter = 2
        found = False
        
        # 执行Dijkstra算法
        while pq and not found:
            current_dist, current_node = heapq.heappop(pq)
            
            # 如果节点已经访问过，跳过
            if current_node in visited:
                continue
                
            visited.add(current_node)
            
            # 检查是否为目标节点
            if current_node == goal:
                found = True
                steps.append(f"Step {step_counter}: Pop node {current_node} with distance {current_dist} from priority queue. This is our target node {goal}. We have found the shortest path with distance {current_dist}.")
                break
            
            # 更新邻居距离
            updates = []
            for neighbor, weight in adj_list[current_node]:
                if neighbor not in visited:
                    new_dist = current_dist + weight
                    if new_dist < distance[neighbor]:
                        old_dist = distance[neighbor] if distance[neighbor] != float('inf') else "infinity"
                        distance[neighbor] = new_dist
                        predecessor[neighbor] = current_node
                        heapq.heappush(pq, (new_dist, neighbor))
                        updates.append(f"distance[{neighbor}] from {old_dist} to {new_dist}")
            
            if updates:
                update_desc = ", ".join(updates)
                steps.append(f"Step {step_counter}: Pop node {current_node} with distance {current_dist}. Update neighbors: {update_desc}. Priority queue updated accordingly.")
            else:
                steps.append(f"Step {step_counter}: Pop node {current_node} with distance {current_dist}. No distance updates needed for its neighbors.")
            
            step_counter += 1
        
        # 如果没有找到路径（对于连通图不会发生）
        if not found:
            steps.append(f"Step {step_counter}: The priority queue is empty and we didn't find a path to node {goal}. This shouldn't happen in a connected graph.")
            return {"steps": steps, "shortest_distance": None, "path": []}
        
        # 重建最短路径
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = predecessor.get(current)
        path.reverse()
        
        # 添加最后一步：总结
        steps.append(f"Step {step_counter}: Reconstruct the shortest path from {start} to {goal}: {' -> '.join([str(p) for p in path])}. The shortest distance is {distance[goal]}.")
        
        paragraph_description += '\n'.join(steps)
        
        return {
            "shortest_distance": distance[goal],
            "path": path,
            "reasoning_steps": paragraph_description
        }
    
    @staticmethod
    def draw_and_save_graph(adj_list, file_path):
        """
        使用 networkx 绘制并保存加权无向图，使用与生成时相同的网格布局。需要安装 networkx 和 matplotlib。
        """
        if nx is None or plt is None:
            return
        
        G = nx.Graph()
        # 添加节点、边（带权重）
        for u in adj_list:
            G.add_node(u)
            for v, weight in adj_list[u]:
                if v > u:  # 避免重复添加 (u,v) 与 (v,u)
                    G.add_edge(u, v, weight=weight)
        
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
        fig_width = max(10, max_x + 4)
        fig_height = max(8, max_y + 3)
        
        # 绘制图形
        plt.figure(figsize=(fig_width, fig_height))
        
        # 为每个节点分配颜色，同一节点的边使用相同颜色
        import matplotlib.cm as cm
        import numpy as np
        
        edges = list(G.edges())
        nodes = sorted(G.nodes())
        num_nodes = len(nodes)
        
        # 实现边着色算法，确保相邻边和交叉边颜色不同
        def edge_coloring(graph_edges, all_nodes, node_positions):
            """
            边着色算法：确保任何相邻的边（共享同一顶点的边）和交叉的边都有不同颜色
            使用贪心算法为每条边分配颜色
            """
            
            def segments_intersect(p1, q1, p2, q2):
                """
                检查两条线段是否相交
                p1, q1: 第一条线段的两个端点
                p2, q2: 第二条线段的两个端点
                返回 True 如果线段相交，False 否则
                """
                def on_segment(p, q, r):
                    """检查点q是否在线段pr上"""
                    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
                
                def orientation(p, q, r):
                    """找到三点的方向"""
                    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
                    if val == 0:
                        return 0  # 共线
                    return 1 if val > 0 else 2  # 顺时针或逆时针
                
                o1 = orientation(p1, q1, p2)
                o2 = orientation(p1, q1, q2)
                o3 = orientation(p2, q2, p1)
                o4 = orientation(p2, q2, q1)
                
                # 一般情况
                if o1 != o2 and o3 != o4:
                    return True
                
                # 特殊情况：三点共线且点在线段上
                if (o1 == 0 and on_segment(p1, p2, q1)) or \
                   (o2 == 0 and on_segment(p1, q2, q1)) or \
                   (o3 == 0 and on_segment(p2, p1, q2)) or \
                   (o4 == 0 and on_segment(p2, q1, q2)):
                    return True
                
                return False
            
            # 构建约束关系：包括相邻边和交叉边
            edge_to_conflicting = {}
            for i, (u1, v1) in enumerate(graph_edges):
                conflicting_edges = []
                for j, (u2, v2) in enumerate(graph_edges):
                    if i != j:
                        # 情况1：两条边相邻（共享一个顶点）
                        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                            conflicting_edges.append(j)
                        else:
                            # 情况2：两条边在几何上相交
                            p1 = pos[u1]
                            q1 = pos[v1] 
                            p2 = pos[u2]
                            q2 = pos[v2]
                            
                            if segments_intersect(p1, q1, p2, q2):
                                conflicting_edges.append(j)
                
                edge_to_conflicting[i] = conflicting_edges
            
            # 贪心着色：为每条边分配最小可用颜色
            edge_color_indices = {}  # 边索引 -> 颜色索引
            
            for i in range(len(graph_edges)):
                # 收集冲突边已使用的颜色
                used_colors = set()
                for conflicting_edge_idx in edge_to_conflicting[i]:
                    if conflicting_edge_idx in edge_color_indices:
                        used_colors.add(edge_color_indices[conflicting_edge_idx])
                
                # 找到最小未使用的颜色索引
                color_idx = 0
                while color_idx in used_colors:
                    color_idx += 1
                edge_color_indices[i] = color_idx
            
            return edge_color_indices
        
        # 执行边着色算法
        if len(edges) > 0:
            edge_color_indices = edge_coloring(edges, nodes, pos)
            max_color_idx = max(edge_color_indices.values()) if edge_color_indices else 0
            
            # 生成足够多的不同颜色
            colors_needed = max_color_idx + 1
            if colors_needed > 0:
                # 使用不同的色彩映射来获得更多区分度的颜色
                color_maps = [cm.tab10, cm.Set1, cm.Dark2, cm.Set2, cm.Pastel1, cm.Pastel2]
                all_edge_colors = []
                
                for i in range(colors_needed):
                    colormap_idx = i // 10  # 每个colormap最多使用10种颜色
                    color_idx_in_map = i % 10
                    
                    if colormap_idx < len(color_maps):
                        colormap = color_maps[colormap_idx]
                        color = colormap(color_idx_in_map / 10.0)
                    else:
                        # 如果需要更多颜色，生成随机颜色
                        np.random.seed(i)
                        color = np.random.rand(4)
                        color[3] = 1.0  # 确保不透明
                    
                    # 将颜色加深，增强对比度
                    if len(color) >= 3:
                        darker_color = [max(0.1, c * 0.7) for c in color[:3]] + [1.0]
                        all_edge_colors.append(darker_color)
                    else:
                        all_edge_colors.append(color)
                
                # 为每条边分配对应的颜色
                edge_colors = []
                for i in range(len(edges)):
                    color_idx = edge_color_indices[i]
                    edge_colors.append(all_edge_colors[color_idx])
            else:
                edge_colors = [[0.2, 0.2, 0.5, 1.0]]
        else:
            edge_colors = [[0.2, 0.2, 0.5, 1.0]]
        
        # 绘制边（使用更深的颜色）
        nx.draw_networkx_edges(G, pos, width=3, alpha=0.9, edge_color=edge_colors, 
                              edgelist=edges)
        
        # 绘制节点（使用醒目的样式）
        nx.draw_networkx_nodes(G, pos, node_size=1200, node_color='lightblue', 
                              edgecolors='darkblue', linewidths=2)
        
        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, font_size=18, font_weight='bold', font_color='black')
        
        # 绘制边权重标签（智能避免重叠并使用对应颜色）
        edge_labels = nx.get_edge_attributes(G, 'weight')
        
        # 计算标签位置，避免重叠
        edge_label_pos = {}
        label_positions = []
        # 创建边到颜色的映射（基于边着色算法结果）
        edge_color_map = {}
        for i, (u, v) in enumerate(edges):
            edge_key = tuple(sorted([u, v]))
            edge_color_map[edge_key] = edge_colors[i]
        
        for (u, v) in edge_labels:
            # 计算边的中点
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            # 计算边的方向向量
            dx, dy = x2 - x1, y2 - y1
            length = (dx**2 + dy**2)**0.5
            
            if length > 0:
                # 标准化方向向量
                dx_norm, dy_norm = dx / length, dy / length
                
                # 计算垂直于边的偏移向量
                offset_x, offset_y = -dy_norm * 0.2, dx_norm * 0.2
                
                # 初始标签位置
                label_x, label_y = mid_x + offset_x, mid_y + offset_y
                
                # 检查是否与现有标签位置重叠
                min_distance = 0.4  # 最小标签间距
                attempts = 0
                max_attempts = 12
                
                while attempts < max_attempts:
                    overlapping = False
                    for existing_pos in label_positions:
                        dist = ((label_x - existing_pos[0])**2 + (label_y - existing_pos[1])**2)**0.5
                        if dist < min_distance:
                            overlapping = True
                            break
                    
                    if not overlapping:
                        break
                    
                    # 尝试不同的偏移位置
                    attempts += 1
                    angle = attempts * 30  # 每次旋转30度
                    import math
                    cos_a, sin_a = math.cos(math.radians(angle)), math.sin(math.radians(angle))
                    offset_distance = 0.2 + attempts * 0.08
                    label_x = mid_x + offset_distance * cos_a
                    label_y = mid_y + offset_distance * sin_a
                
                edge_label_pos[(u, v)] = (label_x, label_y)
                label_positions.append((label_x, label_y))
            else:
                edge_label_pos[(u, v)] = (mid_x, mid_y)
                label_positions.append((mid_x, mid_y))
        
        # 绘制标签（使用与边完全一致的颜色）
        for (u, v), weight in edge_labels.items():
            x, y = edge_label_pos[(u, v)]
            edge_key = tuple(sorted([u, v]))
            edge_color = edge_color_map.get(edge_key, [0.5, 0.5, 0.5, 1.0])
            
            # 使用与边完全相同的颜色
            label_color = edge_color
            
            plt.text(x, y, str(weight), fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                             edgecolor=label_color, linewidth=2),
                    ha='center', va='center', zorder=10, color=label_color)
        
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
        plt.savefig(file_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
        plt.close()
    
    def __init__(self, output_folder="output/shortest_path"):
        """
        初始化最短路径生成器。
        
        Args:
            output_folder: 输出文件夹路径
        """
        super().__init__(output_folder)
    
    def _get_difficulty_params(self, difficulty):
        """
        根据难度级别获取相应的参数配置。
        
        Args:
            difficulty: 难度级别（1-5）
            
        Returns:
            dict: 包含难度参数的字典
        """
        params = {}
        
        if difficulty == 1:
            params["num_nodes"] = 6
            params["edge_prob"] = 0.2
            params["min_weight"] = 1
            params["max_weight"] = 5
            params["min_dis"] = [3, 7, 4, 5, 6, 8, 4, 5, 6, 9]
            params["min_path_nodes"] = 2
            params["max_path_nodes"] = 5
        elif difficulty == 2:
            params["num_nodes"] = 9
            params["edge_prob"] = 0.2
            params["min_weight"] = 1
            params["max_weight"] = 5
            params["min_dis"] = [3, 7, 4, 5, 6, 8, 4, 5, 6, 9]
            params["min_path_nodes"] = 3
            params["max_path_nodes"] = 6
        elif difficulty == 3:
            params["num_nodes"] = 12
            params["edge_prob"] = 0.15
            params["min_weight"] = 1
            params["max_weight"] = 8
            params["min_dis"] = [8, 12, 10, 9, 11, 15, 8, 9, 10, 13]
            params["min_path_nodes"] = 4
            params["max_path_nodes"] = 7
        elif difficulty == 4:
            params["num_nodes"] = 16
            params["edge_prob"] = 0.15
            params["min_weight"] = 1
            params["max_weight"] = 10
            params["min_dis"] = [12, 15, 13, 14, 16, 18, 15, 17, 19, 16, 18, 20]
            params["min_path_nodes"] = 5
            params["max_path_nodes"] = 10
        elif difficulty == 5:
            params["num_nodes"] = 25
            params["edge_prob"] = 0.1
            params["min_weight"] = 1
            params["max_weight"] = 10
            params["min_dis"] = [15, 20, 18, 16, 19, 25, 15, 22, 20, 17]
            params["min_path_nodes"] = 6
            params["max_path_nodes"] = 15
        
        return params
    
    def generate(self, num_cases=10, difficulty=1, output_folder=None):
        """
        实现BaseGenerator的generate方法，生成特定难度级别的最短路径问题。
        
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
        
        # 获取难度参数
        params = self._get_difficulty_params(difficulty)
        num_nodes = params["num_nodes"]
        edge_prob = params["edge_prob"]
        min_weight = params["min_weight"]
        max_weight = params["max_weight"]
        min_dis = params["min_dis"]
        

        min_path_nodes = params["min_path_nodes"]
        max_path_nodes = params["max_path_nodes"]
        
        with_image_data = []
        
        for i in range(num_cases):
            # 设置当前问题的种子
            current_seed = time.time()
            
            # 1) 生成连通的加权无向图
            adjacency_list = self.generate_random_connected_weighted_graph(
                num_nodes, edge_prob, min_weight, max_weight, seed=current_seed)
            
            # 2) 随机选两个不同节点 s, t
            s = 0
            t = num_nodes - 1
            # s, t = random.sample(range(num_nodes), 2)
            
            # 3) 求最短路长度和路径
            dist = self.dijkstra_shortest_distance(adjacency_list, s, t)
            solution_steps = self.generate_solution_steps(adjacency_list, s, t)
            path_length = len(solution_steps["path"])
            
            # 检查距离和路径节点数目是否满足要求
            attempts = 0
            max_attempts = 100
            while (dist < min_dis[i % len(min_dis)] or 
                  path_length < min_path_nodes or 
                  path_length > max_path_nodes) and attempts < max_attempts:
                current_seed = time.time()
                adjacency_list = self.generate_random_connected_weighted_graph(
                    num_nodes, edge_prob, min_weight, max_weight, seed=current_seed)
                s, t = random.sample(range(num_nodes), 2)
                dist = self.dijkstra_shortest_distance(adjacency_list, s, t)
                solution_steps = self.generate_solution_steps(adjacency_list, s, t)
                path_length = len(solution_steps["path"])
                attempts += 1
            
            # 4) 最终的解法步骤已经在上面生成了
            
            # 5) 生成图像并保存
            image_filename = f"graph_{difficulty}_{i}.png"
            image_path = os.path.join(self.image_dir, image_filename)
            self.draw_and_save_graph(adjacency_list, image_path)
            
            # 6) 生成题目和答案
            #   带图片版
            question_text_with_image = (
                f"Given a weighted undirected graph below, what is the shortest distance "
                f"from node {s} to node {t}? Answer with a number (can be integer or decimal)."
            )
            
            #   不带图片版
            adj_str = "\n".join([f"{u}: {[(neighbor, weight) for neighbor, weight in adjacency_list[u]]}" for u in sorted(adjacency_list.keys())])
            question_text_no_image = (
                f"Given a weighted undirected graph with adjacency list below (format: node: [(neighbor, weight), ...]), what is the shortest distance "
                f"from node {s} to node {t}? Answer with a number (can be integer or decimal)."
                f"{adj_str}\n"
            )
            
            # dist 为 Dijkstra 算法求出的最短距离
            answer_text = str(int(dist)) if dist == int(dist) else str(dist)  # 如果是整数就显示为整数
            
            # 7) 生成带图/不带图题目信息
            with_image_entry = {
                "index": f"shortest_distance_weighted_with_image_{difficulty}_{i}",
                "category": "shortest_distance_weighted",
                "question": question_text_with_image,
                "question_language": question_text_no_image,
                "difficulty": difficulty,
                "image": f"images/{image_filename}",
                "answer": answer_text,
                # "cot": solution_steps["reasoning_steps"],
            }
            with_image_data.append(with_image_entry)
        
        
        self.save_annotations(with_image_data, self.output_folder)
        
        metrics_file = os.path.join(output_folder, "metrics.json")
        
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump({"score_function": "simple_str_match"}, f, ensure_ascii=False, indent=4)

    
    def generate_all_difficulties(self, num_cases=6, seed=None, min_path_nodes=None, max_path_nodes=None):
        """
        生成所有难度级别的问题集。
        
        Args:
            num_cases: 每个难度级别要生成的问题数量
            seed: 随机种子，默认为None
            min_path_nodes: 最短路径中的最少节点数
            max_path_nodes: 最短路径中的最多节点数
            
        Returns:
            所有生成的问题列表
        """
        problems = []
        
        # 如果提供了种子，为每个难度级别设置不同的种子
        if seed is None:
            seed = time.time()
        
        for difficulty in [1, 2, 3, 4, 5]:
            # 为每个难度使用稍微不同的种子，以确保多样性
            difficulty_seed = seed + difficulty
            problems.extend(self.generate(
                num_cases=num_cases, 
                difficulty=difficulty,
                seed=difficulty_seed,
                min_path_nodes=min_path_nodes,
                max_path_nodes=max_path_nodes
            ))

        self.save_annotations(problems, self.output_folder)
        
        metrics_file = os.path.join(self.output_folder, "metrics.json")
        
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump({"score_function": "simple_str_match"}, f, ensure_ascii=False, indent=4)
        

def main():
    parser = argparse.ArgumentParser(description="生成带权无向图最短路径问题")
    parser.add_argument("--num_cases", type=int, default=6, help="要生成的每个难度级别的问题数量")
    parser.add_argument("--output_folder", type=str, default="output/shortest_distance_problems", help="输出文件夹")
    parser.add_argument("--difficulty", type=int, default=0, help="问题难度级别（1-5），0表示生成所有难度级别")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--min_path_nodes", type=int, default=None, help="最短路径中的最少节点数（默认值根据难度级别变化）")
    parser.add_argument("--max_path_nodes", type=int, default=None, help="最短路径中的最多节点数（默认值根据难度级别变化）")
    args = parser.parse_args()
    
    # 创建最短路径生成器
    generator = ShortestPathGenerator(output_folder=args.output_folder)
    
    # 生成问题
    if args.difficulty == 0:
        # 生成所有难度级别的问题
        generator.generate_all_difficulties(
            num_cases=args.num_cases, 
            seed=args.seed,
            min_path_nodes=args.min_path_nodes,
            max_path_nodes=args.max_path_nodes
        )
    else:
        # 生成特定难度级别的问题
        generator.generate(
            num_cases=args.num_cases, 
            difficulty=args.difficulty, 
            seed=args.seed,
            min_path_nodes=args.min_path_nodes,
            max_path_nodes=args.max_path_nodes
        )
    
    # 问题已经在生成过程中保存了
    print(f"完成! 文件已保存到文件夹: {args.output_folder}")
    print("带权最短路径问题已生成。")


if __name__ == "__main__":
    main()