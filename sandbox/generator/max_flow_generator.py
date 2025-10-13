import argparse
import os
import json
import random
import time
import numpy as np
from .base_generator import BaseGenerator

try:
    import networkx as nx
    import matplotlib.pyplot as plt
except ImportError:
    print("Please install 'networkx' and 'matplotlib' to run max flow and draw images.")
    nx = None
    plt = None


class MaxFlowGenerator(BaseGenerator):

    def __init__(self, output_folder="output/max_flow"):
        super().__init__(output_folder)
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

    @staticmethod
    def _balance_node_capacities(adj_list, layer_node_ids, capacity_range):
        """
        重新计算边的容量，使每个节点的入边容量和出边容量大致平衡
        
        Args:
            adj_list: 邻接表（将被修改）
            layer_node_ids: 每层的节点ID列表
            capacity_range: 容量范围 (min, max)
        """
        min_cap, max_cap = capacity_range
        
        # 计算每个节点的入边数和出边数
        in_degree = {}
        out_degree = {}
        
        for u in adj_list:
            out_degree[u] = len(adj_list[u])
            for v, _ in adj_list[u]:
                in_degree[v] = in_degree.get(v, 0) + 1
        
        # 为每个节点设置默认入度和出度为0
        for layer_nodes in layer_node_ids:
            for node in layer_nodes:
                in_degree.setdefault(node, 0)
                out_degree.setdefault(node, 0)
        
        # 为每个节点分配一个目标通量
        node_target_flow = {}
        for layer_nodes in layer_node_ids:
            for node in layer_nodes:
                # 根据节点的度数来调整目标通量
                total_degree = in_degree[node] + out_degree[node]
                if total_degree > 0:
                    base_flow = random.randint(min_cap * 2, max_cap * 2)
                    # 度数更高的节点应该有更大的通量
                    degree_multiplier = 1 + (total_degree - 1) * 0.5
                    node_target_flow[node] = int(base_flow * degree_multiplier)
                else:
                    node_target_flow[node] = random.randint(min_cap, max_cap)
        
        # 第一步：初始化所有边的容量
        for u in adj_list:
            if out_degree[u] > 0:
                # 计算该节点每条出边的平均容量
                avg_out_cap = node_target_flow[u] // out_degree[u]
                remainder = node_target_flow[u] % out_degree[u]
                
                # 为每条出边分配容量
                for i, (v, _) in enumerate(adj_list[u]):
                    base_cap = max(min_cap, avg_out_cap)
                    extra = 1 if i < remainder else 0
                    final_cap = max(min_cap, min(max_cap, base_cap + extra))
                    adj_list[u][i] = (v, final_cap)
        
        # 第二步：调整中间节点，使入边容量总和与出边容量总和接近
        for layer_idx in range(1, len(layer_node_ids) - 1):  # 跳过源层和汇层
            for node in layer_node_ids[layer_idx]:
                if in_degree[node] > 0 and out_degree[node] > 0:
                    # 计算当前入边容量总和
                    in_cap_sum = 0
                    for u in adj_list:
                        for v, cap in adj_list[u]:
                            if v == node:
                                in_cap_sum += cap
                    
                    # 调整出边容量，使其接近入边容量总和
                    if out_degree[node] > 0:
                        # 目标出边容量总和：直接使用入边容量总和，加上一些随机性
                        target_out_cap = in_cap_sum + random.randint(-max_cap//2, max_cap//2)
                        # 确保目标容量在合理范围内
                        target_out_cap = max(min_cap * out_degree[node], 
                                        min(max_cap * out_degree[node], target_out_cap))
                        
                        avg_out_cap = target_out_cap // out_degree[node]
                        remainder = target_out_cap % out_degree[node]
                        
                        for i, (v, _) in enumerate(adj_list[node]):
                            base_cap = max(min_cap, avg_out_cap)
                            extra = 1 if i < remainder else 0
                            final_cap = max(min_cap, min(max_cap, base_cap + extra))
                            adj_list[node][i] = (v, final_cap)

    @staticmethod
    def generate_layered_dag(num_layers=4, nodes_per_layer=(1, 2, 2, 1),
                            capacity_range=(1, 10), seed=None):
        """
        生成一个分层的有向无环图，确保没有边交叉的布局，并为每条边设置随机容量。
        
        :param num_layers: 总层数
        :param nodes_per_layer: 各层节点数量列表，如 (1,2,2,1) 表示有4层，第1层1个节点、第2层2个节点...
                                - 要求 len(nodes_per_layer) == num_layers
                                - 第一层为源点所在层，最后一层为汇点所在层
        :param capacity_range: (cap_min, cap_max)，边容量范围
        :param seed: 随机种子，用于可重复生成
        :return: 
        - adj_list: dict, 形如 {node_id: [(to_id, capacity), ...]}
        - pos: dict, 每个节点的坐标 {node_id: (x, y)}，用于画图时避免边交叉
        - source: 源点节点的编号
        - sink: 汇点节点的编号
        """
        seed=time.time()
        
        if len(nodes_per_layer) != num_layers:
            raise ValueError("len(nodes_per_layer) must match num_layers")
        
        # 依次为每层的节点编号，记录每个节点的 (layer, index_in_layer)
        # node_id 从0开始
        node_id = 0
        layer_node_ids = []
        for layer_idx in range(num_layers):
            layer_count = nodes_per_layer[layer_idx]
            ids_in_this_layer = list(range(node_id, node_id + layer_count))
            layer_node_ids.append(ids_in_this_layer)
            node_id += layer_count
        
        total_nodes = node_id
        
        # 为了画图无交叉：把第 i 层的节点在 x=i 这一竖线上均分排布
        # y 坐标可以让同层的节点上下错开
        pos = {}
        for i, layer_ids in enumerate(layer_node_ids):
            count = len(layer_ids)
            # 如果某层只有一个节点，就放在 y=0
            # 如果有多个，就在 [-count/2, ..., count/2] 之间均匀排
            start_y = - (count - 1) / 2.0
            for idx_in_layer, node in enumerate(layer_ids):
                y = start_y + idx_in_layer
                pos[node] = (i, y)  # x = i, y = ...
        
        # 构造邻接表
        adj_list = {n: [] for n in range(total_nodes)}
        
        # 在层与层之间连边：只从 layer i -> layer i+1
        # 保证每层节点与下一层完全连通：每个节点都连接到下一层，下一层每个节点都被连接到
        for i in range(num_layers - 1):
            from_layer = layer_node_ids[i]
            to_layer = layer_node_ids[i+1]
            
            # 第一步：确保下一层的每个节点都至少被连接到一次
            # 随机打乱顺序，避免总是连接到同一个节点
            shuffled_from_layer = from_layer.copy()
            random.shuffle(shuffled_from_layer)
            
            for idx, v in enumerate(to_layer):
                # 为每个下层节点至少分配一个上层节点连接
                u = shuffled_from_layer[idx % len(shuffled_from_layer)]
                # 暂时用占位符容量，稍后会重新计算
                adj_list[u].append((v, 1))
            
            # 第二步：确保每个上层节点都至少连接到一个下层节点
            for u in from_layer:
                # 检查该节点是否已经有连接到下层的边
                has_connection = any(v in to_layer for v, _ in adj_list[u])
                if not has_connection:
                    # 如果没有连接，随机选择一个下层节点连接
                    v = random.choice(to_layer)
                    # 暂时用占位符容量，稍后会重新计算
                    adj_list[u].append((v, 1))
            
            # 第三步：添加额外的随机边以增加连通性
            for u in from_layer:
                for v in to_layer:
                    # 检查是否已经存在边
                    if not any(target == v for target, _ in adj_list[u]):
                        # 以一定概率添加额外边
                        if random.random() < 0.3:  # 可以调整这个概率
                            # 暂时用占位符容量，稍后会重新计算
                            adj_list[u].append((v, 1))
        
        # 重新计算容量，使每个节点的入边容量和出边容量大致平衡
        # 由于这是静态方法，我们需要使用类名来调用其他静态方法
        MaxFlowGenerator._balance_node_capacities(adj_list, layer_node_ids, capacity_range)
        
        # 排序出边，让输出更整齐
        for n in adj_list:
            adj_list[n].sort(key=lambda x: x[0])
        
        source = layer_node_ids[0][0]
        sink = layer_node_ids[-1][-1]
        return adj_list, pos, source, sink


    @staticmethod
    def build_networkx_digraph(adj_list):
        """
        将带容量的邻接表转换为 networkx.DiGraph 以便求最大流。
        """
        G = nx.DiGraph()
        for u in adj_list:
            G.add_node(u)
            for v, cap in adj_list[u]:
                G.add_edge(u, v, capacity=cap)
        return G

    @staticmethod
    def solve_max_flow(adj_list, source, sink):
        """
        使用 networkx 的最大流算法求解，从 source 到 sink 的最大流值、以及 flow dict。
        """
        G = MaxFlowGenerator.build_networkx_digraph(adj_list)
        flow_value, flow_dict = nx.maximum_flow(G, source, sink)
        return flow_value, flow_dict
    

    def generate_solution_steps(self,adj_list, source, sink):
        """
        生成最大流问题的详细解决步骤，包括Ford-Fulkerson算法的执行过程。
        
        Args:
            adj_list: 图的邻接表（带容量）
            source: 源点
            sink: 汇点
            
        Returns:
            dict: 包含解法步骤的字典
        """
        G = MaxFlowGenerator.build_networkx_digraph(adj_list)
        
        # 创建一个段落式的解法描述
        paragraph_description = f"To find the maximum flow from node {source} to node {sink}, I'll use the Ford-Fulkerson algorithm. This algorithm repeatedly finds augmenting paths and increases flow until no more augmenting paths can be found."
        
        # 初始化步骤列表
        steps = []
        steps.append(f"Step 1: Initialize the flow on all edges to 0. The initial flow value is 0. Prepare to find augmenting paths.")
        
        # 初始化流和残差网络
        flow_dict = {u: {v: 0 for v, _ in adj_list[u]} for u in adj_list}
        total_flow = 0
        step_counter = 2
        
        # 描述初始状态
        flow_description = "Current flow network state:\n"
        for u in sorted(adj_list.keys()):
            for v, cap in sorted(adj_list[u]):
                flow_description += f"Flow from {u} to {v}: 0/{cap}\n"
        steps.append(f"Step {step_counter-1}.1: {flow_description}")
        step_counter += 1
        
        # 手动实现Ford-Fulkerson算法以跟踪步骤
        iteration = 1
        while True:
            # 构建残差网络
            residual_network = {u: [] for u in adj_list}
            for u in adj_list:
                for v, cap in adj_list[u]:
                    # 正向边：剩余容量 = 原容量 - 已使用流量
                    residual_cap = cap - flow_dict[u].get(v, 0)
                    if residual_cap > 0:
                        residual_network[u].append((v, residual_cap))
                    
                    # 反向边：容量等于已使用的流量
                    back_flow = flow_dict[u].get(v, 0)
                    if back_flow > 0:
                        if v not in residual_network:
                            residual_network[v] = []
                        residual_network[v].append((u, back_flow))
            
            # 描述残差网络
            residual_description = f"Iteration {iteration}: Residual Network State:\n"
            for u in sorted(residual_network.keys()):
                if residual_network[u]:
                    edges = []
                    for v, cap in sorted(residual_network[u]):
                        edges.append(f"{v}(cap={cap})")
                    residual_description += f"Node {u} can reach: {', '.join(edges)}\n"
            steps.append(f"Step {step_counter-1}.2: {residual_description}")
            
            # 使用BFS在残差网络中寻找增广路径
            path, bottleneck = MaxFlowGenerator.find_augmenting_path(residual_network, source, sink)
            
            if not path:
                steps.append(f"Step {step_counter}: No more augmenting paths can be found. The algorithm terminates. The maximum flow is {total_flow}.")
                break
            
            # 记录找到的增广路径
            path_str = " -> ".join(str(node) for node in path)
            steps.append(f"Step {step_counter}: Found an augmenting path: {path_str}, with bottleneck capacity = {bottleneck}.")
            
            # 更新流
            update_description = f"Updating flows along the path {path_str}:\n"
            for i in range(len(path)-1):
                u, v = path[i], path[i+1]
                
                # 检查是正向边还是反向边
                is_forward = v in [edge[0] for edge in adj_list[u]]
                
                if is_forward:
                    # 正向边：增加流量
                    if v not in flow_dict[u]:
                        flow_dict[u][v] = 0
                    old_flow = flow_dict[u][v]
                    flow_dict[u][v] += bottleneck
                    
                    # 找出这条边的原始容量
                    original_cap = next(cap for to_v, cap in adj_list[u] if to_v == v)
                    update_description += f"  Edge {u}->{v}: flow {old_flow}/{original_cap} -> {flow_dict[u][v]}/{original_cap} (increased by {bottleneck})\n"
                else:
                    # 反向边：减少流量
                    if u not in flow_dict[v]:
                        flow_dict[v][u] = 0
                    old_flow = flow_dict[v][u]
                    flow_dict[v][u] -= bottleneck
                    
                    # 找出这条边的原始容量
                    original_cap = next(cap for to_u, cap in adj_list[v] if to_u == u)
                    update_description += f"  Edge {v}->{u}: flow {old_flow}/{original_cap} -> {flow_dict[v][u]}/{original_cap} (decreased by {bottleneck})\n"
            
            total_flow += bottleneck
            update_description += f"After this augmentation, the total flow is now {total_flow}."
            steps.append(f"Step {step_counter+1}: {update_description}")
            
            # 当前完整流网络状态
            flow_status = "Current complete flow network state:\n"
            for u in sorted(adj_list.keys()):
                for v, cap in sorted(adj_list[u]):
                    current_flow = flow_dict[u].get(v, 0)
                    flow_status += f"Flow from {u} to {v}: {current_flow}/{cap}\n"
            steps.append(f"Step {step_counter+1}.1: {flow_status}")
            
            step_counter += 2
            iteration += 1
        
        # 作为验证，使用networkx的maximum_flow算法
        flow_value, nx_flow_dict = nx.maximum_flow(G, source, sink)
        if flow_value != total_flow:
            steps.append(f"Note: The manually calculated maximum flow ({total_flow}) differs from the result of the networkx library ({flow_value}). We'll use the networkx result as the final answer.")
            total_flow = flow_value
        
        # 总结
        steps.append(f"Final Conclusion: The maximum flow from node {source} to node {sink} is {total_flow}.")
        
        paragraph_description += "\n\n" + "\n".join(steps)
        
        return {
            "max_flow": total_flow,
            "flow_dict": nx_flow_dict,
            "reasoning_steps": paragraph_description
        }

    @staticmethod
    def find_augmenting_path(residual_network, source, sink):
        """
        在残差网络中使用BFS寻找从source到sink的增广路径
        
        Returns:
            tuple: (path, bottleneck) 路径和瓶颈容量
        """
        from collections import deque
        
        # 初始化
        visited = {source: None}  # 记录前驱节点以便回溯路径
        capacities = {source: float('inf')}  # 记录到每个节点的路径上的瓶颈容量
        queue = deque([source])
        
        # BFS寻找路径
        while queue and sink not in visited:
            node = queue.popleft()
            
            for neighbor, capacity in residual_network[node]:
                if neighbor not in visited and capacity > 0:
                    visited[neighbor] = node
                    capacities[neighbor] = min(capacities[node], capacity)
                    queue.append(neighbor)
        
        # 如果找不到路径到sink
        if sink not in visited:
            return [], 0
        
        # 回溯构建路径
        path = [sink]
        node = sink
        while node != source:
            node = visited[node]
            path.append(node)
        path.reverse()
        
        return path, capacities[sink]

    @staticmethod
    def verify_max_flow(adj_list, flow_value, flow_dict, source, sink):
        """
        对求得的流进行验证：
        1. 对每条边 (u->v)，检查 0 <= flow[u][v] <= capacity(u->v)
        2. 对非源非汇的节点，流入 == 流出
        3. 在残差网络中没有增广路径（因为我们使用networkx的算法，理论上应无误，但这里演示一下）。
        如果都通过，返回 True，否则 False。
        """
        G = MaxFlowGenerator.build_networkx_digraph(adj_list)
        
        # (1) 容量约束 + 非负
        for u in flow_dict:
            for v in flow_dict[u]:
                flow_uv = flow_dict[u][v]
                if flow_uv < 0:
                    return False
                if not G.has_edge(u, v):
                    # 不在原图中的流必须是0
                    if flow_uv != 0:
                        return False
                else:
                    cap = G[u][v]['capacity']
                    if flow_uv > cap:
                        return False
        
        # (2) 流守恒
        for node in G.nodes():
            if node != source and node != sink:
                in_flow = 0
                out_flow = 0
                # 计算流入
                for pred in flow_dict:
                    if node in flow_dict[pred]:
                        in_flow += flow_dict[pred][node]
                # 计算流出
                for succ in flow_dict[node]:
                    out_flow += flow_dict[node][succ]
                if abs(in_flow - out_flow) > 1e-9:
                    return False
        
        # (3) 无增广路：等价于在残差网络中, 无从 source 到 sink 的可行路径
        # NetworkX 可以帮我们构建残差网络:
        R = nx.algorithms.flow.build_residual_network(G, 'capacity')
        # 将 flow_dict 中的流量同步到残差图
        for u in flow_dict:
            for v in flow_dict[u]:
                flow_uv = flow_dict[u][v]
                if R.has_edge(u, v):
                    R[u][v]['capacity'] -= flow_uv  # 正向边容量减少
                if R.has_edge(v, u):
                    R[v][u]['capacity'] += flow_uv  # 反向边容量增加
        
        # 在残差网络 R 中做一次DFS/BFS，看能否从 source 到达 sink
        visited = set()
        stack = [source]
        while stack:
            curr = stack.pop()
            if curr == sink:
                # 如果能到达sink，说明还有增广路
                return False
            for nbr in R[curr]:
                if R[curr][nbr]['capacity'] > 0 and nbr not in visited:
                    visited.add(nbr)
                    stack.append(nbr)
        
        return True

    @staticmethod
    def draw_dag(adj_list, pos, file_path):
        """
        按给定的 pos 布局绘制 DAG，并标注容量，保存为 PNG。
        这里我们确保每层只朝下一层连边，从左到右布局，不会产生可见交叉。
        """
        if nx is None or plt is None:
            return
        
        G = MaxFlowGenerator.build_networkx_digraph(adj_list)
        
        plt.figure(figsize=(12, 9))  # 更大的图形尺寸
        
        # 设置节点样式
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', 
                            edgecolors='black', linewidths=1.5)
        
        # 设置节点标签样式
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        # 为不同源节点的边分配不同颜色
        edge_colors = {}
        color_map = plt.cm.tab10  # 使用tab10颜色映射
        for i, u in enumerate(sorted(adj_list.keys())):
            color_idx = i % 10  # 循环使用10种颜色
            for v, _ in adj_list[u]:
                edge_colors[(u, v)] = color_map(color_idx)
        
        # 绘制边，使用颜色区分和弧形样式
        for i, (u, v, d) in enumerate(G.edges(data=True)):
            # 为每条边设置不同的弧度，避免重叠
            rad = 0.1
            if i % 2 == 0:
                rad = 0.15
            
            # 绘制单条边，使用指定颜色
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                                arrows=True, 
                                arrowstyle='-|>', 
                                arrowsize=15,
                                width=2.0,
                                edge_color=[edge_colors.get((u, v), 'black')],
                                connectionstyle=f'arc3,rad={rad}')
        
        # 标注容量，并根据边的多少动态调整标签位置
        edge_labels = {(u,v): d['capacity'] for u,v,d in G.edges(data=True)}
        
        # 根据目标节点分组边以避免位置冲突
        target_to_edges = {}
        for u, v in G.edges():
            if v not in target_to_edges:
                target_to_edges[v] = []
            target_to_edges[v].append((u, v))
        
        # 为每个目标节点的所有入边设置不同的标签位置
        for target, edges in target_to_edges.items():
            if len(edges) > 1:
                # 如果一个节点有多个入边，均匀分布标签位置
                positions = [0.3, 0.5, 0.7]  # 基本位置选项
                if len(edges) > 3:
                    # 如果入边超过3个，增加更多位置选项
                    step = 1.0 / (len(edges) + 1)
                    positions = [step * (i+1) for i in range(len(edges))]
                
                # 为每条边分配位置
                for i, edge in enumerate(edges):
                    label_pos = positions[i % len(positions)]
                    G.edges[edge]['label_pos'] = label_pos
        
        # 绘制边标签，使用分配的标签位置和强化的背景
        for (u, v), label in edge_labels.items():
            pos_u = pos[u]
            pos_v = pos[v]
            
            # 获取为该边分配的标签位置，默认为0.5
            label_pos = G.edges.get((u, v), {}).get('label_pos', 0.5)
            
            # 计算标签坐标（考虑弧形）
            rad = 0.1 if G.edges.get((u, v), {}).get('rad', 0.1) else 0.15
            # 弧形边的中点偏移
            mid_x = (pos_u[0] * (1-label_pos) + pos_v[0] * label_pos)
            mid_y = (pos_u[1] * (1-label_pos) + pos_v[1] * label_pos)
            # 根据弧度进行小偏移
            if abs(pos_v[0] - pos_u[0]) > 0.1:  # 避免除零错误
                slope = (pos_v[1] - pos_u[1]) / (pos_v[0] - pos_u[0])
                perpendicular_slope = -1/slope if slope != 0 else float('inf')
                angle = np.arctan(perpendicular_slope)
                offset = rad * 0.5  # 根据弧度大小调整偏移量
                mid_x += np.cos(angle) * offset
                mid_y += np.sin(angle) * offset
            
            # 获取边的颜色以匹配标签边框
            edge_color = edge_colors.get((u, v), 'black')
            
            # 为标签添加更显眼的背景和边框
            plt.text(mid_x, mid_y, label, 
                    fontsize=10,
                    fontweight='bold',
                    color='black', 
                    horizontalalignment='center',
                    verticalalignment='center',
                    bbox=dict(facecolor='white', 
                            edgecolor=edge_color,  # 边框颜色与边匹配
                            boxstyle='round,pad=0.5',
                            alpha=0.9,  # 高不透明度
                            linewidth=2.0))  # 加粗边框
        
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(file_path, dpi=150, bbox_inches='tight')  # 更高分辨率，并裁剪多余空白
        plt.close()

    def _get_difficulty_params(self, difficulty):
        params = {}
        if difficulty == 1:
            params["num_layers"] = 3
            params["nodes_per_layer"] = (1, 2, 1)
        elif difficulty == 2:
            params["num_layers"] = 4
            params["nodes_per_layer"] = (1, 2, 2, 1)
        elif difficulty == 3:
            params["num_layers"] = 5
            params["nodes_per_layer"] = (1, 2, 3, 2, 1)
        elif difficulty == 4:
            params["num_layers"] = 5
            params["nodes_per_layer"] = (1, 3, 3, 3, 1)
        elif difficulty == 5:
            params["num_layers"] = 6
            params["nodes_per_layer"] = (1, 3, 4, 4, 3, 1)
        return params

    def generate(self, num_cases=5, difficulty=1, seed=time.time(), output_folder="max_flow_layered_problems"):
        return self.generate_max_flow_problems(num_cases, difficulty, seed, output_folder)
    
    def generate_max_flow_problems(self, num_cases=5, difficulty=1, seed=time.time(), output_folder="max_flow_layered_problems"):

        os.makedirs(output_folder, exist_ok=True)
        out_dir = output_folder
        img_dir = os.path.join(out_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        
        with_image_data = []
        
        params = self._get_difficulty_params(difficulty)
        num_layers = params["num_layers"]
        nodes_per_layer = params["nodes_per_layer"]
        
        for i in range(num_cases):
            # 1) 生成一个分层DAG，保证不交叉
            #    举例：4层，分别 1,2,2,1 个节点 (含源点和汇点)
            #    可根据需要更改
            adj_list, pos, source, sink = MaxFlowGenerator.generate_layered_dag(
                num_layers=num_layers,
                nodes_per_layer=nodes_per_layer,
                capacity_range=(1, 10),
                seed=seed
            )
            
            # 2) 计算最大流
            flow_val, flow_dict = MaxFlowGenerator.solve_max_flow(adj_list, source, sink)
            
            # 3) 生成详细的解法步骤
            solution_steps = self.generate_solution_steps(adj_list, source, sink)
            
            # 4) 验证结果
            is_correct = MaxFlowGenerator.verify_max_flow(adj_list, flow_val, flow_dict, source, sink)
            
            # 5) 绘制并保存图
            image_name = f"dag_{difficulty}_{i}.png"
            image_path = os.path.join(img_dir, image_name)
            MaxFlowGenerator.draw_dag(adj_list, pos, image_path)
            
            # 6) 准备题目文本
            # 将邻接表转成字符串形式
            lines = []
            for u in sorted(adj_list.keys()):
                edges_str = ", ".join(f"({v},{cap})" for v,cap in adj_list[u])
                lines.append(f"{u}: [{edges_str}]")
            adj_str = "\n".join(lines)
            
            question_with_image = (
                f"Below is a layered directed acyclic graph (DAG) with capacities on each edge. "
                f"Compute the maximum flow from node {source} to node {sink}. "
                "Answer with the maximum flow value (an integer)."
            )
            question_no_image = (
                f"Below is a layered directed acyclic graph (DAG) adjacency list (with capacities). "
                f"Compute the maximum flow from node {source} to node {sink}. "
                "Answer with the maximum flow value (an integer).\n\n"
                f"{adj_str}\n"
            )
            
            answer_text = str(flow_val) if is_correct else f"(VerificationFailed) - {flow_val}"
            
            with_image_entry = {
                "index": f"max_flow_with_image_{difficulty}_{i}",
                "category": "max_flow",
                "question": question_with_image,
                "question_language": question_no_image,
                "difficulty": difficulty,
                "image": f"images/{image_name}",
                "answer": answer_text,
            }
            
            with_image_data.append(with_image_entry)

        self.save_annotations(with_image_data, output_folder)
        
        metrics_file = os.path.join(output_folder, "metrics.json")
        
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump({"score_function": "simple_str_match"}, f, ensure_ascii=False, indent=4)
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate max flow problems")
    parser.add_argument("--num_cases", type=int, default=6, help="Number of problems to generate")
    parser.add_argument("--output_folder", type=str, default="max_flow_layered_problems", help="Output folder")
    args = parser.parse_args()
    
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    problems = []

    generator = MaxFlowGenerator(output_folder=output_folder)
    
    for difficulty in [1, 2, 3, 4, 5]:
        problems.extend(generator.generate_max_flow_problems(num_cases=args.num_cases, difficulty=difficulty, output_folder=output_folder))
    
    with_image_path = os.path.join(output_folder, "annotations.json")
    
    with open(with_image_path, "w", encoding="utf-8") as fw:
        json.dump(problems, fw, ensure_ascii=False, indent=4)

    metrics_file = os.path.join(output_folder, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump({"score_function": "simple_str_match"}, f, ensure_ascii=False, indent=4)

    print(f"Done! All files saved in: {output_folder}")
    print("You can check annotations.json for the problems & answers.")
