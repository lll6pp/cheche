# -*- coding: utf-8 -*-

"""
无人车补货路径规划平台 - 后端（终极合并版）
所有功能在一个文件中，极速部署，无超时！
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import math
import random
import json
import time
from typing import List, Dict, Tuple

app = Flask(__name__, static_folder='.')
CORS(app)

# ========== 1. 工具函数 ==========
def haversine(lng1, lat1, lng2, lat2):
    R = 6371.0
    lng1_rad, lat1_rad = math.radians(lng1), math.radians(lat1)
    lng2_rad, lat2_rad = math.radians(lng2), math.radians(lat2)
    dlat, dlng = lat2_rad - lat1_rad, lng2_rad - lng1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(dlng/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# 直接使用直线距离，极速，不调用外部API
def get_drive_distance(lng1, lat1, lng2, lat2):
    return haversine(lng1, lat1, lng2, lat2)

def two_opt(route, dist_matrix):
    best_route = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                if j - i == 1:
                    continue
                new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                old_dist = dist_matrix[best_route[i-1]][best_route[i]] + dist_matrix[best_route[j-1]][best_route[j]]
                new_dist = dist_matrix[best_route[i-1]][best_route[j-1]] + dist_matrix[best_route[i]][best_route[j]]
                if new_dist < old_dist:
                    best_route = new_route
                    improved = True
                    break
            if improved:
                break
    return best_route

# ========== 2. 数据管理类 ==========
class DeliveryScenario:
    def __init__(self):
        self.nodes = []
        self.capacity = 0
        self.distance_matrix = []
        self.cost_per_km = 2.0
        self.fixed_cost = 50.0
        self.node_count = 0

    def get_total_demand(self):
        return sum(node[3] for node in self.nodes)

    def get_node(self, idx: int):
        return self.nodes[idx]

    def get_distance(self, i: int, j: int):
        return self.distance_matrix[i][j]

    def get_route_distance(self, route: List[int]):
        if len(route) < 2:
            return 0.0
        total = 0.0
        for k in range(len(route)-1):
            total += self.get_distance(route[k], route[k+1])
        return total

    def get_route_cost(self, route: List[int]):
        dist = self.get_route_distance(route)
        return dist * self.cost_per_km + self.fixed_cost

    def check_capacity(self, route: List[int]):
        total_demand = sum(self.nodes[i][3] for i in route)
        return total_demand <= self.capacity

# ========== 3. ALNS算子 ==========
class ALNSOperators:
    def __init__(self, random_seed: int = 42):
        random.seed(random_seed)

    def random_removal(self, route: List[int], q: int):
        if len(route) <= 2:
            return route.copy(), []
        q = min(q, len(route) - 2)
        removed_indices = random.sample(range(len(route)), q)
        removed_nodes = [route[i] for i in removed_indices]
        remaining_route = [route[i] for i in range(len(route)) if i not in removed_indices]
        return remaining_route, removed_nodes

    def greedy_insertion(self, scenario, route: List[int], removed_nodes: List[int]):
        route_copy = route.copy()
        nodes_to_insert = removed_nodes.copy()
        while nodes_to_insert:
            best_node = None
            best_position = -1
            best_cost = float('inf')
            for node in nodes_to_insert:
                for pos in range(len(route_copy) + 1):
                    if len(route_copy) == 0:
                        cost = 0
                    elif pos == 0:
                        cost = scenario.get_distance(node, route_copy[0])
                    elif pos == len(route_copy):
                        cost = scenario.get_distance(route_copy[-1], node)
                    else:
                        original = scenario.get_distance(route_copy[pos-1], route_copy[pos])
                        new_dist = scenario.get_distance(route_copy[pos-1], node) + scenario.get_distance(node, route_copy[pos])
                        cost = new_dist - original
                    test_route = route_copy[:pos] + [node] + route_copy[pos:]
                    if scenario.check_capacity(test_route):
                        if cost < best_cost:
                            best_cost = cost
                            best_node = node
                            best_position = pos
            if best_node is not None:
                route_copy.insert(best_position, best_node)
                nodes_to_insert.remove(best_node)
            else:
                route_copy.append(nodes_to_insert[0])
                nodes_to_insert.pop(0)
        return route_copy

    def apply_operator(self, scenario, route: List[int], operator_id: int):
        q = max(1, int(len(route) * 0.2))
        destroyed_route, removed_nodes = self.random_removal(route, q)
        return self.greedy_insertion(scenario, destroyed_route, removed_nodes)

# ========== 4. 策略网络（简化版） ==========
class PolicyNetwork:
    def __init__(self, state_dim: int = 20, action_dim: int = 12, hidden_dim: int = 64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.W1 = [[random.random() * 0.1 for _ in range(hidden_dim)] for _ in range(state_dim)]
        self.b1 = [0.0 for _ in range(hidden_dim)]
        self.W2 = [[random.random() * 0.1 for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.b2 = [0.0 for _ in range(hidden_dim)]
        self.W3 = [[random.random() * 0.1 for _ in range(action_dim)] for _ in range(hidden_dim)]
        self.b3 = [0.0 for _ in range(action_dim)]

    def relu(self, x):
        return [max(0, v) for v in x]

    def softmax(self, x):
        max_val = max(x)
        exp_x = [math.exp(v - max_val) for v in x]
        sum_exp = sum(exp_x)
        return [v / sum_exp for v in exp_x]

    def forward(self, state):
        h1 = [sum(state[i] * self.W1[i][j] for i in range(len(state))) + self.b1[j] for j in range(self.hidden_dim)]
        a1 = self.relu(h1)
        h2 = [sum(a1[i] * self.W2[i][j] for i in range(self.hidden_dim)) + self.b2[j] for j in range(self.hidden_dim)]
        a2 = self.relu(h2)
        h3 = [sum(a2[i] * self.W3[i][j] for i in range(self.hidden_dim)) + self.b3[j] for j in range(self.action_dim)]
        return self.softmax(h3)

    def select_action(self, state):
        probs = self.forward(state)
        r = random.random()
        cumsum = 0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                return i, probs
        return len(probs) - 1, probs

# ========== 5. DRL-ALNS求解器 ==========
class DRLALNSSolver:
    def __init__(self, random_seed: int = 42):
        random.seed(random_seed)
        self.alns_operators = ALNSOperators(random_seed)
        self.policy_net = PolicyNetwork()

    def generate_initial_solution(self, scenario):
        n = scenario.node_count
        unvisited = set(range(n))
        route = [0]
        unvisited.remove(0)
        current = 0
        while unvisited:
            best_next = None
            best_dist = float('inf')
            for node in unvisited:
                dist = scenario.get_distance(current, node)
                if dist < best_dist:
                    best_dist = dist
                    best_next = node
            route.append(best_next)
            unvisited.remove(best_next)
            current = best_next
        return route

    def solve(self, scenario, max_iterations: int = 50):
        current_route = self.generate_initial_solution(scenario)
        best_route = current_route.copy()
        best_distance = scenario.get_route_distance(best_route)
        temperature = 10.0
        for iteration in range(max_iterations):
            state = [random.random() for _ in range(20)]
            action, _ = self.policy_net.select_action(state)
            new_route = self.alns_operators.apply_operator(scenario, current_route, action)
            new_distance = scenario.get_route_distance(new_route)
            if new_distance < best_distance:
                best_route = new_route.copy()
                best_distance = new_distance
            if new_distance < scenario.get_route_distance(current_route):
                current_route = new_route
            else:
                delta = new_distance - scenario.get_route_distance(current_route)
                if random.random() < math.exp(-delta / temperature):
                    current_route = new_route
            temperature = max(0.1, temperature * 0.95)
        return best_route

# ========== 6. Flask路由 ==========
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/solve', methods=['POST'])
def solve():
    data = request.json
    depots = data['depots']
    points = data['points']
    capacity = data.get('capacity', 20)
    num_vehicles = data.get('num_vehicles', 2)
    cost_per_km = data.get('cost_per_km', 2.0)
    fixed_cost = data.get('fixed_cost', 50.0)

    if not depots or not points:
        return jsonify({'error': '缺少数据'}), 400

    # 构建坐标列表
    all_coords = [(d['lng'], d['lat']) for d in depots] + [(p['lng'], p['lat']) for p in points]
    total_nodes = len(all_coords)
    num_depots = len(depots)

    print(f"计算距离矩阵... 共 {total_nodes} 个节点")
    dist_matrix = [[0.0] * total_nodes for _ in range(total_nodes)]
    for i in range(total_nodes):
        for j in range(total_nodes):
            if i != j:
                dist_matrix[i][j] = get_drive_distance(all_coords[i][0], all_coords[i][1],
                                                       all_coords[j][0], all_coords[j][1])
        print(f"完成 {i+1}/{total_nodes} 行")

    # 分配补货点
    depot_assign = {i: [] for i in range(num_depots)}
    remaining_stock = {i: depots[i].get('stock', 0) for i in range(num_depots)}
    point_indices = list(range(num_depots, total_nodes))
    point_indices.sort(key=lambda idx: points[idx - num_depots]['demand'], reverse=True)

    for p_idx in point_indices:
        demand = points[p_idx - num_depots]['demand']
        best_depot = None
        best_dist = float('inf')
        for d_idx in range(num_depots):
            if remaining_stock[d_idx] >= demand:
                d = dist_matrix[d_idx][p_idx]
                if d < best_dist:
                    best_dist = d
                    best_depot = d_idx
        if best_depot is not None:
            depot_assign[best_depot].append(p_idx)
            remaining_stock[best_depot] -= demand
        else:
            d_idx = min(range(num_depots), key=lambda i: dist_matrix[i][p_idx])
            depot_assign[d_idx].append(p_idx)

    # 构建路线
    all_routes = []
    total_distance = 0.0
    total_cost = 0.0

    for depot_idx in range(num_depots):
        assigned = depot_assign[depot_idx]
        if not assigned:
            continue

        assigned.sort(key=lambda idx: dist_matrix[depot_idx][idx])
        groups = []
        cur_group = []
        cur_load = 0
        for p_idx in assigned:
            demand = points[p_idx - num_depots]['demand']
            if cur_load + demand <= capacity:
                cur_group.append(p_idx)
                cur_load += demand
            else:
                if cur_group:
                    groups.append(cur_group)
                cur_group = [p_idx]
                cur_load = demand
        if cur_group:
            groups.append(cur_group)

        if len(groups) > num_vehicles:
            extra = []
            for g in groups[num_vehicles-1:]:
                extra.extend(g)
            groups = groups[:num_vehicles-1]
            groups.append(extra)

        for veh_idx, group in enumerate(groups):
            ordered = [depot_idx]
            remaining = group.copy()
            current = depot_idx
            while remaining:
                best_next = None
                best_d = float('inf')
                for cand in remaining:
                    d = dist_matrix[current][cand]
                    if d < best_d:
                        best_d = d
                        best_next = cand
                ordered.append(best_next)
                remaining.remove(best_next)
                current = best_next

            if len(ordered) > 3:
                point_part = ordered[1:]
                optimized_part = two_opt(point_part, dist_matrix)
                ordered = [depot_idx] + optimized_part

            dist_vehicle = 0.0
            for i in range(len(ordered)-1):
                dist_vehicle += dist_matrix[ordered[i]][ordered[i+1]]
            cost_vehicle = dist_vehicle * cost_per_km + fixed_cost
            total_distance += dist_vehicle
            total_cost += cost_vehicle

            route_coords = [all_coords[idx] for idx in ordered]
            all_routes.append({
                'depot_index': depot_idx,
                'vehicle_index': len(all_routes) + 1,
                'route': route_coords,
                'distance': dist_vehicle,
                'cost': cost_vehicle
            })
            print(f"车辆 {len(all_routes)}: {len(group)} 个补货点, 距离 {dist_vehicle:.2f} km")

    print("规划完成!")
    return jsonify({
        'total_distance': total_distance,
        'total_cost': total_cost,
        'routes': all_routes
    })

if __name__ == '__main__':
    from waitress import serve
    print("🚀 后端服务器已启动，请访问 http://127.0.0.1:5000")
    serve(app, host='0.0.0.0', port=5000)