# -*- coding: utf-8 -*-

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import requests
import math
import random
import copy
import time
from typing import List, Dict, Tuple, Optional

app = Flask(__name__, static_folder='.')
CORS(app)

AMAP_WEB_KEY = "9a4d808c69ba920b2b1ed057a5df3714"

def haversine(lng1, lat1, lng2, lat2):
    R = 6371.0
    lng1_rad, lat1_rad = math.radians(lng1), math.radians(lat1)
    lng2_rad, lat2_rad = math.radians(lng2), math.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlng = lng2_rad - lng1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(dlng/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def get_drive_distance(lng1, lat1, lng2, lat2):
    url = "https://restapi.amap.com/v3/direction/driving"
    params = {
        "origin": f"{lng1},{lat1}",
        "destination": f"{lng2},{lat2}",
        "output": "json",
        "key": AMAP_WEB_KEY
    }
    try:
        resp = requests.get(url, params=params, timeout=15).json()
        if resp.get("status") == "1" and resp.get("route"):
            distance_m = resp["route"]["paths"][0]["distance"]
            return int(distance_m) / 1000.0
    except Exception as e:
        print(f"高德API请求失败，使用直线距离: {e}")
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
                new_dist = dist_matrix[new_route[i-1]][new_route[i]] + dist_matrix[new_route[j-1]][new_route[j]]
                if new_dist < old_dist:
                    best_route = new_route
                    improved = True
                    break
            if improved:
                break
    return best_route

class DeliveryScenario:
    def __init__(self):
        self.nodes = []
        self.depot_indices = []
        self.point_indices = []
        self.capacity = 0
        self.distance_matrix = []
        self.cost_per_km = 2.0
        self.fixed_cost = 50.0
        self.node_count = 0

    def initialize(self, depots, points, capacity, cost_per_km, fixed_cost):
        self.nodes = []
        self.depot_indices = []
        self.point_indices = []
        for i, d in enumerate(depots):
            self.nodes.append((i, d['lng'], d['lat'], 0, d.get('stock', 100)))
            self.depot_indices.append(i)
        offset = len(depots)
        for j, p in enumerate(points):
            self.nodes.append((offset + j, p['lng'], p['lat'], p['demand'], 0))
            self.point_indices.append(offset + j)
        self.node_count = len(self.nodes)
        self.capacity = capacity
        self.cost_per_km = cost_per_km
        self.fixed_cost = fixed_cost
        self._compute_distance_matrix()

    def _compute_distance_matrix(self):
        n = self.node_count
        self.distance_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            _, x1, y1, _, _ = self.nodes[i]
            for j in range(i + 1, n):
                _, x2, y2, _, _ = self.nodes[j]
                dist = get_drive_distance(x1, y1, x2, y2)
                self.distance_matrix[i][j] = dist
                self.distance_matrix[j][i] = dist
            print(f"距离矩阵: 完成 {i+1}/{n} 行")

    def get_distance(self, i, j):
        return self.distance_matrix[i][j]

    def get_route_distance(self, route):
        if len(route) < 2:
            return 0.0
        total = 0.0
        for k in range(len(route)-1):
            total += self.get_distance(route[k], route[k+1])
        return total

    def get_route_cost(self, route):
        return self.get_route_distance(route) * self.cost_per_km + self.fixed_cost

    def check_capacity(self, route):
        total = sum(self.nodes[i][3] for i in route)
        return total <= self.capacity

    def get_demand(self, node_idx):
        return self.nodes[node_idx][3]

    def get_coords(self, node_idx):
        return (self.nodes[node_idx][1], self.nodes[node_idx][2])

class ALNSOperators:
    def __init__(self, random_seed=42):
        random.seed(random_seed)

    def random_removal(self, route, q):
        if len(route) <= 2:
            return route.copy(), []
        q = min(q, len(route) - 2)
        indices = random.sample(range(len(route)), q)
        removed = [route[i] for i in indices]
        remaining = [route[i] for i in range(len(route)) if i not in indices]
        return remaining, removed

    def worst_removal(self, scenario, route, q):
        if len(route) <= 2:
            return route.copy(), []
        q = min(q, len(route) - 2)
        route_copy = route.copy()
        removed = []
        for _ in range(q):
            if len(route_copy) <= 2:
                break
            max_saving = -1
            best_idx = -1
            for i in range(len(route_copy)):
                if i == 0:
                    saving = scenario.get_distance(route_copy[i], route_copy[i+1])
                elif i == len(route_copy) - 1:
                    saving = scenario.get_distance(route_copy[i-1], route_copy[i])
                else:
                    old = scenario.get_distance(route_copy[i-1], route_copy[i]) + scenario.get_distance(route_copy[i], route_copy[i+1])
                    new = scenario.get_distance(route_copy[i-1], route_copy[i+1])
                    saving = old - new
                if saving > max_saving:
                    max_saving = saving
                    best_idx = i
            if best_idx != -1:
                removed.append(route_copy[best_idx])
                route_copy.pop(best_idx)
        return route_copy, removed

    def greedy_insertion(self, scenario, route, removed_nodes):
        route_copy = route.copy()
        nodes_to_insert = removed_nodes.copy()
        while nodes_to_insert:
            best_node = None
            best_pos = -1
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
                        old = scenario.get_distance(route_copy[pos-1], route_copy[pos])
                        new = scenario.get_distance(route_copy[pos-1], node) + scenario.get_distance(node, route_copy[pos])
                        cost = new - old
                    test_route = route_copy[:pos] + [node] + route_copy[pos:]
                    if scenario.check_capacity(test_route):
                        if cost < best_cost:
                            best_cost = cost
                            best_node = node
                            best_pos = pos
            if best_node is not None:
                route_copy.insert(best_pos, best_node)
                nodes_to_insert.remove(best_node)
            else:
                route_copy.append(nodes_to_insert[0])
                nodes_to_insert.pop(0)
        return route_copy

    def regret_insertion(self, scenario, route, removed_nodes):
        route_copy = route.copy()
        nodes_to_insert = removed_nodes.copy()
        while nodes_to_insert:
            max_regret = -1
            best_node = None
            best_pos = -1
            for node in nodes_to_insert:
                costs = []
                for pos in range(len(route_copy) + 1):
                    if len(route_copy) == 0:
                        cost = 0
                    elif pos == 0:
                        cost = scenario.get_distance(node, route_copy[0])
                    elif pos == len(route_copy):
                        cost = scenario.get_distance(route_copy[-1], node)
                    else:
                        old = scenario.get_distance(route_copy[pos-1], route_copy[pos])
                        new = scenario.get_distance(route_copy[pos-1], node) + scenario.get_distance(node, route_copy[pos])
                        cost = new - old
                    test_route = route_copy[:pos] + [node] + route_copy[pos:]
                    if scenario.check_capacity(test_route):
                        costs.append((cost, pos))
                if len(costs) >= 2:
                    costs.sort(key=lambda x: x[0])
                    regret = costs[1][0] - costs[0][0]
                    if regret > max_regret:
                        max_regret = regret
                        best_node = node
                        best_pos = costs[0][1]
            if best_node is not None:
                route_copy.insert(best_pos, best_node)
                nodes_to_insert.remove(best_node)
            else:
                route_copy.append(nodes_to_insert[0])
                nodes_to_insert.pop(0)
        return route_copy

    def apply_operator(self, scenario, route, operator_id):
        q = max(1, int(len(route) * 0.2))
        if operator_id == 0:
            destroyed, removed = self.random_removal(route, q)
        elif operator_id == 1:
            destroyed, removed = self.worst_removal(scenario, route, q)
        else:
            destroyed, removed = self.random_removal(route, q)
        if operator_id == 2:
            return self.greedy_insertion(scenario, destroyed, removed)
        elif operator_id == 3:
            return self.regret_insertion(scenario, destroyed, removed)
        else:
            return self.greedy_insertion(scenario, destroyed, removed)

class PolicyNetwork:
    def __init__(self, state_dim=20, action_dim=6, hidden_dim=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.W1 = [[random.uniform(-0.1, 0.1) for _ in range(hidden_dim)] for _ in range(state_dim)]
        self.b1 = [0.0] * hidden_dim
        self.W2 = [[random.uniform(-0.1, 0.1) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.b2 = [0.0] * hidden_dim
        self.W3 = [[random.uniform(-0.1, 0.1) for _ in range(action_dim)] for _ in range(hidden_dim)]
        self.b3 = [0.0] * action_dim

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
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                return i
        return self.action_dim - 1

    def encode_state(self, scenario, route, iteration, max_iterations, operator_history):
        route_dist = scenario.get_route_distance(route)
        total_demand = sum(scenario.get_demand(i) for i in route)
        cap_util = total_demand / scenario.capacity if scenario.capacity > 0 else 0
        progress = iteration / max_iterations if max_iterations > 0 else 0
        avg_dist = route_dist / len(route) if len(route) > 0 else 0
        if operator_history:
            op_avg = sum(operator_history) / len(operator_history)
            op_std = math.sqrt(sum((x - op_avg) ** 2 for x in operator_history) / len(operator_history)) if len(operator_history) > 1 else 0
        else:
            op_avg = 0
            op_std = 0
        return [
            min(route_dist / 100, 1.0),
            min(total_demand / 100, 1.0),
            cap_util,
            min(avg_dist / 50, 1.0),
            progress,
            len(route) / 50,
            1.0 if total_demand <= scenario.capacity else 0.0,
            op_avg / 6,
            op_std / 6,
            random.random()
        ]

class DRLALNSSolver:
    def __init__(self, random_seed=42):
        random.seed(random_seed)
        self.alns = ALNSOperators(random_seed)
        self.policy = PolicyNetwork(action_dim=6)
        self.max_iterations = 200
        self.temperature = 10.0
        self.cooling_rate = 0.95
        self.min_temperature = 0.1
        self.operator_history = []

    def generate_initial_solution(self, scenario, start_node=0):
        n = scenario.node_count
        unvisited = set(range(n))
        unvisited.discard(start_node)
        route = [start_node]
        current = start_node
        while unvisited:
            best_next = None
            best_dist = float('inf')
            for node in unvisited:
                dist = scenario.get_distance(current, node)
                if dist < best_dist:
                    best_dist = dist
                    best_next = node
            if best_next is None:
                break
            route.append(best_next)
            unvisited.remove(best_next)
            current = best_next
        return route

    def solve(self, scenario, max_iterations=None, use_drl=True):
        if max_iterations is None:
            max_iterations = self.max_iterations
        current_route = self.generate_initial_solution(scenario)
        best_route = current_route.copy()
        best_distance = scenario.get_route_distance(best_route)
        temperature = self.temperature
        self.operator_history = []
        for iteration in range(max_iterations):
            state = self.policy.encode_state(scenario, current_route, iteration, max_iterations, self.operator_history)
            if use_drl:
                action = self.policy.select_action(state)
            else:
                action = random.randint(0, 5)
            self.operator_history.append(action)
            if len(self.operator_history) > 20:
                self.operator_history.pop(0)
            new_route = self.alns.apply_operator(scenario, current_route, action)
            if not scenario.check_capacity(new_route):
                continue
            new_distance = scenario.get_route_distance(new_route)
            if new_distance < best_distance:
                best_route = new_route.copy()
                best_distance = new_distance
                current_route = new_route
            elif new_distance < scenario.get_route_distance(current_route):
                current_route = new_route
            else:
                delta = new_distance - scenario.get_route_distance(current_route)
                if random.random() < math.exp(-delta / temperature):
                    current_route = new_route
            temperature = max(self.min_temperature, temperature * self.cooling_rate)
        return best_route

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/solve', methods=['POST'])
def solve():
    data = request.json
    depots = data.get('depots', [])
    points = data.get('points', [])
    capacity = data.get('capacity', 20)
    num_vehicles = data.get('num_vehicles', 2)
    cost_per_km = data.get('cost_per_km', 2.0)
    fixed_cost = data.get('fixed_cost', 50.0)

    if not depots or not points:
        return jsonify({'error': '缺少起点或补货点数据'}), 400

    scenario = DeliveryScenario()
    scenario.initialize(depots, points, capacity, cost_per_km, fixed_cost)

    num_depots = len(depots)
    total_nodes = scenario.node_count

    depot_assign = {i: [] for i in range(num_depots)}
    remaining_stock = {i: depots[i].get('stock', 100) for i in range(num_depots)}

    point_indices = list(range(num_depots, total_nodes))
    point_indices.sort(key=lambda idx: scenario.get_demand(idx), reverse=True)

    for p_idx in point_indices:
        demand = scenario.get_demand(p_idx)
        best_depot = None
        best_dist = float('inf')
        for d_idx in range(num_depots):
            if remaining_stock[d_idx] >= demand:
                d = scenario.get_distance(d_idx, p_idx)
                if d < best_dist:
                    best_dist = d
                    best_depot = d_idx
        if best_depot is not None:
            depot_assign[best_depot].append(p_idx)
            remaining_stock[best_depot] -= demand
        else:
            d_idx = min(range(num_depots), key=lambda i: scenario.get_distance(i, p_idx))
            depot_assign[d_idx].append(p_idx)

    all_routes = []
    total_distance = 0.0
    total_cost = 0.0

    for d_idx in range(num_depots):
        assigned = depot_assign[d_idx]
        if not assigned:
            continue

        assigned.sort(key=lambda idx: scenario.get_distance(d_idx, idx))

        groups = []
        cur_group = []
        cur_load = 0
        for p_idx in assigned:
            demand = scenario.get_demand(p_idx)
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

        for group in groups:
            temp_scenario = DeliveryScenario()
            temp_depots = [depots[d_idx]]
            temp_points = []
            for idx in group:
                node = scenario.nodes[idx]
                temp_points.append({'lng': node[1], 'lat': node[2], 'demand': node[3]})
            temp_scenario.initialize(temp_depots, temp_points, capacity, cost_per_km, fixed_cost)

            solver = DRLALNSSolver()
            if len(group) > 1:
                opt_route = solver.solve(temp_scenario, max_iterations=50)
            else:
                opt_route = [0, 1]

            route_indices = [d_idx]
            for node_idx in opt_route:
                if node_idx == 0:
                    continue
                original_idx = group[node_idx - 1] if node_idx - 1 < len(group) else group[0]
                route_indices.append(original_idx)

            if len(route_indices) > 3:
                point_part = route_indices[1:]
                opt_point_part = two_opt(point_part, scenario.distance_matrix)
                route_indices = [d_idx] + opt_point_part

            dist_veh = 0.0
            for i in range(len(route_indices)-1):
                dist_veh += scenario.get_distance(route_indices[i], route_indices[i+1])
            cost_veh = dist_veh * cost_per_km + fixed_cost
            total_distance += dist_veh
            total_cost += cost_veh

            route_coords = [scenario.get_coords(idx) for idx in route_indices]

            all_routes.append({
                'depot_index': d_idx,
                'vehicle_index': len(all_routes) + 1,
                'route': route_coords,
                'distance': dist_veh,
                'cost': cost_veh
            })

    return jsonify({
        'total_distance': total_distance,
        'total_cost': total_cost,
        'routes': all_routes
    })

if __name__ == '__main__':
    from waitress import serve
    print("🚀 无人车补货路径规划平台后端已启动")
    print("📍 访问地址: http://127.0.0.1:5000")
    serve(app, host='0.0.0.0', port=5000)