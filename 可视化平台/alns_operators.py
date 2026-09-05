# -*- coding: utf-8 -*-

import random
import math
from typing import List, Tuple


class ALNSOperators:

    def __init__(self, random_seed: int = 42):
        random.seed(random_seed)
        self.destroy_operators = {
            0: self.random_removal,
            1: self.worst_removal,
            2: self.related_removal
        }
        self.repair_operators = {
            0: self.greedy_insertion,
            1: self.regret_insertion,
            2: self.best_insertion
        }
        self.total_operators = 6

    def get_operator_list(self) -> List[str]:
        return [
            "随机移除 (Random Removal)",
            "最差移除 (Worst Removal)",
            "相关移除 (Related Removal)",
            "贪婪插入 (Greedy Insertion)",
            "后悔插入 (Regret Insertion)",
            "最优插入 (Best Insertion)"
        ]

    def apply_operator(self, scenario, route: List[int], operator_id: int) -> List[int]:
        if operator_id < 3:
            q = max(1, int(len(route) * 0.2))
            destroyed_route, removed_nodes = self.destroy_operators[operator_id](scenario, route, q)
            new_route = self.greedy_insertion(scenario, destroyed_route, removed_nodes)
            return new_route
        else:
            q = max(1, int(len(route) * 0.2))
            destroyed_route, removed_nodes = self.random_removal(scenario, route, q)
            if operator_id == 3:
                return self.greedy_insertion(scenario, destroyed_route, removed_nodes)
            elif operator_id == 4:
                return self.regret_insertion(scenario, destroyed_route, removed_nodes)
            else:
                return self.best_insertion(scenario, destroyed_route, removed_nodes)

    def random_removal(self, scenario, route: List[int], q: int) -> Tuple[List[int], List[int]]:
        if len(route) <= 2:
            return route.copy(), []
        q = min(q, len(route) - 2)
        removed_indices = random.sample(range(len(route)), q)
        removed_nodes = [route[i] for i in removed_indices]
        remaining_route = [route[i] for i in range(len(route)) if i not in removed_indices]
        return remaining_route, removed_nodes

    def worst_removal(self, scenario, route: List[int], q: int) -> Tuple[List[int], List[int]]:
        if len(route) <= 2:
            return route.copy(), []
        q = min(q, len(route) - 2)
        route_copy = route.copy()
        removed_nodes = []

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
                    original = (scenario.get_distance(route_copy[i-1], route_copy[i]) +
                                scenario.get_distance(route_copy[i], route_copy[i+1]))
                    new_dist = scenario.get_distance(route_copy[i-1], route_copy[i+1])
                    saving = original - new_dist
                if saving > max_saving:
                    max_saving = saving
                    best_idx = i
            if best_idx != -1:
                removed_nodes.append(route_copy[best_idx])
                route_copy.pop(best_idx)

        return route_copy, removed_nodes

    def related_removal(self, scenario, route: List[int], q: int) -> Tuple[List[int], List[int]]:
        if len(route) <= 2:
            return route.copy(), []
        q = min(q, len(route) - 2)
        route_copy = route.copy()

        seed_idx = random.randrange(len(route_copy))
        seed_node = route_copy[seed_idx]
        removed = [seed_node]
        route_copy.pop(seed_idx)

        while len(removed) < q and route_copy:
            seed_info = scenario.get_node(seed_node)
            seed_x, seed_y, seed_demand = seed_info[1], seed_info[2], seed_info[3]

            best_similarity = float('inf')
            best_idx = -1
            for i, node in enumerate(route_copy):
                node_info = scenario.get_node(node)
                x, y, demand = node_info[1], node_info[2], node_info[3]
                dist = math.sqrt((x - seed_x)**2 + (y - seed_y)**2)
                demand_diff = abs(demand - seed_demand)
                similarity = dist + 0.5 * demand_diff
                if similarity < best_similarity:
                    best_similarity = similarity
                    best_idx = i
            if best_idx != -1:
                removed.append(route_copy[best_idx])
                route_copy.pop(best_idx)

        return route_copy, removed

    def greedy_insertion(self, scenario, route: List[int], removed_nodes: List[int]) -> List[int]:
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
                        new_dist = (scenario.get_distance(route_copy[pos-1], node) +
                                    scenario.get_distance(node, route_copy[pos]))
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

    def regret_insertion(self, scenario, route: List[int], removed_nodes: List[int]) -> List[int]:
        route_copy = route.copy()
        nodes_to_insert = removed_nodes.copy()

        while nodes_to_insert:
            max_regret = -1
            best_node = None
            best_position = -1

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
                        original = scenario.get_distance(route_copy[pos-1], route_copy[pos])
                        new_dist = (scenario.get_distance(route_copy[pos-1], node) +
                                    scenario.get_distance(node, route_copy[pos]))
                        cost = new_dist - original
                    costs.append((cost, pos))

                costs.sort(key=lambda x: x[0])
                if len(costs) >= 2:
                    regret = costs[1][0] - costs[0][0]
                else:
                    regret = costs[0][0]

                if regret > max_regret:
                    max_regret = regret
                    best_node = node
                    best_position = costs[0][1]

            if best_node is not None:
                route_copy.insert(best_position, best_node)
                nodes_to_insert.remove(best_node)
            else:
                route_copy.append(nodes_to_insert[0])
                nodes_to_insert.pop(0)

        return route_copy

    def best_insertion(self, scenario, route: List[int], removed_nodes: List[int]) -> List[int]:
        route_copy = route.copy()
        nodes_to_insert = removed_nodes.copy()

        while nodes_to_insert:
            best_combination = None
            best_total_distance = float('inf')

            for node in nodes_to_insert:
                for pos in range(len(route_copy) + 1):
                    test_route = route_copy[:pos] + [node] + route_copy[pos:]
                    if scenario.check_capacity(test_route):
                        total_dist = scenario.get_route_distance(test_route)
                        if total_dist < best_total_distance:
                            best_total_distance = total_dist
                            best_combination = (node, pos)

            if best_combination is not None:
                node, pos = best_combination
                route_copy.insert(pos, node)
                nodes_to_insert.remove(node)
            else:
                route_copy.append(nodes_to_insert[0])
                nodes_to_insert.pop(0)

        return route_copy


class ALNSSolver:

    def __init__(self, random_seed: int = 42):
        self.operators = ALNSOperators(random_seed)
        self.operator_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.operator_scores = [0.0] * 6
        self.operator_counts = [0] * 6
        self.decay = 0.9

    def generate_initial_solution(self, scenario) -> List[int]:
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

    def select_operator(self) -> int:
        total_weight = sum(self.operator_weights)
        r = random.random() * total_weight
        cumulative = 0
        for i, w in enumerate(self.operator_weights):
            cumulative += w
            if r <= cumulative:
                return i
        return len(self.operator_weights) - 1

    def apply_operator(self, scenario, route: List[int], operator_id: int = None) -> List[int]:
        if operator_id is None:
            operator_id = self.select_operator()
        return self.operators.apply_operator(scenario, route, operator_id)

    def update_weights(self, operator_id: int, score: float):
        self.operator_scores[operator_id] += score
        self.operator_counts[operator_id] += 1
        if self.operator_counts[operator_id] > 0:
            avg_score = self.operator_scores[operator_id] / self.operator_counts[operator_id]
            self.operator_weights[operator_id] = self.operator_weights[operator_id] * self.decay + \
                                                 (1 - self.decay) * avg_score

    def solve(self, scenario, max_iterations: int = 100) -> List[int]:
        current_route = self.generate_initial_solution(scenario)
        best_route = current_route.copy()
        best_distance = scenario.get_route_distance(best_route)

        for iteration in range(max_iterations):
            op_id = self.select_operator()
            new_route = self.apply_operator(scenario, current_route, op_id)
            new_distance = scenario.get_route_distance(new_route)

            if new_distance < best_distance:
                score = 1.0
                best_route = new_route.copy()
                best_distance = new_distance
            elif new_distance < scenario.get_route_distance(current_route):
                score = 0.5
            else:
                score = 0.1

            self.update_weights(op_id, score)

            if new_distance < scenario.get_route_distance(current_route):
                current_route = new_route
            else:
                temperature = max(0.1, 1.0 - iteration / max_iterations)
                delta = new_distance - scenario.get_route_distance(current_route)
                if random.random() < math.exp(-delta / temperature):
                    current_route = new_route

        return best_route


if __name__ == "__main__":
    print("ALNS 算子管理模块测试")
    import sys
    sys.path.append('.')
    from data_manager import DeliveryScenario

    scenario = DeliveryScenario().generate_random_scenario(node_count=10, capacity=30)
    alns = ALNSSolver()
    initial = alns.generate_initial_solution(scenario)
    print("初始解距离:", scenario.get_route_distance(initial))

    ops = ALNSOperators()
    for i, name in enumerate(ops.get_operator_list()):
        new_route = ops.apply_operator(scenario, initial, i)
        print(f"{name}: 距离={scenario.get_route_distance(new_route):.2f}")

    best = alns.solve(scenario, max_iterations=50)
    print("ALNS 最优路径距离:", scenario.get_route_distance(best))
    print("ALNS 算子管理模块测试完成")