# -*- coding: utf-8 -*-

import json
import random
import math
from typing import List, Dict, Tuple


class DeliveryScenario:

    def __init__(self):
        self.nodes = []
        self.capacity = 0
        self.distance_matrix = []
        self.cost_per_km = 2.0
        self.fixed_cost = 50.0
        self.node_count = 0

    def generate_random_scenario(self, node_count: int = 20,
                                 area_size: float = 50.0,
                                 demand_range: Tuple[int, int] = (1, 10),
                                 capacity: int = 50,
                                 seed: int = 42):
        random.seed(seed)
        self.nodes = []
        self.capacity = capacity
        self.node_count = node_count

        for i in range(node_count):
            x = random.uniform(0, area_size)
            y = random.uniform(0, area_size)
            demand = random.randint(demand_range[0], demand_range[1])
            self.nodes.append((i, x, y, demand))

        self._compute_distance_matrix()
        return self

    def load_from_file(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.nodes = [(item['id'], item['x'], item['y'], item['demand'])
                      for item in data['nodes']]
        self.capacity = data['capacity']
        self.cost_per_km = data.get('cost_per_km', 2.0)
        self.fixed_cost = data.get('fixed_cost', 50.0)
        self.node_count = len(self.nodes)
        self._compute_distance_matrix()
        return self

    def save_to_file(self, filepath: str):
        data = {
            'capacity': self.capacity,
            'cost_per_km': self.cost_per_km,
            'fixed_cost': self.fixed_cost,
            'nodes': [{'id': n[0], 'x': n[1], 'y': n[2], 'demand': n[3]}
                      for n in self.nodes]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _compute_distance_matrix(self):
        n = self.node_count
        self.distance_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                _, x1, y1, _ = self.nodes[i]
                _, x2, y2, _ = self.nodes[j]
                dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                self.distance_matrix[i][j] = dist
                self.distance_matrix[j][i] = dist

    def get_total_demand(self) -> int:
        return sum(node[3] for node in self.nodes)

    def get_node(self, idx: int):
        return self.nodes[idx]

    def get_distance(self, i: int, j: int) -> float:
        return self.distance_matrix[i][j]

    def get_route_distance(self, route: List[int]) -> float:
        if len(route) < 2:
            return 0.0
        total = 0.0
        for k in range(len(route)-1):
            total += self.get_distance(route[k], route[k+1])
        return total

    def get_route_cost(self, route: List[int]) -> float:
        dist = self.get_route_distance(route)
        return dist * self.cost_per_km + self.fixed_cost

    def check_capacity(self, route: List[int]) -> bool:
        total_demand = sum(self.nodes[i][3] for i in route)
        return total_demand <= self.capacity

    def summary(self) -> str:
        return (f"场景概要: 节点数={self.node_count}, 卡车容量={self.capacity}, "
                f"总需求={self.get_total_demand()}, 成本参数: 每公里{self.cost_per_km}元, "
                f"固定发车{self.fixed_cost}元")


if __name__ == "__main__":
    scenario = DeliveryScenario().generate_random_scenario(node_count=10, capacity=20)
    print(scenario.summary())
    route = list(range(10))
    print("路径距离:", scenario.get_route_distance(route), "公里")
    print("路径成本:", scenario.get_route_cost(route), "元")
    print("容量检查:", scenario.check_capacity(route))