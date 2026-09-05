# -*- coding: utf-8 -*-
"""
无人车补货路径规划平台 - 后端（真实距离 + 2-opt 优化）
"""
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import requests
import math
import time

app = Flask(__name__, static_folder='.')
CORS(app)

AMAP_WEB_KEY = "9a4d808c69ba920b2b1ed057a5df3714"

def haversine(lng1, lat1, lng2, lat2):
    R = 6371.0
    lng1_rad, lat1_rad = math.radians(lng1), math.radians(lat1)
    lng2_rad, lat2_rad = math.radians(lng2), math.radians(lat2)
    dlat, dlng = lat2_rad - lat1_rad, lng2_rad - lng1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(dlng/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def get_drive_distance(lng1, lat1, lng2, lat2):
    """获取两点间真实驾车距离（公里），失败时返回直线"""
    url = "https://restapi.amap.com/v3/direction/driving"
    params = {
        "origin": f"{lng1},{lat1}",
        "destination": f"{lng2},{lat2}",
        "output": "json",
        "key": AMAP_WEB_KEY
    }
    try:
        resp = requests.get(url, params=params, timeout=10).json()
        if resp.get("status") == "1" and resp.get("route"):
            return int(resp["route"]["paths"][0]["distance"]) / 1000.0
    except:
        pass
    return haversine(lng1, lat1, lng2, lat2)

def two_opt(route, dist_matrix):
    """2-opt 局部搜索优化，返回优化后的路径（不含起点）"""
    best_route = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                if j - i == 1:
                    continue
                new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                # 计算距离变化
                old_dist = dist_matrix[best_route[i-1]][best_route[i]] + dist_matrix[best_route[j-1]][best_route[j]]
                new_dist = dist_matrix[best_route[i-1]][best_route[j-1]] + dist_matrix[best_route[i]][best_route[j]]
                if new_dist < old_dist:
                    best_route = new_route
                    improved = True
                    break
            if improved:
                break
    return best_route

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

    all_coords = [ (d['lng'], d['lat']) for d in depots ] + [ (p['lng'], p['lat']) for p in points ]
    total_nodes = len(all_coords)
    num_depots = len(depots)

    print("计算真实距离矩阵...")
    dist_matrix = [[0.0]*total_nodes for _ in range(total_nodes)]
    for i in range(total_nodes):
        for j in range(total_nodes):
            if i != j:
                dist_matrix[i][j] = get_drive_distance(all_coords[i][0], all_coords[i][1],
                                                       all_coords[j][0], all_coords[j][1])
                time.sleep(0.03)
        print(f"完成 {i+1}/{total_nodes} 行")

    # 分配补货点给起点
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

    all_routes = []
    total_distance = 0.0
    total_cost = 0.0

    for depot_idx in range(num_depots):
        assigned = depot_assign[depot_idx]
        if not assigned:
            continue

        # 按离起点距离排序后，用容量分组
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
                if cur_group: groups.append(cur_group)
                cur_group = [p_idx]
                cur_load = demand
        if cur_group: groups.append(cur_group)

        # 限制车辆数
        if len(groups) > num_vehicles:
            extra = []
            for g in groups[num_vehicles-1:]:
                extra.extend(g)
            groups = groups[:num_vehicles-1]
            groups.append(extra)

        for veh_idx, group in enumerate(groups):
            # 初始顺序：最近邻
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

            # 对补货点部分进行 2-opt 优化（不含起点）
            if len(ordered) > 3:
                # 提取补货点序列（去掉起点）
                point_part = ordered[1:]
                optimized_part = two_opt(point_part, dist_matrix)
                ordered = [depot_idx] + optimized_part

            # 计算真实距离和成本
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
            print(f"车辆 {len(all_routes)}: 补货点 {len(group)} 个，优化后距离 {dist_vehicle:.2f} km")

    print("规划完成")
    return jsonify({
        'total_distance': total_distance,
        'total_cost': total_cost,
        'routes': all_routes
    })

if __name__ == '__main__':
    from waitress import serve
    print("后端服务器已启动，请访问 http://127.0.0.1:5000")
    serve(app, host='0.0.0.0', port=5000)