# -*- coding: utf-8 -*-

import os
import sys
import time
import json

from data_manager import DeliveryScenario
from drl_model import DRLTrainer, StateEncoder, DRLConfig
from alns_operators import ALNSOperators, ALNSSolver
from solver import DRLALNSSolver, SolverConfig
from evaluate import ModelEvaluator, create_test_scenarios


def print_banner():
    banner = """
==============================================================
    无人车补货路径深度强化学习智能规划软件 V1.0
    DRL-ALNS Hybrid Intelligent Path Planning Software
==============================================================
    """
    print(banner)


def print_menu():
    menu = """
请选择操作：
  1. 生成随机配送场景
  2. 导入配送场景数据
  3. 训练 DRL 模型
  4. 路径规划求解（DRL-ALNS）
  5. 传统 ALNS 求解
  6. 模型评估与算法对比
  7. 查看场景概要
  8. 保存当前场景
  9. 退出程序
    """
    print(menu)


def generate_scenario():
    print("\n--- 生成随机配送场景 ---")
    try:
        node_count = int(input("请输入快递柜数量（如 20）: "))
        capacity = int(input("请输入卡车容量（如 50）: "))
        area_size = float(input("请输入区域大小（公里，如 50）: "))
        seed = int(input("请输入随机种子（如 42）: "))
    except ValueError:
        print("输入无效，使用默认值")
        node_count = 20
        capacity = 50
        area_size = 50.0
        seed = 42

    scenario = DeliveryScenario().generate_random_scenario(
        node_count=node_count,
        capacity=capacity,
        area_size=area_size,
        seed=seed
    )
    print(scenario.summary())
    return scenario


def load_scenario():
    print("\n--- 导入配送场景数据 ---")
    filepath = input("请输入 JSON 文件路径（如 scenario.json）: ").strip()
    if not filepath:
        filepath = "scenario.json"
    try:
        scenario = DeliveryScenario().load_from_file(filepath)
        print("导入成功！")
        print(scenario.summary())
        return scenario
    except FileNotFoundError:
        print(f"错误：文件 {filepath} 不存在")
        return None
    except Exception as e:
        print(f"导入失败: {e}")
        return None


def train_drl_model(scenario):
    print("\n--- 训练 DRL 模型 ---")
    if scenario is None:
        print("请先生成或导入场景")
        return None

    config = DRLConfig()
    config.print_config()

    modify = input("是否修改默认超参数？(y/n): ").strip().lower()
    if modify == 'y':
        try:
            config.training_epochs = int(input(f"训练轮数（默认{config.training_epochs}）: ") or config.training_epochs)
            config.learning_rate = float(input(f"学习率（默认{config.learning_rate}）: ") or config.learning_rate)
            config.discount_factor = float(input(f"折扣因子（默认{config.discount_factor}）: ") or config.discount_factor)
        except ValueError:
            print("输入无效，使用默认值")

    trainer = DRLTrainer(
        state_dim=20,
        action_dim=12,
        hidden_dim=config.hidden_dim,
        learning_rate=config.learning_rate,
        discount_factor=config.discount_factor
    )
    encoder = StateEncoder()

    alns_ops = ALNSOperators()

    print(f"\n开始训练，共 {config.training_epochs} 轮...")
    for epoch in range(config.training_epochs):
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{config.training_epochs}")

    save_path = input("\n请输入模型保存路径（如 drl_model.pth）: ").strip()
    if not save_path:
        save_path = "drl_model.pth"
    trainer.save_model(save_path)
    print(f"模型已保存到: {save_path}")

    try:
        trainer.plot_training_curves("training_curves.png")
    except:
        pass

    return trainer


def solve_path(scenario, trainer=None):
    print("\n--- 路径规划求解（DRL-ALNS） ---")
    if scenario is None:
        print("请先生成或导入场景")
        return

    encoder = StateEncoder()
    solver = DRLALNSSolver(drl_trainer=trainer, state_encoder=encoder)

    try:
        max_iter = int(input(f"最大迭代次数（默认{solver.max_iterations}）: ") or solver.max_iterations)
    except ValueError:
        max_iter = solver.max_iterations

    use_drl = True
    if trainer is None:
        use_drl = False
        print("未提供训练模型，将使用随机算子选择模式")

    result = solver.solve(scenario, max_iterations=max_iter, use_drl=use_drl, verbose=True)
    solver.print_solution(scenario)
    return result


def traditional_alns_solve(scenario):
    print("\n--- 传统 ALNS 求解 ---")
    if scenario is None:
        print("请先生成或导入场景")
        return

    alns = ALNSSolver()
    try:
        max_iter = int(input(f"最大迭代次数（默认100）: ") or 100)
    except ValueError:
        max_iter = 100

    start_time = time.time()
    route = alns.solve(scenario, max_iterations=max_iter)
    elapsed = time.time() - start_time

    distance = scenario.get_route_distance(route)
    cost = scenario.get_route_cost(route)

    print(f"求解完成！")
    print(f"路径距离: {distance:.2f} km")
    print(f"运营成本: {cost:.2f} 元")
    print(f"求解时间: {elapsed:.2f} 秒")
    return route


def evaluate_models(scenario):
    print("\n--- 模型评估与算法对比 ---")
    if scenario is None:
        print("请先生成或导入场景")
        return

    print("创建测试场景集...")
    scenarios = create_test_scenarios(num_scenarios=3, node_counts=[scenario.node_count],
                                      capacity=scenario.capacity)

    def drl_alns_algorithm(sc):
        solver = DRLALNSSolver()
        result = solver.solve(sc, max_iterations=50, use_drl=False, verbose=False)
        return result

    def traditional_alns_algorithm(sc):
        alns = ALNSSolver()
        route = alns.solve(sc, max_iterations=50)
        return {
            'route': route,
            'distance': sc.get_route_distance(route),
            'cost': sc.get_route_cost(route)
        }

    def greedy_algorithm(sc):
        solver = DRLALNSSolver()
        return solver.solve_greedy(sc)

    algorithms = {
        'DRL-ALNS': drl_alns_algorithm,
        'Traditional ALNS': traditional_alns_algorithm,
        'Greedy': greedy_algorithm
    }

    evaluator = ModelEvaluator()
    report = evaluator.compare_algorithms(algorithms, scenarios, num_runs=2)
    evaluator.print_comparison_report(report)

    save = input("是否保存对比报告？(y/n): ").strip().lower()
    if save == 'y':
        evaluator.save_report(report, 'comparison_report.json')
        evaluator.plot_comparison(report, 'comparison_plot.png')

    return report


def view_scenario(scenario):
    if scenario is None:
        print("当前没有场景")
        return
    print("\n--- 场景概要 ---")
    print(scenario.summary())
    print("前5个节点信息 (id, x, y, demand):")
    for i in range(min(5, scenario.node_count)):
        print(f"  {scenario.get_node(i)}")


def save_scenario(scenario):
    if scenario is None:
        print("当前没有场景")
        return
    filepath = input("请输入保存路径（如 scenario.json）: ").strip()
    if not filepath:
        filepath = "scenario.json"
    scenario.save_to_file(filepath)
    print(f"场景已保存到: {filepath}")


def main():
    print_banner()

    current_scenario = None
    trainer = None

    while True:
        print_menu()
        choice = input("请输入选项（1-9）: ").strip()

        if choice == '1':
            current_scenario = generate_scenario()
        elif choice == '2':
            current_scenario = load_scenario()
        elif choice == '3':
            trainer = train_drl_model(current_scenario)
        elif choice == '4':
            solve_path(current_scenario, trainer)
        elif choice == '5':
            traditional_alns_solve(current_scenario)
        elif choice == '6':
            evaluate_models(current_scenario)
        elif choice == '7':
            view_scenario(current_scenario)
        elif choice == '8':
            save_scenario(current_scenario)
        elif choice == '9':
            print("感谢使用，再见！")
            break
        else:
            print("无效选项，请重新输入")

        input("\n按回车键继续...")


if __name__ == "__main__":
    main()