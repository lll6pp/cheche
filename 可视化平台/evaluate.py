# -*- coding: utf-8 -*-

import numpy as np
import time
from typing import List, Dict, Tuple
import json


class ModelEvaluator:

    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        self.results = {}

    def evaluate_algorithm(self, algorithm_func, scenario_list: List,
                           algorithm_name: str, num_runs: int = 5) -> Dict:
        all_distances = []
        all_costs = []
        all_times = []

        for scenario in scenario_list:
            run_distances = []
            run_costs = []
            run_times = []

            for run in range(num_runs):
                start_time = time.time()
                result = algorithm_func(scenario)
                end_time = time.time()

                distance = result.get('distance', float('inf'))
                cost = result.get('cost', float('inf'))
                elapsed = end_time - start_time

                run_distances.append(distance)
                run_costs.append(cost)
                run_times.append(elapsed)

            all_distances.extend(run_distances)
            all_costs.extend(run_costs)
            all_times.extend(run_times)

        stats = {
            'algorithm': algorithm_name,
            'num_scenarios': len(scenario_list),
            'num_runs_per_scenario': num_runs,
            'total_runs': len(all_distances),
            'best_distance': min(all_distances),
            'worst_distance': max(all_distances),
            'avg_distance': np.mean(all_distances),
            'std_distance': np.std(all_distances),
            'avg_cost': np.mean(all_costs),
            'avg_time': np.mean(all_times),
            'all_distances': all_distances,
            'all_costs': all_costs,
            'all_times': all_times
        }

        self.results[algorithm_name] = stats
        return stats

    def compare_algorithms(self, algorithms_dict: Dict[str, callable],
                           scenario_list: List, num_runs: int = 5) -> Dict:
        report = {}
        for name, func in algorithms_dict.items():
            print(f"正在评估: {name} ...")
            stats = self.evaluate_algorithm(func, scenario_list, name, num_runs)
            report[name] = stats

        if len(report) > 1:
            baseline_name = list(report.keys())[0]
            baseline_avg = report[baseline_name]['avg_distance']
            for name in report:
                if name != baseline_name:
                    improvement = (baseline_avg - report[name]['avg_distance']) / baseline_avg * 100
                    report[name]['improvement_vs_baseline'] = improvement

        return report

    def print_comparison_report(self, report: Dict):
        print("\n" + "=" * 70)
        print("算法性能对比报告")
        print("=" * 70)

        for name, stats in report.items():
            print(f"\n算法: {name}")
            print(f"  场景数: {stats['num_scenarios']}, 每场景运行次数: {stats['num_runs_per_scenario']}")
            print(f"  最优距离: {stats['best_distance']:.2f} km")
            print(f"  最差距离: {stats['worst_distance']:.2f} km")
            print(f"  平均距离: {stats['avg_distance']:.2f} km")
            print(f"  标准差: {stats['std_distance']:.2f} km")
            print(f"  平均成本: {stats['avg_cost']:.2f} 元")
            print(f"  平均耗时: {stats['avg_time']:.4f} 秒")
            if 'improvement_vs_baseline' in stats:
                print(f"  相对基准改进: {stats['improvement_vs_baseline']:.2f}%")

        print("=" * 70)

    def save_report(self, report: Dict, filepath: str):
        summary = {}
        for name, stats in report.items():
            summary[name] = {
                'algorithm': stats['algorithm'],
                'num_scenarios': stats['num_scenarios'],
                'num_runs_per_scenario': stats['num_runs_per_scenario'],
                'total_runs': stats['total_runs'],
                'best_distance': stats['best_distance'],
                'worst_distance': stats['worst_distance'],
                'avg_distance': stats['avg_distance'],
                'std_distance': stats['std_distance'],
                'avg_cost': stats['avg_cost'],
                'avg_time': stats['avg_time']
            }
            if 'improvement_vs_baseline' in stats:
                summary[name]['improvement_vs_baseline'] = stats['improvement_vs_baseline']

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"对比报告已保存到: {filepath}")

    def plot_comparison(self, report: Dict, savepath: str = None):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            algo_names = list(report.keys())
            distances = [report[name]['all_distances'] for name in algo_names]
            avg_distances = [report[name]['avg_distance'] for name in algo_names]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            bp = ax1.boxplot(distances, labels=algo_names, patch_artist=True)
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
            for patch, color in zip(bp['boxes'], colors[:len(algo_names)]):
                patch.set_facecolor(color)
            ax1.set_ylabel('Distance (km)')
            ax1.set_title('Distance Distribution Comparison')
            ax1.grid(True, alpha=0.3)

            bars = ax2.bar(algo_names, avg_distances, color=colors[:len(algo_names)])
            ax2.set_ylabel('Average Distance (km)')
            ax2.set_title('Average Distance Comparison')
            ax2.grid(True, alpha=0.3)
            for bar, val in zip(bars, avg_distances):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f'{val:.2f}', ha='center', va='bottom')

            plt.tight_layout()
            if savepath:
                plt.savefig(savepath, dpi=150)
                print(f"对比图已保存到: {savepath}")
            plt.close()
        except Exception as e:
            print(f"绘制对比图时出错: {e}")

    def generate_detailed_report(self, report: Dict) -> str:
        lines = []
        lines.append("=" * 70)
        lines.append("模型评估与对比详细报告")
        lines.append("=" * 70)
        lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        for name, stats in report.items():
            lines.append(f"算法名称: {name}")
            lines.append(f"  测试场景数: {stats['num_scenarios']}")
            lines.append(f"  每场景运行次数: {stats['num_runs_per_scenario']}")
            lines.append(f"  总运行次数: {stats['total_runs']}")
            lines.append(f"  最优解距离: {stats['best_distance']:.4f} km")
            lines.append(f"  最差解距离: {stats['worst_distance']:.4f} km")
            lines.append(f"  平均距离: {stats['avg_distance']:.4f} km")
            lines.append(f"  距离标准差: {stats['std_distance']:.4f} km")
            lines.append(f"  平均成本: {stats['avg_cost']:.4f} 元")
            lines.append(f"  平均耗时: {stats['avg_time']:.4f} 秒")
            if 'improvement_vs_baseline' in stats:
                lines.append(f"  相对改进: {stats['improvement_vs_baseline']:.2f}%")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)


def create_test_scenarios(num_scenarios: int = 5, node_counts: List[int] = None,
                          capacity: int = 50, seed: int = 42) -> List:
    from data_manager import DeliveryScenario

    if node_counts is None:
        node_counts = [20, 30, 40, 50, 60]

    scenarios = []
    for i in range(num_scenarios):
        nc = node_counts[i % len(node_counts)]
        scenario = DeliveryScenario().generate_random_scenario(
            node_count=nc,
            capacity=capacity,
            seed=seed + i
        )
        scenarios.append(scenario)

    return scenarios


if __name__ == "__main__":
    print("模型评估与对比模块测试")

    from data_manager import DeliveryScenario
    from solver import DRLALNSSolver
    from alns_operators import ALNSSolver

    scenarios = create_test_scenarios(num_scenarios=3, node_counts=[15, 20, 25], capacity=40)

    def drl_alns_algorithm(scenario):
        solver = DRLALNSSolver()
        result = solver.solve(scenario, max_iterations=50, use_drl=False, verbose=False)
        return result

    def traditional_alns_algorithm(scenario):
        alns = ALNSSolver()
        route = alns.solve(scenario, max_iterations=50)
        return {
            'route': route,
            'distance': scenario.get_route_distance(route),
            'cost': scenario.get_route_cost(route)
        }

    def greedy_algorithm(scenario):
        solver = DRLALNSSolver()
        result = solver.solve_greedy(scenario)
        return result

    algorithms = {
        'DRL-ALNS': drl_alns_algorithm,
        'Traditional ALNS': traditional_alns_algorithm,
        'Greedy': greedy_algorithm
    }

    evaluator = ModelEvaluator()
    report = evaluator.compare_algorithms(algorithms, scenarios, num_runs=2)

    evaluator.print_comparison_report(report)

    evaluator.save_report(report, 'comparison_report.json')

    print("\n模型评估与对比模块测试完成")