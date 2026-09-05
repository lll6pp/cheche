# -*- coding: utf-8 -*-

import numpy as np
import random
import math
import os
from typing import List, Tuple, Dict


class PolicyNetwork:

    def __init__(self, state_dim: int = 20, action_dim: int = 12,
                 hidden_dim: int = 64, lr: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.b2 = np.zeros((1, hidden_dim))
        self.W3 = np.random.randn(hidden_dim, action_dim) * 0.1
        self.b3 = np.zeros((1, action_dim))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, state):
        self.z1 = state @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        probs = self.softmax(self.z3)
        return probs

    def select_action(self, state):
        probs = self.forward(state)
        action = np.random.choice(self.action_dim, p=probs[0])
        return action, probs[0]

    def update(self, states, actions, advantages):
        for state, action, advantage in zip(states, actions, advantages):
            probs = self.forward(state)
            grad = probs.copy()
            grad[0, action] -= 1.0
            self.W3 -= self.lr * advantage * (self.a2.T @ grad)
            self.b3 -= self.lr * advantage * grad
            delta3 = grad @ self.W3.T
            delta3[self.a2 <= 0] = 0
            self.W2 -= self.lr * advantage * (self.a1.T @ delta3)
            self.b2 -= self.lr * advantage * delta3
            delta2 = delta3 @ self.W2.T
            delta2[self.a1 <= 0] = 0
            self.W1 -= self.lr * advantage * (state.T @ delta2)
            self.b1 -= self.lr * advantage * delta2


class StateEncoder:

    def __init__(self, max_nodes: int = 100, max_operators: int = 12):
        self.max_nodes = max_nodes
        self.max_operators = max_operators
        self.state_dim = 20

    def encode(self, scenario, route: List[int], iteration: int,
               max_iterations: int, operator_history: List[int] = None) -> np.ndarray:
        route_distance = scenario.get_route_distance(route)
        total_demand = sum(scenario.get_node(i)[3] for i in route)
        capacity_utilization = total_demand / scenario.capacity if scenario.capacity > 0 else 0
        if len(route) > 1:
            avg_dist = route_distance / (len(route) - 1)
        else:
            avg_dist = 0
        progress = iteration / max_iterations if max_iterations > 0 else 0
        node_ratio = len(route) / scenario.node_count if scenario.node_count > 0 else 0
        capacity_satisfied = 1.0 if total_demand <= scenario.capacity else 0.0
        solution_quality = 1.0
        if operator_history:
            op_hist_avg = sum(operator_history) / len(operator_history)
        else:
            op_hist_avg = 0

        state = np.array([
            route_distance,
            total_demand,
            capacity_utilization,
            avg_dist,
            progress,
            node_ratio,
            capacity_satisfied,
            solution_quality,
            op_hist_avg,
            float(iteration % 100) / 100.0,
            float(len(route) % 50) / 50.0,
            float(scenario.node_count) / 100.0,
            float(scenario.capacity) / 100.0,
            float(total_demand) / 100.0,
            1.0 if route_distance > 0 else 0.0,
            1.0 if capacity_utilization > 0.8 else 0.0,
            1.0 if progress > 0.5 else 0.0,
            1.0 if capacity_satisfied else 0.0,
            random.random(),
            0.5
        ])
        return state.reshape(1, -1)


class DRLTrainer:

    def __init__(self, state_dim: int = 20, action_dim: int = 12,
                 hidden_dim: int = 64, learning_rate: float = 0.001,
                 discount_factor: float = 0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.hidden_dim = hidden_dim

        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, learning_rate)

        self.loss_history = []
        self.reward_history = []
        self.action_history = []

    def select_action(self, state):
        return self.policy_net.select_action(state)

    def compute_reward(self, old_distance: float, new_distance: float,
                       capacity_satisfied: bool) -> float:
        if not capacity_satisfied:
            return -10.0
        if new_distance < old_distance:
            improvement = old_distance - new_distance
            reward = 1.0 + improvement * 10.0
        elif new_distance > old_distance:
            deterioration = new_distance - old_distance
            reward = -1.0 - deterioration * 2.0
        else:
            reward = 0.0
        return reward

    def train_episode(self, scenario, encoder, alns_solver,
                      max_iterations: int = 100):
        initial_route = alns_solver.generate_initial_solution(scenario)
        current_route = initial_route.copy()
        current_distance = scenario.get_route_distance(current_route)

        states = []
        actions = []
        rewards = []
        operator_history = []

        for iteration in range(max_iterations):
            state = encoder.encode(scenario, current_route, iteration,
                                   max_iterations, operator_history)
            action, _ = self.select_action(state)
            actions.append(action)
            states.append(state)

            new_route = alns_solver.apply_operator(scenario, current_route, action)
            new_distance = scenario.get_route_distance(new_route)
            capacity_satisfied = scenario.check_capacity(new_route)
            reward = self.compute_reward(current_distance, new_distance, capacity_satisfied)
            rewards.append(reward)

            if reward >= 0 or random.random() < 0.3:
                current_route = new_route
                current_distance = new_distance

            operator_history.append(action)
            if len(operator_history) > 10:
                operator_history.pop(0)

        cumulative_rewards = []
        running_reward = 0
        for r in reversed(rewards):
            running_reward = r + self.discount_factor * running_reward
            cumulative_rewards.insert(0, running_reward)

        self.policy_net.update(states, actions, cumulative_rewards)

        total_reward = sum(rewards)
        avg_loss = np.mean(np.abs(cumulative_rewards)) if cumulative_rewards else 0
        self.loss_history.append(avg_loss)
        self.reward_history.append(total_reward)
        return total_reward, self.loss_history

    def save_model(self, filepath: str):
        np.savez(filepath,
                 W1=self.policy_net.W1, b1=self.policy_net.b1,
                 W2=self.policy_net.W2, b2=self.policy_net.b2,
                 W3=self.policy_net.W3, b3=self.policy_net.b3,
                 loss_history=np.array(self.loss_history),
                 reward_history=np.array(self.reward_history))
        print(f"模型已保存到: {filepath}")

    def load_model(self, filepath: str):
        data = np.load(filepath)
        self.policy_net.W1 = data['W1']
        self.policy_net.b1 = data['b1']
        self.policy_net.W2 = data['W2']
        self.policy_net.b2 = data['b2']
        self.policy_net.W3 = data['W3']
        self.policy_net.b3 = data['b3']
        self.loss_history = list(data['loss_history'])
        self.reward_history = list(data['reward_history'])

    def plot_training_curves(self, savepath: str = None):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.plot(self.loss_history, label='Loss', color='blue')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss Curve')
            ax1.legend()
            ax1.grid(True)
            ax2.plot(self.reward_history, label='Total Reward', color='red')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Reward')
            ax2.set_title('Training Reward Curve')
            ax2.legend()
            ax2.grid(True)
            plt.tight_layout()
            if savepath:
                plt.savefig(savepath, dpi=150)
                print(f"训练曲线已保存到: {savepath}")
            plt.close()
        except Exception as e:
            print(f"绘制训练曲线时出错: {e}")

    def get_training_statistics(self) -> Dict:
        return {
            'total_episodes': len(self.reward_history),
            'avg_reward': np.mean(self.reward_history) if self.reward_history else 0,
            'max_reward': max(self.reward_history) if self.reward_history else 0,
            'min_reward': min(self.reward_history) if self.reward_history else 0,
            'avg_loss': np.mean(self.loss_history) if self.loss_history else 0,
            'last_reward': self.reward_history[-1] if self.reward_history else 0
        }


class DRLConfig:

    def __init__(self):
        self.training_epochs = 100
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        self.hidden_dim = 64
        self.max_iterations_per_episode = 100
        self.reward_function_type = 'distance_improvement'
        self.entropy_coefficient = 0.01
        self.batch_size = 32
        self.use_gpu = False
        self.random_seed = 42

    def to_dict(self):
        return {
            'training_epochs': self.training_epochs,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'hidden_dim': self.hidden_dim,
            'max_iterations_per_episode': self.max_iterations_per_episode,
            'reward_function_type': self.reward_function_type,
            'entropy_coefficient': self.entropy_coefficient,
            'batch_size': self.batch_size,
            'use_gpu': self.use_gpu,
            'random_seed': self.random_seed
        }

    def print_config(self):
        print("=" * 50)
        print("DRL 训练超参数配置:")
        for key, value in self.to_dict().items():
            print(f"  {key}: {value}")
        print("=" * 50)


if __name__ == "__main__":
    print("DRL 模型训练模块测试（纯 numpy 版）")
    config = DRLConfig()
    config.print_config()
    trainer = DRLTrainer(state_dim=20, action_dim=12, hidden_dim=config.hidden_dim)
    print("策略网络创建成功")
    print("DRL 模型训练模块测试完成")