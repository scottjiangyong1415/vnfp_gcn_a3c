# baseline_greedy.py
# 贪心基线：First-Fit, Best-Fit, Random-Fit
# 复用你现有的VNEEnv，带仿真时钟、VNR指数lifetime=500、超时资源回收
import numpy as np
import random
from env import VNEEnv
from config import SUB, VNR

class GreedyVNE:
    def __init__(self, env, strategy="ff"):
        """
        strategy:
            "ff"  -> First-Fit  从0号节点往后找第一个资源够的底物节点
            "bf"  -> Best-Fit   找资源足够、剩余CPU最小的节点（紧凑放置）
            "rf"  -> Random-Fit 在所有合法节点里随机选一个
        """
        self.env = env
        self.strategy = strategy
        self.N_sub = env.N_sub

    def select_node(self, demand_cpu):
        """根据策略选择底物节点，返回选中id，无合法节点返回None"""
        candidate = []
        for n in range(self.N_sub):
            if self.env.sub.cpu[n] >= demand_cpu:
                candidate.append((n, self.env.sub.cpu[n]))
        if len(candidate) == 0:
            return None

        if self.strategy == "ff":
            # First-Fit：下标从小到大，第一个满足的
            candidate.sort(key=lambda x: x[0])
            return candidate[0][0]
        elif self.strategy == "bf":
            # Best-Fit：满足需求前提下，剩余CPU最小（最紧）
            candidate.sort(key=lambda x: x[1])
            return candidate[0][0]
        elif self.strategy == "rf":
            # Random-Fit：随机采样
            idx = random.choice(candidate)[0]
            return idx
        else:
            raise ValueError("strategy must be ff/bf/rf")

    def run(self, episode_num=100):
        """运行基线仿真，统计指标"""
        accept_cnt = 0
        total_vnr = 0
        reward_sum = 0.0

        for ep in range(episode_num):
            obs = self.env.reset()
            done = False
            while not done:
                vnr = self.env.current_vnr
                v_ptr = self.env.v_ptr
                req_cpu = vnr["nodes"][v_ptr]["cpu"]

                # 贪心选节点
                chosen_node = self.select_node(req_cpu)
                if chosen_node is None:
                    # 没有可用节点，step会拒绝当前VNR
                    obs, r, done, info = self.env.step(0)
                else:
                    obs, r, done, info = self.env.step(chosen_node)

                reward_sum += r
                if info["accept"] is not None:
                    total_vnr += 1
                    if info["accept"]:
                        accept_cnt += 1

        accept_ratio = accept_cnt / total_vnr if total_vnr > 0 else 0.0
        print(f"==== Strategy: {self.strategy} ====")
        print(f"Total VNR: {total_vnr}, Accepted: {accept_cnt}")
        print(f"Acceptance Ratio: {accept_ratio:.4f}")
        print(f"Average Reward: {reward_sum / total_vnr:.4f}\n")
        return accept_ratio


if __name__ == "__main__":
    # 初始化环境，和RL完全一致的参数
    env = VNEEnv(SUB, VNR)
    # 依次跑三种贪心基线
    for s in ["ff", "bf", "rf"]:
        agent = GreedyVNE(env, strategy=s)
        agent.run(episode_num=100)
