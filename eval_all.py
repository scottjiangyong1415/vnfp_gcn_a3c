# eval_all.py
# 统一评估脚本：FF, BF, RF, Kipf-GCN-A3C, ChebNet-A3C
# 输出 result.csv，所有方法在相同仿真配置下对比
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from env import VNEEnv
from config import SUB, VNR

# ===================== 1. Kipf GCN 模块（原论文基线） =====================
class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, L_norm):
        x = L_norm @ x @ self.weight
        x = x + self.bias
        return x

class GCNVNE(nn.Module):
    def __init__(self, in_dim, hid_dim, act_num):
        super().__init__()
        self.g1 = GCNConv(in_dim, hid_dim)
        self.g2 = GCNConv(hid_dim, hid_dim)
        self.actor = nn.Linear(hid_dim, act_num)
        self.critic = nn.Linear(hid_dim, 1)

    def forward(self, x, L_norm, lambda_max=None):
        h = F.relu(self.g1(x, L_norm))
        h = F.relu(self.g2(h, L_norm))
        hg = torch.mean(h, dim=0, keepdim=True)
        logits = self.actor(hg)
        val = self.critic(hg)
        return logits, val

# ===================== 2. ChebNet 模块（你的方法） =====================
class ChebConv(nn.Module):
    def __init__(self, in_dim, out_dim, K):
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.randn(K, in_dim, out_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, L, lambda_max):
        N = x.size(0)
        I = torch.eye(N, device=x.device, dtype=x.dtype)
        L_scaled = (2.0 / lambda_max) * L - I
        Tx = [x]
        if self.K > 1:
            Tx.append(L_scaled @ x)
        for k in range(2, self.K):
            Tk = 2 * L_scaled @ Tx[-1] - Tx[-2]
            Tx.append(Tk)
        out = 0.0
        for k in range(self.K):
            out = out + Tx[k] @ self.weight[k]
        out = out + self.bias
        return out

class ChebNetVNE(nn.Module):
    def __init__(self, in_dim, hid_dim, act_num, K):
        super().__init__()
        self.K = K
        self.c1 = ChebConv(in_dim, hid_dim, K)
        self.c2 = ChebConv(hid_dim, hid_dim, K)
        self.actor = nn.Linear(hid_dim, act_num)
        self.critic = nn.Linear(hid_dim, 1)

    def forward(self, x, L, lambda_max):
        h = F.relu(self.c1(x, L, lambda_max))
        h = F.relu(self.c2(h, L, lambda_max))
        hg = torch.mean(h, dim=0, keepdim=True)
        logits = self.actor(hg)
        val = self.critic(hg)
        return logits, val

# ===================== 3. 贪心策略类 FF/BF/RF =====================
class GreedyAgent:
    def __init__(self, env, strategy="ff"):
        self.env = env
        self.strategy = strategy
        self.N_sub = env.N_sub

    def select_node(self, demand_cpu):
        candidate = []
        for n in range(self.N_sub):
            if self.env.sub.cpu[n] >= demand_cpu:
                candidate.append((n, self.env.sub.cpu[n]))
        if len(candidate) == 0:
            return None
        if self.strategy == "ff":
            candidate.sort(key=lambda x: x[0])
            return candidate[0][0]
        elif self.strategy == "bf":
            candidate.sort(key=lambda x: x[1])
            return candidate[0][0]
        elif self.strategy == "rf":
            return random.choice(candidate)[0]
        else:
            raise ValueError("strategy must be ff/bf/rf")

    def run_one_episode(self):
        obs = self.env.reset()
        ep_reward = 0.0
        accept_cnt = 0
        total_vnr = 0
        sum_util = 0.0
        while True:
            vnr = self.env.current_vnr
            v_ptr = self.env.v_ptr
            req_cpu = vnr["nodes"][v_ptr]["cpu"]
            node = self.select_node(req_cpu)
            if node is None:
                obs, r, _, info = self.env.step(0)
            else:
                obs, r, _, info = self.env.step(node)
            ep_reward += r
            # 统计VNR结果
            if info["accept"] is not None:
                total_vnr += 1
                if info["accept"]:
                    accept_cnt += 1
                # 简单资源利用率：已用CPU/总CPU
                cpu_used = self.env.sub.cpu_orig.sum() - self.env.sub.cpu.sum()
                cpu_total = self.env.sub.cpu_orig.sum()
                util = cpu_used / cpu_total if cpu_total > 0 else 0.0
                sum_util += util
            # 这里不设episode终止，跑固定VNR数量，下面主函数控制
            if total_vnr >= EVAL_VNR_NUM:
                break
        ar = accept_cnt / total_vnr if total_vnr > 0 else 0.0
        avg_util = sum_util / total_vnr if total_vnr > 0 else 0.0
        return total_vnr, accept_cnt, ar, ep_reward / total_vnr, avg_util

# ===================== 4. RL模型评估（贪心推理，不采样） =====================
def eval_rl_model(env, model, model_type="cheb", K=3):
    obs = env.reset()
    ep_reward = 0.0
    accept_cnt = 0
    total_vnr = 0
    sum_util = 0.0
    L_np, lambda_max = env.get_spectral()
    L_tensor = torch.from_numpy(L_np).float()
    while True:
        x = torch.from_numpy(obs).float()
        with torch.no_grad():
            if model_type == "kipf":
                logits, val = model(x, L_tensor)
            else:
                logits, val = model(x, L_tensor, lambda_max)
        # greedy：取最大logit，不探索
        act = torch.argmax(logits, dim=1).item()
        obs, r, _, info = env.step(act)
        ep_reward += r
        if info["accept"] is not None:
            total_vnr += 1
            if info["accept"]:
                accept_cnt += 1
            cpu_used = env.sub.cpu_orig.sum() - env.sub.cpu.sum()
            cpu_total = env.sub.cpu_orig.sum()
            util = cpu_used / cpu_total if cpu_total > 0 else 0.0
            sum_util += util
        if total_vnr >= EVAL_VNR_NUM:
            break
    ar = accept_cnt / total_vnr if total_vnr > 0 else 0.0
    avg_util = sum_util / total_vnr if total_vnr > 0 else 0.0
    return total_vnr, accept_cnt, ar, ep_reward / total_vnr, avg_util

# ===================== 主评估入口 =====================
if __name__ == "__main__":
    # 评估配置：每个方法固定评估多少条VNR，保证公平对比
    EVAL_VNR_NUM = 200
    SEED = 123
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 结果列表
    results = []
    # 待评估方法列表
    methods = ["ff", "bf", "rf", "kipf_gcn_a3c", "chebnet_a3c"]

    # ------------ 贪心基线 FF BF RF ------------
    for m in ["ff", "bf", "rf"]:
        print(f"\n==== Evaluating {m.upper()} ====")
        env = VNEEnv(SUB, VNR)
        agent = GreedyAgent(env, strategy=m)
        tv, ac, ar, avg_r, avg_u = agent.run_one_episode()
        print(f"Total:{tv}, Accepted:{ac}, AR:{ar:.4f}, AvgR:{avg_r:.4f}, Util:{avg_u:.4f}")
        results.append([m, tv, ac, ar, avg_r, avg_u])

    # ------------ Kipf GCN-A3C 基线 ------------
    print(f"\n==== Evaluating Kipf-GCN-A3C ====")
    env_gcn = VNEEnv(SUB, VNR)
    gcn_model = GCNVNE(in_dim=7, hid_dim=64, act_num=env_gcn.N_sub)
    # ========= 【重点】在这里加载你训练好的权重 =========
    # gcn_model.load_state_dict(torch.load("./weights/gcn_a3c_best.pth"))
    gcn_model.eval()
    tv, ac, ar, avg_r, avg_u = eval_rl_model(env_gcn, gcn_model, model_type="kipf")
    print(f"Total:{tv}, Accepted:{ac}, AR:{ar:.4f}, AvgR:{avg_r:.4f}, Util:{avg_u:.4f}")
    results.append(["kipf_gcn_a3c", tv, ac, ar, avg_r, avg_u])

    # ------------ ChebNet-A3C 你的方法 ------------
    print(f"\n==== Evaluating ChebNet-A3C ====")
    env_cheb = VNEEnv(SUB, VNR)
    cheb_model = ChebNetVNE(in_dim=7, hid_dim=64, act_num=env_cheb.N_sub, K=3)
    # ========= 【重点】在这里加载你训练好的权重 =========
    # cheb_model.load_state_dict(torch.load("./weights/cheb_a3c_best.pth"))
    cheb_model.eval()
    tv, ac, ar, avg_r, avg_u = eval_rl_model(env_cheb, cheb_model, model_type="cheb")
    print(f"Total:{tv}, Accepted:{ac}, AR:{ar:.4f}, AvgR:{avg_r:.4f}, Util:{avg_u:.4f}")
    results.append(["chebnet_a3c", tv, ac, ar, avg_r, avg_u])

    # 输出csv
    with open("result.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "total_vnr", "accepted_vnr", "accept_ratio", "avg_reward", "avg_utilization"])
        writer.writerows(results)
    print("\n✅ All done! Result saved to result.csv")
