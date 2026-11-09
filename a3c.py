# a3c.py
# A3C Worker：多线程异步worker，每个worker独立和环境交互，异步更新全局网络
import torch
import torch.optim as optim
from torch.distributions import Categorical
import threading
import numpy as np
from chebnet import ChebNetVNE
from env import VNEEnv

class A3CWorker(threading.Thread):
    def __init__(self, wid, cfg, sub_cfg, vnr_cfg, global_model, global_opt, lock):
        super().__init__()
        self.wid = wid                 # worker编号
        self.cfg = cfg
        self.env = VNEEnv(sub_cfg, vnr_cfg)
        self.L_np, self.lambda_max = self.env.get_spectral()
        self.L = torch.from_numpy(self.L_np).float()
        self.global_model = global_model   # 全局共享网络
        self.opt = global_opt              # 全局优化器
        self.lock = lock                   # 线程锁，保护全局参数更新
        # 本地模型，每个worker独立副本
        self.local_model = ChebNetVNE(
            in_dim=cfg["in_dim"],
            hid_dim=cfg["hidden_dim"],
            act_num=self.env.N_sub,
            K=cfg["cheb_K"]
        )
        self.gamma = cfg["gamma"]
        self.max_ep = cfg["max_episode"]
        self.update_freq = cfg["update_freq"]

    def run(self):
        """线程主循环：持续收集轨迹，定期更新全局网络"""
        for ep in range(self.max_ep):
            obs = self.env.reset()
            logps = []    # 保存动作对数概率
            vals = []     # 保存critic状态价值
            rews = []     # 保存即时奖励
            while True:
                x = torch.from_numpy(obs).float()
                # ChebNet前向传播
                logits, v = self.local_model(x, self.L, self.lambda_max)
                # 分类分布，从logits采样动作
                dist = Categorical(logits=logits)
                act = dist.sample()
                logp = dist.log_prob(act)
                # 执行动作，环境步进
                obs, r, done, info = self.env.step(act.item())

                logps.append(logp)
                vals.append(v)
                rews.append(r)

                # 收集够指定步数，执行一次更新
                if len(rews) >= self.update_freq:
                    self.update(logps, vals, rews, obs, done)
                    logps.clear()
                    vals.clear()
                    rews.clear()
                if done:
                    break
            # 打印评估信息
            if ep % self.cfg["eval_interval"] == 0:
                print(f"Worker {self.wid} | Ep {ep} | Accept: {self.env.accepted}, Reject: {self.env.rejected}")

    def update(self, logps, vals, rews, next_obs, done):
        """A3C损失计算、梯度更新"""
        R = 0.0
        # 最后一步非终止，用critic估计后续回报
        if not done:
            x = torch.from_numpy(next_obs).float()
            _, v = self.local_model(x, self.L, self.lambda_max)
            R = v.item()
        # 逆序计算折扣回报
        ret = []
        for r in reversed(rews):
            R = r + self.gamma * R
            ret.insert(0, R)
        ret = torch.tensor(ret, dtype=torch.float32)
        logps = torch.stack(logps)
        vals = torch.cat(vals).squeeze()
        # 优势函数 A = 回报 - 状态价值
        adv = ret - vals

        # actor损失：最大化优势，等价最小化 -logp * A
        loss_a = -(logps * adv.detach()).mean()
        # critic损失：MSE拟合真实回报
        loss_c = adv.pow(2).mean()
        # 总损失
        loss = loss_a + 0.5 * loss_c

        # 先拉取最新全局参数到本地网络
        with self.lock:
            self.local_model.load_state_dict(self.global_model.state_dict())
        self.local_model.zero_grad()
        loss.backward()
        # 把本地梯度累加至全局网络，全局优化器更新
        with self.lock:
            for gp, lp in zip(self.global_model.parameters(), self.local_model.parameters()):
                if gp.grad is None:
                    gp.grad = torch.zeros_like(gp.data)
                gp.grad += lp.grad
            self.opt.step()
            self.opt.zero_grad()
