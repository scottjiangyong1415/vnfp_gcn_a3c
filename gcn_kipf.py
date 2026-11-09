# gcn_kipf.py
# Kipf一阶GCN（原论文GCN），作为ChebNet的RL基线
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim)*0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, L_norm):
        # L_norm = D^{-1/2} A D^{-1/2} 归一化邻接矩阵（Kipf标准）
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
        # lambda_max只是为了接口和ChebNet对齐，Kipf GCN不用
        h = F.relu(self.g1(x, L_norm))
        h = F.relu(self.g2(h, L_norm))
        hg = torch.mean(h, dim=0, keepdim=True)
        logits = self.actor(hg)
        val = self.critic(hg)
        return logits, val
