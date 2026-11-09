# chebnet.py
# ChebNet实现：切比雪夫谱域图卷积网络，替换原文Kipf一阶GCN
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChebConv(nn.Module):
    """
    ChebNet卷积层
    in_dim: 输入特征维度
    out_dim: 输出特征维度
    K: 切比雪夫多项式阶数
    """
    def __init__(self, in_dim, out_dim, K):
        super().__init__()
        self.K = K
        # 权重参数 [K, in_dim, out_dim]，每阶多项式对应一组权重
        self.weight = nn.Parameter(torch.randn(K, in_dim, out_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, L, lambda_max):
        """
        x: 节点特征矩阵 [N, in_dim]
        L: 归一化对称拉普拉斯 [N,N]
        lambda_max: 拉普拉斯最大特征值
        return: 卷积后的节点特征 [N, out_dim]
        """
        N = x.size(0)
        I = torch.eye(N, device=x.device, dtype=x.dtype)
        # 将拉普拉斯缩放至 [-1, 1]，切比雪夫多项式定义域要求
        L_scaled = (2.0 / lambda_max) * L - I

        # 递推计算切比雪夫多项式 T_0, T_1, ... T_{K-1}
        Tx = []
        Tx.append(x)                     # T0(x) = x
        if self.K > 1:
            Tx.append(L_scaled @ x)      # T1(x) = L_scaled * x
        for k in range(2, self.K):
            # 递推公式 Tk = 2*L*T_{k-1} - T_{k-2}
            Tk = 2 * L_scaled @ Tx[-1] - Tx[-2]
            Tx.append(Tk)

        # K阶多项式加权求和
        out = 0.0
        for k in range(self.K):
            out = out + Tx[k] @ self.weight[k]
        out = out + self.bias
        return out

class ChebNetVNE(nn.Module):
    """
    完整ChebNet网络：两层ChebConv + actor/critic头，A3C使用
    actor：输出动作logits，选择底物节点
    critic：输出状态价值V(s)
    """
    def __init__(self, in_dim, hid_dim, act_num, K):
        super().__init__()
        self.K = K
        self.c1 = ChebConv(in_dim, hid_dim, K)
        self.c2 = ChebConv(hid_dim, hid_dim, K)
        self.actor = nn.Linear(hid_dim, act_num)
        self.critic = nn.Linear(hid_dim, 1)

    def forward(self, x, L, lambda_max):
        # 第一层图卷积+激活
        h = F.relu(self.c1(x, L, lambda_max))
        # 第二层图卷积+激活
        h = F.relu(self.c2(h, L, lambda_max))
        # 全局均值池化，所有节点特征取平均，得到图级特征（和论文对齐）
        hg = torch.mean(h, dim=0, keepdim=True)
        logits = self.actor(hg)
        val = self.critic(hg)
        return logits, val
