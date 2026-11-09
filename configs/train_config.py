# configs/train_config.py
# 训练相关超参数配置文件：A3C、GCN、奖励权重
import torch

TRAIN = {
    "lr": 1e-4,                # 学习率：Adam优化器步长，控制参数更新幅度
    "gamma": 0.99,             # 折扣因子 γ：未来奖励的衰减系数；越接近1越看重长期收益
    "tau": 1.0,                # A3C异步更新系数，这里直接完整同步全局网络
    "beta_entropy": 0.01,      # 熵正则系数：鼓励策略探索，避免过早收敛到确定性动作
    "num_workers": 4,          # A3C并行worker数量，多进程并行采样轨迹
    "max_ep": 3000,            # 最大训练episode总数
    "update_global_freq": 5,   # 每收集多少步轨迹，worker向全局网络推送梯度
    "gcn_k": 3,                # ChebNet切比雪夫多项式阶数 K=3，对应论文3阶谱GCN
    "hidden_dim": 64,          # GCN输出隐藏特征维度
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu") # 自动选择GPU/CPU
}

REWARD = {
    "alpha_r": 1.0,            # 收益项权重
    "alpha_c": 0.3,            # 资源开销项权重
    "alpha_s": 0.1             # 资源均衡方差项权重
}
