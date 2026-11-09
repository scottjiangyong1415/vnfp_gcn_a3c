
# config.py
# 配置文件：统一管理训练、底物网络、VNR生成超参数
import torch

# 训练相关超参
TRAIN = {
    "lr": 1e-4,               # Adam学习率
    "gamma": 0.99,            # 回报折扣因子，未来奖励衰减系数
    "max_episode": 2000,      # 最大训练Episode数量
    "update_freq": 10,        # 收集多少条轨迹后，异步更新全局网络
    "eval_interval": 50,      # 每间隔多少episode打印一次评估信息
    "seed": 123,              # 随机种子，保证实验可复现
    "cheb_K": 3,              # ChebNet切比雪夫多项式阶数，控制感受野大小
    "hidden_dim": 64,         # 图卷积隐藏层维度
    "in_dim": 7               # 底物节点特征维度，和论文状态定义保持一致
}

# 底物物理网络参数
SUB = {
    "node_num": 20,           # 底物网络节点总数
    "cpu_low": 50,            # 节点CPU资源下限
    "cpu_high": 100,          # 节点CPU资源上限
    "bw_low": 50,             # 链路带宽资源下限
    "bw_high": 100            # 链路带宽资源上限
}

# VNR虚拟网络请求生成参数
VNR = {
    "lam": 0.08,              # VNR泊松到达率：单位仿真时间到来的请求数量
    "alpha": 0.4,             # Waxman图生成参数α，控制边生成概率
    "beta": 0.1,              # Waxman图生成参数β，控制距离衰减
    "min_v_node": 2,          # 单个VNR最少虚拟节点数
    "max_v_node": 6,          # 单个VNR最多虚拟节点数
    "cpu_demand_range": [10, 30],  # 虚拟节点CPU需求区间
    "bw_demand_range": [10, 30],   # 虚拟链路带宽需求区间
    "max_queue": 8,           # VNR等待队列最大长度
    "lifetime_mean": 500.0    # VNR平均生存时间，指数分布均值=500（论文原文）
}

# 设备配置
DEVICE = torch.device("cpu")
