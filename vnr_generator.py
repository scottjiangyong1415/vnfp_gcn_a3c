# vnr_generator.py
# VNR生成器：生成Waxman拓扑的虚拟网络请求，包含节点、链路、lifetime（指数分布均值500）
import numpy as np
import networkx as nx

class VNRGenerator:
    def __init__(self, cfg):
        self.alpha = cfg["alpha"]
        self.beta = cfg["beta"]
        self.min_n = cfg["min_v_node"]
        self.max_n = cfg["max_v_node"]
        self.cpu_r = cfg["cpu_demand_range"]
        self.bw_r = cfg["bw_demand_range"]
        self.lifetime_mean = cfg["lifetime_mean"] # VNR平均生存时间

    def generate(self):
        """生成一条完整VNR请求"""
        # 随机选择该VNR的虚拟节点数量
        n = np.random.randint(self.min_n, self.max_n+1)
        # 生成Waxman图作为虚拟网络拓扑
        g = nx.waxman_graph(n, alpha=self.alpha, beta=self.beta)

        # 生成每个虚拟节点的CPU需求
        v_nodes = []
        for _ in range(n):
            c = np.random.uniform(self.cpu_r[0], self.cpu_r[1])
            v_nodes.append({"cpu": c})

        # 生成每条虚拟链路的带宽需求
        v_links = {}
        for u, v in g.edges():
            b = np.random.uniform(self.bw_r[0], self.bw_r[1])
            v_links[(u, v)] = b

        # 指数分布生成生存时间，scale=均值500，论文要求
        lifetime = np.random.exponential(self.lifetime_mean)

        return {
            "nodes": v_nodes,
            "links": v_links,
            "v_graph": g,
            "lifetime": lifetime,
            "arrival_time": 0.0
        }
