# substrate.py
# 底物网络类：维护物理节点CPU、链路带宽资源，计算归一化拉普拉斯矩阵，供ChebNet使用
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigh

class Substrate:
    def __init__(self, cfg):
        # 读取底物网络节点数量
        self.N = cfg["node_num"]
        # 生成随机几何图作为底物拓扑
        self.G = nx.random_geometric_graph(self.N, radius=0.4)

        # 初始化节点CPU资源，同时保存原始资源，用于reset重置
        self.cpu = np.random.randint(cfg["cpu_low"], cfg["cpu_high"], size=self.N).astype(np.float32)
        self.cpu_orig = self.cpu.copy()

        # 初始化链路带宽字典，双向链路带宽相等，保存原始带宽
        self.bw = dict()
        self.bw_orig = dict()
        for u, v in self.G.edges():
            b = np.random.randint(cfg["bw_low"], cfg["bw_high"])
            self.bw[(u, v)] = float(b)
            self.bw[(v, u)] = float(b)
            self.bw_orig[(u, v)] = float(b)
            self.bw_orig[(v, u)] = float(b)

        # 构造邻接矩阵A
        A = nx.adjacency_matrix(self.G).todense().astype(np.float32)
        # 计算每个节点的度
        deg = np.sum(A, axis=1).A1
        # D^{-1/2}，加极小值防止除0
        deg_sqrt_inv = np.diag(1.0 / (np.sqrt(deg)+1e-8))
        I = np.eye(self.N)
        # 归一化对称拉普拉斯矩阵 L_sym = I - D^{-1/2} A D^{-1/2}
        L_sym = I - deg_sqrt_inv @ A @ deg_sqrt_inv
        self.L = L_sym.astype(np.float32)

        # ========= ChebNet需要：对拉普拉斯做特征分解 =========
        # eigh专门用于对称矩阵特征分解，返回特征值、特征向量
        eigvals, eigvecs = eigh(self.L)
        self.eigvals = eigvals
        self.eigvecs = eigvecs
        self.lambda_max = float(eigvals.max())  # 拉普拉斯最大特征值，用于缩放切比雪夫多项式

    def reset(self):
        """重置底物资源，回到初始状态，episode开始时调用"""
        self.cpu = self.cpu_orig.copy()
        for k in self.bw:
            self.bw[k] = self.bw_orig[k]

    def consume_cpu(self, n, req):
        """消耗节点CPU资源：n底物节点编号，req需求CPU
        返回True代表资源充足，分配成功；False资源不足
        """
        if self.cpu[n] >= req:
            self.cpu[n] -= req
            return True
        return False

    def release_cpu(self, n, req):
        """VNR到期，释放节点CPU资源"""
        self.cpu[n] += req

    def consume_bw(self, u, v, req):
        """消耗u-v链路带宽资源"""
        if self.bw.get((u, v), 0) >= req:
            self.bw[(u, v)] -= req
            return True
        return False

    def release_bw(self, u, v, req):
        """VNR到期，释放链路带宽资源"""
        self.bw[(u, v)] += req

    def get_spectral_info(self):
        """对外接口：返回拉普拉斯矩阵L和最大特征值lambda_max，送入ChebNet"""
        return self.L, self.lambda_max
