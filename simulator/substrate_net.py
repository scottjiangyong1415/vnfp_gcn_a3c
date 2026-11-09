# simulator/substrate_net.py
# 底物网络Gs：管理物理节点CPU、链路带宽、拉普拉斯矩阵、资源分配/释放
import networkx as nx
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigh

class SubstrateNet:
    def __init__(self, cfg):
        self.cfg = cfg                          # 读取环境配置
        self.num_nodes = cfg["substrate_nodes"] # 底物节点数量
        # 生成Waxman拓扑图
        self.G = nx.waxman_graph(self.num_nodes, cfg["waxman_alpha"], cfg["waxman_beta"])
        # Waxman随机图有可能不连通，循环重新生成直到连通
        while not nx.is_connected(self.G):
            self.G = nx.waxman_graph(self.num_nodes, cfg["waxman_alpha"], cfg["waxman_beta"])
        self.node_cpu = {}                      # 节点总CPU容量（固定不变）
        self.node_cpu_remain = {}               # 节点剩余可用CPU（动态变化）
        self.edge_bw = {}                       # 链路总带宽容量（固定不变）
        self.edge_bw_remain = {}                # 链路剩余可用带宽（动态变化）
        self.init_resource()                    # 初始化资源
        self.L = self.get_laplacian()           # 计算图拉普拉斯矩阵（用于谱GCN）

    def init_resource(self):
        # 初始化所有节点CPU资源
        for n in self.G.nodes:
            cap = np.random.randint(self.cfg["substrate_cpu_low"], self.cfg["substrate_cpu_high"]+1)
            self.node_cpu[n] = cap
            self.node_cpu_remain[n] = cap
        # 初始化双向链路带宽，无向图 u-v 和 v-u 视为同一条链路
        for u,v in self.G.edges:
            bw = np.random.randint(self.cfg["substrate_bw_low"], self.cfg["substrate_bw_high"]+1)
            self.edge_bw[(u,v)] = bw
            self.edge_bw[(v,u)] = bw
            self.edge_bw_remain[(u,v)] = bw
            self.edge_bw_remain[(v,u)] = bw

    def get_laplacian(self):
        # 计算图拉普拉斯矩阵 L = D - A
        A = nx.adjacency_matrix(self.G).astype(float) # 邻接矩阵
        deg = np.array(A.sum(axis=1)).flatten()       # 每个节点的度
        D = diags(deg)                                # 度对角矩阵
        L = D - A
        return L.todense()

    def reset(self):
        # 重置底物网络资源，仿真episode重置时调用
        self.init_resource()
        self.L = self.get_laplacian()

    def can_allocate_node(self, n, req_cpu):
        # 判断节点n剩余CPU是否满足本次虚拟节点需求
        return self.node_cpu_remain[n] >= req_cpu

    def alloc_node(self, n, req_cpu):
        # 分配CPU：剩余资源减去需求
        self.node_cpu_remain[n] -= req_cpu

    def free_node(self, n, req_cpu):
        # 释放CPU：VNR到期，资源归还
        self.node_cpu_remain[n] += req_cpu

    def can_allocate_link(self, u, v, req_bw):
        # 判断链路u-v剩余带宽是否满足虚拟链路带宽需求
        return self.edge_bw_remain.get((u,v),0) >= req_bw

    def alloc_link(self, u, v, req_bw):
        # 分配链路带宽
        self.edge_bw_remain[(u,v)] -= req_bw
        self.edge_bw_remain[(v,u)] -= req_bw

    def free_link(self, u, v, req_bw):
        # 释放链路带宽，VNR生命周期结束时调用
        self.edge_bw_remain[(u,v)] += req_bw
        self.edge_bw_remain[(v,u)] += req_bw
