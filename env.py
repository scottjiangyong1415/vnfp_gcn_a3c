# env.py
# VNE强化学习环境：状态构建、动作执行、节点嵌入、链路嵌入、资源超时回收
import numpy as np
import networkx as nx
from substrate import Substrate
from vnr_generator import VNRGenerator

class VNEEnv:
    def __init__(self, sub_cfg, vnr_cfg):
        # 初始化底物网络
        self.sub = Substrate(sub_cfg)
        # 初始化VNR生成器
        self.vnr_gen = VNRGenerator(vnr_cfg)
        self.sub_G = self.sub.G
        self.N_sub = sub_cfg["node_num"]
        self.max_q = vnr_cfg["max_queue"]

        self.vnr_queue = []          # VNR等待队列：已经到达、还未开始嵌入的请求
        self.current_vnr = None      # 当前正在嵌入的VNR
        self.v_ptr = 0               # 当前待映射的虚拟节点下标，逐个映射
        self.mapping = dict()        # 映射字典：虚拟节点id -> 底物节点id

        self.accepted = 0            # 累计成功嵌入VNR数量
        self.rejected = 0            # 累计被拒绝VNR数量

        self.sim_time = 0.0          # 全局仿真时钟，每一步+1
        # 已成功嵌入的VNR列表，用于到期回收资源
        # 每个元素：{"mapping":映射表, "expire_time":到期时刻, "vnr":原始vnr数据}
        self.embedded_vnr_list = []

    def get_spectral(self):
        """向外提供ChebNet需要的谱信息：L矩阵、最大特征值"""
        return self.sub.get_spectral_info()

    def reset(self):
        """episode重置：清空队列、资源、仿真时钟、统计量"""
        self.sub.reset()
        self.vnr_queue.clear()
        self.embedded_vnr_list.clear()
        self.accepted = 0
        self.rejected = 0
        self.sim_time = 0.0
        self._fill_queue()
        self._load_next_vnr()
        return self._state()

    def _fill_queue(self):
        """填充VNR等待队列，维持队列长度不小于上限"""
        while len(self.vnr_queue) < self.max_q:
            vnr = self.vnr_gen.generate()
            vnr["arrival_time"] = self.sim_time
            self.vnr_queue.append(vnr)

    def _load_next_vnr(self):
        """从等待队列取出下一个VNR作为当前待嵌入请求"""
        if len(self.vnr_queue) == 0:
            self._fill_queue()
        self.current_vnr = self.vnr_queue.pop(0)
        self.v_ptr = 0
        self.mapping.clear()

    def _release_expired(self):
        """资源回收核心函数：遍历所有已嵌入VNR，判断是否到期，到期归还CPU、带宽"""
        remain_list = []
        for item in self.embedded_vnr_list:
            # 判断当前仿真时间是否超过到期时间
            if self.sim_time >= item["expire_time"]:
                # 回收节点CPU
                for v, sub_n in item["mapping"].items():
                    req_cpu = item["vnr"]["nodes"][v]["cpu"]
                    self.sub.release_cpu(sub_n, req_cpu)
                # 回收链路带宽：重新查找当时嵌入的最短路径，释放带宽
                for (u, v), bw_req in item["vnr"]["links"].items():
                    su = item["mapping"][u]
                    sv = item["mapping"][v]
                    try:
                        path = nx.shortest_path(self.sub_G, su, sv)
                        for i in range(len(path)-1):
                            a = path[i]
                            b = path[i+1]
                            self.sub.release_bw(a, b, bw_req)
                    except:
                        pass
            else:
                # 未到期，保留在列表中
                remain_list.append(item)
        self.embedded_vnr_list = remain_list

    def _state(self):
        """构造状态观测矩阵 N_sub × 7，和论文状态定义对齐"""
        n = self.N_sub
        feat = np.zeros((n, 7), dtype=np.float32)
        for i in range(n):
            feat[i, 0] = self.sub.cpu[i]                     # 底物节点剩余CPU
            feat[i, 1] = sum(self.sub.bw.get((i,j),0) for j in range(n)) # 节点邻接剩余带宽之和
            feat[i, 2] = self.current_vnr["nodes"][self.v_ptr]["cpu"]    # 当前待映射虚拟节点CPU需求
            feat[i, 3:7] = 0.0
        return feat

    def _link_embed(self):
        """链路嵌入策略：论文采用最短路径嵌入虚拟链路"""
        vg = self.current_vnr["v_graph"]
        for (u, v), bw_req in self.current_vnr["links"].items():
            su = self.mapping[u]
            sv = self.mapping[v]
            try:
                path = nx.shortest_path(self.sub_G, su, sv)
            except nx.NetworkXNoPath:
                return False
            # 依次占用路径上每条链路带宽
            for i in range(len(path)-1):
                a = path[i]
                b = path[i+1]
                if not self.sub.consume_bw(a,b,bw_req):
                    return False
        return True

    def step(self, action):
        """
        环境步进函数
        action: agent选择的底物节点编号，用来映射当前虚拟节点v_ptr
        return: next_obs, reward, done, info
        """
        reward = 0.0
        done = False
        vnr = self.current_vnr
        v_req = vnr["nodes"][self.v_ptr]["cpu"]

        # 1. 节点资源不足，映射失败，拒绝该VNR
        if not self.sub.consume_cpu(action, v_req):
            reward = -1.0
            self.rejected += 1
            self._load_next_vnr()
            next_obs = self._state()
            return next_obs, reward, done, {"accept": False}

        # 2. 节点资源充足，完成当前虚拟节点映射
        self.mapping[self.v_ptr] = action
        self.v_ptr += 1

        # 判断：是否所有虚拟节点都映射完成
        if self.v_ptr >= len(vnr["nodes"]):
            # 全部节点映射完成，执行链路嵌入
            link_ok = self._link_embed()
            if link_ok:
                reward = 2.0
                self.accepted += 1
                # 嵌入成功，计算到期时间，加入已嵌入列表
                expire_t = self.sim_time + vnr["lifetime"]
                self.embedded_vnr_list.append({
                    "mapping": self.mapping.copy(),
                    "expire_time": expire_t,
                    "vnr": vnr
                })
            else:
                # 节点映射成功、链路嵌入失败，VNR依然拒绝
                reward = -0.5
                self.rejected += 1
            # 加载下一条VNR
            self._load_next_vnr()
        else:
            # 只是单个虚拟节点映射成功，还需要继续映射其他虚拟节点，小奖励
            reward = 0.1

        # 仿真时间向前推进1单位
        self.sim_time += 1.0
        # 检查并释放到期VNR资源
        self._release_expired()
        # 补充VNR队列
        self._fill_queue()
        next_obs = self._state()
        return next_obs, reward, done, {"accept": link_ok if self.v_ptr == 0 else None}
