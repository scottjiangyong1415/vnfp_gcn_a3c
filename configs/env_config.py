# configs/env_config.py
# VNE仿真环境参数：底物网络、虚拟网络请求VNR、流量、拓扑、生存时间
ENV = {
    "substrate_nodes": 15,          # 底物物理网络节点总数
    "substrate_cpu_low": 50,        # 底物节点CPU资源下限
    "substrate_cpu_high": 100,      # 底物节点CPU资源上限
    "substrate_bw_low": 50,         # 底物链路带宽下限
    "substrate_bw_high": 100,       # 底物链路带宽上限
    "vnr_min_nodes": 2,             # 单个VNR最少虚拟节点数
    "vnr_max_nodes": 6,             # 单个VNR最多虚拟节点数
    "vnr_cpu_low": 5,               # VNR虚拟节点CPU需求下限
    "vnr_cpu_high": 30,             # VNR虚拟节点CPU需求上限
    "vnr_bw_low": 5,                # VNR虚拟链路带宽需求下限
    "vnr_bw_high": 30,              # VNR虚拟链路带宽需求上限
    "lam": 0.2,                     # 泊松到达率λ：每个仿真时间步平均到达VNR数量
    "max_vnr_queue": 10,            # VNR等待队列最大长度；队列满，新到达请求直接丢弃
    "waxman_alpha": 0.4,            # Waxman图α：基础连边概率上限，控制拓扑稠密程度
    "waxman_beta": 0.1,             # Waxman图β：距离衰减系数，远距离节点连边概率衰减快慢
    "lifetime_mean":
