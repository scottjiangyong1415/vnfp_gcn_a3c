# main.py
# 程序入口：初始化全局模型、创建多worker线程，启动训练
import torch
import threading
import numpy as np
from config import TRAIN, SUB, VNR
from chebnet import ChebNetVNE
from a3c import A3CWorker

if __name__ == "__main__":
    torch.manual_seed(TRAIN["seed"])
    np.random.seed(TRAIN["seed"])
    # 临时环境实例，获取动作空间大小
    env_tmp = VNEEnv(SUB, VNR)
    act_num = env_tmp.N_sub

    # 全局共享模型
    global_model = ChebNetVNE(
        in_dim=TRAIN["in_dim"],
        hid_dim=TRAIN["hidden_dim"],
        act_num=act_num,
        K=TRAIN["cheb_K"]
    )
    global_opt = torch.optim.Adam(global_model.parameters(), lr=TRAIN["lr"])
    lock = threading.Lock()

    # 启动worker
    worker_num = 2
    workers = []
    for i in range(worker_num):
        w = A3CWorker(i, TRAIN, SUB, VNR, global_model, global_opt, lock)
        workers.append(w)
        w.start()
    # 等待所有worker训练结束
    for w in workers:
        w.join()
    print("Training finished")
