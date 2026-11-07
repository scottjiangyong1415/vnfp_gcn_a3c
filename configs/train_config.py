import torch

TRAIN = {
    "lr": 1e-4,
    "gamma": 0.99,
    "tau": 1.0,
    "beta_entropy": 0.01,
    "num_workers": 4,
    "max_ep": 3000,
    "update_global_freq": 5,
    "gcn_k": 3,
    "hidden_dim": 64,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

REWARD = {
    "alpha_r": 1.0,
    "alpha_c": 0.3,
    "alpha_s": 0.1
}
