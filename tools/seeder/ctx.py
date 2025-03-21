import torch


class TorchSeedContext:
    def __init__(self, seed):
        self.seed = seed
        self.state = None  # 用于存储当前 RNG 状态

    def __enter__(self):
        self.state = torch.random.get_rng_state()  # 备份当前 RNG 状态
        torch.manual_seed(self.seed)  # 设置新的随机种子

    def __exit__(self, type, value, traceback):
        torch.random.set_rng_state(self.state)  # 恢复原来的 RNG 状态
