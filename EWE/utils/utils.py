import torch
import numpy as np
import random

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # 用于保证CUDA 卷积运算的结果确定
    torch.backends.cudnn.benchmark = False      # 用于保证数据变化的情况下，减少网络效率的变化