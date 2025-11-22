# audioShieldNet/asnet_1/audioshieldnet/utils/seed.py

import random, numpy as np, torch
def fix_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
