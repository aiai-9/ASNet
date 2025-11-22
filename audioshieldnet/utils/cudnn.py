# audioShieldNet/asnet_1/audioshieldnet/utils/cudnn.py

import torch, torch.backends.cudnn as cudnn
def tune_cudnn():
    cudnn.benchmark = True
    cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass
