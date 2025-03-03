import random
import torch
import numpy as np

def set_seeds(seed, fully_deterministic=True):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if fully_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)


def is_rct(config):
    if (config.get("patient_function") == "uniform") and (config.get("treatment_function") == "random"):
        return True
    return False