import random
import numpy as np
import torch
import pytorch3d.ops as ops
import pytorch3d.transforms as tt

## Meta method for setting a unique random seed for all random functions in all libraries ##

def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
