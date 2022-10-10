import numpy as np
import torch

def generate_cartesian_grid(range, size=None, step=None):
    if step is None:
        p_per_dim= np.ceil(np.cbrt(size))
        step_x = (range[0,1]-range[0,0])/p_per_dim
        step_y = (range[1,1]-range[1,0])/p_per_dim
        step_z = (range[2,1]-range[2,0])/p_per_dim
    else:
        step_x=step_y=step_z=step
    x = torch.arange(range[0,0], range[0,1], step_x)
    y = torch.arange(range[1,0], range[1,1], step_y) 
    z = torch.arange(range[2,0], range[2,1], step_z)

    grid = torch.cartesian_prod(x, y, z)

    return grid
