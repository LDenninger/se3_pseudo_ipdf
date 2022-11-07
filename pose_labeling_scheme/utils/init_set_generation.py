import torch
import pytorch3d.transforms as tt
import math

def generate_init_set_inv(init_rotation, num_init=None):

    device = init_rotation.device

    # 180 degrees around x-axis
    rot_x = -torch.eye(3)
    rot_x[0,0] = -rot_x[0,0]

    # 180 degrees around y-axis
    rot_y = -torch.eye(3)
    rot_y[1,1] = -rot_y[1,1]

    # 180 degrees around z-axis
    rot_z = -torch.eye(3)
    rot_z[2,2] = -rot_z[2,2]

    init_set = torch.stack([rot_x, rot_y, rot_z])

    return init_set

def generate_init_set_r(init_rotation, offset=0):
    lx = []
    ly = []
    lz = []

    xflip = tt.euler_angles_to_matrix(torch.tensor([0,math.pi,0]), "XYZ")
    yflip = tt.euler_angles_to_matrix(torch.tensor([0,0,math.pi]), "XYZ")
    zflip = tt.euler_angles_to_matrix(torch.tensor([math.pi,0,0]), "XYZ")

    offset_x = tt.euler_angles_to_matrix(torch.tensor([offset,0,0]), "XYZ")
    offset_y = tt.euler_angles_to_matrix(torch.tensor([0, offset,0]), "XYZ")
    offset_z = tt.euler_angles_to_matrix(torch.tensor([0,0, offset]), "XYZ")
        
    rot_x = tt.euler_angles_to_matrix(torch.tensor([0.25*math.pi,0,0]), "XYZ")
    lx.append(init_rotation @ rot_x @ offset_x)

    rot_y = tt.euler_angles_to_matrix(torch.tensor([0,0.25*math.pi,0]), "XYZ")
    ly.append(init_rotation @ rot_y @ offset_y)

    rot_z = tt.euler_angles_to_matrix(torch.tensor([0,0,0.25*math.pi]), "XYZ")
    lz.append(init_rotation @ rot_z @ offset_z)

    

    for i in range(6):
        lz.append(lz[-1]@rot_x)
        lz.append(lz[-1]@rot_y)
        lz.append(lz[-1]@rot_z)

    return torch.stack((lx+ly+lz))


def generate_init_set_noise(init_rotation, num_init=5):

    device = init_rotation.device

    init_rotation = init_rotation.float()

    if len(init_rotation.shape)==2:
        init_rotation = init_rotation.unsqueeze(0)
    
    tiny_rotation = (torch.rand((num_init,3))/5 -0.1).float().to(device)
    noise_rotation = tt.euler_angles_to_matrix(tiny_rotation, 'ZYX')
    init_set = torch.bmm(noise_rotation, torch.repeat_interleave(init_rotation, num_init, dim=0))

    init_set = torch.cat([init_rotation, init_set])
    
    return init_set