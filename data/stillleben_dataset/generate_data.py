import numpy as np
import torch
import pytorch3d.transforms as tt
from scipy.spatial.transform import Rotation as R


def generate_dataset(file_name=None, mode=0):
    if mode==0:
        dataset_poses = _gen_method_01()
    if mode==1:
        dataset_poses = _gen_method_random()
    if file_name is not None:
        try:
            torch.save(dataset_poses, file_name)
        except:
            print(f"Dataset could not been saved to: {file_name}")

        print(f"Dataset was saved to: {file_name}")
    return dataset_poses

def _gen_method_random():
    random_rotations = R.random(20000)
    random_rotations = torch.from_numpy(random_rotations.as_matrix())

    pose_set = torch.repeat_interleave(torch.eye(4).unsqueeze(0), random_rotations.shape[0], dim=0)
    pose_set[:,:3,:3] = random_rotations


    return pose_set

def _gen_method_01():

    num = 100

    init_rotation = torch.eye(3)


    rot_x = _gen_x_revolt(num=num)
    rot_y = _gen_y_revolt(base_rotation = rot_x[-1], num=num)
    rot_z = _gen_z_revolt(base_rotation = rot_y[-1],num=num)

    rotation_set = torch.cat([rot_x, rot_y, rot_z])
    pose_set = torch.repeat_interleave(torch.eye(4).unsqueeze(0), rotation_set.shape[0], dim=0)

    pose_set[:,:3,:3] = rotation_set

    return pose_set


def _gen_x_revolt(base_rotation = torch.eye(3), num=0):

    rotations = [base_rotation]

    rotations_step = tt.euler_angles_to_matrix(torch.tensor([2*np.pi/num,0,0]), 'XYZ').float()

    for i in range(num):
        rotations.append(rotations_step @ rotations[-1])
    
    return torch.stack(rotations)

def _gen_y_revolt(base_rotation = torch.eye(3), num=0):

    rotations = [base_rotation]

    rotations_step = tt.euler_angles_to_matrix(torch.tensor([0, 2*np.pi/num, 0]), 'XYZ').float()

    for i in range(num):
        rotations.append(rotations_step @ rotations[-1])
    
    return torch.stack(rotations)

def _gen_z_revolt(base_rotation = torch.eye(3), num=0):

    rotations = [base_rotation]

    rotations_step = tt.euler_angles_to_matrix(torch.tensor([0,0, 2*np.pi/num]), 'XYZ').float()

    for i in range(num):
        rotations.append(rotations_step @ rotations[-1])
    
    return torch.stack(rotations)