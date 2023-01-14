import os
import json
import numpy as np
from math import pi
import random
import torchvision
import torch
from torch.utils.data import Dataset
from pytorch3d.transforms import euler_angles_to_matrix
from .renderer import Renderer
from PIL import Image
import pytorch3d.transforms as tt
import math
import ipdb
from scipy.spatial.transform import Rotation as R

class YCBPoseDataset(Dataset):
    def __init__(self, obj_id, img_size=(224, 224),
                poses=None,
                translation=False,
                remove_texture=False):
        super().__init__()

        self.img_size = img_size
        self.obj_id = obj_id

        self.ResNetTransform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.renderer = Renderer(
                                    obj_id = obj_id,
                                    img_size = img_size,
        )

        self.poses = poses
        self.gamma = 0.12
        self.translation = translation


    def __getitem__(self, idx):
        if self.poses is not None:
            pose = self.poses[idx]
        else:
            rotation = torch.from_numpy(R.random().as_matrix())
            if self.translation==True:
                x, y, z = [random.uniform(-0.1,0.1) for i in range(3)]
            else:
                x=y=z = 0

            translation = torch.tensor([x, y, z]).float() 
            pose = torch.eye(4)
            pose[:3,:3] = rotation
            pose[:3,-1] = translation

        self.renderer.set_obj_pose(pose)
        img = self.renderer.render()
        img = img.permute(2, 0, 1).float()/255.

        ground_truth = self.produce_ground_truth_analytical(self.renderer.get_object_pose_in_camera())

        image_edit = img.clone()
        image_edit = torchvision.transforms.functional.adjust_gamma(image_edit, gamma=self.gamma)
        image_edit = self.ResNetTransform(image_edit)

        return {
            'image': image_edit,
            'image_raw': img,
            'image_original': img,
            'obj_pose_in_camera': ground_truth,
            'obj_id': self.obj_id
        }

    def produce_ground_truth_analytical(self, ground_truth):
        if self.obj_id == 3:
            ground_truth = ground_truth.float()
            rot_mag = np.random.uniform(0,2*math.pi)
            rotation = tt.euler_angles_to_matrix(torch.tensor([rot_mag,0,0]), 'ZYX').float()
            flip_random = np.random.randint(4)
            if flip_random == 0 or flip_random == 1:
                flip = torch.eye(3).float()
            elif flip_random == 2:
                flip = tt.euler_angles_to_matrix(torch.tensor([0, np.pi, 0]), 'ZYX').float()
            else:
                flip = tt.euler_angles_to_matrix(torch.tensor([0, 0, np.pi]), 'ZYX').float()
            ground_truth[:3,:3] = ground_truth[:3,:3] @ (flip @ rotation)
        
        if self.obj_id == 4:
            """ground_truth = ground_truth.float()
            rot_mag = math.pi/2
            rot_x = tt.euler_angles_to_matrix(torch.tensor([rot_mag,0,0]), 'ZYX').float()
            rot_y = tt.euler_angles_to_matrix(torch.tensor([0, rot_mag, 0]), 'ZYX').float()
            rot_z = tt.euler_angles_to_matrix(torch.tensor([0,0, rot_mag]), 'ZYX').float()
            rot = torch.stack([rot_x, rot_y, rot_z])
            ground_truth [:3,:3] = ground_truth[:3,:3] @ rot[np.random.randint(3)]"""
            syms = torch.zeros(4,3,3)
            syms[0] = torch.eye(3)
            syms[1] = tt.euler_angles_to_matrix(torch.Tensor([0,0, np.pi]), 'ZYX').float()
            syms[2] = tt.euler_angles_to_matrix(torch.Tensor([0,np.pi,0 ]), 'ZYX').float()
            syms[3] = syms[1] @ syms[2]
            ground_truth[:3,:3] = ground_truth[:3,:3].double() @ syms[np.random.randint(syms.shape[0])].double()

        
        if self.obj_id == 5:
            ground_truth=ground_truth.float()
            rot_mag = np.random.uniform(0,2*math.pi)
            rotation = tt.euler_angles_to_matrix(torch.tensor([rot_mag,0,0]), 'ZYX').float()
            ground_truth[:3,:3] = ground_truth[:3,:3] @ rotation
        
        return ground_truth

    def __len__(self):
        
        return self.poses.shape[0] if self.poses is not None else 20000
