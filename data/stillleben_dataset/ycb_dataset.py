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
import ipdb
from scipy.spatial.transform import Rotation as R

class YCBPoseDataset(Dataset):
    def __init__(self, obj_id, img_size=(224, 224),
                poses=None,
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
            pose[:,:3,-1] = translation

        self.renderer.set_obj_pose(pose)
        img = self.renderer.render()
        img = img.permute(2, 0, 1).float()/255.

        image_edit = img.clone()
        image_edit = torchvision.transforms.functional.adjust_gamma(image_edit, gamma=self.gamma)
        image_edit = self.ResNetTransform(image_edit)

        return {
            'image': image_edit,
            'image_raw': img,
            'image_original': img,
            'obj_pose_in_camera': self.renderer.get_object_pose_in_camera(),
            'obj_id': self.obj_id
        }

    def __len__(self):
        
        return self.poses.shape[0] if self.poses is not None else 20000
