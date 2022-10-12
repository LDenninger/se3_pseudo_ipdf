import os
import numpy as np
import torch
from torch.utils.data import Dataset
import ipdb
import math
import open3d as o3d
import random
import torchvision 
import pytorch3d.transforms as tt
from torch.utils.data import DataLoader
import yaml
from yaml import CLoader as Loader, CDumper as Dumper
from pathlib import Path as P



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224,244)


def load_tabletop_dataset(hyper_param, obj_id, validation_only=False):


    # Validation data
    data_val = TLESSPoseDataset(obj_id=obj_id,
                                train_set=False,
                                ground_truth_mode=0,
                                occlusion=hyper_param['occlusion'],
                                device=DEVICE)
    val_loader = DataLoader(dataset=data_val, batch_size=hyper_param['batch_size_val'], drop_last=True ,shuffle=True, num_workers=4)

    # Training data
    if not validation_only:
        data_train = TLESSPoseDataset(obj_id=obj_id,
                                        train_set=True,
                                        ground_truth_mode=hyper_param['train_mode'],
                                        occlusion=hyper_param['occlusion'],
                                        device=DEVICE)
        train_loader = DataLoader(dataset=data_train, batch_size=hyper_param['batch_size'], drop_last=True,shuffle=True, num_workers=8)
        return train_loader, val_loader

    return val_loader


class TLESSPoseDataset(Dataset):
    def __init__(self, 
                    obj_id,
                    ground_truth_mode=0,
                    gamma=0.6,
                    train_set=True,
                    train_as_test=False,
                    occlusion=False):
        """
        Arguments:
            ground_truth_mode: 0: original GT, 1: pseudo GT
        
        """
        super().__init__()

        if train_set or train_as_test:
            self.data_dir = os.path.join("/home/nfs/inf6/data/datasets/T-Less/t-less_v2/train_kinect", str(obj_id).zfill(2))
            self.len = 1296 # 1296
            if train_as_test:
                self.len = 216 # take every 6th element in the train set
            if ground_truth_mode==0:
                with open(os.path.join(self.data_dir, "gt.yml"), "r") as f:
                    self.ground_truth = yaml.load(f, Loader=Loader)

        else:
            self.data_dir = os.path.join("/home/nfs/inf6/data/datasets/T-Less/t-less_v2/test_kinect/")
            # Extract target frames for the test dataset
            target_frame_set = torch.load(f"/home/nfs/inf6/data/datasets/T-Less/t-less_v2/test_kinect/target_frames/{str(obj_id).zfill(2)}.pth")
            self.target_frames = []
            for target in target_frame_set:
                path = P(target[0])
                frame_id = path.stem
                set_id = path.parent.parent.stem
                target_id = target[1]
                self.target_frames.append({
                    "set_id": set_id,
                    "frame_id": frame_id,
                    "obj_id": target_id
                })
            
            self.len = len(self.target_frames)

            self.set_gt = []
            for set in range(1,21):    
                with open(os.path.join(self.data_dir, str(set).zfill(2),"gt.yml"), "r") as f:
                    self.set_gt.append(yaml.load(f, Loader=Loader))
            
        self.train_as_test = train_as_test
        self.obj_id = obj_id
        self.train_set = train_set
        self.ground_truth_mode = ground_truth_mode
        self.occlusion = occlusion
        self.gamma = gamma

        self.Resizer = torchvision.transforms.Resize(IMG_SIZE)
        self.ResNetTransform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.OccTransform = torchvision.transforms.RandomErasing(p=0.2, scale=(0.1,0.6), inplace=True)

    def __getitem__(self, idx):
        if self.train_set or self.train_as_test:
            if self.train_as_test:
                idx *= 6
            elif not self.train_set and idx%6==0:
                return self.__getitem__((idx+1)%self.__len__())

            data_dir = os.path.join(self.data_dir, "crops", str(idx).zfill(4), "00_quad_crop_mask_resized.pth")
            try:
                image = torch.load(data_dir)
            except:
                return self.__getitem__((idx+1)%self.__len__())

            image_edit = torch.clone(image)
            image_edit = self.apply_augmentation(image_edit)

            ground_truth = self.load_ground_truth(idx)

            if ground_truth is None:
                return self.__getitem__((idx+1)%self.__len__())

            return {
                'image': image_edit,
                'image_raw': image,
                'image_original': image,
                'obj_pose_in_camera': ground_truth
            }

        else:
            frame_info = self.target_frames[idx]
            
            data_dir = os.path.join(self.data_dir, frame_info["set_id"], "crops", frame_info["frame_id"], (str(frame_info["obj_id"]).zfill(2)+"_quad_crop_mask_resized.pth"))
            data_dir_full = os.path.join(self.data_dir, frame_info["set_id"], "rgb", (frame_info["frame_id"]+"_rgb.pth"))
            try:
                image = torch.load(data_dir)
                image_original = torch.load(data_dir_full)
            except:
                return self.__getitem__((idx+1)%self.__len__())
        
            image_edit = torch.clone(image)
            image_edit = self.apply_augmentation(image_edit, validation=True)
            
            ground_truth = self.load_ground_truth(idx)

            if ground_truth is None:
                return self.__getitem__((idx+1)%self.__len__())
            
            return {
                'image': image_edit,
                'image_raw': image,
                'image_original': image_original,
                'obj_pose_in_camera': ground_truth
            }    



    def apply_augmentation(self, image, validation=False):
        # Data augmentation for invariance
        # Random zero padding

        image_edit = image
        """if self.occlusion and not validation:
            image_edit = torchvision.transforms.Pad(np.random.randint(10))(image)"""

        image_edit = self.Resizer(image_edit)
        
        image_edit = torchvision.transforms.functional.adjust_gamma(image_edit, gamma=self.gamma)

        image_edit = self.ResNetTransform(image_edit)
        
        if self.occlusion and not validation:
            image_edit = self.OccTransform(image_edit)
        
        return image_edit



    def load_ground_truth(self, idx):
        if not self.train_set and not self.train_as_test:
            frame_info = self.target_frames[idx]

            gt = torch.eye(4)

            frame_gt = self.set_gt[int(frame_info["set_id"])-1][int(frame_info["frame_id"])]

            for obj in frame_gt:
                if obj['obj_id'] == self.obj_id:
                    gt[:3,:3] = torch.reshape(torch.Tensor(obj['cam_R_m2c']), (3,3))
                    gt[:3,-1] = torch.Tensor(obj['cam_t_m2c'])/1000
            return gt
        if self.ground_truth_mode==0:
            gt = torch.eye(4)
            gt[:3,:3] = torch.reshape(torch.Tensor(self.ground_truth[idx][0]['cam_R_m2c']), (3,3))
            gt[:3,-1] = torch.Tensor(self.ground_truth[idx][0]['cam_t_m2c'])

        elif self.ground_truth_mode==1:
            try:
                pseudo_gt_set  = torch.load(os.path.join(self.data_dir, "pseudo_gt", ("cleaned_"+str(idx).zfill(4)+".pth")))
            except:
                #print(f"No pseudo ground truth found for frame {idx}")
                return None
            gt = pseudo_gt_set[np.random.randint(pseudo_gt_set.shape[0])]

        return gt


    def __len__(self):
        return self.len
