import json
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import data
import ipdb
import math
import open3d as o3d
import random
import utils
import torchvision
import data
import registration

VERBOSE = True


# img size: (200,400)

class TabletopWorkDataset(Dataset):
    OBJ_ID = 3
    BB_SIZE = (560,560)
    def __init__(self, data_dir, length = 50000, save_dir=None, mesh_data=None, diameter=0,
                verbose=False,
                device="cpu"):
        """
        Dataloader for the RGBD dataset to work on the dataset using different modes:

        Arguments:
            start: Start index of the interval of images to use from the dataset
            end: End index of the interval of images to use from the dataset
            mode: Defines the mode of the dataset which determines the actions taken on the dataset:
                0: Dataset is initialized to generate pseudo ground truths
                1: Dataset is initialized to return the pseudo ground truths and images used for training
        """
        super().__init__()

        self.data_dir = data_dir
        self.device = device
        self.verbose = VERBOSE
        self.mesh_data = mesh_data
        self.diameter = diameter

        self.meta_info = load_meta_info(data_dir)
        self.obj_id = self.meta_info[2]['OBJECT_ID']

        self.obj_model_sl = utils.load_cad_models(self.obj_id)
        #self.obj_model_sl = None
        
        self.len = length


    def __getitem__(self, idx):
        # Define the frame from the given index
        frame_id = str(idx).zfill(6)
        data_frame_dir = os.path.join(self.data_dir, frame_id)
        pseudo_gt_dir = os.path.join(data_frame_dir, "pseudo_gt.pth")

        # Load the data needed by the pose labeling scheme
        try:
            image = torch.load(os.path.join(data_frame_dir,"rgb_tensor.pt"))
            seg_data = torch.load(os.path.join(data_frame_dir, "seg_data.pt"))
            depth_data = torch.load(os.path.join(data_frame_dir, "seg_data.pt"))
        except:    
            try:
                meta_data = torch.load(os.path.join(data_frame_dir, "meta_data.pt"))
                image = torch.from_numpy(meta_data['rgb_tensor'][...,:3])
                seg_data = torch.from_numpy(meta_data['seg_tensor'].astype("int32"))
                depth_data = torch.from_numpy(meta_data['depth_tensor'])
            except:
                print(f"Data for frame {idx} could not been loaded!")
                return 0
        if VERBOSE:
            torchvision.utils.save_image(image.permute(2,0,1)/255., "output/tabletop/org_image.png")
            torchvision.utils.save_image(depth_data.unsqueeze(0), "output/tabletop/depth_image.png")

        seg_mask = (seg_data==self.obj_id).int()
        depth_data = depth_data * seg_mask


        intrinsic = torch.tensor([2/self.meta_info[0][0,0], 2/self.meta_info[0][1,1],image.shape[1]/2, image.shape[0]/2])# (fx, fy, cx, cy)
        projection_matrix = self.meta_info[0]

        # Start the pseudo ground truth generation
        pseudo_gt_saved = None
        if os.path.exists(pseudo_gt_dir):
            pseudo_gt_saved = torch.load(pseudo_gt_dir)

        # Generate pseudo ground truth and save them
        pseudo_ground_truth = registration.pseudo_labeling_scheme(pts_canonical=self.mesh_data[0],
                                                                    seg_data=seg_data,
                                                                    depth_data=depth_data,
                                                                    obj_id=self.obj_id,
                                                                    diameter = self.diameter,
                                                                    intrinsic=intrinsic,
                                                                    projection_matrix=projection_matrix,
                                                                    obj_model_sl=self.obj_model_sl,
                                                                    device=self.device)
        if pseudo_ground_truth is None:
            return -1
        if not self.verbose:
            self.save_pseudo_ground_truth(pseudo_ground_truth.cpu(), frame_id)

        return 1

    def save_pseudo_ground_truth(self, pseudo_ground_truth_set, frame_id):
        save_dir = os.path.join(self.data_dir, frame_id, "pseudo_gt.pth")
        if os.path.exists(save_dir):
            pgt_exist = torch.load(save_dir)
            pseudo_ground_truth_set = torch.cat((pgt_exist, pseudo_ground_truth_set))
        torch.save(pseudo_ground_truth_set, save_dir)
            



    def __len__(self):
        return self.len

def load_meta_info(data_dir):
    meta_data = torch.load(os.path.join(data_dir, "000000", "meta_data.pt"))

    # Assumption that the camera calibration is consistent for all frames
    projection_matrix = meta_data['projection_matrix']
    view_matrix = meta_data['view_matrix']

    seg_id = meta_data['seg_ids']

    return projection_matrix, view_matrix, seg_id
