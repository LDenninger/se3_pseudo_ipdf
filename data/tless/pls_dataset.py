import os
import numpy as np
import torch
from torch.utils.data import Dataset
import data
import ipdb
import math
import open3d as o3d
import random
import torchvision 
import pytorch3d.transforms as tt
from torch.utils.data import DataLoader
import yaml
import utils
import viz
import json
import registration


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TLESSWorkDataset(Dataset):
    def __init__(self, config, start=0, mode=0):
        
        super().__init__()

        self.data_dir = os.path.join("/home/nfs/inf6/data/datasets/T-Less/t-less_v2/train_kinect", str(config["obj_id"]).zfill(2))
        self.pseudo_save_dir = os.path.join("/home/nfs/inf6/data/datasets/T-Less/t-less_v2/train_kinect", str(config["obj_id"]).zfill(2), "pseudo_gt")
        self.len = 1296 if start==0 else 1296-start
        self.mode = mode
        self.start = start
        self.config=config


        with open(os.path.join(self.data_dir, "gt.yml"), "r") as f:
            self.ground_truth = yaml.safe_load(f)
        self.mesh_data, self.diameter = utils.load_tless_object_model(config["obj_id"],model_type=2)
        self.obj_id = config["obj_id"]

        with open(os.path.join(self.data_dir, "info.yml"), "r") as f:
            self.meta_info = yaml.safe_load(f)

        self.obj_model_sl = utils.load_cad_models(config["obj_id"])



    def __getitem__(self, idx):
        """Adjust the data of the given frame idx

        Modes:
            0: Generate pseudo gt
            1: Load and return the given data for each frame
        
        
        """
        zfill_len = 4
        idx += self.start
        frame_id = str(idx).zfill(zfill_len)
        data_dir = os.path.join(self.data_dir,"rgb", (frame_id+"_rgb.pth"))#
        depth_dir = os.path.join(self.data_dir,"depth_clean", (frame_id+".pth"))
        seg_dir = os.path.join(self.data_dir, "seg", (frame_id+".pth")) #"00_"+
        if self.config["skip"] and os.path.exists(os.path.join(self.pseudo_save_dir, (frame_id+".pth"))):
            return 1
        try:
            image = torch.load(data_dir)
            seg_data = torch.load(seg_dir)
            depth_data = torch.load(depth_dir)
            loaded = True

        except:
            image =  -torch.eye(4)
            seg_data =  -torch.eye(4)
            depth_data =  -torch.eye(4)
            loaded=False

        if self.config["verbose"]:

            torchvision.utils.save_image(image / 255., "output/tless_2/org_img.png")
            torchvision.utils.save_image(torch.clip(seg_data, 0, 1).float(), "output/tless_2/seg_image.png")
            torchvision.utils.save_image(depth_data/1000, "output/tless_2/depth_image.png")

        K = self.meta_info[idx]['cam_K']
        intrinsic = torch.tensor([K[0], K[4], K[2], K[5]]) # (fx, fy, cx, cy)

        if self.mode==0:
            if not loaded:
                return 0
            pseudo_ground_truth = registration.pseudo_labeling_scheme(pts_canonical=self.mesh_data[0],
                                                                        seg_data=seg_data,
                                                                        depth_data=depth_data,
                                                                        obj_id=self.config["obj_id"],
                                                                        diameter = self.diameter,
                                                                        intrinsic=intrinsic,
                                                                        obj_model_sl=self.obj_model_sl)
            if pseudo_ground_truth is None:
                return -1
            if not self.config["verbose"]:
                self.save_pseudo_ground_truth(pseudo_ground_truth.cpu(), frame_id)

            return 1
        if self.mode==1:

            if loaded:
                try:
                    pseudo_gt_dir = os.path.join(self.data_dir,"pseudo_gt", (frame_id+ ".pth"))
                    pseudo_gt = torch.load(pseudo_gt_dir)
                    loaded = True
                except:
                    pseudo_gt = -torch.eye(4)
                    loaded=False
            return {
                "image": image,
                "pseudo_gt": pseudo_gt,
                "seg_image": seg_data,
                "depth_image": depth_data,
                "intrinsic": intrinsic,
                "loaded": loaded
            }


    def save_pseudo_ground_truth(self, pseudo_ground_truth_set, frame_id):
        save_dir = os.path.join(self.pseudo_save_dir, (frame_id+".pth"))
        if os.path.exists(save_dir):
            pgt_exist = torch.load(save_dir)
            pseudo_ground_truth_set = torch.cat((pgt_exist, pseudo_ground_truth_set))
        torch.save(pseudo_ground_truth_set, save_dir)



    def load_ground_truth(self, idx):
        if self.ground_truth_mode==0:
            gt = torch.eye(4)
            rotation = self.ground_truth[idx][0]['cam_R_m2c']
            gt[:3,:3] = torch.reshape(torch.FloatTensor(rotation), (3,3))
            gt[:3,-1] = torch.FloatTensor(self.ground_truth[idx][0]['cam_t_m2c'])

        
        return gt

    def __len__(self):
        return self.len

