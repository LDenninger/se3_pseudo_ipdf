import os
import torch
from torch.utils.data import Dataset
import torchvision 
import yaml

import data


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TLESSWorkDataset(Dataset):
    def __init__(self, config, start=0):
        
        super().__init__()

        self.data_dir = os.path.join("/home/nfs/inf6/data/datasets/T-Less/t-less_v2/train_kinect", str(config["obj_id"]).zfill(2))
        self.pseudo_save_dir = os.path.join("/home/nfs/inf6/data/datasets/T-Less/t-less_v2/train_kinect", str(config["obj_id"]).zfill(2), "pseudo_gt")
        self.len = 1296 if start==0 else 1296-start
        self.start = start
        self.config=config


        with open(os.path.join(self.data_dir, "gt.yml"), "r") as f:
            self.ground_truth = yaml.safe_load(f)
        self.obj_id = config["obj_id"]

        with open(os.path.join(self.data_dir, "info.yml"), "r") as f:
            self.meta_info = yaml.safe_load(f)




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

        return {
                "image": image,
                "seg_image": seg_data,
                "depth_image": depth_data,
                "intrinsic": intrinsic,
                "loaded": loaded
            }

    def __len__(self):
        return self.len

