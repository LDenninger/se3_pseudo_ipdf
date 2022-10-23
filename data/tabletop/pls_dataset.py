import os
import torch
from torch.utils.data import Dataset
import torchvision

import data

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# img size: (200,400)

class TabletopWorkDataset(Dataset):
    OBJ_ID = 3
    BB_SIZE = (560,560)
    def __init__(self, config, return_pgt=False, cleaned_pgt=True, return_gt=False, start=0, end=20000):
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

        self.data_dir = data.id_to_path[config["obj_id"]]
        
        self.config = config

        self.meta_info = load_meta_info(self.data_dir)
        self.obj_id = self.meta_info[2]['OBJECT_ID']
        
        self.len = end-start
        self.start = start

        self.return_pgt = return_pgt
        self.cleaned_pgt = cleaned_pgt
        self.return_gt = return_gt

    def __getitem__(self, idx):
        # Define the frame from the given index
        frame_id = str(idx).zfill(6)
        data_frame_dir = os.path.join(self.data_dir, frame_id)
        # Load the data needed by the pose labeling scheme
 
        try:
            image = torch.load(os.path.join(data_frame_dir,"rgb_tensor.pt"))
            seg_data = torch.load(os.path.join(data_frame_dir, "seg_tensor.pt"))
            depth_data = torch.load(os.path.join(data_frame_dir, "depth_tensor.pt"))
            loaded=True
        except:    
            try:
                meta_data = torch.load(os.path.join(data_frame_dir, "meta_data.pt"))
                image = torch.from_numpy(meta_data['rgb_tensor'][...,:3])
                seg_data = torch.from_numpy(meta_data['seg_tensor'].astype("int32"))
                depth_data = torch.from_numpy(meta_data['depth_tensor'])
                loaded=True
            except:
                image =  -torch.eye(4)
                seg_data =  -torch.eye(4)
                depth_data =  -torch.eye(4)
                print(f"Data for frame {idx} could not been loaded!")
                loaded=False
                
        if self.config["verbose"]:
            torchvision.utils.save_image(image.permute(2,0,1)/255., "output/pose_labeling_scheme/org_image.png")
            torchvision.utils.save_image(depth_data.unsqueeze(0), "output/pose_labeling_scheme/depth_image.png")

        #seg_mask = (seg_data==self.obj_id).int()
        #depth_data = depth_data * seg_mask
        intrinsic = torch.tensor([2/self.meta_info[0][0,0], 2/self.meta_info[0][1,1],image.shape[1]/2, image.shape[0]/2])# (fx, fy, cx, cy)

        pseudo_ground_truth = -torch.eye(4)
        ground_truth = -torch.eye(4)

        if self.return_pgt:
            try:
                if self.cleaned_pgt:
                    pseudo_ground_truth = torch.load(os.path.join(self.data_dir, frame_id, "cleaned_pseudo_gt.pth"))
                else:
                    pseudo_ground_truth = torch.load(os.path.join(self.data_dir, frame_id, "pseudo_gt.pth"))
            except:
                pseudo_ground_truth = -torch.eye(4)

                loaded=False
        if pseudo_ground_truth is None or pseudo_ground_truth.shape[0]==0:
            pseudo_ground_truth = -torch.eye(4)
            loaded=False

        if self.return_gt:
            ground_truth = torch.load(os.path.join(self.data_dir, frame_id, "ground_truth.pt"))

        

        return {
            "image": image,
            "seg_image": seg_data,
            "depth_image": depth_data,
            "intrinsic": intrinsic,
            "pseudo_gt": pseudo_ground_truth,
            "ground_truth": ground_truth,
            "index": idx,
            "loaded": loaded
        }

    def __len__(self):
        return self.len

def load_meta_info(data_dir):
    meta_data = torch.load(os.path.join(data_dir, "000000", "meta_data.pt"))

    # Assumption that the camera calibration is consistent for all frames
    projection_matrix = meta_data['projection_matrix']
    view_matrix = meta_data['view_matrix']

    seg_id = meta_data['seg_ids']

    return projection_matrix, view_matrix, seg_id
