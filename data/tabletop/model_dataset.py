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


from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
    PointsRasterizationSettings, PointsRenderer,AlphaCompositor,PointsRasterizer,
    PerspectiveCameras, FoVOrthographicCameras, SoftPhongShader
)

# img size: (200,400)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
conv = torch.eye(3)
conv[1,1] = -1
conv[2,2] = -1

PSEUDO_GROUND_TRUTH_FILE = "pseudo_gt_thesis.pth"
CLEAN_PSEUDO_GROUND_TRUTH_FILE = "cleaned_pseudo_gt_thesis.pth"


class TabletopPoseDataset(Dataset):
    BB_SIZE = (560,560)
    def __init__(self, data_dir, length, obj_id, img_size=(224, 224),
                train_mode=True,
                full_data=False,
                pseudo_gt=False,
                single_gt=False,
                gamma=0.12,
                train_set=True,
                bb_crop=False,
                mask=False,
                full=False,
                occlusion=False,
                preset_occ=False,
                trans_out=False,
                include_trans_img=False,
                device="cpu"):
        super().__init__()

        self.single_gt = single_gt
        self.train_set = train_set
        self.obj_id = obj_id
        self.data_dir = data_dir
        self.len_complete = length
        self.device = device
        self.img_size = img_size
        self.bb_crop = bb_crop
        self.mask = mask
        self.full = full
        self.occlusion = occlusion
        self.preset_occ=preset_occ
        self.train_mode = train_mode
        self.full_data = full_data
        self.pseudo_gt = pseudo_gt
        self.include_trans_img=include_trans_img,
        self.Resizer = torchvision.transforms.Resize(size=self.img_size)
        self.ResNetTransform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.OccTransform = torchvision.transforms.RandomErasing(p=0.8, scale=(0.1,0.5), inplace=True)
        self.gamma = gamma
       
        if train_set==True:
            self.len = length
        else:
            self.len = 5000
        
        if obj_id==4 or obj_id==6:
            self.CUBOID_SYMS = torch.from_numpy(self.get_cuboid_syms()).float()

    def __getitem__(self, idx):
        # Set index for training or validation set

        if self.train_set == False:
            idx = 15000+idx
        data = os.path.join(self.data_dir, str(idx).zfill(6))

        if self.train_mode == True:
            # Dataloader for the training process
            ground_truth = self.load_ground_truth(idx)

            if ground_truth is None or ground_truth.shape[0]==0:
                return self.__getitem__((idx+1)%self.__len__())
            if not self.preset_occ:
                image = self.load_image(idx)
            else:
                image = self.load_image(idx,occlusion=True)
            if image is None:
                return self.__getitem__((idx+1)%self.__len__())
             
            

            return {
                'image': image,
                'obj_pose_in_camera': ground_truth,
                'obj_id': self.obj_id
            }

        else:
            # Dataloader used for evaluation
            ground_truth = self.load_ground_truth(idx)
            if not self.preset_occ:
                image = self.load_image(idx)
                image_raw = self.load_image(idx, edit=False)
                image
            else:
                image = self.load_image(idx, occlusion=True)
                image_raw = self.load_image(idx, edit=False, occlusion=True)



            if image is None:
                return self.__getitem__((idx+1)%self.__len__())

            if self.full_data:
                try:
                    depth_data = torch.load(os.path.join(data, "depth_tensor.pt"))
                    seg_data = torch.load(os.path.join(data, "seg_tensor.pt"))
                    image_full = torch.load(os.path.join(data, "rgb_tensor.pt"))[...,:3].permute(2,0,1) / 255.
                    
                    image_full_resize_mask = torch.load(os.path.join(data, "resize_mask_rgb_tensor.pt")) / 255.


                except:
                    return self.__getitem__((idx+1)%self.__len__())
            

                return {
                    'image': image,
                    'image_raw': image_raw,
                    'image_original': image_full,
                    'image_trans_input': image_full_resize_mask,
                    'depth_image': depth_data,
                    'seg_image': seg_data,
                    'obj_pose_in_camera': ground_truth,
                    'obj_id': self.obj_id

                }
            return {
                'image': image,
                'image_raw': image_raw,
                'obj_pose_in_camera': ground_truth,
                'obj_id': self.obj_id
            }
            


    def load_image(self, idx, edit=True, preset_occ=False):
        data = os.path.join(self.data_dir, str(idx).zfill(6))            
        try:
            if not self.full:
                if self.bb_crop and self.mask:
                    if preset_occ:
                        image = torch.load(os.path.join(data, "mask_occ_crop_rgb_tensor.pt.pt"))
                    else:
                        image = torch.load(os.path.join(data, "mask_crop_rgb_tensor.pt"))
                elif self.bb_crop and not self.mask:
                    if preset_occ:
                        image = torch.load(os.path.join(data, "mask_occ_crop_rgb_tensor.pt.pt"))
                    else:
                        image = torch.load(os.path.join(data, "crop_rgb_tensor.pt"))
                elif not self.bb_crop and self.mask:
                    if preset_occ:
                        image = torch.load(os.path.join(data, "mask_occ_crop_rgb_tensorpt"))
                    else:
                        image = torch.load(os.path.join(data, "mask_rgb_tensor.pt"))
                elif not self.bb_crop and not self.mask:
                    if preset_occ:
                        image = torch.load(os.path.join(data, "occ_rgb_tensorpt"))
                    else:
                        image = torch.load(os.path.join(data, "rgb_tensor.pt"))[...,:3]
                        image = image.permute(2,0,1)
                        image = self.Resizer(image)
            else:
                if self.mask:
                    if preset_occ:
                        image = torch.load(os.path.join(data, "resize_mask_occ_rgb_tensor.pt"))
                        
                    else:
                        image = torch.load(os.path.join(data, "resize_mask_rgb_tensor.pt"))
                else:
                    if preset_occ:
                        image = torch.load(os.path.join(data, "resize_occ_rgb_tensor.pt"))

                    image = torch.load(os.path.join(data, "rgb_tensor.pt"))
                    image = image.permute(2,0,1)
                    image = self.Resizer(image)


        except:
            return None

        image_edit = image / 255.

        if edit==True:
            if self.occlusion == True and not preset_occ:
                

                self.OccTransform(image_edit)
            image_edit = torchvision.transforms.functional.adjust_gamma(image_edit, gamma=self.gamma)
            image_edit = self.ResNetTransform(image_edit)
        
        return image_edit
    
    def load_ground_truth(self, idx):
        """ Load the ground-truth pose. The mode determines how the ground truth is produced.
        Modes:
            0: Original ground-truth the image was rendered with
            1: Pseudo ground-truth produced offline with the pose labeling scheme
            2: Simulate pseudo ground-truth annotations using the knowledge about the symmetries
            4: Produce pseudo ground-truths online using the pose labeling scheme
        """
        data = os.path.join(self.data_dir, str(idx).zfill(6))
        if self.single_gt==True:
            try:
                    ground_truth = torch.load(os.path.join(data, "gt.pt"))
            except:
                try:
                    ground_truth = torch.load(os.path.join(data, "ground_truth.pt"))
                except:
                    return None
        else:
            if self.pseudo_gt:
                try:
                    pseudo_gt_set = torch.load(os.path.join(data, CLEAN_PSEUDO_GROUND_TRUTH_FILE))

                except:
                    if self.obj_id == 5:
                        try:
                            pseudo_gt_set = torch.load(os.path.join(data, "cleaned_pseudo_gt.pth"))

                        except:
                            return None
                    else:
                        return None
                ground_truth = pseudo_gt_set[np.random.randint(pseudo_gt_set.shape[0])]
                #if self.obj_id==4:
                #    ground_truth = self.produce_ground_truth_analytical(ground_truth)

            else:
                try:
                    ground_truth = torch.load(os.path.join(data, "gt.pt"))
                except:
                    try:
                        ground_truth = torch.load(os.path.join(data, "ground_truth.pt"))
                    except:
                        return None

                ground_truth = self.produce_ground_truth_analytical(ground_truth)
            
        return ground_truth
        

    def get_cuboid_syms(self):
        cuboid_seeds = [np.eye(3)]
        cuboid_seeds.append(np.float32([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]))
        for i in range(3):
            cuboid_seeds.append(np.diag(np.roll([-1, -1, 1], i)))

        cuboid_syms = []
        for rotation_matrix in cuboid_seeds:
            cuboid_syms.append(rotation_matrix)
            cuboid_syms.append(np.roll(rotation_matrix, 1, axis=0))
            cuboid_syms.append(np.roll(rotation_matrix, -1, axis=0))
        cuboid_syms = np.stack(cuboid_syms, 0)
        
        return cuboid_syms

    
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
        return self.len

    def _get_bb_resized_img(self,resizer, rgb_img, seg_data, obj_pixels,full_size=(1080,1920)):
        
        left_x, right_x = torch.min(obj_pixels[1]), torch.max(obj_pixels[1])
        top_y, low_y = torch.min(obj_pixels[0]), torch.max(obj_pixels[0])
        offset_x = torch.div((self.BB_SIZE[1] - (right_x - left_x)),2, rounding_mode="trunc")
        offset_y = torch.div((self.BB_SIZE[0] - (low_y - top_y)), 2, rounding_mode="trunc")
        left_x = max(0, left_x.item() - offset_x.item())
        right_x = self.BB_SIZE[1] + left_x

        if right_x > full_size[1]:
            right_x = full_size[1]
            left_x = full_size[1] - self.BB_SIZE[1]
        top_y = max(0, top_y.item() - offset_y.item())
        low_y = self.BB_SIZE[0] + top_y
        if low_y > full_size[0]:
            low_y = full_size[0]
            top_y = full_size[0] - self.BB_SIZE[0]

        seg_data = seg_data[top_y:low_y, left_x:right_x]
        rgb_img = rgb_img[:,:,:3]
        rgb_img = rgb_img[top_y:low_y, left_x:right_x].permute(2, 0, 1)
        rgb_img = resizer(rgb_img)

        return rgb_img

