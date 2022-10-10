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
import torchvision

from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
    PointsRasterizationSettings, PointsRenderer,AlphaCompositor,PointsRasterizer,
    PerspectiveCameras, FoVOrthographicCameras, SoftPhongShader
)

# img size: (200,400)

class TabletopTestDataset(Dataset):
    OBJ_ID = 3
    BB_SIZE = (560,560)
    def __init__(self, data_dir, length, img_size=(200, 356), mesh_data=None, diameter=0,
                start=0,
                end=0,
                mode=0,
                num_gt=5,
                icp_init_file=None,
                verbose=True,
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

        self.mode = mode

        self.data_dir = data_dir
        self.len_complete = length
        self.device = device
        self.verbose = verbose
        self.img_size = img_size
        self.start = start
        self.end = end
        self.num_gt = num_gt
        self.icp_init_file=icp_init_file
        self.resizer = torchvision.transforms.Resize((self.img_size))
  
        self.mesh_data = mesh_data
        self.diameter = diameter

        self.obj_id = load_meta_info(data_dir)[2]['OBJECT_ID']
        
        self.len = end-start
        # Initializing the silhouette renderer
        trans = torch.tensor([[0, 0, 0]])
        rot = torch.eye(3).unsqueeze(0)
        # Perspective camera
        cameras = FoVPerspectiveCameras(device=device, R=rot, T=trans,
                                znear=0.1, zfar=3)

        # Silhoutte renderer
        raster_settings = PointsRasterizationSettings(
            image_size=(1080,1920),  
            radius=0.02,
            points_per_pixel=1
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        self.reference_renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=(0.,0.,0.,1.))
        )

        # Silhoutte renderer
        blend_params = BlendParams(sigma=1e-8, gamma=1e-8, background_color=[0,0,0,1])
        sigma = 1e-4
        raster_settings = RasterizationSettings(
            image_size=(1080,1920),  
            blur_radius=0, #np.log(1. / 1e-4 - 1.)*sigma,
            faces_per_pixel=1
        )


        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        self.silhouette_renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )


    def __getitem__(self, idx):
        idx = self.start+idx
        data = os.path.join(self.data_dir, str(idx).zfill(6))
        # Mode used for training. It only returns the needed fields: cropped image and ground truth
        if self.mode==0:
            try:
                image = torch.load(os.path.join(data,"rgb_tensor.pt"))
                seg_data = torch.load(os.path.join(data, "seg_data.pt"))
                depth_data = torch.load(os.path.join(data,"depth_data.pt"))
            except:
                try:
                    meta_data = torch.load(os.path.join(data, "meta_data.pt"))
                    image = torch.from_numpy(meta_data['rgb_tensor'][...,:3])
                    seg_data = torch.from_numpy(meta_data['seg_tensor'].astype("int32"))
                    depth_data = torch.from_numpy(meta_data['depth_tensor'])

                
                except:
                    return {
                    'pseudo_gt': -torch.eye(4),
                    'converged': False
                    }

            if seg_data.max()!= self.obj_id:
                return {
                'pseudo_gt': -torch.eye(4),
                'converged': False
                }
            if self.verbose:
                torchvision.utils.save_image(image.permute(2,0,1) / 255., "output/org_img.png")

            ground_truth = generate_pseudo_gt(height=image.shape[0],width=image.shape[1],projection_matrix=meta_data['projection_matrix'],
                                    mesh_data=self.mesh_data,
                                    diameter=self.diameter,
                                    renderer=self.silhouette_renderer,
                                    ref_renderer=self.reference_renderer,
                                    depth_image=depth_data,
                                    seg_image=seg_data,
                                    object_id=self.obj_id,
                                    verbose=self.verbose,
                                    device=self.device)
            
            #torch.save(ground_truth, os.path.join(data, "pseudo_gt.pth"))
            return {
                'pseudo_gt': ground_truth,
                'converged': True
            }
        # Mode for loading the data
        if self.mode==1:
            try:
                image = torch.load(os.path.join(data,"rgb_tensor.pt"))
                seg_data = torch.load(os.path.join(data, "seg_data.pt"))
            except:
                try:
                    meta_data = torch.load(os.path.join(data, "meta_data.pt"))
                    image = torch.from_numpy(meta_data['rgb_tensor'][...,:3])
                    seg_data = torch.from_numpy(meta_data['seg_tensor'].astype("int32"))

                
                except:
                    return {
                    'image': -torch.eye(4),
                    'cropped_image': -torch.eye(4),
                    'pseudo_gt': -torch.eye(4),
                    'ground_truth':-torch.eye(4)
                    }

            crop_image = self.resizer(image.to(self.device))
            pseudo_gt = []
            for i in range(self.num_gt):
                if os.path.exists(os.path.join(data, f"pseudo_gt_{i}.pt"))==True:
                    pseudo_gt.append(torch.load(os.path.join(data, f"pseudo_gt_{i}.pt")))
            if len(pseudo_gt)==0:
                pseudo_gt = -torch.eye(4)
            else:
                pseudo_gt = torch.stack(pseudo_gt)

            return {
                'image': image,
                'cropped_image': crop_image.cpu(),
                'pseudo_gt': pseudo_gt,
                'ground_truth': meta_data['obj_in_cam']
            }

        if self.mode==2:
            meta_data = torch.load(os.path.join(data, "meta_data.pt"))
            seg_data = torch.from_numpy(meta_data['seg_tensor'].astype("int32"))
            if seg_data.max()!=3:
                return -1
            object_pixel = (seg_data==3)
            object_pixel = torch.nonzero(object_pixel)
            if object_pixel.numel() == 0:
                return -1
            image = torch.from_numpy(meta_data['rgb_tensor'][...,:3]).permute(2,0,1)
            crop_image = self._get_bb_resized_img(self.resizer, image, seg_data, object_pixel)
            torch.save(crop_image, os.path.join(data, "crop_rgb_tensor.pt"))
            torch.save(image, os.path.join(data, "rgb_tensor.pt"))
            torchvision.utils.save_image(crop_image/ 255., os.path.join(data, "crop_rgb_image_debug.png"))
            return 1
        
        if self.mode==3:
            meta_data = torch.load(os.path.join(data, "meta_data.pt"))
            image = torch.from_numpy(meta_data['rgb_tensor'][...,:3])
            seg_data = torch.from_numpy(meta_data['seg_tensor'].astype("int32"))
            depth = torch.from_numpy(meta_data['depth_tensor'])
            # image = torch.load(os.path.join(data, "rgb_tensor.pt"), map_location="cpu")[...,:3]
            # seg_data = torch.load(os.path.join(data, "seg_tensor.pt"), map_location="cpu")[...,0]
            if seg_data.max()!=3:
                return -1
            non_object_pixel = (seg_data!=3)
            non_object_pixel = torch.nonzero(non_object_pixel)
            if non_object_pixel.numel() == 0:
                return -1
            torch.save(image, os.path.join(data, "rgb_tensor.pt"))
            torch.save(seg_data, os.path.join(data, "seg_tensor.pt"))
            torch.save(depth, os.path.join(data, "depth_tensor.pt"))
            image[non_object_pixel[:,0], non_object_pixel[:,1]] = torch.tensor([0,0,0], dtype=torch.uint8)
            object_pixel = (seg_data==3)
            object_pixel = torch.nonzero(object_pixel)
            if object_pixel.numel() == 0:
                return -1
            image = image.permute(2,0,1)
            crop_img = self._get_bb_resized_img(self.resizer, image, seg_data, object_pixel)
            torch.save(image, os.path.join(data,"mask_rgb_tensor.pt"))
            torch.save(crop_img, os.path.join(data,"mask_crop_rgb_tensor.pt"))

            return 1

        if self.mode==4:
            try:
                image_org = torch.load(os.path.join(data, "rgb_tensor.pt"))
                seg_data = torch.load(os.path.join(data, "seg_tensor.pt"))
            except:
                return -1
            image = torch.clone(image_org)
            # Mask the image

            if seg_data.max()!=3:
                return -1
            non_object_pixel = (seg_data!=3)
            non_object_pixel = torch.nonzero(non_object_pixel)
            if non_object_pixel.numel() == 0:
                return -1

            image[non_object_pixel[:,0], non_object_pixel[:,1]] = torch.tensor([0,0,0], dtype=torch.uint8)
            image = image.permute(2,0,1)
            image_org = image_org.permute(2,0,1)

            image = self.resizer(image)
            image_org = self.resizer(image_org)

            torch.save(image, os.path.join(data, "resize_mask_rgb_tensor.pt"))
            torch.save(image_org, os.path.join(data, "resize_rgb_tensor.pt"))

            return 1


    def __len__(self):
        return self.len

    def _get_bb_resized_img(self,resizer, rgb_img, seg_data, obj_pixels,full_size=(1080,1920)):
        left_x, right_x = torch.min(obj_pixels[:, 1]), torch.max(obj_pixels[:, 1])
        top_y, low_y = torch.min(obj_pixels[:, 0]), torch.max(obj_pixels[:, 0])
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
        rgb_img = rgb_img[:,top_y:low_y, left_x:right_x]
        rgb_img = resizer(rgb_img)

        return rgb_img

def load_meta_info(data_dir):
    meta_data = torch.load(os.path.join(data_dir, "000000", "meta_data.pt"))

    # Assumption that the camera calibration is consistent for all frames
    projection_matrix = meta_data['projection_matrix']
    view_matrix = meta_data['view_matrix']

    seg_id = meta_data['seg_ids']

    return projection_matrix, view_matrix, seg_id