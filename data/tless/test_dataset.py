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

from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
    PointsRasterizationSettings, PointsRenderer,AlphaCompositor,PointsRasterizer,
    PerspectiveCameras, FoVOrthographicCameras, SoftPhongShader
)

class TLESSTestDataset(Dataset):
    def __init__(self, length,
                    obj_id,
                    mode=0,
                    ground_truth_mode=0,
                    gamma=0.12,
                    train_set=True,
                    occlusion=False,
                    verbose=True,
                    device="cpu"):
        
        super().__init__()

        if train_set:
            self.data_dir = os.path.join("/home/nfs/inf6/data/datasets/T-Less/t-less_v2/train_kinect", str(obj_id).zfill(2))
            self.len = 1295
            if ground_truth_mode==0:
                with open(os.path.join(self.data_dir, "gt.yml"), "r") as f:
                    self.ground_truth = yaml.safe_load(f)

        else:
            self.data_dir = os.path.join("/home/nfs/inf6/data/datasets/T-Less/t-less_v2/test_kinect", str(obj_id).zfill(2))


        self.mode = mode
        self.train_set = train_set
        self.len_complete = length
        self.ground_truth_mode = ground_truth_mode
        self.occlusion = occlusion
        self.device = device
        self.mesh_data, self.diameter = utils.load_tless_object_model(obj_id, device=device)
        self.verbose=True


        with open(os.path.join(self.data_dir, "info.yml"), "r") as f:
            self.meta_info = yaml.safe_load(f)

        self.ResNetTransform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.OccTransform = torchvision.transforms.RandomErasing(p=0.8, scale=(0.1,0.5), inplace=True)



    def __getitem__(self, idx):
        data_dir = os.path.join(self.data_dir,"rgb", (str(idx).zfill(4)+"_rgb.pth"))
        depth_dir = os.path.join(self.data_dir,"depth", (str(idx).zfill(4)+".pth"))
        seg_dir = os.path.join(self.data_dir, "seg", (str(idx).zfill(4)+".pth"))
        try:
            image = torch.load(data_dir)
            seg_data = torch.load(seg_dir)
            depth_data = torch.load(depth_dir)
        except:
            return self.__getitem__((idx+1)%self.__len__())

        if self.verbose:
            torchvision.utils.save_image(image / 255., "output/tless_2/org_img.png")
            torchvision.utils.save_image(seg_data, "output/tless_2/seg_image.png")
            torchvision.utils.save_image(depth_data/1000, "output/tless_2/depth_image.png")



        if self.mode==0:
            pseudo_gt_dir = os.path.join(self.data_dir,"pseudo_gt", (str(idx).zfill(4)+".pth"))
            
            K = self.meta_info[idx]['cam_K']

            intrinsic = torch.tensor([K[0], K[4], K[2], K[5]]) # (fx, fy, cx, cy)
            reference_renderer, silhouette_renderer = self.create_renderer(intrinsic)
            pseudo_ground_truth = generate_pseudo_gt(
                                                    height = image.shape[1], width=image.shape[2],
                                                    diameter=self.diameter,
                                                    intrinsic=intrinsic,
                                                    mesh_data=self.mesh_data,
                                                    renderer=silhouette_renderer,
                                                    ref_renderer=reference_renderer,
                                                    dataset=1,
                                                    object_id=1,
                                                    depth_image=depth_data,
                                                    seg_image=seg_data,
                                                    verbose=True,
                                                    device=self.device)
            
            torch.save(pseudo_ground_truth, pseudo_gt_dir)

            return 1
        
        if self.mode==1:
            # Load complete set of data for each frame
            K = self.meta_info[idx]['cam_K']

            intrinsic = torch.tensor([K[0], K[4], K[2], K[5]]) # (fx, fy, cx, cy)
            K = torch.reshape(torch.FloatTensor(K), (3,3))
            return {
                'image': image/255.,
                'seg_image': seg_data,
                'depth_image': depth_data,
                'intrinsic': intrinsic,
                'calibration_matrix': K,
            }

        if self.mode==2:
            K = self.meta_info[idx]['cam_K']

            intrinsic = torch.tensor([K[0], K[4], K[2], K[5]]) # (fx, fy, cx, cy)
            reference_renderer, silhouette_renderer = self.create_renderer(intrinsic)

            pseudo_rotation, pseudo_translation = data.differentiable_renderer(
                                                                                    depth_image=depth_data,
                                                                                    seg_image=seg_data,
                                                                                    mesh_data=self.mesh_data,
                                                                                    renderer=silhouette_renderer,
                                                                                    ref_renderer=reference_renderer,
                                                                                    intrinsic=intrinsic,
                                                                                    verbose=True,
                                                                                    device=self.device)

            return 1

        image_edit = torch.clone(image)
        image_edit = self.ResNetTransform(image_edit)

        if self.occlusion:
            image_edit = self.OccTransform(image_edit)

        ground_truth = self.load_ground_truth(idx)

        return {
            'image': image_edit,
            'image_raw': image,
            'obj_pose_in_camera': ground_truth
        }

            


    def load_ground_truth(self, idx):
        if self.ground_truth_mode==0:
            gt = torch.eye(4)
            gt[:3,:3] = self.ground_truth[str(idx)]['cam_R_m2c']
            gt[:3,-1] = self.ground_truth[str(idx)]['cam_t_m2c']

        
        return gt
    
    def create_renderer(self, intrinsic):
                # Initializing the silhouette renderer
        trans = torch.tensor([[0, 0, 0]])
        rot = torch.eye(3).unsqueeze(0)
        # Perspective camera
        cameras = PerspectiveCameras(
            focal_length = intrinsic[:2].unsqueeze(0),
            principal_point = intrinsic[2:].unsqueeze(0),
            R=rot,
            T=trans,
            in_ndc=False,
            image_size=torch.tensor([[400,400]]),
            device=self.device
        )

        # Reference renderer
        raster_settings = PointsRasterizationSettings(
            image_size=(400,400),  
            radius=0.02,
            points_per_pixel=1
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        reference_renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=(0.,0.,0.,1.))
        )

        # Silhoutte renderer
        blend_params = BlendParams(sigma=1e-8, gamma=1e-8, background_color=[0,0,0,1])
        sigma = 1e-4
        raster_settings = RasterizationSettings(
            image_size=(400,400),  
            blur_radius=0, #np.log(1. / 1e-4 - 1.)*sigma,
            faces_per_pixel=1
        )


        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        silhouette_renderer = utils.MeshRendererWithFragments(
            rasterizer=rasterizer,
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )

        return reference_renderer, silhouette_renderer


    def __len__(self):
        return self.len