import os
import numpy as np
from math import pi
import stillleben as sl
import torch
from pytorch3d.transforms import euler_angles_to_matrix
import ipdb

obj_id_to_file= {
    3: ["obj_000001.ply", 1],
    4: ["obj_000002.ply", 2],
    5: ["obj_000013.ply", 13],
    6: ["obj_000003.ply", 3],
    8: ["obj_000014.ply", 14]
}


class Renderer:
    FOV_X = 20.0 * pi / 180.0
    
    def __init__(self, obj_id, img_size=(640, 480)):
        camera_position = torch.Tensor([0, 1e-20, -2.5])
        bbox_scale = 0.7
    
        self.img_size = img_size
        mesh_path = os.path.join("/home/nfs/inf6/data/datasets/YCB_VIDEO_DATASET/YCB_Video_Dataset", "models_bop",obj_id_to_file[obj_id][0])
        self.mesh = sl.Mesh.load_threaded([mesh_path])[0]
        self.mesh.scale_to_bbox_diagonal(bbox_scale)
        self.obj=sl.Object(self.mesh)
        self.scene = sl.Scene(self.img_size)
        self.scene.add_object(self.obj)
        #ipdb.set_trace()
        self.scene.set_camera_hfov(self.FOV_X)
        self.scene.set_camera_look_at(position=camera_position,
                                      look_at=torch.Tensor([0., 0., 0.]))
        #self.scene.light_position= torch.tensor([0,1,-3.5])
        #self.scene.light_directions = torch.tenso r([0,0,0])
        #ipdb.set_trace()
        self.scene.choose_random_light_direction()
        self.scene.light_position = torch.tensor([0,2,-1.5])
        #self.scene.light_directions[0] = torch.tensor([0,0,0])
        #self.scene.light_colors[0] = torch.tensor([300,300,300])
        self.scene.ambient_light = torch.tensor([1,1,1])
        
        #self.scene.background_color = torch.tensor([0.1, 0.1, 0.1, 1.0])
        
        self.sl_renderer = sl.RenderPass(shading='phong')
        self.Rt = torch.eye(4)

    def render(self):
        view = self.sl_renderer.render(self.scene)
        rgb_img = view.rgb()[:,:,:3]
        return rgb_img

    def set_obj_orientation(self, rot_mat):
        '''
        set object orientation (3x3 rotation matrix)
        '''
        xs = rot_mat.shape
        assert xs[-1] == 3 and xs[-2] == 3, 'Expected rot_mat:  3x3 tensor'
        self.Rt[:3, :3] = rot_mat.clone()
        self.scene.objects[0].set_pose(self.Rt)
    
    def set_obj_pose(self, pose):
        '''
        set object pose (4x4 SE(3) transformation matrix)
        '''
        xs = pose.shape
        assert xs[-1] == 4 and xs[-2] == 4, 'Expected pose: 4x4 tensor'
        self.Rt = pose.clone()
        self.scene.objects[0].set_pose(self.Rt)

    def get_pointcloud_canonical(self):
        return self.downsampled_mesh.points

    def get_pointcloud_rotated(self):
        return self.get_pointcloud_canonical() @ torch.transpose(self.Rt[:3, :3], -2, -1)

    def get_camera_pose(self):
        '''
        camera pose in world frame
        '''
        return self.scene.camera_pose()

    def get_object_pose_in_world(self):
        '''
        pose of object in world frame
        '''
        return self.Rt

    def get_object_pose_in_camera(self):
        '''
        pose of object in camera frame
        '''
        return  torch.inverse(self.get_camera_pose()) @  self.Rt
