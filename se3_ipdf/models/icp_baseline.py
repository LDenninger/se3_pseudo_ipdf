import numpy as np
import torch
import torch.nn as nn
import pytorch3d.ops as ops
import pytorch3d.transforms as tt


class IterativeClosestPoint(nn.Module):
    """Module encapsuling the ICP-algorithm used as a baseline for evaluating the ImplicitIPDF models
    
    """

    def __init__(self, initialization_mode, relative_rmse_thr, 
                max_iterations,
                projection_matrix,
                img_size=(1080,1920),
                object_id=3,
                verbose = False,
                device="cpu"):
        super(IterativeClosestPoint, self).__init__()
        self.init_mode = initialization_mode
        self.relative_rmse_thr = relative_rmse_thr
        self.max_iterations = max_iterations
        self.intrinsic = torch.tensor([2/projection_matrix[0,0], 2/projection_matrix[1,1],img_size[1]/2, img_size[0]/2])
        self.object_id = object_id
        self.img_size = img_size
        self.device = device
        self.verbose = verbose
    
    def forward(self, input, model_points):

        pc_canon = model_points

        depth_image = input['depth_image']
        seg_image = input['seg_image']

        batch_size = depth_image.shape[0]

        # Extract observed point cloud from the image
        pc_obs = self.convert_rgbd_to_pointcloud(seg_image, depth_image).to(self.device).unsqueeze(0)


        init_scale = torch.ones(batch_size).to(self.device)
        if self.init_mode == 0:
            ## Random Initialization ##
            init_rot = tt.random_rotations(batch_size, dtype=torch.float32).to(self.device)
            
            # Calculate middle point of the observed point clouds for the initialization of the translation
            init_trans = torch.mean(pc_obs, dim=1).to(self.device)
        
        if self.init_mode == 1:
            ground_truth = input['obj_pose_in_camera'].float().to(self.device)

            rand_num = torch.from_numpy(np.random.uniform(-np.pi/4,np.pi/4, 3))
            noise_rotation = torch.repeat_interleave(tt.euler_angles_to_matrix(rand_num, 'ZYX').unsqueeze(0), batch_size, dim=0).float().to(self.device)
            #noise_rotation = torch.eye(3).unsqueeze(0).float().to(self.device)
            #init_rot = torch.bmm(noise_rotation, ground_truth[:,:3,:3])
            init_rot = ground_truth[:,:3,:3]
            init_rot = torch.bmm(noise_rotation, init_rot)
            init_trans = -(ground_truth[:,:3,-1]@init_rot.squeeze())
            #init_trans = torch.mean(pc_obs, dim=1).to(self.device)

        #p_init = pc_obs@init_rot + init_trans
        #viz.visualize_pointclouds(p_init.squeeze().cpu(), pc_canon.squeeze().cpu(), filename="output/pc_init.png")

        init_transformation = ops.points_alignment.SimilarityTransform(init_rot, init_trans, init_scale)

        icp_solution = ops.iterative_closest_point(pc_obs, pc_canon, init_transformation, 
                                                        relative_rmse_thr = self.relative_rmse_thr, 
                                                        max_iterations=self.max_iterations ,allow_reflection = False, verbose=self.verbose)
        final_rotation = icp_solution.RTs[0].detach()
        #final_rotation = icp_solution.RTs[0].detach()
        final_translation = -(icp_solution.RTs[1].detach().squeeze()@torch.transpose(final_rotation, -2, -1))
        #final_pointcloud = icp_solution.Xt
        #final_translation = icp_solution.RTs[1].detach().squeeze()

        estimation = torch.repeat_interleave(torch.eye(4).unsqueeze(0), batch_size, dim=0).to(self.device)
        estimation[:,:3,:3] = final_rotation
        estimation[:,:3,-1] = final_translation
        #p_gt = pc_canon@ground_truth[0,:3,:3].T + ground_truth[0,:3,-1]
        #viz.visualize_pointclouds(pc_canon.squeeze().cpu(), final_pointcloud.squeeze().cpu(), filename="output/pc_final.png")
        return estimation
    
    def _predict_rotation(self, input, model_points):
        return self.forward(input, model_points)

    def convert_rgbd_to_pointcloud(self, seg_image, depth_image):
        """Main function to extract the point cloud of the object in the image

        Arguments:
            image: Batch of RGB images
            seg_data: Segmentation data containing information about the object pixels
            depth: Depth data for every pixel in the RGB image
        """
        def _set_obj_pixel(seg):
            seg_pixel = []
            object_pixel = seg==self.object_id
            object_pixel = torch.nonzero(object_pixel).cpu()
            return object_pixel
        with torch.no_grad():
            object_pixel = _set_obj_pixel(seg_image.squeeze())

            x = -(torch.sub(object_pixel[:,1], self.intrinsic[2]))/(self.img_size[1])
            y = (torch.sub(object_pixel[:,0],self.intrinsic[3]))/(self.img_size[0])
            d = depth_image[0,object_pixel[:,0],object_pixel[:,1]]

            points = torch.cat(((d*self.intrinsic[0]*x).unsqueeze(-1),(d*self.intrinsic[1]*y).unsqueeze(-1),d.unsqueeze(-1)), dim=-1)
            points = points[::6]

            
        return points

