import torch
import pytorch3d.ops as ops
from scipy.spatial.transform import Rotation as R
import copy
import time

import numpy as np
import ipdb

import registration
import viz
import utils

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pose_labeling_scheme(pts_canonical, seg_data, depth_data, diameter, intrinsic, obj_model_sl, config):

    # Extract and process point clouds
    # Scale everything to cm for optimization
    pts_observed = utils.generate_pointcloud_from_rgbd(config["dataset"], seg_data, depth_data, intrinsic)
    pts_canonical = pts_canonical*100
    diameter = diameter*100
    # Construct open3d point clouds. Additionally downsampling, fpfh features, and normals approximation
    source_down_global, source_fpfh_global = registration.preprocess_point_cloud(pts_canonical, config["voxel_size_global"])
    target_down_global, target_fpfh_global = registration.preprocess_point_cloud(pts_observed, config["voxel_size_global"])

    source_down_local, source_fpfh_local = registration.preprocess_point_cloud(pts_canonical, config["voxel_size_local"])
    target_down_local, target_fpfh_local = registration.preprocess_point_cloud(pts_observed, config["voxel_size_local"])

    # Construct tensors containing the downsampled point cloud points 
    pts_observed_down = torch.from_numpy(np.asarray(target_down_local.points)).float().to(DEVICE).unsqueeze(0)
    pts_canonical_down = torch.from_numpy(np.asarray(source_down_local.points)).float().to(DEVICE).unsqueeze(0)




    if config["verbose"]:
        print("_____________Pose Labeliling Scheme_______________\n")
        print("\n:: Object Info ::\n")
        print("\nModel diameter: ", diameter)
        print("\nVoxel size (global opt.): ", config["voxel_size_global"])
        print("\nVoxel size (local opt.): ", config["voxel_size_local"])
        print("\nDownsampled Pts observed (global opt.): ", source_down_global.points)
        print("\nDownsampled Pts canonical (global opt.): ", target_down_global.points)
        print("\nDownsampled Pts observed (local opt.): ", source_down_local.points)
        print("\nDownsampled Pts canonical (local opt.): ", target_down_local.points)

    pseudo_ground_truth_set = []

    icp_iter = 0
    for iter in range(config["max_iteration"]):

        if icp_iter == config["icp_iteration"]:
            break

        if config["verbose"]:
            print("\n\n:: Global Registration ::")
            print("\n\nStarting Fast Global Registration...\n")

        # Random initial rotation for FGR
        if (pseudo_ground_truth_set != []):
            icp_iter += 1
            offset = 0.1*np.pi*(iter-1)
            set = registration.generate_init_set_r(pseudo_ground_truth_set[-1][:3,:3], offset)
            init_transform = torch.repeat_interleave(torch.eye(4).unsqueeze(0), set.shape[0], dim=0)
            init_transform[:,:3,-1] = torch.repeat_interleave((pseudo_ground_truth_set[-1][:3,-1]*100).unsqueeze(0), set.shape[0], dim=0)
            init_transform[:,:3,:3] = set
            init_transform = init_transform

        else:
            init_rotation = torch.from_numpy(R.random().as_matrix())
            init_transform_global = torch.eye(4)
            init_transform_global[:3,:3] = init_rotation

            source_down_global.transform(init_transform_global)

                ## Fast Global Registration ##

            # Compute an initial transformation using Fast Global Registration
            init_transform, result = registration.fast_global_registration(target_down=target_down_global, source_down=source_down_global, 
                                                                            source_fpfh=source_fpfh_global, 
                                                                            target_fpfh=target_fpfh_global,
                                                                            diameter=diameter,
                                                                            hyper_param=config)

            # Compute initial transform for ICP
            init_transform[:3,:3] = init_transform[:3,:3] @ init_transform_global[:3,:3].numpy()

            # Define set of initial transformaions for ICP
            init_transform = torch.from_numpy(init_transform).unsqueeze(0)
            if config["verbose"]:
                t =init_transform.squeeze().to(DEVICE).float()
                points_rendered = torch.from_numpy(np.asarray(source_down_local.points)).to(DEVICE).float() @ t[:3,:3].T + t[:3,-1]
                viz.visualize_pointclouds(torch.from_numpy(np.asarray(target_down_local.points)).float(), points_rendered.cpu(), "output/tless_2/global_result.png")

        if config["verbose"]:
            print("\nFast Global Registration finished!")
            print("\nFast Global Registration result: \n", result)
            print("\n\n:: Local Registration ::")
            print("\nStart Iterative-Closest-Point...\n")

        # Construct torch tensors containing points and normals from open3d point cloud structures
        pts_canonical = pts_canonical.to(DEVICE)

        ## ICP ##
        # Start Iterative-Closest-Point algorithm for local optimization
        pseudo_transformation = registration.iterative_closest_point(pts_observed_down, pts_canonical_down, 
                                                                                        init_transform,
                                                                                        hyper_param=config)

        # Scale translation back to meters
        pseudo_transformation[:,:3,-1] /= 100
        U, S, V = torch.linalg.svd(pseudo_transformation[:,:3,:3])
        pseudo_transformation[:,:3,:3] = torch.bmm(U, V)
        if not config['verbose']:
            converged = registration.check_convergence_batchwise( depth_original=depth_data,
                                                                obj_model=obj_model_sl, 
                                                                transformation_set=pseudo_transformation,
                                                                threshold=config['threshold'],
                                                                intrinsic=intrinsic,
                                                                verbose=config['verbose'],
                                                                device=DEVICE)
        else:
            converged, d_max, d_avg = registration.check_convergence_batchwise( depth_original=depth_data,
                                                                obj_model=obj_model_sl, 
                                                                transformation_set=pseudo_transformation,
                                                                threshold=config['threshold'],
                                                                intrinsic=intrinsic,
                                                                verbose=config['verbose'],
                                                                device=DEVICE)
        conv_ind = torch.nonzero(converged)
        for ind in conv_ind:
            pseudo_ground_truth_set.append(pseudo_transformation[ind].squeeze())

        if config["verbose"]:
            # Visualize results of one iteration of the pipeline
            for (i, pgt) in enumerate(pseudo_transformation):
                pgt[:3,-1] *= 100
                # PLT visualization
                final_pcd = copy.deepcopy(source_down_local)
                final_pcd.transform(pgt)
                pts_obs = torch.from_numpy(np.asarray(target_down_local.points))
                pts_final = torch.from_numpy(np.asarray(final_pcd.points))
                viz.visualize_pointclouds(pts_obs, pts_final, "output/tless_2/registration_result.png")

                #print("\nFast Global Registration result: \n", result)
                #final_pcd = copy.deepcopy(source_down_local)
                #final_pcd = final_pcd.transform(pgt)
                #viz.draw_pointcloud_o3d(source_down, lookat=pseudo_translation)
                #viz.draw_pointcloud_o3d(target_down, lookat=pseudo_translation)
                print("\nPose Labeling scheme finished!\n")
                print("\n_________________________________________________\n")
                print(f":: Results of ICP initialization no. {i+1} ::\n")
                print("\nFinal Transformation:\n ", pgt)
                #print("\nNormals Convergence threshold: ", hyper_param["converge_norm"])
                #print("\nL2 distance: Mean: ", distance_mean, " Median: ", distance_median)
                #print("\nNormal angle: Mean: ", angle_mean, ", Median: ", angle_median)
                print("Convergence results:")
                print ("Maximum distance: ", d_max,"Mean Distance: ", d_avg)
                print("\nConverged: ", converged[i])
                #print("\nConverge: ", converge)
                print("\n_________________________________________________\n")

                #viz.draw_registration_result(final_pcd, target_down_local, np.identity(4), lookat=pseudo_transformation[:3,-1].cpu().numpy())
        #ipdb.set_trace()
    if pseudo_ground_truth_set == []:
        return None
    return torch.stack(pseudo_ground_truth_set)