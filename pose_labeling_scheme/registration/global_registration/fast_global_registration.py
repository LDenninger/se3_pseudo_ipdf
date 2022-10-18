import open3d as o3d
import numpy as np

def fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, hyper_param, diameter, device="cpu"):


    #distance_threshold = hyper_param["voxel_size_global"] * 2
    distance_threshold = hyper_param["global_dist_threshold"] # 0.4
    options = o3d.pipelines.registration.FastGlobalRegistrationOption(
        decrease_mu = True,
        division_factor = 4, # obj 7: 0.5
        iteration_number = 200,
        maximum_correspondence_distance = distance_threshold, # 0.01 (8)
        maximum_tuple_count = 1000,
        tuple_scale = 0.9,
        use_absolute_scale=False
    )

    if hyper_param["verbose"]:
        print(":: Apply fast global registration with distance threshold %.3f" \
                % distance_threshold)

    result = o3d.pipelines.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, options)

    final_transformation = np.copy(result.transformation)

    #if hyper_param["verbose"]:
        #draw_registration_result(source_down, target_down, np.identity(4), lookat=init_transform[:3,-1,None])
        #viz.draw_registration_result(source_down, target_down, result.transformation, lookat=final_transformation[:3,-1,None])
        #t = torch.from_numpy(final_transformation).to(device).float()
        #points_rendered = torch.from_numpy(np.asarray(source_down.points)).to(device).float() @ t[:3,:3].T + t[:3,-1]
        #viz.visualize_pointclouds(torch.from_numpy(np.asarray(target_down.points)).float(), points_rendered.cpu(), "output/tless_2/global_result.png")
        
    return final_transformation, result