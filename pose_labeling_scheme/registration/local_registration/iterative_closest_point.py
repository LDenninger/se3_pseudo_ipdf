import pose_labeling_scheme
import torch
import pytorch3d.ops as ops


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def iterative_closest_point(points_observed, points_canonical,
                            initial_transform, 
                            hyper_param):
    # Check dimensions of the given point cloud and initial transform and adjust them to fit
    if len(points_observed.shape)==2:
        points_observed = points_observed.unsqueeze(0)
    if len(points_canonical.shape)==2:
        points_canonical = points_canonical.unsqueeze(0)
    if len(initial_transform.shape)==2:
        initial_transform = initial_transform.unsqueeze(0)
    if initial_transform.shape[0]!=points_observed.shape[0]:
        assert points_observed.shape[0]==1
        points_observed = torch.repeat_interleave(points_observed, initial_transform.shape[0], dim=0)
    
    if points_canonical.shape[0]!=points_observed.shape[0]:
        assert points_canonical.shape[0]==1
        points_canonical = torch.repeat_interleave(points_canonical, points_observed.shape[0], dim=0)
    


    # Initial transform for ICP must be provided

    init_rotation = initial_transform[:,:3,:3].float().to(DEVICE)
    init_translation = initial_transform[:,:3,-1].float().to(DEVICE)

    init_translation = -torch.bmm(init_translation.unsqueeze(1), init_rotation).squeeze(1)
    init_rotation = torch.transpose(init_rotation, -2, -1)

    # Inverse transformation due to the optimizations pts_obs ---> pts_can

    initial_guess = ops.points_alignment.SimilarityTransform(torch.transpose(init_rotation, -2, -1), init_translation, torch.ones(init_rotation.shape[0]).to(DEVICE))

    points_observed = points_observed.to(DEVICE)
    points_canonical = points_canonical.to(DEVICE)

    #pts_obs_ = torch.repeat_interleave(points_observed.unsqueeze(0), num_pseudo_gt, dim=0)
    #pts_can_ = torch.repeat_interleave(points_canonical.unsqueeze(0), num_pseudo_gt, dim=0)

    icp_solution = ops.iterative_closest_point(points_observed, points_canonical,
                                                initial_guess, 
                                                relative_rmse_thr = 1e-4, 
                                                max_iterations=100 ,allow_reflection = False, verbose=hyper_param["verbose"])
    # Invert the results of ICP to get the transform: PC_CANONICAL ---> PC_OBS
    # Attention here the rotation matrix is in the pytorch convention, meaning it is applied from the right hand side
    # pts @ R


    # Convert rotation matrix to fit the regular convention being applied from the left hand side
    pseudo_rotation = torch.transpose(icp_solution.RTs[0].detach(), -2, -1)

    # Inverse transformation
    pseudo_translation = -torch.bmm(icp_solution.RTs[1].detach().unsqueeze(1), pseudo_rotation).squeeze()
    pseudo_rotation = torch.transpose(pseudo_rotation, -2, -1)
    
    # icp point cloud
    points_icp = icp_solution.Xt

    pseudo_transformation = torch.repeat_interleave(torch.eye(4).unsqueeze(0), pseudo_rotation.shape[0], dim=0)
    pseudo_transformation[:,:3,:3] = pseudo_rotation
    pseudo_transformation[:,:3,-1] = pseudo_translation

    return pseudo_transformation

