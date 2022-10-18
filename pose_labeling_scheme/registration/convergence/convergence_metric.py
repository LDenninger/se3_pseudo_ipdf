import torch
import pytorch3d.ops as ops

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Distance functions to evaluate the quality of the alignment of two point clouds ##
def knn_l2_distance(pc_can, pc_obs, rotation=None, translation=None):
    if rotation is not None and translation is not None:
            pc_rendered = registration.transform_points(pc_can, rotation, translation)
    else:
            pc_rendered = pc_can

    nn_dist, idx, nn = ops.knn_points(pc_obs, pc_rendered)
    nn_dist = torch.mean(nn_dist.squeeze(-1), dim=-1)
    return nn_dist