import torch
import pytorch3d.transforms as tt
import pytorch3d.ops as ops
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluation_acc_error(pseudo_gt, ground_truth, obj_id):

    ground_truth_set = produce_ground_truth_set(ground_truth, obj_id)


    geo_d = geo_dist_pairwise(pseudo_gt, ground_truth_set)

    try:
        min_geodesic_dist = torch.min(geo_d, dim=-1)[0]
    except:
        return None

    mean = torch.mean(min_geodesic_dist).numpy()
    return np.rad2deg(mean)

def evaluation_translation_error(pseudo_gt, ground_truth):

    pseudo_trans = pseudo_gt[...,:3,-1]
    trans_gt = ground_truth[:3,-1]

    dist = torch.sqrt(torch.sum((pseudo_trans-trans_gt)**2, -1))

    mean = torch.mean(dist)

    return mean.numpy()

def evaluation_recall_error(pseudo_gt, ground_truth, obj_id):

    ground_truth_set = produce_ground_truth_set(ground_truth[0], obj_id, m=True).float()
    try:
        if ground_truth.shape[0]==1:
            ground_truth = ground_truth.squeeze()
            pseudo_gt = ground_truth.T.float() @  pseudo_gt[0].float()
        else:
            pgt = []
            for i in range(ground_truth.shape[0]):
                pgt.append(ground_truth[i].T.float() @ pseudo_gt[i].float())
            pseudo_gt = torch.cat(pgt, dim=0)
    except:
        import ipdb; ipdb.set_trace()
    geo_d = geo_dist_pairwise(ground_truth_set, pseudo_gt)
    try:
        min_geodesic_dist = torch.min(geo_d, dim=-1)[0]
    except:
        return None

    mean = torch.mean(min_geodesic_dist).numpy()
    
    return np.rad2deg(mean)

def evaluation_mann(pseudo_gt):
    
    distance = geo_dist_pairwise(pseudo_gt[:,:3,:3], pseudo_gt[:,:3,:3])
    min_dist = []

    for (i, d) in enumerate(distance):
        d_ = torch.cat((d[:i], d[(i+1):]))
        min_dist.append(torch.min(d_))

    mean = np.mean(np.rad2deg(min_dist))

    return mean

def evaluation_adds(pseudo_gt, ground_truth, object_model):



    pseudo_gt = pseudo_gt.to(DEVICE).float()
    ground_truth = ground_truth.to(DEVICE).float()
    object_model = object_model.float()



    batch_size = pseudo_gt.shape[0]


    """if obj_id==5:
        pseudo_gt[:,:3,-1] = torch.repeat_interleave(torch.mean(pseudo_gt[:,:3,-1], dim=0).unsqueeze(0), batch_size, dim=0)"""



    point_cloud_gt = object_model @ ground_truth[:3,:3].T + ground_truth[:3,-1]
    point_cloud_gt = torch.repeat_interleave(point_cloud_gt.unsqueeze(0), batch_size, dim=0)

    point_cloud_pgt = torch.einsum('aj,bjk->bak', object_model, torch.transpose(pseudo_gt[:,:3,:3], -2, -1)) +  pseudo_gt[:,:3,-1].unsqueeze(1)

    adds_distance, idx, nn = ops.knn_points(point_cloud_pgt, point_cloud_gt)
    adds_distance = torch.sqrt(adds_distance).squeeze()
    adds_distance = torch.mean(adds_distance, dim=-1)


    return adds_distance.tolist()



def visualize_pointclouds(p_1, p_2, filename="output/test.png"):
    """ p_1 is painted green and p_2 is painted red 
    
    """
    center = torch.mean(p_1, dim=0, keepdim=True).squeeze()

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(p_1[:,0], p_1[:,1], p_1[:,2], s=1, color="green", alpha=0.3)
    ax.scatter(p_2[:,0], p_2[:,1], p_2[:,2], s=1, color="red", alpha=0.3)
    ax.set_xlim(center[0]-0.2, center[0]+0.2)
    ax.set_ylim(center[1]-0.2, center[1]+0.2)
    ax.set_zlim(center[2]-0.2, center[2]+0.2)

    fig.savefig(filename)
    plt.close()



def geo_dist_pairwise(r1, r2):
    """Computes pairwise geodesic distances between two sets of rotation matrices.
    Argumentss:
        r1: [N, 3, 3] numpy array
        r2: [M, 3, 3] numpy array
    Returns:
        [N, M] angular distances.
    """

    prod = torch.einsum('aij,bkj->abik', r1, r2)
    trace = torch.einsum('baii->ba', prod)
    return torch.arccos(torch.clip((trace - 1.0) / 2.0, -1.0, 1.0))


def geo_dist(r1, r2):
    """Computes pairwise geodesic distances between two sets of rotation matrices.
    Argumentss:
        r1: [N, 3, 3] numpy array
        r2: [N, 3, 3] numpy array
    Returns:
        N angular distances.
    """

    prod = r1 @ torch.transpose(r2, -2, -1)
    trace = torch.einsum('bii->b', prod)
    return torch.arccos(torch.clip((trace - 1.0) / 2.0, -1.0, 1.0))


def produce_ground_truth_set(rotation_gt, obj_id, num=200, m=False):

    device = rotation_gt.device

    if obj_id == 3:
        device = rotation_gt.device
        rot_mag = torch.linspace(-np.pi,np.pi, steps=int(num/2))
        rotation = torch.zeros(int(num/2),3)
        rotation[:,0] = rot_mag
        rotation = tt.euler_angles_to_matrix(rotation, 'ZYX').to(device).float()
        flip = tt.euler_angles_to_matrix(torch.repeat_interleave(torch.tensor([[0, 0, np.pi]]),int(num/2),dim=0), 'ZYX').to(device).float()
        rotation_flip = flip @ rotation
        syms = torch.cat([rotation, rotation_flip]).to(device).float()
        ground_truth_set = rotation_gt.float() @ syms
    if obj_id==4:
        syms = torch.zeros(4,3,3)
        syms[0] = torch.eye(3)
        syms[1] = tt.euler_angles_to_matrix(torch.Tensor([0,0, np.pi]), 'ZYX').to(device).float()
        syms[2] = tt.euler_angles_to_matrix(torch.Tensor([0,np.pi,0 ]), 'ZYX').to(device).float()
        syms[3] = syms[1] @ syms[2]

        ground_truth_set = rotation_gt.float() @ syms.float().to(device)


    if obj_id==5:

        rot_mag = torch.linspace(-np.pi,np.pi, steps=int(num/2))
        rotation = torch.zeros(int(num/2),3)
        rotation[:,0] = rot_mag
        rotation = tt.euler_angles_to_matrix(rotation, 'ZYX')
        syms = rotation.to(device).float()
        ground_truth_set = rotation_gt.float() @ syms

    if m:
        return syms

    return ground_truth_set
