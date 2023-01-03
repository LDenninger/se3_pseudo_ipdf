import torch
import pytorch3d.transforms as tt
import numpy as np

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
    
    import ipdb; ipdb.set_trace()
    distance = geo_dist_pairwise(pseudo_gt[:,:3,:3], pseudo_gt[:,:3,:3])
    min_dist = []

    for (i, d) in enumerate(distance):
        d_ = torch.cat((d[:i], d[(i+1):]))
        min_dist.append(torch.min(d_)[0])

    mean = np.mean(np.rad2deg(min_dist))

    return mean



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
