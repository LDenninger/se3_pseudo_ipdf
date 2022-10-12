import torch
import roma

def calc_angular_dist(rotations):
    prod = torch.einsum('aij,bkj->abik', rotations[:,:3,:3], rotations[:,:3,:3])
    trace = torch.einsum('baii->ba', prod)
    geo_dist = torch.arccos(torch.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    angular_dist = torch.rad2deg(geo_dist)

    return angular_dist

def check_duplicates(pseudo_gt, angular_threshold=3):
    cleaned_pseudo_gt = []
    ind_already_in = []
    
    angular_dist = calc_angular_dist(pseudo_gt)
   
    #angular_dist = geo_dist

    for (i, pgt) in enumerate(pseudo_gt):
  
        if not i in ind_already_in:
            under_threshold = torch.nonzero(angular_dist[i] <= angular_threshold)
            if len(under_threshold.shape)==2:
                under_threshold = under_threshold.squeeze()
            if len(under_threshold.shape)==0:
                under_threshold = under_threshold.unsqueeze(0)
            cleaned_pseudo_gt.append(pgt)
            ind_already_in += under_threshold.tolist()
    
    return torch.stack(cleaned_pseudo_gt)

def check_duplicates_averaging(pseudo_gt, angular_threshold=15):
    cleaned_pseudo_gt = []
    ind_already_in = []
    
    angular_dist = calc_angular_dist(pseudo_gt)
   
    #angular_dist = geo_dist

    for (i, pgt) in enumerate(pseudo_gt):
  
        if not i in ind_already_in:
            under_threshold = torch.nonzero(angular_dist[i] <= angular_threshold)
            if len(under_threshold.shape)==2:
                under_threshold = under_threshold.squeeze()
            if len(under_threshold.shape)==0:
                under_threshold = under_threshold.unsqueeze(0)
            close_pgt_set = torch.sum(pseudo_gt[under_threshold,:3,:3], dim=0)
            average_rotation = roma.special_procrustes(close_pgt_set)
            pgt[:3,:3] = average_rotation
            cleaned_pseudo_gt.append(pgt)
            ind_already_in += under_threshold.tolist()
    
    return torch.stack(cleaned_pseudo_gt)