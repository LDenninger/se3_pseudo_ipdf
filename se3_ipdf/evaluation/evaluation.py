import numpy as np
import torch
from tqdm import tqdm
import pytorch3d.transforms as tt
import pytorch3d.ops as ops
import json
import math


def eval_llh(model, dataset, mode=0,
                    eval_grid_size=2**15,
                    num_eval_iter=1,
                    verbose=False,
                    device= "cpu"):
    
    """Distribution-based evaluation functions for pose estimation.
    Arguments:
        model: The IPDF model.
        dataset: The dataset of images paired with single-valued ground truth
            rotations.
        batch_size: batch size for chunking up the evaluation operations.
        eval_grid_size: if supplied, sets the resolution of the grid to use for
            evaluation.
        skip_spread_eval: Whether to skip the spread calculation, which can
            take a while and is uninformative for the three shapes (tetX, cylO,
            sphereX) without the full set of ground truth annotations.
        number_eval_iter: stop evaluation after this number of steps.
    Returns:
        Average log likelihood and average spread (in degrees)
    """
    llh_rot_all = []
    llh_trans_all = []
    llh_all = []
    progress_bar = tqdm(enumerate(dataset), total=num_eval_iter)  
    for (step, batch) in progress_bar:
        if num_eval_iter is not None and step >= num_eval_iter:
            break
        
        images = batch['image'].to(device).float()
        trans_gt = batch['obj_pose_in_camera'][:,:3,-1].to(device).float()
        rot_gt = batch['obj_pose_in_camera'][:,:3,:3].to(device).float()
        with torch.no_grad():

            if mode==2:
                so3_grid, cartesian_grid, rotation_prob, translation_prob = model.output_pdf(images)

                max_inds = find_closest_rot_inds(so3_grid, rot_gt).squeeze()
                max_inds = np.array(max_inds.cpu())

                probabilities = rotation_prob.cpu().detach().numpy()

                prob_gt = np.float32([probabilities[i][max_inds[i]] for i in range (max_inds.shape[0])])
                llh = np.log(prob_gt*so3_grid.shape[0]/np.pi**2)
                llh_rot_all.append(llh)

                max_inds = find_closest_trans_inds(cartesian_grid, trans_gt).squeeze()
                max_inds = np.array(max_inds.cpu())

                probabilities = translation_prob.cpu().detach().numpy()

                prob_gt = np.float32([probabilities[i][max_inds[i]] for i in range (max_inds.shape[0])])
                llh = np.log(prob_gt*cartesian_grid.shape[0]/1**3)

                llh_trans_all.append(llh)

                continue

            if mode==0:
                so3_grid, probabilities = model.output_pdf(images)
                max_inds = find_closest_rot_inds(so3_grid, rot_gt).squeeze()
                max_inds = np.array(max_inds.cpu())

                probabilities = probabilities.cpu().detach().numpy()

                prob_gt = np.float32([probabilities[i][max_inds[i]] for i in range (max_inds.shape[0])])
                prob_gt = np.clip(prob_gt, a_min=1e-10, a_max=10000000)
                llh = np.log(prob_gt*so3_grid.shape[0]/np.pi**2)
            
            if mode==1:
                cartesian_grid, probabilities = model.output_pdf(images)
                max_inds = find_closest_trans_inds(cartesian_grid, trans_gt)
                max_inds = np.array(max_inds.cpu())
        
                probabilities = probabilities.cpu().detach().numpy()

                prob_gt = np.float32([probabilities[i][max_inds[i]] for i in range (max_inds.shape[0])])
                prob_gt = np.clip(prob_gt, a_min=1e-10, a_max=10000000)
                llh = np.log(prob_gt*cartesian_grid.shape[0]/1**3)

            llh_all.append(llh)

        if verbose == True:
            print("Evaluation step ", step+1,": loglikelihood: ", np.mean(llh))
    if mode==0 or mode==1:
        return np.mean(llh_all)
    if mode==2:
        llh_all = llh_rot_all + llh_trans_all
        return np.mean(llh_rot_all), np.mean(llh_trans_all), np.mean(llh_all)

def eval_recall_error(model, dataset,
                    hyper_param,
                    eval_grid_size=2**15,
                    threshold=1e-5):
    """Evaluation of the mean/median angular error of a set of symmetric poses to multiple pose estimates.
    
    """
    device = next(model.parameters()).device

    # Get symmetries for the T-Less object
    if hyper_param["dataset"] == "tless":
        obj_sym = []
        sym_set = produce_ground_truth_set_tless(hyper_param["obj_id"])
        for sym in sym_set:
            obj_sym.append(torch.from_numpy(sym['R']))
        obj_sym = torch.stack(obj_sym).float().to(device)

    error_mean = []
    error_median = []

    progress_bar = tqdm(enumerate(dataset), total=hyper_param['num_val_iter'])
    with torch.no_grad():
        for (i, input_) in progress_bar:
            if i == hyper_param['num_val_iter']:
                break      
            images = input_['image'].to(device)
            pose_gt = input_['obj_pose_in_camera'].to(device)
            rot_gt = input_['obj_pose_in_camera'][:,:3,:3].to(device)
            
            if hyper_param["dataset"] == "tabletop":
                obj_id = input_["obj_id"][0]
                ground_truth_set = produce_ground_truth_set(rot_gt, obj_id)
            elif hyper_param["dataset"] == "tless":
                obj_id = input_["obj_id"]
                ground_truth_set = torch.einsum('bij,ajk->baik', rot_gt.float(), obj_sym)


            so3_grid, probabilities = model.output_pdf(images)

            prediction_idx = probabilities > threshold
            
            if prediction_idx.int().max()==0:
                break
            for j in range(probabilities.shape[0]):
                idx = torch.nonzero(prediction_idx[j]).squeeze()
                    
                geodesic_dist = geo_dist_pairwise(ground_truth_set[j], so3_grid[idx])
                try:
                    min_geodesic_dist = torch.min(geodesic_dist, dim=-1)[0]
                except:
                    break
                error_mean.append(torch.mean(min_geodesic_dist))
                error_median.append(torch.median(min_geodesic_dist))

    if error_mean==[] or error_median==[]:
        return -1, -1

    return np.rad2deg(torch.mean(torch.stack(error_mean)).item()), np.rad2deg(torch.mean(torch.stack(error_median)).item())


def eval_accuracy_angular_error(model, dataset,
                                    hyper_param,
                                    gradient_ascent=False):
    device = next(model.parameters()).device
    
    if hyper_param["dataset"] == "tless":
        obj_sym = []
        sym_set = produce_ground_truth_set_tless(hyper_param["obj_id"])
        for sym in sym_set:
            obj_sym.append(torch.from_numpy(sym['R']))
        obj_sym = torch.stack(obj_sym).float().to(device)

    geodesic_errors = []
    progress_bar = tqdm(enumerate(dataset), total=hyper_param['num_val_iter'])
    estimations = []

    for (i, input_) in progress_bar:
        if i == hyper_param['num_val_iter']:
            break

        img = input_['image'].to(device)
        rot_gt = input_['obj_pose_in_camera'][:,:3,:3].to(device)
        pose_gt = input_['obj_pose_in_camera'].to(device)
        obj_id = input_["obj_id"][0]
        if hyper_param["dataset"] == "tabletop":
            ground_truth_set = produce_ground_truth_set(rot_gt,obj_id)
        elif hyper_param["dataset"] == "tless":
            ground_truth_set = torch.einsum('bij,ajk->baik', rot_gt.float(), obj_sym)


        model_estimation = model.predict_rotation(img, gradient_ascent=gradient_ascent)
        for j in range(model_estimation.shape[0]):
            geodesic_dist = geo_dist(torch.repeat_interleave(model_estimation[j].unsqueeze(0), ground_truth_set.shape[1], dim=0), ground_truth_set[j]).detach().cpu().numpy()
            min_geo_dist = np.min(geodesic_dist)
            geodesic_errors.append(min_geo_dist)

    geodesic_errors = np.rad2deg(geodesic_errors)
    mean_angular_error = np.mean(geodesic_errors)
    accuracy5 = np.average(geodesic_errors <= 5)
    accuracy15 = np.average(geodesic_errors <= 15)
    accuracy30 = np.average(geodesic_errors <= 30)


    return mean_angular_error, accuracy5, accuracy15, accuracy30                


def eval_spread(model, dataset,
                    hyper_param,
                    eval_grid_size=2**15):
    device = next(model.parameters()).device
    so3_grid = model._generate_queries(eval_grid_size, mode='grid')
    spreads = []

    if hyper_param["dataset"] == "tless":
        obj_sym = []
        sym_set = produce_ground_truth_set_tless(hyper_param["obj_id"])
        for sym in sym_set:
            obj_sym.append(torch.from_numpy(sym['R']))
        obj_sym = torch.stack(obj_sym).float().to(device)

    progress_bar = tqdm(enumerate(dataset), total=hyper_param['num_val_iter'])
    with torch.no_grad():
        for (i, input_) in progress_bar:
            if i == hyper_param['num_val_iter']:
                break
            images = input_['image'].to(device)
            pose_gt = input_['obj_pose_in_camera'].to(device)
            rot_gt = input_['obj_pose_in_camera'][:,:3,:3].to(device)

            if hyper_param["dataset"] == "tabletop":
                ground_truth_set = produce_ground_truth_set(rot_gt,hyper_param["obj_id"])
            elif hyper_param["dataset"] == "tless":
                ground_truth_set = torch.einsum('bij,ajk->baik', rot_gt.float(), obj_sym)


            bs, num_init = ground_truth_set.shape[:2]
            so3_grid = so3_grid.to(device)
            probabilities = get_prob(model, images, so3_grid)
            probabilities = probabilities.to(device)
            ground_truth_set = torch.flatten(ground_truth_set, 0, 1)
            geodesic_dist = geo_dist_pairwise(ground_truth_set, so3_grid)
            ground_truth_set = ground_truth_set.cpu()
            so3_grid = so3_grid.cpu()
            geodesic_dist = torch.reshape(geodesic_dist, (bs, num_init, so3_grid.shape[0]))

            min_geodesic_dist = torch.min(geodesic_dist, dim=-1)
            for j in range(probabilities.shape[0]):
                spread = torch.sum(probabilities[j,min_geodesic_dist[1][j]]*min_geodesic_dist[0][j])
                spreads.append(spread.cpu())


    return torch.mean(torch.stack(spreads)).item()

        

def eval_adds(model, dataset,
                batch_size,
                model_points,
                threshold_list,
                eval_iter,
                diameter,
                mode=0,
                gradient_ascent=False,
                device="cpu"):
    # Take median from the k lowest/highest values to prevent the impact of unique outliers

    model_points = torch.repeat_interleave(model_points.unsqueeze(0), batch_size, dim=0)                
    progress_bar = tqdm(enumerate(dataset), total=eval_iter)

    threshold_distance = torch.arange(0.001,0.1, 0.001)
    num_examples = batch_size * eval_iter
    
    adds_distance_list = []
    adds_results = []

    model_points = model_points.to(device).float()

    for (i, input) in progress_bar:
        if i == eval_iter:
            break
        images = input['image'].float().to(device)
        pose_gt = input['obj_pose_in_camera'].float().to(device)

        if mode==0:
            rotation_estimation = model.predict_rotation(images, gradient_ascent)
            model_estimation = pose_gt
            model_estimation[:,:3,:3] = rotation_estimation
            
        elif mode==1:
            translation_estimation = model.predict_translation(images, gradient_ascent)
            model_estimation = pose_gt
            model_estimation[:,:3, -1] = translation_estimation
        else:
            image_trans = input['image_trans_input'].to(device)
            model_estimation = model.predict_pose(image_rot=images, image_trans=image_trans, gradient_ascent=gradient_ascent)

        # Transform point clouds
        pc_ground_truth = torch.bmm(model_points, torch.transpose(pose_gt[:,:3,:3], -2, -1)) + pose_gt[:,:3,-1].unsqueeze(1)
        pc_rendered = torch.bmm(model_points, torch.transpose(model_estimation[:,:3,:3], -2, -1)) + model_estimation[:,:3,-1].unsqueeze(1)
        adds_distance, idx, nn = ops.knn_points(pc_ground_truth, pc_rendered)
        adds_distance = torch.sqrt(adds_distance)
        adds_distance = torch.mean(adds_distance.squeeze(-1), dim=-1)
        #visualize_pointclouds(pc_ground_truth.squeeze().cpu(), pc_rendered.squeeze().cpu(), filename="output/pc_test.png")
        
        adds_distance_list.append(adds_distance)
    adds_distances = torch.stack(adds_distance_list)
    mean_distance = torch.mean(adds_distances).item()

    for threshold in threshold_distance:

        num_under_threshold = adds_distances <= threshold
        num_under_threshold = torch.count_nonzero(num_under_threshold)
        num_under_threshold = num_under_threshold / num_examples
        adds_results.append(num_under_threshold.item())
    
    return adds_results, threshold_distance, mean_distance

def eval_translation_error(model, dataset,
                            eval_iter,
                            batch_size,
                            eval_accuracy=False,
                            model_points=None,
                            threshold_list=[],
                            gradient_ascent=True):
    def euclidean_distance(p_1, p_2):
        dist = (p_1-p_2)**2
        dist = torch.sqrt(torch.sum(dist, dim=-1))
        return dist
    
    device = next(model.parameters()).device

    
    distance_list = []
    progress_bar = tqdm(enumerate(dataset), total=eval_iter)
    for (i, input) in progress_bar:
        if i==eval_iter:
            break
        image = input['image'].to(device)
        trans_gt = input['obj_pose_in_camera'][:,:3,-1].to(device)

        model_estimation = model.predict_translation(image, gradient_ascent)

        distance = euclidean_distance(trans_gt, model_estimation)
        distance_list.append(distance)
    
    translation_error = torch.stack(distance_list)
    if eval_accuracy:
        assert model_points is not None

        num_examples = batch_size * eval_iter

        maximum = torch.median(torch.topk(model_points[:,0], k=5)[0])
        minimum = torch.median(torch.topk(model_points[:,0], k=5, largest=False)[0])       

        model_diameter = maximum-minimum
        threshold_distance = torch.FloatTensor(threshold_list) * model_diameter

        accuracy_list = []
        for threshold in threshold_distance:

            accuracy_at_t = translation_error <= threshold
            accuracy_at_t = torch.count_nonzero(accuracy_at_t)
            accuracy_at_t = accuracy_at_t / num_examples

            accuracy_list.append(accuracy_at_t.item())
        
        return torch.mean(translation_error).item(), threshold_distance, accuracy_list
        

    
    return torch.mean(translation_error).item()

def scale_to_original(grid):
    trans_range = torch.tensor([[-0.1,0.1],[-0.1,0.1],[-0.1,0.1]])

    grid[:,-1,0] = 0.2*torch.sub(grid[:,-1,0], torch.min(grid[:,-1,0]))/(torch.max(grid[:,-1,0])-torch.min(grid[:,-1,0]))+(-0.1)
    grid[:,-1,1] = 0.2*torch.sub(grid[:,-1,1], torch.min(grid[:,-1,1]))/(torch.max(grid[:,-1,1])-torch.min(grid[:,-1,1]))+(-0.1)
    grid[:,-1,2] = 0.2*torch.sub(grid[:,-1,2], torch.min(grid[:,-1,2]))/(torch.max(grid[:,-1,2])-torch.min(grid[:,-1,2]))+(-0.1)
    
    return grid

def get_prob(model, images, grid):
    """"Returns the probabilites of the query rotations without scaling"""
    device = images.device

    query_rotations = torch.reshape(grid, [-1, model.query_dim]).to(device)
    query_rotations = model._positional_encoding(query_rotations).to(device)
    query_rotations = torch.repeat_interleave(query_rotations.unsqueeze(dim=0), images.shape[0], dim=0)
    logits = torch.squeeze(model((images.float(), query_rotations.float())), dim=-1)
    return torch.nn.Softmax(dim=-1)(logits)

def get_cuboid_syms():
    cuboid_seeds = [np.eye(3)]
    cuboid_seeds.append(np.float32([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]))
    for i in range(3):
        cuboid_seeds.append(np.diag(np.roll([-1, -1, 1], i)))

    cuboid_syms = []
    for rotation_matrix in cuboid_seeds:
        cuboid_syms.append(rotation_matrix)
        cuboid_syms.append(np.roll(rotation_matrix, 1, axis=0))
        cuboid_syms.append(np.roll(rotation_matrix, -1, axis=0))
    get_cuboid_syms = np.stack(cuboid_syms, 0)
    
    return get_cuboid_syms

def produce_ground_truth_set(rotation_gt, obj_id, num=200):

    device = rotation_gt.device

    if obj_id == 3:
        device = rotation_gt.device
        rot_mag = torch.linspace(-np.pi,np.pi, steps=int(num/2))
        rotation = torch.zeros(int(num/2),3)
        rotation[:,0] = rot_mag
        rotation = tt.euler_angles_to_matrix(rotation, 'ZYX').to(device).float()
        flip = tt.euler_angles_to_matrix(torch.repeat_interleave(torch.tensor([[0, 0, np.pi]]),int(num/2),dim=0), 'ZYX').to(device).float()
        rotation_flip = flip @ rotation
        sym = torch.cat([rotation, rotation_flip]).to(device)
        ground_truth_set = torch.einsum('bij,ajk->baik', rotation_gt.float(), sym)
    if obj_id==4 or obj_id==6:
        syms = torch.zeros(4,3,3)

        syms[0] = torch.eye(3)
        syms[1] = tt.euler_angles_to_matrix(torch.Tensor([0,0, np.pi]), 'ZYX').float()
        syms[2] = tt.euler_angles_to_matrix(torch.Tensor([0,np.pi,0 ]), 'ZYX').float()
        syms[3] = syms[1] @ syms[2]
        syms = syms.to(device)
        ground_truth_set = torch.einsum('bij,ajk->baik', rotation_gt.float(), syms)


    if obj_id==5:

        rot_mag = torch.linspace(-np.pi,np.pi, steps=int(num/2))
        rotation = torch.zeros(int(num/2),3)
        rotation[:,0] = rot_mag
        rotation = tt.euler_angles_to_matrix(rotation, 'ZYX')
        ground_truth_set = rotation.to(device)
        ground_truth_set = torch.einsum('bij,ajk->baik', rotation_gt.float(), ground_truth_set)



    return ground_truth_set

def produce_ground_truth_set_tless(obj_id, cont_step=0.01):
    """Returns a set of symmetry transformations for an object model.
    :param model_info: See files models_info.json provided with the datasets.
    :param max_sym_disc_step: The maximum fraction of the object diameter which
        the vertex that is the furthest from the axis of continuous rotational
        symmetry travels between consecutive discretized rotations.
    :return: The set of symmetry transformations.
    """
    ## External helper functions ##
    def unit_vector(data, axis=None, out=None):
        """Return ndarray normalized by length, i.e. Euclidean norm, along axis.
        """
        if out is None:
            data = np.array(data, dtype=np.float64, copy=True)
            if data.ndim == 1:
                data /= math.sqrt(np.dot(data, data))
            return data
        else:
            if out is not data:
                out[:] = np.array(data, copy=False)
                data = out
        length = np.atleast_1d(np.sum(data * data, axis))
        np.sqrt(length, length)
        if axis is not None:
            length = np.expand_dims(length, axis)
        data /= length
        if out is None:
            return data


    def rotation_matrix(angle, direction, point=None):
        """Return matrix to rotate about axis defined by point and direction.
        """
        sina = math.sin(angle)
        cosa = math.cos(angle)
        direction = unit_vector(direction[:3])
        # rotation matrix around unit vector
        R = np.diag([cosa, cosa, cosa])
        R += np.outer(direction, direction) * (1.0 - cosa)
        direction *= sina
        R += np.array([[0.0, -direction[2], direction[1]],
                            [direction[2], 0.0, -direction[0]],
                            [-direction[1], direction[0], 0.0]])
        M = np.identity(4)
        M[:3, :3] = R
        if point is not None:
            # rotation not around origin
            point = np.array(point[:3], dtype=np.float64, copy=False)
            M[:3, 3] = point - np.dot(R, point)
        return M


    ## External loading script taken from the BOP-toolkit repository ##
    # Load the model infos containing the symmetries
    model_info_path = "/home/nfs/inf6/data/datasets/T-Less/t-less-bop/models_cad/models_info.json"
    with open(model_info_path, "r") as f:
      model_info = json.load(f)
    model_info = model_info[str(obj_id)]

    # Discrete symmetries.
    trans_disc = [{'R': np.eye(3), 't': np.array([[0, 0, 0]]).T}]  # Identity.
    if 'symmetries_discrete' in model_info:
        for sym in model_info['symmetries_discrete']:
            sym_4x4 = np.reshape(sym, (4, 4))
            R = sym_4x4[:3, :3]
            t = sym_4x4[:3, 3].reshape((3, 1))
            trans_disc.append({'R': R, 't': t})

    # Discretized continuous symmetries.
    trans_cont = []
    if 'symmetries_continuous' in model_info:
        for sym in model_info['symmetries_continuous']:
            axis = np.array(sym['axis'])
            offset = np.array(sym['offset']).reshape((3, 1))

        # (PI * diam.) / (max_sym_disc_step * diam.) = discrete_steps_count
        discrete_steps_count = int(np.ceil(np.pi / cont_step))

        # Discrete step in radians.
        discrete_step = 2.0 * np.pi / discrete_steps_count

        for i in range(1, discrete_steps_count):
            R = rotation_matrix(i * discrete_step, axis)[:3, :3]
            t = -R.dot(offset) + offset
            trans_cont.append({'R': R, 't': t})

    # Combine the discrete and the discretized continuous symmetries.
    trans = []
    for tran_disc in trans_disc:
        if len(trans_cont):
            for tran_cont in trans_cont:
                R = tran_cont['R'].dot(tran_disc['R'])
                t = tran_cont['R'].dot(tran_disc['t']) + tran_cont['t']
                trans.append({'R': R, 't': t})
        else:
            trans.append(tran_disc)

    return trans


def min_geo_dist(r1, r2):
    """Computes pairwise geodesic distances between two sets of rotation matrices.
    Argumentss:
        r1: [N, 3, 3] numpy array
        r2: [M, 3, 3] numpy array
    Returns:
        [N, M] angular distances.
    """
    trace = torch.einsum('aij,bij->ab', r1, r2)
    max_trace = torch.max(trace,dim=-1)[0]
    return torch.arccos(torch.clip((max_trace - 1.0) / 2.0, -1.0, 1.0))

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


def translation_dist(gt_matrices, grid):
    eucl_dist = torch.cdist(gt_matrices, grid[:,-1])
    return eucl_dist

def find_closest_rot_inds(rot_grid, rot_gt):
    device = rot_gt.device
    rot_grid = rot_grid.to(device)
    if len(rot_gt.shape)==3:
        rot_gt = torch.unsqueeze(rot_gt, 0)

    traces = torch.einsum('gij,lkij->glk', rot_grid, rot_gt)
    max_i = torch.argmax(traces, dim=0)
    return max_i

def find_closest_trans_inds(trans_grid, trans_gt):
    device = trans_gt.device
    trans_grid = trans_grid.to(device)
    dist = torch.cdist(trans_gt, trans_grid)
    min_i = torch.argmin(dist, dim=-1)
    return min_i


def find_closest_transform_inds(grid, ground_truth):
    def norm(input):
        bs = input.shape[0]
        grid_size = input.shape[1]
       
        inp_min = torch.repeat_interleave(torch.min(input, dim=-1)[0].unsqueeze(1), grid_size, dim=1) 
        inp_max = torch.repeat_interleave(torch.max(input, dim=-1)[0].unsqueeze(1), grid_size, dim=1) 
        input = (input-inp_min)/(inp_max-inp_min)
        return input
    def dist(query, ground_truth):
        dist = torch.cdist(ground_truth.unsqueeze(0), query.unsqueeze(0))
        return dist.squeeze(0)
    device = ground_truth.device
    rot_grid = grid[:,:3,:3].to(device)
    trans_grid = grid[:,-1,:].to(device)
    #if len(ground_truth.shape)==3:
    #    ground_truth = torch.unsqueeze(ground_truth, 0)
    rot_dist = geo_dist_pairwise(ground_truth[:,:3,:3], rot_grid)
    norm_rot_dist = norm(rot_dist)

    trans_dist = dist(trans_grid, ground_truth[:,:3,-1])
    norm_trans_dist = norm(trans_dist)
    dist = torch.add(rot_dist, trans_dist,  alpha=1)
    min_i = torch.argmin(dist, dim=-1)

    return min_i


