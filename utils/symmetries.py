import torch
import numpy as np
import pytorch3d.transforms as tt
import math
import json


def get_symmetry_ground_truth(rotation_gt, obj_id, dataset, num=720):

    if dataset=="tabletop":
        if obj_id==3:
            flip = tt.euler_angles_to_matrix(torch.repeat_interleave(torch.tensor([[0, 0, np.pi]]),int(num/2),dim=0), 'ZYX').float()
            rot_mag = torch.linspace(-np.pi,np.pi, steps=int(num/2))
            rotation = torch.zeros(int(num/2),3)
            rotation[:,0] = rot_mag
            rotation = tt.euler_angles_to_matrix(rotation, 'ZYX')
            rotation_flip = flip @ rotation
            ground_truth_set = torch.cat([rotation, rotation_flip])
            ground_truth_set = rotation_gt.float() @ ground_truth_set
            
        if obj_id==4:
            syms = torch.zeros(4,3,3)
            syms[0] = torch.eye(3)
            syms[1] = tt.euler_angles_to_matrix(torch.Tensor([0,0, np.pi]), 'ZYX').float()
            syms[2] = tt.euler_angles_to_matrix(torch.Tensor([0,np.pi,0 ]), 'ZYX').float()
            syms[3] = syms[1] @ syms[2]
            ground_truth_set = rotation_gt.float() @ syms
            

        if obj_id==5:
            rot_mag = torch.linspace(-np.pi,np.pi, steps=int(num/2))
            rotation = torch.zeros(int(num/2),3)
            rotation[:,0] = rot_mag
            rotation = tt.euler_angles_to_matrix(rotation, 'ZYX')
            ground_truth_set = rotation
            ground_truth_set = rotation_gt.float() @ ground_truth_set
        
        if obj_id==8:
            ground_truth_set = rotation_gt

    elif dataset=="tless":
        obj_sym = []
        sym_set = produce_ground_truth_set_tless(obj_id)
        for sym in sym_set:
            obj_sym.append(torch.from_numpy(sym['R']))
        obj_sym = torch.stack(obj_sym).float()
        ground_truth_set = rotation_gt.float() @ obj_sym

    return ground_truth_set

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