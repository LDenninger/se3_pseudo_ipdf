import cv2
from matplotlib import pyplot as plt
import numpy as np
import stillleben as sl
sl.init_cuda()

import torch
import torchvision
import kornia as K
import ipdb
from kornia import morphology as morph
from pytorch3d.ops import knn_points


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
kernel = torch.ones(5, 5).to(DEVICE)


def imshow(input: torch.Tensor, name: str):
    out = torchvision.utils.make_grid(input, nrow=2, padding=5)
    out_np: np.ndarray = K.utils.tensor_to_image(out)
    plt.axis('off')
    plt.imsave(name, out_np)


def render_depth_image(obj_model, transformation, intrinsic, config):
    # Render the depth using the cad model
    K = intrinsic # (fx, fy, cx, cy)
    scene = sl.Scene(config["resolution"])
    scene.set_camera_intrinsics(*K)
    scene.ambient_light = torch.tensor([0.9, 0.9, 0.9])
    obj = sl.Object(obj_model)
    # import ipdb; ipdb.set_trace() 
    obj.set_pose(transformation.squeeze())
    scene.add_object(obj)
    renderer = sl.RenderPass()
    result = renderer.render(scene)

    depth = result.depth()
    depth[depth==3000] = 0

    return depth


def process_tensor(im_tensor):

    x_gray = im_tensor.unsqueeze(1)
    x_canny: torch.Tensor = K.filters.canny(x_gray)[0] / 10
    #print ('x_canny' , x_canny.min().item(), x_canny.max().item(), ) 
    x_canny = morph.dilation(x_canny, kernel)
    # imshow(x_canny.clamp(0., 1.))
    return x_canny


def binarize(img, threshold = 0.25):
    img [img > threshold] = 1
    img [img <= threshold] = 0.0
    return img

def matcher(img_1, img_2, threshold=None, verbose=False):
    idx_1 = torch.nonzero(img_1.squeeze(1)) # Nx3
    idx_2 = torch.nonzero(img_2.squeeze()).unsqueeze(0)

    B = img_1.shape[0]
    d_max = torch.zeros(B)
    d_avg = torch.zeros(B)

    if verbose:
        diff_img = torch.zeros_like(img_1)

    for b in range(B):
        _idx_1 = idx_1[ idx_1[:, 0] == b ][..., 1:].unsqueeze(0) # Nx3
        knn = knn_points(_idx_1.float(), idx_2.float())
        d_1 = knn.dists.squeeze()
        #print ('-'*10, d_1.max(), d_1.mean())
        knn = knn_points(idx_2.float(), _idx_1.float())
        d_2 = knn.dists.squeeze()

        d_max[b] = torch.max(torch.cat((d_1,d_2)))

        d_avg[b] =torch.max(torch.stack([d_1.mean(dim=-1), d_2.mean(dim=-1)]), dim=0)[0]

        if verbose:
            diff_img[b] = torch.zeros_like(img_1[b])
            diff_img[b,:,_idx_1[0,:,0],_idx_1[0,:,1]] = d_1
            diff_img[b,:,idx_2[0,:,0],idx_2[0,:,1]] += d_2
            diff_img[b] /= 2
            diff_img[b] -= diff_img[b].min()
            diff_img[b] /= diff_img[b].max()
            

    #print ('-'*10, d_2.max(), d_2.mean())
    #print ('-'*10,"Max: ", d_max,"Mean: ", d_avg)
    #print('-'*10, f" Treshold: max: {threshold[0]}, mean: {threshold[1]}")

    converged = torch.bitwise_and(d_max <= threshold[0], d_avg <= threshold[1])
    if verbose:
        imshow(diff_img, "output/tless_2/edge_diff.png")
        return converged, d_max, d_avg
    return converged

def check_convergence_batchwise(obj_model, transformation_set, depth_original, threshold, intrinsic, config):
    # Render the depth from the final pseudo transformation
    depth_rendered = []
    for (i, transformation) in enumerate(transformation_set):
        depth_rendered.append(render_depth_image(obj_model, transformation, intrinsic, config))
    depth_rendered = torch.stack(depth_rendered).to(DEVICE)
    depth_original = depth_original.unsqueeze(0).to(DEVICE)
    
    # Scale to cm
    depth_rendered = depth_rendered * 100 
    depth_original = depth_original / 10

    # Compute edge image of the final result using Canny edge detector
    edge_rendered = process_tensor(depth_rendered)
    edge_rendered = binarize(edge_rendered)   # binarize to get sharp edges

    edge_original = process_tensor(depth_original)
    edge_original = binarize(edge_original)

    # Match the edges and calculate the distance, determine if the alignment is tight enough

    if config["verbose"]:
        imshow(edge_original, "output/tless_2/edge_org.png")
        imshow(edge_rendered, "output/tless_2/edge_rend.png")

        return matcher(edge_rendered, edge_original, threshold, config["verbose"])

    return matcher(edge_rendered, edge_original, threshold)