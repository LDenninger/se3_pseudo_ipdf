
from tqdm import tqdm
import ipdb
import torch

from .so3_mollweide_projection import visualize_so3_rotations

def visualize_pseudo_gt(dataset, hyper_param, save_path):


    for (i, input) in  enumerate(dataset):

        
        idx = input["index"].squeeze(0)

        if not input["loaded"]:
            print(f"\nNo pseudo ground truth found for frame {idx}")
            continue
        
        print(f"\nVisualization of the pseudo ground truth of frame {idx}")
        dupl_rot = torch.flatten(torch.repeat_interleave(input["pseudo_gt"][...,:3,:3], 2, dim=0), 0, 1)
        visualize_so3_rotations(
            rotations=dupl_rot,
            obj_id=hyper_param["obj_id"],
            dataset=hyper_param["dataset"],
            rotations_gt=input["ground_truth"].squeeze()[:3,:3],
            display_gt_set=True,
            save_path=save_path
        )

        ipdb.set_trace()