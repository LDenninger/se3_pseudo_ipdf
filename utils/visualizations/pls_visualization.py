
from tqdm import tqdm
import ipdb

from .so3_mollweide_projection import visualize_so3_rotations

def visualize_pseudo_gt(dataset, hyper_param, save_path):


    for (i, input) in  enumerate(dataset):
        
        idx = input["index"].squeeze(0)
        print(f"\nVisualization of the pseudo ground truth of frame {idx}")
        visualize_so3_rotations(
            rotations=input["pseudo_gt"].squeeze(0)[...,:3,:3],
            obj_id=hyper_param["obj_id"],
            dataset=hyper_param["dataset"],
            rotations_gt=input["ground_truth"].squeeze()[:3,:3],
            display_gt_set=True,
            save_path=save_path
        )

        ipdb.set_trace()