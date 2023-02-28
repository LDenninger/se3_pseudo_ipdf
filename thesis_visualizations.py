import argparse
from tqdm import tqdm
import torch
import torchvision 
from pathlib import Path as P
import ipdb

import data
import utils
import utils.visualizations as visualizations

## Script to generate the visualizations used in the thesis ##

IMG_SIZE= (500,500)
DIR = P("output/thesis_demo")

def visualize_symmetries(obj_id, mode):
    dataset, dataset_size = data.load_demonstration_dataset(obj_id=obj_id, mode=mode, img_size=IMG_SIZE)

    progress_bar = tqdm(enumerate(dataset), total=dataset_size)

    for (i, input) in progress_bar:
        if i==dataset_size:
            break
        image = input["image"]
        pose_gt = input["pose_gt"]

        torchvision.utils.save_image(image, str(DIR/f"image_{str(i)}.png") )

        sym_set = utils.get_symmetry_ground_truth(pose_gt.squeeze()[:3,:3], obj_id=obj_id, dataset="tabletop")

        pose_gt = torch.repeat_interleave(pose_gt, sym_set.shape[0], dim=0)
        pose_gt[:,:3,:3] = sym_set

        visualizations.visualize_so3_rotations(
            rotations = pose_gt[:,:3,:3],
            obj_id=obj_id,
            save_path=str(DIR/f"rot_sym_{str(i)}.png")
        )
        visualizations.visualize_translation(
            translation_gt = pose_gt[0,:3,-1],
            save_path=str(DIR/f"trans_{str(i)}.png")
        )




if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters for training")
    parser.add_argument("-obj_id", type=int)
    parser.add_argument("-mode", type=int, default = 0)

    args = parser.parse_args()


    visualize_symmetries(obj_id = args.obj_id, mode = args.mode)

