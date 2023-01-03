import os
import yaml
from pathlib import Path as P
import torch
import torchvision
import ipdb
import pytorch3d.transforms as tt
import math
from tqdm import tqdm

import data


def script_1():
    EXP_NAME_LIST = ["tabletop_3_can_resnet18_0_4", "tabletop_3_can_resnet18_1_3", "tabletop_3_can_resnet18_2_3","tabletop_3_can_resnet50_3", "tabletop_3_can_convnextT_3", "tabletop_3_can_convnextS_3","tabletop_3_can_convnextB_3", "tabletop_3_can_vgg_3", 
                    "tabletop_3_crackerbox_resnet18_0_3", "tabletop_3_crackerbox_resnet18_1_3", "tabletop_3_crackerbox_resnet18_2_3","tabletop_3_crackerbox_resnet50_3", "tabletop_3_crackerbox_convnextT_3", "tabletop_3_crackerbox_convnextS_3","tabletop_3_crackerbox_convnextB_3", "tabletop_3_crackerbox_vgg_3", 
                    "tabletop_3_bowl_resnet18_0_3", "tabletop_3_bowl_resnet18_1_3", "tabletop_3_bowl_resnet18_2_3","tabletop_3_bowl_resnet50_3", "tabletop_3_bowl_convnextT_3", "tabletop_3_bowl_convnextS_3","tabletop_3_bowl_convnextB_3", "tabletop_3_bowl_vgg_3"]
    OBJ_ID = 3
    MATERIAL = False

    data = P(data.id_to_path[OBJ_ID] if MATERIAL else data.id_to_path_uniform[OBJ_ID])

    """for i in range(10000):
        ind = str(i).zfill(6)
        img = torch.load(str(data/P(ind)/P("rgb_tensor.pt")))

        name = f"output/rgb_{OBJ_ID}.png" if MATERIAL else f"output/rgb_{OBJ_ID}_uni.png"
        torchvision.utils.save_image(img.permute(2,0,1)/255., name)

        ipdb.set_trace()"""


    for n in EXP_NAME_LIST:

        f1 = P("experiments/") / P(("exp_"+n)) / P("config_rotation.yaml")
        f2 = P("experiments/") / P(("exp_"+n)) / P("config_translation.yaml")


        with open(f1, "r") as f:
            config = yaml.safe_load(f)
        
        config["num_epochs"] = 100
        #config["num_train_iter"] = 200
        #config["warmup_steps"] = 20


        with open(f1, "w") as f:
            config = yaml.safe_dump(config, f)
        
        with open(f2, "r") as f:
            config = yaml.safe_load(f)

        config["num_epochs"] = 30
        config["num_fourier_comp"] = 2
        config["num_val_iter"] = 20
        config["eval_freq"] = 100



        with open(f2, "w") as f:
            config = yaml.safe_dump(config, f)

def script_2():
    OBJ_ID = 5
    path = [P(data.id_to_path[OBJ_ID]), P(data.id_to_path_uniform[OBJ_ID])]
    for p in path:
        progress_bar = tqdm(range(15000), total=15000)
        for i in progress_bar:
            if i==15000:
                break
            index = str(i).zfill(6)
            dPath = p / index / "cleaned_pseudo_gt_thesis.pth"
            try:
                pgt = torch.load(str(dPath))
            except:
                print(f"pgt does not exist for frame {index}")

            correction_matrix = tt.euler_angles_to_matrix(torch.tensor([math.pi, 0,0]), "XYZ")
            pgt[:,:3,-1] = pgt[:,:3,-1] @ correction_matrix.T

            torch.save(pgt, str(dPath))

if __name__=="__main__":
    script_2()