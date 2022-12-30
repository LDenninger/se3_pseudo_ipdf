import os
import yaml
from pathlib import Path as P
import torch
import torchvision
import ipdb

import data

EXP_NAME_LIST = ["tabletop_3_can_resnet18_0_3", "tabletop_3_can_resnet18_1_2", "tabletop_3_can_resnet18_2_2","tabletop_3_can_resnet50_2", "tabletop_3_can_convnextT_2", "tabletop_3_can_convnextS_2","tabletop_3_can_convnextB_2", "tabletop_3_can_vgg_2", 
                "tabletop_3_crackerbox_resnet18_0_2", "tabletop_3_crackerbox_resnet18_1_2", "tabletop_3_crackerbox_resnet18_2_2","tabletop_3_crackerbox_resnet50_2", "tabletop_3_crackerbox_convnextT_2", "tabletop_3_crackerbox_convnextS_2","tabletop_3_crackerbox_convnextB_2", "tabletop_3_crackerbox_vgg_2", 
                "tabletop_3_bowl_resnet18_0_2", "tabletop_3_bowl_resnet18_1_2", "tabletop_3_bowl_resnet18_2_2","tabletop_3_bowl_resnet50_2", "tabletop_3_bowl_convnextT_2", "tabletop_3_bowl_convnextS_2","tabletop_3_bowl_convnextB_2", "tabletop_3_bowl_vgg_2"]
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
    
    config["num_train_iter"] = 200
    config["warmup_steps"] = 20


    with open(f1, "w") as f:
        config = yaml.safe_dump(config, f)
    
    with open(f2, "r") as f:
        config = yaml.safe_load(f)



    with open(f2, "w") as f:
        config = yaml.safe_dump(config, f)