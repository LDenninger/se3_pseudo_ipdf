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

    EXP_NAME_LIST = [
        "tabletop_3_bowl_4", "tabletop_3_bowl_ana_2", "tabletop_3_bowl_single_2", "tabletop_3_bowl_uni_4",
        "tabletop_3_can_3", "tabletop_3_can_ana_2", "tabletop_3_can_single_2", "tabletop_3_can_uni_3",
        "tabletop_3_crackerbox_3", "tabletop_3_crackerbox_ana_2", "tabletop_3_crackerbox_single_2", "tabletop_3_crackerbox_uni_3"
    ]
    
    #EXP_NAME_LIST = [ "tabletop_3_bowl_ana_1", "tabletop_3_bowl_single_1", "tabletop_3_can_ana_1", "tabletop_3_can_single_1", "tabletop_3_crackerbox_ana_1", "tabletop_3_crackerbox_single_1"]

    #OBJ_ID = 3
    #MATERIAL = [True, False, True, True, False, True, True, False, True]


    """for i in range(10000):
        ind = str(i).zfill(6)
        img = torch.load(str(data/P(ind)/P("rgb_tensor.pt")))

        name = f"output/rgb_{OBJ_ID}.png" if MATERIAL else f"output/rgb_{OBJ_ID}_uni.png"
        torchvision.utils.save_image(img.permute(2,0,1)/255., name)

        ipdb.set_trace()"""


    for (i, n) in enumerate(EXP_NAME_LIST):

        f1 = P("experiments/") / P(("exp_"+n)) / P("config_rotation.yaml")
        f2 = P("experiments/") / P(("exp_"+n)) / P("config_translation.yaml")


        with open(f1, "r") as f:
            config = yaml.safe_load(f)
        
        #config["num_epochs"] = 100
        #config["num_train_iter"] = 200
        #config["warmup_steps"] = 20
        #config["num_val_iter"] = 40


        with open(f1, "w") as f:
            config = yaml.safe_dump(config, f)
        
        with open(f2, "r") as f:
            config = yaml.safe_load(f)
        #config["length"] = 15000
        #config["backbone"] = "convnext_tiny"
        ##config["obj_id"] = [config["obj_id"]]
        config["num_epochs"] = 30
        config["num_train_iter"] = 400
        #config["single_gt"] = False
        #config["pseudo_gt"] = True
        config["num_fourier_comp"] = 2
        config["num_val_iter"] = 30
        #config["eval_freq"] = 2



        with open(f2, "w") as f:
            config = yaml.safe_dump(config, f)

def script_2():
    OBJ_ID = 5
    import ipdb; ipdb.set_trace()
    path = [P(data.id_to_path[OBJ_ID])]
    for p in path:
        progress_bar = tqdm(range(15000), total=15000)
        for i in progress_bar:
            if i==15000:
                break
            index = str(i).zfill(6)
            dPath = p / index / "cleaned_pseudo_gt.pth"
            try:
                pgt = torch.load(str(dPath))
            except:
                continue
                print(f"pgt does not exist for frame {index}")
            
            if (pgt[:,2,-1]<=0).any():
                continue

            correction_matrix = tt.euler_angles_to_matrix(torch.tensor([math.pi, 0,0]), "XYZ")
            pgt[:,:3,-1] = torch.mean(pgt[:,:3,-1] @ correction_matrix.T,dim=0)

            torch.save(pgt, str(dPath))

def script_3():
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
                os.remove(str(dPath))
            except:
                continue

def script_4():
    OBJ_ID = 5
    path = [P(data.id_to_path_uniform[OBJ_ID])]
    import ipdb; ipdb.set_trace()
    for p in path:
        progress_bar = tqdm(range(15000), total=15000)
        for i in progress_bar:
            if i==15000:
                break
            index = str(i).zfill(6)
            dPath_1 = p / index / "cleaned_pseudo_gt_thesis.pth"
            dPath_2 = p / index / "cleaned_pseudo_gt.pth"

            try:
                pgt_1 = torch.load(str(dPath_1))
                pgt_2 = torch.load(str(dPath_2))

            except:
                print(f"pgt does not exist for frame {index}")
                continue

            trans_gt = torch.mean(pgt_2[:,:3,-1], dim=0)
            pgt_1[:,:3,-1] = torch.repeat_interleave(trans_gt.unsqueeze(0), pgt_1.shape[0], dim=0)

            torch.save(pgt_1, str(dPath_1)) 

def script_5():
    return None

if __name__=="__main__":
    script_1()