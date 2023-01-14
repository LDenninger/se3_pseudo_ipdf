import torch
from pathlib import Path as P

import data
EXP_NAME = ["demonstration_can_2", "demonstration_box_2", "demonstration_bowl_2"]


if __name__=="__main__":
        


    for (i, exp_name) in enumerate(EXP_NAME):

        path = P("experiments") / ("exp_"+exp_name)

        poses = data.generate_dataset(mode = 1)

        torch.save(poses, str(path/"dataset"/"train_dataset.pt"))

