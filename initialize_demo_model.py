import torch
from pathlib import Path as P
import os
import yaml

import data
import config
EXP_NAME = ["demonstration_can_1", "demonstration_box_1", "demonstration_bowl_1"]
OBJ_ID = [3,4,5]


def create_filesystem(exp_name):
    path = P("experiments") / ("exp_"+exp_name)
    os.makedirs(str(path))
    os.makedirs(str(path/"dataset"))
    os.makedirs(str(path/"models"))
    os.makedirs(str(path/"visualizations"))

def load_config(exp_name, obj_id):
    path = P("experiments") / ("exp_"+exp_name)
    hyper_param = config.load_demo_config(obj_id)

    try:
        with open(str(path/"config_rotation.yaml"), "w") as f:
            yaml.safe_dump(hyper_param, f)
    except:
        return False
    
    return True



if __name__=="__main__":

    for (i, exp_name) in enumerate(EXP_NAME):
        try:
            create_filesystem(exp_name)
            if not load_config(exp_name, OBJ_ID[i]):
                print("Config could not been loaded. Please add config file manually.")
        except:
            print("Something went wrong!")
        

