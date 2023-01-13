import torch
from pathlib import Path as P
import os

import data
import config
EXP_NAME = []
OBJ_ID = []


def create_filesystem(exp_name):
    path = P("experiments") / ("exp_"+exp_name)
    os.makedirs(str(path/"dataset"))
    os.makedirs(str(path/"models"))
    os.makedirs(str(path/"visualizations"))

def load_config(obj_id):
    hyper_param = config.demo


if __name__=="__main__":

    for exp_name in EXP_NAME:
        create_filesystem(exp_name)


