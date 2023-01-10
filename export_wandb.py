import wandb
import torch
import argparse
from pathlib import Path as P
import numpy as np

wandb_api = wandb.Api()

EXP_NAMES=["tabletop_3_bowl_uni_3", "tabletop_3_bowl_3", "tabletop_3_can_2", "tabletop_3_can_uni_1", "tabletop_3_crackerbox_1", "tabletop_3_crackerbox_uni_1"]
ROTATION_DIR = P("ipdf_se3/SO3_IPDF")
TRANSLATION_DIR = P("ipdf_se3/Translation_IPDF")
SAVE_PATH = P("output")


def save_data(variable_names, data, exp_name):

    f_name = str(SAVE_PATH/(exp_name+"_"+str(args.model)+".txt"))

    with open(f_name, "w") as f:
        f.write("\n")
        for (i, var) in enumerate(variable_names):
            f.write(var+":\n")
            for (j, dt) in enumerate(data[i]):
                f.write(f"({str(j+1)}, {str(dt)})\n")
            f.write("\n")




def export_data(variable_names, exp_name):

    if args.model==0:
        path = ROTATION_DIR / exp_name
    else:
        path = TRANSLATION_DIR / exp_name

    data_frame = wandb_api.run(str(path)).history()
    export_data = []

    for var in variable_names:
        data = data_frame[var].to_numpy().astype(float)
        data = data[~np.isnan(data)]
        
        export_data.append(data)
    
    save_data(variable_names, export_data, exp_name)

    return export_data



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-exp_name", type=str)
    parser.add_argument("-model", type=int)

    args = parser.parse_args()

    if args.model==0:
        var_names = ["TrainLoss", "Loglikelihood", "MeanAngularError", "RecallMeanAngularError"]
    elif args.model==1:
        var_names = ["TrainLoss", "Loglikelihood", "EstimateError"]


    for exp_name in EXP_NAMES:
        export_data(var_names, exp_name)


