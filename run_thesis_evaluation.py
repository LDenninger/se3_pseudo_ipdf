import torch
from pathlib import Path as P
import yaml
import os

import se3_ipdf.models as models
import se3_ipdf.evaluation as evaluation
import utils.visualizations as vis
import data
import utils

## Script to run the evaluation performed for the thesis ##


BASE_DIR = "experiments/aa_evaluation_models"



def save_adds(adds, threshold, auc, save_dir):
    assert len(adds)==len(threshold)
    f_name = str(P(save_dir)/"adds_.txt")

    with open(f_name, "w") as f:
        f.write("\n")
        f.write(f"AUC: {str(auc)}\n")
        for (i, adds) in enumerate(adds):
            f.write(f"({str(threshold[i].item())}, {str(adds)})\n")
    

def model_evaluation(exp_dir):

    if os.path.exists(str(exp_dir/"evaluations"/"adds.txt")):
        return False
    import ipdb; ipdb.set_trace()

    utils.set_random_seed(42)

    print("_"*60)
    print(f"Starting evaluation for model under path: {str(exp_dir)}")

   
    config_file_name = os.path.join(str(exp_dir/ "models" /"config_rotation.yaml"))
    with open(config_file_name, 'r') as f:
        hyper_param_rot = yaml.safe_load(f)

    # Load the config file for the translation model
    config_file_name = os.path.join(str(exp_dir / "models" / "config_translation.yaml"))
    with open(config_file_name, 'r') as f:
        hyper_param_trans = yaml.safe_load(f)

    # Load the object model
    if hyper_param_rot["dataset"]=="tabletop":
        obj_model, diameter = data.load_ycbv_object_model(hyper_param_rot["obj_id"][0], pointcloud_only=True)
    elif hyper_param_rot["dataset"]=="tless":
        obj_model, diameter = data.load_tless_object_model(hyper_param_rot["obj_id"], pointcloud_only=True)

    model = models.load_evaluation_model(hyper_param_rot, hyper_param_trans, exp_dir)

    dataset = data.load_single_model_dataset(hyper_param=hyper_param_rot, validation_only=True, translation=True)

    #vis.visualize_ensamble_model(model=model, dataset=dataset, save_dir= str(exp_dir / "visualizations"), hyper_param=hyper_param_rot)

    print("Visualization done!")

    adds, threshold, mean_adds, auc = evaluation.adds_evaluation(model=model, dataset=dataset, hyper_param_rot=hyper_param_rot, hyper_param_trans=hyper_param_trans, diameter=diameter, model_points=obj_model)

    save_adds(adds, threshold, auc, save_dir = str(exp_dir / "evaluations"))

    return True

if __name__=="__main__":

    exp_dir_list = P(BASE_DIR).glob("*")

    utils.set_random_seed(42)
    for exp_dir in exp_dir_list:
        try:
            model_evaluation(exp_dir)
        except:
            print("Model failed to be evaluated ")
            continue        