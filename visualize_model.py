import torch
import argparse
import os 
import yaml

import se3_ipdf.models as models
import se3_ipdf.evaluation as evaluation
import data
import utils.visualizations as visualizations
import utils

EXP_NAME_LIST = [
  #"tabletop_4_can_uni_3",
  #"tabletop_4_can_3",
  "tabletop_4_can_occ_3",
  #"tabletop_4_can_single_2",
  #"tabletop_4_can_ana_3"




]
ROT_EPOCH_LIST = ["30"]*5
TRANS_EPOCH_LIST = ["final"]*5

def visualize(exp_name, mode):

    exp_dir = "experiments/exp_" + exp_name
    utils.set_random_seed(24)

    if mode==0:

        # Load the config file for the rotation model
        config_file_name = os.path.join(exp_dir, "config_rotation.yaml")
        with open(config_file_name, 'r') as f:
            hyper_param_rot = yaml.safe_load(f)

        # Load the config file for the translation model
        config_file_name = os.path.join(exp_dir, "config_translation.yaml")
        with open(config_file_name, 'r') as f:
            hyper_param_trans = yaml.safe_load(f)
        model = models.load_ensamble_model(hyper_param_rot=hyper_param_rot, hyper_param_trans=hyper_param_trans, arguments=args, exp_name=exp_name)
        obj_id = hyper_param_rot["obj_id"][0]
        dataset = data.load_single_model_dataset(hyper_param_rot, translation=True, validation_only=True)

        visualizations.visualize_ensamble_model(model=model, dataset=dataset, save_dir=os.path.join(exp_dir, "visualizations"), hyper_param=hyper_param_rot)


    elif mode==1:
        # Load the config file for the rotation model
        config_file_name = os.path.join(exp_dir, "config_rotation.yaml")
        with open(config_file_name, 'r') as f:
            hyper_param_rot = yaml.safe_load(f)

        
        model = models.load_rotation_model(hyper_param=hyper_param_rot, exp_name=exp_name, arguments=args)[0]
        dataset_list = data.load_model_dataset(hyper_param_rot, validation_only=True)
        for (i, dataset) in enumerate(dataset_list):
            obj_id = hyper_param_rot["obj_id"][i]
            visualizations.visualize_rotation_model(model=model, dataset=dataset, save_dir=os.path.join(exp_dir, f"visualizations/obj_0{obj_id}"), hyper_param=hyper_param_rot)
    
    elif mode==2:
        # Load the config file for the translation model
        config_file_name = os.path.join(exp_dir, "config_translation.yaml")
        with open(config_file_name, 'r') as f:
            hyper_param_trans = yaml.safe_load(f)

        
        model = models.load_translation_model(hyper_param=hyper_param_trans,exp_name=exp_name, arguments=args)[0]

        dataset = data.load_single_model_dataset(hyper_param_trans, validation_only=True)

        visualizations.visualize_translation_model(model=model, dataset=dataset, save_dir=os.path.join(exp_dir, "visualizations"), hyper_param=hyper_param_trans)

    else:
        print("\nPlease define a model to be loaded and evaluated!")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters for visualization of the model output")
    parser.add_argument("-mode", type=int, default=0)

    args = parser.parse_args()

    for (i, exp_name) in enumerate(EXP_NAME_LIST):
        args.rot_epoch = ROT_EPOCH_LIST[i]
        args.trans_epoch = TRANS_EPOCH_LIST[i]
        visualize(exp_name, args.mode)
    # Determine which model type is to be evaluated

