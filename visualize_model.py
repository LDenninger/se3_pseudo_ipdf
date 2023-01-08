import torch
import argparse
import os 
import yaml

import se3_ipdf.models as models
import se3_ipdf.evaluation as evaluation
import data
import utils.visualizations as visualizations

EXP_NAME_LIST = [
    "tabletop_3_bowl_4",
    "tabletop_3_bowl_single_2",
    "tabletop_3_bowl_uni_4",
     "tabletop_3_can_3",
    "tabletop_3_can_uni_3",
     "tabletop_3_crackerbox_3",
    "tabletop_3_crackerbox_single_2",

]
ROT_EPOCH_LIST = ["10", "20", "10", "40", "40", "40", "40"]
TRANS_EPOCH_LIST = ["20", "10", "20", "20", "20", "20", "20"]

def visualize(exp_name, rot_epoch, trans_epoch, mode):

    exp_dir = "experiments/exp_" + exp_name


    if mode==0:

        # Load the config file for the rotation model
        config_file_name = os.path.join(exp_dir, "config_rotation.yaml")
        with open(config_file_name, 'r') as f:
            hyper_param_rot = yaml.safe_load(f)

        # Load the config file for the translation model
        config_file_name = os.path.join(exp_dir, "config_translation.yaml")
        with open(config_file_name, 'r') as f:
            hyper_param_trans = yaml.safe_load(f)
        
        model = models.load_ensamble_model(hyper_param_rot=hyper_param_rot, hyper_param_trans=hyper_param_trans, arguments=args)

        dataset = data.load_dataset(hyper_param_rot, validation_only=True)

        visualizations.visualize_ensamble_model(model=model, dataset=dataset, hyper_param=hyper_param_rot)


    elif mode==1:
        # Load the config file for the rotation model
        config_file_name = os.path.join(exp_dir, "config_rotation.yaml")
        with open(config_file_name, 'r') as f:
            hyper_param_rot = yaml.safe_load(f)

        
        model = models.load_rotation_model(hyper_param=hyper_param_rot, exp_name=args.exp_name, arguments=args)[0]
        dataset_list = data.load_model_dataset(hyper_param_rot, validation_only=True)
        for (i, dataset) in enumerate(dataset_list):
            obj_id = hyper_param_rot["obj_id"][i]
            visualizations.visualize_rotation_model(model=model, dataset=dataset, save_dir=os.path.join(exp_dir, f"visualizations/obj_0{obj_id}"), hyper_param=hyper_param_rot)
    
    elif mode==2:
        # Load the config file for the translation model
        config_file_name = os.path.join(exp_dir, "config_translation.yaml")
        with open(config_file_name, 'r') as f:
            hyper_param_trans = yaml.safe_load(f)

        
        model = models.load_translation_model(hyper_param=hyper_param_trans,exp_name=args.exp_name, arguments=args)[0]

        dataset = data.load_model_dataset(hyper_param_trans, validation_only=True)

        visualizations.visualize_translation_model(model=model, dataset=dataset[0], save_dir=os.path.join(exp_dir, "visualizations"), hyper_param=hyper_param_trans)

    else:
        print("\nPlease define a model to be loaded and evaluated!")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters for visualization of the model output")
    parser.add_argument("-mode", type=int, default=0)

    args = parser.parse_args()

    for (i, exp_name) in enumerate(EXP_NAME_LIST):
        visualize(exp_name, ROT_EPOCH_LIST[i], TRANS_EPOCH_LIST[i], args.mode)
    # Determine which model type is to be evaluated

