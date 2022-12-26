import torch
import argparse
import os 
import yaml

import se3_ipdf.models as models
import se3_ipdf.evaluation as evaluation
import data
import utils.visualizations as visualizations

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters for visualization of the model output")
    parser.add_argument("-exp_name", type=str, help="Name of the experiment")
    parser.add_argument("-rot_epoch", type=str, default=None, help="Epoch the checkpoint to load the rotation-model is taken from")
    parser.add_argument("-trans_epoch", type=str, default=None, help="Epoch the checkpoint to load the rotation-model is taken from")

    args = parser.parse_args()

    exp_dir = "experiments/exp_" + args.exp_name

    # Determine which model type is to be evaluated
    if args.rot_epoch is not None and args.trans_epoch is not None:

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


    elif args.rot_epoch is not None:
        # Load the config file for the rotation model
        config_file_name = os.path.join(exp_dir, "config_rotation.yaml")
        with open(config_file_name, 'r') as f:
            hyper_param_rot = yaml.safe_load(f)

        
        model = models.load_rotation_model(hyper_param=hyper_param_rot, exp_name=args.exp_name, arguments=args)[0]
        dataset_list = data.load_model_dataset(hyper_param_rot, validation_only=True)
        for (i, dataset) in enumerate(dataset_list):
            visualizations.visualize_rotation_model(model=model, dataset=dataset, save_dir=os.path.join(exp_dir, f"visualizations/obj_0{i+3}"), hyper_param=hyper_param_rot)
    
    elif args.trans_epoch is not None:
        # Load the config file for the translation model
        config_file_name = os.path.join(exp_dir, "config_translation.yaml")
        with open(config_file_name, 'r') as f:
            hyper_param_trans = yaml.safe_load(f)

        
        model = models.load_translation_model(hyper_param=hyper_param_trans,exp_name=args.exp_name, arguments=args)

        dataset = data.load_model_dataset(hyper_param_trans, validation_only=True)

        visualizations.visualize_translation_model(model=model, dataset=dataset, save_dir=os.path.join(exp_dir, "visualizations"), hyper_param=hyper_param_trans)[0]

    else:
        print("\nPlease define a model to be loaded and evaluated!")
