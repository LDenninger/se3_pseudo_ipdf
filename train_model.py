import argparse
import wandb
import yaml
import os
import torch

import utils
import data
import config
import se3_ipdf
import se3_ipdf.models as models


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Define multiple pre-defined experiments to run, automatically used, if no experiment is provided as an argument ##
EXP_NAME_LIST = ["tabletop_2_can_6", "tabletop_2_can_occ_7", "tabletop_2_can_res1_6", "tabletop_2_can_res2_8", "tabletop_2_bowl_7", "tabletop_2_bowl_occ_7", "tabletop_2_bowl_res1_7", "tabletop_2_bowl_res2_6", "tabletop_2_crackerbox_5", "tabletop_2_crackerbox_occ_5", "tabletop_2_crackerbox_res1_5", "tabletop_2_crackerbox_res2_5"]
MODEL_TYPE = [0]*12


def train_model():
    wandb.login()
    # Set up Weights'n'Biases logging
    if args.log:
        if args.model==0:
            config_file_name = os.path.join(exp_dir, "config_rotation.yaml")
            with open(config_file_name, 'r') as f:
                hyper_param = yaml.safe_load(f)     
            #with wandb.init(mode="disabled"):
            with wandb.init(project="SO3_IPDF", entity="ipdf_se3", config=hyper_param, resume="allow",
                            name=exp_name,
                            id=exp_name):

                ## Run the training for the rotation model ##

                utils.set_random_seed(hyper_param['random_seed'])
                chkpt_dir = os.path.join(exp_dir, 'models_rotation') 

                print("Config file was loaded from: " + config_file_name + "\n")

                
                train_loader, val_loader = data.load_model_dataset(hyper_param)
                
                model, optimizer, start_epoch = models.load_rotation_model(hyper_param, args, exp_name)

                wandb.watch(model, log='all', log_freq=10)

                se3_ipdf.run_rotation_training(model=model, 
                                                train_dataset=train_loader,
                                                val_dataset=val_loader,
                                                optimizer=optimizer,
                                                hyper_param=hyper_param,
                                                checkpoint_dir=os.path.join(exp_dir,"models_rotation"),
                                                start_epoch=start_epoch)
            
        elif args.model==1:
            # with wandb.init(mode="disabled"):
            config_file_name = os.path.join(exp_dir, "config_translation.yaml")
            with open(config_file_name, 'r') as f:
                hyper_param = yaml.safe_load(f)
            with wandb.init(project="Translation_IPDF", entity="ipdf_se3", config=hyper_param, resume="allow",
                            name=exp_name,
                            id=exp_name):
                ## Run the training for the translation model ##
                

                utils.set_random_seed(hyper_param['random_seed'])
                chkpt_dir = os.path.join(exp_dir, 'models_translation') 

                wandb.config = hyper_param
                print("Config file was loaded from: " + config_file_name + "\n")

                train_loader, val_loader = data.load_model_dataset(hyper_param, args)
                
                model, optimizer, start_epoch = models.load_translation_model(hyper_param, args, exp_name)
                wandb.watch(model, log='all', log_freq=10)

                se3_ipdf.run_translation_training(model=model, 
                                                train_dataset=train_loader,
                                                val_dataset=val_loader,
                                                optimizer=optimizer,
                                                hyper_param=hyper_param,
                                                checkpoint_dir=os.path.join(exp_dir,"models_rotation"),
                                                start_epoch=start_epoch)
    
    else:
        if args.model==0:
            config_file_name = os.path.join(exp_dir, "config_rotation.yaml")
            with open(config_file_name, 'r') as f:
                hyper_param = yaml.safe_load(f)     
            #with wandb.init(mode="disabled"):
            with wandb.init(mode="disabled"):

                ## Run the training for the rotation model ##

                utils.set_random_seed(hyper_param['random_seed'])
                chkpt_dir = os.path.join(exp_dir, 'models_rotation') 

                print("Config file was loaded from: " + config_file_name + "\n")

                
                train_loader, val_loader = data.load_model_dataset(hyper_param)
                
                model, optimizer, start_epoch = models.load_rotation_model(hyper_param, args, exp_name)

                wandb.watch(model, log='all', log_freq=10)

                se3_ipdf.run_rotation_training(model=model, 
                                                train_dataset=train_loader,
                                                val_dataset=val_loader,
                                                optimizer=optimizer,
                                                hyper_param=hyper_param,
                                                checkpoint_dir=os.path.join(exp_dir,"models_rotation"),
                                                start_epoch=start_epoch)
            
        elif args.model==1:
            # with wandb.init(mode="disabled"):
            config_file_name = os.path.join(exp_dir, "config_translation.yaml")
            with open(config_file_name, 'r') as f:
                hyper_param = yaml.safe_load(f)
            with wandb.init(mode="disabled"):
                ## Run the training for the translation model ##
                

                utils.set_random_seed(hyper_param['random_seed'])
                chkpt_dir = os.path.join(exp_dir, 'models_translation') 

                wandb.config = hyper_param
                print("Config file was loaded from: " + config_file_name + "\n")

                train_loader, val_loader = data.load_model_dataset(hyper_param, args)
                
                model, optimizer, start_epoch = models.load_translation_model(hyper_param, args, exp_name)
                wandb.watch(model, log='all', log_freq=10)

                se3_ipdf.run_translation_training(model=model, 
                                                train_dataset=train_loader,
                                                val_dataset=val_loader,
                                                optimizer=optimizer,
                                                hyper_param=hyper_param,
                                                checkpoint_dir=os.path.join(exp_dir,"models_rotation"),
                                                start_epoch=start_epoch)
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters for training")
    parser.add_argument('-exp_name', type=str, default=None, help="experiment's name")
    parser.add_argument('-c_rot', metavar='PATH', type=str, default=None, dest="rot_epoch", help="Checkpoint epoch for the rotation model")
    parser.add_argument('-c_trans', metavar='PATH', type=str, default=None, dest="trans_epoch", help="Checkpoint epoch for the translation model")
    parser.add_argument('-model', type=int, default=0, help="0: Rotation model, 1: Translation model")
    parser.add_argument('-wandb', action="store_true", dest="log", help="Observed training using wandb")
    args = parser.parse_args()


    if args.exp_name is not None:
        experiment_dir_list = [("experiments/exp_" + args.exp_name)]
        exp_names = [args.exp_name]
        model_type_list = [args.model]
    else:
        experiment_dir_list = ["experiments/exp_" + exp_name for exp_name in EXP_NAME_LIST]
        exp_names = EXP_NAME_LIST
        model_type_list = MODEL_TYPE
    
    assert len(model_type_list)==len(experiment_dir_list)

    for (i, exp_dir) in enumerate(experiment_dir_list):
        exp_name = exp_names[i]

        print("_"*40)
        print(f"Start training model (type {model_type_list[i]}) in experiment {exp_dir}...")
        print("_"*40)
        
        train_model()


    #autograd.set_detect_anomaly(True)
    

  
