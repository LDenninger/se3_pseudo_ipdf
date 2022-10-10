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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters for training")
    parser.add_argument('-exp_name', type=str, help="experiment's name")
    parser.add_argument('-c_rot', metavar='PATH', type=str, default=None, dest="rot_epoch", help="Checkpoint epoch for the rotation model")
    parser.add_argument('-c_trans', metavar='PATH', type=str, default=None, dest="trans_epoch", help="Checkpoint epoch for the translation model")
    parser.add_argument('-model', type=int, default=0, help="0: Rotation model, 1: Translation model")
    parser.add_argument('-wandb', action="store_true", help="Observed training using wandb")
    args = parser.parse_args()

    exp_dir = "experiments/exp_" + args.exp_name
    #autograd.set_detect_anomaly(True)

    wandb.login()
    # Set up Weights'n'Biases logging
    if args.wandb:
        if args.model==0:
            config_file_name = os.path.join(exp_dir, "config_rotation.yaml")
            with open(config_file_name, 'r') as f:
                hyper_param = yaml.safe_load(f)     
            #with wandb.init(mode="disabled"):
            with wandb.init(project="SO3_IPDF", entity="ipdf_se3", config=hyper_param, resume="allow",
                            name=args.exp_name,
                            id=args.exp_name):

                ## Run the training for the rotation model ##

                utils.set_random_seed(hyper_param['random_seed'])
                chkpt_dir = os.path.join(exp_dir, 'models_rotation') 

                print("Config file was loaded from: " + config_file_name + "\n")

                
                train_loader, val_loader = data.load_model_dataset(hyper_param)
                
                model, optimizer, start_epoch = models.load_rotation_model(hyper_param, args)

                wandb.watch(model, log='all', log_freq=10)

                se3_ipdf.run_rotation_training(hyper_param, device=DEVICE)
            
        elif args.model==1:
            # with wandb.init(mode="disabled"):
            config_file_name = os.path.join(exp_dir, "config_translation.yaml")
            with open(config_file_name, 'r') as f:
                hyper_param = yaml.safe_load(f)
            with wandb.init(project="Translation_IPDF", entity="ipdf_se3", config=hyper_param, resume="allow",
                            name=args.exp_name,
                            id=args.exp_name):
                ## Run the training for the translation model ##
                

                utils.set_random_seed(hyper_param['random_seed'])
                chkpt_dir = os.path.join(exp_dir, 'models_translation') 

                wandb.config = hyper_param
                print("Config file was loaded from: " + config_file_name + "\n")

                train_loader, val_loader = data.load_dataset(hyper_param, args)
                
                model, optimizer, start_epoch = models.load_translation_model(hyper_param)
                wandb.watch(model, log='all', log_freq=10)

                se3_ipdf.run_translation_training(hyper_param, device=DEVICE)
    
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

                
                train_loader, val_loader = data.load_dataset(hyper_param)
                
                model, optimizer, start_epoch = models.load_rotation_model(hyper_param, args)

                wandb.watch(model, log='all', log_freq=10)

                se3_ipdf.run_rotation_training(hyper_param, device=DEVICE)
            
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

                train_loader, val_loader = data.load_dataset(hyper_param, args)
                
                model, optimizer, start_epoch = models.load_translation_model(hyper_param)
                wandb.watch(model, log='all', log_freq=10)

                se3_ipdf.run_translation_training(hyper_param, device=DEVICE)