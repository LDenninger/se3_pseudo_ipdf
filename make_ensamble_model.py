import torch
import argparse
import os
import yaml

import se3_ipdf.models as models

EXP_NAME_LIST = ["tabletop_3_bowl_4", "tabletop_3_bowl_uni_3", "tabletop_3_can_2", "tabletop_3_can_uni_2","tabletop_3_crackerbox_2", "tabletop_3_crackerbox_uni_2", ]

ROT_EPOCH_LIST = ["20", "20", "60", "60", "30", "30"]
TRANS_EPOCH_LIST = ["20"]*2+["final"]*4

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters for visualization of the model output")
    parser.add_argument("-exp_name", type=str, help="Name of the experiment")
    parser.add_argument("-rot_epoch", type=str, help="Epoch the checkpoint to load the rotation-model is taken from")
    parser.add_argument("-trans_epoch", type=str, help="Epoch the checkpoint to load the rotation-model is taken from")

    args = parser.parse_args()

    for (i, exp_name) in enumerate(EXP_NAME_LIST):
        exp_dir = "experiments/exp_" + exp_name
        args.rot_epoch = ROT_EPOCH_LIST[i]
        args.trans_epoch = TRANS_EPOCH_LIST[i]
        # Load the config file for the rotation model
        config_file_name = os.path.join(exp_dir, "config_rotation.yaml")
        with open(config_file_name, 'r') as f:
            hyper_param_rot = yaml.safe_load(f)

        # Load the config file for the translation model
        config_file_name = os.path.join(exp_dir, "config_translation.yaml")
        with open(config_file_name, 'r') as f:
            hyper_param_trans = yaml.safe_load(f)

        save_path = os.path.join(exp_dir, "models_ensamble", f"ensamble_{args.rot_epoch}_{args.trans_epoch}.pth")
        # Load the model
        model = models.load_ensamble_model(hyper_param_rot=hyper_param_rot, hyper_param_trans=hyper_param_trans, arguments=args, exp_name=exp_name, init_mode=True)

        # Save the model
        torch.save(model.state_dict(), save_path)

        print(f"Ensamble model was saved to {save_path}!")