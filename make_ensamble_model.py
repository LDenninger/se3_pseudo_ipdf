import torch
import argparse
import os
import yaml

import se3_ipdf.models as models
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters for visualization of the model output")
    parser.add_argument("-exp_name", type=str, help="Name of the experiment")
    parser.add_argument("-rot_epoch", type=str, help="Epoch the checkpoint to load the rotation-model is taken from")
    parser.add_argument("-trans_epoch", type=str, help="Epoch the checkpoint to load the rotation-model is taken from")

    args = parser.parse_args()

    assert args.rot_epoch is not None and args.trans_epoch is not None

    exp_dir = "experiments/exp_" + args.exp_name

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
    model = models.load_ensamble_model(hyper_param_rot=hyper_param_rot, hyper_param_trans=hyper_param_trans, arguments=args, init_mode=True)

    # Save the model
    torch.save(model.state_dict(), save_path)

    print(f"Ensamble model was saved to {save_path}!")