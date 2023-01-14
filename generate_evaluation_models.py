
import torch
import os
from pathlib import Path as P
import yaml
import se3_ipdf.models as models
import argparse

PATH_NAME = P("experiments/aa_evaluation_models_3")

EXP_NAME_LIST = [
"tabletop_4_box_3",
"tabletop_4_box_ana_2",
"tabletop_4_box_ana_occ_2",
"tabletop_4_box_single_2",
"tabletop_4_box_single_occ_2",
"tabletop_4_box_uni_3",
"tabletop_4_box_occ_3",


]

ROT_EPOCH_LIST = ["40"]*6+["30"]
TRANS_EPOCH_LIST = ["20"]*7
def generate_file_structure(exp_name):

    p = PATH_NAME / exp_name
    try:
        os.makedirs(str(p))
        os.makedirs(str(p/"visualizations"))
        os.makedirs(str(p/"evaluations"))
        os.makedirs(str(p/"models"))

    except:
        print("Filesystem could not been created!")
        return False

    return True

def load_experiment(exp_name, args):

    exp_name_full = P(("exp_"+exp_name))
    exp_dir = "experiments"/exp_name_full


    config_file_name = os.path.join(str(exp_dir /"config_rotation.yaml"))
    with open(config_file_name, 'r') as f:
        hyper_param_rot = yaml.safe_load(f)

    # Load the config file for the translation model
    config_file_name = os.path.join(str(exp_dir / "config_translation.yaml"))
    with open(config_file_name, 'r') as f:
        hyper_param_trans = yaml.safe_load(f)
        
    model = models.load_ensamble_model(hyper_param_rot=hyper_param_rot, hyper_param_trans=hyper_param_trans, exp_name=exp_name, arguments=args)

    save_path = PATH_NAME / exp_name[:-2] / "models"

    torch.save(model.state_dict(), str(save_path / f"model_{args.rot_epoch}_{args.trans_epoch}.pth"))

    with open(str(save_path/"config_rotation.yaml"), 'w') as f:
            yaml.safe_dump(hyper_param_rot, f, default_flow_style=False)
    with open(str(save_path/"config_translation.yaml"), 'w') as f:
            yaml.safe_dump(hyper_param_trans, f, default_flow_style=False)

    print("Ensamble model successfully loaded!")


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Hyperparameters for evaluation")
    parser.add_argument("-exp_name", type=str, help="Name of the experiment")
    parser.add_argument("-rot_epoch", type=str, help="Epoch the checkpoint to load the rotation-model is taken from")
    parser.add_argument("-trans_epoch", type=str, help="Epoch the checkpoint to load the rotation-model is taken from")
    parser.add_argument('-r_seed', metavar='INTEGER', type=int, default='42', help='Random seed used for evaluation')
    parser.add_argument("--mode", type=int, help="evaluation mode: 0: evaluation on rotation model, 1: evaluation on translation model, 2: full evaluation on the ensamble model")
    parser.add_argument('--wandb', action='store_true', help='Log data to WandB')

    args = parser.parse_args()

    assert len(EXP_NAME_LIST) == len(ROT_EPOCH_LIST)
    assert len(EXP_NAME_LIST) == len(TRANS_EPOCH_LIST)


    for (i, exp_name) in enumerate(EXP_NAME_LIST):

        rot_epoch = ROT_EPOCH_LIST[i]
        trans_epoch = TRANS_EPOCH_LIST[i]

        clean_exp_name = exp_name[:-2]
        args.rot_epoch = rot_epoch
        args.trans_epoch = trans_epoch

        ret = generate_file_structure(clean_exp_name)
        load_experiment(exp_name, args)


        