import yaml
import os 
import argparse

import config

EXP_NAME_LIST = ["tabletop_2_can_uni_1","tabletop_2_can_uni_occ_1","tabletop_2_crackerbox_uni_1","tabletop_2_crackerbox_uni_occ_1","tabletop_2_bowl_uni_1","tabletop_2_bowl_uni_occ_1"]
DATASET_LIST = ["tabletop"]*6
OBJ_ID_LIST = [3]*2+[4]*2+ [5]*2
MODEL_TYPE = [0]*6


def initiate_experiment(exp_dir, model_type, dataset, obj_id):
    os.makedirs(exp_dir)
    vis_dir = os.path.join(exp_dir, "visualizations")
    os.makedirs(vis_dir)
    data_dir = os.path.join(exp_dir, "datasets")
    os.makedirs(data_dir)

    if model_type==0:
        config_file = os.path.join(exp_dir, "config_rotation.yaml")

        chkpt_dir = os.path.join(exp_dir, "models_rotation")
        os.makedirs(chkpt_dir)

        rot_config_data = config.load_model_rotation_config(dataset, obj_id)
        
        with open(config_file, 'w') as f:
            yaml.safe_dump(rot_config_data, f, default_flow_style=False)

    elif model_type==1:
        config_file = os.path.join(exp_dir, "config_translation.yaml")

        chkpt_dir = os.path.join(exp_dir, "models_translation")
        os.makedirs(chkpt_dir)

        trans_config_data = config.load_model_translation_config(dataset, obj_id)
        with open(config_file, 'w') as f:
            yaml.safe_dump(trans_config_data, f, default_flow_style=False)

    elif model_type==2:
        config_file_rotation = os.path.join(exp_dir, "config_rotation.yaml")
        config_file_translation = os.path.join(exp_dir, "config_translation.yaml")

        rot_config_data = config.load_model_rotation_config(dataset, obj_id)
        trans_config_data = config.load_model_translation_config(dataset, obj_id)

        with open(config_file_rotation, 'w') as f:
            yaml.safe_dump(rot_config_data, f, default_flow_style=False)
        with open(config_file_translation, 'w') as f:
            yaml.safe_dump(trans_config_data, f, default_flow_style=False)

        chkpt_dir_rot = os.path.join(exp_dir, "models_rotation")
        chkpt_dir_trans = os.path.join(exp_dir, "models_translation")
        chkpt_dir_ensamble = os.path.join(exp_dir, "models_ensamble")
        os.makedirs(chkpt_dir_rot)
        os.makedirs(chkpt_dir_trans)
        os.makedirs(chkpt_dir_ensamble)

    elif model_type==4:
        with open(config_file, 'w') as f:
            yaml.safe_dump(icp_config_data, f, default_flow_style=False)
    





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters for the experiment")
    parser.add_argument("-exp_name", type=str, default=None, help="name of the experiment")
    parser.add_argument("-dataset", type=str, default=None, help="dataset the experiment is performed on")
    parser.add_argument("-obj_id", type=str, default=None, help="object id of the object used in the experiment")
    parser.add_argument("-model", type=int, help="model type: 0: Single rotation IPDF-model, 1: Single translation IPDF-model, 2: Ensamble IPDF-model, 3: ICP-model", default=0)
    args = parser.parse_args()


    if args.exp_name is not None:
        exp_dir = 'experiments/exp_' + args.exp_name
        model_type = args.model
        if not os.path.isdir(exp_dir):
            initiate_experiment(exp_dir, model_type, args.dataset, args.obj_id)
            print("Experiment directory was created.")
        else:
            print("Experiment directory already exists.")
    
    else:
        for (i, exp_name) in enumerate(EXP_NAME_LIST):
            exp_dir = 'experiments/exp_' + exp_name
            if not os.path.isdir(exp_dir):
                initiate_experiment(exp_dir, MODEL_TYPE[i], DATASET_LIST[i], OBJ_ID_LIST[i])
                print("Experiment directory was created.")
            else:
                print(f"Experiment directory ({exp_dir}) already exists.")

