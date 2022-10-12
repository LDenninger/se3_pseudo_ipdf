from pathlib import Path as P
import yaml

from .pls_config import tless_pls_config_data, tabletop_pls_config_data
from .se3_ipdf_config import tabletop_rot_config_data, tabletop_trans_config_data, tless_rot_config_data, tless_trans_config_data


MODEL_CONFIG_PATH = P("config/se3_ipdf_config")
PLS_CONFIG_PATH = P("config/pls_config")

## Loading scripts ##

def load_model_rotation_config(dataset: str, obj_id: str):
    
    config_path = MODEL_CONFIG_PATH / dataset / f"config_rotation_{obj_id}.yml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except:
        print("\nNo specific configuration file found for the dataset/object...\nConfig will be generated...")
        ret_code = generate_model_rotation_config(dataset, obj_id)
        if ret_code==-1:
            return ret_code
        return load_model_rotation_config(dataset, obj_id)
    
    return config

def load_model_translation_config(dataset: str, obj_id: str):
    
    config_path = MODEL_CONFIG_PATH / dataset / f"config_translation_{obj_id}.yml"

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except:
        print("\nNo specific configuration file found for the dataset/object...\nConfig will be generated...")
        ret_code = generate_model_translation_config(dataset, obj_id)
        if ret_code==-1:
            return ret_code
        return load_model_translation_config(dataset, obj_id)
    
    return config

def load_pls_config(dataset: str, obj_id: str):

    config_path = PLS_CONFIG_PATH / dataset / f"config_{obj_id}.yml"

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except:
        print("\nNo specific configuration file found for the dataset/object...\nConfig will be generated...")
        ret_code = generate_pls_config(dataset, obj_id)
        if ret_code==-1:
            return ret_code
        return load_pls_config(dataset, obj_id)
    
    return config

## Generation scripts ##

def generate_model_rotation_config(dataset: str, obj_id: str):

    config_path = MODEL_CONFIG_PATH / dataset / f"config_rotation_{obj_id}.yml"
    
    try:
        
        if dataset=="tless":
            config = tless_rot_config_data
        elif dataset=="tabletop":
            config = tabletop_rot_config_data
        else:
            print("\nDataset is not defined!")
            return -1
        
        # set dataset and obj_id in config file
        config["dataset"] = dataset
        config["obj_id"] = obj_id
        
        with open(str(config_path), "w") as f:
            yaml.safe_dump(config, f)
        print(f"\nConfig was generated and saved to {config_path}")
        
    except Exception:
        print(Exception)
        print("\nConfig could not been created!")

    return 1
def generate_model_translation_config(dataset: str, obj_id: str):

    config_path = MODEL_CONFIG_PATH / dataset / f"config_translation_{obj_id}.yml"
    

    try:
        if dataset=="tless":
            config = tless_trans_config_data
        elif dataset=="tabletop":
            config = tabletop_trans_config_data
        else:
            print("\nDataset is not defined!")
            return -1

        # set dataset and obj_id in config file
        config["dataset"] = dataset
        config["obj_id"] = obj_id

        with open(str(config_path), "w") as f:
            yaml.safe_dump(config, f)
        print(f"\nConfig was generated and saved to {config_path}")
        
    except Exception:
        print("\nConfig could not been created!")

    return 1

def generate_pls_config(dataset: str, obj_id: str):

    config_path = PLS_CONFIG_PATH / dataset / f"config_{obj_id}.yml"

    try:
        if dataset=="tless":
            config = tless_pls_config_data
        elif dataset=="tabletop":
            config = tabletop_pls_config_data
        else:
            print("\nDataset is not defined!")
            return -1
        
        # set dataset and obj_id in config file
        config["dataset"] = dataset
        config["obj_id"] = obj_id

        with open(str(config_path), "w") as f:
            yaml.safe_dump(config, f)
            
    except:
        print("\nConfig could not been created!")
    
    return 1
    
