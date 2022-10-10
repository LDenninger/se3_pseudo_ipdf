from pathlib import Path as P
import yaml
import models
import pose_labeling_scheme

MODEL_CONFIG_PATH = P("models")
PLS_CONFIG_PATH = P("pose_labeling_scheme")

## Loading scripts ##

def load_model_rotation_config(dataset: str, obj_id: str):
    
    config_path = MODEL_CONFIG_PATH / dataset / f"config_rotation_{obj_id}.yml"

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except:
        print("\nNo specific configuration file found for the dataset/object...\nConfig will be generated...")
        generate_model_rotation_config(dataset, obj_id)
        return load_model_rotation_config(dataset, obj_id)
    
    return config

def load_model_translation_config(dataset: str, obj_id: str):
    
    config_path = MODEL_CONFIG_PATH / dataset / f"config_translation_{obj_id}.yml"

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except:
        print("\nNo specific configuration file found for the dataset/object...\nConfig will be generated...")
        generate_model_translation_config(dataset, obj_id)
        return load_model_translation_config(dataset, obj_id)
    
    return config

def load_pls_config(dataset: str, obj_id: str):

    config_path = PLS_CONFIG_PATH / dataset / f"config_{obj_id}.yml"

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except:
        print("\nNo specific configuration file found for the dataset/object...\nDefault config is loaded!\n")
        config = pose_labeling_scheme.pls_config_data
    
    return config

## Generation scripts ##

def generate_model_rotation_config(dataset: str, obj_id: str):

    config_path = MODEL_CONFIG_PATH / dataset / f"config_rotation_{obj_id}.yml"
    

    try:
        config = models.rot_config_data
        
        with open(config_path, "r") as f:
            yaml.dump(config, f)
        print(f"\nConfig was generated and saved to {config_path}")
        
    except Exception:
        print(Exception)
        print("\nConfig could not been created!")

def generate_model_translation_config(dataset: str, obj_id: str):

    config_path = MODEL_CONFIG_PATH / dataset / f"config_translation_{obj_id}.yml"
    

    try:
        config = models.trans_config_data

        with open(config_path, "r") as f:
            yaml.dump(config, f)
        print(f"\nConfig was generated and saved to {config_path}")
        
    except Exception:
        print(Exception)
        print("\nConfig could not been created!")

def generate_pls_config(dataset: str, obj_id: str):

    config_path = PLS_CONFIG_PATH / dataset / f"config_{obj_id}.yml"

    try:
        config = pose_labeling_scheme.pls_config_data

        with open(config_path, "r") as f:
            yaml.dump(config, f)
            
    except:
        print(Exception)
        print("\nConfig could not been created!")
    
