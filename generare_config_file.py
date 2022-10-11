import argparse


import config


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters for the configuration")
    parser.add_argument("-dataset", type=str, help="dataset the config file is generated for")
    parser.add_argument("-obj_id", type=str, help="object the config file is generated for")
    parser.add_argument("--pls", action="store_true", help="Configuration is produced for the pose labeling scheme")

    args = parser.parse_args()
    

    if args.pls:
        config.generate_pls_config(args.dataset, args.obj_id)
    
    else:
        config.generate_model_rotation_config(args.dataset, args.obj_id)
        config.generate_model_translation_config(args.dataset, args.obj_id)
    
    print("\nConfiguration files were successfully created!")


