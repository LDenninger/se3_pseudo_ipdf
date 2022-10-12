import argparse
from numpy import save
from tqdm import tqdm
from pathlib import Path as P


import utils.visualizations as visualizations
import utils
import config
import data

SAVE_PATH = P("output/pose_labeling_scheme")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters for the pose labeling scheme")
    parser.add_argument("-dataset", type=str, help="Dataset to run the PLS on")
    parser.add_argument("-obj_id", type=int, help="Object to run the PLS on")
    parser.add_argument("-f_name", type=str, default="pgt_test.png", help="File name the visualization is saved to")
    parser.add_argument("--clean", action="store_true", help="Visualize cleaned pgt")
    parser.add_argument("-start", type=int, default=0, help="Frame index to start the dataset from")
    parser.add_argument("-rs", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    utils.set_random_seed(args.rs)
    
    # Load the config file corresponding to dataset and object
    config = config.load_pls_config(args.dataset, args.obj_id)
    
    # Load the data
    data_loader = data.load_pls_dataset(config, start=args.start, return_gt=True, return_pgt=True, cleaned_pgt=args.clean)

    save_path = str(SAVE_PATH / args.f_name)

    print(f"\nOutput of each iteration will be saved to: {save_path}")


    visualizations.visualize_pseudo_gt(dataset=data_loader, hyper_param=config, save_path=save_path)

