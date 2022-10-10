import argparse
from tqdm import tqdm

import data
import utils
import config

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Hyperparameters for the pose labeling scheme")
    parser.add_argument("-dataset", type=str, help="Dataset to run the PLS on")
    parser.add_argument("-obj_id", type=int, help="Object to run the PLS on")
    parser.add_argument("-rs", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    utils.set_random_seed(args.rs)
    
    # Load the config file corresponding to dataset and object
    config = config.load_pls_config(args.dataset, args.obj_id)

    # Load the data
    data_loader = data.load_pls_dataset(config)

    progress_bar = tqdm(enumerate(data_loader), total= config["length"])

    failed = []
    
    for (i, input) in progress_bar:
        if i==1296:
            break
        if input != 1:
            failed.append(i)
            
    print("_"*20)
    print("\nPseudo labeling scheme finished")
    print("\nFailed frames:", failed)
        