import argparse
from numpy import save
from tqdm import tqdm
from pathlib import Path as P
import torch


import utils.visualizations as visualizations
import utils
import config
import data
import pose_labeling_scheme as pls

SAVE_PATH = P("output/pose_labeling_scheme")
LENGTH = 200


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters for the pose labeling scheme")
    parser.add_argument("-dataset", type=str, help="Dataset to run the PLS on")
    parser.add_argument("-obj_id", type=int, help="Object to run the PLS on")
    parser.add_argument("--mat", default=False, action="store_true")
    parser.add_argument("-f_name", type=str, default="pgt_test.png", help="File name the visualization is saved to")
    parser.add_argument("--clean", action="store_true", help="Visualize cleaned pgt")
    parser.add_argument("-start", type=int, default=0, help="Frame index to start the dataset from")
    parser.add_argument("-rs", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    utils.set_random_seed(args.rs)
    
    # Load the config file corresponding to dataset and object
    config = config.load_pls_config(args.dataset, args.obj_id)

    if args.mat:
        data_path = P(data.id_to_path)
    else:
        data_path = P(data.id_to_path_uniform)
    
    # Load the data

    save_path = str(SAVE_PATH / args.f_name)

    print(f"\nOutput of each iteration will be saved to: {save_path}")


    progress_bar = tqdm(range(LENGTH), total=LENGTH)

    error = []
    recall_error = []

    for i in progress_bar:
        idx = str(i).zfill(6)
        p = data_path / idx 
        try:
            pgt = torch.load(str(p/"pseudo_gt.pth"))
            ground_truth = torch.load(str(p/"ground_truth.pt"))
        except:
            try:
                ground_truth = torch.load(str(p/"gt.pt"))
            except:
                continue

        error.append(pls.evaluation_acc_error(pgt, ground_truth, config))
        recall_error.append(pls.evaluation_recall_error(pgt, ground_truth, config))


