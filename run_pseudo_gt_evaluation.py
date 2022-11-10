import argparse
from numpy import save
from tqdm import tqdm
from pathlib import Path as P
import torch
import numpy as np


import utils.visualizations as visualizations
import utils
import config
import data
import pose_labeling_scheme as pls

SAVE_PATH = P("output/pose_labeling_scheme")
LENGTH = 2300
OBJ_ID = [5]

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters for the pose labeling scheme")
    parser.add_argument("-start", type=int, default=0, help="Frame index to start the dataset from")
    parser.add_argument("-rs", type=int, default=42, help="Random seed")
    parser.add_argument("--clean", default=False,action="store_true", help="Random seed")
    parser.add_argument("--uni", default=False,action="store_true", help="Random seed")


    args = parser.parse_args()

    utils.set_random_seed(args.rs)
    
    # Load the config file corresponding to dataset and object

    # Load the data
    for obj_id in OBJ_ID:
        progress_bar = tqdm(range(LENGTH), total=LENGTH)
        if args.uni:
            data_path = P(data.id_to_path_uniform[obj_id])
        else:
            data_path = P(data.id_to_path[obj_id])


        error = []
        recall_error = []

        for i in progress_bar:
            idx = str(args.start+i).zfill(6)
            p = data_path / idx 
            try:
                if args.clean:
                    pgt = torch.load(str(p/"cleaned_pseudo_gt.pth"))
                else:
                    pgt = torch.load(str(p/"pseudo_gt.pth"))
                ground_truth = torch.load(str(p/"ground_truth.pt"))
            except:
                try:
                    ground_truth = torch.load(str(p/"gt.pt"))
                except:
                    continue
            if pgt is None:
                continue    
            error.append(pls.evaluation_acc_error(pgt[:,:3,:3], ground_truth[:3,:3], obj_id))
            recall_error.append(pls.evaluation_recall_error(pgt[:,:3,:3], ground_truth[:3,:3], obj_id))
        p_error = np.mean(error)
        r_error = np.mean(recall_error)
        print("_"*60)
        print(f"\nObject no. {obj_id}: Angular precision error: {p_error}, Angular recall error: {r_error}\n")
        print("_"*60)

