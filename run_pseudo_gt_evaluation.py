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
LENGTH = 5000
OBJ_ID = [3]

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
                    pgt = torch.load(str(p/"cleaned_pseudo_gt.pth"))[:,:3,:3]
                else:
                    pgt = torch.load(str(p/"pseudo_gt.pth"))[:,:3,:3]
                ground_truth = torch.load(str(p/"ground_truth.pt"))[:3,:3]
            except:
                try:
                    ground_truth = torch.load(str(p/"gt.pt"))[:3,:3]
                except:
                    continue
            if pgt is None:
                continue    
            error.append(pls.evaluation_acc_error(pgt, ground_truth, obj_id))
            recall_error.append(pls.evaluation_recall_error([pgt], ground_truth.unsqueeze(0), obj_id))

        p_error = np.mean(error)
        r_error = np.mean(recall_error)
        print("_"*60)
        print(f"\nObject no. {obj_id}: Angular precision error: {p_error}, Angular recall error: {r_error}\n")
        print("_"*60)
        failed=False
        for n in [10, 20]:
            l = LENGTH
            progress_bar = tqdm(range(l), total=l)
            for i in progress_bar:
                ground_truth = []
                pgt = []
                failed=False
                for j in range(n):
                    idx = str(args.start+i+j).zfill(6)
                    p = data_path / idx 
                    try:
                        if args.clean:
                            pgt.append(torch.load(str(p/"cleaned_pseudo_gt.pth"))[:,:3,:3])
                        else:
                            pgt.append(torch.load(str(p/"pseudo_gt.pth"))[:,:3,:3])
                    except:
                        failed = True
                    try:
                        ground_truth.append(torch.load(str(p/"ground_truth.pt"))[:3,:3])
                    except:
                        try:
                            ground_truth.append(torch.load(str(p/"gt.pt"))[:3,:3])
                        except:
                            failed=True
                if failed:
                    continue
                
                recall_error.append(pls.evaluation_recall_error(pgt, torch.stack(ground_truth), obj_id))
            
            r_error = np.mean(recall_error)
            print("_"*60)
            print(f"\nObject no. {obj_id}: Angular recall error with {n} images: {r_error}\n")
            print("_"*60)
        
        

