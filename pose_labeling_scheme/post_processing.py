import torch
from tqdm import tqdm
import ipdb
import os

from .registration import check_convergence_batchwise
from .utils import check_duplicates_averaging
from ..data import id_to_path

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run_convergence_check(dataset, obj_model, obj_model_sl, hyper_param):
    """Iterate through the pseudo ground truth already produced for the dataset using the 
    pose labeling scheme. The already existing pseudo ground truth are checked for convergence
    using the render-and-compare framework and saved to a new file.
    
    """

    progress_bar = tqdm(enumerate(dataset), total=len(dataset))

    for (i, input) in progress_bar:

        if not input["loaded"]:
            print("\nData could not been loaded!")

        if not hyper_param["verbose"]:
            converged = check_convergence_batchwise(depth_original=input["depth_image"].squeeze(),
                                                    obj_model=obj_model_sl, 
                                                    transformation_set=input["pseudo_gt"].squeeze(),
                                                    threshold=hyper_param['threshold'],
                                                    intrinsic=input["intrinsic"].squeeze(),
                                                    verbose=False)
            conv_ind = torch.nonzero(converged).squeeze()
            conv_pgt = input["pseudo_gt"].squeeze()[conv_ind]
            if hyper_param["dataset"]=="tless":
                data_dir = "/home/nfs/inf6/data/datasets/T-Less/t-less_v2/train_kinect"
                torch.save(conv_pgt, os.path.join(data_dir, str(hyper_param["obj_id"]), "pseudo_gt", ("cleaned_"+str(i).zfill(4)+ ".pth")))
            elif hyper_param["dataset"]=="tabletop":
                data_dir = id_to_path[hyper_param["obj_id"]]
                torch.save(conv_pgt, os.path.join(data_dir, str(i).zfill(6), "cleaned_pseudo_gt.pth"))

        else:
            converged, d_max, d_avg = check_convergence_batchwise(depth_original=input["depth_image"].squeeze(),
                                                                obj_model=obj_model_sl, 
                                                                transformation_set=input["pseudo_gt"].squeeze(),
                                                                threshold=hyper_param['threshold'],
                                                                intrinsic=input["intrinsic"].squeeze(),
                                                                verbose=True)
            for j in range(converged.shape[0]):
                print("_"*20)
                print(f"\nConvergence Check for frame {i}, PGT no. {j}:\n")
                print("\nMaximum distance: ", d_max[j],"Mean Distance: ", d_avg[j])
                print("\nConverged: ", converged[j])
                print("\n")
                print("_"*20)
                print("\n")
            ipdb.set_trace()

def run_duplicate_check(dataset, hyper_param, angular_threshold=15):

    progress_bar = tqdm(enumerate(dataset), total=len(dataset))

    for (i, input) in progress_bar:

        cleaned_pseudo_gt = check_duplicates_averaging(input["pseudo_gt"], angular_threshold=angular_threshold)

        if hyper_param["dataset"]=="tless":
                data_dir = "/home/nfs/inf6/data/datasets/T-Less/t-less_v2/train_kinect"
                torch.save(cleaned_pseudo_gt, os.path.join(data_dir, str(hyper_param["obj_id"]), "pseudo_gt", ("cleaned_"+str(i).zfill(4)+ ".pth")))
        elif hyper_param["dataset"]=="tabletop":
            data_dir = id_to_path[hyper_param["obj_id"]]
            torch.save(cleaned_pseudo_gt, os.path.join(data_dir, str(i).zfill(6), "cleaned_pseudo_gt.pth"))
    
    

