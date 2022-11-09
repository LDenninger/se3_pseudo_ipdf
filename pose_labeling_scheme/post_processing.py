import torch
from tqdm import tqdm
import ipdb
import os

from .registration import check_convergence_batchwise
from .utils import check_duplicates_averaging, id_to_path, id_to_path_uniform

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
                                                    intrinsic=input["intrinsic"].squeeze(),
                                                    config=hyper_param)
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
                                                                intrinsic=input["intrinsic"].squeeze(),
                                                                config=hyper_param)
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

    non_exist = []

    for (i, input) in progress_bar:
        
        if not input["loaded"]:
            non_exist.append(i)
            continue
        if input["pseudo_gt"] is None:
            non_exist.append(i)
            continue
        
        cleaned_pseudo_gt = check_duplicates_averaging(input["pseudo_gt"].squeeze(0).to(DEVICE), angular_threshold=angular_threshold)
        cleaned_pseudo_gt = cleaned_pseudo_gt.cpu()

        if hyper_param["dataset"]=="tless":
                data_dir = "/home/nfs/inf6/data/datasets/T-Less/t-less_v2/train_kinect"
                torch.save(cleaned_pseudo_gt, os.path.join(data_dir, str(hyper_param["obj_id"]), "pseudo_gt", ("cleaned_"+str(i).zfill(4)+ ".pth")))
        elif hyper_param["dataset"]=="tabletop":
            if hyper_param["material"]:
                data_dir = id_to_path[hyper_param["obj_id"]]
            else:
                data_dir = id_to_path_uniform[hyper_param["obj_id"]]
            torch.save(cleaned_pseudo_gt, os.path.join(data_dir, str(i).zfill(6), "cleaned_pseudo_gt.pth"))
    
    print("Frames without pseudo ground truth:", non_exist)

def run_convention_cleanup(dataset, hyper_param):

    progress_bar = tqdm(enumerate(dataset), total=len(dataset))

    non_exist = []

    conv = torch.eye(3)
    conv[1,1] *= -1
    conv[2,2] *= -1


    for (i, input) in progress_bar:
        
        if not input["loaded"]:
            non_exist.append(i)
            continue
        pgt = input["pseudo_gt"].squeeze(0)
        cleaned_pseudo_gt =  pgt
        cleaned_pseudo_gt[:,:3,:3] =pgt[:,:3,:3] @ conv
        cleaned_pseudo_gt[:,:3,:3] = conv @ cleaned_pseudo_gt[:,:3,:3]

        if hyper_param["dataset"]=="tless":
                data_dir = "/home/nfs/inf6/data/datasets/T-Less/t-less_v2/train_kinect"
                torch.save(cleaned_pseudo_gt, os.path.join(data_dir, str(hyper_param["obj_id"]), "pseudo_gt", ("cleaned_"+str(i).zfill(4)+ ".pth")))
        elif hyper_param["dataset"]=="tabletop":
            if hyper_param["material"]:
                data_dir = id_to_path[hyper_param["obj_id"]]
            else:
                data_dir = id_to_path_uniform[hyper_param["obj_id"]]
            torch.save(cleaned_pseudo_gt, os.path.join(data_dir, str(i).zfill(6), "cleaned_pseudo_gt.pth"))
    
    print("Conversion was undone!")

