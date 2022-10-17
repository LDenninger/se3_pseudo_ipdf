import torch
import argparse
from tqdm import tqdm
import os
import ipdb

import data
import utils
import config
from pose_labeling_scheme import pose_labeling_scheme
from pose_labeling_scheme.utils import convert_points_opencv_opengl
import utils.visualizations as visualizations

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Hyperparameters for the pose labeling scheme")
    parser.add_argument("-dataset", type=str, help="Dataset to run the PLS on")
    parser.add_argument("-obj_id", type=int, help="Object to run the PLS on")
    parser.add_argument("-rs", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    utils.set_random_seed(args.rs)
    
    # Load the config file corresponding to dataset and object
    config = config.load_pls_config(args.dataset, args.obj_id)

    # Save directory for the pseudo ground truth
    pseudo_save_dir = os.path.join("/home/nfs/inf6/data/datasets/T-Less/t-less_v2/train_kinect", str(config["obj_id"]).zfill(2), "pseudo_gt")
    

    # Load the data
    data_loader = data.load_pls_dataset(config)

    progress_bar = tqdm(enumerate(data_loader), total= config["length"])

    failed = []

    # Load the object model
    if args.dataset=="tabletop":
        object_model, diameter = data.load_ycbv_object_model(args.obj_id, pointcloud_only=True)
        object_model = convert_points_opencv_opengl(object_model) # Convert opeGL to openCV
        object_model_sl = data.load_sl_cad_model(args.dataset, args.obj_id)
        # Convert canonical point clouds to fit the OpenCV convention

    elif args.dataset=="tless":
        object_model, diameter = data.load_tless_object_model(args.obj_id, pointcloud_only=True)
        object_model_sl = data.load_sl_cad_model(args.dataset, args.obj_id)
    else:
        print("\nNo object model for the given dataset/object!")
    
    object_model = object_model.to(DEVICE)

    for (i, input) in progress_bar:

        if i==1296:
            break
        if not input["loaded"]:
            continue

        
        seg_data = input["seg_image"].to(DEVICE).squeeze()
        depth_data = input["depth_image"].to(DEVICE).squeeze()
        intrinsic = input["intrinsic"].to(DEVICE).squeeze()
        
        pseudo_ground_truth = pose_labeling_scheme(pts_canonical=object_model,
                                                    seg_data=seg_data,
                                                    depth_data=depth_data,
                                                    diameter = diameter,
                                                    intrinsic=intrinsic,
                                                    obj_model_sl=object_model_sl,
                                                    config=config)
        # Save the failed frames
        if pseudo_ground_truth is None:
            failed.append(i)

        # Save the pseudo ground truth for the current frame
        if not config["verbose"]:
            if args.dataset=="tless":
                save_dir = os.path.join(pseudo_save_dir, (str(i).zfill(4)+".pth"))
                if os.path.exists(save_dir):
                    pgt_exist = torch.load(save_dir)
                    pseudo_ground_truth = torch.cat((pgt_exist, pseudo_ground_truth))
                torch.save(pseudo_ground_truth, save_dir)

            elif args.dataset=="tabletop":
                save_dir = os.path.join(data.id_to_path[args.obj_id], str(i).zfill(6), "pseudo_gt.pth")
                if False and os.path.exists(save_dir):
                    pgt_exist = torch.load(save_dir)
                    pseudo_ground_truth = torch.cat((pgt_exist, pseudo_ground_truth))
                torch.save(pseudo_ground_truth, save_dir)
        else:
            visualizations.visualize_so3_rotations(
                rotations=pseudo_ground_truth,
                dataset=config["dataset"],
                obj_id=config["obj_id"],
                rotations_gt=input["ground_truth"],
                display_gt_set=True,
                save_path="output/pose_labeling_scheme/pgt_final_result.png"
            )
            ipdb.set_trace()
        
        
            
    print("_"*20)
    print("\nPseudo labeling scheme finished")
    print("\nFailed frames:", failed)
        