import torch
import argparse
from tqdm import tqdm
import os
import ipdb

import data
import utils
import config as c
from pose_labeling_scheme import pose_labeling_scheme
from pose_labeling_scheme.utils import convert_points_opencv_opengl
import utils.visualizations as visualizations

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

OBJ_ID_LIST = [8]*2
MATERIAL_LIST = [True, False]
DATASET_LIST = ["tabletop"]*2

SAVE_NAME = "pseudo_gt_thesis.pth"

def run_pose_labeling_scheme(dataset, obj_id, material):
    # Load the config file corresponding to dataset and object
    config = c.load_pls_config(dataset, obj_id)
    if args.dataset=="tabletop":
        config["material"] = material
    config["verbose"] = args.verbose
    # Save directory for the pseudo ground truth
    pseudo_save_dir = os.path.join("/home/nfs/inf6/data/datasets/T-Less/t-less_v2/train_kinect", str(config["obj_id"]).zfill(2), "pseudo_gt")
    

    # Load the data
    data_loader = data.load_pls_dataset(config, material=material, start=args.start)

    progress_bar = tqdm(enumerate(data_loader), total= config["length"]-args.start)

    failed = []

    # Load the object model
    if dataset=="tabletop":
        object_model, diameter = data.load_ycbv_object_model(obj_id, pointcloud_only=True)
        object_model = convert_points_opencv_opengl(object_model) # Convert opeGL to openCV
        object_model_sl = data.load_sl_cad_model(dataset, obj_id)
        # Convert canonical point clouds to fit the OpenCV convention

    elif dataset=="tless":
        object_model, diameter = data.load_tless_object_model(obj_id, pointcloud_only=True)
        object_model_sl = data.load_sl_cad_model(dataset, obj_id)
    else:
        print("\nNo object model for the given dataset/object!")
    
    object_model = object_model.to(DEVICE)

    print("_"*60)
    print(f"Pose Labeling Scheme for object no. {obj_id} in {dataset} dataset (material = {material})\n")
    print("Configuration:\n")
    print(config)
    print("_"*60)

    for (i, input) in progress_bar:

        if i==config["length"]-args.start:
            break
        
        if config["skip"]:
            if dataset=="tless" and os.path.exists(os.path.join(pseudo_save_dir, (str(i).zfill(4)+".pth"))):
                continue
            elif dataset=="tabletop" and os.path.exists(os.path.join(data.id_to_path[obj_id] if material else data.id_to_path_uniform[obj_id], str(i).zfill(6), SAVE_NAME)):
                continue
        if not input["loaded"]:
            failed.append(i)
            continue
        
        seg_data = input["seg_image"].to(DEVICE).squeeze()
        depth_data = input["depth_image"].to(DEVICE).squeeze()
        intrinsic = input["intrinsic"].to(DEVICE).squeeze()
        if not (seg_data==config["obj_id"]).any():
            failed.append(i)
            continue
        try:
            pseudo_ground_truth = pose_labeling_scheme(pts_canonical=object_model,
                                                        seg_data=seg_data,
                                                        depth_data=depth_data,
                                                        diameter = diameter,
                                                        intrinsic=intrinsic,
                                                        obj_model_sl=object_model_sl,
                                                        config=config)
        except:
            pseudo_ground_truth = None
        # Save the failed frames
        if pseudo_ground_truth is None:
            failed.append(i)
            continue

        # Save the pseudo ground truth for the current frame
        if not config["verbose"]:
            if args.dataset=="tless":
                save_dir = os.path.join(pseudo_save_dir, (str(i).zfill(4)+".pth"))
                if os.path.exists(save_dir):
                    pgt_exist = torch.load(save_dir)
                    pseudo_ground_truth = torch.cat((pgt_exist, pseudo_ground_truth))
                torch.save(pseudo_ground_truth, save_dir)

            elif dataset=="tabletop":
                if material:
                    save_dir = os.path.join(data.id_to_path[obj_id], str(input["index"].item()).zfill(6), SAVE_NAME)
                else:
                    save_dir = os.path.join(data.id_to_path_uniform[obj_id], str(input["index"].item()).zfill(6), SAVE_NAME)
                if False and os.path.exists(save_dir):
                    pgt_exist = torch.load(save_dir)
                    if pgt_exist is not None:
                        pseudo_ground_truth = torch.cat((pgt_exist, pseudo_ground_truth))
                torch.save(pseudo_ground_truth, save_dir)
        else:
            visualizations.visualize_so3_rotations(
                rotations=pseudo_ground_truth[:,:3,:3],
                dataset=config["dataset"],
                obj_id=config["obj_id"],
                rotations_gt=input["ground_truth"][:,:3,:3],
                display_gt_set=True,
                save_path="output/pose_labeling_scheme/pgt_final_result.png"
            )
            ipdb.set_trace()
        
        
            
    print("_"*20)
    print("\nPseudo labeling scheme finished")
    print("\nFailed frames:", failed)
        


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Hyperparameters for the pose labeling scheme")
    parser.add_argument("-dataset", type=str, default=None, help="Dataset to run the PLS on")
    parser.add_argument("-obj_id", type=int, default=None, help="Object to run the PLS on")
    parser.add_argument("-start", type=int, default=0)
    parser.add_argument("-rs", type=int, default=42, help="Random seed")
    parser.add_argument("--uni", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args()

    utils.set_random_seed(args.rs)
    
    if args.dataset is not None:
        assert args.obj_id is not None
        OBJ_ID_LIST = [args.obj_id]
        MATERIAL_LIST = [ not args.uni]
        DATASET_LIST = [args.dataset]

    for i in range(len(DATASET_LIST)):
        run_pose_labeling_scheme(DATASET_LIST[i], OBJ_ID_LIST[i], MATERIAL_LIST[i])

