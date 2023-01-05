import argparse
from tqdm import tqdm


import data
import utils
import config
import pose_labeling_scheme

OBJ_ID_LIST = [5]*2
MATERIAL_LIST = [True, False]
DATASET_LIST = ["tabletop"]*2
SAVE_FILE = "cleaned_pseudo_gt_thesis.pth"
ANG_THRESHOLD = [5]*2


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Hyperparameters for validating the pseudo gt")
    parser.add_argument("-dataset", type=str, help="Dataset to run the PLS on")
    parser.add_argument("-obj_id", type=int, help="Object to run the PLS on")
    parser.add_argument("-mode", type=int, default=0, help="Mode for validation: 0: Duplicate check, 1: Convergence validation, 2: Convention cleanup for tabletop")
    parser.add_argument("-ang_th", type=int, default=15, help="Angular threshold for the duplicate check")
    parser.add_argument("-rs", type=int, default=42, help="Random seed")
    parser.add_argument("--clean", action="store_true", default=False, help="Use the already cleaned pseudo gt set")
    args = parser.parse_args()

    utils.set_random_seed(args.rs)


    # Run the validation

    for i in range(len(DATASET_LIST)):

        # Load the config file corresponding to dataset and object
        pls_config = config.load_pls_config(DATASET_LIST[i], OBJ_ID_LIST[i])
        pls_config["material"] = MATERIAL_LIST[i]

        # Load the data
        data_loader = data.load_pls_dataset(pls_config, material=MATERIAL_LIST[i], return_pgt=True, cleaned_pgt=args.clean)

        if args.mode == 0:
            pose_labeling_scheme.run_duplicate_check(dataset=data_loader, hyper_param=pls_config, angular_threshold=ANG_THRESHOLD[i], save_file=SAVE_FILE)
        elif args.mode == 1:
            # Load the object model
            if args.dataset=="tabletop":
                object_model, diameter = data.load_ycbv_object_model(args.obj_id, pointcloud_only=True)
                object_model_sl = data.load_sl_cad_model(args.dataset, args.obj_id)
            elif args.dataset=="tless":
                object_model, diameter = data.load_tless_object_model(args.obj_id, pointcloud_only=True)
                object_model_sl = data.load_sl_cad_model(args.dataset, args.obj_id)
            else:
                print("\nNo object model for the given dataset/object!")

            pose_labeling_scheme.run_convergence_check(dataset=data_loader, obj_model=object_model, obj_model_sl=object_model_sl, hyper_param=pls_config)
        elif args.mode == 2:
            pose_labeling_scheme.run_convention_cleanup(dataset=data_loader, hyper_param=pls_config)

    print("\nValidation done!")