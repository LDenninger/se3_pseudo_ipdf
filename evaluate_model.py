import wandb
import torch
import argparse
import errno
import os
import yaml


import se3_ipdf.evaluation as evaluation
import se3_ipdf.models as models
import data


def tune(model, dataset, hyper_param_rot, model_points=None):

    llh_rot_list = []
    mean_error_list = []
    median_error_list = []
    mae_list = []
    acc15_list = []
    acc30_list = []


    for (i, dataset_obj) in enumerate(dataset):

        # Evaluation of the rotation estimate accuracy

        print("______________________________________\nStart computing Accuracy:\n")
        mae, acc5, acc15, acc30 = evaluation.eval_accuracy_angular_error(model, dataset_obj, hyper_param_rot,
                                                        gradient_ascent=True)
        print("\nMean Angular Error: ", mae)
        print("Accuracy5: ", acc5)
        print("Accuracy15: ", acc15)
        print("Accuracy30: ", acc30)
        print("\n")

            
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Hyperparameters for evaluation")
    parser.add_argument("-exp_name", type=str, help="Name of the experiment")
    parser.add_argument("-rot_epoch", type=str, help="Epoch the checkpoint to load the rotation-model is taken from")
    parser.add_argument("-trans_epoch", type=str, help="Epoch the checkpoint to load the rotation-model is taken from")
    parser.add_argument('-r_seed', metavar='INTEGER', type=int, default='42', help='Random seed used for evaluation')
    parser.add_argument("--mode", type=int, help="evaluation mode: 0: evaluation on rotation model, 1: evaluation on translation model, 2: full evaluation on the ensamble model")
    parser.add_argument('--wandb', action='store_true', help='Log data to WandB')

    args = parser.parse_args()
    #Check if the experiment directory exists
    exp_dir = "experiments/exp_"+args.exp_name
    if not os.path.isdir(exp_dir):
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), exp_dir)

    # Define the directory the evaluation results are saved to
    
    wandb.login()


    if args.wandb == True:

        if args.mode==0:
            with wandb.init(project="SO3_IPDF", entity="ipdf_se3", resume="allow", id=args.exp_name):
                wandb.run.name = args.exp_name

                config_file_name = os.path.join(exp_dir, "config_rotation.yaml")
                with open(config_file_name, 'r') as f:
                    hyper_param = yaml.safe_load(f)

                # Load the model
                model = models.load_rotation_model(
                    hyper_param=hyper_param,
                    arguments=args
                )[0]
                # Load the dataset
                dataset = data.load_model_dataset(hyper_param=hyper_param, validation_only=True)

                # Load the object model
                if hyper_param["dataset"]=="tabletop":
                    obj_model, diameter = data.load_ycbv_object_model(hyper_param["obj_id"], pointcloud_only=True)
                elif hyper_param["dataset"]=="tless":
                    obj_model, diameter = data.load_tless_object_model(hyper_param["obj_id"], pointcloud_only=True)
                else:
                    print("Object model for dataset/object was not found!")

                save_dir=os.path.join(exp_dir, os.path.join("experiments", ("evaluation_"+args.rot_epoch)))
                if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)
                evaluation.rotation_model_evaluation(model=model, dataset=dataset, hyper_param_rot=hyper_param, model_points=obj_model)
        if args.mode==1:
            with wandb.init(project="Translation_IPDF", entity="ipdf_se3", resume="allow", id=args.exp_name):
                wandb.run.name = args.exp_name

                config_file_name = os.path.join(exp_dir, "config_translation.yaml")
                with open(config_file_name, 'r') as f:
                    hyper_param = yaml.safe_load(f)
                # Load the model
                model = models.load_translation_model(
                    hyper_param=hyper_param,
                    arguments=args

                )
                # Load the dataset
                dataset = data.load_model_dataset(hyper_param=hyper_param, validation_only=True)

                # Load the object model
                if hyper_param["dataset"]=="tabletop":
                    obj_model, diameter = data.load_ycbv_object_model(hyper_param["obj_id"], pointcloud_only=True)
                elif hyper_param["dataset"]=="tless":
                    obj_model, diameter = data.load_tless_object_model(hyper_param["obj_id"], pointcloud_only=True)
                else:
                    print("Object model for dataset/object was not found!")


                save_dir=os.path.join(exp_dir, os.path.join("experiments", ("evaluation_"+args.trans_epoch)))
                if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)
                evaluation.translation_model_evaluation(model=model, dataset=dataset, hyper_param_trans=hyper_param, model_points=obj_model)

        if args.mode==2:
            with wandb.init(project="SE3_ENSAMBLE", entity="ipdf_se3", resume="allow", id=args.exp_name):
                wandb.run.name = args.exp_name +"_"+args.rot_epoch+"_"+args.trans_epoch

                # Load translation model configuration
                config_file_name = os.path.join(exp_dir, "config_translation.yaml")
                with open(config_file_name, 'r') as f:
                    hyper_param_trans = yaml.safe_load(f)

                # Load rotation model configuration
                config_file_name = os.path.join(exp_dir, "config_rotation.yaml")
                with open(config_file_name, 'r') as f:
                    hyper_param_rot = yaml.safe_load(f)

                # Load the ensamble model
                model = models.load_ensamble_model(
                    hyper_param_rot=hyper_param_rot,
                    hyper_param_trans=hyper_param_trans,
                    arguments=args
                )

                # Load the dataset
                dataset = data.load_model_dataset(hyper_param=hyper_param, validation_only=True)

                # Load the object model
                if hyper_param["dataset"]=="tabletop":
                    obj_model, diameter = data.load_ycbv_object_model(hyper_param["obj_id"], pointcloud_only=True)
                elif hyper_param["dataset"]=="tless":
                    obj_model, diameter = data.load_tless_object_model(hyper_param["obj_id"], pointcloud_only=True)
                else:
                    print("Object model for dataset/object was not found!")


                save_dir=os.path.join(exp_dir, os.path.join("experiments", ("evaluation_"+args.rot_epoch+"_"+args.trans_epoch)))
                if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)

                evaluation.full_evaluation(model=model, dataset=dataset, hyper_param_rot=hyper_param_rot, hyper_param_trans=hyper_param_trans, model_points=obj_model)
    
    else:
        if args.mode==0:

            with wandb.init(mode='disabled'):
                wandb.run.name = args.exp_name
                config_file_name = os.path.join(exp_dir, "config_rotation.yaml")
                with open(config_file_name, 'r') as f:
                    hyper_param = yaml.safe_load(f)

                # Load the model
                model = models.load_rotation_model(
                    hyper_param=hyper_param,
                    arguments=args
                )[0]
                # Load the dataset
                dataset = data.load_model_dataset(hyper_param=hyper_param, validation_only=True)

                # Load the object model
                if hyper_param["dataset"]=="tabletop":
                    obj_model, diameter = data.load_ycbv_object_model(hyper_param["obj_id"][0], pointcloud_only=True)
                elif hyper_param["dataset"]=="tless":
                    obj_model, diameter = data.load_tless_object_model(hyper_param["obj_id"][0], pointcloud_only=True)
                else:
                    print("Object model for dataset/object was not found!")

                save_dir=os.path.join(exp_dir, os.path.join("experiments", ("evaluation_"+args.rot_epoch)))
                if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)
                #evaluation.rotation_model_evaluation(model=model, dataset=dataset, hyper_param_rot=hyper_param, model_points=obj_model)
                tune(model=model, dataset=dataset, hyper_param_rot=hyper_param, model_points=obj_model)


        if args.mode==1:
            with wandb.init(mode='disabled'):

                wandb.run.name = args.exp_name

                config_file_name = os.path.join(exp_dir, "config_translation.yaml")
                with open(config_file_name, 'r') as f:
                    hyper_param = yaml.safe_load(f)

                # Load the model
                model = models.load_translation_model(
                    hyper_param=hyper_param,
                    arguments=args

                )
                # Load the dataset
                dataset = data.load_model_dataset(hyper_param=hyper_param, validation_only=True)

                # Load the object model
                if hyper_param["dataset"]=="tabletop":
                    obj_model, diameter = data.load_ycbv_object_model(hyper_param["obj_id"], pointcloud_only=True)
                elif hyper_param["dataset"]=="tless":
                    obj_model, diameter = data.load_tless_object_model(hyper_param["obj_id"], pointcloud_only=True)
                else:
                    print("Object model for dataset/object was not found!")


                save_dir=os.path.join(exp_dir, os.path.join("experiments", ("evaluation_"+args.trans_epoch)))
                if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)

                evaluation.translation_model_evaluation(model=model, dataset=dataset, hyper_param_trans=hyper_param, model_points=obj_model)


        if args.mode==2:
            with wandb.init(mode='disabled'):

                wandb.run.name = args.exp_name +"_"+args.rot_epoch+"_"+args.trans_epoch

                save_dir=os.path.join(exp_dir, os.path.join("experiments", ("evaluation_"+args.rot_epoch+"_"+args.trans_epoch)))
                if not os.path.isdir(save_dir):
                        os.makedirs(save_dir)
                evaluation.full_evaluation(args, visualization=args.vis)
    
    wandb.finish()


