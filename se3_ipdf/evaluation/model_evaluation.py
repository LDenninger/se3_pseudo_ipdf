import torch
import wandb

from .evaluation import eval_llh, eval_adds, eval_accuracy_angular_error, eval_recall_error, eval_spread, eval_translation_error, eval_adds

THRESHOLD = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
DEVICE= torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def full_evaluation(model, dataset, hyper_param_rot, hyper_param_trans, model_points):
    """
        Full evaluation run on the ensamble model. This includes evaluation of the seperate models using the
        evaluation metrics for rotation and translation respectively and a combined evaluation of the ensamble model.
    
    
    """


    # Evaluation of the Loglikelihood
    print("______________________________________\nStart computing Loglikelihood:\n")

    llh_rot, llh_trans, llh_comb = eval_llh(model, dataset, mode=2,
                                            num_eval_iter = hyper_param_rot['num_val_iter'],
                                            device = DEVICE)
    
    print("\nLoglikelihood rotation: ", llh_rot)
    print("\nLoglikelihood translation: ", llh_trans)
    print("\nLoglikelihood combined: ", llh_comb)
    print("\n")

    # Evaluation of the recall error
    print("______________________________________\nStart computing Recall Error:\n")

    mean_error, median_error = eval_recall_error(model, dataset,
                                                eval_iter=hyper_param_rot['num_val_iter'])
    print("\nMean Recall Error: ", mean_error)
    print("\nMedian Recall Error: ", median_error)

    # Evaluation of the translation estimate distance
    print("______________________________________\nStart computing translation estimation distance:\n")

    distance, threshold_distance, accuracy = eval_translation_error(model, dataset=dataset,
                                eval_accuracy=True,
                                batch_size=hyper_param_rot['batch_size_val'],
                                model_points=model_points,
                                threshold_list=THRESHOLD,
                                eval_iter=hyper_param_trans['num_val_iter'],
                                gradient_ascent=True)
    print("\nMean Euclidean Distance: ", distance)
    print("\nThreshold (percentage): ", THRESHOLD)
    print("\nThreshold (in centimeters): ", (100*threshold_distance).tolist())
    print("\nAccuracy (at threshold): ", accuracy)

    print("\n")
    
    # Evaluation of the rotation estimate accuracy

    print("______________________________________\nStart computing Accuracy:\n")
            
    mae, acc15, acc30 = eval_accuracy_angular_error(model, dataset,
                                                    eval_iter=40,
                                                    gradient_ascent=True)
    print("\nMean Angular Error: ", mae)
    print("Accuracy15: ", acc15)
    print("Accuracy30: ", acc30)
    print("\n")


    print("______________________________________\nStart computing ADD-S:\n")

    adds, threshold_distance_adds, mean_distance_adds = eval_adds(model,dataset,
                                                                batch_size=hyper_param_rot['batch_size_val'],
                                                                model_points=model_points,
                                                                threshold_list=THRESHOLD,
                                                                eval_iter=hyper_param_rot['num_val_iter'],
                                                                gradient_ascent=True,
                                                                device=DEVICE)
    print("\nMean ADD-S Distance: ", mean_distance_adds)
    print("\nThreshold (percentage): ", THRESHOLD)
    print("\nThreshold (in centimeters): ", (100*threshold_distance_adds).tolist())
    print("\nADD-S (at threshold): ", adds)

    print("\n\n")
    print("______________________________________\nEvaluation finished!\n")

    # Log evaluation metrics using WandB
    wandb.log({
        "RotationLoglikelihood": llh_rot,
        "TranslationLoglikelihood": llh_trans,
        "EnsambleLoglikelihood": llh_comb,
        "MeanAngularError": mae,
        "Accuracy15": acc15,
        "Accuracy30": acc30,
        "RecallMeanAngularError": mean_error,
        "RecallMedianAngularError": median_error,
        "MeanEuclideanDistance": distance,
    }, step=0)

    data = [[x,y] for (x,y) in zip(THRESHOLD, adds)]
    perc_adds_table = wandb.Table(data=data,columns=["threshold", "add_s"])

    data = [[x,y] for (x,y) in zip((100*threshold_distance).tolist(), adds)]
    dist_adds_table = wandb.Table(data=data,columns=["threshold", "add_s"])

    data = [[x,y] for (x,y) in zip(THRESHOLD, accuracy)]
    perc_accuracy_table = wandb.Table(data=data,columns=["threshold", "accuracy"])

    data = [[x,y] for (x,y) in zip((100*threshold_distance).tolist(), accuracy)]
    distance_accuracy_table = wandb.Table(data=data,columns=["threshold", "acucracy"])


    wandb.log({
        "adds_at_percentage_threshold": wandb.plot.line(perc_adds_table, "threshold", "add_s",
                                                            title="ADD-S vs threshold in percentage"),
        "adds_at_distance_threshold": wandb.plot.line(dist_adds_table, "threshold", "add_s",
                                                            title="ADD-S vs threshold in cm"),
        "trans_accuracy_at_percentage_threshold": wandb.plot.line(perc_accuracy_table, "threshold", "accuracy",
                                                            title="Translation accuracy vs threshold in percentage"),
        "trans_accuracy_at_distance_threshold": wandb.plot.line(distance_accuracy_table, "threshold", "accuracy",
                                                            title="Translation accuracy vs threshold in cm")
                                                            
    }, step=0)

    
    """
    eval_file = os.path.join(save_dir, "eval_results.txt")

    with open(config_file_name_rot, 'r') as f:
        config_content_rot = f.read()

    with open(config_file_name_trans, 'r') as f:
        config_content_trans = f.read()
    
    with open(eval_file, 'w') as f:
        f.write("Meta Data:\n\n")
        f.write("Evaluated experiment: "+ exp_dir+ "\n")
        f.write("Model: "+ args.model+"\n")
        f.write("Random Seed: "+ str(args.r) +"\n")
        f.write("______________________________\n")

        f.write("Evaluation Metrics:\n\n")

        f.write("Rotation model evaluation:\n")
        f.write("Loglikelihood (Precision): "+ str(llh_rot)+ "\n")
        f.write("Mean Angular Error (Precision): "+ str(mae)+ "\n")
        f.write("Accuracy15 (Precision): "+ str(acc15)+ "\n")
        f.write("Accuracy30 (Precision): "+ str(acc30)+ "\n")
        f.write("Mean geodesic Error (Recall): "+ str(mean_error)+ "\n")
        f.write("Median geodesic Error (Recall): "+ str(median_error)+ "\n")

        f.write("\nTranslation model evaluation:\n")
        f.write("Loglikelihood (Precision): "+ str(llh_trans)+ "\n")
        f.write("Mean Euclidean Distance: "+ str(distance)+ "\n")
        f.write("Threshold (percentage): "+ ",".join(str(x) for x in THRESHOLD)+ "\n")
        f.write("Threshold (in meters): "+ ",".join(str(x) for x in (100*threshold_distance).tolist())+ "\n")
        f.write("Accuracy (at threshold): "+ ",".join(str(x) for x in accuracy)+ "\n")

        f.write("\nCombined model evaluation")
        f.write("Loglikelihood (precision): "+ str(llh_comb)+ "\n")
        f.write("ADD-S (at threshold) (Precision): " + ",".join(str(x) for x in adds)+ "\n")
        f.write("Threshold (percentage): "+ ",".join(str(x) for x in THRESHOLD)+ "\n")
        f.write("Threshold (in meters): "+ ",".join(str(x) for x in (100*threshold_distance_adds).tolist())+ "\n")
        f.write("Mean Distance: "+ str(mean_distance_adds) + "\n")

        f.write("______________________________\n")
        f.write("Rotation Config File:\n\n")
        f.write(config_content_rot)
        f.write("______________________________\n")
        f.write("Translation Config File:\n\n")
        f.write(config_content_trans)
        f.write("______________________________\n")

        print("Saved results to: ", eval_file)

    """

def rotation_model_evaluation(model, dataset, hyper_param_rot, model_points):

    # Evaluation of the Loglikelihood
    print("______________________________________\nStart computing Loglikelihood:\n")

    llh_rot = eval_llh(model, dataset, mode=0,
                        num_eval_iter = hyper_param_rot['num_val_iter'],
                        device = DEVICE)
    
    print("\nLoglikelihood rotation: ", llh_rot)
    print("\n")

    # Evaluation of the recall error
    print("______________________________________\nStart computing Recall Error:\n")
    mean_error, median_error = eval_recall_error(model, dataset, hyper_param_rot)
    print("\nMean Recall Error: ", mean_error)
    print("\nMedian Recall Error: ", median_error)

    # Evaluation of the rotation estimate accuracy

    print("______________________________________\nStart computing Accuracy:\n")
    mae, acc15, acc30 = eval_accuracy_angular_error(model, dataset, hyper_param_rot,
                                                    gradient_ascent=True)
    print("\nMean Angular Error: ", mae)
    print("Accuracy15: ", acc15)
    print("Accuracy30: ", acc30)
    print("\n")


    """print("______________________________________\nStart computing ADD-S:\n")

    adds, threshold_distance_adds, mean_distance_adds = eval_adds(model,dataset,
                                                                batch_size=hyper_param_rot['batch_size_val'],
                                                                model_points=model_points,
                                                                threshold_list=THRESHOLD,
                                                                eval_iter=hyper_param_rot['num_val_iter'],
                                                                gradient_ascent=True,
                                                                mode=0,
                                                                device=device_1)
    print("\nMean ADD-S Distance: ", mean_distance_adds)
    print("\nThreshold (percentage): ", THRESHOLD)
    print("\nThreshold (in centimeters): ", (100*threshold_distance_adds).tolist())
    print("\nADD-S (at threshold): ", adds)
    """
    print("\n\n")
    print("______________________________________\nEvaluation finished!\n")

    # Log evaluation metrics using WandB
    wandb.log({
        "RotationLoglikelihood": llh_rot,
        "MeanAngularError": mae,
        "Accuracy15": acc15,
        "Accuracy30": acc30,
        "RecallMeanAngularError": mean_error,
        "RecallMedianAngularError": median_error,
    })

    """data = [[x,y] for (x,y) in zip(THRESHOLD, adds)]
    perc_adds_table = wandb.Table(data=data,columns=["threshold", "add_s"])

    data = [[x,y] for (x,y) in zip((100*threshold_distance_adds).tolist(), adds)]
    dist_adds_table = wandb.Table(data=data,columns=["threshold", "add_s"])
    wandb.log({
        "adds_at_percentage_threshold": wandb.plot.line(perc_adds_table, "threshold", "add_s",
                                                            title="ADD-S vs threshold in percentage"),
        "adds_at_distance_threshold": wandb.plot.line(dist_adds_table, "threshold", "add_s",
                                                            title="ADD-S vs threshold in cm")
                                                            
    })"""

    
def translation_model_evaluation(model, dataset, hyper_param_trans, model_points):

    # Evaluation of the Loglikelihood
    print("______________________________________\nStart computing Loglikelihood:\n")

    llh_trans= eval_llh(model, dataset, mode=1,
                        num_eval_iter = hyper_param_trans['num_val_iter'],
                        device = DEVICE)
    
    print("\nLoglikelihood translation: ", llh_trans)
    print("\n")

    # Evaluation of the translation estimate distance
    print("______________________________________\nStart computing translation estimation distance:\n")

    distance, threshold_distance, accuracy = eval_translation_error(model, dataset=dataset,
                                eval_accuracy=True,
                                batch_size=hyper_param_trans['batch_size_val'],
                                model_points=model_points,
                                threshold_list=THRESHOLD,
                                eval_iter=hyper_param_trans['num_val_iter'],
                                gradient_ascent=True)
    print("\nMean Euclidean Distance: ", distance)
    print("\nThreshold (percentage): ", THRESHOLD)
    print("\nThreshold (in centimeters): ", (100*threshold_distance).tolist())
    print("\nAccuracy (at threshold): ", accuracy)

    print("\n")

    # Log evaluation metrics using WandB
    wandb.log({
        "TranslationLoglikelihood": llh_trans,
        "MeanEuclideanDistance": distance,
    })

    data = [[x,y] for (x,y) in zip(THRESHOLD, accuracy)]
    perc_accuracy_table = wandb.Table(data=data,columns=["threshold", "accuracy"])

    data = [[x,y] for (x,y) in zip((100*threshold_distance).tolist(), accuracy)]
    distance_accuracy_table = wandb.Table(data=data,columns=["threshold", "acucracy"])


    wandb.log({
        "trans_accuracy_at_percentage_threshold": wandb.plot.line(perc_accuracy_table, "threshold", "accuracy",
                                                            title="Translation accuracy vs threshold in percentage"),
        "trans_accuracy_at_distance_threshold": wandb.plot.line(distance_accuracy_table, "threshold", "accuracy",
                                                            title="Translation accuracy vs threshold in cm")
                                                            
    })