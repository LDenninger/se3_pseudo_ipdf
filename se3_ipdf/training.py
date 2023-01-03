import numpy as np
import os
import torch
from tqdm import tqdm
from PIL import Image
import wandb


from .evaluation import eval_llh, eval_translation_error, rotation_model_evaluation, translation_model_evaluation


# Global variables
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


  
def cosine_decay(step, hyper_param):
    warmup_factor = min(step, hyper_param["warmup_steps"]) / hyper_param["warmup_steps"]
    decay_step = max(step - hyper_param["warmup_steps"], 0) / (hyper_param['num_train_iter']- hyper_param["warmup_steps"])
    return hyper_param['learning_rate'] * warmup_factor * (1 + np.cos(decay_step * np.pi)) / 2

def run_single_epoch(model, data_loader, hyper_param, num_iter, mode=0, optimizer=None):
    progress_bar = tqdm(enumerate(data_loader), total=num_iter)
    epoch_losses = []
    #ipdb.set_trace()

    for (i, input_) in progress_bar:
        #load the images from the batch
        img = input_['image']
        img = img.to(DEVICE).float()
        if mode==0:
            ground_truth = input_['obj_pose_in_camera'][:,:3,:3].to(DEVICE).float()
        else:
            ground_truth = input_['obj_pose_in_camera'][:,:3,-1].to(DEVICE).float()

        loss = model.compute_loss(img, ground_truth)

        epoch_losses.append(loss.item())
        
        #update the learning rate using the cosine decay
        for g in optimizer.param_groups:
            lr = cosine_decay(i+1, hyper_param)
            g['lr'] = lr
        #backward pass through the network
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)

        optimizer.step()

        if i == num_iter:
            break
    mean_epoch_loss = np.mean(epoch_losses)
    return mean_epoch_loss

def run_rotation_training(model, train_dataset, val_dataset, optimizer, hyper_param, checkpoint_dir, start_epoch=0):
    """In each epoch the model is trained for num_train_iter iterations defined in the configuration.
    The evaluation step includes calculating the loglikelihood and the spread on a num_eval_step iterations.
    The model is saved in pre-defined intervals, defined in the configuration. If additionally the full evaluation is enabled,
    also all defined evaluation metrics are calculated 
    
    """
    train_losses = []
    loglikelihood = []
    mean_errors = []
    median_errors = []
    num_epochs = hyper_param['num_epochs']
    for epoch in range(start_epoch+1, num_epochs+1):
        # training
        model.train()
        #run a single training epoch
        train_loss = run_single_epoch(model=model,
                                      data_loader=train_dataset,
                                      hyper_param=hyper_param,
                                      num_iter=hyper_param['num_train_iter'],
                                      mode=0,
                                      optimizer=optimizer)
        # validation
        model.eval()

        with torch.no_grad():
            
            llh = []

            for dataset in val_dataset:
                llh.append(eval_llh(model, dataset=dataset,
                                            num_eval_iter=hyper_param['num_val_iter'], 
                                            mode=0,
                                            device=DEVICE))
        train_losses.append(train_loss)
        loglikelihood.append(llh)
        # log the loss values 
        wandb.log({
            'TrainLoss': train_loss,
            'Loglikelihood': sum(llh)/len(llh)
        })
        print("Epoch:", epoch, "....", "TrainLoss: ", train_loss, "Loglikelihood: ", llh)
        # save a checkpoint
        if ((epoch % hyper_param['save_freq'] == 0) and epoch>=hyper_param['start_save']) or epoch == num_epochs:
            chkpt_name = f'checkpoint_{epoch}.pth' if epoch != num_epochs else f'checkpoint_final.pth'
            chkpt_path = os.path.join(checkpoint_dir, chkpt_name)
            torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                       },
                       chkpt_path
            )
            # For every saved model run a full evaluation if full_evaluation is set to true
        if hyper_param['full_eval']==True and epoch % hyper_param["eval_freq"]==0:
            rotation_model_evaluation(model=model, dataset=val_dataset, hyper_param_rot=hyper_param)
    print("Training finished.")
    print("The final IPDF model was saved to: ", os.path.join(checkpoint_dir, 'checkpoint_final.pth'))
    print("\nFinal evaluation metrics:\n")
    print("Train Loss:\n", train_losses)
    print("\nLoglikelihood:\n", loglikelihood)
    print("\nMean Error:\n", mean_errors)
    print("\nMedian Error:\n", median_errors)


def run_translation_training(model, train_dataset, val_dataset, optimizer, hyper_param, checkpoint_dir, start_epoch=0):
    """In each epoch the model is trained for num_train_iter iterations defined in the configuration.
    The evaluation step includes calculating the loglikelihood and the spread on a num_eval_step iterations.
    
    """
    num_epochs = hyper_param['num_epochs']
    for epoch in range(start_epoch+1, num_epochs+1):
        # training
        model.train()
        #run a single training epoch
        train_loss = run_single_epoch(model=model,
                                      data_loader=train_dataset,
                                      hyper_param=hyper_param,
                                      num_iter=hyper_param['num_train_iter'],
                                      mode=1,
                                      optimizer=optimizer)

        # validation
        model.eval()
        error = eval_translation_error(model, dataset=val_dataset,
                                    batch_size=hyper_param['batch_size_val'],
                                    eval_iter=hyper_param['num_val_iter'],
                                    gradient_ascent=True)
        # log the loss values 
        wandb.log({
            'TrainLoss': train_loss,
            'EstimateError': error
        })

        print("Epoch:", epoch, "....", "TrainLoss: ", train_loss, "Estimate Error: ", error)
        # save a checkpoint
        if ((epoch % hyper_param['save_freq'] == 0) and epoch>=hyper_param['start_save']) or epoch == num_epochs:
            chkpt_name = f'checkpoint_{epoch}.pth' if epoch != num_epochs else f'checkpoint_final.pth'
            chkpt_path = os.path.join(checkpoint_dir, chkpt_name)
            torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                       },
                       chkpt_path
            )
        # For every saved model run a full evaluation if full_evaluation is set to true
        """if hyper_param['full_eval']==True and epoch % hyper_param["eval_freq"]==0:
            translation_model_evaluation(model=model, dataset=val_dataset)"""
    print("Training finished.")
    print("The final IPDF model was saved to: ", os.path.join(checkpoint_dir, 'checkpoint_final.pth'))


