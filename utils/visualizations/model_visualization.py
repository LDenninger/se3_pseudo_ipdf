import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import numpy as np
import torchvision as tv
import os

import utils.visualizations as visualizations

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
world_to_cam = Rotation.from_euler('XYZ', np.float32([0.1, 0.2, 0.3])).as_matrix()


def visualize_rotation_model(model, dataset, hyper_param, save_dir, num_batch=1, batch_size=8, displ_threshold=1e-4):

    progress_bar = tqdm(enumerate(dataset), total=num_batch)
    
    for (step, batch) in progress_bar:
        if step==num_batch:
            break
        
        # Extract data
        img = batch['image'].to(DEVICE)
        img_raw = batch['image_raw']
        img_org = batch['image_original']
        rot_gt = batch['obj_pose_in_camera'][:,:3,:3]

        # Produce IPDF
        query_rotation, probabalities = model.output_pdf(img.float())
        probabalities = torch.squeeze(probabalities, dim=-1)

        img = img.cpu()
        query_rotation = query_rotation.cpu()
        probabalities = probabalities.cpu()

        # Loop for saving the visualizations and image data
        for i in range(batch_size):

            # Save image data
            tv.utils.save_image(img_raw[i],os.path.join(save_dir, f"image_raw_{step}_{i}.png"))
            tv.utils.save_image(img[i],os.path.join(save_dir, f"image_edit_{step}_{i}.png"))
            tv.utils.save_image(img_org[i],os.path.join(save_dir, f"image_org_{step}_{i}.png"))

            # Save the visualizations of the model output
            visualizations.visualize_so3_probabilities(rotations=query_rotation, probabilities=probabalities.detach()[i], rotations_gt=rot_gt[i],
                                                    dataset=hyper_param["dataset"],
                                                    obj_id=hyper_param["obj_id"],
                                                    canonical_rotation=world_to_cam,
                                                    save_path=os.path.join(save_dir, f"visualization_rot_{step}_{i}.png"),
                                                    display_gt_set=True,
                                                    display_threshold_probability=displ_threshold)

def visualize_translation_model(model, dataset, hyper_param, save_dir, num_batch=1, batch_size=8, displ_threshold=1e-4):
    progress_bar = tqdm(enumerate(dataset), total=num_batch)
    
    for (step, batch) in progress_bar:
        if step==num_batch:
            break
        
        # Extract data
        img = batch['image'].to(DEVICE)
        img_raw = batch['image_raw']
        img_org = batch['image_original']
        trans_gt = batch['obj_pose_in_camera'][:,:3,-1]

        # Produce IPDF
        query_translation, probabalities = model.output_pdf(img.float())
        probabalities = torch.squeeze(probabalities, dim=-1)

        img = img.cpu()
        query_rotation = query_rotation.cpu()
        probabalities = probabalities.cpu()

        # Loop for saving the visualizations and image data
        for i in range(batch_size):

            # Save image data
            tv.utils.save_image(img_raw[i],os.path.join(save_dir, f"image_raw_{step}_{i}.png"))
            tv.utils.save_image(img[i],os.path.join(save_dir, f"image_edit_{step}_{i}.png"))
            tv.utils.save_image(img_org[i],os.path.join(save_dir, f"image_org_{step}_{i}.png"))

            # Save the visualizations of the model output
            visualizations.visualize_translation_probabilities(translations=query_translation, probabilities=probabalities.detach()[i], translation_gt=trans_gt[i],
                                                    save_path=os.path.join(save_dir, f"visualization_rot_{step}_{i}.png"),
                                                    display_threshold_probability=displ_threshold)

def visualize_ensamble_model(model, dataset, hyper_param, save_dir, num_batch=1, batch_size=8, displ_threshold=1e-4):
    progress_bar = tqdm(enumerate(dataset), total=num_batch)
    
    for (step, batch) in progress_bar:
        if step==num_batch:
            break
            
        img = batch['image'].to(DEVICE)
        img_raw = batch['image_raw']
        img_org = batch['image_original']
        rot_gt = batch['obj_pose_in_camera'][:,:3,:3]
        trans_gt = batch['obj_pose_in_camera'][:,:3,-1]

        query_rotation, query_translation, prob_rot, prob_trans = model.output_pdf(img.float())
        prob_rot = torch.squeeze(prob_rot, dim=-1)
        prob_trans = torch.squeeze(prob_trans, dim=-1)

        query_rotation = query_rotation.cpu()
        prob_rot = prob_rot.cpu()
        query_translation = query_translation.cpu()
        prob_trans = prob_trans.cpu()

        for i in range(batch_size):

            # Save image data
            tv.utils.save_image(img_raw[i],os.path.join(save_dir, f"image_raw_{step}_{i}.png"))
            tv.utils.save_image(img[i],os.path.join(save_dir, f"image_edit_{step}_{i}.png"))
            tv.utils.save_image(img_org[i],os.path.join(save_dir, f"image_org_{step}_{i}.png"))

            # Save the visualizations of the model rotation output
            visualizations.visualize_so3_probabilities(rotations=query_rotation, probabilities=prob_rot.detach()[i], rotations_gt=rot_gt[i],
                                                    dataset=hyper_param["dataset"],
                                                    obj_id=hyper_param["obj_id"],
                                                    canonical_rotation=world_to_cam,
                                                    save_path=os.path.join(save_dir, f"visualization_rot_{step}_{i}.png"),
                                                    display_gt_set=True,
                                                    display_threshold_probability=displ_threshold)
            
            # Save the visualizations of the model translation output
            visualizations.visualize_translation_probabilities(translations=query_translation, probabilities=prob_trans.detach()[i], translation_gt=trans_gt[i],
                                                    save_path=os.path.join(save_dir, f"visualization_rot_{step}_{i}.png"),
                                                    display_threshold_probability=displ_threshold)