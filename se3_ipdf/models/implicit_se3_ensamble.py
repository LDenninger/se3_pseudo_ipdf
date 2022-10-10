import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImplicitSE3_Ensamble(nn.Module):
    """ Model reconstructs a pose distribution over SE(3). It encapsulates
    two models responsible for predicting the rotation in SO(3) and the translation in R³ respectively.
    Both models show the same architecture and are seperately trained. The models are combined in one ensamble model.
    
    """

    def __init__(self, rotation_model,  translation_model):
        super(ImplicitSE3_Ensamble, self).__init__()
        
        self.rotation_model = rotation_model
        self.translation_model = translation_model


    def forward(self, input):
        
        out_rot = self.rotation_model(input)
        out_trans = self.translation_model(input)

        return out_rot, out_trans

    def predict_probability(self, images, pose):
        """Predict the probability of the given pose by forwarding it through the models seperately
        
        """
        device = images.device
        rotation_prob = self.rotation_model._predict_probability(images, pose[:,:3,:3])
        translation_prob = self.translation_model._predict_probability(images, pose[:,:3,-1])

        pose_probability = rotation_prob[:,-1]*translation_prob[:,-1]

        return pose_probability

    def predict_pose(self, images, gradient_ascent=False):
        """The models predict the rotation and translation respectively using gradient ascent.
        The rotation concatenated with the translation is given out as the single most likely pose
        """

        rotation_estimate = self.rotation_model.predict_rotation(images, gradient_ascent=gradient_ascent)
        translation_estimate = self.translation_model.predict_translation(images, gradient_ascent=gradient_ascent)

        pose_estimate = torch.repeat_interleave(torch.eye(4).unsqueeze(0), images.shape[0], dim=0).to(images.device)
        pose_estimate[:,:3,:3] = rotation_estimate
        pose_estimate[:,:3,-1] = translation_estimate

        return pose_estimate
    
    def predict_translation(self, images, gradient_ascent=False):
        """Predict the most likely translation
        """

        translation_estimate = self.translation_model.predict_translation(images, gradient_ascent=gradient_ascent)
        return translation_estimate
    
    def predict_rotation(self, images, gradient_ascent=False):
        """Predict the most lilkely rotation
        """
        rotation_estimate = self.rotation_model.predict_rotation(images, gradient_ascent=gradient_ascent)
        return rotation_estimate



    def output_pdf(self, images, rotation=True, translation=True, joint_distribution=False):
        with torch.no_grad():
            if rotation and translation:
                query_rotations, rotation_prob = self.rotation_model.output_pdf(images)
                if joint_distribution:
                    query_translation = self.translation_model.predict_rotation(images, gradient_ascent=True)
                    translation_prob = torch.ones_like(query_translation)
                else:
                    query_translation, translation_prob = self.translation_model.output_pdf(images)
                return query_rotations, query_translation, rotation_prob, translation_prob
            if rotation:
                query_rotations, rotation_prob = self.rotation_model.output_pdf(images)
                return query_rotations, rotation_prob
            if translation:
                if joint_distribution:
                    query_translation = self.translation_model.predict_rotation(images, gradient_ascent=True)
                    translation_prob = torch.ones_like(query_translation)
                else:
                    query_translation, translation_prob = self.translation_model.output_pdf(images)
                return query_translation, translation_prob

        

