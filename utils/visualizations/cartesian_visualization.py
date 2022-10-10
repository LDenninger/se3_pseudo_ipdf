import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import matrix_to_euler_angles

def visualize_translation_probabilities(translations,
                                        probabilities, 
                                        translations_gt=None, 
                                        save_path=None, 
                                        display_threshold_probability=0,
                                        return_img=False):
    
    assert isinstance(translations, torch.Tensor)
    assert isinstance(translations_gt, torch.Tensor)
    assert isinstance(probabilities, torch.Tensor)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatterpoint_scaling = 4e2
    probabilities = probabilities.numpy()
    which_to_display = (probabilities > display_threshold_probability)

    display_color = np.array([[0,0,1]])
    ax.scatter(
            translations[which_to_display,0], 
            translations[which_to_display,1], 
            translations[which_to_display,2],
            s=scatterpoint_scaling * probabilities[which_to_display],
            c=display_color,
            alpha=0.3)

    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)
    ax.set_zlim(-1,0)


    if translations_gt is not None:
        gt_color = np.array([[1,0,0]])
        ax.scatter(
            translations_gt[0], 
            translations_gt[1], 
            translations_gt[2],
            s=10,
            c=gt_color)
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    
    return fig
    
