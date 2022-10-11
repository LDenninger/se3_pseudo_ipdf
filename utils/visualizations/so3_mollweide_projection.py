import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import matrix_to_euler_angles

from utils import get_symmetry_ground_truth

def visualize_so3_probabilities(rotations,
                                probabilities,
                                obj_id,
                                dataset="tabletop",
                                rotations_gt=None,
                                ax=None,
                                fig=None,
                                save_path=None,
                                display_gt_set=False,
                                display_threshold_probability=0,
                                show_color_wheel=True,
                                canonical_rotation=Rotation.from_euler('XYZ', np.float32([0.1, 0.2, 0.3])).as_matrix(),
                                return_img=False):
    """Plot a single distribution on SO(3) using the tilt-colored method.
    Args:
      rotations: [N, 3, 3] tensor of rotation matrices
      probabilities: [N] tensor of probabilities
      rotations_gt: [N_gt, 3, 3] or [3, 3] ground truth rotation matrices
      ax: The matplotlib.pyplot.axis object to paint
      fig: The matplotlib.pyplot.figure object to paint
      display_threshold_probability: The probability threshold below which to omit
        the marker
      to_image: If True, return a tensor containing the pixels of the finished
        figure; if False return the figure itself
      show_color_wheel: If True, display the explanatory color wheel which matches
        color on the plot with tilt angle
      canonical_rotation: A [3, 3] rotation matrix representing the 'display
        rotation', to change the view of the distribution.  It rotates the
        canonical axes so that the view of SO(3) on the plot is different, which
        can help obtain a more informative view.
    Returns:
      A matplotlib.pyplot.figure object, or a tensor of pixels if to_image=True.
    """
    assert isinstance(rotations, torch.Tensor)
    #assert isinstance(rotations_gt, torch.Tensor)
    assert isinstance(probabilities, torch.Tensor)

    def _show_single_marker(ax, rotation, marker, edgecolors=True,
                          facecolors=False):
      eulers = matrix_to_euler_angles(rotation, 'XYZ')
      cmap = plt.cm.hsv
      xyz = rotation[:, 0]
      tilt_angle = eulers[0]
      longitude = np.arctan2(xyz[2], -xyz[0])
      latitude = np.arcsin(xyz[1])
      expr = 0.5 + tilt_angle / 2 / np.pi
      color = cmap(expr.item())
      
      ax.scatter(longitude, latitude, s=500,
                edgecolors=color if edgecolors else 'none',
                facecolors=facecolors if facecolors else 'none',
                marker=marker,
                linewidth=4)

    def _generate_llc(rotations):
      '''
      Generate longitute, latitude, color for all rotations
      '''
      llc_dict = defaultdict(list)
      for rotation in rotations:
        r = Rotation.from_matrix(rotation.numpy())
        eulers = torch.from_numpy(r.as_euler('XYZ'))
        cmap = plt.cm.hsv
        
        eulersPyTorch = matrix_to_euler_angles(rotation, 'XYZ')

        # 2 * PI == 0
        close_idx = torch.isclose(eulers.abs().float(), torch.full(eulers.shape,np.pi))
        eulers[close_idx] = 0.
        xyz = rotation[:, 0]
        tilt_angle = eulers[0]
        longitude = np.arctan2(xyz[0], -xyz[1])
        latitude = np.arcsin(xyz[2])
        expr = 0.5 + tilt_angle / 2 / np.pi
        color = cmap(expr.item())
        llc_dict[longitude.item(), latitude.item()].append(color)
      for k, v in llc_dict.items():
          llc_dict[k] = sorted(v)
      return llc_dict

    def _show_llc(ax, llc_dict, marker, edgecolors=True, facecolors=False):
      # Main functio for ground truth marker
        for ll, colors in llc_dict.items():
          num_colors = len(colors)
          for ci, color in enumerate(colors):
            ax.scatter(ll[0], ll[1], s=200 * (ci + 1) ,
                      edgecolors=color if edgecolors else 'none',
                      facecolors=facecolors if facecolors else 'none',
                      marker=marker,
                      linewidth=2)
        for ll, colors in llc_dict.items():
          ax.scatter(ll[0], ll[1], s=200 ,
                        edgecolors='none',
                        facecolors='#ffffff',
                        marker=marker,
                        linewidth=2)

          

        
    if ax is None:
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='mollweide')
    if rotations_gt is not None and len(rotations_gt.shape) == 2:
      rotations_gt = rotations_gt.unsqueeze(0)
    display_rotations = rotations @ canonical_rotation
    cmap = plt.cm.hsv
    scatterpoint_scaling = 6e2
    eulers_queries = matrix_to_euler_angles(display_rotations, 'XYZ')
    xyz = display_rotations[:, :, 0]
    tilt_angles = eulers_queries[:, 0]

    longitudes = np.arctan2(xyz[:, 0], -xyz[:, 1])
    latitudes = np.arcsin(xyz[:, 2])
    probabilities = probabilities.numpy()
    which_to_display = (probabilities > display_threshold_probability)

    if rotations_gt is not None:
      # The visualization is more comprehensible if the GT
      # rotation markers are behind the output with white filling the interior.
      if display_gt_set==True:
        rotations_gt = get_symmetry_ground_truth(rotations_gt, obj_id, dataset)
      display_rotations_gt = rotations_gt @ canonical_rotation

      
      #for rotation in display_rotations_gt:
        #_show_single_marker(ax, rotation, 'o')
      
      llc_map = _generate_llc(display_rotations_gt)
      _show_llc(ax, llc_map, 'o')
      
      # Cover up the centers with white markers
      # for rotation in display_rotations_gt:
      #  _show_single_marker(ax, rotation, 'o', edgecolors=False,
      #                      facecolors='#ffffff')

    # Display the distribution
    ax.scatter(
        longitudes[which_to_display],
        latitudes[which_to_display],
        s=scatterpoint_scaling * probabilities[which_to_display],
        c=cmap(0.5 + tilt_angles[which_to_display] / 2. / np.pi,
        ))
   # ipdb.set_trace()
    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if show_color_wheel:
      # Add a color wheel showing the tilt angle to color conversion.
      ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection='polar')
      theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
      radii = np.linspace(0.4, 0.5, 2)
      _, theta_grid = np.meshgrid(radii, theta)
      colormap_val = 0.5 + theta_grid / np.pi / 2.
      ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap)
      ax.set_yticklabels([])
      ax.set_xticklabels([r'90$\degree$', None,
                          r'180$\degree$', None,
                          r'270$\degree$', None,
                          r'0$\degree$'], fontsize=14)
      ax.spines['polar'].set_visible(False)
      plt.text(0.5, 0.5, 'Tilt', fontsize=14,
              horizontalalignment='center',
              verticalalignment='center', transform=ax.transAxes)

    if save_path is None:
      plt.show()
    else:
      plt.savefig(save_path)
      
    return fig


def visualize_so3_rotations(rotations,
                                obj_id,
                                dataset="tabletop",
                                rotations_mark=None,
                                rotations_gt=None,
                                ax=None,
                                display_gt_set=False,
                                fig=None,
                                save_path=None,
                                show_color_wheel=True,
                                canonical_rotation=Rotation.from_euler('XYZ', np.float32([0.1, 0.2, 0.3])).as_matrix(),
                                labels=None):
    """Plot a single distribution on SO(3) using the tilt-colored method.
    Args:
      rotations: [N, 3, 3] tensor of rotation matrices
      probabilities: [N] tensor of probabilities
      rotations_gt: [N_gt, 3, 3] or [3, 3] ground truth rotation matrices
      ax: The matplotlib.pyplot.axis object to paint
      fig: The matplotlib.pyplot.figure object to paint
      display_threshold_probability: The probability threshold below which to omit
        the marker
      to_image: If True, return a tensor containing the pixels of the finished
        figure; if False return the figure itself
      show_color_wheel: If True, display the explanatory color wheel which matches
        color on the plot with tilt angle
      canonical_rotation: A [3, 3] rotation matrix representing the 'display
        rotation', to change the view of the distribution.  It rotates the
        canonical axes so that the view of SO(3) on the plot is different, which
        can help obtain a more informative view.
    Returns:
      A matplotlib.pyplot.figure object, or a tensor of pixels if to_image=True.
    """
    assert isinstance(rotations, torch.Tensor)
    #assert isinstance(rotations_gt, torch.Tensor)

    def _show_single_marker(ax, rotation, marker, edgecolors=True,
                          facecolors=False):
      eulers = matrix_to_euler_angles(rotation, 'XYZ')
      cmap = plt.cm.hsv
      xyz = rotation[:, 0]
      tilt_angle = eulers[0]
      longitude = np.arctan2(xyz[2], -xyz[0])
      latitude = np.arcsin(xyz[1])
      expr = 0.5 + tilt_angle / 2 / np.pi
      color = cmap(expr.item())
      
      ax.scatter(longitude, latitude, s=500,
                edgecolors=color if edgecolors else 'none',
                facecolors=facecolors if facecolors else 'none',
                marker=marker,
                linewidth=4)

    def _generate_llc(rotations):
      '''
      Generate longitute, latitude, color for all rotations
      '''
      llc_dict = defaultdict(list)
      for rotation in rotations:
        r = Rotation.from_matrix(rotation.numpy())
        eulers = torch.from_numpy(r.as_euler('XYZ'))
        cmap = plt.cm.hsv
        
        eulersPyTorch = matrix_to_euler_angles(rotation, 'XYZ')

        # 2 * PI == 0
        close_idx = torch.isclose(eulers.abs().float(), torch.full(eulers.shape,np.pi))
        eulers[close_idx] = 0.
        xyz = rotation[:, 0]
        tilt_angle = eulers[0]
        longitude = np.arctan2(xyz[0], -xyz[1])
        latitude = np.arcsin(xyz[2])
        expr = 0.5 + tilt_angle / 2 / np.pi
        color = cmap(expr.item())
        llc_dict[longitude.item(), latitude.item()].append(color)
      for k, v in llc_dict.items():
          llc_dict[k] = sorted(v)
      return llc_dict

    def _show_llc(ax, llc_dict, marker, edgecolors=True, facecolors=False):
        for ll, colors in llc_dict.items():
          num_colors = len(colors)
          for ci, color in enumerate(colors):
            ax.scatter(ll[0], ll[1], s=250 * (ci + 1) ,
                      edgecolors=color if edgecolors else 'none',
                      facecolors=facecolors if facecolors else 'none',
                      marker=marker,
                      linewidth=1)
        for ll, colors in llc_dict.items():
          ax.scatter(ll[0], ll[1], s=200 ,
                        edgecolors='none',
                        facecolors='#ffffff',
                        marker=marker,
                        linewidth=2)
    
        
    if ax is None:
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='mollweide')
    if rotations_gt is not None and len(rotations_gt.shape) == 2:
      rotations_gt = rotations_gt.unsqueeze(0)
    display_rotations = rotations @ canonical_rotation
    cmap = plt.cm.hsv
    scatterpoint_scaling = 4e2
    eulers_queries = matrix_to_euler_angles(display_rotations, 'XYZ')
    xyz = display_rotations[:, :, 0]
    tilt_angles = eulers_queries[:, 0]

    longitudes = np.arctan2(xyz[:, 0], -xyz[:, 1])
    latitudes = np.arcsin(xyz[:, 2])

  
    if rotations_gt is not None:
      # The visualization is more comprehensible if the GT
      # rotation markers are behind the output with white filling the interior.
      if display_gt_set==True:
        rotations_gt = get_symmetry_ground_truth(rotations_gt, obj_id, dataset)
      display_rotations_gt = rotations_gt @ canonical_rotation

      #for rotation in display_rotations_gt:
        #_show_single_marker(ax, rotation, 'o')
      
      llc_map = _generate_llc(display_rotations_gt)
      _show_llc(ax, llc_map, 'o')
      
      # Cover up the centers with white markers
      # for rotation in display_rotations_gt:
      #  _show_single_marker(ax, rotation, 'o', edgecolors=False,
      #                      facecolors='#ffffff')
    if rotations_mark is not None:
      for rotation in rotations_mark:
        _show_single_marker(ax, rotation, 'o',
                            edgecolors="#000000")
                            
      
    # Display the distribution
    ax.scatter(
        longitudes,
        latitudes,
        s=scatterpoint_scaling*0.2,
        c=cmap(0.5 + tilt_angles / 2. / np.pi,
        ))
   # ipdb.set_trace()
    #Labels can be given to annotate single rotations
    if labels is not None:
      #ipdb.set_trace()
      text_list = []
      for i, lbl in enumerate(labels):
        #ax.annotate(round(lbl.item(), 2), (longitudes[i], latitudes[i]))
        text_list.append(ax.text(longitudes[i], latitudes[i], round(lbl.item(), 2),rotation=0, color='black', fontsize=13))
      #ipdb.set_trace()
      #adt.adjust_text(text_list)
      
    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if show_color_wheel:
      # Add a color wheel showing the tilt angle to color conversion.
      ax = fig.add_axes([0.86, 0.17, 0.12, 0.12], projection='polar')
      theta = np.linspace(-3 * np.pi / 2, np.pi / 2, 200)
      radii = np.linspace(0.4, 0.5, 2)
      _, theta_grid = np.meshgrid(radii, theta)
      colormap_val = 0.5 + theta_grid / np.pi / 2.
      ax.pcolormesh(theta, radii, colormap_val.T, cmap=cmap)
      ax.set_yticklabels([])
      ax.set_xticklabels([r'90$\degree$', None,
                          r'180$\degree$', None,
                          r'270$\degree$', None,
                          r'0$\degree$'], fontsize=14)
      ax.spines['polar'].set_visible(False)
      plt.text(0.5, 0.5, 'Tilt', fontsize=14,
              horizontalalignment='center',
              verticalalignment='center', transform=ax.transAxes)

    if save_path is None:
      plt.show()
    else:
      plt.savefig(save_path)