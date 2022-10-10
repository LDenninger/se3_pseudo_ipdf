import healpy as hp  # pylint: disable=g-import-not-at-top
import numpy as np
import torch
from pytorch3d.transforms import euler_angles_to_matrix

def generate_healpix_grid(recursion_level=None, size=None):
    """
    Generates an equivolumetric grid on SO(3) following Yershova et al. (2010).
    Uses a Healpix grid on the 2-sphere as a starting point and then tiles it
    along the 'tilt' direction 6*2**recursion_level times over 2pi.
    Args:
        recursion_level: An integer which determines the level of resolution of the
            grid. The final number of points will be 72*8**recursion_level.
            A recursion_level of 2 (4k points) was used for training and 5 (2.4M points)
            for evaluation.
        size: A number of rotations to be included in the grid.  The nearest grid
            size in log space is returned.
    Returns:
        (N, 3, 3) array of rotation matrices, where N=72*8**recursion_level.
    """

    assert not(recursion_level is None and size is None)
    if size:
        recursion_level = max(int(np.round(np.log(size/72.)/np.log(8.))), 0)
    number_per_side = 2**recursion_level
    number_pix = hp.nside2npix(number_per_side)
    s2_points = hp.pix2vec(number_per_side, np.arange(number_pix))
    s2_points = np.stack([*s2_points], 1)

    # Take these points on the sphere and
    azimuths = np.arctan2(s2_points[:, 1], s2_points[:, 0])
    tilts = np.linspace(0, 2 * np.pi, 6 * 2**recursion_level, endpoint=False)
    polars = np.arccos(s2_points[:, 2])
    grid_rots_mats = []
    for tilt in tilts:
        # Build up the rotations from Euler angles, zyz formiat
        rot_mats = euler_angles_to_matrix(torch.stack([torch.from_numpy(azimuths),
                                                       torch.zeros(number_pix),
                                                       torch.zeros(number_pix)], 1),
                                          convention='XYZ')
        rot_mats = rot_mats @ euler_angles_to_matrix(torch.stack([torch.zeros(number_pix),
                                                                  torch.zeros(number_pix),
                                                                  torch.from_numpy(polars)], 1),
                                                     convention='XYZ')
        rot_mats = rot_mats.float() @ torch.unsqueeze(
            euler_angles_to_matrix(torch.Tensor([tilt, 0., 0.]), convention='XYZ'), dim=0)
        grid_rots_mats.append(rot_mats)

    grid_rots_mats = torch.cat(grid_rots_mats, dim=0)
    return grid_rots_mats

