import torch

from .convention_transforms import convert_points_opencv_opengl

def generate_pointcloud_from_rgbd(dataset: str, seg_image, depth_image, intrinsic, obj_id=1):
    """Generate a point cloud from an RGBD image in the camera coordinate system. The camera coordinate system
    follows the OpenCV convention. In the tabletop the camera follows OpenGL convention and thus the point cloud
    must be transformed to OpenCV convention before returning.
    The segmentation image and intrinsic must be provided.
    The point cloud is return in cm.
    
    """
    if dataset=="tless":
        # Extract point cloud
        pointcloud = convert_rgbd_to_pointcloud_tless(seg_image, depth_image, intrinsic, obj_id)

        # Scale to cm
        pointcloud /= 10
    
    elif dataset=="tabletop":
        # Extract point cloud
        pointcloud = convert_rgbd_to_pointcloud_tabletop(seg_image, depth_image, intrinsic, obj_id)
        pointcloud = convert_points_opencv_opengl(pointcloud)
        # Scale to cm
        pointcloud *= 100
    
    return pointcloud


def convert_rgbd_to_pointcloud_tless(seg_image, depth_image, intrinsic, obj_id=1):
        """Main function to extract the point cloud of the object in the image

        Arguments:
                image: Batch of RGB images
                seg_data: Segmentation data containing information about the object pixels
                depth: Depth data for every pixel in the RGB image
        """
        def _set_obj_pixel():
                seg_pixel = []
                object_pixel = seg_image==obj_id
                object_pixel = torch.nonzero(object_pixel)

                return object_pixel

        with torch.no_grad():

                object_pixel = _set_obj_pixel()
                x = (torch.sub(object_pixel[:,1],intrinsic[2]))
                y = (torch.sub(object_pixel[:,0],intrinsic[3]))
                d = depth_image[object_pixel[:,0],object_pixel[:,1]]

                points = torch.cat(((d*x/intrinsic[0]).unsqueeze(-1),(d*y/intrinsic[1]).unsqueeze(-1),d.unsqueeze(-1)), dim=-1)
                points = points

        return points



def convert_rgbd_to_pointcloud_tabletop(seg_image, depth_image, intrinsic, obj_id=1):
        """Main function to extract the point cloud of the object in the image

        Arguments:
                image: Batch of RGB images
                seg_data: Segmentation data containing information about the object pixels
                depth: Depth data for every pixel in the RGB image
        """
        def _set_obj_pixel():
                seg_pixel = []
                object_pixel = seg_image==obj_id
                object_pixel = torch.nonzero(object_pixel)

                return object_pixel

        with torch.no_grad():

                object_pixel = _set_obj_pixel()
                x = -(torch.sub(object_pixel[:,1],intrinsic[2]))/depth_image.shape[1]
                y = (torch.sub(object_pixel[:,0],intrinsic[3]))/depth_image.shape[0]
                d = depth_image[object_pixel[:,0],object_pixel[:,1]]

                points = torch.cat(((d*x*intrinsic[0]).unsqueeze(-1),(d*y*intrinsic[1]).unsqueeze(-1),d.unsqueeze(-1)), dim=-1)
                points = points

        return points