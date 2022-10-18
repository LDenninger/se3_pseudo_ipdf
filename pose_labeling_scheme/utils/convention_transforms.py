import torch



def convert_transformation_opencv_opengl(transformation):
    """Transforms a rigid body transformation from OpenCV to OpenGL convention and back.

    Arguments:
        transformation: rigid body transformations, shape: [N,4,4]/[4,4]
    """
    
    CONV_MATRIX = torch.eye(3)
    CONV_MATRIX[1,1] *= -1
    CONV_MATRIX[2,2] *= -1
    CONV_MATRIX = CONV_MATRIX.type(transformation.dtype)
    CONV_MATRIX = CONV_MATRIX.to(transformation.device)

    if len(transformation.shape)==2:
        transformation = transformation.unsqueeze(0)

    transformation[:,:3,:3] = CONV_MATRIX @ transformation[:,:3,:3]
    transformation[:,:3,-1] = transformation[:,:3,-1] @ CONV_MATRIX

    return transformation

def convert_points_opencv_opengl(points):
    """Transforms a point cloud from OpenCV to OpenGL convention and back.

    Arguments:
        points: point cloud, shape: [N,3]
    """
    CONV_MATRIX = torch.eye(3)
    CONV_MATRIX[1,1] *= -1
    CONV_MATRIX[2,2] *= -1
    CONV_MATRIX = CONV_MATRIX.type(points.dtype)
    CONV_MATRIX = CONV_MATRIX.to(points.device)

    points = points @ CONV_MATRIX

    return points
