import torch
import matplotlib.pyplot as plt




def visualize_transformations(pc_canon, transform_1, transform_2, filename):
    """Visualizations of two point clouds transformed by the provided rigid transformation matrix.
        The point cloud produced using the first transformation is visualized in green, the second one in red.
    
    """

    p_1 = pc_canon @ transform_1[:3,:3].T + transform_1[:3,-1]
    p_2 = pc_canon @ transform_2[:3,:3].T + transform_2[:3,-1]
    
    center = torch.mean(p_1, dim=0, keepdim=True)

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(p_1[:,0], p_1[:,1], p_1[:,2], s=1, color="green", alpha=0.3)
    ax.scatter(p_2[:,0], p_2[:,1], p_2[:,2], s=1, color="red", alpha=0.3)

    ax.set_xlim(center[0]-0.5, center[0]+0.5)
    ax.set_ylim(center[1]-0.5, center[1]+0.5)
    ax.set_zlim(center[2]-0.5, center[2]+0.5)

    fig.savefig(filename)
    plt.close()

def visualize_pointclouds(p_1, p_2, filename):
    """ p_1 is painted green and p_2 is painted red 
    
    """
    center = torch.mean(p_1, dim=0, keepdim=True).squeeze()

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(p_1[:,0], p_1[:,1], p_1[:,2], s=1, color="green", alpha=0.3)
    ax.scatter(p_2[:,0], p_2[:,1], p_2[:,2], s=1, color="red", alpha=0.3)
    ax.set_xlim(center[0]-10, center[0]+10)
    ax.set_ylim(center[1]-10, center[1]+10)
    ax.set_zlim(center[2]-10, center[2]+10)

    fig.savefig(filename)
    plt.close()