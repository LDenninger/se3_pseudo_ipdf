import os
import yaml
import json
import torch
from pytorch3d.io import load_ply
from pytorch3d.renderer import TexturesVertex
import stillleben as sl


obj_id_to_file= {
    3: ["obj_000001.ply", 1],
    4: ["obj_000002.ply", 2],
    5: ["obj_000013.ply", 13],
    6: ["obj_000003.ply", 3],
    8: ["obj_000014.ply", 14]
}


def load_tless_object_model(obj_id, pointcloud_only=False, model_type=2, device="cpu"):
    """ Load the 3D model of an TLESS-object and return it in meters.
    
    """
    if model_type==0:
        # CAD model
        model_path = os.path.join("/home/nfs/inf6/data/datasets/T-Less/t-less_v2/models_cad", ("obj_"+str(obj_id).zfill(2)+".ply"))
    if model_type==1:
        # Subdivided CAD model
        model_path = os.path.join("/home/nfs/inf6/data/datasets/T-Less/t-less_v2/models_cad_subdivided", ("obj_"+str(obj_id).zfill(2)+".ply"))
    if model_type==2:
        # Reconstructed model
        model_path = os.path.join("/home/nfs/inf6/data/datasets/T-Less/t-less_v2/models_reconst", ("obj_"+str(obj_id).zfill(2)+".ply"))


    with open("/home/nfs/inf6/data/datasets/T-Less/t-less_v2/models_cad/models_info.yml", "r") as f:
        models_info = yaml.safe_load(f)
    diameter = models_info[obj_id]["diameter"]/1000

    verts ,faces = load_ply(model_path)
    verts /= 1000

    if not pointcloud_only:
        verts_rgb = torch.ones(verts.shape[0], 3)[None]
        textures = TexturesVertex(verts_features=verts_rgb.to(device))
        mesh_data = [verts.to(device), faces.to(device), textures.to(device)]
        return mesh_data, diameter

    return verts.to(device), diameter


def load_ycbv_object_model(obj_id, pointcloud_only=False, downsample=False, device="cpu"):

    """ Load the 3D model of an ycbv-object and return it in meters.
    
    """
 
    if downsample:
        model_path = os.path.join("/home/nfs/inf6/data/datasets/YCB_VIDEO_DATASET/YCB_Video_Dataset", "models_bop_compat_eval",obj_id_to_file[obj_id][0])
    else:
        model_path = os.path.join("/home/nfs/inf6/data/datasets/YCB_VIDEO_DATASET/YCB_Video_Dataset", "models_bop",obj_id_to_file[obj_id][0])
    with open("/home/nfs/inf6/data/datasets/YCB_VIDEO_DATASET/YCB_Video_Dataset/models_bop/models_info.json", 'r') as f:
        models_info = json.load(f)
    # Scale to meters
    diameter = models_info[str(obj_id_to_file[obj_id][1])]["diameter"]/1000

    verts ,faces = load_ply(model_path)
    verts = verts/1000

    if not pointcloud_only:
        verts_rgb = torch.ones(verts.shape[0], 4)[None]
        textures = TexturesVertex(verts_features=verts_rgb.to(device))
        mesh_data = [verts.to(device), faces.to(device), textures.to(device)]
        return mesh_data, diameter

    return verts.to(device), diameter

def load_sl_cad_model(dataset, obj_id):
    """Load a TLESS or YCB-V object model using the stillleben loader. It is return in a stillleben.Mesh-object in meters.
    """

    if dataset=="tabletop":
        model_path = [os.path.join("/home/nfs/inf6/data/datasets/YCB_VIDEO_DATASET/YCB_Video_Dataset", "models_bop",obj_id_to_file[obj_id][0])]
        # Load the model
        model = sl.Mesh.load_threaded(model_path)[0]
        pts = model.points
        pts /= 1000
        model.set_new_positions(pts)
    
    elif dataset=="tless":
          # model path
        model_path = [os.path.join("/home/nfs/inf6/data/datasets/T-Less/t-less_v2/models_cad", f"obj_{obj_id:02d}.ply")]
        # Load the model

        model = sl.Mesh.load_threaded(model_path)[0]
        pts = model.points
        pts /= 1000
        model.set_new_positions(pts)
    
    else:
        print("\nNo object models corresponding to the given dataset!")
        return None

    return model
