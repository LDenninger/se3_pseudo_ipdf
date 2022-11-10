import torch
from pathlib import Path as P
from tqdm import tqdm
import os

id_to_path = {
    3: "/home/nfs/inf6/data/datasets/IPDF_tabletop/can_material_texture",
    4: "/home/nfs/inf6/data/datasets/IPDF_tabletop/cracker_box_material_texture",
    5: "/home/nfs/inf6/data/datasets/IPDF_tabletop/bowl_material_texture"
}

id_to_path_uniform = {
    3: "/home/nfs/inf6/data/datasets/IPDF_tabletop/uniform_texture",
    4: "/home/nfs/inf6/data/datasets/IPDF_tabletop/cracker_box_uniform_texture",
    5: "/home/nfs/inf6/data/datasets/IPDF_tabletop/bowl_uniform_texture"
}

OBJ_ID = 3
material = True
length = 20000

data_path = P(id_to_path[OBJ_ID] if material else id_to_path_uniform[OBJ_ID])
b = tqdm(range(length), total=length)
for i in b:
    idx = str(i).zfill(6)
    if os.path.exists(str(data_path / idx / "ground_truth.pt")):
        continue
    p = data_path / idx / "meta_data.pt"
    try:
        d = torch.load(str(p))
    except:
        continue
    
    torch.save(d["obj_in_cam"], str(data_path / idx / "ground_truth.pt"))
