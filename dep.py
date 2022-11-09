import torch
import data
from pathlib import Path as P
from tqdm import tqdm

OBJ_ID = 3
material = True
length = 20000

data_path = P(data.id_to_path[OBJ_ID] if material else data.id_to_path_uniform[OBJ_ID])
b = tqdm(range(length), total=length)
for i in b:
    idx = str(i).zfill(6)
    p = data_path / idx / "meta_data.pt"
    d = torch.load(str(p))
    torch.save(d["obj_in_cam"], str(data_path / idx / "ground_truth.pt"))
