import torch
import torchvision
from pathlib import Path as P
from tqdm import tqdm
import argparse

import data
import utils

## Script to generate and save occlusion images for the photorealistic dataset ##


img_size=(224,224)
BB_SIZE = (560,560)



OccTransform = torchvision.transforms.RandomErasing(p=0.8, scale=(0.1,0.5), inplace=True)
Resizer = torchvision.transforms.Resize(size=img_size)

def occlude_bounding_box(rgb_img, seg_data, obj_id,full_size=(1080,1920)):

    object_pixel = seg_data==obj_id
    object_pixel = torch.nonzero(object_pixel).cpu()
    if object_pixel.numel() == 0:
        return None
    left_x, right_x = torch.min(object_pixel[:,1]), torch.max(object_pixel[:,1])
    top_y, low_y = torch.min(object_pixel[:,0]), torch.max(object_pixel[:,0])
    offset_x = torch.div((BB_SIZE[1] - (right_x - left_x)),2, rounding_mode="trunc")
    offset_y = torch.div((BB_SIZE[0] - (low_y - top_y)), 2, rounding_mode="trunc")
    left_x = max(0, left_x.item() - offset_x.item())
    right_x = BB_SIZE[1] + left_x

    if right_x > full_size[1]:
        right_x = full_size[1]
        left_x = full_size[1] - BB_SIZE[1]
    top_y = max(0, top_y.item() - offset_y.item())
    low_y = BB_SIZE[0] + top_y
    if low_y > full_size[0]:
        low_y = full_size[0]
        top_y = full_size[0] - BB_SIZE[0]

    seg_data = seg_data[top_y:low_y, left_x:right_x]
    rgb_img = rgb_img[:,:,:3].permute(2,0,1)


    crop_img = rgb_img[:,top_y:low_y, left_x:right_x]

    occ_img = OccTransform(crop_img)

    rgb_img[:,top_y:low_y, left_x:right_x] = occ_img

    return rgb_img, occ_img, seg_data



def generate_occlusion_image(data_path):
    

    progress_bar = tqdm(range(20000), total=20000)

    for i in progress_bar:
        if i == 20000:
            break

        index = str(i).zfill(6)

        path = data_path / index

        try:
            image = torch.load(str(path/"rgb_tensor.pt"))
            seg_data = torch.load(str(path/"seg_tensor.pt"))
        except:
            print("file does not exist!")
            continue

        occ_img, occ_crop_img, seg_data_crop = occlude_bounding_box(image, seg_data, id)
        seg_mask = torch.repeat_interleave((seg_data==id).int().unsqueeze(-1), 3, dim=-1).permute(2,0,1)
        mask_occ_img = seg_mask * occ_img
        seg_mask = torch.repeat_interleave((seg_data_crop==id).int().unsqueeze(-1), 3, dim=-1).permute(2,0,1)
        mask_occ_crop_img = seg_mask * occ_crop_img

        resize_occ_img = Resizer(occ_img)
        resize_occ_crop_img = Resizer(occ_crop_img)
        resize_mask_occ_crop_img = Resizer(mask_occ_crop_img)
        resize_mask_occ_img = Resizer(mask_occ_img)


        seg_mask = torch.repeat_interleave((seg_data==id).int().unsqueeze(-1), 3, dim=-1)
        mask_img = seg_mask * image

        torch.save(occ_img, str(path/"occ_rgb_tensor.pt"))
        torch.save(resize_occ_img, str(path/"resize_occ_rgb_tensor.pt"))
        torch.save(resize_occ_crop_img, str(path/"resize_occ_crop_rgb_tensor.pt"))

        torch.save(mask_occ_crop_img, str(path/"mask_occ_crop_rgb_tensor.pt"))
        torch.save(mask_occ_img, str(path/"mask_occ_rgb_tensor.pt"))
        torch.save(resize_mask_occ_crop_img, str(path/"resize_mask_occ_crop_rgb_tensor.pt"))
        torch.save(resize_mask_occ_img, str(path/"resize_mask_occ_rgb_tensor.pt"))







OBJ_ID = [3]

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-obj_id", type=int)
    args = parser.parse_args()

    id = args.obj_id

    data_path = P(data.id_to_path[id])
    generate_occlusion_image(data_path)
    data_path = P(data.id_to_path_uniform[id])
    generate_occlusion_image(data_path)

        
