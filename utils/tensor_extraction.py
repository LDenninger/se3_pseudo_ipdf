import torch
import torchvision as tv
from tqdm import tqdm
import os


img_size=(224,224)
BB_SIZE = (560,560)

resizer = tv.transforms.Resize(size=img_size)


def crop_bounding_box(rgb_img, seg_data, obj_id,full_size=(1080,1920)):

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
    rgb_img = rgb_img[:,:,:3]
    rgb_img = rgb_img[top_y:low_y, left_x:right_x].permute(2, 0, 1)
    rgb_img = resizer(rgb_img)

    return rgb_img

def tensor_extraction(data_dir, obj_id, frame_list):

    loop = tqdm(enumerate(frame_list), total=len(frame_list))

    for (i, step) in loop:
        if i == len(frame_list):
            break
        
        # Define frame id and directory
        ind = str(step).zfill(6)
        frame_dir = os.path.join(data_dir, ind)
    
        # Extract the data from the meta data tensor

        meta_data = torch.load(os.path.join(frame_dir, "meta_data.pt")) 

        # image: [H,W,C]
        image = torch.from_numpy(meta_data['rgb_tensor'][...,:3])
        seg_data = torch.from_numpy(meta_data['seg_tensor'].astype("int32"))
        depth_data = torch.from_numpy(meta_data['depth_tensor'])
        ground_truth = meta_data['obj_in_cam']

        seg_mask = torch.repeat_interleave((seg_data==obj_id).int().unsqueeze(-1), 3, dim=-1)
        mask_img = seg_mask * image
        # Create a resized image for the input of the ResNet
        mask_cropped_img = crop_bounding_box(mask_img, seg_data, obj_id)
        if mask_cropped_img is None:
            print("Frame is corrupted: ", step)
            continue
        cropped_img = crop_bounding_box(image, seg_data, obj_id)

        resized_img = resizer(torch.clone(image.permute(2,0,1)))
        resized_cropped_img = resizer(torch.clone(cropped_img))
        resized_mask_cropped_img = resizer(torch.clone(mask_cropped_img))
        resized_mask_img = resizer(torch.clone(mask_img))


        # Save the extracted tensors seperately
        
        torch.save(image, os.path.join(frame_dir, "rgb_tensor.pt"))
        torch.save(seg_data, os.path.join(frame_dir, "seg_tensor.pt"))
        torch.save(depth_data, os.path.join(frame_dir, "depth_tensor.pt"))
        torch.save(ground_truth, os.path.join(frame_dir, "ground_truth.pt"))
        torch.save(resized_img, os.path.join(frame_dir, "resize_rgb_tensor.pt"))
        torch.save(mask_img, os.path.join(frame_dir, "mask_rgb_tensor.pt"))
        torch.save(mask_cropped_img, os.path.join(frame_dir, "mask_crop_rgb_tensor.pt"))
        torch.save(resized_cropped_img, os.path.join(frame_dir, "resize_crop_rgb_tensor.pt"))
        torch.save(resized_mask_cropped_img, os.path.join(frame_dir, "resize_mask_crop_rgb_tensor.pt"))
        torch.save(resized_mask_img, os.path.join(frame_dir, "resize_mask_rgb_tensor.pt"))   