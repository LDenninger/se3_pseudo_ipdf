
import torch

import data
from torch.utils.data import DataLoader



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IPDF = False


def load_single_model_dataset(hyper_param, validation_only=False):

    # Validation data
    if hyper_param["material"]:
        data_dir = data.id_to_path[hyper_param["obj_id"][0]]
    else:
        data_dir = data.id_to_path_uniform[hyper_param["obj_id"][0]]

    if hyper_param["dataset"]=="tabletop":
        data_val = data.TabletopPoseDataset(data_dir=data_dir, 
                                            obj_id = hyper_param["obj_id"][0],
                                            img_size= hyper_param['img_size'],
                                            bb_crop=hyper_param['crop_image'],
                                            mask=hyper_param['mask'],
                                            full=hyper_param['full_img'],
                                            pseudo_gt = False,
                                            single_gt=True,
                                            length=hyper_param["length"],
                                            train_set=False,
                                            train_mode=False if validation_only else True,
                                            full_data=validation_only,
                                            occlusion=hyper_param['occlusion'],
                                            device=DEVICE)
    if hyper_param["dataset"]=="tless":
        data_val = data.TLESSPoseDataset(obj_id = hyper_param['obj_id'],
                                            ground_truth_mode=0,
                                            occlusion=hyper_param["occlusion"],
                                            train_set=False,
                                            train_as_test=hyper_param["train_as_test"])

    val_loader = DataLoader(dataset=data_val, batch_size=hyper_param['batch_size_val'], drop_last=True ,shuffle=True, num_workers=8)

    # Training data
    if not validation_only:
        if hyper_param["dataset"]=="tabletop":
            data_train = data.TabletopPoseDataset(data_dir=data_dir,
                                                obj_id = hyper_param["obj_id"][0],
                                                img_size= hyper_param['img_size'],
                                                bb_crop=hyper_param['crop_image'],
                                                mask=hyper_param['mask'],
                                                full=hyper_param['full_img'],
                                                pseudo_gt=hyper_param["pseudo_gt"],
                                                single_gt=hyper_param["single_gt"],
                                                length=hyper_param["length"],
                                                train_set=True,
                                                train_mode=True,
                                                occlusion=hyper_param['occlusion'],
                                                device=DEVICE)
        if hyper_param["dataset"]=="tless":
            data_train = data.TLESSPoseDataset(obj_id = hyper_param['obj_id'],
                                            ground_truth_mode=1 if hyper_param["pseudo_gt"] else 0,
                                            train_set=True,
                                            occlusion=hyper_param["occlusion"])
        train_loader = DataLoader(dataset=data_train, batch_size=hyper_param['batch_size'], drop_last=True,shuffle=True, num_workers=8)
        return train_loader, val_loader

    return val_loader

def load_model_dataset(hyper_param, include_translation=False, validation_only=False):

    validation_datasets = []
    training_datasets = []

    #import ipdb; ipdb.set_trace()

    for obj_id in hyper_param["obj_id"]:
        # Validation data
        if hyper_param["material"]:
            data_dir = data.id_to_path[obj_id]
        else:
            data_dir = data.id_to_path_uniform[obj_id]

        if hyper_param["dataset"]=="tabletop":
            data_val = data.TabletopPoseDataset(data_dir=data_dir, 
                                                obj_id = obj_id,
                                                img_size= hyper_param['img_size'],
                                                bb_crop=hyper_param['crop_image'],
                                                mask=hyper_param['mask'],
                                                full=hyper_param['full_img'],
                                                pseudo_gt = False,
                                                single_gt=True,
                                                length=hyper_param["length"],
                                                train_set=False,
                                                train_mode=False if validation_only else True,
                                                full_data=validation_only,
                                                occlusion=hyper_param['occlusion'],
                                                device=DEVICE)
        if hyper_param["dataset"]=="tless":
            data_val = data.TLESSPoseDataset(obj_id = hyper_param['obj_id'],
                                                ground_truth_mode=0,
                                                occlusion=hyper_param["occlusion"],
                                                train_set=False,
                                                train_as_test=hyper_param["train_as_test"])
        validation_datasets.append(data_val)

        # Training data
        if not validation_only:
            if hyper_param["dataset"]=="tabletop":
                data_train = data.TabletopPoseDataset(data_dir=data_dir,
                                                    obj_id = obj_id,
                                                    img_size= hyper_param['img_size'],
                                                    bb_crop=hyper_param['crop_image'],
                                                    mask=hyper_param['mask'],
                                                    full=hyper_param['full_img'],
                                                    pseudo_gt=hyper_param["pseudo_gt"],
                                                    single_gt=hyper_param["single_gt"],
                                                    length=hyper_param["length"],
                                                    train_set=True,
                                                    train_mode=True,
                                                    occlusion=hyper_param['occlusion'],
                                                    device=DEVICE)
            if hyper_param["dataset"]=="tless":
                if hyper_param["pseudo_gt"]:
                    gt_mode = 1
                elif hyper_param["single_gt"]:
                    gt_mode = 0
                else:
                    gt_mode = 2

                data_train = data.TLESSPoseDataset(obj_id = obj_id,
                                                ground_truth_mode= gt_mode,
                                                train_set=True,
                                                occlusion=hyper_param["occlusion"])
            training_datasets.append(data_train)
    
    val_loader_list = [DataLoader(dataset=dat, batch_size=hyper_param['batch_size_val'], drop_last=True ,shuffle=True, num_workers=8) for dat in validation_datasets]
    if validation_only:
        return val_loader_list

    training_concat_dataset = torch.utils.data.ConcatDataset(training_datasets)
    train_loader = DataLoader(dataset=training_concat_dataset, batch_size=hyper_param['batch_size'], drop_last=True,shuffle=True, num_workers=8)
    return train_loader, val_loader_list

def load_pls_dataset(hyper_param, material=None, start=0,return_gt=False, return_pgt=False, cleaned_pgt=False):
    
    if hyper_param["dataset"]=="tless":
        dataset = data.TLESSWorkDataset(
            config=hyper_param,
            start=start,
            return_pgt=return_pgt,
            return_gt=return_gt or hyper_param["verbose"],
            cleaned_pgt=cleaned_pgt
        )
    elif hyper_param["dataset"]=="tabletop":
        dataset = data.TabletopWorkDataset(
            config=hyper_param,
            start=start,
            return_pgt=return_pgt,
            return_gt=return_gt or hyper_param["verbose"],
            material= material if material is not None else hyper_param["material"],
            cleaned_pgt=cleaned_pgt
        )
    data_loader = DataLoader(dataset=dataset, batch_size=1, drop_last=False, shuffle=False, num_workers=0)

    return data_loader
    
def load_demonstration_dataset(obj_id, mode, img_size=(224,224)):

    poses = data.generate_dataset(mode=mode)
    size = poses.shape[0]

    dataset = data.YCBPoseDataset(
        obj_id = obj_id,
        img_size = img_size,
        poses = poses
    )

    data_loader = DataLoader(dataset=dataset, batch_size=1, drop_last=False, shuffle=False, num_workers = 0)

    return data_loader, size