
import torch

import data
from torch.utils.data import DataLoader



DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IPDF = False




def load_model_dataset(hyper_param, validation_only=False):

    # Validation data

    if hyper_param["dataset"]=="tabletop":
        data_val = data.RGBDPoseDataset(data_dir=data.id_to_path[hyper_param["obj_id"]], 
                                            obj_id = hyper_param["obj_id"],
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
                                            occlusion=False,
                                            train_set=False,
                                            train_as_test=False,
                                            device=DEVICE)

    val_loader = DataLoader(dataset=data_val, batch_size=hyper_param['batch_size_val'], drop_last=True ,shuffle=True, num_workers=0)

    # Training data
    if not validation_only:
        if hyper_param["dataset"]=="tabletop":
            data_train = data.RGBDPoseDataset(data_dir=data.id_to_path[hyper_param["obj_id"]],
                                                obj_id = hyper_param["obj_id"],
                                                img_size= hyper_param['img_size'],
                                                bb_crop=hyper_param['crop_image'],
                                                mask=hyper_param['mask'],
                                                full=hyper_param['full_img'],
                                                pseudo_gt=hyper_param["pseudo_gt"],
                                                single_gt=IPDF,
                                                length=hyper_param["length"],
                                                train_set=True,
                                                train_mode=True,
                                                occlusion=hyper_param['occlusion'],
                                                device=DEVICE)
        if hyper_param["dataset"]=="tless":
            data_train = data.TLESSPoseDataset(obj_id = hyper_param['obj_id'],
                                            ground_truth_mode=0,
                                            train_set=True,
                                            occlusion=True,
                                            device=DEVICE)
        train_loader = DataLoader(dataset=data_train, batch_size=hyper_param['batch_size'], drop_last=True,shuffle=True, num_workers=0)
        return train_loader, val_loader

    return val_loader

def load_pls_dataset(hyper_param):
    
    if hyper_param["dataset"]=="tless":
        dataset = data.TLESSWorkDataset(
            config=hyper_param
        )
    elif hyper_param["dataset"]=="tabletop":
        dataset = data.TabletopWorkDataset(
            config=hyper_param
        )
    data_loader = DataLoader(dataset=dataset, batch_size=1, drop_last=False, shuffle=False)

    return data_loader
    
    