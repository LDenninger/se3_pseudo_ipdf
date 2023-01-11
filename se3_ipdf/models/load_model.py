import torch
import os
from pathlib import Path as P
from torch.optim import Adam

from .convNext import load_convnext_model
from .backbones import ResNet
from .implicit_so3 import ImplicitSO3
from .implicit_translation import ImplicitTranslation
from .implicit_se3_ensamble import ImplicitSE3_Ensamble

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
translation_range = torch.tensor([[-0.5,0.5],[-0.5,0.5],[-1,0]])

resnet_feature_dim = {
    18:{
        0: 512,
        1: 25088,
        2: 50176
    },
    50:{
        0: 2048,
        1: 100352,
        2: 200704
    }
}

convnext_feature_dim = {
    "tiny" : {
        0: 1000,
        1: 768,
        2: 37632,
        3: 75264,
    }
}

def load_backbone(hyper_param):
    import ipdb; ipdb.set_trace()
    if hyper_param["backbone"]=="resnet18":
        feature_extractor = ResNet(depth=18, layer=hyper_param["backbone_layer"], pretrained=True)
        feature_dim = resnet_feature_dim[18][hyper_param["backbone_layer"]]
    elif hyper_param["backbone"]=="resnet50":
        feature_extractor = ResNet(depth=50, layer=hyper_param["backbone_layer"], pretrained=True)
        feature_dim = resnet_feature_dim[50][hyper_param["backbone_layer"]]
    elif hyper_param["backbone"]=="convnext_tiny":
        feature_extractor = load_convnext_model(size="tiny", remove_layer=hyper_param["backbone_layer"])
        feature_dim = convnext_feature_dim["tiny"][hyper_param["backbone_layer"]]
    elif hyper_param["backbone"]=="convnext_small":
        feature_extractor = load_convnext_model(size="small", remove_layer=hyper_param["backbone_layer"])
        feature_dim = 1000
    elif hyper_param["backbone"]=="convnext_base":
        feature_extractor = load_convnext_model(size="base", remove_layer=hyper_param["backbone_layer"])
        feature_dim = 1000
    elif hyper_param["backbone"]=="convnext_large":
        feature_extractor = load_convnext_model(size="large", remove_layer=hyper_param["backbone_layer"])
        feature_dim = 0
    elif hyper_param["backbone"]=="vgg":
        feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
        feature_dim= 1000

    return feature_extractor, feature_dim




def load_translation_model(hyper_param, arguments, exp_name=None, load_model=None, only_model=False, init_model=False):
    """Load the translation model. Either it is completely new initialized or the model state is load from
    a checkpoint file.

    
    """

    feature_extractor, feature_dim = load_backbone(hyper_param)


    model = ImplicitTranslation(feature_extractor=feature_extractor, feat_dim=feature_dim, # 100352
                                   mlp_layer_sizes=hyper_param['mlp_layers'],
                                   num_fourier_comp=hyper_param['num_fourier_comp'],
                                   num_train_queries=hyper_param['num_train_queries'],
                                   translation_range = translation_range)
    
    model = model.to(DEVICE)
    
    if init_model:
        return model

    optimizer = Adam(model.parameters(), lr=hyper_param['learning_rate'])
    start_epoch=0

    if arguments.trans_epoch is not None:
        load_model = "checkpoint_" + arguments.trans_epoch +".pth"

    if load_model is not None:
        chkpt_dir = os.path.join(("experiments/exp_"+exp_name), 'models_translation') 

        chkpt_path = os.path.join(chkpt_dir, load_model)

        try:
            checkpoint = torch.load(chkpt_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print("Checkpoint was loaded from: " + chkpt_path + "\n")
        except:
            print("Translation model could not be loaded...\n")
    
    model = model.to(DEVICE)

    if only_model:
        return model
    

    return model, optimizer, start_epoch

def load_rotation_model(hyper_param, arguments, exp_name=None, load_model=None, init_model=False, only_model=False):

    ## Load the rotation-model from a given checkpoint. If no checkpoint is provided the model will be newly initialized ##

    feature_extractor, feature_dim = load_backbone(hyper_param)
    
    model = ImplicitSO3(feature_extractor=feature_extractor,
                                    feat_dim=feature_dim, # 
                                    mlp_layer_sizes=hyper_param['mlp_layers'],
                                    num_fourier_comp=hyper_param['num_fourier_comp'],
                                    num_train_queries=hyper_param['num_train_queries'],
                                    num_eval_queries=2**19)
    model = model.to(DEVICE)
    
    if init_model:
        return model
    optimizer = Adam(model.parameters(), lr=hyper_param['learning_rate'])
    start_epoch=0

    if arguments.rot_epoch is not None:
        load_model = "checkpoint_" + arguments.rot_epoch +".pth"

    if load_model is not None:     
        chkpt_dir = os.path.join(("experiments/exp_"+ exp_name), 'models_rotation') 

        chkpt_path = os.path.join(chkpt_dir, load_model)
        try:
            checkpoint = torch.load(chkpt_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print("Checkpoint was loaded from: " + chkpt_path + "\n")
        except:
            print("Rotation model could not be loaded...\n")
    
    model = model.to(DEVICE)

    if only_model:
        return model
    return model, optimizer, start_epoch

def load_ensamble_model(hyper_param_rot, hyper_param_trans, arguments, exp_name=None, init_mode=False):

    model_rot = load_rotation_model(hyper_param_rot, arguments, exp_name, only_model=True, init_model=(not init_mode))

    model_trans = load_translation_model(hyper_param_trans, arguments, exp_name, only_model=True, init_model=(not init_mode))
    # Initialize ensamble SE(3) model

    model_ensamble = ImplicitSE3_Ensamble(
        rotation_model=model_rot,
        translation_model=model_trans,
    )

    if not init_mode:
        model_path = os.path.join("experiments/exp_"+ exp_name, os.path.join("models_ensamble",  ("ensamble_"+arguments.rot_epoch+"_"+arguments.trans_epoch+".pth")))

        try:
            model_ensamble.load_state_dict(torch.load(model_path))
            print("Model was load from: ", model_path) 
        except:
            print("Model could not be loaded...\nTrying to initialize new ensamble model from existing models...\n")
            return load_ensamble_model(hyper_param_rot, hyper_param_trans, arguments, exp_name, init_mode=True)

    model_ensamble = model_ensamble.to(DEVICE)

    return model_ensamble


def load_evaluation_model(hyper_param_rot, hyper_param_trans, exp_dir):

    feature_extractor, feature_dim = load_backbone(hyper_param_rot)
    
    model_rot = ImplicitSO3(feature_extractor=feature_extractor,
                                    feat_dim=feature_dim, # 
                                    mlp_layer_sizes=hyper_param_rot['mlp_layers'],
                                    num_fourier_comp=hyper_param_rot['num_fourier_comp'],
                                    num_train_queries=hyper_param_rot['num_train_queries'],
                                    num_eval_queries=2**19)


    feature_extractor, feature_dim = load_backbone(hyper_param_trans)

    model_trans = ImplicitTranslation(feature_extractor=feature_extractor, feat_dim=feature_dim, # 100352
                                mlp_layer_sizes=hyper_param_trans['mlp_layers'],
                                num_fourier_comp=hyper_param_trans['num_fourier_comp'],
                                num_train_queries=hyper_param_trans['num_train_queries'],
                                translation_range = translation_range)
    model_ensamble = ImplicitSE3_Ensamble(
        rotation_model=model_rot,
        translation_model=model_trans,
    )

    try:
        model_path = next((P(exp_dir) / "models").glob('*.pth'))

        model_ensamble.load_state_dict(torch.load(str(model_path)))
        print("Model was load from: ", str(model_path)) 
    except:
        print("Model could not been loaded from: ", str(model_path))

    model_ensamble = model_ensamble.to(DEVICE)

    return model_ensamble