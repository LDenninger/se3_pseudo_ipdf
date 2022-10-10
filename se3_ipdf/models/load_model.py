import torch
import os
from torch.optim import Adam

from .implicit_so3 import ImplicitSO3
from .implicit_translation import ImplicitTranslation
from .implicit_se3_ensamble import ImplicitSE3_Ensamble

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
translation_range = torch.tensor([[-0.5,0.5],[-0.5,0.5],[-1,0]])

def load_translation_model(hyper_param, arguments, load_model=None, init_model=False):
    """Load the translation model. Either it is completely new initialized or the model state is load from
    a checkpoint file.

    
    """


    model = ImplicitTranslation(resnet_depth=18, feat_dim=512, # 100352
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
        chkpt_dir = os.path.join(("experiments/exp_"+ arguments.exp_name), 'models_translation') 

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
    

    return model, optimizer, start_epoch

def load_rotation_model(hyper_param, arguments, load_model=None, init_model=False):
    ## Load the rotation-model from a given checkpoint. If no checkpoint is provided the model will be newly initialized ##
    model = ImplicitSO3(resnet_depth=hyper_param['resnet_depth'], 
                                    feat_dim=512 if hyper_param['resnet_depth']==18 else 2048, # 
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
        chkpt_dir = os.path.join(("experiments/exp_"+ arguments.exp_name), 'models_rotation') 

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

  

    return model, optimizer, start_epoch

def load_ensamble_model(hyper_param_rot, hyper_param_trans, arguments, init_mode=False):

    model_rot = load_rotation_model(hyper_param_rot, arguments, init_model=(not init_mode))

    model_trans = load_translation_model(hyper_param_trans, arguments, init_model=(not init_mode))
    # Initialize ensamble SE(3) model

    model_ensamble = ImplicitSE3_Ensamble(
        rotation_model=model_rot,
        translation_model=model_trans
    )

    if not init_mode:
        model_path = os.path.join(arguments.exp_dir, os.path.join("models_ensamble",  arguments.model))

        try:
            model_ensamble.load_state_dict(torch.load(model_path)['model_state_dict'])
            print("Model was load from: ", model_path) 
        except IOError:
            print("Model could not be loaded...\nTrying to initialize new ensamble model from existing models...\n")
            return load_ensamble_model(hyper_param_rot, hyper_param_trans, arguments, init_mode=True)

    model_ensamble = model_ensamble.to(DEVICE)

    return model_ensamble