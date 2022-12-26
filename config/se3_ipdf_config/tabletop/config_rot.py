tabletop_rot_config_data = {
    'num_epochs': 30,
    'num_train_iter': 200,
    'num_val_iter': 60,
    'start_save': 2,
    'save_freq': 4,
    'batch_size': 64,
    'batch_size_val': 16,
    'learning_rate': 1e-4,
    'num_fourier_comp': 3,
    'img_size': (224, 224),
    'crop_image': True,
    'train_mode': 2,
    'mlp_layers': [256]*4,
    'random_seed': 23,
    'warmup_steps': 20,
    'num_train_queries': 2**12,
    'dataset': "tless",
    'obj_id': [5],
    'mask': True,
    'backbone': "resnet18",
    'backbone_layer': 0,
    'pseudo_gt': True,
    'single_gt': False,
    'occlusion': False,
    'full_img': False,
    'full_eval': True,
    'length': 15000,
    'material': True
}
