tabletop_rot_config_data = {
    'num_epochs': 30,
    'num_train_iter': 400,
    'num_val_iter': 60,
    'start_save': 2,
    'save_freq': 10,
    'eval_freq': 5,
    'batch_size': 32,
    'batch_size_val': 16,
    'learning_rate': 1e-4,
    'num_fourier_comp': 3,
    'img_size': (224, 224),
    'crop_image': True,
    'train_mode': 2,
    'mlp_layers': [256]*3,
    'random_seed': 44,
    'warmup_steps': 40,
    'num_train_queries': 2**12,
    'dataset': "tabletop",
    'obj_id': [5],
    'mask': True,
    'backbone': "convnext_tiny",
    'backbone_layer': 1,
    'pseudo_gt': True,
    'single_gt': False,
    'occlusion': False,
    'full_img': False,
    'full_eval': True,
    'length': 15000,
    'material': True
}
