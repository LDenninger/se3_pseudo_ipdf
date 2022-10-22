tless_rot_config_data = {
    'num_epochs': 100,
    'num_train_iter': 80,
    'num_val_iter': 40,
    'start_save': 2,
    'save_freq': 5,
    'batch_size': 16,
    'batch_size_val': 8,
    'learning_rate': 1e-4,
    'num_fourier_comp': 2,
    'img_size': (224, 224)*2,
    'train_mode': 2,
    'mlp_layers': [256]*2,
    'random_seed': 23,
    'warmup_steps': 8,
    'num_train_queries': 2**12,
    'dataset': "tless",
    'obj_id': 5,
    'resnet_layer': 0,
    'resnet_depth': 18,
    'train_as_test': True,
    'pseudo_gt': True,
    'occlusion': True,
    'full_eval': True
}
