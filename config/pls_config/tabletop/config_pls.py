tabletop_pls_config_data = {
    "dataset": "tabletop",
    "conv_metric": "l2",
    "obj_id": 7,
    "length": 20000,
    "skip": False,
    "resolution": (400,400),
    "max_iteration": 40,
    "icp_iteration": 1,
    "voxel_size_global": 0.2, #TLESS: 0.001
    "voxel_size_local": 0.2,  #TLESS: 0.001
    "threshold": (800, 20, 2),
    "l2_threshold": 5,
    "verbose": False,
    "global_dist_threshold": 0.4
}