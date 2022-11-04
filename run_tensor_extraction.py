from utils import tensor_extraction

START_LIST = [0]*2
END_LIST = [20000]*2
DATA_DIR_LIST = ["/home/nfs/inf6/data/datasets/IPDF_tabletop/bowl_uniform_texture", "/home/nfs/inf6/data/datasets/IPDF_tabletop/cracker_box_uniform_texture"]
OBJ_ID = [5, 4]


if __name__=="__main__":


    for (i, data_dir) in enumerate(DATA_DIR_LIST):
    # Define splits for the processes
        print("Start: ", START_LIST[i])
        print("End: ", END_LIST[i])
        split = range(START_LIST[i], END_LIST[i])

        tensor_extraction(data_dir, OBJ_ID[i], split)
