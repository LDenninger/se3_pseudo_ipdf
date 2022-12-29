from utils import tensor_extraction, tensor_correction
from data import id_to_path, id_to_path_uniform


START_LIST = [0]*2
END_LIST = [20000]*2
DATA_DIR_LIST = [""]
OBJ_ID = [8]*2


if __name__=="__main__":


    for (i, obj_id) in enumerate(OBJ_ID):
        if i%2==0:
            data_dir = id_to_path[obj_id]
        else:
            data_dir = id_to_path_uniform[obj_id]
    # Define splits for the processes
        print("Start: ", START_LIST[i])
        print("End: ", END_LIST[i])
        split = range(START_LIST[i], END_LIST[i])

        try:
            tensor_extraction(data_dir, OBJ_ID[i], split)
        except:
            continue
