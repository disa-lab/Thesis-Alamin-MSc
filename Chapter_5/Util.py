import pickle
import numpy as np
import os

class Util:
    FEATURES = ["ROP", "Torque", "Weight_on_Bit"]
    TARGET = "Accel"
    DT = "Date_Time"
    def static_load_obj(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def static_save_obj(obj, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        # print("Saved object to a file: %s" % (str(file_path)))

    def get_smooth_data(data, N):
        ret = np.cumsum(data, dtype=float)
        ret[N:] = ret[N:] - ret[:-N]
        ret_avg =  ret[N - 1:] / N
        ret = np.concatenate([data[:N-1], ret_avg])
        assert len(data) == len(ret)
        return ret

    def create_directory(dir):
        if(not os.path.exists(dir)):
            print("Creating directory %s." % dir)
            os.makedirs(dir)
        else:
            print("Directory %s already exists and so returning." % dir)