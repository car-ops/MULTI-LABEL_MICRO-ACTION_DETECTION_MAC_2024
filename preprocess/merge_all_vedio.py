import os
import glob
import shutil

train_path = "/data/train"
val_path = "/data/val"
test_path = "/data/test"
save_path = "/data/all/"

os.makedirs(save_path, exist_ok=True)

train_path_list = glob.glob(train_path + "/*.mp4")
val_path_list = glob.glob(val_path + "/*.mp4")
test_path_list = glob.glob(test_path + "/*.mp4")
all_path_list = train_path_list + val_path_list + test_path_list

for vedio_path in all_path_list:
    vedio_name = os.path.basename(vedio_path)
    shutil.copy(vedio_path, os.path.join(save_path, vedio_name))
