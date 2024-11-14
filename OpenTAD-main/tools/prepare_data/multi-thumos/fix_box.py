import os
import glob
import json
import pickle

# def read_json(json_path):
#     with open(json_path, 'r') as file:
#         data = json.load(file)
#     return data

# json_pathes = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/data/trace2_new/redraw/"
# json_path_list = glob.glob(json_pathes + "*.json")
# fix_dict = {}
# for json_path in json_path_list:
#     json_info = read_json(json_path)
#     vedio_name = os.path.basename(json_path).split(".")[0] + ".mp4"
#     shapes = json_info['shapes']
#     for shape in shapes:
#         point_1, point_2 = shape['points']
#         x_min = round(min(point_1[0], point_2[0]), 4)
#         x_max = round(max(point_1[0], point_2[0]), 4)
#         y_min = round(min(point_1[1], point_2[1]), 4)
#         y_max = round(max(point_1[1], point_2[1]), 4)
#     fix_dict[vedio_name] = [x_min, y_min, x_max, y_max]

# pk_path = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/code/OpenTAD-main/weights/trace2_instances_test_all.pickle"
# with open(pk_path, 'rb') as f:
#     obj = pickle.load(f)
# obj.update(fix_dict)

# with open('/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/code/OpenTAD-main/weights/trace2_instances_test_all_fix.pickle', 'wb') as f:
#     pickle.dump(obj, f)

pk_path = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/code/OpenTAD-main/weights/trace2_instances_test_all_fix.pickle"
with open(pk_path, 'rb') as f:
    obj = pickle.load(f)

pre_pk_path = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/code/OpenTAD-main/weights/trace1_all_fix_all_trace2_all_fix_all_last.pickle"
with open(pre_pk_path, 'rb') as f:
    obj2 = pickle.load(f)
obj2.update(obj)
with open('/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/code/OpenTAD-main/weights/trace1_all_fix_all_trace2_all_fix_all_last_trace2_test.pickle', 'wb') as f:
    pickle.dump(obj2, f)
