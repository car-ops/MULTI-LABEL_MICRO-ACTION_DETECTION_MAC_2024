import os
import pickle
import glob
import shutil

ori_pickle_path = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/code/OpenTAD-main/weights/trace1_all_fix_all_trace2_all_fix_all.pickle"
remain_pickle_path = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/code/OpenTAD-main/weights/instances_remain.pickle"
remain_train_pickle_path = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/code/OpenTAD-main/weights/instances_train_remain.pickle"
vedio_all_path = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/data/trace2/vedios_all/"
save_path = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/data/trace2/remain_train/"

vedio_all_list = [os.path.basename(vedio_path) for vedio_path in glob.glob(vedio_all_path + "*.mp4")]
# with open('/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/code/OpenTAD-main/weights/trace1_all_fix_all_trace2_all_fix_all_last.pickle', 'rb') as fr:
#     box_datas = pickle.load(fr)
# vedio_name_list = []
# for vedio_path in vedio_all_list:
#     vedio_name = os.path.basename(vedio_path)
#     if vedio_name in box_datas:
#         pass
#     else:
#         print(vedio_name)
#         vedio_name_list.append(vedio_name)
# print(len(vedio_name_list))

# exit()

with open(ori_pickle_path, 'rb') as file:
    box_datas = pickle.load(file)
with open(remain_pickle_path, 'rb') as file:
    remian_box_datas = pickle.load(file)
with open(remain_train_pickle_path, 'rb') as file:
    train_remian_box_datas = pickle.load(file)


trace_bbox_dict = {}
for vedio_name, bbox in box_datas.items():
    vedio_name = vedio_name.split("_split")[0].split("mp4")[0]
    if vedio_name in trace_bbox_dict:
        trace_bbox_dict[vedio_name].append(bbox)
    else:
        trace_bbox_dict[vedio_name] = []
        trace_bbox_dict[vedio_name].append(bbox)

trace_bbox_merge_dict = {}
for vedio_name, bboxes in trace_bbox_dict.items():
    x_min = 1000
    x_max = 0
    y_min = 1000
    y_max = 0
    for bbox in bboxes:
        if x_min > bbox[0]:
            x_min = bbox[0]
        if y_min > bbox[1]:
            y_min = bbox[1]
        if x_max < bbox[2]:
            x_max = bbox[2]
        if y_max < bbox[3]:
            y_max = bbox[3]
    trace_bbox_merge_dict[vedio_name + ".mp4"] = [round(x_min, 4), round(y_min, 4), round(x_max, 4), round(y_max, 4)]
trace_bbox_merge_dict.update(remian_box_datas)
trace_bbox_merge_dict.update(train_remian_box_datas)
trace_bbox_merge_dict.update(box_datas)
with open('/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/code/OpenTAD-main/weights/trace1_all_fix_all_trace2_all_fix_all_last.pickle', 'wb') as fr:
    pickle.dump(trace_bbox_merge_dict, fr)
