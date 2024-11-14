import os
import glob
import json
import cv2

video_path_root = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/data/trace2/vedios/"
video_path_all = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/data/trace2/vedios_all/"
train_csv_path = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/data/trace2/DevelopPhase_Data/annotations/train_aug.csv"
val_csv_path = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/data/trace2/DevelopPhase_Data/annotations/val.csv"
test_csv_path = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/data/trace2_new/annotations/test.csv"
label_name_txt = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/data/videos/annotations/label_name.txt"
tarce_one_train_txt = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/data/videos/annotations/train_list_videos_aug.txt"
tarce_one_val_txt = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/data/videos/annotations/val_list_videos.txt"

tarce_one_train_file = open(tarce_one_train_txt, "r")
tarce_one_train_info_list = tarce_one_train_file.readlines()
tarce_one_val_file = open(tarce_one_val_txt, "r")
tarce_one_val_info_list = tarce_one_val_file.readlines()
tarce_one_info_dict = {}
for tarce_one_train_info in tarce_one_train_info_list+tarce_one_val_info_list:
    tarce_one_train_info = tarce_one_train_info.strip()
    tarce_one_info_dict[tarce_one_train_info.split(" ")[0]] = tarce_one_train_info.split(" ")[1]

vedio_path_list = glob.glob(video_path_root + "/*/*.mp4")
vedio_name_list = [os.path.basename(vedio_path).split(".")[0] for vedio_path in vedio_path_list]

label_name_file = open(label_name_txt, "r") 
label_name_list = [label_info.strip().split("\t")[1] for label_info in label_name_file.readlines()]

train_csv_file = open(train_csv_path, "r")
train_csv_info = train_csv_file.readlines()

val_csv_file = open(val_csv_path, "r")
val_csv_info = val_csv_file.readlines()

test_csv_file = open(test_csv_path, "r")
test_csv_info = test_csv_file.readlines()

multi_dict = {}
database_dict = {}

for i, train_info in enumerate(train_csv_info):
    if i == 0:
        pass
    else:
        _, _, video_id, start_frame, end_frame, class_label, total_frames = train_info.split(",")
        fps = 30
        if (video_id in vedio_name_list):
            video_index = vedio_name_list.index(video_id)
            video_path = vedio_path_list[video_index] 
        elif (video_id.lower() in vedio_name_list):
            video_index = vedio_name_list.index(video_id.lower())
            video_id = video_id.lower()
            video_path = vedio_path_list[video_index]
        elif ((video_id.upper() in vedio_name_list)):
            video_index = vedio_name_list.index(video_id.upper())
            video_id = video_id.upper()
            video_path = vedio_path_list[video_index]
        if video_id in database_dict:
            clip_info = {}
            clip_info["label"] = label_name_list[int(class_label)]
            clip_info["segment"] = [round(float(start_frame) / fps, 2), round(float(end_frame) / fps, 2)]
            database_dict[video_id]["annotations"].append(clip_info)
        else:
            database_dict[video_id] = {}
            database_dict[video_id]["duration"] = round(float(total_frames) / fps, 2)
            database_dict[video_id]["frame"] = int(total_frames)
            database_dict[video_id]["subset"] = "training"
            database_dict[video_id]["annotations"] = []
            clip_info = {}
            clip_info["label"] = label_name_list[int(class_label)]
            clip_info["segment"] = [round(float(start_frame) / fps, 2), round(float(end_frame) / fps, 2)]
            database_dict[video_id]["annotations"].append(clip_info)

for i, val_info in enumerate(val_csv_info):
    if i == 0:
        pass
    else:
        _, video_id, start_frame, end_frame, class_label, total_frames = val_info.split(",")
        fps = 30
        if (video_id in vedio_name_list):
            video_index = vedio_name_list.index(video_id)
            video_path = vedio_path_list[video_index] 
        elif (video_id.lower() in vedio_name_list):
            video_index = vedio_name_list.index(video_id.lower())
            video_id = video_id.lower()
            video_path = vedio_path_list[video_index]
        elif ((video_id.upper() in vedio_name_list)):
            video_index = vedio_name_list.index(video_id.upper())
            video_id = video_id.upper()
            video_path = vedio_path_list[video_index]
        if video_id in database_dict:
            clip_info = {}
            clip_info["label"] = label_name_list[int(class_label)]
            clip_info["segment"] = [round(float(start_frame) / fps, 2), round(float(end_frame) / fps, 2)]
            database_dict[video_id]["annotations"].append(clip_info)
        else:
            database_dict[video_id] = {}
            database_dict[video_id]["duration"] = round(float(total_frames) / fps, 2)
            database_dict[video_id]["frame"] = int(total_frames)
            database_dict[video_id]["subset"] = "training"
            database_dict[video_id]["annotations"] = []
            clip_info = {}
            clip_info["label"] = label_name_list[int(class_label)]
            clip_info["segment"] = [round(float(start_frame) / fps, 2), round(float(end_frame) / fps, 2)]
            database_dict[video_id]["annotations"].append(clip_info)


for i, test_info in enumerate(test_csv_info):
    if i == 0:
        pass
    else:
        _, video_id, start_frame, end_frame, class_label, total_frames, fps = test_info.split(",")
        fps = int(float(fps.strip()))
        if (video_id in vedio_name_list):
            video_index = vedio_name_list.index(video_id)
            video_path = vedio_path_list[video_index] 
        elif (video_id.lower() in vedio_name_list):
            video_index = vedio_name_list.index(video_id.lower())
            video_id = video_id.lower()
            video_path = vedio_path_list[video_index]
        elif ((video_id.upper() in vedio_name_list)):
            video_index = vedio_name_list.index(video_id.upper())
            video_id = video_id.upper()
            video_path = vedio_path_list[video_index]
        if video_id in database_dict:
            clip_info = {}
            clip_info["label"] = label_name_list[int(class_label)]
            clip_info["segment"] = [round(float(start_frame) / fps, 2), round(float(end_frame) / fps, 2)]
            database_dict[video_id]["annotations"].append(clip_info)
        else:
            database_dict[video_id] = {}
            database_dict[video_id]["duration"] = round(float(total_frames) / fps, 2)
            database_dict[video_id]["frame"] = int(total_frames)
            database_dict[video_id]["subset"] = "validation"
            database_dict[video_id]["annotations"] = []
            clip_info = {}
            clip_info["label"] = label_name_list[int(class_label)]
            clip_info["segment"] = [round(float(start_frame) / fps, 2), round(float(end_frame) / fps, 2)]
            database_dict[video_id]["annotations"].append(clip_info)


for vedio_name, class_label in tarce_one_info_dict.items():
    video_id = vedio_name.split(".")[0]
    video_path = os.path.join(video_path_all, vedio_name)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # duration_frame = cap.get(cv2.CAP_PROP_FORMAT)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # total_seconds = duration_frame / fps
    if video_id in database_dict:
        clip_info = {}
        clip_info["label"] = label_name_list[int(class_label)]
        clip_info["segment"] = [0, round(float(total_frames) / 30, 2)]
        database_dict[video_id]["annotations"].append(clip_info)
    else:
        database_dict[video_id] = {}
        database_dict[video_id]["duration"] = round(float(total_frames) / 30, 2)
        database_dict[video_id]["frame"] = int(total_frames)
        database_dict[video_id]["subset"] = "training"
        database_dict[video_id]["annotations"] = []
        clip_info = {}
        clip_info["label"] = label_name_list[int(class_label)]
        clip_info["segment"] = [0, round(float(total_frames) / 30, 2)]
        database_dict[video_id]["annotations"].append(clip_info)

multi_dict["database"] = database_dict
multi_dict["taxonomy"] = {}
multi_dict["version"] = {}

with open('multithumos_anno_all.json', 'w') as f:
    json.dump(multi_dict, f)

