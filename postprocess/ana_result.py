import os
import json
import zipfile

def read_json_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


category_idx_path = "/data/annotations/category_idx.txt"
result_detection_path = "/postprocess/result_detection.json"
csv_file_path = "/postprocess/prediction.csv"
zip_file_path = "/postprocess/submission.zip"
anno_path = "/annotations/anno_all.json"

category_idx_file = open(category_idx_path, "r")
category_idx_list = [category.strip() for category in category_idx_file.readlines()]
result = read_json_from_file(result_detection_path)
anno_info = read_json_from_file(anno_path)
database = anno_info['database']

fp = open(csv_file_path, "w")
fp.writelines(",video_id,t_start,t_end,class,score\n")

total_cout = 0

detection_result = result['results']

sorted_keys = sorted(detection_result.keys())
sorted_detection_result = {k: detection_result[k] for k in sorted_keys}

for vedio_name, detection_infos in sorted_detection_result.items():
    for detection_info in detection_infos:
        segment = detection_info['segment']
        start_frame = max(abs(float(segment[0])), 0)
        end_frame = min(float(segment[1]), float(database[vedio_name]["frame"]/30.0))
        label_name = detection_info['label']
        label = int(category_idx_list.index(label_name))
        score = round(detection_info['score'], 4)
        # if score > 0:
        fp.writelines("{},{},{},{},{},{}\n".format(total_cout, vedio_name, start_frame, end_frame, label, score))
        total_cout += 1


with zipfile.ZipFile(zip_file_path, 'w') as zipf:
    zipf.write(csv_file_path, os.path.basename(csv_file_path))