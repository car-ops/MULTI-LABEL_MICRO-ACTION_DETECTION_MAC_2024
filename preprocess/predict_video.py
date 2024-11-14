import os, pickle, tqdm, glob
import numpy as np
from ultralytics import YOLO
# Load a model
import multiprocessing
model = YOLO('/weights/yolov8x.pt')  # load an official model
pickle_path = "/weights/pickle"
video_foloder="/data/all/*.mp4"
all_vedio_list = glob.glob(video_foloder)
all_vedio_len = len(all_vedio_list)
avg_vedio_len = all_vedio_len // 64 + 1
avg_vedio_list = []
for i in range(64):
    avg_vedio_list.append(all_vedio_list[i * avg_vedio_len: (i + 1) * avg_vedio_len])


def predict(video_list, i, device):
    all_bbox_dict={}
    if len(video_list) > 0:
        for path_video in tqdm.tqdm(video_list):
            print(path_video)
            results = model.predict(source=path_video, stream=True, device=device, classes=0, iou=0.45, conf=0.5, save=False, save_txt=False, verbose=False)

            video_name = os.path.basename(path_video)
            xyxy_list = []
            for result in results:
                bboxes=result.boxes
                bboxes=bboxes.xyxy.cpu().numpy()
                if bboxes.shape[0]>0:
                    xyxy_list.append(bboxes[0])

            if xyxy_list:
                xyxy = np.stack(xyxy_list)
                bbox = np.min(xyxy, axis=0)
                bbox = np.round(bbox).tolist()  # [x1,y1,x2,y2]
                all_bbox_dict[video_name] = bbox
            else:
                all_bbox_dict[video_name]=[]
                print(video_name)
        save_pickle_path = os.path.join(pickle_path, str(i) + ".pickle")
        with open(save_pickle_path, 'wb') as fr:
            pickle.dump(all_bbox_dict, fr)


processes= 64 #进程数根据机器的性能情况填写
pool = multiprocessing.Pool(processes)

for i, vedio_list in enumerate(avg_vedio_list):
    device = i // 16
    pool.apply_async(predict, (vedio_list, i, device))

pool.close()
pool.join()

pickle_list = glob.glob(pickle_path + "/*.pickle")

data_dict = {}
for pk_path in pickle_list:
    with open(pk_path, 'rb') as f:
        obj = pickle.load(f)
        # print(obj)
        # data.append(obj)
        data_dict.update(obj)

with open('/weights/trace2_instances_all.pickle', 'wb') as f:
    # new_dict = {old_key + ".mp4": value for old_key, value in data_dict.items()}
    pickle.dump(data_dict, f)
