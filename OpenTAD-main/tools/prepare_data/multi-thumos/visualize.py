import pickle
import cv2, os, glob, tqdm


mode='/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/data/trace2_new/test/'
with open('/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/code/OpenTAD-main/weights/trace2_instances_test_all.pickle', 'rb') as fr:
    temp=pickle.load(fr)

list_video=glob.glob(r"{}/*.mp4".format(mode))
list_video.sort()
for path_video in tqdm.tqdm(list_video):
    video_name = os.path.basename(path_video)
    cap = cv2.VideoCapture(path_video)
    assert cap.isOpened(), "video cannot be opened"
    ret, frame = cap.read()
    bbox = temp[video_name]  # [x1, y1, x2, y2]

    frame = cv2.rectangle(
        frame, 
        [int(bbox[0]), int(bbox[1])], 
        [int(bbox[2]), int(bbox[3])], 
        (0,255,0), 3)
    cv2.imwrite("/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/data/trace2_new/images/{}".format(video_name.replace(".mp4", ".jpg")), frame)


    # fps = cap.get(cv2.CAP_PROP_FPS)
    # timeF = int(fps)
    # i=0
    # while cap.isOpened():
    #     ret,frame = cap.read() #按帧读取视频
        
        #到视频结尾时终止
        # if ret is False :
        #     break
        #每隔timeF帧进行存储操作
        # if (n % timeF == 0) :
        #     i += 1
        #     print('保存第 %s 张图像' % i)
        #     save_image_dir = os.path.join(save_dir,'%s.jpg' % i)
        #     print('save_image_dir: ', save_image_dir)
        #     cv2.imwrite(save_image_dir,frame) #保存视频帧图像
        # n = n + 1
        # cv2.waitKey(1) #延时1ms
    cap.release()
