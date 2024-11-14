import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

def draw_plot(draw_data, save_path, status):
    plt.figure(figsize=(16, 12))
    sns.countplot(x='class', data=draw_data, saturation=0.75)
    counts = draw_data['class'].value_counts()
    counts_sort = counts.sort_index(ascending=True)
    plt.title('Trace 2 Distribution of veideos ' + status)
    for index, value in counts_sort.iteritems():  # 在Pandas版本0.24及以前使用
        plt.text(index, value, value, ha="center", va="bottom")
    plt.savefig(os.path.join(save_path, status + ".png"))

train_csv_path = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/data/trace2/DevelopPhase_Data/annotations/train.csv"
train_data = pd.read_csv(train_csv_path, sep=',')
save_path = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/code/OpenTAD-main/tools/prepare_data/multi-thumos"
draw_plot(train_data, save_path, "train")
exit()
# print(train_data)
train_data = train_data.rename(columns={'class': 'class_id'})
save_path = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/code/OpenTAD-main/tools/prepare_data/multi-thumos"
# draw_plot(train_data, save_path, "train")
counts = train_data['class_id'].value_counts()
counts_df = counts.to_frame()
counts_df.columns = ["counts"]
counts_df_select = counts_df[counts_df.counts < 100]
aug_repeat_dict = {}
for index, value in counts_df_select.iterrows():
    # select_id.append(index)
    # print(type(index), int(value))
    aug_repeat_dict[index] = int(math.log2(100 // int(value))) + 1
# exit()
# print(aug_repeat_dict)
# exit()
# train_data_copy = copy.deepcopy(train_data)


for id, repeat_times in aug_repeat_dict.items():
    for i in range(repeat_times):
        train_data = train_data.append(train_data[train_data.class_id == id], ignore_index=True)
# counts_last = train_data['class_id'].value_counts()
# counts_sort = counts_last.sort_index(ascending=True)
# print(counts_sort)
# print(train_data)
# exit()
train_data = train_data.rename(columns={'class_id': 'class'})
train_data.to_csv("/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/data/trace2/DevelopPhase_Data/annotations/train_aug.csv", sep=',', index=True)
