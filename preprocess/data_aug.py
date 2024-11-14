import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import copy


train_csv_path = "/data/annotations/train.csv"
train_data = pd.read_csv(train_csv_path, sep=',')
train_data = train_data.rename(columns={'class': 'class_id'})

counts = train_data['class_id'].value_counts()
counts_df = counts.to_frame()
counts_df.columns = ["counts"]
counts_df_select = counts_df[counts_df.counts < 100]
aug_repeat_dict = {}
for index, value in counts_df_select.iterrows():
    aug_repeat_dict[index] = int(math.log2(100 // int(value))) + 1

train_data_copy = copy.deepcopy(train_data)


for id, repeat_times in aug_repeat_dict.items():
    for i in range(repeat_times):
        train_data = train_data.append(train_data[train_data.class_id == id], ignore_index=True)

train_data = train_data.rename(columns={'class_id': 'class'})
train_data.to_csv("/data/annotations/train_aug.csv", sep=',', index=True)
