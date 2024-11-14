import os

csv_path = "/home/ai-trainning-server/work/meixun/MICRO-ACTION_RECOGNITION/code/OpenTAD-main/bash/exps/multithumos/adatad/e2e_actionformer_videomae_s_768x1_160_adapter/prediction.csv"
csv_file = open(csv_path, "r")
csv_infoes = csv_file.readlines()
results_list = []
for csv_info in csv_infoes:
    csv_info = csv_info.strip()
    row = csv_info.split(",")
    results_list.append((float(float(row[2]) / 30), float(float(row[3]) / 30), float(row[5]), int(row[4]), row[1]))

print(results_list)
    