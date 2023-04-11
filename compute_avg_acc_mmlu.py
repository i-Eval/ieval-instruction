# compute the average acc of all mmlu tasks
# the accuracy is at the last line of the log file, e.g., Accuracy: 0.1836734693877551

import os
import sys
import re

def get_avg_acc_mmlu(log_dir,model_name,task_name):
    acc_list = {}
    for filename in os.listdir(log_dir):
        if model_name in filename and task_name in filename:
            with open(os.path.join(log_dir, filename), 'r') as f:
                for line in f:
                    if line.startswith('Accuracy:'):
                        acc = float(line.split(' ')[1].strip())
                        acc_list[filename] = acc

    # print acc for each task line by line
    for k,v in acc_list.items():
        print(k,v)
    return sum(acc_list.values())/len(acc_list)

print(get_avg_acc_mmlu(sys.argv[1],sys.argv[2],sys.argv[3]))