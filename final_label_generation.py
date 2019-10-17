import os
import numpy as np

index_root = "D:/Datasets/LRW-1000/MAVSR2019_val_task3_kws/kws/images"

video_list = sorted(os.listdir(index_root))

index_root = "D:/Datasets/LRW-1000/info/trn_1000.txt"

with open(index_root, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines = [line.strip().split(',') for line in lines]
    pinyins = sorted(np.unique([line[2] for line in lines] ))

output_file = "output_1.txt"

# print (pinyins)

# assert(0)

final_result = {}

for i in range(1, 1001):
    final_result[i] = []


with open(output_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    # lines = [line.strip().split(' ') for line in lines]
    for line in lines:
        # print (line )
        path = line.split(' ')[0]
        # print (line.split('[')[1])
        words = line.split('[')[1].split(', ')
        # assert (0)
        for word in words:
            # print (word.strip('[').strip(']').strip("'"))
            # print (pinyins.index(word.strip('[').strip(']').strip("'")  ))
            # print (video_list.index (path))
            final_result[pinyins.index( word.strip().strip(']').strip("'") ) + 1 ].append(video_list.index (path) + 1 )
            #assert (0)

f = open("submit_result.txt", "w")

for key, value in final_result.items():
    output = str(key) 
    for result in value:
        output = output + "_" + str(result)
    f.write(output + "\n")

