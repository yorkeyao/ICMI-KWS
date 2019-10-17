import os
import numpy as np
from scipy.special import softmax
import math
import time 
from sklearn.metrics import average_precision_score

pinyins = None



with open("D:/Datasets/LRW-1000/info/trn_1000.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines = [line.strip().split(',') for line in lines]
    pinyins = sorted(np.unique([line[2] for line in lines]))

def convert_task3_label_to_video_label():
    task3_label_dir = os.path.expanduser("D:/Datasets/LRW-1000/MAVSR2019_val_task3_kws/kws/info")

    video_dict = {}

    for num, label_file in enumerate(os.listdir(task3_label_dir)):
        label_file_path = os.path.join(task3_label_dir, label_file)

        label_file_num = int(label_file.strip('.txt'))

        with open(label_file_path, 'r', encoding = 'utf-8') as lf:
            labels_info = lf.readlines()

        labels_info = [item.strip('\n') for item in labels_info]

        video_id = labels_info[0]

        # if video_id != "004499b75f5456aa7f866b7f5252b73c":
        #     continue

        labels_info.pop(0)
        labels_info.pop(-1)

        while '' in labels_info:
            labels_info.remove('')

        for label in labels_info:
            label_info = label.split(',')

            label_pinyin = label_info[1]

            if video_id in video_dict.keys():
                video_dict[video_id].append(label_pinyin)
            else:
                video_dict[video_id] = [label_pinyin]

            # print (label_pinyin, int(float(label_info[2])*25)+1 , int(float(label_info[3])*25)+1)
    return video_dict


video_dist_label = convert_task3_label_to_video_label()



output_file_path = "C:/Users/YorkeYao/Desktop/MAVSRC/sd/checkpoint_prob_1"

output_files = os.listdir(output_file_path)

print ("begin iteration")


average_hehe = 0

pred_prob_all = []

video_dist_label_all = []

# print (pinyins)

for output_file in output_files:
    pre_video_id = output_file.split('.')[0]
    pred_prob = np.load (output_file_path +"/"+ output_file)
    pred_prob_all.append (pred_prob)
    video_dist_label_all.append (pre_video_id)

print (np.shape ( np.array (pred_prob_all)))

ranking_by_top_k = True

final_result = {}
final_result_score = {}
final_result_ground_truth = {}

for i in pinyins:
    final_result[i] = []
    final_result_ground_truth[i] = []
    final_result_score[i] = []


average_IOU = 0
print ("final_weight_softmax.npy")
weight = np.load ("final_weight_softmax.npy")

pre_video_length_all = []

index_root = "D:/Datasets/LRW-1000/MAVSR2019_val_task3_kws/kws/images"

for path in video_dist_label_all:
    pre_video_length_all.append (len ( os.listdir (os.path.join(index_root, path))))

for z in range( len (video_dist_label_all) ):
    # pred_prob = np.multiply ( np.array(pred_prob_all[z]), np.array(weight) )
    pred_prob = np.array(pred_prob_all[z])
    pre_video_id = video_dist_label_all[z]

    sorted_index = sorted(enumerate(pred_prob), key=lambda x:x[1] )

    # print (sorted_index)

    # assert (0)

    pred_word = [pinyins[int(i[0])] for i in sorted_index ]
    # pred_prob_new = [pred_prob[int(i[0])] for i in sorted_index ]

    # pred_prob = np.log (softmax (pred_prob) )                           
    # pred_word = [pinyins[int(i)] for i in  np.where (pred_prob > - 10000000000000000000000000000000 )[0]]    
    # print (pred_word)
    # print (video_dist_label[pre_video_id])
    word_cnt = 0
    for word in set (pred_word):
        word_cnt = word_cnt + 1
        final_result[word ].append (pre_video_id)
        # print (pred_prob[ pinyins.index(word) ])
        # print (pinyins.index(word))
        # assert (0)
        
        if word_cnt > (1000 - int (pre_video_length_all[z]/13) ):
            ratio = (word_cnt - (1000 - int (pre_video_length_all[z]/13) ) ) / int (pre_video_length_all[z]/13)
            # print (ratio)
            final_result_score[ word ].append ( pred_prob[ pinyins.index(word) ] ) 
        else:
            final_result_score[ word ].append ( pred_prob[ pinyins.index(word) ] )
    # assert (0)
    for word in  set (video_dist_label[pre_video_id] ):
        final_result_ground_truth[ word ].append (pre_video_id)
    x = len( set( pred_word) & set (video_dist_label[pre_video_id] ))
    y = len( set( pred_word) | set (video_dist_label[pre_video_id] ))
    average_IOU  = average_IOU  + (x / y)
print (average_IOU  / len(video_dist_label_all))


# print (final_result.keys())
# print (final_result_score.keys())
# print ()
aps = np.zeros (1000)
# print (aps)
cnt = 0

f = open("submit_result.txt", "w")


video_list = sorted(os.listdir(index_root))

length_mAP = {}

for i in range (1, 8):
    length_mAP[i] = []

for key, value in final_result_score.items():
    sorted_index = sorted(enumerate(value), key=lambda x:x[1], reverse = True)

    ranked_list = [ final_result[key][i[0]] for i in sorted_index ]

    score_list = [ final_result_score[key][i[0]] for i in sorted_index  ]

    existence_list = [ ]

    output = str(pinyins.index(key) ) 

    for i in ranked_list:
        if i in final_result_ground_truth[key]:
            existence_list.append(True)
        else:
            existence_list.append (False)
        output = output + "_" + str(video_list.index(i))
   
    if np.sum (existence_list) > 0:
        aps[cnt] = average_precision_score(existence_list, score_list)
        length_mAP[len (key.split(' '))].append (aps[cnt]) 
    else:
        length_mAP[len (key.split(' '))].append(0)
    f.write(output + "\n")

    cnt = cnt + 1

np.save ("detection_ap.npy", aps)
print (np.mean (aps))
print ("##############################################")
for i in range (1, 8):
    print (np.mean (length_mAP[i]))






