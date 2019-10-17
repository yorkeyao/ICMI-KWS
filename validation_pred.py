from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data import LipreadingDataset_val
from data import LipreadingDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import numpy as np

from sklearn.metrics import confusion_matrix    
import matplotlib.pyplot as plt    # 绘图库
import plotly.plotly as py
import plotly
#py.__version__
import plotly.graph_objs as go
plotly.tools.set_credentials_file(username='yy.fy.cn', api_key='o70mhkDG8wqIjEr6ZgVA')

import time

def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')

def convert_task3_label_to_video_label():
    task3_label_dir = os.path.expanduser("D:\Datasets\LRW-1000\MAVSR2019_val_task3_kws\kws\info")

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


class Validator():
    def __init__(self, options, mode):
    
        self.usecudnn = options["general"]["usecudnn"]

        self.batchsize = options["input"]["batchsize"]    

        self.validationdataset_ori = LipreadingDataset(options[mode]["data_root"], 
                                                options[mode]["index_root"], 
                                                options[mode]["padding"],                                                
                                                False)    
        self.validationdataset = LipreadingDataset_val(options[mode]["data_root"], 
                                                options[mode]["index_root"], 
                                                options[mode]["padding"],                                                
                                                False)
                                                
        self.tot_data = len(self.validationdataset)
        self.validationdataloader = DataLoader(
                                    self.validationdataset,
                                    batch_size=options["input"]["batchsize"],
                                    shuffle=False,
                                    num_workers=options["input"]["numworkers"],
                                    drop_last=False
                                )
        self.mode = mode

        self.video_dist_label = convert_task3_label_to_video_label()

        self.average_IOU = 0

        self.video_id_cnt = 0

        self.time0 = time.time()

        
    def __call__(self, model):
        with torch.no_grad():
            print("Starting {}...".format(self.mode))
            # count = np.zeros((len(self.validationdataset.pinyins)))
            # print (len(self.validationdataset.pinyins))
            self.pinyins = self.validationdataset_ori.get_pinyins()

            self.video_list = self.validationdataset.get_video_list()

            video_list_check_list = np.zeros(len(self.video_list))
            # assert (0)
            validator_function_pred = model.validator_function_pred()
            model.eval()
            # if(self.usecudnn):
            # net = nn.DataParallel(model).cuda()
            net = model.cuda() 
            num_samples = 0            

            all_labels = []
            all_predictions = []

            output_list = []
            # print (self.video_list)

            for i_batch, sample_batched in enumerate(self.validationdataloader):

                input = Variable(sample_batched['temporalvolume']).cuda()
                # labels = Variable(sample_batched['label']).cuda()
                length = Variable(sample_batched['length']).cuda()

                op = sample_batched['op']

                ed = sample_batched['ed']

                path = sample_batched['path']

                video_length = sample_batched['video_length']

                # if path == "004499b75f5456aa7f866b7f5252b73c":
                #     continue
                    # assert(0)
            
                # print (input[0])
                # print (np.shape (input))

                model = model.cuda()
                #print(np.shape (input))
                outputs = net(input)
                #print (np.shape (outputs))
                #assert (0)
                (values, predition, averageEnergies) = validator_function_pred(outputs, length)

                pre_video_id = None

                pre_video_length = None

                # time0 = time.time()

                for i in range(0, len(predition)):
                    if video_list_check_list[self.video_list.index(path[i])] == 1:
                        # print (path[i])
                        output_list.append (averageEnergies[i].cpu().numpy())
                        pre_video_id = path[i]
                        pre_video_length = video_length[i]

                    elif video_list_check_list[self.video_list.index(path[i])] == 0:
                        if len(output_list) != 0 and pre_video_id != None and pre_video_id in self.video_dist_label.keys():
                            
                            self.video_id_cnt = self.video_id_cnt + 1

                            

                            pred_prob = np.max(output_list, axis=0)

                            np.save ("checkpoint_prob_30/" + pre_video_id + ".npy", np.array(pred_prob))

                            # pred_word = [self.pinyins[int(i)] for i in  np.where (pred_prob > -0.5 )[0]]

                            sorted_index = sorted(enumerate(pred_prob), key=lambda x:x[1])


                            # print (sorted_index)

                            pred_word = [self.pinyins[int(i[0])] for i in  sorted_index[- int (pre_video_length/13):] ]


                            # print (pred_word)

                            # print (pred_word)

                            # assert (0)

                            print ( set( pred_word) )

                            print ( set (self.video_dist_label[pre_video_id]))

                            output = str(pre_video_id) + " " + str (pred_word) 

                            print (output)

                            f = open('output_10.txt','a')

                            f.write(output + "\n" )

                            f.close()

                            x = len( set( pred_word) & set (self.video_dist_label[pre_video_id] ))

                            y = len( set( pred_word) | set (self.video_dist_label[pre_video_id] ))
                            
                            # print (x)

                            # print (y)

                            print ("Num:", self.video_id_cnt," this_term:",  x / y )   

                            self.average_IOU  = self.average_IOU  + (x / y)

                            print ( "average:", self.average_IOU  / self.video_id_cnt, "time needed:", (time.time() - self.time0) / self.video_id_cnt )

                            # time0 = time.time()

                        # print (np.shape (averageEnergies[i].cpu().numpy()))
                        output_list = [averageEnergies[i].cpu().numpy()]
                        video_list_check_list[self.video_list.index(path[i])] = 1
                        pre_video_id = path[i]
                        pre_video_length = video_length[i]

                continue
                # print (values)

                # predition = [ self.pinyins[label] for label in predition ]

                # for i in range(0, len(predition)):
                #     if values[i] > -5:
                #         output = str(path[i]) + " " + str(op[i].cpu().numpy()) + " " + str(ed[i].cpu().numpy()) + " " + str(predition[i]) + " " + str(values[i].cpu().numpy())
                #         f.write(output + "\n" )# print (output)

                

                # _, maxindices = vector.cpu().max(1)

 
                # all_labels.extend (labels.cpu().numpy()[:,0])  
                # all_predictions.extend (maxindices.cpu().numpy())

                # argmax = (-vector.cpu().numpy()).argsort()
                # for i in range(input.size(0)):
                #     p = list(argmax[i]).index(labels[i])
                #     count[p:] += 1          
                # #print (count)          
                # num_samples += input.size(0)
                
                # if i_batch % 50 == 0:
                #     print('i_batch/tot_batch:{}/{},corret/tot:{}/{},current_acc:{}'.format(i_batch,len(self.validationdataloader),count[0],len(self.validationdataset),1.0*count[0]/num_samples))    

                # print (len(all_labels))
                # if len (all_labels) > 100:
                #     break            
                #break
        
        # all_labels = np.array(all_labels).flatten()
        # all_predictions = np.array(all_predictions).flatten()
        # print (all_labels)
        # print (all_predictions)
        assert (0)
        all_length_labels = [ pinyin_lengh[label] for label in all_labels ]
        all_length_predictions = [ pinyin_lengh[label] for label in all_predictions ]
            
        #print ()
            #all_length_labels.append 
        cm = confusion_matrix(all_length_labels, all_length_predictions, )
        
        #np.save ("confusion_matrix.npy", cm)

        #print (self.pinyins[::-1])
        #assert (0)
        #cm = np.load ("confusion_matrix.npy")
        print (cm)
        print (cm.sum(axis=1))
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  
        #cm = cm[:100, :100]
        print (cm)
        pinyin_lengh_name = [1,2,3,4,5,6,7]
        #assert (0)
        trace = go.Heatmap(z=cm,
                   x=pinyin_lengh_name,
                   y=pinyin_lengh_name)
        #print (self.pinyins)
        data=[trace]
        py.iplot(data, filename='labelled-heatmap-length')
        
        plot_confusion_matrix(cm, pinyin_lengh_name, "Confusion Matrix for Pinyins")
        plt.savefig('HAR_cm.png', format='png')
        plt.show()
        assert (0)
        
        return count/num_samples
