from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
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


class Validator():
    def __init__(self, options, mode):
    
        self.usecudnn = options["general"]["usecudnn"]

        self.batchsize = options["input"]["batchsize"]        
        self.validationdataset = LipreadingDataset(options[mode]["data_root"], 
                                                options[mode]["index_root"], 
                                                options[mode]["padding"],                                                
                                                False)
                                                
        self.tot_data = len(self.validationdataset)
        self.validationdataloader = DataLoader(
                                    self.validationdataset,
                                    batch_size=options["input"]["batchsize"],
                                    shuffle=options["input"]["shuffle"],
                                    num_workers=options["input"]["numworkers"],
                                    drop_last=False
                                )
        self.mode = mode
        
    def __call__(self, model):
        with torch.no_grad():
            print("Starting {}...".format(self.mode))
            count = np.zeros((len(self.validationdataset.pinyins)))
            # print (len(self.validationdataset.pinyins))
            # assert (0)
            validator_function = model.validator_function()
            model.eval()
            # if(self.usecudnn):
            #     net = nn.DataParallel(model).cuda()
            net = model.cuda() 
            num_samples = 0            

            self.pinyins = self.validationdataset.get_pinyins()

            #print (self.pinyins)

            cnt = 0
            pinyin_lengh = []
            for pinyin in self.pinyins:
                cnt = cnt + 1
                #print (pinyin)
                pinyin_lengh.append(len(pinyin.split(" ")))
                #print (len(pinyin.split(" ")))
                # if cnt > 5:
                #     assert (0)
            print (max (pinyin_lengh))
            #assert (0)

            all_labels = []
            all_predictions = []

            for i_batch, sample_batched in enumerate(self.validationdataloader):

                input = Variable(sample_batched['temporalvolume']).cuda()
                labels = Variable(sample_batched['label']).cuda()
                length = Variable(sample_batched['length']).cuda()
            


                model = model.cuda()
                #print(np.shape (input))
                outputs = net(input)
                #print (np.shape (outputs))
                #assert (0)
                (vector, top1) = validator_function(outputs, length, labels)

                


                _, maxindices = vector.cpu().max(1)

 
                all_labels.extend (labels.cpu().numpy()[:,0])  
                all_predictions.extend (maxindices.cpu().numpy())

                argmax = (-vector.cpu().numpy()).argsort()
                for i in range(input.size(0)):
                    p = list(argmax[i]).index(labels[i])
                    count[p:] += 1          
                #print (count)          
                num_samples += input.size(0)
                
                if i_batch % 50 == 0:
                    print('i_batch/tot_batch:{}/{},corret/tot:{}/{},current_acc:{}'.format(i_batch,len(self.validationdataloader),count[0],len(self.validationdataset),1.0*count[0]/num_samples))    

                # print (len(all_labels))
                # if len (all_labels) > 100:
                #     break            
                #break
        # all_labels = np.array(all_labels).flatten()
        # all_predictions = np.array(all_predictions).flatten()
        # print (all_labels)
        # print (all_predictions)
        
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
