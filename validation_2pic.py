from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data import LipreadingDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import numpy as np


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
    
    def validator_function(self, modelOutput, length, labels, total=None, wrong=None):

        #print (type (modelOutput))
        print (np.shape (modelOutput))
        averageEnergies = torch.sum(modelOutput.data, 1)
        for i in range(modelOutput.size(0)):
            #print(modelOutput[i,:length[i]].sum(0).shape)
            averageEnergies[i] = modelOutput[i,:length[i]].sum(0)
        print (np.shape (averageEnergies))
        maxvalues, maxindices = torch.max(averageEnergies, 1)

        count = 0

        for i in range(0, labels.squeeze(1).size(0)):
            l = int(labels.squeeze(1)[i].cpu())
            if total is not None:
                if l not in total:
                    total[l] = 1
                else:
                    total[l] += 1 
            if maxindices[i] == labels.squeeze(1)[i]:
                count += 1
            else:
                if wrong is not None:
                    if l not in wrong:
                        wrong[l] = 1
                    else:
                        wrong[l] += 1

        return (averageEnergies, count)

    def __call__(self, model_resnet, model_fc, model_classifier):
        with torch.no_grad():
            print("Starting {}...".format(self.mode))
            count = np.zeros((len(self.validationdataset.pinyins)))
            #validator_function = self.validator_function()

            for k in range(2):
                model_resnet[k].eval()
                model_fc[k].eval()
            model_classifier.eval()

            #if(self.usecudnn):
            #    net = nn.DataParallel(model).cuda()
                
            num_samples = 0      
            cnt_all = 0      
            for i_batch, sample_batched in enumerate(self.validationdataloader):

                input = Variable(sample_batched['temporalvolume']).cuda()
                labels = Variable(sample_batched['label']).cuda()
                length = Variable(sample_batched['length']).cuda()
                
                #print (length)
                output_resnet = []
                output_fc = []

                img_init = input [:,:,0,:,:]
                img_mid = input [:,:, int(np.shape (input)[2] / 2), : ,:]

                img_input = [img_init, img_mid]
                #print (np.shape (img_input[k]))
                #assert (0)
                #print (np.shape (labels.view(-1)))
                #assert (0)
                for k in range(2):
                    output_resnet.append(model_resnet[k](img_input[k]))
                    output_fc.append(model_fc[k](output_resnet[k]))

                output_model = torch.cat((output_fc[0], output_fc[1]), 1)

                outputs = model_classifier(output_model)

                '''
                print (np.shape(outputs))
                (vector, top1) = self.validator_function(outputs, length, labels)
                _, maxindices = vector.cpu().max(1)
                argmax = (-vector.cpu().numpy()).argsort()
                for i in range(input.size(0)):
                    p = list(argmax[i]).index(labels[i])
                    count[p:] += 1     
                '''               
                _, predict = torch.max(outputs, 1)
                #print (predict)
                #print (labels.view(-1))
                cnt_all += np.sum((predict.data.cpu().numpy() == labels.view(-1).data.cpu().numpy()))
                num_samples += input.size(0)
                #print (cnt_all)
                #print (num_samples)
                print('i_batch/tot_batch:{}/{},corret/tot:{}/{},current_acc:{}'.format(i_batch,len(self.validationdataloader),cnt_all,len(self.validationdataset),1.0*cnt_all/num_samples))                
                #assert (0)
                #break
        return cnt_all/num_samples
