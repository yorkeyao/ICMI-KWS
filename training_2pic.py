from torch.autograd import Variable
import torch
import torch.optim as optim
from datetime import datetime, timedelta
from data import LipreadingDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn as nn
import os
import pdb
import math
import numpy as np
import torch


def timedelta_string(timedelta):
    totalSeconds = int(timedelta.total_seconds())
    hours, remainder = divmod(totalSeconds,60*60)
    minutes, seconds = divmod(remainder,60)
    return "{:0>2} hrs, {:0>2} mins, {:0>2} secs".format(hours, minutes, seconds)

def output_iteration(loss, i, time, totalitems):

    avgBatchTime = time / (i+1)
    estTime = avgBatchTime * (totalitems - i)
    
    print("Iteration: {:0>8},Elapsed Time: {},Estimated Time Remaining: {},Loss:{}".format(i, timedelta_string(time), timedelta_string(estTime),loss))


class Trainer():

    tot_iter = 0
    writer = SummaryWriter()    
    
    def __init__(self, options):
                                
        self.usecudnn = options["general"]["usecudnn"]

        self.batchsize = options["input"]["batchsize"]

        self.statsfrequency = options["training"]["statsfrequency"]       

        self.learningrate = options["training"]["learningrate"]

        self.modelType = options["training"]["learningrate"]

        self.weightdecay = options["training"]["weightdecay"]
        self.momentum = options["training"]["momentum"]
        
        self.save_prefix = options["training"]["save_prefix"]
        
        self.trainingdataset = LipreadingDataset(options["training"]["data_root"], 
                                                options["training"]["index_root"], 
                                                options["training"]["padding"], 
                                                True)
        
        self.trainingdataloader = DataLoader(
                                    self.trainingdataset,
                                    batch_size=options["input"]["batchsize"],
                                    shuffle=options["input"]["shuffle"],
                                    num_workers=options["input"]["numworkers"],
                                    drop_last=True)
        

    def learningRate(self, epoch):
        decay = math.floor((epoch - 1) / 5)
        return self.learningrate * pow(0.5, decay)

    def __call__(self, model_resnet, model_fc, model_classifier, optimizer, epoch):
        #set up the loss function.
        '''
        if(self.usecudnn):
            for k in range(2):
                model_resnet[k] = nn.DataParallel(model_resnet[k]).cuda()
                model_fc[k] = nn.DataParallel(model_fc[k]).cuda()
            model_classifier = nn.DataParallel(model_classifier).cuda()
        '''
        #transfer the model to the GPU.       
        print ("AMD Yes !")
        
        print ("Nvidia No !")

        criterion = torch.nn.CrossEntropyLoss()
        
        criterion = criterion.cuda()

        #print (model_resnet)
        #assert (0)
        startTime = datetime.now()
        print("Starting training...")
        for i_batch, sample_batched in enumerate(self.trainingdataloader):

            #if i_batch > 1000:
            #    break
            optimizer.zero_grad()
            input = Variable(sample_batched['temporalvolume'])
            labels = Variable(sample_batched['label'])
            length = Variable(sample_batched['length'])

            if(self.usecudnn):
                input = input.cuda()
                labels = labels.cuda()

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

            output = model_classifier(output_model)
            loss = criterion(output, labels.view(-1))
            
            #print (np.shape (img_init))
            #print (np.shape (img_mid))

            #print (type(input))
            
            #print (np.shape (input))
            #assert (0)
            #outputs = net(input)
            #loss = criterion(outputs, labels.squeeze(1))
            loss.backward()
            optimizer.step()
            sampleNumber = i_batch * self.batchsize

            if(sampleNumber % self.statsfrequency == 0):
                currentTime = datetime.now()
                output_iteration(loss.cpu().detach().numpy(), sampleNumber, currentTime - startTime, len(self.trainingdataset))
                Trainer.writer.add_scalar('Train Loss', loss, Trainer.tot_iter)
            Trainer.tot_iter += 1
            #break

        print("Epoch completed, saving state...")
        #torch.save(model.state_dict(), "{}_{:0>8}.pt".format(self.save_prefix, epoch))       
