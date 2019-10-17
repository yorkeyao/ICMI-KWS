from __future__ import print_function
from models.Dense3D import Dense3D
import torch
import toml
from training_2pic import Trainer
from validation_2pic import Validator
from models import LipRead
import torch.nn as nn
import os
import sys
from collections import OrderedDict   
import csv
import numpy as np
import json
import scipy.io as sio
from collections import defaultdict
import matplotlib.pyplot as plt
from models.resnet_2pic import *
import torch.optim as optim


def build_model_2pic(num_classes=1000):

    model_resnet, model_fc, model_classifier = Modified_Model_npic(num_pic=2, num_classes=num_classes)

    # for i in range(2):
    #     model_resnet[i].load_state_dict(torch.load('./CK_fine_tune_2pic_model/Accuracy_Best_model_resnet_{}.pkl'.format(i)))
    #     model_fc[i].load_state_dict(torch.load('./CK_fine_tune_2pic_model/Accuracy_Best_model_fc_{}.pkl'.format(i)))

    model = resnet18(pretrained=True)
    pretrained_dict = model.state_dict()
    model_dict = model_resnet[0].state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)

    for i in range(2):
        model_resnet[i].load_state_dict(model_dict)

    return model_resnet, model_fc, model_classifier


print("Loading options...")
with open(sys.argv[1], 'r') as optionsFile:
    options = toml.loads(optionsFile.read())

if(options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
    print("Running cudnn benchmark...")
    torch.backends.cudnn.benchmark = True

#os.environ['CUDA_VISIBLE_DEVICES'] = options["general"]['gpuid']
    
torch.manual_seed(options["general"]['random_seed'])

#Create the model.

model_resnet, model_fc, model_classifier = build_model_2pic()

optimizer = optim.Adam([
            {'params':model_resnet[0].parameters(), 'lr':0.00001},
            {'params':model_resnet[1].parameters(), 'lr':0.00001},
            {'params':model_fc[0].parameters(), 'lr':0.01},
            {'params':model_fc[1].parameters(), 'lr':0.01},
            {'params':model_classifier.parameters(), 'lr':0.01}
            ], weight_decay=0.00001)

for k in range(2):
    model_resnet[k] = model_resnet[k].cuda()
    model_fc[k] = model_fc[k].cuda()
model_classifier = model_classifier.cuda()
#model = LipRead(options)
'''
model = Dense3D(options)

if(options["general"]["loadpretrainedmodel"]):
    # remove paralle module
    pretrained_dict = torch.load(options["general"]["pretrainedmodelpath"])
    # load only exists weights
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
    print('matched keys:',len(pretrained_dict))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
'''

#Move the model to the GPU.
#       
if(options["general"]["usecudnn"]):        
    torch.cuda.manual_seed(options["general"]['random_seed'])
    torch.cuda.manual_seed_all(options["general"]['random_seed'])

if(options["training"]["train"]):
    trainer = Trainer(options)
if(options["validation"]["validate"]):   
    validator = Validator(options, 'validation')
if(options['test']['test']):   
    tester = Validator(options, 'test')
    
for epoch in range(options["training"]["startepoch"], options["training"]["epochs"]):
    if(options["training"]["train"]):
        trainer(model_resnet, model_fc, model_classifier, optimizer, epoch)
    if(options["validation"]["validate"]):        
        validator(model_resnet, model_fc, model_classifier)
        '''
        print('-'*21)
        print('{:<10}|{:>10}'.format('Top #', 'Accuracy'))
        for i in range(5):
            print('{:<10}|{:>10}'.format(i, result[i]))
        print('-'*21)
        '''
            
    if(options['test']['test']):
        tester(model_resnet, model_fc, model_classifier)
        '''
        print('-'*21)
        print('{:<10}|{:>10}'.format('Top #', 'Accuracy'))
        for i in range(5):
            print('{:<10}|{:>10}'.format(i, result[i]))
        print('-'*21)
        '''
    
Trainer.writer.close()