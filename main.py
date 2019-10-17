import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#from __future__ import print_function
from models.Dense3D import Dense3D
import torch
import toml
from training import Trainer
from validation_pred import Validator
from models import LipRead
import torch.nn as nn

import sys
from collections import OrderedDict   
import csv
import numpy as np
import json
import scipy.io as sio
from collections import defaultdict
import matplotlib.pyplot as plt

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':

    print("Loading options...")
    with open(sys.argv[1], 'r') as optionsFile:
        options = toml.loads(optionsFile.read())

    if(options["general"]["usecudnnbenchmark"] and options["general"]["usecudnn"]):
        print("Running cudnn benchmark...")
        torch.backends.cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = options["general"]['gpuid']
        
    torch.manual_seed(options["general"]['random_seed'])

    #Create the model.

    model = LipRead(options)
    '''
    model = Dense3D(options)
    '''
    if(options["general"]["loadpretrainedmodel"]):
        # remove paralle module
        pretrained_dict = torch.load(options["general"]["pretrainedmodelpath"])
        # load only exists weights
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        print('matched keys:',len(pretrained_dict))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print ("load successful")

    #Move the model to the GPU.
    criterion = model.loss()        
    if(options["general"]["usecudnn"]):        
        torch.cuda.manual_seed(options["general"]['random_seed'])
        torch.cuda.manual_seed_all(options["general"]['random_seed'])

    if(options["training"]["train"]):
        trainer = Trainer(options)
    if(options["validation"]["validate"]):   
        validator = Validator(options, 'validation')
    if(options['test']['test']):   
        tester = Validator(options, 'test')

    val_acc = 0
    test_acc = 0    

    for epoch in range(options["training"]["startepoch"], options["training"]["epochs"]):
        if(options["training"]["train"]):
            print ("Begin training for one epoch")
            trainer(model, epoch)
        change = False
        if(options["validation"]["validate"]):        
            result = validator(model)
            print('-'*21)
            print('{:<10}|{:>10}'.format('Top #', 'Accuracy'))
            for i in range(5):
                print('{:<10}|{:>10}'.format(i, result[i]))
            if result[0] > val_acc:
                change = True
                val_acc = result[0]
            print('-'*21)
                
        if(options['test']['test']):
            result = tester(model)
            print('-'*21)
            print('{:<10}|{:>10}'.format('Top #', 'Accuracy'))
            for i in range(5):
                print('{:<10}|{:>10}'.format(i, result[i]))
            if change:
                test_acc = result[0]
            print('-'*21)
        print ("Epoch:", epoch, "Best val acc:", val_acc, "and its test acc:", test_acc)
        
    Trainer.writer.close()