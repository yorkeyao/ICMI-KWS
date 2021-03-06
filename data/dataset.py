from torch.utils.data import Dataset
from .preprocess import *
import os
import glob
import numpy as np
import random

class LipreadingDataset(Dataset):

    def __init__(self, data_root, index_root, padding, augment=True, pinyins=None, **kwargs):
        self.padding = padding
        self.data = []
        self.data_root = data_root
        self.padding = padding
        self.augment = augment
        self.tot_time = 0
        self.pinyins = None

        print (index_root)
        
        with open(index_root, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip().split(',') for line in lines]
            pinyins = sorted(np.unique([line[2] for line in lines]))
            self.tot_time = sum ([float(line[4]) - float(line[3]) for line in lines])
            self.data = [(line[0], int(float(line[3])*25)+1, int(float(line[4])*25)+1, pinyins.index(line[2])) for line in lines]
            self.data = list(filter(lambda data: data[2]-data[1] <= self.padding, self.data))
            self.lengths = [data[2]-data[1] for data in self.data]
            self.pinyins = pinyins
            
        print('index file:',index_root)
        print('num of pinyins:',len(pinyins))
        print('num of data:',len(self.data))
        print('max video length',np.max(self.lengths))
        print('tot time', self.tot_time)

        #return pinyins
        self.length_1 = 0
        self.length_1_cnt = 0
        self.length_2 = 0
        self.length_2_cnt = 0
        self.length_3 = 0
        self.length_3_cnt = 0
    
    
    def __len__(self):
        return len(self.data)

    def get_pinyins(self):
        return self.pinyins

    def __getitem__(self, idx):
        #load video into a tensor
        (path, op, ed, label) = self.data[idx]         
        vidframes = load_images(os.path.join(self.data_root, path), op, ed)
        # print (np.shape (vidframes))
        length = len(vidframes)

        # length = ed - op

        # print (self.pinyins[label].split(" ") )

        # print ( type (len (self.pinyins[label].split(" ") )))

        # pinyin_length = len (self.pinyins[label].split(" ") )

        # if pinyin_length == 1:
        #     # print ("length", length)
        #     self.length_1 = self.length_1 + length
        #     self.length_1_cnt = self.length_1_cnt + 1
        #     # print (self.length_1)
        #     # print (self.length_1_cnt)
        #     # print("1_average:", length)
        
        # if pinyin_length == 2:
        #     self.length_2 = self.length_2 + length
        #     self.length_2_cnt = self.length_2_cnt + 1
        #     print("2_average:", length)

        # if pinyin_length == 3:
        #     self.length_3 = self.length_3 + length
        #     self.length_3_cnt = self.length_3_cnt + 1
        #     print("3_average:", self.length_3 / self.length_3_cnt)

        # assert (0)

        # print (label)

        # print ("label:", self.pinyins[label], "length:", length)

        # assert (0)
        temporalvolume = bbc(vidframes, self.padding, self.augment)
        return {'temporalvolume': temporalvolume, 
            'label': torch.LongTensor([label]), 
            'length': length}

