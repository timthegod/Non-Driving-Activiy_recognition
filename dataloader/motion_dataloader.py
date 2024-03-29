# this code is origin from jefferyHuang git repo, and editted by TingYu Yang 
# create dataloader for temporal(motion) stream
import numpy as np
import pickle
from PIL import Image
import time
import shutil
import random
import argparse

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .split_train_test_video import *
 
class motion_dataset(Dataset):  
    def __init__(self, dic, in_channel, root_dir, mode, transform=None):
        #Generate a 16 Frame clip
        self.keys=dic.keys()
        self.values=dic.values()
        self.root_dir = root_dir
        self.transform = transform
        self.mode=mode
        self.in_channel = in_channel
        self.img_rows=224
        self.img_cols=224

    # for stacking 10 optical flow as the input for temporal(motion) stream
    def stackopf(self):
        name = self.video
        flowNet = self.root_dir + name
        
        flow = torch.FloatTensor(3*self.in_channel,self.img_rows,self.img_cols)
        i = int(self.clips_idx)
        if i == 0:
            i = 1
        for j in range(self.in_channel):
            idx = i + j
            # idx = str(idx)
            frame_idx = 'frame_'+ '{:06d}'.format(idx)
            flowNet_image = flowNet +'/' + frame_idx +'.jpg'
            
            imgFlowNet_image=(Image.open(flowNet_image))

            # split RGB channel of optical flow
            red, green, blue = imgFlowNet_image.split()
            FLOWNET_R = self.transform(red)
            FLOWNET_G = self.transform(green)
            FLOWNET_B = self.transform(blue)
            
            flow[3*(j-1),:,:] = FLOWNET_R
            flow[3*(j-1)+1,:,:] = FLOWNET_G
            flow[3*(j-1)+2,:,:] = FLOWNET_B
  
            imgFlowNet_image.close()  
        return flow

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        #print ('mode:',self.mode,'calling Dataset:__getitem__ @ idx=%d'%idx)
        
        if self.mode == 'train':
            self.video, nb_clips = list(self.keys)[idx].split('-')
            self.clips_idx = random.randint(0,int(nb_clips) - 1)  # randint(1,int(nb_clips)) change to randint(0,int(nb_clips) - 1) 
        elif self.mode == 'val':
            self.video,self.clips_idx = list(self.keys)[idx].split('-')
        else:
            raise ValueError('There are only train and val mode')

        label = list(self.values)[idx]
        label = int(label)-1 
        data = self.stackopf()

        if self.mode == 'train':
            sample = (data,label)
        elif self.mode == 'val':
            sample = (self.video,data,label)
        else:
            raise ValueError('There are only train and val mode')
        return sample



class Motion_DataLoader():
    def __init__(self, BATCH_SIZE, num_workers, in_channel,  path, nda_list, nda_split):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers = num_workers
        self.frame_count={}
        self.in_channel = in_channel
        self.data_path=path
        # split the training and testing videos
        splitter = NDA_splitter(path=nda_list,split=nda_split)
        self.train_video, self.test_video = splitter.split_video()
    

    def load_frame_count(self):
        #print '==> Loading frame number of each video'
        with open('dataloader/dic/frame_count.pickle','rb') as file:
            dic_frame = pickle.load(file)
        file.close()

        for line in dic_frame :
            videoname = line.split('.')[0]
            self.frame_count[videoname]=dic_frame[line] 

    def run(self):
        # load frame count by frame count pickle
        self.load_frame_count()
        # get the training dictionary
        self.get_training_dic()
        # get the 19 sample frames for validation
        self.val_sample19()
        train_loader = self.train()
        val_loader = self.val()

        return train_loader, val_loader, self.test_video
            
    def val_sample19(self):
        self.dic_test_idx = {}
        for video in self.test_video:
            # n,g = video.split('_',1)

            sampling_interval = int((self.frame_count[video]-10)/19) #-10+1 to -10
            for index in range(19):
                clip_idx = index*sampling_interval
                key = video + '-' + str(clip_idx+1)
                self.dic_test_idx[key] = self.test_video[video]
             
    def get_training_dic(self):
        self.dic_video_train={}
        for video in self.train_video:
            
            nb_clips = self.frame_count[video]-10 #-10+1 to -10
            key = video +'-' + str(nb_clips)
            self.dic_video_train[key] = self.train_video[video] 
                            
    def train(self):
        training_set = motion_dataset(dic=self.dic_video_train, in_channel=self.in_channel, root_dir=self.data_path,
            mode='train',
            transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            ]))
        print ('==> Training data :',len(training_set),' videos',training_set[1][0].size())

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
            )

        return train_loader

    def val(self):
        validation_set = motion_dataset(dic= self.dic_test_idx, in_channel=self.in_channel, root_dir=self.data_path ,
            mode ='val',
            transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            ]))
        print ('==> Validation data :',len(validation_set),' frames',validation_set[1][1].size())
        #print validation_set[1]

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)

        return val_loader
