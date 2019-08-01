# this code is origin from jefferyHuang git repo, and editted by TingYu Yang 
# train or evaluate process for temporal (motion) stream
from __future__ import print_function
import numpy as np
import pickle
from PIL import Image
import time
import tqdm
import shutil
from random import randint
import argparse

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import *
from network import *
import dataloader


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='NDA motion stream on resnet101')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', default=1e-2, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--net-size', default=101, type=int, metavar='N', help='size of resnet (default: 101)')
parser.add_argument('--opt', default='flowNet/', type=str, metavar='PATH', help='path to jpg data folder (default: flowNet/)' )
parser.add_argument('--NDAlist', default='NDA_list/', type=str, metavar='PATH', help='path to data list folder (default: NDA_list/)' )
parser.add_argument('--numClass', default=3, type=int, metavar='N',help='number of classes')


def main():
    global arg
    arg = parser.parse_args()
    print (arg)

    #Prepare DataLoader
    data_loader = dataloader.Motion_DataLoader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=8,
                        path=arg.opt,
                        nda_list=arg.NDAlist,
                        nda_split='01',
                        in_channel=10,
                        )
    
    train_loader,test_loader, test_video = data_loader.run()
    #Model 
    model = Motion_CNN(
                        # Data Loader
                        train_loader=train_loader,
                        test_loader=test_loader,
                        # Utility
                        start_epoch=arg.start_epoch,
                        resume=arg.resume,
                        evaluate=arg.evaluate,
                        # Hyper-parameter
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        channel = 10 * 3,
                        test_video=test_video,
                        net_size = arg.net_size,
                        num_class = arg.numClass
                        )
    #Training
    model.run()

class Motion_CNN():
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader, channel,test_video,net_size, num_class):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.best_prec1=0
        self.best_train_loss1 = 100
        self.channel=channel
        self.classes = num_class
        self.test_video=test_video
        self.net_size = net_size

    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        #build model, 4 net-size is supported
        if self.net_size == 101:
            self.model = resnet101(pretrained= True, channel= self.channel, classes = self.classes).cuda()
        elif self.net_size == 50:
            self.model = resnet50(pretrained= True, channel= self.channel, classes = self.classes).cuda()
        elif self.net_size == 34:
            self.model = resnet34(pretrained= True, channel= self.channel, classes = self.classes).cuda()
        elif self.net_size == 18:
            self.model = resnet18(pretrained= True, channel= self.channel, classes = self.classes).cuda()

        #Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)

    def resume_and_evaluate(self):
        # check if needed to resume model
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.best_train_loss1 = checkpoint['best_train_loss1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                  .format(self.resume, checkpoint['epoch'], self.best_prec1))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        # check if we are in evaluate mode
        if self.evaluate:
            self.epoch=0
            prec1, val_loss = self.validate_1epoch()
            with open('record/motion/evel_motion_video_preds.pickle','wb') as f:
                pickle.dump(self.dic_video_level_preds,f)
            f.close()
            return
    
    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True
        
        # start epoch 
        for self.epoch in range(self.start_epoch, self.nb_epochs):
            train_loss = self.train_1epoch()
            prec1, val_loss = self.validate_1epoch()
            # is_best = (prec1 > self.best_prec1) or (train_loss < self.best_train_loss1 and prec1 == self.best_prec1)
            # Compare if the best model occured
            if prec1 > self.best_prec1:
                is_best = True
            elif prec1 == self.best_prec1 and train_loss < self.best_train_loss1:
                is_best = True
            else:
                is_best = False
            #lr_scheduler
            self.scheduler.step(val_loss)
            # save the best model
            if is_best:
                self.best_prec1 = prec1
                self.best_train_loss1 = train_loss
                with open('record/motion/motion_video_preds.pickle','wb') as f:
                    pickle.dump(self.dic_video_level_preds,f)
                f.close() 

            # save the checkpoint model, code mantained in util.py
            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'best_train_loss1': self.best_train_loss1,
                'optimizer' : self.optimizer.state_dict()
            },is_best,'record/motion/checkpoint.pth.tar','record/motion/model_best.pth.tar')

    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top3 = AverageMeter()
        #switch to train mode
        self.model.train()    
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader)
        for i, (data,label) in enumerate(progress):

            # measure data loading time
            data_time.update(time.time() - end)
            
            label = label.cuda(async=True)
            input_var = Variable(data).cuda()
            target_var = Variable(label).cuda()

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec3 = accuracy(output.data, label, topk=(1, 3))
            losses.update(loss.data.cpu().numpy(), data.size(0))
            top1.update(prec1.data.cpu().numpy(), data.size(0))
            top3.update(prec3.data.cpu().numpy(), data.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        info = {'Epoch':[self.epoch],
                'Batch Time':[np.round(batch_time.avg,3)],
                'Data Time':[np.round(data_time.avg,3)],
                'Loss':[np.round(losses.avg,5)],
                'Prec@1':[np.round(top1.avg,4)],
                'Prec@3':[np.round(top3.avg,4)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, 'record/motion/opf_train.csv','train')

        return np.round(losses.avg,5)

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))

        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top3 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds={}
        end = time.time()
        progress = tqdm(self.test_loader)
        for i, (keys,data,label) in enumerate(progress):
            
            #data = data.sub_(127.353346189).div_(14.971742063)
            label = label.cuda(async=True)
            with torch.no_grad():
                data_var = Variable(data).cuda(async=True)
                label_var = Variable(label).cuda(async=True)

            # compute output
            output = self.model(data_var)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            #Calculate video level prediction
            preds = output.data.cpu().numpy()
            nb_data = preds.shape[0]
            for j in range(nb_data):
                videoName = keys[j].split('-',1)[0] # ApplyMakeup_g01_c01
                if videoName not in self.dic_video_level_preds.keys():
                    self.dic_video_level_preds[videoName] = preds[j,:]
                else:
                    self.dic_video_level_preds[videoName] += preds[j,:]
                    
        #Frame to video level accuracy
        video_top1, video_top3, video_loss = self.frame2_video_level_accuracy()
        info = {'Epoch':[self.epoch],
                'Batch Time':[np.round(batch_time.avg,3)],
                'Loss':[np.round(video_loss,5)],
                'Prec@1':[np.round(video_top1,3)],
                'Prec@3':[np.round(video_top3,3)]
                }
        record_info(info, 'record/motion/opf_test.csv','test')
        return video_top1, video_loss
    # calculate frame to video accuracy
    def frame2_video_level_accuracy(self):
     
        correct = 0
        video_level_preds = np.zeros((len(self.dic_video_level_preds),self.classes))
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        ii=0
        for key in sorted(self.dic_video_level_preds.keys()):
            name = key.split('-',1)[0]

            preds = self.dic_video_level_preds[name]
            label = int(self.test_video[name])-1
                
            video_level_preds[ii,:] = preds
            video_level_labels[ii] = label
            ii+=1         
            if np.argmax(preds) == (label):
                correct+=1

        #top1 top3
        video_level_labels = torch.from_numpy(video_level_labels).long()
        video_level_preds = torch.from_numpy(video_level_preds).float()

        loss = self.criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda())    
        top1,top3 = accuracy(video_level_preds, video_level_labels, topk=(1,3))     
                            
        top1 = float(top1.numpy())
        top3 = float(top3.numpy())
            
        return top1,top3,loss.data.cpu().numpy()

if __name__=='__main__':
    main()
