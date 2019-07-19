from __future__ import print_function
import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.nn import functional as F #for CAM
from torch import topk

import matplotlib.pyplot as plt
import skimage.transform

import dataloader
from utils import *
from network import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='NDA spatial stream on resnet')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--attentionMap', dest='attentionMap', action='store_true', help='store attention map or not')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--jpeg', default='jpeg/', type=str, metavar='PATH', help='path to jpg data folder (default: jpeg/)' )
parser.add_argument('--NDAlist', default='NDA_list/', type=str, metavar='PATH', help='path to data list folder (default: NDA_list/)' )
parser.add_argument('--net-size', default=50, type=int, metavar='N', help='size of resnet (default: 50)')
parser.add_argument('--imgCropSize', default=224, type=int, metavar='N',help='size of input to network')
parser.add_argument('--numClass', default=3, type=int, metavar='N',help='number of classes')


def main():
    global arg
    arg = parser.parse_args()
    print (arg)

    #Prepare DataLoader
    data_loader = dataloader.spatial_dataloader(
                        BATCH_SIZE=arg.batch_size,
                        num_workers=8,
                        path=arg.jpeg,
                        nda_list =arg.NDAlist,
                        nda_split ='01',
                        crop_size = arg.imgCropSize
                        )
    
    train_loader, test_loader, test_video = data_loader.run()
    #Model 
    model = Spatial_CNN(
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        resume=arg.resume,
                        start_epoch=arg.start_epoch,
                        evaluate=arg.evaluate,
                        attMap=arg.attentionMap,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        test_video=test_video,
                        net_size = arg.net_size,
                        num_class = arg.numClass
    )
    #Training
    model.run()

class Spatial_CNN():
    def __init__(self, nb_epochs, lr, batch_size, resume, start_epoch, attMap, evaluate, train_loader, test_loader, test_video, net_size, num_class):
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.attMap=attMap
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.best_prec1=0
        self.best_train_loss1 = 100
        self.test_video=test_video
        self.classes = num_class
        self.net_size = net_size
        self.cam = {}

    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        #build model
        if self.net_size == 101:
            self.model = resnet101(pretrained= True, channel=3, classes=self.classes).cuda()
        elif self.net_size == 50:
            self.model = resnet50(pretrained= True, channel=3, classes=self.classes).cuda()
        elif self.net_size == 34:
            self.model = resnet34(pretrained= True, channel=3, classes=self.classes).cuda()
        elif self.net_size == 18:
            self.model = resnet18(pretrained= True, channel=3, classes=self.classes).cuda()
        #Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)
    
    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.best_train_loss1 = checkpoint['best_train_loss1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {}) (best_train_loss1 {})"
                  .format(self.resume, checkpoint['epoch'], self.best_prec1, self.best_train_loss1))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            self.epoch = 0
            prec1, val_loss = self.validate_1epoch()
            with open('record/spatial/evel_spatial_video_preds.pickle','wb') as f:
                pickle.dump(self.dic_video_level_preds,f)
            f.close()
            for key in self.cam:
                over, img = self.cam[key]
                self.vis_cam(key, over, img)
            return

    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True
        
        for self.epoch in range(self.start_epoch, self.nb_epochs):
            train_loss = self.train_1epoch()
            prec1, val_loss = self.validate_1epoch()
            
            if prec1 > self.best_prec1:
                is_best = True
            elif prec1 == self.best_prec1 and train_loss < self.best_train_loss1:
                is_best = True
            else:
                is_best = False
            #lr_scheduler
            self.scheduler.step(val_loss)
            # save model
            if is_best:
                self.best_prec1 = prec1
                self.best_train_loss1 = train_loss
                with open('record/spatial/spatial_video_preds.pickle','wb') as f:
                    pickle.dump(self.dic_video_level_preds,f)
                f.close()
            
            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'best_train_loss1': self.best_train_loss1,
                'optimizer' : self.optimizer.state_dict()
            },is_best,'record/spatial/checkpoint.pth.tar','record/spatial/model_best.pth.tar')

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
        for i, (data_dict,label) in enumerate(progress):

    
            # measure data loading time
            data_time.update(time.time() - end)
            
            label = label.cuda(async=True)
            target_var = Variable(label).cuda()

            # compute output
            output = Variable(torch.zeros(len(data_dict['img1']),self.classes).float()).cuda()
            for i in range(len(data_dict)):
                key = 'img'+str(i)
                data = data_dict[key]
                input_var = Variable(data).cuda()
                output += self.model(input_var)

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
        record_info(info, 'record/spatial/rgb_train.csv','train')

        return np.round(losses.avg,5)

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        video_loss = AverageMeter()
        video_top1 = AverageMeter()
        video_top3 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()

        if self.evaluate and self.attMap:
            class SaveFeatures():
                features=None
                def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
                def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
                def remove(self): self.hook.remove()
            final_layer = self.model._modules.get('layer4')
            activated_features = SaveFeatures(final_layer)

            def getCAM(feature_conv, weight_fc, class_idx):
                # weight_fc.shape (11, 2048)
                cam_img_batch = []
                XX, nc, h, w = feature_conv.shape #XX: batch size (32, 2048, 7, 7)
                # print(feature_conv.shape)
                # print(weight_fc.shape)
                for features_i, index in enumerate(class_idx):
                    # index, ex: [5]
                    feature_conv_class = feature_conv[features_i]
                    # print(feature_conv_class.shape)
                    cam = weight_fc[index[0]].dot(feature_conv_class.reshape((nc, h*w)))
                    cam = cam.reshape(h, w)
                    cam = cam - np.min(cam)
                    cam_img = cam / np.max(cam)
                    cam_img_batch.append([cam_img])
                return cam_img_batch

            self.dic_video_level_preds={}
            end = time.time()
            progress = tqdm(self.test_loader)
            for batch_i, (keys,ind , data,label) in enumerate(progress):
                # keys: list of class ex: ('mobileBrowse_g01_c02', 'mobileBrowse_g01_c02', 'mobileBrowse_g01_c02'...)
                # data.shape: (batch size, 3, 244, 244)
                label = label.cuda(async=True)
                data_var = Variable(data, requires_grad=True).cuda(async=True)
                label_var = Variable(label).cuda(async=True)

                # compute output
                output = self.model(data_var)

                #for CAM
                pred_probabilities = F.softmax(output, dim = 1)
                # pred_probabilities = output.data.cpu().numpy()
                activated_features.remove()

                weight_softmax_params = list(self.model._modules.get('fc_custom').parameters())
                weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())

                class_idx = topk(pred_probabilities,1)[1].cpu().data.int().numpy() #shape: (batch size, 1)
                overlay = getCAM(activated_features.features, weight_softmax, class_idx ) #len overlay = batch-size
                # print(len(overlay))

                # ch, w, h = data[class_idx[0][0]].cpu().data.numpy().shape
                # print(data.cpu().data.numpy().shape) 
                # print(keys)
                # print(ind.cpu().data.numpy())
                
                for ind_save, image in enumerate(overlay):
                    out_img = data[ind_save].cpu().data.numpy().transpose(1,2,0) # transpose from (3,244,244) to (244,244,3)

                    out_img = out_img - min(out_img.flatten())
                    out_img = out_img / max(out_img.flatten())

                    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
                    ax.imshow(out_img)
                    ax.imshow(skimage.transform.resize(image[0], out_img.shape[0:2]), alpha=0.5, cmap='jet')
                    root = os.path.join('attention_map', keys[ind_save])
                    # print(keys[ind_save],int(ind.cpu().data.numpy()[ind_save]))
                    if not os.path.isdir(root):
                        os.mkdir(root)
                    fig.savefig(os.path.join(root, '{}_{:02d}.jpg'.format(keys[ind_save], int(ind.cpu().data.numpy()[ind_save]))))   # save the figure to file
                    plt.close(fig) 

                # self.cam[keys] = (overlay, data[class_idx])
                
                # self.vis_cam(keys, overlay, data[class_idx].cpu().data.numpy().reshape(w, h, ch))
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                #Calculate video level prediction
                preds = output.data.cpu().numpy()
                # print(self.dic_video_level_preds)
                nb_data = preds.shape[0]
                for j in range(nb_data):
                    videoName = keys[j].split('/',1)[0]
                    if videoName not in self.dic_video_level_preds.keys():
                        self.dic_video_level_preds[videoName] = preds[j,:]
                    else:
                        self.dic_video_level_preds[videoName] += preds[j,:]
        else:
            self.dic_video_level_preds={}
            end = time.time()
            progress = tqdm(self.test_loader)
            for i, (keys, ind, data,label) in enumerate(progress):
                
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
                    videoName = keys[j].split('/',1)[0]
                    if videoName not in self.dic_video_level_preds.keys():
                        self.dic_video_level_preds[videoName] = preds[j,:]
                    else:
                        self.dic_video_level_preds[videoName] += preds[j,:]

        video_top1, video_top3, video_loss = self.frame2_video_level_accuracy()
            

        info = {'Epoch':[self.epoch],
                'Batch Time':[np.round(batch_time.avg,3)],
                'Loss':[np.round(video_loss,5)],
                'Prec@1':[np.round(video_top1,3)],
                'Prec@3':[np.round(video_top3,3)]}
        record_info(info, 'record/spatial/rgb_test.csv','test')
        return video_top1, video_loss

    def frame2_video_level_accuracy(self):
            
        correct = 0
        video_level_preds = np.zeros((len(self.dic_video_level_preds), self.classes))
        video_level_labels = np.zeros(len(self.dic_video_level_preds))
        ii=0
        for name in sorted(self.dic_video_level_preds.keys()):
        
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
            
        top1,top3 = accuracy(video_level_preds, video_level_labels, topk=(1,3))
        loss = self.criterion(Variable(video_level_preds).cuda(), Variable(video_level_labels).cuda())     
                            
        top1 = float(top1.numpy())
        top3 = float(top3.numpy())
            
        #print(' * Video level Prec@1 {top1:.3f}, Video level Prec@5 {top3:.3f}'.format(top1=top1, top3=top3))
        return top1,top3,loss.data.cpu().numpy()

    def vis_cam(self, name, over, img):
        print(name)
        imshow((img * 255).astype(np.uint8))
        imshow(skimage.transform.resize(over[0], over.shape[1:3]), alpha=0.5, cmap='jet')






if __name__=='__main__':
    main()
