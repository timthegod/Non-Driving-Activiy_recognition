from matplotlib import pyplot as plt
import pickle
import argparse
import numpy as np
import torch
from utils import *
import dataloader

parser = argparse.ArgumentParser(description='NDA fusion stream on resnet101')
parser.add_argument('--jpeg', default='jpeg/', type=str, metavar='PATH', help='path to jpg data folder (default: jpeg/)' )
parser.add_argument('--NDAlist', default='NDA_list/', type=str, metavar='PATH', help='path to data list folder (default: NDA_list/)' )
parser.add_argument('--rgbPred', default='record/spatial/spatial_video_preds.pickle', type=str, metavar='PATH', help='path to rbg prediction pickle' )
parser.add_argument('--optPred', default='record/motion/motion_video_preds.pickle', type=str, metavar='PATH', help='path to optical flow prediction pickle' )
parser.add_argument('--imgCropSize', default=224, type=int, metavar='N',help='size of input to network')
parser.add_argument('--numClass', default=3, type=int, metavar='N',help='number of classes')

if __name__ == '__main__':
    global arg
    arg = parser.parse_args()
    print (arg)

    rgb_preds = arg.rgbPred
    opf_preds = arg.optPred

    with open(rgb_preds,'rb') as f:
        rgb =pickle.load(f)
    f.close()
    with open(opf_preds,'rb') as f:
        opf =pickle.load(f)
    f.close()

    dataloader = dataloader.spatial_dataloader(BATCH_SIZE=1, num_workers=1, 
                                    path=arg.jpeg, 
                                    nda_list=arg.NDAlist,
                                    nda_split='01',
                                    crop_size=arg.imgCropSize
                                    )
    train_loader,val_loader,test_video = dataloader.run()

    rgb_video_level_preds = np.zeros((len(rgb.keys()),arg.numClass))
    opt_video_level_preds = np.zeros((len(rgb.keys()),arg.numClass))
    fuse_video_level_preds = np.zeros((len(rgb.keys()),arg.numClass))
    video_level_labels = np.zeros(len(rgb.keys()))
    # correct=0
    ii=0
    for name in sorted(rgb.keys()):   
        r = rgb[name]
        o = opf[name]

        label = int(test_video[name])-1
        
        rgb_video_level_preds[ii,:] = r
        opt_video_level_preds[ii,:] = o
        fuse_video_level_preds[ii,:] = (r+o) # add prediction of r and o 
        video_level_labels[ii] = label
        ii+=1         
        #if np.argmax(r+o) == (label):
            #correct+=1

    video_level_labels = torch.from_numpy(video_level_labels).long()
    rgb_video_level_preds = torch.from_numpy(rgb_video_level_preds).float()
    opt_video_level_preds = torch.from_numpy(opt_video_level_preds).float()
    fuse_video_level_preds = torch.from_numpy(fuse_video_level_preds).float()
        
    rgb_top1,rgb_top3 = accuracy(rgb_video_level_preds, video_level_labels, topk=(1,3))
    opt_top1,opt_top3 = accuracy(opt_video_level_preds, video_level_labels, topk=(1,3))
    fuse_top1,fuse_top3 = accuracy(fuse_video_level_preds, video_level_labels, topk=(1,3))

    rgb_top1 = float(rgb_top1.numpy())
    rgb_top3 = float(rgb_top3.numpy())                           
    opt_top1 = float(opt_top1.numpy())
    opt_top3 = float(opt_top3.numpy())                           
    fuse_top1 = float(fuse_top1.numpy())
    fuse_top3 = float(fuse_top3.numpy())                           
    print ('RGB Acc: top1:', rgb_top1, 'top3:',rgb_top3)
    print ('OPT Acc: top1:', opt_top1, 'top3:',opt_top3)
    print ('FUSE Acc: top1:',fuse_top1, 'top3:', fuse_top3)
