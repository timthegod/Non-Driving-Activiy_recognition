import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from .split_train_test_video import *
from skimage import io, color, exposure

class spatial_dataset(Dataset):  
    def __init__(self, dic, root_dir, mode, transform=None):
 
        self.keys = dic.keys()
        self.values=dic.values()
        self.root_dir = root_dir
        self.mode =mode
        self.transform = transform

    def __len__(self):
        return len(self.keys)

    def load_ucf_image(self,video_name, index):
       # if video_name.split('_')[0] == 'HandstandPushups':
       #     n,g = video_name.split('_',1)
       #     name = 'HandStandPushups_'+g
       #     path = self.root_dir + 'v_' + name + '/frame'
       #    #path = self.root_dir + 'HandstandPushups'+'/separated_images/v_'+name+'/v_'+name+'_'
       # else:
        path = self.root_dir + video_name + '/frame_' 
            #path = self.root_dir + video_name.split('_')[0]+'/separated_images/v_'+video_name+'/v_'+video_name+'_'
        a = str(index)
        b = a.zfill(6)
        img = Image.open(path +str(b)+'.jpg')
        transformed_img = self.transform(img)
        img.close()

        return transformed_img

    def __getitem__(self, idx):

        if self.mode == 'train':
            video_name, nb_clips = list(self.keys)[idx].split(' ')
            nb_clips = int(nb_clips)
            clips = []
            clips.append(random.randint(1, int(nb_clips/3)))
            clips.append(random.randint(int(nb_clips/3), int(nb_clips*2/3)))
            clips.append(random.randint(int(nb_clips*2/3), nb_clips+1))
            
        elif self.mode == 'val':
            video_name, index = list(self.keys)[idx].split(' ')
            index =abs(int(index))
        else:
            raise ValueError('There are only train and val mode')

        label = list(self.values)[idx]
        label = int(label)-1
        
        if self.mode=='train':
            data ={}
            for i in range(len(clips)):
                key = 'img'+str(i)
                index = clips[i]
                data[key] = self.load_ucf_image(video_name, index)
                    
            sample = (data, label)
        elif self.mode=='val':
            data = self.load_ucf_image(video_name,index)
            sample = (video_name, index, data, label)
        else:
            raise ValueError('There are only train and val mode')
           
        return sample

class spatial_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, path, nda_list, nda_split, crop_size):

        self.BATCH_SIZE=BATCH_SIZE
        self.num_workers=num_workers
        self.data_path=path
        self.crop_size = crop_size
        self.frame_count ={}
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
        self.load_frame_count()
        self.get_training_dic()
        self.val_sample20()
        train_loader = self.train()
        val_loader = self.validate()

        return train_loader, val_loader, self.test_video

    def get_training_dic(self):
        print( '==> Generate frame numbers of each training video')
        self.dic_training={}
        # 1 frame per video
        for video in self.train_video:
            # 'video' here is key ex: Book_g01_c01
            #print videoname
            nb_frame = self.frame_count[video]-10+1
            # 'key' here is ex: 'Book_g01_c01 100-10+1' 
            # combine video name and frame count
            key = video+' '+ str(nb_frame)
            # value for dic_training is the class label
            self.dic_training[key] = self.train_video[video]
                    
    def val_sample20(self):
        print ('==> sampling testing frames')
        self.dic_testing={}
        # 20 frame per video
        for video in self.test_video:
            nb_frame = self.frame_count[video]-10+1
            interval = int(nb_frame/19)
            for i in range(19):
                frame = i*interval
                key = video+ ' '+str(frame+1)
                self.dic_testing[key] = self.test_video[video]      

    def train(self):
        training_set = spatial_dataset(dic=self.dic_training, root_dir=self.data_path, mode='train', transform = transforms.Compose([
                transforms.Resize(256), 
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        print ('==> Training data :',len(training_set),'frames')
        print( training_set[1][0]['img1'].size())

        train_loader = DataLoader(
            dataset=training_set, 
            batch_size=self.BATCH_SIZE,
            shuffle=True,
            num_workers=self.num_workers)
        return train_loader

    def validate(self):
        validation_set = spatial_dataset(dic=self.dic_testing, root_dir=self.data_path, mode='val', transform = transforms.Compose([
                transforms.Resize([self.crop_size,self.crop_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ]))
        
        print ('==> Validation data :',len(validation_set),'frames')
        print (validation_set[1][2].size())

        val_loader = DataLoader(
            dataset=validation_set, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.num_workers)
        return val_loader





if __name__ == '__main__':
    
    dataloader = spatial_dataloader(BATCH_SIZE=1, num_workers=1, 
                                path='/home/ubuntu/data/UCF101/spatial_no_sampled/', 
                                nda_list='/home/ubuntu/cvlab/pytorch/ucf101_two_stream/github/nda_list/',
                                nda_split='01')
    train_loader,val_loader,test_video = dataloader.run()
