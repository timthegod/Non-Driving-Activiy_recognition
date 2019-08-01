# this code is origin from jefferyHuang git repo, and editted by TingYu Yang 
# create dictionary for data loading
import os, pickle


class NDA_splitter():
    def __init__(self, path, split):
        self.path = path
        self.split = split

    def get_action_index(self):
        self.action_label={}
        with open(self.path+'classInd.txt') as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        for line in content:
            label,action = line.split(' ')
            #print label,action
            if action not in self.action_label.keys():
                self.action_label[action]=label

    def split_video(self):
        self.get_action_index()
        for path,subdir,files in os.walk(self.path):
            for filename in files:
                if filename.split('.')[0] == 'trainlist'+self.split:
                    train_video = self.file2_dic(self.path+filename)
                if filename.split('.')[0] == 'testlist'+self.split:
                    test_video = self.file2_dic(self.path+filename)
        print( '==> (Training video, Validation video):(', len(train_video),len(test_video),')')
        self.train_video = train_video
        self.test_video = test_video

        return self.train_video, self.test_video

    def file2_dic(self,fname):
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip('\r\n') for x in content]
        f.close()
        dic={}
        for line in content:
            #print line
            video = line.split('/',1)[1].split(' ',1)[0]
            key = video.split('.',1)[0]
            label = self.action_label[line.split('/')[0]]   
            dic[key] = int(label)
            #print key,label
        return dic


if __name__ == '__main__':
    path = 'NDA_list/'
    split = '01'
    splitter = NDA_splitter(path=path,split=split)
    train_video,test_video = splitter.split_video()
    print( len(train_video),len(test_video))
