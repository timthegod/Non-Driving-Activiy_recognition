import pickle

with open('dataloader/dic/frame_count.pickle' ,'rb') as file:
    dic = pickle.load(file)
file.close()
print(dic)

