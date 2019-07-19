import pickle, os
import argparse
import pandas as pd

def record_info(info,filename):
	     
    df = pd.DataFrame.from_dict(info)
    column_names = ['RGB Accuracy','Optical Flow Accuracy','Fuse Accuracy']
    
    if not os.path.isfile(filename):
        df.to_csv(filename,index=False,columns=column_names)
    else: # else it exists so append without writing the header
        df.to_csv(filename,mode = 'a',header=False,index=False,columns=column_names)   


parser = argparse.ArgumentParser(description='Read rgb and optical flow prediction pickle')
parser.add_argument('--rgbPred', default='record/spatial/spatial_video_preds.pickle', type=str, metavar='PATH', help='path to rbg prediction pickle' )
parser.add_argument('--optPred', default='record/motion/motion_video_preds.pickle', type=str, metavar='PATH', help='path to optical flow prediction pickle' )

global arg
arg = parser.parse_args()
print (arg)

with open(arg.rgbPred,'rb') as file:
    rgb_dic_frame = pickle.load(file)
file.close()

with open(arg.optPred,'rb') as file:
    opt_dic_frame = pickle.load(file)
file.close()

classes = []
class_file = open('NDA_list/classInd.txt', 'r')
for line in class_file:
	if line[-1] == '\n':
		name = line.split(' ')[1]
		name = name.replace('\n', '')
		classes.append(name)
print(classes)
rgb_correct = 0
opt_correct = 0
fuse_correct = 0
total = 0
for name in sorted(rgb_dic_frame.keys()):
	current_class = name.split('_')[0]
	r = rgb_dic_frame[name]
	r = r/sum(abs(r))
	o = opt_dic_frame[name]
	o = o/sum(abs(o))
	print(name, r, o)
	fuse = r+o
	rgb_predict_class = classes[r.tolist().index(max(r))]
	optical_flow_predict_class = classes[o.tolist().index(max(o))]
	fuse_predict_class = classes[fuse.tolist().index(max(fuse))]

	if fuse_predict_class == current_class:
		fuse_correct += 1
	if rgb_predict_class == current_class:
		rgb_correct += 1
	if optical_flow_predict_class == current_class:
		opt_correct += 1
	total += 1
print('RGB Accuracy = ' + str(rgb_correct/total))
print('Optical Flow Accuracy = ' + str(opt_correct/total))
print('Fuse Accuracy = ' + str(fuse_correct/total))

RGB_A = rgb_correct/total
OPT_A = opt_correct/total
FUSE_A = fuse_correct/total


info = {'RGB Accuracy':[RGB_A], 'Optical Flow Accuracy':[OPT_A],'Fuse Accuracy':[FUSE_A]}
record_info(info, 'record/inference/acc.csv')
