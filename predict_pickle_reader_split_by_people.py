import pickle
import argparse
import matplotlib.pyplot as plt 
parser = argparse.ArgumentParser(description='Read rgb and optical flow prediction pickle')
parser.add_argument('--rgbPred', default='record/spatial/evel_spatial_video_preds.pickle', type=str, metavar='PATH', help='path to rbg prediction pickle' )
parser.add_argument('--optPred', default='record/motion/evel_motion_video_preds.pickle', type=str, metavar='PATH', help='path to optical flow prediction pickle' )
parser.add_argument('--classes', default='NDA_list/classInd.txt', type=str, metavar='PATH', help='path to optical flow prediction pickle' )

global arg
arg = parser.parse_args()
print (arg)

with open(arg.rgbPred,'rb') as file:
    rgb_dic_frame = pickle.load(file)
file.close()

with open(arg.optPred,'rb') as file:
    opt_dic_frame = pickle.load(file)
file.close()

def add_correct_dic(people_name, people_dic):
	if people_name not in people_dic:
		people_dic[people_name] = 1
	else:
		people_dic[people_name] += 1

def add_dic(people_name, people_dic):
	if people_name not in people_dic:
		people_dic[people_name] = 1
	else:
		people_dic[people_name] += 1		


classes = []
class_file = open(arg.classes, 'r')
for line in class_file:
	if line[-1] == '\n':
		name = line.split(' ')[1]
		name = name.replace('\n', '')
		classes.append(name)

people_occur = {}

people_rgb_correct = {}
people_opt_correct = {}
people_fuse_correct = {}

rgb_correct = 0
opt_correct = 0
fuse_correct = 0

total = 0

for name in sorted(rgb_dic_frame.keys()):
	current_class = name.split('_')[0]
	current_name = name.split('_')[1]
	add_dic(current_name, people_occur)
	r = rgb_dic_frame[name]
	r = r/sum(abs(r))
	o = opt_dic_frame[name]
	o = o/sum(abs(o))
	# print(name, r, o)
	fuse = r+o
	rgb_predict_class = classes[r.tolist().index(max(r))]
	optical_flow_predict_class = classes[o.tolist().index(max(o))]
	fuse_predict_class = classes[fuse.tolist().index(max(fuse))]

	if fuse_predict_class == current_class:
		add_correct_dic(current_name, people_fuse_correct)
		fuse_correct += 1
	if rgb_predict_class == current_class:
		add_correct_dic(current_name, people_rgb_correct)
		rgb_correct += 1
	if optical_flow_predict_class == current_class:
		add_correct_dic(current_name, people_opt_correct)
		opt_correct += 1
	total += 1

print('RGB Accuracy = ' + str(rgb_correct/total))
print('Optical Flow Accuracy = ' + str(opt_correct/total))
print('Fuse Accuracy = ' + str(fuse_correct/total))

for people in people_occur:
	print(people, 'RGB', people_rgb_correct[people]/people_occur[people])
	print(people, 'OPT', people_opt_correct[people]/people_occur[people])
	print(people, 'FUSE', people_fuse_correct[people]/people_occur[people])

list_people_fuse_accuracy = []
list_people_rgb_accuracy = []
list_people_opt_accuracy = []
list_people_name = []

for items in people_fuse_correct.items():
	name, corrects = items
	list_people_fuse_accuracy.append(people_fuse_correct[name]/people_occur[name])
	list_people_rgb_accuracy.append(people_rgb_correct[name]/people_occur[name])
	list_people_opt_accuracy.append(people_opt_correct[name]/people_occur[name])
	list_people_name.append(name)

fig = plt.figure(figsize=(14,6))
fig.add_subplot(131)
x = range(len(list_people_name))
barList_r = plt.bar(x, list_people_rgb_accuracy)
plt.xticks(x, list_people_name)
plt.title('rgb')
plt.grid()

fig.add_subplot(132)
x = range(len(list_people_name))
barList_r = plt.bar(x, list_people_opt_accuracy)
plt.xticks(x, list_people_name)
plt.title('opt')
plt.grid()

fig.add_subplot(133)
x = range(len(list_people_name))
barList_r = plt.bar(x, list_people_fuse_accuracy)
plt.xticks(x, list_people_name)
plt.title('fuse')
plt.grid()
plt.show()
