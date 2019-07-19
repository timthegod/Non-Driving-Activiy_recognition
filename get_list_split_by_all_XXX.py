import os 
from collections import OrderedDict
from random import shuffle
import argparse

parser = argparse.ArgumentParser(description='get train and test lsit both sampled and no_sampled')
parser.add_argument('--jpg-path', default='jpeg_1080', type=str, metavar='PATH', help='path to jpeg folder')

train_list = open('NDA_list/trainlist01.txt', 'w')
test_list = open('NDA_list/testlist01.txt', 'w')
n_train_list = open('n_NDA_list/trainlist01.txt', 'w')
n_test_list = open('n_NDA_list/testlist01.txt', 'w')

global arg
arg = parser.parse_args()
print(arg)

src_jpg = arg.jpg_path

class_file = open('NDA_list/classInd.txt', 'r')
category_ind = {}
for line in class_file:
	if line.split(' ')[1][-1] == '\n':
		name = line.split(' ')[1]
		name = name.replace('\n', '')
		category_ind[name] = line.split(' ')[0]
	else:
		category_ind[line.split(' ')[1]] = line.split(' ')[0]

category_ind_sorted_by_value = OrderedDict(sorted(category_ind.items(), key=lambda x: x[1]))

all_file_name = []
no_sample_file = []
# initialize 2D list 
for _ in range(len(category_ind_sorted_by_value)):
	tmp = []
	all_file_name.append(tmp)
# append video name to corresponding index list
for dirc in os.listdir(src_jpg):
	if dirc != '.DS_Store':
		current_class_name = dirc.split('_')[0]
		if current_class_name in category_ind:
			all_file_name[int(category_ind[current_class_name]) - 1].append(dirc + '.mp4')
for i in range(len(all_file_name)):
	shuffle(all_file_name[i])
	l = len(all_file_name[i])
	tmp = all_file_name[i][:int(l * 0.2)]
	for name in tmp:
		no_sample_file.append(name)

print(no_sample_file)


for key in category_ind_sorted_by_value.keys():
	ind = {} # g has how many c
	for file in os.listdir(src_jpg):
		if file.split('_')[0] == key:
			g = int((file.split('_c')[0]).split('g')[1])
			c = int(file.split('_c')[1])
			if g not in ind:
				ind[g] = c
			else:
				if ind[g] < c:
					ind[g] = c
	ind_sorted_by_key = OrderedDict(sorted(ind.items(), key=lambda x: x[0]))
	for g in ind_sorted_by_key.keys():
		for c in range(1, ind_sorted_by_key[g] + 1):
			if c > (ind_sorted_by_key[g]*4/5):
				train = False
			else:
				train = True
			video_name = key + '_g{:02d}'.format(g) + '_c{:02d}.mp4'.format(c)
			
			if video_name not in no_sample_file:
				if train:
					out = '{}/{} {}\n'.format(key, video_name, category_ind_sorted_by_value[key])
					train_list.write(out)
				else:
					out = '{}/{}\n'.format(key, video_name)
					test_list.write(out)
			else:
				class_name = video_name.split('_')[0]
				n_train_out = '{}/{} {}\n'.format(class_name, video_name, category_ind_sorted_by_value[class_name])
				n_train_list.write(n_train_out)

				n_test_out = out = '{}/{}\n'.format(class_name, video_name)
				n_test_list.write(n_test_out)

# for no in no_sample_file:
# 	class_name = no.split('_')[0]
# 	n_train_out = '{}/{} {}\n'.format(class_name, no, category_ind_sorted_by_value[class_name])
# 	n_train_list.write(n_train_out)

# 	n_test_out = out = '{}/{}\n'.format(class_name, no)
# 	n_test_list.write(n_test_out)


