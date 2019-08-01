# get the categoy from the jpeg data
import os 
from collections import OrderedDict
from random import shuffle
import argparse

parser = argparse.ArgumentParser(description='get train and test lsit both sampled and no_sampled')
parser.add_argument('--jpg-path', default='jpeg_1080', type=str, metavar='PATH', help='path to jpeg folder')

classInd = open('NDA_list/classInd.txt', 'w')
n_classInd = open('n_NDA_list/classInd.txt', 'w')


global arg
arg = parser.parse_args()
print(arg)


src_jpg = arg.jpg_path
class_name = []
for dirc in os.listdir(src_jpg):
	if os.path.isdir(os.path.join(src_jpg, dirc)):
		name = dirc.split('_')[0]
		if name not in class_name:
			class_name.append(name)
class_name.sort()

for i, name in enumerate(class_name):
	out = '{} {}\n'.format(i + 1, name)
	print(out)
	classInd.write(out)
	n_classInd.write(out)