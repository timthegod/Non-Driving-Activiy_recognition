import pickle
import os
import argparse

parser = argparse.ArgumentParser(description='get train and test lsit both sampled and no_sampled')
parser.add_argument('--jpg-path', default='jpeg_1080', type=str, metavar='PATH', help='path to jpeg folder')

global arg
arg = parser.parse_args()
print(arg)

src_jpg = arg.jpg_path

src_path = []
for dirc in os.listdir(src_jpg):
	if dirc != '.DS_Store':
		src_path.append(os.path.join(src_jpg, dirc))

out_dic = {}

for path in src_path:
	fc = 0
	for frame_name in os.listdir(path):
		if frame_name.split('.')[1] == 'jpg':
			fc += 1
	file_name = path.split('/')[1]
	out_dic[file_name] = fc

pickle_out = open("dataloader/dic/frame_count.pickle","wb")
pickle.dump(out_dic, pickle_out)
pickle_out.close()
