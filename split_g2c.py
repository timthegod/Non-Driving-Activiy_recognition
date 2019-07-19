import os 
import argparse
from shutil import copyfile

parser = argparse.ArgumentParser(description='split g into c')
parser.add_argument('--jpg-path', default='cropped_jpeg_1080', type=str, metavar='PATH', help='path to jpeg folder')
parser.add_argument('--opt-path', default='cropped_flowNet_1080', type=str, metavar='PATH', help='path to opt folder')
parser.add_argument('--fps', default=24, type=int, metavar='N', help='frame per second')
parser.add_argument('--sec', default=3, type=int, metavar='N', help='how many second to cut each video')

global arg
arg = parser.parse_args()
print(arg)

jpg_src_path = arg.jpg_path
opt_src_path = arg.opt_path

jpg_des_path = 'jpeg_1080'
opt_des_path = 'flowNet_1080'

if not os.path.isdir(jpg_des_path):
	os.mkdir(jpg_des_path)
if not os.path.isdir(opt_des_path):
	os.mkdir(opt_des_path)
print('jpg splitting...')

c_dic = {}

for fold in os.listdir(jpg_src_path):
	if os.path.isdir(os.path.join(jpg_src_path, fold)):
		image_cnt = 0
		for file in os.listdir(os.path.join(jpg_src_path, fold)):
			if '.jpg' in file:
				image_cnt += 1
		
		each_cnt = arg.fps * arg.sec
		c_cnt = int(image_cnt/each_cnt)
		c_dic[fold] = c_cnt
		
		c = 1
		ind = 1
		c_name = fold + '_c{:02d}'.format(c)
		if not os.path.isdir(os.path.join(jpg_des_path, c_name)):
			os.mkdir(os.path.join(jpg_des_path, c_name))
		src_root = os.path.join(jpg_src_path, fold)
		des_root = os.path.join(jpg_des_path, c_name)

		break_sign = False
		for i in range(1,image_cnt+1):
			if ind > each_cnt:
				c += 1
				if c > c_cnt:
					break_sign = True
				ind = 1
				c_name = fold + '_c{:02d}'.format(c)
				if not os.path.isdir(os.path.join(jpg_des_path, c_name)) and not break_sign:
					os.mkdir(os.path.join(jpg_des_path, c_name))
					src_root = os.path.join(jpg_src_path, fold)
					des_root = os.path.join(jpg_des_path, c_name)
			if break_sign:
				print(fold, image_cnt, c_cnt, c)
				break
			copyfile(os.path.join(src_root, 'frame_{:06d}.jpg'.format(i)), os.path.join(des_root, 'frame_{:06d}.jpg'.format(ind)))
			ind += 1

print('flowNet splitting...')
for fold in os.listdir(opt_src_path):
	if os.path.isdir(os.path.join(opt_src_path, fold)):
		image_cnt = 0
		for file in os.listdir(os.path.join(opt_src_path, fold)):
			if '.jpg' in file:
				image_cnt += 1
		
		each_cnt = arg.fps * arg.sec - 1
		c_cnt = c_dic[fold]

		c = 1
		ind = 1
		c_name = fold + '_c{:02d}'.format(c)
		if not os.path.isdir(os.path.join(opt_des_path, c_name)):
			os.mkdir(os.path.join(opt_des_path, c_name))
		src_root = os.path.join(opt_src_path, fold)
		des_root = os.path.join(opt_des_path, c_name)

		break_sign = False
		i = 1
		while not break_sign:
			if ind > each_cnt:
				c += 1
				i += 1 # ex no.72 is not needed
				if c > c_cnt:
					break_sign = True
				ind = 1
				c_name = fold + '_c{:02d}'.format(c)
				if not os.path.isdir(os.path.join(opt_des_path, c_name)) and not break_sign:
					os.mkdir(os.path.join(opt_des_path, c_name))
					src_root = os.path.join(opt_src_path, fold)
					des_root = os.path.join(opt_des_path, c_name)
			if break_sign:
				print(fold, image_cnt, c_cnt, c)
				break
			copyfile(os.path.join(src_root, 'frame_{:06d}.jpg'.format(i)), os.path.join(des_root, 'frame_{:06d}.jpg'.format(ind)))
			ind += 1
			i += 1




