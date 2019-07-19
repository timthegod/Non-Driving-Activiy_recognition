import os
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Make demo video')
parser.add_argument('--jpg-path', default='cropped_jpeg_1080', type=str, metavar='PATH', help='path to jpeg folder')
parser.add_argument('--opt-path', default='cropped_flowNet_1080', type=str, metavar='PATH', help='path to opt folder')
parser.add_argument('--class-name', default='tabletGame_g01', type=str, help='class name')

global arg
arg = parser.parse_args()
print(arg)

cag_name = arg.class_name
rbg_root = os.path.join(arg.jpg_path) + cag_name
opt_root = os.path.join(arg.opt_path) + cag_name

opt_image = []
rbg_image= []
if '480' in opt_root:
	height = 300
	width = 300
else:
	height = 600
	width = 600
size = (height, width)

opt_first_img = np.zeros([height, width, 3], dtype=np.uint8)
opt_image.append(opt_first_img)
ind = 1
for file in os.listdir(opt_root):
	if '.' in file:
		if file.split('.')[1] == 'jpg':
			opt_frame_name = os.path.join(opt_root, 'frame_{:06d}.jpg'.format(ind))
			opt = cv2.imread(opt_frame_name)
			opt = cv2.resize(opt, size)
			opt_image.append(opt)

			rbg_frame_name = os.path.join(rbg_root, 'frame_{:06d}.jpg'.format(ind))
			rgb = cv2.imread(rbg_frame_name)
			rgb = cv2.resize(rgb, size)
			rbg_image.append(rgb)

			ind += 1

rbg_frame_name = os.path.join(rbg_root, 'frame_{:06d}.jpg'.format(ind))
rgb = cv2.imread(rbg_frame_name)
rbg_image.append(rgb)

# opt_last_img = np.zeros([height, width, 3], dtype=np.uint8)
# opt_image.append(opt_last_img)

print(len(rbg_image), len(opt_image))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
w = rbg_image[0].shape[0]
h = opt_image[0].shape[0]
print(w, h)
out = cv2.VideoWriter('demo_{}.mp4'.format(cag_name), fourcc, 24, (int(w)*2, int(h)))

for i in range(len(rbg_image)):
	vis = np.concatenate((opt_image[i], rbg_image[i]), axis=1)
	# cv2.imshow('a', vis)
	# cv2.waitKey(10)
	out.write(vis)
out.release()










