import os
import cv2
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Make demo video')
parser.add_argument('--jpg-path', default='cropped_jpeg_1080', type=str, metavar='PATH', help='path to jpeg folder')
parser.add_argument('--opt-path', default='cropped_flowNet_1080', type=str, metavar='PATH', help='path to opt folder')
parser.add_argument('--participant', default='g03', type=str, help='participant name')
parser.add_argument('--numberOfVideo', default=35, metavar='N', type=int, help='number of test video')
parser.add_argument('--rgbPred', default='record/spatial/evel_spatial_video_preds.pickle', type=str, metavar='PATH', help='path to rbg prediction pickle' )
parser.add_argument('--optPred', default='record/motion/evel_motion_video_preds.pickle', type=str, metavar='PATH', help='path to optical flow prediction pickle' )
parser.add_argument('--classes', default='NDA_list/classInd.txt', type=str, metavar='PATH', help='path to optical flow prediction pickle' )
global arg
arg = parser.parse_args()
print(arg)

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

participant_name = arg.participant

# calculate prediction percentage matrix
percentage_cate_dic = {}
fuse_highest_percentage_cate_dic = {}
rgb_highest_percentage_cate_dic = {}
opt_highest_percentage_cate_dic = {}
cate_image_path = []
for name in sorted(rgb_dic_frame.keys()):
	if arg.participant in name:
		cate_image_path.append(name)
		r = rgb_dic_frame[name]
		r = r/sum(abs(r))
		o = opt_dic_frame[name]
		o = o/sum(abs(o))
		fuse = r+o
		fuse = fuse - min(fuse)
		fuse = fuse/sum(fuse)*100
		percentage_cate_dic[name] = fuse

		rgb_predict_class = classes[r.tolist().index(max(r))]
		optical_flow_predict_class = classes[o.tolist().index(max(o))]
		fuse_predict_class = classes[fuse.tolist().index(max(fuse))]

		fuse_highest_percentage_cate_dic[name] = fuse_predict_class
		rgb_highest_percentage_cate_dic[name] = rgb_predict_class
		opt_highest_percentage_cate_dic[name] = optical_flow_predict_class
# print(cate_image_path)
opt_image = []
rbg_image= []

height = 500
width = 500
size = (height, width)

rbg_root = arg.jpg_path
opt_root = arg.opt_path
font = cv2.FONT_HERSHEY_SIMPLEX
num_video = arg.numberOfVideo
for ram, cate in enumerate(cate_image_path):
	if ram>num_video:
		break
	ground_truth = cate.split('_g')[0]
	opt_first_img = np.zeros([height, width, 3], dtype=np.uint8)
	cv2.putText(opt_first_img,'FlowNet 2.0 optical flow',(10, 30), font, 1,(0,255,255),1,cv2.LINE_AA)
	predict_frame = np.zeros([height, width, 3], dtype=np.uint8)

	cv2.putText(predict_frame,'Percentage after fuse',(10, 30), font, 1,(0,255,255),1,cv2.LINE_AA)
	for position, className in enumerate(classes):
		cv2.putText(predict_frame,className+': ',(10, (position+2)*30), font, 1,(255,255,255),1,cv2.LINE_AA)
		cv2.putText(predict_frame,'0.000%',(250, (position+2)*30), font, 1,(0,255,0),1,cv2.LINE_AA)  

	cv2.putText(predict_frame,'RGB stream:     ',(10, (position+3)*30), font, 1,(255,255,0),1,cv2.LINE_AA)
	cv2.putText(predict_frame,'Optical flow:   ',(10, (position+4)*30), font, 1,(255,255,0),1,cv2.LINE_AA)
	cv2.putText(predict_frame,'Fused:         ',(10, (position+5)*30), font, 1,(255,255,0),1,cv2.LINE_AA)
	cv2.putText(predict_frame,'Ground Truth:  '+ ground_truth,(10, (position+6)*30), font, 1,(255,255,0),1,cv2.LINE_AA)
			
	combined = np.concatenate((opt_first_img, predict_frame), axis=1)
	opt_image.append(combined)
	ind = 1
	rbg_cate_root = os.path.join(rbg_root, cate)
	opt_cate_root = os.path.join(opt_root, cate)

	for file in os.listdir(opt_cate_root):
		if '.' in file:
			if file.split('.')[1] == 'jpg':
				opt_frame_name = os.path.join(opt_cate_root, 'frame_{:06d}.jpg'.format(ind))
				opt = cv2.imread(opt_frame_name)
				opt = cv2.resize(opt, size)
				cv2.putText(opt,'FlowNet 2.0 optical flow',(10, 30), font, 1,(0,255,255),1,cv2.LINE_AA)

				predict_frame = np.zeros([height, width, 3], dtype=np.uint8)
				cv2.putText(predict_frame,'Percentage after fuse',(10, 30), font, 1,(0,255,255),1,cv2.LINE_AA)
				for position, className in enumerate(classes):
					if className == fuse_highest_percentage_cate_dic[cate]:
						color = (0,0,255)
					else:
						color = (255,255,255)
					cv2.putText(predict_frame,className,(10, (position+2)*30), font, 1,color,1,cv2.LINE_AA)
					cv2.putText(predict_frame,str(round((percentage_cate_dic[cate])[position], 3))+'%',(250, (position+2)*30), font, 1,(0,255,0),1,cv2.LINE_AA)  				
				
				cv2.putText(predict_frame,'RGB stream:   ' + rgb_highest_percentage_cate_dic[cate],(10, (position+3)*30), font, 1,(255,255,0),1,cv2.LINE_AA)
				cv2.putText(predict_frame,'Optical flow:   ' + opt_highest_percentage_cate_dic[cate],(10, (position+4)*30), font, 1,(255,255,0),1,cv2.LINE_AA)
				cv2.putText(predict_frame,'Fused:         ' + fuse_highest_percentage_cate_dic[cate],(10, (position+5)*30), font, 1,(255,255,0),1,cv2.LINE_AA)
				cv2.putText(predict_frame,'Ground Truth:  '+ ground_truth,(10, (position+6)*30), font, 1,(255,255,0),1,cv2.LINE_AA)
				combined = np.concatenate((opt, predict_frame), axis=1)
				opt_image.append(combined)

				rbg_frame_name = os.path.join(rbg_cate_root, 'frame_{:06d}.jpg'.format(ind))
				rgb = cv2.imread(rbg_frame_name)
				rgb = cv2.resize(rgb, size)
				cv2.putText(rgb,'RGB frame',(10, 30), font, 1,(255,255,255),1,cv2.LINE_AA)
				rbg_image.append(rgb)

				ind += 1

	rbg_frame_name = os.path.join(rbg_cate_root, 'frame_{:06d}.jpg'.format(ind))
	rgb = cv2.imread(rbg_frame_name)
	rgb = cv2.resize(rgb, size)
	cv2.putText(rgb,'RGB frame',(10, 30), font, 1,(255,255,255),1,cv2.LINE_AA)
	rbg_image.append(rgb)
	print("current frames: ", len(rbg_image))

# opt_last_img = np.zeros([height, width, 3], dtype=np.uint8)
# opt_image.append(opt_last_img)


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
w = rbg_image[0].shape[0]
h = rbg_image[0].shape[0]
print(w, h)
out = cv2.VideoWriter('demo_{}.mp4'.format(participant_name), fourcc, 24, (int(w)*3, int(h)))
print('writing video ...')
for i in range(len(rbg_image)):
	vis = np.concatenate((rbg_image[i], opt_image[i]), axis=1)
	# cv2.imshow('a', vis)
	# cv2.waitKey(10)
	out.write(vis)
out.release()
print('Done.')










