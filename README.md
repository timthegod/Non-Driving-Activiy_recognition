# Non-Driving-Activiy_recognition
We use a spatial and motion stream cnn with ResNet50 for modeling video information in NDAs (Non-Driving-Activities) dataset that collected ourselves. NDAs are one of driver's activity, however,  when level 3 Autonomous Vehicle is in autopilot mode, the primary task of drivers are no longer driving. But we still need to understand what the drivers are doing to know the reaction time, because if the driver needs to take control the reaction time of the period became a crucial information.
For the motion stream we use FlowNet 2.0 to generate the optical flow.

## 1. Data (Not open sourced yet)
### 1.1 Spatial input data -> rgb videos
* This is the original video from our NDAs dataset.
The size is about 3G.

### 1.2 Temporal(motion) input data -> optical flow frame
* In temporal stream, we used FlowNet 2.0 to extract the optical flow.
* You could follow the [FlowNet2.0 Colab Notebook](https://drive.google.com/open?id=1H5EtxZYbVWRyc2MwdAA1wJWP9jsYAqWU) for obtaining the optical flow of the video stream

## 2. Model
### 2.1 Spatial Convolutional Neural Network (Spatial CNN)
* We use ResNet50 first pre-trained with ImageNet to be the main structure, the input of the Spatial CNN is the frame that extract from the original video stream. The input shape is (3, 244, 244). The resize and data augmentation are all automotically done in the code.

### 2.2 Temporal Convolutional Neural Network (temporal CNN)
* Input data of temporal cnn is a stack of optical flow images which is a RGB-visualised optiacal flow image using FlowNet2.0, So the input shape is (30, 224, 224) which can be considered as a 30-channel image, 30 comes from RGB (3-channel) times 10 images per stack.
* In order to utilize ImageNet pre-trained weight on our model, we have to modify the weights of the first convolution layer pre-trained  with ImageNet from (64, 3, 7, 7) to (64, 30, 7, 7).

## 3. Training methodology
###  3.1 Spatial cnn
* For every videos in a mini-batch, we randomly select 3 frames from each video. Then a consensus among the frames will be derived as the video-level prediction for calculating loss.

###  3.1 Temporal cnn
* In every mini-batch, we randomly select 16 (batch size) videos from the training videos and futher randomly select 1 stacked optical flow in each video. 

### 3.3 Data augmentation
* Both stream apply the same data augmentation technique such as random cropping.

## 4. Testing method
* For every testing videos, we uniformly sample 19 frames in each video and the video level prediction is the voting result of all 19 frame level predictions.

## 5. Performace
   
 network       | top1  |
---------------|:-----:|
Spatial stream | 80.8% | 
Motion stream  | 83.3% | 
Fusion         | 96.4% |   

## 6. Pre-trained Model

* [Spatial resnet50](https://drive.google.com/open?id=1yAqPsX52jSthPVczzNCX8kizwOXRS9g8)
* [Motion resnet50](https://drive.google.com/open?id=1p_o4Ca2arikxHwQ-hHQHpMZA2twyQvqa)

## 7. Testing by yourself Device
* The below is the link of my Google Colab notebook, all the instruction could be found on it. If there is any problem, leave a issue on this git repository.
* [Google Colab notebook](https://colab.research.google.com/drive/119H6hfq19yO_OXuinyo44rFsgsxwhedh)


## Acknowledgment
* This code is modified from [jeffreyhuang](https://github.com/jeffreyhuang1/two-stream-action-recognition). All the edition is completed by myself in order to fit the NDAs Recognition project.
* The optical flow images was estimated using FlowNet 2.0 from [NVIDIA git repository](https://github.com/NVIDIA/flownet2-pytorch)




   
   
