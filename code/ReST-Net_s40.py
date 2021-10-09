 """
 -------------------------------------------------------------------------
 Copyright 2018 Guangdong Key Laboratory of Intelligent Information 
 Processing and Shenzhen Key Laboratory of Media Security, Shenzhen University.
 All Rights Reserved.
 -------------------------------------------------------------------------
 Contact: libin@szu.edu.cn | 2150130427@email.szu.edu.cn | March 2018
          http://
 -------------------------------------------------------------------------
 ReST-Net[1] is trained in two phases. In the first phase, each 
 sub-net is pre-trained independently with a fullyconnected layer and a 
 Softmax function, to classify cover and stego images. as showed in 
 pretrain_subnet1_s40.py、pretrain_subnet2_s40.py and pretrain_subnet3_s40.py
 Once the pre-training is done, the parameters in the sub-nets are frozen 
 without further training, and the fully-connected layers are discarded.
 In the second phase, a new fully-connected layer with 768 (256 × 3) input 
 neurons is fed with the concatenated output feature vectors from the final 
 convolutional groups of all three sub-nets. Such a fully-connected layer with 
 a Softmax function is trained and acts as the final classification module.
 -------------------------------------------------------------------------
 Input:  BOSSBase v1.01 dataset [2] and and S-UNIWARD [3] for data embedding, 
         with the payload 0.4 bpp
 Output: the training model of ReST-Net
 -------------------------------------------------------------------------
 [1] B. Li, W. Wei, A. Ferreira, and S. Tan, ReST-Net: Diverse Activation 
 Modules and Parallel Sub-nets based CNN for Spatial Image Steganalysis, 
 IEEE Signal Processing Letters,Under review.
 [2] P. B. T. Filler and T. Pevny, “Break our steganographic system—the 
 ins and outs of organizing BOSS,” in Proc. 13th Information Hiding
 Workshop (IH’2011), 2011, pp. 59–70.
 [3] V. Holub, J. Fridrich, and T. Denemark, “Universal distortion function
 for steganography in an arbitrary domain,” EURASIP Journal on Information
 Security, vol. 2014, no. 1, pp. 1–13, 2014.
 -------------------------------------------------------------------------
"""
# import some toolkits
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os
import random
from scipy import ndimage
from pandas import DataFrame

# setup 
BATCH_SIZE = 20
IMAGE_SIZE = 512
NUM_CHANNEL = 1
NUM_LABELS = 2
NUM_ITER =10000
NUM_SAVE = 400
NUM_SHOWTRAIN = 50 #show result eveary epoch 
NUM_SHOWVALIDATION = 400
LEARNING_RATE =0.001
LEARNING_RATE_DECAY = 0.1
MOMENTUM = 0.9
DECAY_STEP = 5000

# the path of dataset
path1 = '/data/BOSS'
path2 = '/data/UNIWARD40'
saver_path = os.path.abspath(os.path.dirname(os.getcwd()))

# randomly shuffling dataset
fileList = []
for (dirpath,dirnames,filenames) in os.walk(path1):  #0~5000 for training  5001~10000 for testing
    fileList = filenames
np.set_printoptions(threshold='nan')
random.seed(1234)
random.shuffle(fileList)

# the input of model
x = tf.placeholder(tf.float32,shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
y = tf.placeholder(tf.float32,shape=[BATCH_SIZE,NUM_LABELS])
is_train = tf.placeholder(tf.bool,name='is_train')

# images were preprocessed with 14 nonlinear filters
# nonlinearly processed with max and min operation 
def multi_minmax(input):
    input_shape = input.get_shape()
    min = tf.reduce_min(input,3)
    max = tf.reduce_max(input,3)
    min = tf.reshape(min,[input_shape[0].value,input_shape[1].value,input_shape[2].value,1])
    max = tf.reshape(max,[input_shape[0].value,input_shape[1].value,input_shape[2].value,1])
    output = tf.concat([min,max],3)
    return output


#D1
hpf1 = np.zeros( [5,5,1,4],dtype=np.float32 ) #[height,width,input,output]
hpf1[1:4, 1:4, 0, 0] = np.array([[0,1,0], [0,-1,0], [0,0,0]], dtype=np.float32)
hpf1[1:4, 1:4, 0, 1] = np.array([[0,0,0], [0,-1,1], [0,0,0]], dtype=np.float32)
hpf1[1:4, 1:4, 0, 2] = np.array([[0,0,0], [0,-1,0], [0,1,0]], dtype=np.float32)
hpf1[1:4, 1:4, 0, 3] = np.array([[0,0,0], [1,-1,0], [0,0,0]], dtype=np.float32)
kernel0_1 = tf.Variable(hpf1,name="kernel0")
conv0_1 = tf.nn.conv2d(x,kernel0_1,[1,1,1,1],'SAME',name="conv0_1")
convminmax1 = multi_minmax(conv0_1)

#D2
hpf2 = np.zeros( [5,5,1,4],dtype=np.float32 ) #[height,width,input,output]
hpf2[1:4, 1:4, 0, 0] = np.array([[0,0,1], [0,-1,0], [0,0,0]], dtype=np.float32)
hpf2[1:4, 1:4, 0, 1] = np.array([[0,0,0], [0,-1,0], [0,0,1]], dtype=np.float32)
hpf2[1:4, 1:4, 0, 2] = np.array([[0,0,0], [0,-1,0], [1,0,0]], dtype=np.float32)
hpf2[1:4, 1:4, 0, 3] = np.array([[1,0,0], [0,-1,0], [0,0,0]], dtype=np.float32)
kernel0_2 = tf.Variable(hpf2,name="kernel0")
conv0_2 = tf.nn.conv2d(x,kernel0_2,[1,1,1,1],'SAME',name="conv0_2")
convminmax2 = multi_minmax(conv0_2)


#D3
hpf3 = np.zeros( [5,5,1,4],dtype=np.float32 ) #[height,width,input,output]
hpf3[1:4,1:4,0,0] = np.array([[0,1,0],[0,-2,0],[0,1,0]],dtype=np.float32)
hpf3[1:4,1:4,0,1] = np.array([[0,0,1],[0,-2,0],[1,0,0]],dtype=np.float32)
hpf3[1:4,1:4,0,2] = np.array([[0,0,0],[1,-2,1],[0,0,0]],dtype=np.float32)
hpf3[1:4,1:4,0,3] = np.array([[1,0,0],[0,-2,0],[0,0,1]],dtype=np.float32)
kernel0_3 = tf.Variable(hpf3,name="kernel0")
conv0_3 = tf.nn.conv2d(x,kernel0_3,[1,1,1,1],'SAME',name="conv0_3")
convminmax3 = multi_minmax(conv0_3)

#D4
hpf4 = np.zeros( [5,5,1,4],dtype=np.float32 ) #[height,width,input,output]
hpf4[:,:,0,0] = np.array( [ [0,0,-1,0,0],[0,0,3,0,0],[0,0,-3,0,0],[0,0,1,0,0],[0,0,0,0,0] ]  ,dtype=np.float32)
hpf4[:,:,0,1] = np.array( [ [0,0,0,0,0],[0,0,0,0,0],[0,1,-3,3,-1],[0,0,0,0,0],[0,0,0,0,0] ]  ,dtype=np.float32)
hpf4[:,:,0,2] = np.array( [ [0,0,0,0,0],[0,0,1,0,0],[0,0,-3,0,0],[0,0,3,0,0],[0,0,-1,0,0] ]  ,dtype=np.float32)
hpf4[:,:,0,3] = np.array( [ [0,0,0,0,0],[0,0,0,0,0],[-1,3,-3,1,0],[0,0,0,0,0],[0,0,0,0,0] ]  ,dtype=np.float32)
kernel0_4 = tf.Variable(hpf4,name="kernel0")
conv0_4 = tf.nn.conv2d(x,kernel0_4,[1,1,1,1],'SAME',name="conv0_4")
convminmax4 = multi_minmax(conv0_4)

#D5
hpf5 = np.zeros( [5,5,1,4],dtype=np.float32 ) #[height,width,input,output]
hpf5[:,:,0,0] = np.array( [ [0,0,0,0,-1],[0,0,0,3,0],[0,0,-3,0,0],[0,1,0,0,0],[0,0,0,0,0] ]  ,dtype=np.float32)
hpf5[:,:,0,1] = np.array( [ [0,0,0,0,0],[0,1,0,0,0],[0,0,-3,0,0],[0,0,0,3,0],[0,0,0,0,-1] ]  ,dtype=np.float32)
hpf5[:,:,0,2] = np.array( [ [0,0,0,0,0],[0,0,0,1,0],[0,0,-3,0,0],[0,3,0,0,0],[-1,0,0,0,0] ]  ,dtype=np.float32)
hpf5[:,:,0,3] = np.array( [ [-1,0,0,0,0],[0,3,0,0,0],[0,0,-3,0,0],[0,0,0,1,0],[0,0,0,0,0] ]  ,dtype=np.float32)
kernel0_5 = tf.Variable(hpf5,name="kernel0")
conv0_5 = tf.nn.conv2d(x,kernel0_5,[1,1,1,1],'SAME',name="conv0_5")
convminmax5 = multi_minmax(conv0_5)

#D6
hpf6 = np.zeros( [5,5,1,4],dtype=np.float32 ) #[height,width,input,output]
hpf6[1:4,1:4,0,0] = np.array(  [[-1,2,-1],[2,-4,2],[0,0,0]]  ,dtype=np.float32)
hpf6[1:4,1:4,0,1] = np.array(  [[-1,2,0],[2,-4,0],[-1,2,0]]  ,dtype=np.float32)
hpf6[1:4,1:4,0,2] = np.array(  [[0,0,0],[2,-4,2],[-1,2,-1]]  ,dtype=np.float32)
hpf6[1:4,1:4,0,3] = np.array(  [[0,2,-1],[0,-4,0],[0,2,-1]]  ,dtype=np.float32)
kernel0_6 = tf.Variable(hpf6,name="kernel0")
conv0_6 = tf.nn.conv2d(x,kernel0_6,[1,1,1,1],'SAME',name="conv0_6")
convminmax6 = multi_minmax(conv0_6)

#D7
hpf7 = np.zeros( [5,5,1,4],dtype=np.float32 ) #[height,width,input,output]
hpf7[:,:,0,0] = np.array( [ [-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[0,0,0,0,0],[0,0,0,0,0] ]  ,dtype=np.float32)
hpf7[:,:,0,1] = np.array( [ [-1,2,-2,0,0],[2,-6,8,0,0],[-2,8,-12,0,0],[2,-6,8,0,0],[-1,2,-2,0,0] ]  ,dtype=np.float32)
hpf7[:,:,0,2] = np.array( [ [0,0,0,0,0],[0,0,0,0,0],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1] ]  ,dtype=np.float32)
hpf7[:,:,0,3] = np.array( [ [0,0,-2,2,-1],[0,0,8,-6,2],[0,0,-12,8,-2],[0,0,8,-6,2],[0,0,-2,2,-1] ]  ,dtype=np.float32)
kernel0_7 = tf.Variable(hpf7,name="kernel0")
conv0_7 = tf.nn.conv2d(x,kernel0_7,[1,1,1,1],'SAME',name="conv0_7")
convminmax7 = multi_minmax(conv0_7)
conv0_0 = tf.concat([convminmax1,convminmax2,convminmax3,convminmax4,convminmax5,convminmax6,convminmax7],3)


# images were preprocessed with 14 SRM filters
hpf11 = np.zeros( [5,5,1,14],dtype=np.float32 ) #[height,width,input,output]
hpf11[1:4,1:4,0,0] = np.array(  [[0,0,0],[1,-2,1],[0,0,0]]  ,dtype=np.float32)
hpf11[1:4,1:4,0,1] = np.array(  [[0,1,0],[0,-2,0],[0,1,0]]  ,dtype=np.float32)
hpf11[1:4,1:4,0,2] = np.array( [[1,0,0],[0,-2,0],[0,0,1]]  ,dtype=np.float32)
hpf11[1:4,1:4,0,3] = np.array( [[0,0,1],[0,-2,0],[1,0,0]]  ,dtype=np.float32)         
hpf11[1:4,1:4,0,4] = np.array( [[-1,2,-1],[2,-4,2],[-1,2,-1]]  ,dtype=np.float32)    
hpf11[:,:,0,5] = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],dtype=np.float32)
hpf11[1:4,1:4,0,10] = np.array( [[-1,2,-1],[2,-4,2],[0,0,0]]  ,dtype=np.float32)
hpf11[1:4,1:4,0,11] = np.array( [[-1,2,0],[2,-4,0],[-1,2,0]]  ,dtype=np.float32)
hpf11[1:4,1:4,0,12] = np.array( [[0,0,0],[2,-4,2],[-1,2,-1]]  ,dtype=np.float32)
hpf11[1:4,1:4,0,13] = np.array( [[0,2,-1],[0,-4,2],[0,2,-1]]  ,dtype=np.float32)
hpf11[:,:,0,6] = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[0,0,0,0,0],[0,0,0,0,0]],dtype=np.float32)
hpf11[:,:,0,7] = np.array([[0,0,0,0,0],[0,0,0,0,0],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]],dtype=np.float32)
hpf11[:,:,0,8] = np.array([[0,0,-2,2,-1],[0,0,8,-6,2],[0,0,-12,8,-2],[0,0,8,-6,2],[0,0,-2,2,-1]],dtype=np.float32)
hpf11[:,:,0,9] = np.array([[-1,2,-2,0,0],[2,-6,8,0,0],[-2,8,-12,0,0],[2,-6,8,0,0],[-1,2,-2,0,0]],dtype=np.float32)

kernel1_1 = tf.Variable(hpf11,name="kernel1")
conv1_0 = tf.nn.conv2d(x,kernel1_1,[1,1,1,1],'SAME',name="conv1_1")


# images were preprocessed with 16 Gabor filters
hpf = np.zeros( [6,6,1,16],dtype=np.float32 ) #[height,width,input,output]
hpf[:,:,0,0] = np.array([[0.0101,-0.0001,-0.0100,-0.0100,-0.0001,0.0101],[0.0748,-0.0006,-0.0742,-0.0742,-0.0006,0.0748],[0.2033,-0.0017,-0.2016,-0.2016,-0.0017,0.2033],[0.2033,-0.0017,-0.2016,-0.2016,-0.0017,0.2033],[0.0748,-0.0006,-0.0742,-0.0742,-0.0006,0.0748],[0.0101,-0.0001,-0.0100,-0.0100,-0.0001,0.0101]
],dtype=np.float32)*10e-05
hpf[:,:,0,1] = np.array([[-0.0001,-0.0095,-0.0224,-0.0014,-0.0000,-0.0000],[0.0002,0.0368,0.2209,0.0341,0.0001,0.0000],[-0.0000,-0.0156,-0.2021,-0.0417,0.0008,0.0000],[0.0000,0.0008,-0.0417,-0.2021,-0.0156,-0.0000],[0.0000,0.0001,0.0341,0.2209,0.0368,0.0002],[-0.0000,-0.0000,-0.0014,-0.0224,-0.0095,-0.0001]
],dtype=np.float32)
hpf[:,:,0,2] = np.array([[0.0001,0.0001,-0.0032,-0.0001,0.0000,0.0000],[0.0001,0.0055,0.0018,-0.0240,-0.0002,0.0000],[-0.0032,0.0018,0.0404,0.0050,-0.0240,-0.0001],[-0.0001,-0.0240,0.0050,0.0404,0.0018,-0.0032],[0.0000,-0.0002,-0.0240,0.0018,0.0055,0.0001],[0.0000,0.0000,-0.0001,-0.0032,0.0001,0.0001]
],dtype=np.float32)
hpf[:,:,0,3] = np.array([[-0.0001,0.0002,-0.0000,0.0000,0.0000,-0.0000],[-0.0095,0.0368,-0.0156,0.0008,0.0001,-0.0000],[-0.0224,0.2209,-0.2021,-0.0417,0.0341,-0.0014],[-0.0014,0.0341,-0.0417,-0.2021,0.2209,-0.0224],[-0.0000,0.0001,0.0008,-0.0156,0.0368,-0.0095],[-0.0000,0.0000,0.0000,-0.0000,0.0002,-0.0001]],dtype=np.float32)
hpf[:,:,0,4] = np.array([[0.0101,0.0748,0.2033,0.2033,0.0748,0.0101],[-0.0001,-0.0006,-0.0017,-0.0017,-0.0006,-0.0001],[-0.0100,-0.0742,-0.2016,-0.2016,-0.0742,-0.0100],[-0.0100,-0.0742,-0.2016,-0.2016,-0.0742,-0.0100],[-0.0001,-0.0006,-0.0017,-0.0017,-0.0006,-0.0001],[0.0101,0.0748,0.2033,0.2033,0.0748,0.0101]
],dtype=np.float32)*10e-05
hpf[:,:,0,5] = np.array([[1.0e-10*-1.04,1.0e-09*7.71,1.0e-05*1.14,1.0e-05*-2.85,0.0002,-0.0001],[1.0e-06*-2.39,0.0001,0.0008,-0.0156,0.0368,-0.0095],[-0.0014,0.0341,-0.0417,-0.2021,0.2209,-0.0224],[-0.0224,0.2209,-0.2021,-0.0417,0.0341,-0.0014],[-0.0095,0.0368,-0.0156,0.0008,0.0001,1.0e-06*-2.39],[-0.0001,0.0002,1.0e-05*-2.85,1.0e-05*1.14,1.0e-09*7.71,1.0e-10*-1.04]
],dtype=np.float32)
hpf[:,:,0,6] = np.array([[1.0e-13*6.98,1.0e-09*2.24,-0.0001,-0.0032,0.0001,0.0001],[1.0e-09*2.24,-0.0002,-0.0240,0.0018,0.0055,0.0001],[-0.0001,-0.0240,0.0050,0.0404,0.0018,-0.0032],[-0.0032,0.0018,0.0404,0.0050,-0.0240,-0.0001],[0.0001,0.0055,0.0018,-0.0240,-0.0002,1.0e-09*2.24],[0.0001,0.0001,-0.0032,-0.0001,1.0e-09*2.24,1.0e-13*6.98]
],dtype=np.float32)  
hpf[:,:,0,7] = np.array([[1.0e-10*-1.04,1.0e-06*-2.39,-0.0014,-0.0224,-0.0095,-0.0001],[1.0e-09*7.71,0.0001,0.0341,0.2209,0.0368,0.0002],[1.0e-06*1.15,0.0008,-0.0417,-0.2021,-0.0156,1.0e-05*-2.85 ],[1.0e-05*-2.85,-0.0156,-0.2021,-0.0417,0.0008,1.0e-06*1.15 ],[0.0002,0.0368,0.2209,0.0341,0.0001,1.0e-09*7.71],[-0.0001,-0.0095,-0.0224,-0.0014,1.0e-06*-2.39,1.0e-10*-1.04]
],dtype=np.float32)


hpf[:,:,0,8] = np.array([[-0.0151,0.0854,-0.0703,-0.0703,0.0854,-0.0151],[-0.0249,0.1408,-0.1158,-0.1158,0.1408,-0.0249],[-0.0320,0.1807,-0.1488,-0.1488,0.1807,-0.0320],[-0.0320,0.1807,-0.1488,-0.1488,0.1807,-0.0320],[-0.0249,0.1408,-0.1158,-0.1158,0.1408,-0.0249],[-0.0151,0.0854,-0.0703,-0.0703,0.0854,-0.0151]
],dtype=np.float32)
hpf[:,:,0,9] = np.array([[0.0051,0.0195,-0.0690,0.0571,-0.0168,0.0018],[0.1092,-0.4300,0.6676,-0.4315,0.1096,-0.0116],[0.0319,-0.2105,0.5263,-0.5363,0.2123,-0.0347],[-0.0347,0.2123,-0.5363,0.5263,-0.2105,0.0319],[-0.0116,0.1096,-0.4315,0.6676,-0.4300,0.1092],[0.0018,-0.0168,0.0571,-0.0690,0.0195,0.0051]
],dtype=np.float32)
hpf[:,:,0,10] = np.array([[0.2073,-0.2300,0.0540,0.0309,-0.0150,0.0019],[-0.2300,0.5634,-0.4869,0.0891,0.0396,-0.0150],[0.0540,-0.4869,0.9290,-0.6252,0.0891,0.0309],[0.0309,0.0891,-0.6252,0.9290,-0.4869,0.0540],[-0.0150,0.0396,0.0891,-0.4869,0.5634,-0.2300],[0.0019,-0.0150,0.0309,0.0540,-0.2300,0.2073]
],dtype=np.float32)
hpf[:,:,0,11] = np.array([[0.0051,0.1092,0.0319,-0.0347,-0.0116,0.0018],[0.0195,-0.4300,-0.2105,0.2123,0.1096,-0.0168],[-0.0690,0.6676,0.5263,-0.5363,-0.4315,0.0571],[0.0571,-0.4315,-0.5363,0.5263,0.6676,-0.0690],[-0.0168,0.1096,0.2123,-0.2105,-0.4300,0.0195],[0.0018,-0.0116,-0.0347,0.0319,0.1092,0.0051]
],dtype=np.float32)
hpf[:,:,0,12] = np.array([[-0.0151,-0.0249,-0.0320,-0.0320,-0.0249,-0.0151],[0.0854,0.1408,0.1807,0.1807,0.1408,0.0854],[-0.0703,-0.1158,-0.1488,-0.1488,-0.1158,-0.0703],[-0.0703,-0.1158,-0.1488,-0.1488,-0.1158,-0.0703],[0.0854,0.1408,0.1807,0.1807,0.1408,0.0854],[-0.0151,-0.0249,-0.0320,-0.0320,-0.0249,-0.0151]
],dtype=np.float32)
hpf[:,:,0,13] = np.array([[0.0018,-0.0116,-0.0347,0.0319,0.1092,0.0051],[-0.0168,0.1096,0.2123,-0.2105,-0.4300,0.0195],[0.0571,-0.4315,-0.5363,0.5263,0.6676,-0.0690],[-0.0690,0.6676,0.5263,-0.5363,-0.4315,0.0571],[0.0195,-0.4300,-0.2105,0.2123,0.1096,-0.0168],[0.0051,0.1092,0.0319,-0.0347,-0.0116,0.0018]
],dtype=np.float32)
hpf[:,:,0,14] = np.array([[0.0019,-0.0150,0.0309,0.0540,-0.2300,0.2073],[-0.0150,0.0396,0.0891,-0.4869,0.5634,-0.2300],[0.0309,0.0891,-0.6252,0.9290,-0.4869,0.0540],[0.0540,-0.4869,0.9290,-0.6252,0.0891,0.0309],[-0.2300,0.5634,-0.4869,0.0891,0.0396,-0.0150],[0.2073,-0.2300,0.0540,0.0309,-0.0150,0.0019]  
],dtype=np.float32)
hpf[:,:,0,15] = np.array([[0.0018,-0.0168,0.0571,-0.0690,0.0195,0.0051],[-0.0116,0.1096,-0.4315,0.6676,-0.4300,0.1092],[-0.0347,0.2123,-0.5363,0.5263,-0.2105,0.0319],[0.0319,-0.2105,0.5263,-0.5363,0.2123,-0.0347],[0.1092,-0.4300,0.6676,-0.4315,0.1096,-0.0116],[0.0051,0.0195,-0.0690,0.0571,-0.0168,0.0018]],dtype=np.float32)
	
kernel2_0 = tf.Variable(hpf,name="kernel2")
conv2_0 = tf.nn.conv2d(x,kernel2_0,[1,1,1,1],'SAME',name="conv2_0")
# create model 
# using name scope to origanize nodes in the graph visualizer
with tf.variable_scope("model1") as scope:
    with tf.variable_scope("Group1") as scope:
        kernel1 = tf.Variable( tf.random_normal( [5,5,14,24],mean=0.0,stddev=0.01 ),name="kernel1" )  
        conv1 = tf.nn.conv2d(conv1_0, kernel1, [1,1,1,1], padding='SAME',name="conv1"  )
        abs1 = tf.abs(conv1,name="abs1")
        bn1 = slim.layers.batch_norm(abs1,is_training=is_train,updates_collections=None,decay=0.05)
        tanh1 = tf.nn.tanh(bn1,name="tanh1")
        pool1 = tf.nn.avg_pool(tanh1, ksize=[1,5,5,1], strides=[1,2,2,1], padding='SAME',name="pool1" )

    with tf.variable_scope("Group2") as scope:
        kernel2_1 = tf.Variable( tf.random_normal( [5,5,24,32],mean=0.0,stddev=0.01 ),name="kernel2_1")
        conv2_1 = tf.nn.conv2d( pool1, kernel2_1, [1,1,1,1], padding="SAME",name="conv2_1"  )
        bn2_1 = slim.layers.batch_norm(conv2_1,is_training=is_train,updates_collections=None,decay=0.05) 
        tanh2_1 = tf.nn.tanh(bn2_1,name="tanh2_1")
        pool2_1 = tf.nn.avg_pool(tanh2_1, ksize=[1,5,5,1], strides=[1,2,2,1], padding='SAME',name="pool2_1" ) 

        kernel2_2= tf.Variable( tf.random_normal( [5,5,24,32],mean=0.0,stddev=0.01 ),name="kernel2_2")
        conv2_2 = tf.nn.conv2d( pool1, kernel2_2, [1,1,1,1], padding="SAME",name="conv2_2"  )
        bn2_2 = slim.layers.batch_norm(conv2_2,is_training=is_train,updates_collections=None,decay=0.05) 
        tanh2_2 = tf.nn.relu(bn2_2,name="relu2_2")
        pool2_2 = tf.nn.avg_pool(tanh2_2, ksize=[1,5,5,1], strides=[1,2,2,1], padding='SAME',name="pool2_2" )

        kernel2_3= tf.Variable( tf.random_normal( [5,5,24,32],mean=0.0,stddev=0.01 ),name="kernel2_3")
        conv2_3 = tf.nn.conv2d( pool1, kernel2_3, [1,1,1,1], padding="SAME",name="conv2_3"  )
        bn2_3 = slim.layers.batch_norm(conv2_3,is_training=is_train,updates_collections=None,decay=0.05) 
        tanh2_3 = tf.sigmoid(bn2_3,name="sigmoid2_3")
        pool2_3 = tf.nn.avg_pool(tanh2_3, ksize=[1,5,5,1], strides=[1,2,2,1], padding='SAME',name="pool2_2" )

        pool2 = tf.concat([pool2_1,pool2_2,pool2_3],3)
    with tf.variable_scope("Group3") as scope:
        kernel3 = tf.Variable( tf.random_normal( [1,1,96,64],mean=0.0,stddev=0.01 ),name="kernel3" )
        conv3 = tf.nn.conv2d( pool2, kernel3, [1,1,1,1], padding="SAME",name="conv3"  )
        bn3 = slim.layers.batch_norm(conv3,is_training=is_train,updates_collections=None,decay=0.05)  
        relu3 = tf.nn.relu(bn3,name="bn3")
        pool3 = tf.nn.avg_pool(relu3, ksize=[1,5,5,1], strides=[1,2,2,1], padding="SAME",name="pool3" ) 

    with tf.variable_scope("Group4") as scope:
        kernel4_1 = tf.Variable( tf.random_normal([3,3,64,96],mean=0.0,stddev=0.01 ),name="kernel4_1" )
        conv4_1 = tf.nn.conv2d( pool3, kernel4_1, [1,1,1,1], padding="SAME",name="conv4_1"  )
        bn4_1 = slim.layers.batch_norm(conv4_1,is_training=is_train,updates_collections=None,decay=0.05)       
        relu4_1 = tf.nn.relu(bn4_1,name="relu4_1")
        pool4_1 = tf.nn.avg_pool(relu4_1, ksize=[1,5,5,1], strides=[1,2,2,1], padding="SAME",name="pool4_1" ) 

        kernel4_2 = tf.Variable( tf.random_normal( [3,3,64,96],mean=0.0,stddev=0.01 ),name="kernel4_2" )
        conv4_2 = tf.nn.conv2d( pool3, kernel4_2, [1,1,1,1], padding="SAME",name="conv4_2"  )
        bn4_2 = slim.layers.batch_norm(conv4_2,is_training=is_train,updates_collections=None,decay=0.05)       
        relu4_2 = tf.nn.tanh(bn4_2,name="tanh4_2")
        pool4_2 = tf.nn.avg_pool(relu4_2, ksize=[1,5,5,1], strides=[1,2,2,1], padding="SAME",name="pool4_2" ) 

        kernel4_3 = tf.Variable( tf.random_normal( [3,3,64,96],mean=0.0,stddev=0.01 ),name="kernel4_3" )
        conv4_3 = tf.nn.conv2d( pool3, kernel4_3, [1,1,1,1], padding="SAME",name="conv4_3"  )
        bn4_3 = slim.layers.batch_norm(conv4_3,is_training=is_train,updates_collections=None,decay=0.05)       
        relu4_3 = tf.sigmoid(bn4_3,name="sigmoid4_3")
        pool4_3 = tf.nn.avg_pool(relu4_3, ksize=[1,5,5,1], strides=[1,2,2,1], padding="SAME",name="pool4_3" ) 

        pool4 = tf.concat([pool4_1,pool4_2,pool4_3],3)
    with tf.variable_scope("Group5") as scope:
        kernel5 = tf.Variable( tf.random_normal( [1,1,288,256],mean=0.0,stddev=0.01 ),name="kernel5" )
        conv5 = tf.nn.conv2d( pool4, kernel5, [1,1,1,1], padding="SAME",name="conv5"  )
        bn5 = slim.layers.batch_norm(conv5,is_training=is_train,updates_collections=None,decay=0.05)      
        relu5_1 = tf.nn.relu(bn5,name="relu5")
        pool5_1 = tf.nn.avg_pool(relu5_1, ksize=[1,32,32,1], strides=[1,1,1,1], padding="VALID",name="pool5" ) 

    with tf.variable_scope('Group6') as scope:
        pool_shape = pool5_1.get_shape().as_list()
        pool_reshape = tf.reshape( pool5_1, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        weights = tf.Variable( tf.random_normal( [256,2],mean=0.0,stddev=0.01 ),name="weights" )
        bias = tf.Variable( tf.random_normal([2],mean=0.0,stddev=0.01),name="bias" )
        y_ = tf.matmul(pool_reshape, weights) + bias 


with tf.variable_scope("model2") as scope:
    with tf.variable_scope("Group1") as scope:
        kernel1 = tf.Variable( tf.random_normal( [5,5,16,24],mean=0.0,stddev=0.01 ),name="kernel1" )  
        conv1 = tf.nn.conv2d(conv2_0, kernel1, [1,1,1,1], padding='SAME',name="conv1"  )
        abs1 = tf.abs(conv1,name="abs1")
        bn1 = slim.layers.batch_norm(abs1,is_training=is_train,updates_collections=None,decay=0.05)
        tanh1 = tf.nn.tanh(bn1,name="tanh1")
        pool1 = tf.nn.avg_pool(tanh1, ksize=[1,5,5,1], strides=[1,2,2,1], padding='SAME',name="pool1" )

    with tf.variable_scope("Group2") as scope:
        kernel2_1 = tf.Variable( tf.random_normal( [5,5,24,32],mean=0.0,stddev=0.01 ),name="kernel2_1")
        conv2_1 = tf.nn.conv2d( pool1, kernel2_1, [1,1,1,1], padding="SAME",name="conv2_1"  )
        bn2_1 = slim.layers.batch_norm(conv2_1,is_training=is_train,updates_collections=None,decay=0.05) 
        tanh2_1 = tf.nn.tanh(bn2_1,name="tanh2_1")
        pool2_1 = tf.nn.avg_pool(tanh2_1, ksize=[1,5,5,1], strides=[1,2,2,1], padding='SAME',name="pool2_1" ) 

        kernel2_2= tf.Variable( tf.random_normal( [5,5,24,32],mean=0.0,stddev=0.01 ),name="kernel2_2")
        conv2_2 = tf.nn.conv2d( pool1, kernel2_2, [1,1,1,1], padding="SAME",name="conv2_2"  )
        bn2_2 = slim.layers.batch_norm(conv2_2,is_training=is_train,updates_collections=None,decay=0.05) 
        tanh2_2 = tf.nn.relu(bn2_2,name="relu2_2")
        pool2_2 = tf.nn.avg_pool(tanh2_2, ksize=[1,5,5,1], strides=[1,2,2,1], padding='SAME',name="pool2_2" )

        kernel2_3= tf.Variable( tf.random_normal( [5,5,24,32],mean=0.0,stddev=0.01 ),name="kernel2_3")
        conv2_3 = tf.nn.conv2d( pool1, kernel2_3, [1,1,1,1], padding="SAME",name="conv2_3"  )
        bn2_3 = slim.layers.batch_norm(conv2_3,is_training=is_train,updates_collections=None,decay=0.05) 
        tanh2_3 = tf.sigmoid(bn2_3,name="sigmoid2_3")
        pool2_3 = tf.nn.avg_pool(tanh2_3, ksize=[1,5,5,1], strides=[1,2,2,1], padding='SAME',name="pool2_2" )

        pool2 = tf.concat([pool2_1,pool2_2,pool2_3],3)
        #pool2 = tf.dropout(pool2,0.5)
    with tf.variable_scope("Group3") as scope:
        kernel3 = tf.Variable( tf.random_normal( [1,1,96,64],mean=0.0,stddev=0.01 ),name="kernel3" )
        conv3 = tf.nn.conv2d( pool2, kernel3, [1,1,1,1], padding="SAME",name="conv3"  )
        bn3 = slim.layers.batch_norm(conv3,is_training=is_train,updates_collections=None,decay=0.05)  
        relu3 = tf.nn.relu(bn3,name="bn3")
        pool3 = tf.nn.avg_pool(relu3, ksize=[1,5,5,1], strides=[1,2,2,1], padding="SAME",name="pool3" ) 

    with tf.variable_scope("Group4") as scope:
        kernel4_1 = tf.Variable( tf.random_normal([3,3,64,96],mean=0.0,stddev=0.01 ),name="kernel4_1" )
        conv4_1 = tf.nn.conv2d( pool3, kernel4_1, [1,1,1,1], padding="SAME",name="conv4_1"  )
        bn4_1 = slim.layers.batch_norm(conv4_1,is_training=is_train,updates_collections=None,decay=0.05)       
        relu4_1 = tf.nn.relu(bn4_1,name="relu4_1")
        pool4_1 = tf.nn.avg_pool(relu4_1, ksize=[1,5,5,1], strides=[1,2,2,1], padding="SAME",name="pool4_1" ) 

        kernel4_2 = tf.Variable( tf.random_normal( [3,3,64,96],mean=0.0,stddev=0.01 ),name="kernel4_2" )
        conv4_2 = tf.nn.conv2d( pool3, kernel4_2, [1,1,1,1], padding="SAME",name="conv4_2"  )
        bn4_2 = slim.layers.batch_norm(conv4_2,is_training=is_train,updates_collections=None,decay=0.05)       
        relu4_2 = tf.nn.tanh(bn4_2,name="tanh4_2")
        pool4_2 = tf.nn.avg_pool(relu4_2, ksize=[1,5,5,1], strides=[1,2,2,1], padding="SAME",name="pool4_2" ) 

        kernel4_3 = tf.Variable( tf.random_normal( [3,3,64,96],mean=0.0,stddev=0.01 ),name="kernel4_3" )
        conv4_3 = tf.nn.conv2d( pool3, kernel4_3, [1,1,1,1], padding="SAME",name="conv4_3"  )
        bn4_3 = slim.layers.batch_norm(conv4_3,is_training=is_train,updates_collections=None,decay=0.05)       
        relu4_3 = tf.sigmoid(bn4_3,name="sigmoid4_3")
        pool4_3 = tf.nn.avg_pool(relu4_3, ksize=[1,5,5,1], strides=[1,2,2,1], padding="SAME",name="pool4_3" ) 

        pool4 = tf.concat([pool4_1,pool4_2,pool4_3],3)
    with tf.variable_scope("Group5") as scope:
        kernel5 = tf.Variable( tf.random_normal( [1,1,288,256],mean=0.0,stddev=0.01 ),name="kernel5" )
        conv5 = tf.nn.conv2d( pool4, kernel5, [1,1,1,1], padding="SAME",name="conv5"  )
        bn5 = slim.layers.batch_norm(conv5,is_training=is_train,updates_collections=None,decay=0.05)      
        relu5_2 = tf.nn.relu(bn5,name="relu5")
        pool5_2 = tf.nn.avg_pool(relu5_2, ksize=[1,32,32,1], strides=[1,1,1,1], padding="VALID",name="pool5" ) 

    with tf.variable_scope('Group6') as scope:
        pool_shape = pool5_2.get_shape().as_list()
        pool_reshape = tf.reshape( pool5_2, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        weights = tf.Variable( tf.random_normal( [256,2],mean=0.0,stddev=0.01 ),name="weights" )
        bias = tf.Variable( tf.random_normal([2],mean=0.0,stddev=0.01),name="bias" )
        y_ = tf.matmul(pool_reshape, weights) + bias 


with tf.variable_scope("model3") as scope:
    with tf.variable_scope("Group1") as scope:
        kernel1 = tf.Variable( tf.random_normal( [5,5,14,24],mean=0.0,stddev=0.01 ),name="kernel1" )  
        conv1 = tf.nn.conv2d(conv0_0, kernel1, [1,1,1,1], padding='SAME',name="conv1"  )
        abs1 = tf.abs(conv1,name="abs1")
        bn1 = slim.layers.batch_norm(abs1,is_training=is_train,updates_collections=None,decay=0.05)
        tanh1 = tf.nn.tanh(bn1,name="tanh1")
        pool1 = tf.nn.avg_pool(tanh1, ksize=[1,5,5,1], strides=[1,2,2,1], padding='SAME',name="pool1" )


    with tf.variable_scope("Group2") as scope:
        kernel2_1 = tf.Variable( tf.random_normal( [5,5,24,32],mean=0.0,stddev=0.01 ),name="kernel2_1")
        conv2_1 = tf.nn.conv2d( pool1, kernel2_1, [1,1,1,1], padding="SAME",name="conv2_1"  )
        bn2_1 = slim.layers.batch_norm(conv2_1,is_training=is_train,updates_collections=None,decay=0.05) 
        tanh2_1 = tf.nn.tanh(bn2_1,name="tanh2_1")
        pool2_1 = tf.nn.avg_pool(tanh2_1, ksize=[1,5,5,1], strides=[1,2,2,1], padding='SAME',name="pool2_1" ) 

        kernel2_2= tf.Variable( tf.random_normal( [5,5,24,32],mean=0.0,stddev=0.01 ),name="kernel2_2")
        conv2_2 = tf.nn.conv2d( pool1, kernel2_2, [1,1,1,1], padding="SAME",name="conv2_2"  )
        bn2_2 = slim.layers.batch_norm(conv2_2,is_training=is_train,updates_collections=None,decay=0.05) 
        tanh2_2 = tf.nn.relu(bn2_2,name="relu2_2")
        pool2_2 = tf.nn.avg_pool(tanh2_2, ksize=[1,5,5,1], strides=[1,2,2,1], padding='SAME',name="pool2_2" )

        kernel2_3= tf.Variable( tf.random_normal( [5,5,24,32],mean=0.0,stddev=0.01 ),name="kernel2_3")
        conv2_3 = tf.nn.conv2d( pool1, kernel2_3, [1,1,1,1], padding="SAME",name="conv2_3"  )
        bn2_3 = slim.layers.batch_norm(conv2_3,is_training=is_train,updates_collections=None,decay=0.05) 
        tanh2_3 = tf.sigmoid(bn2_3,name="sigmoid2_3")
        pool2_3 = tf.nn.avg_pool(tanh2_3, ksize=[1,5,5,1], strides=[1,2,2,1], padding='SAME',name="pool2_2" )

        pool2 = tf.concat([pool2_1,pool2_2,pool2_3],3)
        #pool2 = tf.dropout(pool2,0.5)
    with tf.variable_scope("Group3") as scope:
        kernel3 = tf.Variable( tf.random_normal( [1,1,96,64],mean=0.0,stddev=0.01 ),name="kernel3" )
        conv3 = tf.nn.conv2d( pool2, kernel3, [1,1,1,1], padding="SAME",name="conv3"  )
        bn3 = slim.layers.batch_norm(conv3,is_training=is_train,updates_collections=None,decay=0.05)  
        relu3 = tf.nn.relu(bn3,name="bn3")
        pool3 = tf.nn.avg_pool(relu3, ksize=[1,5,5,1], strides=[1,2,2,1], padding="SAME",name="pool3" ) 

    with tf.variable_scope("Group4") as scope:
        kernel4_1 = tf.Variable( tf.random_normal([3,3,64,96],mean=0.0,stddev=0.01 ),name="kernel4_1" )
        conv4_1 = tf.nn.conv2d( pool3, kernel4_1, [1,1,1,1], padding="SAME",name="conv4_1"  )
        bn4_1 = slim.layers.batch_norm(conv4_1,is_training=is_train,updates_collections=None,decay=0.05)       
        relu4_1 = tf.nn.relu(bn4_1,name="relu4_1")
        pool4_1 = tf.nn.avg_pool(relu4_1, ksize=[1,5,5,1], strides=[1,2,2,1], padding="SAME",name="pool4_1" ) 

        kernel4_2 = tf.Variable( tf.random_normal( [3,3,64,96],mean=0.0,stddev=0.01 ),name="kernel4_2" )
        conv4_2 = tf.nn.conv2d( pool3, kernel4_2, [1,1,1,1], padding="SAME",name="conv4_2"  )
        bn4_2 = slim.layers.batch_norm(conv4_2,is_training=is_train,updates_collections=None,decay=0.05)       
        relu4_2 = tf.nn.tanh(bn4_2,name="tanh4_2")
        pool4_2 = tf.nn.avg_pool(relu4_2, ksize=[1,5,5,1], strides=[1,2,2,1], padding="SAME",name="pool4_2" ) 

        kernel4_3 = tf.Variable( tf.random_normal( [3,3,64,96],mean=0.0,stddev=0.01 ),name="kernel4_3" )
        conv4_3 = tf.nn.conv2d( pool3, kernel4_3, [1,1,1,1], padding="SAME",name="conv4_3"  )
        bn4_3 = slim.layers.batch_norm(conv4_3,is_training=is_train,updates_collections=None,decay=0.05)       
        relu4_3 = tf.sigmoid(bn4_3,name="sigmoid4_3")
        pool4_3 = tf.nn.avg_pool(relu4_3, ksize=[1,5,5,1], strides=[1,2,2,1], padding="SAME",name="pool4_3" ) 

        pool4 = tf.concat([pool4_1,pool4_2,pool4_3],3)
    with tf.variable_scope("Group5") as scope:
        kernel5 = tf.Variable( tf.random_normal( [1,1,288,256],mean=0.0,stddev=0.01 ),name="kernel5" )
        conv5 = tf.nn.conv2d( pool4, kernel5, [1,1,1,1], padding="SAME",name="conv5"  )
        bn5 = slim.layers.batch_norm(conv5,is_training=is_train,updates_collections=None,decay=0.05)      
        relu5_3 = tf.nn.relu(bn5,name="relu5")
        pool5_3 = tf.nn.avg_pool(relu5_3, ksize=[1,32,32,1], strides=[1,1,1,1], padding="VALID",name="pool5" ) 


    with tf.variable_scope('Group6') as scope:
        pool_shape = pool5_3.get_shape().as_list()
        pool_reshape = tf.reshape( pool5_3, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        weights = tf.Variable( tf.random_normal( [256,2],mean=0.0,stddev=0.01 ),name="weights" )
        bias = tf.Variable( tf.random_normal([2],mean=0.0,stddev=0.01),name="bias" )
        y_ = tf.matmul(pool_reshape, weights) + bias 

with tf.variable_scope("model4") as scope:
    with tf.variable_scope('all_fullconnected') as scope:
        full = tf.concat([pool5_1,pool5_2,pool5_3],3) 
        full_shape = full.get_shape().as_list()
        full_reshape = tf.reshape( full, [full_shape[0], full_shape[1] * full_shape[2] * full_shape[3]])
        weights4_1 = tf.Variable( tf.random_normal( [768,2],mean=0.0,stddev=0.01 ),name="weights" )
        bias4_1 = tf.Variable( tf.random_normal([2],mean=0.0,stddev=0.01),name="bias" )
        final = tf.matmul(full_reshape, weights4_1) + bias4_1 
		
# save weights	
vars = tf.trainable_variables()
params1 = [v for v in vars if ( v.name.startswith('model1/') ) ]
params2 = [v for v in vars if ( v.name.startswith('model2/') ) ]
params3 = [v for v in vars if ( v.name.startswith('model3/') ) ]
params4 = [v for v in vars if ( v.name.startswith('model4/') ) ]

# define loss and optimizer
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(final,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( labels=y,logits=final))
global_step = tf.Variable(0,trainable = False)
decayed_learning_rate=tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEP, LEARNING_RATE_DECAY, staircase=True)
opt = tf.train.MomentumOptimizer(decayed_learning_rate,MOMENTUM).minimize(loss,var_list=params1)
# add summary ops to collect data
tf.summary.scalar('acc',accuracy)
tf.summary.scalar('loss',loss)

# input images and labels
data_x = np.zeros([BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNEL])
data_y = np.zeros([BATCH_SIZE,NUM_LABELS])
bs = 10
for i in range(0,bs):
    data_y[i,1] = 1
for i in range(bs,BATCH_SIZE):
    data_y[i,0] = 1


saver = tf.train.Saver()
merged = tf.summary.merge_all()

with tf.Session() as sess:
    # merge all the summaries and write them out to tmp/ReST-Net_logs
    writer = tf.summary.FileWriter("tmp/ReST-Net_logs/ReST-Net_s40_180326",sess.graph)
	# initialized all the weights 
    tf.global_variables_initializer().run()
	# restore the subnet2
    tf.train.Saver(params2).restore(sess,saver_path+"/saver/subnet1.ckpt")
    tf.train.Saver(params1).restore(sess,saver_path+"/saver/subnet2.ckpt")
    tf.train.Saver(params3).restore(sess,saver_path+"/saver/subnet3.ckpt")
    summary = tf.Summary()	
    count = 0
	# load data and train the model 
    list = [h for h in range(4000)]
    for i in range(1,NUM_ITER+1):
        for j in range(bs):
            if count%4000==0:
                count = count%4000
                random.seed(i)
                random.shuffle(list)
            cover=ndimage.imread(path1+'/'+fileList[list[count]])   
            stego=ndimage.imread(path2+'/'+fileList[list[count]])   
            data_x[j,:,:,0] = cover.astype(np.float32)
            data_x[j+bs,:,:,0] = stego.astype(np.float32)           
            count = count+1
             
        _,temp,l = sess.run([opt,accuracy,loss],feed_dict={x:data_x,y:data_y,is_train:True})
		# record summaries every 10 iters        
        if i%10==0:  
            summary.ParseFromString(sess.run(merged,feed_dict={x:data_x,y:data_y,is_train:True}))
            writer.add_summary(summary, i)
		# show the training loss and accuracy every epoch
        if i%50==0:  
            print ('ReST-Net_s40: batch result')
            print ('epoch:', i)
            print ('loss:', l)
            print ('accuracy:', temp)
            print (' ')

		# validation data every 2 epoches				
        if i%NUM_SHOWVALIDATION==0:
            result1 = np.array([]) #accuracy for training set
            num = i/NUM_SHOWTEST - 1
            val_count = 0
            while val_count<4000:
                for j in range(bs):
                    cover=ndimage.imread(path1+'/'+fileList[val_count])   
                    stego=ndimage.imread(path2+'/'+fileList[val_count])   
                    data_x[j,:,:,0] = cover.astype(np.float32)
                    data_x[j+bs,:,:,0] = stego.astype(np.float32)           
                    val_count = val_count+1
                c1,temp1 = sess.run([loss,accuracy],feed_dict={x:data_x,y:data_y,is_train:False})
                result1 = np.insert(result1,0,temp1)
            summary.value.add(tag='val_acc', simple_value=np.mean(result1))
            writer.add_summary(summary, i)
            print ('validation accuracy:', np.mean(result1))

		# save the model every 4 epoches
        if i%NUM_SAVE==0:
            saver = tf.train.Saver()
            saver.save(sess,saver_path+'/saver/ReST-Net_saver/ReST-Net_s40_'+str(i)+'.ckpt')  			
"""
	
	# test data when the validation accuracy is the best	
	saver.restore(sess,saver_path+"/saver/ReST-Net.ckpt")
            test_result = np.array([]) #accuracy for testing set
            test_count = 5000
            while test_count<10000:
                for j in range(bs):
                    cover=ndimage.imread(path1+'/'+fileList[test_count])   
                    stego=ndimage.imread(path2+'/'+fileList[test_count])   
                    data_x[j,:,:,0] = cover.astype(np.float32)
                    data_x[j+bs,:,:,0] = stego.astype(np.float32)           
                    test_count = test_count+1
                c2,temp2 = sess.run([loss,accuracy],feed_dict={x:data_x,y:data_y,is_train:False})
                test_result= np.insert(test_result,0,temp2)
            print ('Testing :', np.mean(result2))
            print (' ')
"""


