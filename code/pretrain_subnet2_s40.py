# import some toolkits
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import os
import random
from scipy import ndimage
from pandas import DataFrame


# setup 
BATCH_SIZE = 40
IMAGE_SIZE = 512
NUM_CHANNEL = 1
NUM_LABELS = 2
NUM_ITER =200000
NUM_SAVE =5000
NUM_SHOWTRAIN = 200 #show result eveary epoch 
NUM_SHOWVALIDATION = 5000
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
is_train = tf.placeholder(tf.bool,name='IS_TRAIN')



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

# create model 
# using name scope to origanize nodes in the graph visualizer
with tf.variable_scope("model1") as scope:
    with tf.variable_scope("Group1") as scope:
        kernel1 = tf.Variable( tf.random_normal( [5,5,14,24],mean=0.0,stddev=0.01 ),name="kernel1" )  
        conv1 = tf.nn.conv2d(conv1_0, kernel1, [1,1,1,1], padding='SAME',name="conv1"  )
        abs1 = tf.abs(conv1,name="abs")
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
        
		#concat
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

		#concat
        pool4 = tf.concat([pool4_1,pool4_2,pool4_3],3)
    with tf.variable_scope("Group5") as scope:
        kernel5 = tf.Variable( tf.random_normal( [1,1,288,256],mean=0.0,stddev=0.01 ),name="kernel5" )
        conv5 = tf.nn.conv2d( pool4, kernel5, [1,1,1,1], padding="SAME",name="conv5"  )
        bn5 = slim.layers.batch_norm(conv5,is_training=is_train,updates_collections=None,decay=0.05)      
        relu5_1 = tf.nn.relu(bn5,name="relu5")
        pool5_1 = tf.nn.avg_pool(relu5_1, ksize=[1,32,32,1], strides=[1,1,1,1], padding="VALID",name="pool5" ) 

    with tf.variable_scope('Group6') as scope:
	    #fully-connected
        pool_shape = pool5_1.get_shape().as_list()
        pool_reshape = tf.reshape( pool5_1, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        weights = tf.Variable( tf.random_normal( [256,2],mean=0.0,stddev=0.01 ),name="weights" )
        bias = tf.Variable( tf.random_normal([2],mean=0.0,stddev=0.01),name="bias" )
        y_ = tf.matmul(pool_reshape, weights) + bias 
		
# save weights		
vars = tf.trainable_variables()
params1 = [v for v in vars if ( v.name.startswith('model1/') ) ]

# define loss and optimizer
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( labels=y,logits=y_))
global_step = tf.Variable(0,trainable = False)
decayed_learning_rate=tf.train.exponential_decay(LEARNING_RATE, global_step, DECAY_STEP, LEARNING_RATE_DECAY, staircase=True)
opt = tf.train.MomentumOptimizer(decayed_learning_rate,MOMENTUM).minimize(loss,var_list=params1)
# add summary ops to collect data
tf.summary.scalar('acc',accuracy)
tf.summary.scalar('loss',loss)


# input images and labels
data_x = np.zeros([BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNEL])
data_y = np.zeros([BATCH_SIZE,NUM_LABELS])
bs = 20
for i in range(0,bs):
    data_y[i,1] = 1
for i in range(bs,BATCH_SIZE):
    data_y[i,0] = 1


saver = tf.train.Saver()
merged = tf.summary.merge_all()


with tf.Session() as sess:
    # merge all the summaries and write them out to tmp/subnet2_logs
    writer = tf.summary.FileWriter("tmp/subnet2_logs/subnet2_s40_180326",sess.graph)
	# initialized all the weights 
    tf.global_variables_initializer().run()
	# restore the subnet2
    #tf.train.Saver(params1).restore(sess,saver_path+"/saver/subnet2.ckpt")	
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
        
		# record summaries every 100 iters
        if i%100==0:  
            summary.ParseFromString(sess.run(merged,feed_dict={x:data_x,y:data_y,is_train:True}))
            writer.add_summary(summary, i)
		# show the training loss and accuracy every epoch
        if i%NUM_SHOWTRAIN==0:  
            print ('subnet2_s40: batch result')
            print ('epoch:', i)
            print ('loss:', l)
            print ('accuracy:', temp)
            print (' ')
		# validation data every 25 epoches	
        if i%NUM_SHOWVALIDATION==0:
            result1 = np.array([]) #accuracy for training set
            num = i/NUM_SHOWTEST - 1
            val_count = 4000
            while val_count<5000:
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
         
		# save the model every 50 epoches
        if i%NUM_SAVE==0:
            saver = tf.train.Saver()
            saver.save(sess,saver_path+'/saver/subnet2_saver/subnet2_s40_'+str(i)+'.ckpt')  

