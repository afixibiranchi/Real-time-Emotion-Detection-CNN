import tensorflow as tf
import numpy as np
from scipy import misc
import model
import utils
import graph
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from settings import *

data, output_dimension, label = utils.get_dataset(location)
print label

sess = tf.InteractiveSession()

model = model.Model(picture_dimension, learning_rate, output_dimension, enable_dropout, dropout_probability, enable_penalty, penalty)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())

try:
    saver.restore(sess, current_location + "/model.ckpt")
    print "load model.."
except:
    print "start from fresh variables"
        

ACCURACY = []; EPOCH = []; LOST = []
    
for i in xrange(epoch):
    total_cost = 0
    total_accuracy = 0
    last_time = time.time()
    EPOCH.append(i)
        
    for k in xrange(0, data.shape[0] - batch_size, batch_size):
            
        emb_data = np.zeros((batch_size, picture_dimension, picture_dimension, 1), dtype = np.float32)
        emb_data_label = np.zeros((batch_size, output_dimension), dtype = np.float32)
            
        for x in xrange(batch_size):
                
            image = misc.imread(location + data[k + x, 0])
            image = misc.imresize(image, (picture_dimension, picture_dimension))
                
            if len(image.shape) > 2:
                image = np.mean(image, -1)
                
            image = image.reshape([image.shape[0], image.shape[1], 1])
            emb_data_label[x, int(data[k + x, 1])] = 1.0
            emb_data[x, :] = image
           
        _, loss = sess.run([model.optimizer, model.cost], feed_dict = {model.X : emb_data, model.Y : emb_data_label})
        accuracy = sess.run(model.accuracy, feed_dict = {model.X : emb_data, model.Y : emb_data_label})
        total_cost += loss
        total_accuracy += accuracy
        
        
    accuracy = total_accuracy / ((data.shape[0] - batch_size) / (batch_size * 1.0))
    loss = total_cost / ((data.shape[0] - batch_size) / (batch_size * 1.0))
    ACCURACY.append(accuracy)
    LOST.append(loss)
        
    print "epoch: " + str(i + 1) + ", loss: " + str(loss) + ", accuracy: " + str(accuracy) + ", s / epoch: " + str(time.time() - last_time)
    graph.generategraph(EPOCH, ACCURACY, LOST)
    saver.save(sess, current_location + "/model.ckpt")