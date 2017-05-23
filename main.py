import cv2
import tensorflow as tf
import os
import utils
import model
import numpy as np
from scipy import misc
from settings import *

data, output_dimension, label = utils.get_dataset(location)

sess = tf.InteractiveSession()

model = model.Model(picture_dimension, learning_rate, output_dimension, enable_dropout, dropout_probability, enable_penalty, penalty)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())

try:
    saver.restore(sess, current_location + "/model.ckpt")
    print "load model.."
except:
    print "need to train first, exiting.."
    exit(0)
    
cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('/home/project/detectface/haarcascade_frontalface_default.xml')

fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

count = 0

while(True):

    ret, frame = cap.read()
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30), flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    
    for (x, y, w, h) in faces:
        
        img = gray[y: y + h, x: x + w]
        img = misc.imresize(img, (picture_dimension, picture_dimension))
        img = img.reshape([img.shape[0], img.shape[1], 1])
 
        emb_data = np.zeros((1, picture_dimension, picture_dimension, 1), dtype = np.float32)
        
        emb_data[0, :, :, :] = img
        
        if count % 10 == 0:
            prob = sess.run(tf.nn.softmax(model.y_hat), feed_dict = {model.X : emb_data})
            count = 0

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        try:
            for i in xrange(prob.shape[1]):
                cv2.putText(frame, label[i] + ': ' + str(prob[0][i]), (x, y + ((i + 1) * 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
                
        except:
            print 'not found any faces'
    
    out.write(frame)
    
    cv2.imshow('Video', frame)
    
    count += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()