"""
Created on Sun Mar 24 17:39:43 2019


This code is for the SDPII project


@author: AbdAlla Hefny
"""
import numpy as np
import tensorflow as tf
import cv2
import time
import configparser


config = configparser.ConfigParser()
config.read('ET_config.ini')
user_name = config['End User Section']['name of the user directory']
examples_per_class_train = int(config['End User Section']['number of frames for training'])
examples_per_class_test = int(config['End User Section']['number of frames for testing'])
no_frames = int(config['End User Section']['number of frames for prediction'])

classes = ['right', 'forward', 'left', 'closed']


cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 64)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 64)
#cap.set(cv2.CAP_PROP_FPS, 60)

with tf.Session() as sess:

    new_saver = tf.train.import_meta_graph('./'+str(user_name)+'/model_param/my_sdp_model.meta')
    new_saver.restore(sess, './'+str(user_name)+'/model_param/my_sdp_model')
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X:0")
    Y = graph.get_tensor_by_name("Y:0")
    model_prediction = graph.get_tensor_by_name("fully_connected_1/BiasAdd:0")
    predict = tf.nn.softmax(model_prediction)
    
    predictions = []
    start = time.time()   
    for i in range (0,no_frames):
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        pre_X = cv2.resize(frame,(64,64), interpolation = cv2.INTER_AREA)/255.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        pre_X = np.expand_dims(pre_X, 0)
        pre= sess.run(predict, feed_dict={X: pre_X})
        #print("predicted class: "+str(classes[pre.argmax()]))
        predictions.append(classes[pre.argmax()])
    end = time.time()
    cap.release()
    cv2.destroyAllWindows()
    elapsed = end-start
    fp = no_frames/ elapsed
    print("prediction fps: "+ str(fp))
    

        
