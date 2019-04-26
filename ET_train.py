"""
Created on Sun Mar 24 17:39:43 2019


This code is for the SDPII project


@author: AbdAlla Hefny
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
import os
import sys
import configparser
from conv_utils import read_dataset, create_one_hot, initialize_parameters, forward_propagation, shuffle_dataset

config = configparser.ConfigParser()
config.read('ET_config.ini')
user_name = config['End User Section']['name of the user directory']
examples_per_class_train = int(config['End User Section']['number of frames for training'])
examples_per_class_test = int(config['End User Section']['number of frames for testing'])
plot_f = int(config['End User Section']['plot flag'])
print_f = int(config['End User Section']['print flag'])
display_step = int(config['End User Section']['display step'])
plot_flag = True
print_cost = True
if not (plot_f):
    plot_flag = False
if not(print_f):
    print_cost = False
# Training Parameters
learning_rate = 0.001
num_steps = 100  # epochs
#batch_size = 1  # No batches



# Input & Output Parameters
n_H0 = 64
n_W0 = 64
n_C0 = 3
classes = ['right', 'forward', 'left', 'closed']
# Model Architecture
num_conv_layers = 2
filters=[3,3]
channels=[16,12]
strides= [1, 1]
pools = [4, 4]
h_layer_n = 16  # number of neurons in the hidden FC layer


# Make sure that the training dataset exists
# If not, the program terminates
newpath = user_name + '/Reye'
if not os.path.exists(newpath):
    print("You should first have a training dataset for this user....")
    print("Please run ET_daq.py file to collect the dataset....")
    sys.exit()
if not os.listdir(newpath) :
    print("Directory containig training dataset is empty...")
    print("The dataset may be deleted or placed in another directory....")
    print("Please place the dataset in the correct path, or you may run the ET_daq.py file to re-collect the dataset....")
    sys.exit()

# If the dataset exists, create directory to save the model     
newpath = user_name + '/Reye/model_param'
if not os.path.exists(newpath):
    os.makedirs(newpath)    
    
## setup the training dataset and labels
num_classes = len(classes)
# Load the training and testing dataset of the user 
X_train, X_test = read_dataset(user_name, examples_per_class_train, examples_per_class_test, classes)
# create one-hot vector for the training dataset
Y_train, Y_test = create_one_hot(examples_per_class_train, examples_per_class_test)

## setup tensorflow graph                                      
ops.reset_default_graph()
# create placeholders 
X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0), name = "X")
Y = tf.placeholder(tf.float32, shape=(None, num_classes), name = "Y")
# initialize training parameters
parameters = initialize_parameters(num_conv_layers, n_C0, filters, channels)
# Define the forward pass in the tensorflow graph
Z= forward_propagation(X, parameters, num_conv_layers, strides, pools, h_layer_n, num_classes)
# Compute cost: Add cost function to tensorflow graph
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y))
# Define Tensorflow Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
# Evaluate Model
predict_op = tf.argmax(Z, 1)
correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))   
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
saver = tf.train.Saver()
# Initialize all the variables globally
init = tf.global_variables_initializer()

costs =[]
# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    for step in range(num_steps):
        # shuffle the training set for each new epoch
        input_x, input_y = shuffle_dataset(X_train, Y_train)
        #print(type(batch_x))
        _, batch_cost = sess.run([optimizer, cost], feed_dict={X: input_x, Y: input_y})
        if print_cost == True and step % display_step == 0:
            costs.append(batch_cost)
            print ("After step %i, Cost: %f" % (step, batch_cost))
    print("Optimization Finished!")
    saver.save(sess, './' + user_name + '/model_param/my_sdp_model')
    # plot the cost if required
    if (plot_flag):
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
    # Calculate accuracy for training and testing sets 
    #X_test, Y_test = shuffle_dataset(X_test, Y_test)
    print("Training Accuracy:", \
        sess.run(accuracy, feed_dict={X: X_train ,Y: Y_train}))
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: X_test ,Y: Y_test}))

    