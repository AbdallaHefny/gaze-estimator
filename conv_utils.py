import tensorflow as tf
import numpy as np
import cv2

def create_one_hot(examples_perclass_train, examples_perclass_test):
    """
    Creates one-hot vector Y from the images for both traing and testing datasets
    Assumes that total number of classses = 4
    
    Arguments:
    examples_perclass_train -- number of images per class in the train dataset
    examples_perclass_test -- number of images per class in the test dataset
     
    Returns:
    Y_train -- encoded one-hot vector for train set
    Y_test -- encoded one-hot vector for test set
    """    
    
       
    total_examples_train = examples_perclass_train*4
    total_examples_test = examples_perclass_test*4
    Y_train = np.zeros((total_examples_train, 4))
    Y_test = np.zeros((total_examples_test, 4))
    for i in range (4):
        start_c = i*examples_perclass_train
        end_c = start_c + examples_perclass_train
        Y_train[start_c:end_c, i] =1
    for i in range (4):
        start_c = i*examples_perclass_test
        end_c = start_c + examples_perclass_test
        Y_test[start_c:end_c, i] =1
    return Y_train, Y_test

def read_dataset (user, examples_perclass_train, examples_perclass_test, classes): 
    """
    Reads and loads images for both training and testing datasets for a specific user
    
    Arguments:
    user -- path to directory of the user  e.g. 'user_1'  
    examples_peer class_train --  number of images per class to be used in the training
    examples_peer class_test --  number of images per class to be used in the testing
     
    Returns:
    imgs_train -- list of all images (Normalized) in the training  dataset for the specific user
    imgs_train -- list of all images (Normalized) in the testing  dataset for the specific user
    """    
    
    imgs_train=[]
    imgs_test=[]
    for i in range (4):
        for j in range (examples_perclass_train):
            image = cv2.imread(user + '/Reye/' + str(classes[i]) +'_%d.jpg' %j ,1)
            image = cv2.resize(image,(64,64), interpolation = cv2.INTER_AREA)/255.
            imgs_train.append(image)
        for k in range (examples_perclass_train, examples_perclass_train + examples_perclass_test):
            image = cv2.imread(user + '/Reye/' + str(classes[i]) +'_%d.jpg' %k ,1)
            image = cv2.resize(image,(64,64), interpolation = cv2.INTER_AREA)/255.
            imgs_test.append(image)
    return imgs_train, imgs_test



def shuffle_dataset (images, labels):
    """
    
    Arguments:
    images -- images of the training dataset (unshuffelled)
    labels --  one-hot labels (unshuffelled)
     
    Returns:
    images -- images of the training dataset (shuffelled)
    labels --  one-hot labels (shuffelled)
    """    
    indices = np.arange(labels.shape[0])
    np.random.shuffle(indices)
    images = [images[i] for i in indices]
    labels = labels[indices]
    
    return images, labels


def initialize_parameters(num_conv_layers, n_C0, filters, channels):
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        Wn : [fn, fn, n_Cn-1, n_Cn]
                        
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    parameters = {}
    prev_C = n_C0
    for counter in range (num_conv_layers):
        current_f = filters[counter]
        current_C = channels[counter]
        name = "W" + str(counter+1)
        parameters['W{}'.format(counter+1)] = tf.get_variable(name, [current_f, current_f, prev_C, current_C], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
        prev_C = current_C
    return parameters

def forward_propagation(X, parameters, num_conv_layers, strides, pools, h_layer_n, num_classes):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> ........ -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X -- input dataset placeholder, of shape (batch, in_height, in_width, in_channels)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters
    
    
    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    Z = {}
    A = {}
    P = {}
    
    P = X
    for counter in range (num_conv_layers): 
        current_s = strides[counter]
        current_p = pools[counter]
        current_W = parameters['W{}'.format(counter+1)]
        Z = tf.nn.conv2d(P, current_W, strides = [1,current_s,current_s,1], padding = 'VALID')
        #print("output size of conv layer is: " + str(Z.shape))
        A = tf.tanh(Z)
        P = tf.nn.max_pool(A, ksize = [1,current_p,current_p,1], strides = [1,current_p,current_p,1], padding = 'VALID')
        #print("output size of conv layer is: " + str(P.shape))
    # FLATTEN
    flat = tf.contrib.layers.flatten(P)
    
    fc1 = tf.contrib.layers.fully_connected(flat, h_layer_n)
    Z = tf.contrib.layers.fully_connected(fc1, num_classes , activation_fn=None)
    return Z

