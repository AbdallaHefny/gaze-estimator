"""
Created on Sun Mar 24 17:39:43 2019


This code is for the SDPII project


@author: AbdAlla Hefny
"""

import cv2
import time
import os
import winsound
import configparser

config = configparser.ConfigParser()
config.read('ET_config.ini')
user_name = config['End User Section']['name of the user directory']
examples_per_class_train = int(config['End User Section']['number of frames for training'])
examples_per_class_test = int(config['End User Section']['number of frames for testing'])


newpath = user_name
if not os.path.exists(newpath):
    os.makedirs(newpath)
newpath = user_name + '/Reye'
if not os.path.exists(newpath):
    os.makedirs(newpath)    

# Beep Sound settings
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 800  # Set Duration To 800 ms 
# Total number of farmes per class to be collected
frames_per_class = examples_per_class_train + examples_per_class_test


classes = ['right', 'forward', 'left', 'closed']
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 64)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 64)

for i in range (60):
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break  
print("Image acquisition will start now, read the instructions appearing in the console....")
time.sleep(1)
#winsound.Beep(frequency, duration)  
start = time.time()
for j in range (len(classes)):
    print(classes[j] + '........')
    time.sleep(1)
    winsound.Beep(frequency, duration)   # strart acquistion of current class    
    for i in range (frames_per_class):
        ret, frame = cap.read()
        cv2.imwrite(newpath + '/' + classes[j] +'_%d.jpg' %i, frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break  
    winsound.Beep(frequency, duration)  # end acquistion of current class
    time.sleep(1)
end = time.time()
fps = frames_per_class * len(classes) / (end-start - 8 - 8*0.8)
#print(fps)
cap.release()
cv2.destroyAllWindows()