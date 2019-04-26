This is tensorflow (CPU) implementation of a real-time gaze estimator. 
For the current version, it classsifies gazes for only 4 classes: Right, Forward, Left, or closed.
The clasasification layer will be substitued by a regression layer in the future version.
The application was tested on webcam, the prediction speed is 30 fps.
You may try to run the application with faster cameras, the speed can go beyond 30 fps. 

Files description
----------------------------------------------
"ET_config.ini" ----> Configuration file
"ET_daq.py"     ----> Collects Training Dataset. Saves images in correct directory. 
"ET_train.py"   ----> Train the model. Saves the TF model to be used later.
"ET_predict.py" ----> Loads the TF model. Runs the Real-Time Gaze Estimator.
"conv_utils.py" ----> Helping functions for the training.


How to run the application?
------------------------------------------------
- place all the files in your working directory
- open "ET_config.ini" file using any text editor
- specify the user name 
- change the number of frames (if needed)
- In the shell, run the following commands in order:
	python ET_daq.py
	python ET_train.py
	python ET_predict.py

-------------------------------------------------------------------------------------------------------
NO NEED TO CREATE/REMOVE ANY DIRECTORY. THE PROGRAM TAKES CARE OF CREATING AND PLACING ALL DIRECTORIES.
-------------------------------------------------------------------------------------------------------


# To-Do List
-----------------------------------------------
Add a regression head to predict gazes at any direction
Calculate maximum speed of the estimator