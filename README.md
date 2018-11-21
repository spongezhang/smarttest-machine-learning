# Machine Learning & Image Processing Component of SMARTTest

A large-scale image processing/machine learning backend component written in Python. Implemented several image processing,
feature extraction, and data augmentation algorithms, a pre-trained fast region-based convolutional network for object
localization, and a custom convolutional network model for image classification in Python with TensorFlow and OpenCV. The pre- trained Fast R-CNN is imported using the Nanonets App, the custom CNN is included in the local_image_processing.py file. 
In this same file, you will find all the relevant algorithms and methods. serverside_image_processing.py and the 
related requirements.txt files are needed for the Google Cloud Function which uses a Flask server under the hood. The 
hello_world function in serverside_image_processing.py is the function to be executed for online prediction and can be 
reached by a simple API call. Note that the relative paths of training and testing images, and saving location of various 
image processing samples are relative and would have to be changed. Some of them, including the training and testing folders,
have to obviously be filled in with sample images.
