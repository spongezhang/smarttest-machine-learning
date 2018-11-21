# Machine Learning & Image Processing Component of SMARTTest
## DESCRIPTION
A large-scale image processing/machine learning backend component written in Python. Implemented several image processing,
feature extraction, and data augmentation algorithms, a pre-trained fast region-based convolutional network for object
localization, and a custom convolutional network model for image classification in Python with TensorFlow and OpenCV. The pre- trained Fast R-CNN is imported using the Nanonets App, the custom CNN is included in the ```local_image_processing.py file.``` 

## FILES
```local_image_processing.py```: This is the Python code that is used for the local training of the CNN using TensorFlow. For 
this code to work, directories and filepaths specified has to actually exist. Specifically, the training and testing 
directories have to be created and filled in with sample images. (two sets of images: one for validation, one for training) Inside this same file, you will find all the relevant algorithms and methods discussed in the description.

```serverside_image_processing.py```: This is the Python code is used as the Google Cloud Function which uses a Flask server
under the hood. The required Python modules are listed inside the accompanying ```requirements.txt```. The hello_world
function inside this file is the function designed to be executed for online prediction and can be reached by a simple API
call. It gets a base64string from the app, converts it to an image and then applies, in order, the object detection model,
a robust circle detection algorithm, some further image processing techniques, and finally the image classification model.
