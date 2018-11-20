#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 00:16:55 2018

@author: uzaymacar
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import scipy.misc

import os

print(cv2.__version__)
IMAGE_SIZE = 32 #it was originally 28 for a MNIST image, we have chosen 32 in our case.
REJECT = 0
reject_array = []
irregularity_percentage_array = []
train_invalid_count = 0
train_syphilis_count = 0
train_hiv_count = 0
train_dualpositive_count = 0
train_negative_count= 0
test_invalid_count = 0
test_syphilis_count = 0
test_hiv_count = 0
test_dualpositive_count = 0
test_negative_count = 0
ratio_array = []
badcanny_filename_array = []
text_file = open("output.txt", "w")

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""

  features = features[list(features.keys())[0]]
  # remember to close the comments on these tf.Print functions
  # features = tf.Print(features, [tf.shape(features), "shape of features"], summarize = 10)
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # INSTI Test Kit images are IMAGE_SIZE*IMAGE_SIZE(currently 32x32) pixels, and have one color channel.
  # The single color channel is aimed for the simplicity of the convolutional neural netowrk,
  # as they do not convey meaningful information in this case.
  input_layer = tf.reshape(features, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
  #input_layer = tf.Print(input_layer, [tf.shape(input_layer), "shape of input_layer"], summarize = 10)


  # Convolutional Layer #1
  # Computes 32 features using a 3x3 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1]
  # Output Tensor Shape: [batch_size, IMAGE_SIZE, IMAGE_SIZE, 16]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=16,
      kernel_size=[5, 5],
      padding="same",
      data_format='channels_last',
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, IMAGE_SIZE, IMAGE_SIZE, 16]
  # Output Tensor Shape: [batch_size, IMAGE_SIZE/2, IMAGE_SIZE/2, 16]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 3x3 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, IMAGE_SIZE/2, IMAGE_SIZE/2, 16]
  # Output Tensor Shape: [batch_size, IMAGE_SIZE/2, IMAGE_SIZE/2, 32]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, IMAGE_SIZE/2, IMAGE_SIZE/2, 32]
  # Output Tensor Shape: [batch_size, IMAGE_SIZE/4, IMAGE_SIZE/4, 32]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  
  # Convolutional Layer #3
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, IMAGE_SIZE/4, IMAGE_SIZE/4, 32]
  # Output Tensor Shape: [batch_size, IMAGE_SIZE/4, IMAGE_SIZE/4, 64]
  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  
  # Pooling Layer #3
  # Third max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, IMAGE_SIZE/4, IMAGE_SIZE/4, 64]
  # Output Tensor Shape: [batch_size, IMAGE_SIZE/8, IMAGE_SIZE/8, 64]
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
  
  # Convolutional Layer #4
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, IMAGE_SIZE/8, IMAGE_SIZE/8, 64]
  # Output Tensor Shape: [batch_size, IMAGE_SIZE/8, IMAGE_SIZE/8, 128]
  conv4 = tf.layers.conv2d(
      inputs=pool3,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  
  # Pooling Layer #4
  # Fourth max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, IMAGE_SIZE/8, IMAGE_SIZE/8, 128]
  # Output Tensor Shape: [batch_size, IMAGE_SIZE/16, IMAGE_SIZE/16, 128]
  pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, IMAGE_SIZE/16, IMAGE_SIZE/16, 64]
  # Output Tensor Shape: [batch_size, IMAGE_SIZE/16 * IMAGE_SIZE/16 * 64]
  pool4_flat = tf.reshape(pool4, [-1, int(IMAGE_SIZE/16) * int(IMAGE_SIZE/16) * 128])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, IMAGE_SIZE/16 * IMAGE_SIZE/16 * 128]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool4_flat, units=128, activation=tf.nn.relu)

  # Add dropout operation; 0.95 probability that element will be kept
  # We probably want to keep the element because it may convey critical information
  # about the locations of a black dot on the test kit screen.
  dropout = tf.layers.dropout(
          inputs=dense, rate=0.50, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 5]
  #5 is for the total number of possible cases of the INSTI test kits.
  logits = tf.layers.dense(inputs=dropout, units=5)
  
  #For our training labels, we will have:
        #0 = Negative
        #1 = Syphilis
        #2 = HIV
        #3 = Dual Positive
        #4 = Invalid Test

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  
  #export_outputs ={.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.ClassificationOutput(classes=tf.as_string(predictions['classes']))})
  #export_output is to save the model
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, 
                                      export_outputs={'predict': tf.estimator.export.PredictOutput(predictions)})

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels))
  #loss = tf.Print(loss, ["loss : ", loss])
  tf.summary.scalar('Loss', loss)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00000001)
    #rate = tf.train.exponential_decay(0.001, tf.train.get_global_step(), 1, 0.9999)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0003)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"]),
      "precision": tf.metrics.precision(labels=labels, predictions=predictions["classes"]),
      "recall": tf.metrics.recall(labels=labels, predictions=predictions["classes"])
      }
  tf.summary.scalar('Accuracy', eval_metric_ops.get('accuracy'))
  tf.summary.scalar('Precision', eval_metric_ops.get('precision'))
  tf.summary.scalar('Recall', eval_metric_ops.get('recall'))
  
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def get_label(filename, mode):
    if "invalid" in filename or "Invalid" in filename:
        if mode == 0:
            global train_invalid_count
            train_invalid_count += 1
        elif mode == 1:
            global test_invalid_count
            test_invalid_count += 1
        return 4
    elif "syphilis" in filename or "Syphilis" in filename:
        if mode == 0:
            global train_syphilis_count
            train_syphilis_count += 1
        elif mode == 1:
            global test_syphilis_count
            test_syphilis_count += 1
        return 1
    elif "HIV" in filename or "hiv" in filename:
        if mode == 0:
            global train_hiv_count
            train_hiv_count += 1
        elif mode == 1:
            global test_hiv_count
            test_hiv_count += 1
        return 2
    elif "dualpositive" in filename or "DualPositive" in filename:
        if mode == 0:
            global train_dualpositive_count
            train_dualpositive_count += 1
        elif mode == 1:
            global test_dualpositive_count
            test_dualpositive_count += 1
        return 3
    elif "negative" in filename or "Negative" in filename:
        if mode == 0:
            global train_negative_count
            train_negative_count += 1
        elif mode == 1:
            global test_negative_count
            test_negative_count += 1
        return 0

def hough_circle_detection(X_img_file_paths, mode):
     """Circle Detection and Resizing Method"""
     #accepts a file path
     dirname = ""
     if mode == 0:
         dirname = "TrainingFinal"
     elif mode == 1:
         dirname = "Testing"
     
     X_data = list()
     global REJECT
    
     for filename in os.listdir(X_img_file_paths):
         if "jpg" in filename or "jpeg" in filename:
             print(filename)
             
             img = Image.open(X_img_file_paths + "/" + filename)    
             gray = cv2.cvtColor(np.array(img).astype(np.uint8), cv2.COLOR_BGR2GRAY)
             if gray.shape[1]>800:
                 new_width = 800
                 ratio = gray.shape[1]/800
                 new_height = gray.shape[0]/ratio
                 gray = scipy.misc.imresize(gray, (int(new_height), int(new_width)))
             
             gray_gauss = cv2.GaussianBlur(gray, (5, 5), 0)
             gray_smooth = cv2.addWeighted(gray_gauss,1.5,gray,-0.5,0)
             circles = cv2.HoughCircles(gray_smooth, cv2.HOUGH_GRADIENT, 1, 200, param1=30, param2=15, minRadius=int(gray.shape[1]/8.5),maxRadius=int(gray.shape[1]/7))
             
             if circles is None:
                 print('problem')
                 break
                 #circles = cv2.HoughCircles(median_blur, cv2.HOUGH_GRADIENT, 1, max(gray.shape[0], gray.shape[1]) * 2, param1=50, param2=30, minRadius=int(gray.shape[1]/8),maxRadius=int(gray.shape[1]/6))
                 badcanny_filename_array.append(filename)
             
             else:
             
                 center_x = circles[0][0][0]
                 center_y = circles[0][0][1]
                 radius = circles[0][0][2]
                 rectX = int(center_x) - int(radius) 
                 rectY = int(center_y) - int(radius)
                 
                 crop_img = gray[rectY:(rectY+2*int(radius)), rectX:(rectX+2*int(radius))]
                 
                 if center_x > gray.shape[1]/2.5 and center_x < gray.shape[1]/1.5:                    
                     print('I found!')
                     
                     #GETTING THE NORMAL IMAGE
                     crop_img_final = scipy.misc.imresize(crop_img, (IMAGE_SIZE, IMAGE_SIZE))
                     crop_img_final = cv2.fastNlMeansDenoising(crop_img_final)         
                     cv2.imwrite(os.path.join(dirname, str(filename) + dirname + '.png'),crop_img_final)
                     X_data.append(crop_img_final)
                     
                     #GETTING THE ENHANCED VERSION 
                     image_enhanced = cv2.equalizeHist(crop_img_final)
                     cv2.imwrite(os.path.join(dirname, str(filename) + dirname + 'enhanced' + '.png'), image_enhanced)
                     X_data.append(image_enhanced)
                              
                     #GETTING THE THRESHOLDED VERSION
                     image_thresholded = cv2.adaptiveThreshold(crop_img_final, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
                     cv2.imwrite(os.path.join(dirname, str(filename) + dirname + 'thresholded' + '.png'), image_thresholded)
                     X_data.append(image_thresholded)
                     
                     #REFERENCE MASKING THE THRESHOLDED IMAGE (FOR INTEREST ZONES)
                     interest_zones_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
                     cv2.circle(interest_zones_mask,(int(IMAGE_SIZE/2), int(IMAGE_SIZE/4)),int(IMAGE_SIZE/9),255,thickness=-1)
                     cv2.circle(interest_zones_mask,(int(IMAGE_SIZE/2), 3*int(IMAGE_SIZE/4)),int(IMAGE_SIZE/9),255,thickness=-1)
                     cv2.circle(interest_zones_mask,(int(IMAGE_SIZE/4), int(IMAGE_SIZE/2)),int(IMAGE_SIZE/9),255,thickness=-1)
                     
                     interest_zones_img = cv2.bitwise_and(interest_zones_mask, image_thresholded)
                     cv2.imwrite(os.path.join(dirname, str(filename) + dirname + 'MASKEDINTEREST' + '.png'), interest_zones_img)
                     
                     #REFERENCE MASKING THE THRESHOLDED IMAGE
                     #FOR UNINTERESTED ZONES TO DETECT SHADOWS AND ANOMALITIES
                     irregularity_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
                     irregularity_mask = irregularity_mask + 255 #turn from black to white
                     
                     cv2.circle(irregularity_mask,(0, 0), int(IMAGE_SIZE/4),0,thickness=-1) #color top left corner
                     cv2.circle(irregularity_mask,(0, IMAGE_SIZE), int(IMAGE_SIZE/4),0,thickness=-1) #color bottom left corner
                     cv2.circle(irregularity_mask,(IMAGE_SIZE, 0), int(IMAGE_SIZE/4),0,thickness=-1) #color top right corner
                     cv2.circle(irregularity_mask,(IMAGE_SIZE, IMAGE_SIZE), int(IMAGE_SIZE/4),0,thickness=-1) #color bottom right corner
                     
                     cv2.circle(irregularity_mask,(int(IMAGE_SIZE/2), int(IMAGE_SIZE/2)),int(IMAGE_SIZE/2),0,thickness=5) #Marker for the surrounding circular cell membrane edges
                     
                     cv2.circle(irregularity_mask,(int(IMAGE_SIZE/2), int(IMAGE_SIZE/4)),int(IMAGE_SIZE/9),0,thickness=-1) #Spot for Control Zone
                     cv2.circle(irregularity_mask,(int(IMAGE_SIZE/2), 3*int(IMAGE_SIZE/4)),int(IMAGE_SIZE/9),0,thickness=-1) #Spot for HIV Zone
                     cv2.circle(irregularity_mask,(int(IMAGE_SIZE/4), int(IMAGE_SIZE/2)),int(IMAGE_SIZE/9),0,thickness=-1) #Spot for Syphilis Zone
            
                     irregularity_img = cv2.bitwise_and(irregularity_mask, image_thresholded)
                     
                     #LOOP THROUGH THE IRREGULARITY IMAGE TO COUNT THE # OF WHITE PIXELS REPRESTING IRREGULARITIES.
                     total_pixels = irregularity_img.shape[0] * irregularity_img.shape[1]
                     total_white_pixels = 0
                     for i in range(irregularity_img.shape[0]):
                         for j in range(irregularity_img.shape[1]):
                             pixel = irregularity_img[i][j]
                             if (pixel == 255):
                                 total_white_pixels += 1
                    
                     percentage = (total_white_pixels / total_pixels) * 100
                     
                     text_file.write("PHOTO NAME: %s ;" % filename)
                     if (percentage > 5):
                         REJECT += 1
                         reject_array.append(filename)
                         text_file.write("Irregularity Percentage ABOVE LIMIT %s " % str(percentage))
                     else:
                         text_file.write("Irregularity Percentage %s " % str(percentage))
                 
                     text_file.write("\n\n")       
                     cv2.imwrite(os.path.join(dirname, str(filename) + dirname + 'MASKEDIRREGULARITY' + "%" + str(percentage) + '.png'), irregularity_img)
                 
                 else:
                     c = scipy.misc.imresize(crop_img, (IMAGE_SIZE, IMAGE_SIZE))
                     Image.fromarray(c).show()
                     badcanny_filename_array.append(filename)
                     print("COULDN'T FIND!")

     return X_data

def main(unused_argv):
    """Main Method of Script"""
    print(tf.VERSION)
    TRAINING_DATA = []
    TRAINING_LABELS = []
    TEST_DATA = []
    TEST_LABELS = []
    trainingpath = "/Users/uzaymacar/Desktop/SMART/Photos/TrainingImagesFinal"
    testingpath = "/Users/uzaymacar/Desktop/SMART/Photos/ValidationSetCurrent"
    
    circle_images = hough_circle_detection(trainingpath, 0)
    circle_images_final = np.array(circle_images, dtype = np.float32) # Convert to numpy
    TRAINING_DATA = circle_images_final
    numberOfTrainingImages = (int) (TRAINING_DATA.size/(IMAGE_SIZE*IMAGE_SIZE))
    training_copying_factor = 3
    test_copying_factor = 3
    #set the copying factors to 1 if you are not data augmenting
    
    NORMALIZED_TEST_LABELS = []
    TEST_FILE_NAMES = []
    
    for filename in os.listdir(trainingpath):
        if "jpg" in filename or "jpeg" in filename:
            print("train: " + filename)
            for i in range(0, training_copying_factor):
                TRAINING_LABELS.append(get_label(filename, 0))
            
    for filename in os.listdir(testingpath):
        if "jpg" in filename or "jpeg" in filename:
            print("test: " + filename)
            TEST_FILE_NAMES.append(filename)
            label = None
            for i in range(0, test_copying_factor):
                label = get_label(filename,1)
                TEST_LABELS.append(label)
            NORMALIZED_TEST_LABELS.append(label)

    circle_test_images = hough_circle_detection(testingpath, 1)
    text_file.close()
    circle_test_images_final = np.array(circle_test_images, dtype = np.float32) # Convert to numpy
    TEST_DATA = circle_test_images_final
    
    numberOfTestingImages = (int) (TEST_DATA.size/(IMAGE_SIZE*IMAGE_SIZE))
        
    print("Test Images Count: " + str(numberOfTestingImages) + " (test images with a copying factor of " + str(test_copying_factor) + ")")
    print("Test Labels: " + str(TEST_LABELS))
    print("Training Images Count: " + str(numberOfTrainingImages) + " (training images with a copying factor of " + str(training_copying_factor) + ")")
    print("Training Labels: " + str(TRAINING_LABELS))
    print("Training/Test Invalid Count: " + str(train_invalid_count) + "/" + str(test_invalid_count))
    print("Training/Test Negative Count: " + str(train_negative_count) + "/" + str(test_negative_count))
    print("Training/Test Syphilis Count: " + str(train_syphilis_count) + "/" + str(test_syphilis_count))
    print("Training/Test Hiv Count: " + str(train_hiv_count) + "/" + str(test_hiv_count))
    print("Training/Test Dual Positive Count: " + str(train_dualpositive_count) + "/" + str(test_dualpositive_count))
    print(badcanny_filename_array)
    print("REJECTED: " + str(REJECT))
    print(reject_array)
    
    train_data = TRAINING_DATA
    train_labels = np.asarray(TRAINING_LABELS, dtype = np.int32)
    #eval_data = TRAINING_DATA
    #eval_labels = np.asarray(TRAINING_LABELS, dtype = np.int32)
    eval_data = TEST_DATA #change eval data and label to the trainings to see your training error
    eval_labels = np.asarray(TEST_LABELS, dtype = np.int32)

    # Create the Estimator, different versions (instances) tried out are written below
    disease_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/doitmodel101_convnet_model", 
        config = tf.contrib.learn.RunConfig(
                save_checkpoints_steps=20,
                save_checkpoints_secs=None,
                save_summary_steps=40,))      
     
    export_dir_base = '/Users/uzaymacar/Desktop/SMART/FinalSavedModels/'
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
       tensors=tensors_to_log, every_n_iter=50)
    
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=numberOfTrainingImages,
      num_epochs=None,
      shuffle=True)
    disease_classifier.train(
      input_fn=train_input_fn,
      steps=500,
      hooks=[logging_hook])
    
    #for exporting out
    disease_classifier.export_savedmodel(export_dir_base, json_serving_input_receiver_fn)
    
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
    eval_results = disease_classifier.evaluate(input_fn=eval_input_fn)
    test_results = disease_classifier.predict(input_fn=eval_input_fn)
    print(eval_results)

     
    prediction_labels = []
    prediction_possibilities = []
    prediction_minDifferences = []
    
    for i,p in enumerate(test_results):
      #get the class prediction; either 0, 1, 2, 3, or 4.
      class_prediction = p.get('classes')
      prediction_labels.append(int(class_prediction))
      
      #get the class probabilities as an array.
      class_probabilities = p.get('probabilities')
      prediction_possibilities.append(class_probabilities)
      
      #get the possibility difference (out of %100) between the maximum predicted class and the second.
      min_possibility_difference = get_min_possibility_difference(class_probabilities)
      prediction_minDifferences.append(int(min_possibility_difference))
      
      #print the class prediction and min possibility difference.
      print(str(TEST_FILE_NAMES[int(i/test_copying_factor)]), str(class_prediction), str(min_possibility_difference))
    
    final_preds = []
    preds_count = len(prediction_labels)
    for i in range(0, preds_count, test_copying_factor):
        normal_pred_class = prediction_labels[i]
        normal_min_difference = prediction_minDifferences[i]
        enhanced_pred_class = prediction_labels[i+1]
        enhanced_min_difference = prediction_minDifferences[i+1]
        thresholded_pred_class = prediction_labels[i+2]
        thresholded_min_difference = prediction_minDifferences[i+2]
        #if (normal_pred_class == enhanced_pred_class or normal_pred_class == thresholded_pred_class):
            #final_preds.append(normal_pred_class)
        if (normal_pred_class == thresholded_pred_class):
            final_preds.append(normal_pred_class)
            print(TEST_FILE_NAMES[int(i/test_copying_factor)] + str(normal_pred_class) + str(enhanced_pred_class) + str(thresholded_pred_class))
        elif (enhanced_pred_class == thresholded_pred_class):
            final_preds.append(enhanced_pred_class)
            print("B")
        else:
            if(thresholded_min_difference >= enhanced_min_difference and thresholded_min_difference >= thresholded_min_difference):
                final_preds.append(thresholded_pred_class)
            elif(normal_min_difference >= enhanced_min_difference and normal_min_difference >= thresholded_min_difference):
                final_preds.append(normal_pred_class)
            else:
                final_preds.append(enhanced_pred_class)
            
    print("---------------------------------------------------------------------")
    print("FINALIZED TEST RESULTS")
    
    total_labels_count = len(NORMALIZED_TEST_LABELS)
    correct_pred_count = 0
    for i in range(0, total_labels_count):
        if final_preds[i] == NORMALIZED_TEST_LABELS[i]:
            correct_pred_count += 1
            print(TEST_FILE_NAMES[i] + ": ",str(final_preds[i]), " " + str(NORMALIZED_TEST_LABELS[i]), " CORRECT")
        else:
            print(TEST_FILE_NAMES[i] + ": ","GUESSED " + str(final_preds[i]) + " BUT IT WAS " + str(NORMALIZED_TEST_LABELS[i]), " WRONG!")
    
    print("CORRECTLY GUESSED: " + str(correct_pred_count) + " OUT OF " + str(total_labels_count))
    print("ACCURACY: %" + str(float(correct_pred_count/total_labels_count)*100))
    print(ratio_array)

def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_typecasted = tf.cast(image_decoded, tf.float32)
    image_reshaped = tf.reshape(image_typecasted, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    
    return image_reshaped

    
def json_serving_input_receiver_fn():
  inputs = {
    'image': tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1]),
  }
  return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def serving_input_receiver_fn():
    feature_spec = {'image/encoded': tf.FixedLenFeature(shape=[],
        dtype=tf.string)}

    serialized_tf_example = tf.placeholder(dtype=tf.string,name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}

    features = tf.parse_example(serialized_tf_example, feature_spec)
    jpegs = features['image/encoded']
    images = tf.map_fn(_parse_function, jpegs, dtype=tf.float32)

    return tf.estimator.export.ServingInputReceiver(images, receiver_tensors)
 
def get_min_possibility_difference(class_probabilities):
    #normalize the probabilities so they sum to 100 instead of 1.
    class_probabilities_normalized = class_probabilities*100
    #sort the normalized probabilities.
    sorted_probabilities = sorted(class_probabilities_normalized)
    
    #compute the maximum predicted class and the second.
    maxPos = sorted_probabilities[len(sorted_probabilities) - 1]
    secondMaxPos = sorted_probabilities[len(sorted_probabilities) - 2]
    
    #compute the difference between them and return it.
    minDifference = maxPos - secondMaxPos
    return minDifference
 
if __name__ == "__main__":
      tf.app.run()
