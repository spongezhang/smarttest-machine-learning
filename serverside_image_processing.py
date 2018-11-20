import os
import io
import cv2
import scipy.misc
from googleapiclient import discovery
import numpy as np
import flask
from flask import json
from flask import jsonify
import base64
from PIL import Image
import requests
import re
from PIL import ExifTags
from skimage import transform
from skimage import feature


def hello_world(request):
    """Responds to any HTTP request.
    Args:
        request (flask.Request): HTTP request object.
    Returns:
        The response text or any set of values that can be turned into a
        Response object using
        `make_response <http://flask.pocoo.org/docs/0.12/api/#flask.Flask.make_response>`.
    """
    print(cv2.__version__)

    IMAGE_SIZE = 32
    TEST_IMAGE_SIZE = 18
    shape_condition = 1 #specifies the x-coordinate in an numpy array
    X_data = list()
    answers = []
    nparrays = []
    request_json = request.get_json()
    base64string = request_json['image']
    exif = request_json['exif']

    coordinates = nano_object_detection(str(base64string)) #(xmin, xmax, ymin, ymax)
    answers.append(coordinates)

    gray = stringToGray(str(base64string))
    test_initial = scipy.misc.imresize(gray, (TEST_IMAGE_SIZE, TEST_IMAGE_SIZE))
    print('Check Initial:')
    print(np.array(test_initial, dtype = np.float32).tolist())

    gray = gray[coordinates[2]:coordinates[3], coordinates[0]:coordinates[1]]
    test_crop = scipy.misc.imresize(gray, (TEST_IMAGE_SIZE, TEST_IMAGE_SIZE))
    print('Check Object Detection:')
    print(gray.shape[1], gray.shape[0])
    print(np.array(test_crop, dtype = np.float32).tolist())

    gray = rotateAccordingly(base64string, gray, exif)
    test_rotation = scipy.misc.imresize(gray, (TEST_IMAGE_SIZE, TEST_IMAGE_SIZE))
    print('Check Rotation:')
    print(np.array(test_rotation, dtype = np.float32).tolist())

    blurred_gray = cv2.medianBlur(gray,5)
    test_blurred_gray = scipy.misc.imresize(gray, (TEST_IMAGE_SIZE, TEST_IMAGE_SIZE))
    print('Check Blur:')
    print(np.array(test_blurred_gray, dtype = np.float32).tolist())

    edges = feature.canny(gray, sigma=0.33, low_threshold=20, high_threshold=60)
    # Detect two radii
    hough_radii = np.arange(gray.shape[1]/7.75, gray.shape[1]/6.75, 1)
    hough_res = transform.hough_circle(edges, hough_radii)

    # Select the most prominent 5 circles
    accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii, total_num_peaks=3)

    crop_img = []
    for center_y, center_x, radius in zip(cy, cx, radii):
        crop_img
        rectX = int(center_x - radius)
        rectY = int(center_y - radius)
        crop_img = gray[rectY:(rectY+2*int(radius)), rectX:(rectX+2*int(radius))]

        answers.append((center_x, center_y, radius))

        print('I found!')

        crop_img_final = scipy.misc.imresize(crop_img, (IMAGE_SIZE, IMAGE_SIZE))
        crop_img_final_denoised = cv2.fastNlMeansDenoising(crop_img_final)
        X_data.append(crop_img_final)

        image_enhanced = cv2.equalizeHist(crop_img_final_denoised)
        X_data.append(image_enhanced)

        image_thresholded = cv2.adaptiveThreshold(crop_img_final_denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        X_data.append(image_thresholded)

        test_normal = scipy.misc.imresize(crop_img_final, (TEST_IMAGE_SIZE, TEST_IMAGE_SIZE))
        print(np.array(test_normal, dtype = np.float32).tolist())

        test_enhanced = scipy.misc.imresize(image_enhanced, (TEST_IMAGE_SIZE, TEST_IMAGE_SIZE))
        print(np.array(test_enhanced, dtype = np.float32).tolist())

        test_thresholded = scipy.misc.imresize(image_thresholded, (TEST_IMAGE_SIZE, TEST_IMAGE_SIZE))
        print(np.array(test_thresholded, dtype = np.float32).tolist())
        break

    predict_data = np.array(X_data, dtype = np.float32)
    normal_dic = {}
    normal_dic['image'] = predict_data[0].tolist()
    enhanced_dic = {}
    enhanced_dic['image'] = predict_data[1].tolist()
    thresholded_dic = {}
    thresholded_dic['image'] = predict_data[2].tolist()

    nparrays.append(normal_dic['image'])
    nparrays.append(enhanced_dic['image'])
    nparrays.append(thresholded_dic['image'])

    project_id = "smartimageprocessing"
    model_name = "NewModel"
    version_name = "v1"

    old_model_name = "TestModel"
    old_version_name = "GAMMATEST2"

    normal_result_list = predict_json(project_id, old_model_name, normal_dic, old_version_name)[0]
    enhanced_result_list = predict_json(project_id, old_model_name, enhanced_dic, old_version_name)[0]
    thresholded_result_list = predict_json(project_id, old_model_name, thresholded_dic, old_version_name)[0]

    normal_pred_class = int(normal_result_list['classes'])
    answers.append(normal_pred_class)
    normal_min_difference = float(get_min_possibility_difference(normal_result_list['probabilities']))
    answers.append(normal_min_difference)

    enhanced_pred_class = int(enhanced_result_list['classes'])
    answers.append(enhanced_pred_class)
    enhanced_min_difference = float(get_min_possibility_difference(enhanced_result_list['probabilities']))
    answers.append(enhanced_min_difference)

    thresholded_pred_class = int(thresholded_result_list['classes'])
    answers.append(thresholded_pred_class)
    thresholded_min_difference = float(get_min_possibility_difference(thresholded_result_list['probabilities']))
    answers.append(thresholded_min_difference)

    final_prediction = -1

    if (thresholded_min_difference > 90):
        final_prediction = thresholded_pred_class
    elif (thresholded_pred_class == normal_pred_class):
        final_prediction = thresholded_pred_class
    else:
        final_prediction = -1

    answers.append(final_prediction)
    print(answers)
    return flask.make_response(flask.jsonify(final_prediction))

def nano_object_detection(base64_string):
    url = 'https://app.nanonets.com/api/v2/ObjectDetection/Model/30653d5a-7c1c-4b70-b966-e932a5cbca71/LabelFile/'
    data = {'file': io.BytesIO(base64.b64decode(base64_string))}
    response = requests.post(url, auth=requests.auth.HTTPBasicAuth('_SERDrpKlIQVCvYj_SUJ6_UQHuWEF46xEAQSO979Q13', ''), files=data)
    content = response.json()
    boundingBox = content['result'][0]['prediction'][0]
    xmin = int(boundingBox['xmin'])
    ymin = int(boundingBox['ymin'])
    xmax = int(boundingBox['xmax'])
    ymax = int(boundingBox['ymax'])
    return (xmin, xmax, ymin, ymax)

def predict_json(project, model, instances, version):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    service = discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


def get_min_possibility_difference(class_probabilities):
    #normalize the probabilities so they sum to 100 instead of 1.
    class_probabilities_normalized = []
    for prob in class_probabilities:
        normalized_prob = prob*100
        class_probabilities_normalized.append(normalized_prob)

    #sort the normalized probabilities.
    sorted_probabilities = sorted(class_probabilities_normalized)

    #compute the maximum predicted class and the second.
    maxPos = sorted_probabilities[len(sorted_probabilities) - 1]
    secondMaxPos = sorted_probabilities[len(sorted_probabilities) - 2]

    #compute the difference between them and return it.
    minDifference = maxPos - secondMaxPos
    return minDifference

def normalize(image):
    width = image.shape[0]
    height = image.shape[1]
    return flask.make_response(flask.jsonify(str(width)+" "+str(height)))

def imageTo64(image):
    return str(base64.b64encode(image.tobytes()))

def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    #exif=dict((ExifTags.TAGS[k], v) for k, v in image._getexif().items() if k in ExifTags.TAGS)
    #if not exif['Orientation']:
        #return flask.make_response(flask.jsonify('14'))
        #image=image.rotate(90, expand=True)
    return cv2.cvtColor(np.array(image).astype(np.uint8), cv2.COLOR_BGR2RGB)

def stringToGray(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    #exif=dict((ExifTags.TAGS[k], v) for k, v in image._getexif().items() if k in ExifTags.TAGS)
    #if not exif['Orientation']:
        #image=image.rotate(90, expand=True)
    return cv2.cvtColor(np.array(image).astype(np.uint8), cv2.COLOR_BGR2GRAY)

def rotateAccordingly(base64_string, crop_image, exif_data):
    rotation = False
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    correct = cv2.cvtColor(np.array(image).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    for orientation in ExifTags.TAGS.keys() :
        if ExifTags.TAGS[orientation]=='Orientation':
            break

    exif = {}
    if(image._getexif() is None):
        exif = exif_data
        print('isNone')
        print(exif)
        if exif['Orientation'] == 3 :
            rotation = True
            correct = np.rot90(crop_image, k=2)
            print('180 rotation!')
        elif exif['Orientation'] == 6 :
            rotation = True
            correct=np.rot90(crop_image, k=3)
            print('270 rotation!')
        elif exif['Orientation'] == 8 :
            rotation = True
            correct=np.rot90(crop_image, k=1)
            print('90 rotation!')
    else:
        exif=dict(image._getexif().items())
        print('nopee i fouund!!!')
        print(exif)
        if exif[orientation] == 3 :
            rotation = True
            correct = np.rot90(crop_image, k=2)
            print('180 rotation!')
        elif exif[orientation] == 6 :
            rotation = True
            correct=np.rot90(crop_image, k=3)
            print('270 rotation!')
        elif exif[orientation] == 8 :
            rotation = True
            correct=np.rot90(crop_image, k=1)
            print('90 rotation!')

    if rotation:
        return correct
    return crop_image
