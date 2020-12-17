import pandas
import os
import numpy as np
from matplotlib.pyplot import imread
from sklearn.model_selection import train_test_split
import cv2
import dlib
import torch as pt
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import sys

from A1 import A1
from A2 import A2
from B1 import B1
from B2 import B2

use_CNN = False
if len(sys.argv) > 1:
    if sys.argv[1] in ['svm', 'SVM']:
        use_CNN = False
    elif sys.argv[1] in ['CNN', 'cnn']:
        use_CNN = True

size_batch, learning_rate = 40, 0.01
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def run_dlib_shape(image):
    # in this function we load the image, detect the landmarks of the face, and then return the image and the landmarks
    # load the input image, resize it, and convert it to grayscale
    resized_image = image.astype('uint8')

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype('uint8')

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    num_faces = len(rects)

    if num_faces == 0:
        return None, resized_image

    face_areas = np.zeros((1, num_faces))
    face_shapes = np.zeros((136, num_faces), dtype=np.int64)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        temp_shape = predictor(gray, rect)
        temp_shape = shape_to_np(temp_shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)],
        #   (x, y, w, h) = face_utils.rect_to_bb(rect)
        (x, y, w, h) = rect_to_bb(rect)
        face_shapes[:, i] = np.reshape(temp_shape, [136])
        face_areas[0, i] = w * h
    # find largest face and keep
    dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

    return dlibout, resized_image


def get_data(path, field, is_test=False):
    if path in ['cartoon_set', 'cartoon_set_test']:
        file_label = 'file_name'
        reshape_img = True
    else:
        file_label = 'img_name'

    if is_test:
        path = os.path.join('Datasets', 'test', path)
    else:
        path = os.path.join('Datasets', 'train', path)

    df = pandas.read_csv(os.path.join(path, 'labels.csv'), delimiter='\t')

    _X = []
    _y = []

    for i, row in df.iterrows():
        # print(path, file_label, os.path.join(path, 'img', row[file_label]))
        img = imread(os.path.join(path, 'img', row[file_label]))
        if reshape_img:
            img = img[:, :, :3] * 255
        features, resized_img = run_dlib_shape(img)
        if features is not None:
            # calculating the averate coordinates of left and right eyes
            cor1 = (features[42:48].sum(axis=0) // 6).astype(np.int32)
            cor2 = (features[36:42].sum(axis=0) // 6).astype(np.int32)

            if field == 'eye_color':
                # flatten features
                features = features.reshape(-1)

                try:
                    # get the eye color from image
                    colors = np.array([])
                    #                     colors = []
                    # 9*9 pexels region centered with average coordinate
                    for i in range(-3, 4):
                        for j in range(-3, 4):
                            # append eye color to the features
                            colors = np.concatenate(
                                (colors, resized_img[cor1[1] + i, cor1[0] + j], resized_img[cor2[1] + i, cor2[0] + j]))
                    #                             colors.append(resized_img[cor1[1]+i, cor1[0]+j])
                    #                             colors.append(resized_img[cor2[1]+i, cor2[0]+j])
                    colors = np.array(colors)
                    # colors = colors.sum(axis=0) / 9
                    features = np.concatenate((features, colors), axis=None)
                except IndexError:
                    print(features.shape, cor1, cor2, resized_img.shape)
            elif field in ['face_shape', 'gender']:
                # angle between two segments
                cos_value = []
                for i in range(0, 15):
                    v1 = features[i + 1] - features[i]
                    v2 = features[i + 2] - features[i + 1]
                    cos = min(1.0, np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
                    arccos = np.arccos(cos)
                    if arccos != arccos:
                        print(cos, v1, v2)
                    cos_value.append(arccos)

                # flatten features
                features = features.reshape(-1)
                # add angle between two segments
            #                 features = np.concatenate((features, cos_value))
            else:
                # flatten features
                features = features.reshape(-1)
            _X.append(features)
            _y.append(row[field])

    _X = np.array(_X)
    _y = np.array(_y)

    return _X, _y


class GetLoader(pt.utils.data.Dataset):
    # constructor
    def __init__(self, path, field, is_test=False):
        if path in ['cartoon_set', 'cartoon_set_test']:
            self.file_label = 'file_name'
        else:
            self.file_label = 'img_name'
        if not is_test:
            self.path = os.path.join('Datasets', 'train', path)
        else:
            self.path = os.path.join('Datasets', 'test', path)
        self.csv = pd.read_csv(os.path.join(self.path, 'labels.csv'), delimiter='\t')
        self.field = field

    def __getitem__(self, index):
        img_path = os.path.join(self.path, 'img', self.csv[self.file_label][index])
        img = imread(img_path)
        img = np.moveaxis(img, 2, 0)
        labels = self.csv[self.field][index]
        return img, labels

    def __len__(self):
        return self.csv.shape[0]


def data_preprocessing(path, field, use_CNN=True):
    if path in ['celeba', 'celeba_test_set']:
        path = 'celeba'
        test_path = 'celeba_test'
    else:
        path = 'cartoon_set'
        test_path = 'cartoon_set_test'

    if use_CNN:
        # get train and valid dataset
        data = GetLoader(path, field)
        data = DataLoader(data, batch_size=size_batch, shuffle=True)
        train_size = int(0.8 * len(data.dataset))
        test_size = len(data.dataset) - train_size
        train, valid = pt.utils.data.random_split(data.dataset, [train_size, test_size])
        train = DataLoader(train, batch_size=size_batch, shuffle=True)
        valid = DataLoader(valid, batch_size=size_batch, shuffle=True)

        # get test dataset from local images
        test = GetLoader(test_path, field, True)
        test = DataLoader(test, batch_size=size_batch, shuffle=True)
        return train, valid, test
    else:
        _X, _y = get_data(path, field)
        X_train, X_valid, y_train, y_valid = train_test_split(_X, _y, test_size=0.2, random_state=1)
        _X, _y = get_data(test_path, field, True)
        return (X_train, y_train), (X_valid, y_valid), (_X, _y)


# ======================================================================================================================
# Data preprocessing
# data_train, data_val, data_test = data_preprocessing(args...)
# ======================================================================================================================
# Task A1
train, val, test = data_preprocessing('celeba', 'gender')
model_A1 = A1(use_CNN)  # Build model object.
acc_A1_train = model_A1.train(train, val)
acc_A1_test = model_A1.test(test)

# acc_A1_train = model_A1.train(args...) # Train model based on the training set (you should fine-tune your model based on validation set.)
# acc_A1_test = model_A1.test(args...)   # Test model based on the test set.
# Clean up memory/GPU etc...             # Some code to free memory if necessary.


# ======================================================================================================================
# Task A2
# model_A2 = A2(args...)
# acc_A2_train = model_A2.train(args...)
# acc_A2_test = model_A2.test(args...)
train, val, test = data_preprocessing('celeba', 'smiling')
model_A2 = A2(use_CNN)  # Build model object.
acc_A2_train = model_A2.train(train, val)
acc_A2_test = model_A2.test(test)
# Clean up memory/GPU etc...


# ======================================================================================================================
# Task B1
# model_B1 = B1(args...)
# acc_B1_train = model_B1.train(args...)
# acc_B1_test = model_B1.test(args...)
# Clean up memory/GPU etc...
train, val, test = data_preprocessing('cartoon_set', 'face_shape')
model_B1 = B1(use_CNN)  # Build model object.
acc_B1_train = model_B1.train(train, val)
acc_B1_test = model_B1.test(test)

# ======================================================================================================================
# Task B2
# model_B2 = B2(args...)
# acc_B2_train = model_B2.train(args...)
# acc_B2_test = model_B2.test(args...)
train, val, test = data_preprocessing('cartoon_set', 'eye_color')
model_B2 = B2(use_CNN)  # Build model object.
acc_B2_train = model_B2.train(train, val)
acc_B2_test = model_B2.test(test)
# Clean up memory/GPU etc...


# ======================================================================================================================
## Print out your results with following format:
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))
