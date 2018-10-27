
import os
import random
import sys
import warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import gc

from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images, imsave
from skimage.transform import resize
from skimage.morphology import label as sklabel
from skimage.filters import threshold_otsu
from skimage.util import view_as_blocks
from skimage.measure import regionprops

from keras.utils import Progbar

from keras.models import Model, load_model, save_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda, Flatten
from keras.layers.convolutional import Conv2D, Conv2DTranspose,Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K
from keras.optimizers import Adam, Nadam, Adamax
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from skimage.color import rgb2gray
from scipy import ndimage
from scipy.ndimage.morphology import binary_fill_holes
from tqdm import tqdm

from kaggle_utils.keras_cnn import conv_layer,bottonneck_conv_layer,\
    residual_block, dense_set, UNET_256
from kaggle_utils.dill_helper import save_obj, load_obj
import tensorflow as tf

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')


# Data Path
TRAIN_PATH_OLD = 'input/stage1_train/'
TRAIN_PATH = 'input/stage1_train/'
#TRAIN_PATH = 'input/stage1_train_clean/'
TEST_PATH = 'input/stage1_test/'


# Get train and test IDs

#train_ids = next(os.walk(TRAIN_PATH_OLD))[1]
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
total_size = len(train_ids)

CLASS1_SIZE = 256
CLASS2_SIZE = 256
TYPE_CLASS_SIZE = 64

CLASS1 = {
    'black': 0,
    'cell': 1,
    'organ': 2,
    'white': 3,
}

CLASS1_INV = {
    0 : 'black',
    1 : 'cell',
    2 : 'organ',
    3 : 'white'
}

CLASS2_BLACK = {
    'blackcommon' :0,
    'blackHuge' : 1,
    'blackStar' : 2
}

CLASS2_BLACK_INV = {
    0 : 'blackcommon',
    1 : 'blackHuge',
    2 : 'blackStar'
}

CLASS2_WHITE = {
    'whitebig' : 0,
    'whitecommon' : 1,
    'whiteline' : 2,
}

CLASS2_WHITE_INV = {
    0 : 'whitebig',
    1 : 'whitecommon',
    2 : 'whiteline'
}

data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=360.,
                     width_shift_range=0.3,
                     height_shift_range=0.3,
                     fill_mode='reflect',
                     cval = 0,
                     zoom_range=0.2,
                     horizontal_flip=False,
                     vertical_flip=False)


def split_mask_v1(mask):
    thresh = mask.copy().astype(np.uint8)
    im2, contours, hierarchy = cv2.findContours(thresh, 2, 1)
    i = 0 
    for contour in contours:
        if  cv2.contourArea(contour) > 20:
            hull = cv2.convexHull(contour, returnPoints = False)
            defects = cv2.convexityDefects(contour, hull)
            if defects is None:
                continue
            points = []
            dd = []

            #
            # In this loop we gather all defect points 
            # so that they can be filtered later on.
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                d = d / 256
                #print('s[{}]{} e[{}]{} f[{}]{} {}'.format(s, start, e, end, f, far, d))
                dd.append(d)
                
            ddmax = np.max(dd)
            for i in range(len(dd)):
                s,e,f,d = defects[i,0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                if dd[i] > 4.0 and dd[i] / ddmax > 0.2:
                    xdis = abs(start[0] - end[0])
                    ydis = abs(start[1] - end[1])
                    Hor = xdis > ydis
                    if Hor:
                        ymin = min(start[1], end[1])
                        direction = far[1] < ymin
                    else:
                        xmin = min(start[0], end[0])
                        direction = far[0] < xmin
                        
                    if Hor:
                        mov = np.array([0, -1 if direction else 1])
                    else:
                        mov = np.array([-1 if direction else 1, 0])
                        
                    points.append((f, dd[i], Hor, mov))
            
            if len(points) >= 2:
                points.sort(key = lambda x: x[1], reverse=True)
                totcnt = len(points)
                used = []
                for i in range(totcnt):
                    f1 = points[i][0]
                    if f1 in used:
                        continue
                    if (totcnt - len(used)) > 1:
                        
                        p1 = tuple(contour[f1][0])
                        nearest = None
                        nearest_idx = -1
                        min_dist = np.inf
                        for j in range(len(points)):
                            if i != j:
                                f2 = points[j][0]                   
                                if f2 in used:
                                    continue
                                p2 = tuple(contour[f2][0])
                                dist = (p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]) 
                                """
                                if points[i][2] is True:
                                    # horizontal
                                    dist = abs(p1[0] - p2[0])
                                else:
                                    dist = abs(p1[1] - p2[1])
                                """
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest = p2
                                    nearest_idx = f2
                        nearest = np.array(nearest)
                        p1 = np.array(p1)
                        direction = nearest - p1
                        factor = abs(direction[0]) if abs(direction[0]) > abs(direction[1]) else abs(direction[1])
                        direction = (direction / factor).astype(np.int8)
                        p1 -= direction
                        nearest += direction
                        p1 = tuple(p1)
                        nearest = tuple(nearest)
                        
                    else:
                        p1 = tuple(contour[f1][0])
                        mov = points[i][3]
                        nearest = contour[f1][0].copy()
                        nearest += mov
                        while cv2.pointPolygonTest(contour, tuple(nearest), False) >= 0:
                            nearest += mov
                        nearest = tuple(nearest)
                        
                    cv2.line(thresh,p1, nearest, [0, 0, 0], 2)
                    used.append(nearest_idx)
                    used.append(f1)
    return thresh 

def split_and_relabel(mask):
    masks = []
    for i in np.unique(mask):
        if i == 0: # id = 0 is for background
            continue
        mask_i = (mask==i).astype(np.uint8)
        props = regionprops(mask_i, cache=False)
        if len(props) > 0:
            prop = props[0]
            if prop.convex_area/prop.filled_area > 1.1:
                mask_i = split_mask_v1(mask_i)
        masks.append(mask_i)

    masks = np.array(masks)
    masks_combined = np.amax(masks, axis=0)
    labels = sklabel(masks_combined, connectivity = 1)
    return labels

def scale_canal(canal):
    canal -= canal.min()
    imgmax = canal.max()
    if imgmax > 0:
        factor = (255 / imgmax) - 0.05
        canal = (canal * factor).astype(np.uint8)
    return canal

def scale_img_canals(img, IMG_CHANNELS=3):
    if len(img.shape) > 2:
        for i in range(IMG_CHANNELS):
            img[:,:,i] = scale_canal(img[:,:,i])
    else:
        img = scale_canal(img)
                
    return img

def scale_img_gray(img):
    img = img - img.min()
    img_max = img.max()
    if img_max > 0:
        factor = 255 / img_max
        img = (img * factor).astype(np.uint8)
    return img

def processImg(img):
    from skimage.exposure import equalize_adapthist
    if img.mean() > 100:
        img = np.invert(img)

    img = equalize_adapthist(img)
    img = scale_img_canals(img)
    
    return img

# Function read train images and mask return as nump array

def read_train_data(basepath, IMG_SIZE=256, IMG_CHANNELS=1, flex=''):
    X_train = np.zeros((len(train_ids), IMG_SIZE, IMG_SIZE, IMG_CHANNELS), dtype=np.float32)
    Y_train = np.zeros((len(train_ids), IMG_SIZE, IMG_SIZE,1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    
    train_path = "obj/train_img{}[{}].npy".format(flex, IMG_SIZE)
    mask_path = "obj/train_mask{}[{}].npy".format(flex, IMG_SIZE)
    if os.path.isfile(train_path) and os.path.isfile(mask_path):
        print("\nTrain file loaded from memory")
        X_train = np.load(train_path)
        Y_train = np.load(mask_path)
        return X_train,Y_train
    
    print("Generate train data from original")
    a = Progbar(len(train_ids))
    for n, id_ in enumerate(train_ids):
        path = basepath + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,0]
        img = resize(img, (IMG_SIZE, IMG_SIZE), mode='constant', preserve_range=True).astype(np.uint8)
        img = processImg(img)
        img = img.astype(np.float32) / 255
        X_train[n] = img[:,:,np.newaxis]
        
        mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            if len(mask_.shape) > 2:
                mask_ = mask_[:,:,0]
            mask_ = np.expand_dims(resize(mask_, (IMG_SIZE, IMG_SIZE), mode='constant', 
                                        preserve_range=True).astype(np.uint8), axis=-1)
            mask = np.maximum(mask, mask_)
            #mask = ndimage.grey_opening(mask.squeeze(), size=(3,3)).reshape((IMG_HEIGHT, IMG_WIDTH, 1))
       
        Y_train[n] = mask
        a.update(n)
    np.save(train_path, X_train)
    np.save(mask_path, Y_train)
    return X_train,Y_train

# Function to read test images and return as numpy array
def read_test_data(IMG_SIZE=CLASS1_SIZE, IMG_CHANNELS=1):
    X_test = np.zeros((len(test_ids), IMG_SIZE, IMG_SIZE, IMG_CHANNELS), dtype=np.float32)
    sizes_test = []
    
    test_img_path = "obj/test_img{}.npy".format(IMG_SIZE)
    test_size_path = "obj/test_size{}.npy".format(IMG_SIZE)
    
    if os.path.isfile(test_img_path) and os.path.isfile(test_size_path):
        X_test = np.load(test_img_path)
        sizes_test = np.load(test_size_path)
    else:
        b = Progbar(len(test_ids))
        for n, id_ in enumerate(test_ids):
            path = TEST_PATH + id_
            img = imread(path + '/images/' + id_ + '.png')[:,:,0]
            sizes_test.append([img.shape[0], img.shape[1]])
            img = resize(img, (IMG_SIZE, IMG_SIZE), mode='constant', preserve_range=True).astype(np.uint8)
            img = processImg(img)
            img = img.astype(np.float32) / 255
    
            X_test[n] = img[:,:,np.newaxis]
            
            b.update(n)
            
        np.save(test_img_path, X_test)
        np.save(test_size_path, sizes_test)
    return X_test,sizes_test


# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

"""
def rle_encoding(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return runs
""" 
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    return img.reshape(shape)

def prob_to_rles(x, cutoff=0.5):
    """
    lab_img = sklabel(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)
    """    
    
    x = binary_fill_holes(x > cutoff)
    """
    lab_img, nlabels = ndimage.label(x > cutoff)
    for i in range(1, nlabels + 1):
        yield rle_encoding(lab_img == i)
    """    
    lab_img = split_and_relabel(x > cutoff)# 
    for i in range(1, np.max(np.unique(lab_img)) + 1):
        yield rle_encoding(lab_img == i)
    #mask = ndimage.grey_opening(mask.squeeze(), size=(3,3)).reshape((IMG_HEIGHT, IMG_WIDTH, 1))
    

# Iterate over the test IDs and generate run-length encodings for each seperate mask identified by skimage
def mask_to_rle(preds_test_upsampled, ids, cutoff):
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(ids):
        rle = list(prob_to_rles(preds_test_upsampled[n], cutoff))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))
    return new_test_ids,rles

# Metric function
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Loss funtion
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_coef_weighted(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    
    negtrue = K.sum((1 - y_true_f) * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + negtrue + smooth)

def dice_coef_weighted_loss(y_true, y_pred):
    return -dice_coef_weighted(y_true, y_pred)

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def get_unet(IMG_SIZE, BASE_SIZE = 16, IMG_CHANNELS=1):
    inputs = Input((IMG_SIZE, IMG_SIZE, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255) (inputs)
    c1 = Conv2D(BASE_SIZE, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(BASE_SIZE, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    c2 = Conv2D(BASE_SIZE * 2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(BASE_SIZE * 2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(BASE_SIZE * 4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(BASE_SIZE * 4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(BASE_SIZE * 8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(BASE_SIZE * 8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(BASE_SIZE * 16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(BASE_SIZE * 16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(BASE_SIZE * 8, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(BASE_SIZE * 8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(BASE_SIZE * 8, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(BASE_SIZE * 4, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(BASE_SIZE * 4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(BASE_SIZE * 4, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(BASE_SIZE * 2, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(BASE_SIZE * 2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(BASE_SIZE * 2, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(BASE_SIZE, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(BASE_SIZE, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(BASE_SIZE, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

def unet_down(y, kernel_size, dropout):
    shortcut = y
    y = Conv2D(kernel_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (y)
    y = Dropout(dropout) (y)
    y = Conv2D(kernel_size, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (y)
    return concatenate([y, shortcut])

def unet_up(y, x, kernel_size, dropout):
    y = Conv2DTranspose(kernel_size, (2, 2), strides=(2, 2), padding='same') (y)
    y = concatenate([y, x])
    return y

def get_unet_deeper(IMG_SIZE, BASE_SIZE = 16, IMG_CHANNELS=3):
    inputs = Input((IMG_SIZE, IMG_SIZE, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255) (inputs)

    c1 = unet_down(s, BASE_SIZE, 0.1)
    c1 = unet_down(s, BASE_SIZE, 0.1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = unet_down(p1, BASE_SIZE * 2, 0.1)
    c2 = unet_down(p1, BASE_SIZE * 2, 0.1)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = unet_down(p2, BASE_SIZE * 4, 0.2)
    c3 = unet_down(p2, BASE_SIZE * 4, 0.2)
    c3 = unet_down(p2, BASE_SIZE * 4, 0.2)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = unet_down(p3, BASE_SIZE * 8, 0.2)
    c4 = unet_down(p3, BASE_SIZE * 8, 0.2)
    c4 = unet_down(p3, BASE_SIZE * 8, 0.2)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = unet_down(p4, BASE_SIZE * 16, 0.3)
    c5 = unet_down(p4, BASE_SIZE * 16, 0.3)
    c5 = unet_down(p4, BASE_SIZE * 16, 0.3)
    c5 = unet_down(p4, BASE_SIZE * 16, 0.3)


    c6 = unet_up(c5, c4, BASE_SIZE * 8, 0.2)
    c6 = unet_down(c6, BASE_SIZE * 8, 0.2)
    c6 = unet_down(c6, BASE_SIZE * 8, 0.2)
    c6 = unet_down(c6, BASE_SIZE * 8, 0.2)
    
    c7 = unet_up(c6, c3, BASE_SIZE * 4, 0.2)
    c7 = unet_down(c7, BASE_SIZE * 4, 0.2)
    c7 = unet_down(c7, BASE_SIZE * 4, 0.2)
    c7 = unet_down(c7, BASE_SIZE * 4, 0.2)

    c8 = unet_up(c7, c2, BASE_SIZE * 2, 0.1)
    c8 = unet_down(c8, BASE_SIZE * 2, 0.1)
    c8 = unet_down(c8, BASE_SIZE * 2, 0.1)

    c9 = unet_up(c8, c1, BASE_SIZE, 0.1)
    c9 = unet_down(c9, BASE_SIZE, 0.1)
    c9 = unet_down(c9, BASE_SIZE, 0.1)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

def totalunet():
    # get train_data
    train_img,train_mask = read_train_data()
    
    
    # get test_data
    test_img,test_img_sizes = read_test_data()
    
    #@title Default title text
    # get u_net model
    u_net = get_unet()
    
    # fit model on train_data
    print("\nTraining...")
    u_net.fit(train_img,train_mask,batch_size=16,epochs=epochs)
    u_net.save('unet_12.h5')
    
    print("Predicting")
    # Predict on test data
    test_mask = u_net.predict(test_img,verbose=1)
    
    # Create list of upsampled test masks
    test_mask_upsampled = []
    for i in range(len(test_mask)):
        test_mask_upsampled.append(resize(np.squeeze(test_mask[i]),
                                           (test_img_sizes[i][0],test_img_sizes[i][1]), 
                                           mode='constant', preserve_range=True).astype(np.uint8))
    
    test_ids,rles = mask_to_rle(test_mask_upsampled, test_ids, 0.5)
    
    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    
    sub.to_csv('nucles.csv', index=False)

def readmask(paths, IMG_SIZE, erosion = False):
    mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.bool)
    for mask_file in paths:
        mask_ = imread(mask_file)
        if len(mask_.shape) > 2:
            mask_ = mask_[:,:,0]
        
        mask_ = np.expand_dims(resize(mask_, (IMG_SIZE, IMG_SIZE), mode='constant', preserve_range=True).astype(np.uint8), axis=-1)
        
        if erosion:
            loc = ndimage.find_objects(mask_ > 0)[0]

            extend = 10
            new_loc = (
                slice(
                    max(0, loc[0].start - extend),
                    min(mask.shape[0], loc[0].stop + extend),
                    None
                ),
                slice(
                    max(0, loc[1].start - extend),
                    min(mask.shape[1], loc[1].stop + extend),
                    None
                ),
                slice(0, 1, None)
            )

            sub_mask = mask[new_loc]

            if sub_mask.max() > 0 and np.count_nonzero(mask_) > 50:
                cnt = 0
                while np.count_nonzero(mask_[loc]) > 50 and cnt < 10:
                    cnt += 1
                    mask_ = ndimage.binary_erosion(mask_, structure=np.ones((2,2,1))).astype(mask_.dtype)
                #print('after erosion {}'.format(np.count_nonzero(mask_)))

        mask_ = np.where(mask_ > 0, True, False).astype(np.bool)
        mask = np.maximum(mask, mask_)

        #mask = ndimage.grey_opening(mask.squeeze(), size=(3,3)).reshape((IMG_SIZE, IMG_SIZE, 1))
        #mask = ndimage.binary_opening(mask, iterations=4)
   
    return mask

def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp
        
def classifier_reader(path = './input/detail', flex = '', fillhole = True):
    img_dict_path = 'img_dict_noresize{}'.format(flex)
    l1_dict_path = 'l1_dict_noresize{}'.format(flex)
    l2_dict_path = 'l2_dict_noresize{}'.format(flex)
    
    if os.path.isfile('obj/{}.dl'.format(img_dict_path)) and\
        os.path.isfile('obj/{}.dl'.format(l1_dict_path)) and \
        os.path.isfile('obj/{}.dl'.format(l2_dict_path)):
        print('read from file')
        img_dict = load_obj(img_dict_path)
        l1_dict = load_obj(l1_dict_path)
        l2_dict = load_obj(l2_dict_path)
        return img_dict, l1_dict, l2_dict
    
    img_dict = {
        'image': [],
        'mask': [],
        'label': [],
        'size': [],
        'class1': [],
        'class2': []
    }
    
    l1_dict = {

    }
    
    l2_dict = {
    }
    
    def fillclassdict(some_dict, img, cls, img_size):
        #img = resize(img, (img_size, img_size), 
        #           mode='constant', preserve_range=True).astype(np.uint8)
        if cls not in some_dict:
            some_dict[cls] = {'image':[], 'mask':[]}
        some_dict[cls]['image'].append(img)

    for root, dirs, files in os.walk(path):
        if 'images' in root and len(files) == 1:
            f = files[0]
            path = os.path.join(root, f)
            img = imread(path)[:,:,0]

            img = processImg(img)[:,:,np.newaxis]
            img = img.astype(np.float32) / 255
            
            spt = root.split('\\')
            ext = os.path.splitext(str(f))
            
            img_dict['image'].append(img)
            img_dict['label'].append(ext[0])
            
            class1 = spt[1]
            class2 = spt[2]
            img_dict['class1'].append(class1)
            img_dict['class2'].append(class2)
            fillclassdict(l1_dict, img, class1, CLASS1_SIZE)
            
            if 'black' in class2:
                #print(class2)
                fillclassdict(l2_dict, img, class2, CLASS2_SIZE)
            
            print('{} {} {} {}'.format(class1, class2, img.max(), ext[0]))
            
        elif 'masks' in root and len(files) > 0:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool)
            for mask_file in (os.path.join(root, f) for f in files):
                mask_ = imread(mask_file)
                if len(mask_.shape) > 2:
                    mask_ = mask_[:,:,0].astype(np.uint8)
                mask = np.maximum(mask, mask_)
            
            if fillhole:
                mask = binary_fill_holes(mask)
            mask = mask[:,:,np.newaxis]
            img_dict['mask'].append(mask)
            
            spt = root.split('\\')
            class1 = spt[1]
            class2 = spt[2]
            l1_dict[class1]['mask'].append(mask)
            
            if 'black' in class2:
                #print(class2)
                l2_dict[class2]['mask'].append(mask)     
            
    save_obj(img_dict, img_dict_path)
    save_obj(l1_dict, l1_dict_path)
    save_obj(l2_dict, l2_dict_path)

    return img_dict, l1_dict, l2_dict

def class1Model(output_size, CHANNEL = 3):
    from model import conv_block, identity_block
    
    inp_img = Input(shape=(TYPE_CLASS_SIZE, TYPE_CLASS_SIZE, CHANNEL))

    # 51
    
    y = conv_layer(inp_img, 32, kernel_size=(5, 5), padding='same')
    y = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(y)
    
    y = conv_block(y, 3, [64, 64, 256], stage=1, block='a', strides=(1, 1))
    y = identity_block(y, 3, [64, 64, 256], stage=2, block='b')
    y = identity_block(y, 3, [64, 64, 256], stage=3, block='b')
    y = identity_block(y, 3, [64, 64, 256], stage=4, block='b')
    y = identity_block(y, 3, [64, 64, 256], stage=5, block='b')
    
    y = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(y)

    # 1
    # dense layers
    flt = Flatten()(y)
    ds1 = dense_set(flt, 64, activation='relu', drop_rate=0.0)
    ds2 = dense_set(ds1, 32, activation='relu', drop_rate=0.0)
    out = dense_set(ds2, output_size, activation='softmax')

    #y = Conv2D(filters=64, kernel_size=(1,1), strides=(1,1), padding='same')(y)
    
    model = Model(inputs=inp_img, outputs=out)
    #mypotim = Adam(lr=2 * 1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])
    #model.summary()
    return model

def argument_image_only(imgs, argu_cnt, IMG_SIZE=64, CHANNEL = 1):
    
    chunk_size = len(imgs)
    shape = (chunk_size * argu_cnt, IMG_SIZE, IMG_SIZE)
    if CHANNEL > 1:
        shape = (chunk_size * argu_cnt, IMG_SIZE, IMG_SIZE, CHANNEL)
    output = np.zeros(shape, dtype=np.uint8)

    print('Argument train images only ... ')
    sys.stdout.flush()
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    npimgs = np.array(imgs)
    image_generator = image_datagen.flow(npimgs, batch_size=chunk_size)

    a = Progbar(argu_cnt)
    for i in range(argu_cnt):
        chunk = image_generator.next()
        output[i * chunk_size: (i + 1) * chunk_size] = chunk.reshape(chunk_size, IMG_SIZE, IMG_SIZE, CHANNEL)
        a.update(i)
    output = np.split(output, chunk_size * argu_cnt)
    for idx, img in enumerate(output):
        output[idx] = output[idx].squeeze()
    return output

def scaleImgs(imgs, size, channel):
    print('\nscale {} images to {}'.format(len(imgs), size))
    
    shape = (size, size)
    if channel > 1:
        shape = (size, size, channel)
    
    scaled_imgs = []
    for idx, img in enumerate(imgs):
        img_sized = resize(img, shape, mode='constant',preserve_range=True).astype(np.uint8)
        #print(img_sized.max())
        scaled_imgs.append(img_sized)
    return scaled_imgs

def argumentByType(class_dict, IMG_SIZE, CHANNEL = 1):
    scaled_imgs = []
    labels = []

    cnts = [len(class_dict[key]['image']) for key in class_dict.keys()]
    max_cnt = max(cnts)
    for key in class_dict.keys():
        subimgs = class_dict[key]['image']
        scaled_subimgs = scaleImgs(subimgs, IMG_SIZE, CHANNEL)

        scaled_imgs += scaled_subimgs
        labels += [key] * len(scaled_subimgs)

        argucnt = min(10, int(max_cnt / len(scaled_subimgs) - 1))
        if argucnt > 0:
            newimgs = argument_image_only(scaled_subimgs, argucnt, IMG_SIZE, CHANNEL)
            scaled_imgs += newimgs
            labels += [key] * len(newimgs)
    return scaled_imgs, labels

def train_model(img, target, func, name, epochs = 50, arguratio = 5, BATCH_SIZE = 64):
    path = 'model/{}.h5'.format(name)

    img = np.array(img)
    print('\ntrain model with {} imgs {}\n'.format(len(img), img.shape))

    channel = 1
    if len(img.shape) > 3:
        channel = img.shape[3]
    gmodel = func(target.shape[1], channel)
    gmodel.fit(img, target, BATCH_SIZE, epochs=epochs,verbose=1,shuffle=True)
    """
    #gmodel.load_weights(filepath='model_weight_Adam.hdf5')
    x_train, x_valid, y_train, y_valid = train_test_split(
                                                        img,
                                                        target,
                                                        shuffle=True,
                                                        train_size=0.95,
                                                        #random_state=RANDOM_STATE
                                                        )
    
    gen = ImageDataGenerator(**data_gen_args)
    gmodel.fit_generator(gen.flow(x_train, y_train,batch_size=BATCH_SIZE),
               steps_per_epoch=arguratio*len(x_train)/BATCH_SIZE,
               epochs=epochs,
               verbose=1,
               shuffle=True,
               validation_data=(x_valid, y_valid))
    """
    gmodel.save(path)
    K.clear_session()

    
def trainLevelUNet(basepath, level_dict, flex, imgsize, epochs = 50, batch_size=16, 
                   exceptlist = [], argucnt = 20, batch_max = 5,
                   tilesize = 256, cutoff = 230, loadTrain = False):
    #dict_unet = {}
    
    train_keys = []
    for key in level_dict.keys():
        if key not in exceptlist:
            train_keys.append(key)
            
    cnts = [len(level_dict[key]['image']) for key in train_keys]
    max_cnt = max(cnts)
    print('{}\nmax_cnt = {}'.format(cnts, max_cnt))
    for key, subdict in level_dict.items():
        if key in exceptlist:
            print('skip ' + key)
            continue
        
        img = np.array(subdict['image'])
        mask = np.array(subdict['mask'])
        img, mask = splitTrain(img, mask, tilesize, '{}_{}'.format(flex, key), cutoff)
        
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
        seed = 5
        image_generator = image_datagen.flow(img, batch_size = batch_size,seed=seed)
        mask_generator = mask_datagen.flow(mask, batch_size = batch_size,seed=seed)
        train_generator = zip(image_generator, mask_generator)
        
        model_path = 'model/unet{}_{}_{}.h5'.format(flex, key, imgsize)
        u_net = None
        if os.path.isfile(model_path):
            if not loadTrain:
                print('unet_{} exist, continue...'.format(key))
                #dict_unet[key] = load_model(model_path, custom_objects={'dice_coef': dice_coef})
                continue
            else:
                print('\nload unet_{} to train'.format(key))
                #u_net = load_model(model_path, custom_objects={'dice_coef': dice_coef})

                u_net = get_unet(IMG_SIZE = 256, IMG_CHANNELS = 1, BASE_SIZE = 16)
                #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
                adamax = Adamax(lr=0.0005)
                u_net.compile(optimizer=adamax,loss=dice_coef_loss, metrics=[dice_coef])
                u_net.load_weights(model_path)
        
        if u_net is None:
            if basepath is None:
                print('\ntrain from scrach\n')
                u_net = get_unet(IMG_SIZE = 256, IMG_CHANNELS = 1, BASE_SIZE = 16)
                #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
                #u_net.compile(optimizer='adam',loss='binary_crossentropy', metrics=[dice_coef])
                u_net.compile(optimizer='adamax',loss=dice_coef_loss, metrics=[dice_coef])
                
            else:
                print('\nload base model from ' + basepath + '\n')
                #u_net = load_model(basepath, custom_objects={'dice_coef': dice_coef})
                u_net = get_unet(IMG_SIZE = 256, IMG_CHANNELS = 1, BASE_SIZE = 16)
                u_net.compile(optimizer='adamax',loss=dice_coef_loss, metrics=[dice_coef])
                u_net.load_weights(basepath)
        
        print("\nTraining unet_{}...\n".format(key))
        steps = len(img) / batch_size
        steps = 1 if steps < 1 else steps
        steps = int(steps * batch_max)
        if steps * batch_size < 100:
            steps = 20
        history = u_net.fit_generator(train_generator, steps_per_epoch = steps, epochs=epochs)
        
        df_history = pd.DataFrame(history.history)
        df_history.to_csv(model_path + '_history.csv')
        u_net.save(model_path)
        K.clear_session()
        
    return# dict_unet

def plotOrigVsPred(imgs):
    cnt = len(imgs)
    f,ax = plt.subplots(1, cnt)
    for i in range(cnt):
        ax[i].imshow(imgs[i].squeeze())
        
def plotClassImgMas(dic, subtype, idx):
    plotOrigVsPred([dic[subtype]['image'][idx], dic[subtype]['mask'][idx]])
        
def plotTrainSample(i):
    plotOrigVsPred([img_dict['image'][i], img_dict['mask'][i]])
        
def plotTrainResult(i):
    plotOrigVsPred([img_dict['image'][i], img_dict['mask'][i], valid_mask[i]])
    
def plotClassicPreResult(i, cutoff = 0.5):
    plotOrigVsPred([X_pre[i], valid_mask[i], binary_fill_holes(valid_mask[i] > cutoff)])
    
def plotClassiCombine(i, mask_cutoff = 0.5, border_cutoff = 0.5):
    plotOrigVsPred([X_pre[i], valid_mask[i], valid_border[i], combine_mask[i]])
    
def plotTrainResultByLabel(label):
    idx = img_dict['label'].index(label)
    plotTrainResult(idx)
    
def plotPreResult(idx, cutoff = 0.5):
    plotOrigVsPred([X_pre[idx], test_mask[idx], test_mask[idx] > cutoff])
    
def plotOrigTestAndFilter(idx, filterfunc):
    plotOrigVsPred([test_img[idx], filterfunc(test_img[idx])])
    
def plotBatchArgument(imgs,masks):
    cnt = len(imgs)
    f,ax = plt.subplots(2, cnt)
    for i in range(cnt):
        ax[0,i].imshow(imgs[i].squeeze())
        ax[1,i].imshow(masks[i].squeeze())
    
def plotImageMaskGenerator(image_generator, mask_generator):
    plotBatchArgument(image_generator.next().astype(np.uint8), mask_generator.next().astype(np.uint8))
    
def plotMaskBoderCombine(idx, mask_cutoff = 0.5, border_cutoff = 0.5):
    mask_cut = binary_fill_holes(test_mask[idx] > mask_cutoff)
    border_cut = test_border[idx] > border_cutoff
    plotOrigVsPred([X_pre[idx], mask_cut, border_cut, 
                    binary_fill_holes(np.logical_and(mask_cut, np.logical_not(border_cut)))])
    
    
def predictImgType(model, img):
    img4d = resize(img, (TYPE_CLASS_SIZE, TYPE_CLASS_SIZE), mode='constant', preserve_range=True).astype(np.float32)
    img4d = img4d[np.newaxis, :, :, :]
    prob = model.predict(img4d, verbose=0)
    return prob.argmax(axis=-1)[0]
    
def predictImages(tarimgs, tarids, tile_size, model_c1, model_c2, dict_unetl1, dict_unetl2):
    test_mask = []
    for idx, img in enumerate(tarimgs):
        img4d = resize(img, (TYPE_CLASS_SIZE, TYPE_CLASS_SIZE), mode='constant', preserve_range=True).astype(np.float32)
        img4d = img4d[np.newaxis, :, :, :]
        prob = model_c1.predict(img4d, verbose=0)
        pred = prob.argmax(axis=-1)[0]
        
        if pred == 0:
            prob2 = model_c2.predict(img4d, verbose=0)
            pred2 = prob2.argmax(axis=-1)[0]
            unet_l2 = dict_unetl2[CLASS2_BLACK_INV[pred2]]
            print('\n{} type1[{}] type2 [{}]'.format(tarids[idx], pred, pred2))
            
            pre_mask = scaleAndPredictTile(unet_l2, img, tile_size)
            #pre_mask = unet_l2.predict(img[np.newaxis, :, :, :],verbose=0)
        else:
            print('\n{} type1 [{}]'.format(tarids[idx], pred))
            unet_l1 = dict_unetl1[CLASS1_INV[pred]]
            
            pre_mask = scaleAndPredictTile(unet_l1, img, tile_size)
            #pre_mask = unet_l1.predict(img[np.newaxis, :, :, :],verbose=0)
        test_mask.append(pre_mask)
        
    return test_mask

def genRLE_base(mask, ids, cutoff):
    test_ids,rles = mask_to_rle(mask, ids, cutoff)
    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    return sub

def genRLEandSave(mask, orig_img_sizes, ids, imgsize, cutoff, filename):
    test_mask_upsampled = []
    for i in range(len(mask)):
        test_mask_upsampled.append(resize(np.squeeze(mask[i]),
                                           (orig_img_sizes[i][0],orig_img_sizes[i][1]), 
                                           mode='constant', preserve_range=True))
    
    
    sub = genRLE_base(test_mask_upsampled, ids, cutoff)
    sub.to_csv('{}[{},{}].csv'.format(filename, imgsize, cutoff), index=False)

def createMaskBorder(mask_, thick = 1):
    mask_ = binary_fill_holes(mask_).astype(np.uint8) * 255
    tmp = mask_
    tmp = np.maximum(tmp, ndimage.interpolation.shift(mask_, [-thick, thick]))
    tmp = np.maximum(tmp, ndimage.interpolation.shift(mask_, [-thick, -thick]))
    tmp = np.maximum(tmp, ndimage.interpolation.shift(mask_, [thick, thick]))
    tmp = np.maximum(tmp, ndimage.interpolation.shift(mask_, [thick, -thick]))
    tmp = np.maximum(tmp, ndimage.interpolation.shift(mask_, [-thick, 0]))
    tmp = np.maximum(tmp, ndimage.interpolation.shift(mask_, [thick, 0]))
    tmp = np.maximum(tmp, ndimage.interpolation.shift(mask_, [0, thick]))
    tmp = np.maximum(tmp, ndimage.interpolation.shift(mask_, [0, -thick]))
    tmp -= mask_
    return tmp.astype(np.bool)
    

def genStackMask():
    from shutil import copyfile
    CLEAN_PATH = TRAIN_PATH
    IMG_CHANNELS=3
    a = Progbar(len(train_ids))
    for n, id_ in enumerate(train_ids):
        path = TRAIN_PATH_OLD + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    
        directory = CLEAN_PATH + id_
        if not os.path.exists(directory):
            os.makedirs(directory)
    
        imgdir = directory + '/images'
        if not os.path.exists(imgdir):
            os.makedirs(imgdir)
        copyfile(path + '/images/' + id_ + '.png', imgdir + '/' + id_ + '.png')
        
        maskdir = directory + '/masks'
        if not os.path.exists(maskdir):
            os.makedirs(maskdir)

        maskpath = maskdir + '/mask.png'
        if not os.path.isfile(maskpath):
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file)
                mask = np.maximum(mask, mask_)
                border = createMaskBorder(mask_, thick = 2)
                mask = np.logical_and(border, np.logical_not(mask))
            imsave(maskpath, mask.astype(np.uint8) * 255)
        a.update(n)
        
def copy_clean_mask():
    from shutil import copyfile
    srcdir = 'input/stage1_train_border1/'
    outdir = 'input/detail_clean_border'
    rootdir = './input/detail'
    len_root = len(rootdir)
    srclen = len(srcdir)
    for root, dirs, files in os.walk(rootdir):
        for d in dirs:
            fullpath = os.path.join(root, d)
            relativedir = fullpath[len_root:]
            tarpath = outdir + relativedir
            if not os.path.exists(tarpath):
                os.makedirs(tarpath)

        if len(files) > 0:
            spt = root.split('\\')
            srcpath = srcdir + spt[-2] + '/' + spt[-1]
            tarpath = outdir + root[len_root:]
            for r, dd, f in os.walk(srcpath):
                for ff in f:
                    srcfile = os.path.join(r, ff)
                    tarfile = os.path.join(tarpath, ff)
                    copyfile(srcfile, tarfile)

border_path = 'input/stage1_train_border1/'
#train_path = TRAIN_PATH_OLD
thick = 1
def genBorder(thick, train_path, border_path):
    from shutil import copyfile
    IMG_CHANNELS=3
    
    a = Progbar(len(train_ids))
    for n, id_ in enumerate(train_ids):
        path = train_path + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    
        directory = border_path + id_
        if not os.path.exists(directory):
            os.makedirs(directory)
    
        imgdir = directory + '/images'
        if not os.path.exists(imgdir):
            os.makedirs(imgdir)
        copyfile(path + '/images/' + id_ + '.png', imgdir + '/' + id_ + '.png')
        
        maskdir = directory + '/masks'
        if not os.path.exists(maskdir):
            os.makedirs(maskdir)
    
        maskpath = maskdir + '/mask.png'
        if not os.path.isfile(maskpath):
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_ = imread(path + '/masks/' + mask_file)
                border = createMaskBorder(mask_, thick = thick)
                mask = np.maximum(mask, border)
            mask = mask.astype(np.uint8) * 255
            imsave(maskpath, mask)
        a.update(n)
        


def genBorderFromResult(thick, train_path, border_path):
    from shutil import copyfile

    a = Progbar(len(train_ids))
    for n, id_ in enumerate(train_ids):
        path = TRAIN_PATH + id_
    
        directory = border_path + id_
        if not os.path.exists(directory):
            os.makedirs(directory)
    
        imgdir = directory + '/images'
        if not os.path.exists(imgdir):
            os.makedirs(imgdir)
        copyfile(path + '/images/' + id_ + '.png', imgdir + '/' + id_ + '.png')
        
        maskdir = directory + '/masks'
        if not os.path.exists(maskdir):
            os.makedirs(maskdir)
    
        maskpath = maskdir + '/mask.png'
        if not os.path.isfile(maskpath):
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                mask_path = path + '/masks/' + mask_file
                mask_ = imread(mask_path)
                if len(mask_.shape) > 2:
                    mask_ = mask_[:,:,0]
                border = createMaskBorder(mask_, thick = thick)
            border = border.astype(np.uint8) * 255
            imsave(maskpath, border)
        a.update(n)
        
def trainResizeModel(model_path, train_data_path, img_size, data_flex, batch_size = 12, loadTrain = False):
    if os.path.isfile(model_path):
        print('load model from file for training')
        u_net = get_unet(IMG_SIZE = img_size, BASE_SIZE = 16)
        u_net.compile(optimizer='adam',loss=dice_coef_loss, metrics=[dice_coef])
        u_net.load_weights(model_path)
        
        if not loadTrain:
            return u_net
    else:
        u_net = get_unet(IMG_SIZE = img_size, BASE_SIZE = 16)
        u_net.compile(optimizer='adam',loss='binary_crossentropy', metrics=[dice_coef])
        
    
    train_img, train_mask = read_train_data(train_data_path, IMG_SIZE=img_size, flex=data_flex)
        
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    seed = 5
    image_generator = image_datagen.flow(train_img, batch_size = batch_size,seed=seed)
    mask_generator = mask_datagen.flow(train_mask, batch_size = batch_size,seed=seed)
    train_generator = zip(image_generator, mask_generator)
        
    u_net.fit_generator(train_generator, steps_per_epoch = int(total_size / batch_size), epochs=epochs)
    u_net.save(model_path)
    return u_net
        

def trainModelPredict(img_size, model_path, train_data_path, data_flex, test_img, batch_size = 12):
    if os.path.isfile(model_path):
        print('load model from file...')
        u_net = load_model(model_path, 
                           custom_objects={'dice_coef': dice_coef})
    else:
        u_net = get_unet(IMG_SIZE = img_size, BASE_SIZE = 16)
        u_net.compile(optimizer='adam',loss=dice_coef_loss, metrics=[dice_coef])
    
        train_img, train_mask = read_train_data(train_data_path, IMG_SIZE=img_size, flex=data_flex)
        
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
        seed = 5
        image_generator = image_datagen.flow(train_img, batch_size = batch_size,seed=seed)
        mask_generator = mask_datagen.flow(train_mask, batch_size = batch_size,seed=seed)
        train_generator = zip(image_generator, mask_generator)
        
        u_net.fit_generator(train_generator, steps_per_epoch = int(total_size / batch_size), epochs=epochs)
        u_net.save(model_path)
    
    print("Predicting " + data_flex)
    test_mask = u_net.predict(test_img,verbose=1, batch_size=4)
    return test_mask

def read_test_noresize():
    X_test = []
    xname = 'test_img_noresize'
    test_img_path = "obj/{}.dl".format(xname)
    
    if os.path.isfile(test_img_path):
        X_test = load_obj(xname)
    else:
        b = Progbar(len(test_ids))
        for n, id_ in enumerate(test_ids):
            path = TEST_PATH + id_
            img = imread(path + '/images/' + id_ + '.png')[:,:,0]
            img = processImg(img)
            img = img.astype(np.float32) / 255
    
            X_test.append(img[:,:,np.newaxis])
            
            b.update(n)
            
        save_obj(X_test, xname)
    return X_test

def read_train_noresize(basepath, flex, fill = True):
    train_name = 'train_img_noresize[{}]'.format(flex)
    train_path = "obj/{}.dl".format(train_name)
    mask_name = 'train_mask_noresize[{}]'.format(flex)
    mask_path = "obj/{}.dl".format(mask_name)
    if os.path.isfile(train_path) and os.path.isfile(mask_path):
        print("{} file loaded from memory".format(flex))
        sys.stdout.flush()
        
        X_train = load_obj(train_name)
        Y_train = load_obj(mask_name)
        return X_train,Y_train
    
    print("{} generate".format(flex))
    sys.stdout.flush()
        
    X_train = []
    Y_train = []
    a = Progbar(len(train_ids))
    for n, id_ in enumerate(train_ids):
        path = basepath + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,0]
        X_train.append(processImg(img)[:,:,np.newaxis])
        
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            if len(mask_.shape) > 2:
                mask_ = mask_[:,:,0].astype(np.uint8)
            mask = np.maximum(mask, mask_)

        if fill:
            mask = binary_fill_holes(mask)
        
        Y_train.append(mask[:,:,np.newaxis])
        a.update(n)
    save_obj(X_train, train_name)
    save_obj(Y_train, mask_name)
    return X_train,Y_train

def sliceImg(img, size, xcnt, ycnt):
    img = img.astype(np.float32)
    if img.max() > 200:
        img /= 255
    sliced = []
    slice_img = view_as_blocks(img, (size, size, 1)).reshape((xcnt * ycnt, size, size))
    for i, subimg in enumerate(slice_img):
        sliced.append(subimg[:,:,np.newaxis])
    return sliced

def splitTrain(X_train, Y_train, size = 128, flex = '', cutoff = 240):
    train_name = 'train_img_slice[{},{}]'.format(size, flex)
    train_path = "obj/{}.dl".format(train_name)
    mask_name = 'train_mask_slice[{},{}]'.format(size, flex)
    mask_path = "obj/{}.dl".format(mask_name)
    if os.path.isfile(train_path) and os.path.isfile(mask_path):
        print("{} sliced file loaded from memory".format(flex))
        sys.stdout.flush()
        
        img_sliced = load_obj(train_name)
        mask_sliced = load_obj(mask_name)
        return img_sliced, mask_sliced
    
    print("{} generate".format(flex))
    sys.stdout.flush()
    
    img_sliced = []
    mask_sliced = []
    a = Progbar(len(train_ids))
    for idx, img in enumerate(X_train):
        img = X_train[idx]
        mask = Y_train[idx]
        if mask.dtype == np.bool:
            mask = mask.astype(np.uint8) * 255
        xcnt = round(img.shape[0] / size)
        ycnt = round(img.shape[1] / size)
        xsize = xcnt * size
        ysize = ycnt * size
        if xsize != img.shape[0] or ysize != img.shape[1]:
            #print('[{}] resize from [{}] to [{}]'.format(idx, img.shape, (xsize, ysize)))
            img = resize(img, (xsize, ysize), mode='constant', 
                      preserve_range=True)
            mask = resize(mask, (xsize, ysize), mode='constant', 
                      preserve_range=True).astype(np.uint8)
            mask = (mask > cutoff).astype(np.uint8) * 255
        
        img_sliced += sliceImg(img, size, xcnt, ycnt)
        mask_sliced += sliceImg(mask, size, xcnt, ycnt)
        """
        for idx, mask in enumerate(mask_sliced):
            mask_sliced[idx] = mask.astype(np.bool)
        """
        
        a.update(idx)
        
    img_sliced = np.array(img_sliced)
    mask_sliced = np.array(mask_sliced)
    save_obj(img_sliced, train_name)
    save_obj(mask_sliced, mask_name)
                
    return img_sliced, mask_sliced

def predictSplited(model, splited, tile_size):
    res = np.zeros((splited.shape[0] * tile_size, splited.shape[1] * tile_size)).astype(np.float32)
    for x in range(splited.shape[0]):
        for y in range(splited.shape[1]):
            subres = model.predict(splited[x,y][np.newaxis, :, :, :])
            """
            plt.figure()
            plt.imshow(splited[x,y][np.newaxis, :, :, :].squeeze())
            
            plt.figure()
            plt.imshow(subres.squeeze())
            """
            xstart = tile_size * x
            xend = tile_size * (x + 1)
            ystart = tile_size * y
            yend = tile_size * (y + 1)
            #print('sub rect[{},{},{},{}], result[{}]'.format(xstart, xend, ystart, yend, subres.shape))
            res[xstart : xend, ystart : yend] = subres[0,:,:,0]
    return res

def scaleAndPredictTile(model, img, tile_size):
    original_size = (img.shape[0], img.shape[1])
    img = img.astype(np.float32)# / 255
    
    xcnt = round(img.shape[0] / tile_size)
    ycnt = round(img.shape[1] / tile_size)
    xsize = xcnt * tile_size
    ysize = ycnt * tile_size
    if xsize != img.shape[0] or ysize != img.shape[1]:
        #print('resize from {} to {}'.format(img.shape, (xsize, ysize)))
        img = resize(img, (xsize, ysize), mode='constant')
        
        #plt.figure()
        #plt.imshow(img.squeeze())
        
        sliced = view_as_blocks(img, (tile_size, tile_size, 1))

        res = predictSplited(model, sliced[:,:,0,:,:,:], tile_size)
        
        #print('resize from {} to {}'.format(img.shape, original_size))
        
        res = resize(res, original_size, mode='constant')
        #plt.figure()
        #plt.imshow(res.squeeze())
        
    else:
        sliced = view_as_blocks(img, (tile_size, tile_size, 1))
        res = predictSplited(model, sliced[:,:,0,:,:,:], tile_size)
        
    #plt.figure()
    #plt.imshow(res.squeeze())
    return res

def predictTestTile(model, X_pre, tile_size):
    res = []
    for img in X_pre:
        res.append(scaleAndPredictTile(model, img, tile_size))
    return res

def loadSliced(path, flex, tilesize, cutoff, fill):
    X_train, Y_train = read_train_noresize(path, flex, fill)
    X_train_sliced, Y_train_sliced = splitTrain(X_train, Y_train, tilesize, flex, cutoff)
    return X_train_sliced, Y_train_sliced 

def trainModelPredictNoResize(tile_size, model_path, X_train, Y_train, X_pre, 
                              base_size = 32, batch_size = 12, loadtrain = False):
    from keras.callbacks import History
    
    model_existed = False
    if os.path.isfile(model_path):
        model_existed = True
        print('load model from file...')
        sys.stdout.flush()
        
        u_net = get_unet(IMG_SIZE = tile_size, IMG_CHANNELS = 1, BASE_SIZE = base_size)
        adamax = Adamax(lr=0.0005)
        u_net.compile(optimizer=adamax,loss=dice_coef_loss, metrics=[dice_coef])
        u_net.load_weights(model_path)
        #u_net = load_model(model_path, custom_objects={'dice_coef': dice_coef})
        #'dice_coef_loss':dice_coef_loss, 
    else:
        u_net = get_unet(IMG_SIZE = tile_size, IMG_CHANNELS = 1, BASE_SIZE = base_size)
        #u_net = get_unet_deeper(IMG_SIZE = tile_size, IMG_CHANNELS = 1, BASE_SIZE = base_size)
        #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        u_net.compile(optimizer='adamax',loss='binary_crossentropy', metrics=[dice_coef])
        
    if not model_existed or loadtrain:
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
        seed = 5
        image_generator = image_datagen.flow(X_train, batch_size = batch_size,seed=seed)
        mask_generator = mask_datagen.flow(Y_train, batch_size = batch_size,seed=seed)
        train_generator = zip(image_generator, mask_generator)
        
        history = u_net.fit_generator(train_generator, steps_per_epoch = int(total_size / batch_size * 5), 
                            epochs=epochs)
        u_net.save(model_path)
        df_history = pd.DataFrame(history.history)
        df_history.to_csv(model_path + '_history.csv')
        
    #print("Predicting " + data_flex)
    #test_mask = u_net.predict(X_pre,verbose=1, batch_size=4)
    return u_net

def plotSlicePred(X_train, Y_train, model, idx):
    Y_res = model.predict(X_train[idx][np.newaxis,:,:,:])
    plotOrigVsPred([X_train[idx].squeeze(), 
                    Y_train[idx].squeeze(),
                    Y_res.squeeze()])
    
def plotSliceSample(X_train, Y_train, idx):
    plotOrigVsPred([X_train[idx].squeeze(), 
                    Y_train[idx].squeeze()])
    
def checkSize():
    for i in range(len(test_ids)):
        shape1 = X_pre[i][:,:,0].shape
        shape2 = test_mask[i][:,:].shape
        if shape1 != shape2:
            print(test_ids[i] + '{}  {}'.format(shape1, shape2))
            
def combineSlicedMaskAndBorder(masks, borders, mask_cutoff = 0.5, border_cutoff = 0.5):
    res = []
    for idx, mask in enumerate(masks):
        fillhole = binary_fill_holes(np.logical_or(mask > mask_cutoff, borders[idx] > border_cutoff))
        res.append(binary_fill_holes(np.logical_and(fillhole, np.logical_not(borders[idx] > border_cutoff))))
    return res

def trainClass1Classifier(level_dict, classstr, enum, imgsize, channel, epochs = 10, arguratio = 10):
    name = classstr
    path = 'model/{}.h5'.format(name)
    if os.path.isfile(path):
        print('model exist')
        return

    X_train, y_train = argumentByType(level_dict, imgsize, channel)
    y_train = to_categorical(np.array([enum[l] for l in y_train]))
    
    train_model(X_train, y_train, class1Model, classstr, 
                epochs = epochs, arguratio = arguratio, BATCH_SIZE = 128)
    
    return

def trainModels(img_dict, l1_dict, l2_dict, basemodel_path = None, flex = '', epochs = 20, loadTrain = False):
    print('train models start\n')
    trainClass1Classifier(l1_dict, 'class1', CLASS1, epochs = 20, arguratio = 10)
    trainClass1Classifier(l2_dict, 'class2', CLASS2_BLACK, epochs = 20, arguratio = 10)
    trainLevelUNet(basemodel_path, l1_dict, 'l1' + flex, CLASS1_SIZE, epochs = epochs, 
        batch_size = 12, exceptlist = ['black', 'cell'], loadTrain = loadTrain)
    trainLevelUNet(basemodel_path, l2_dict, 'l2' + flex, CLASS2_SIZE, epochs = epochs, batch_size = 12, 
        exceptlist = ['blackcommon', 'blackbig'], 
        argucnt = 20, batch_max = 5, loadTrain = loadTrain)
    print('train models finished\n')
    
def loadClassifierModel(name):
    from model import BatchNorm
    
    path = 'model/{}.h5'.format(name)
    if os.path.isfile(path):
        print('load {}'.format(path))
        return load_model(path, custom_objects={'BatchNorm':BatchNorm, 'dice_coef': dice_coef})
    return None
    
def loadUnet(level_dict, flex, imgsize):
    dict_unet = {}
    for key, subdict in level_dict.items():
        model_path = 'model/unet{}_{}_{}.h5'.format(flex, key, imgsize)
        if os.path.isfile(model_path):
            print('load unet_{}...'.format(key))
            dict_unet[key] = load_model(model_path, 
                     custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef': dice_coef})
    return dict_unet
    
def slice_unet():
    # Setting seed for reproducability
    smooth = 1.
    epochs = 20
    BORDER_PATH = 'input/stage1_train_border1/'
    
    # get train_data
    img_size = 256
    thick = 1
    tile_size = 256
    base_size = 16
    
    model_path = 'model/noresize_noclean_{}_{}.h5'.format(tile_size, base_size)
    border_model_path = 'model/unet_noresize_border_{}_{}_{}.h5'.format(thick, tile_size, base_size)
    
    X_pre = read_test_noresize()
    
    X_train_slice, Y_train_slice = loadSliced(TRAIN_PATH, 'mask', tile_size, cutoff = 200, fill = True)
    
    
    unet_slice = trainModelPredictNoResize(tile_size, model_path, X_train_slice, 
                                           Y_train_slice, X_pre, base_size = base_size,
                                           batch_size = 10,
                                           loadtrain = False)
    
    
    
    unet_slice = trainModelPredictNoResize(tile_size, model_path, X_train_slice, 
                                           Y_train_slice, X_pre, base_size = base_size,
                                           batch_size = 10,
                                           loadtrain = False)
    
    flex = 'slice'
    loadTrain = False
    epochs = 20
    mask_path = 'slice_mask'
    
    
    img_dict, l1_dict, l2_dict = classifier_reader()
    
    trainModels(img_dict, l1_dict, l2_dict, basemodel_path = model_path, flex = flex, epochs = epochs, loadTrain = loadTrain)
    
    
    model_c1 = loadClassifierModel('class1')
    model_c2 = loadClassifierModel('class2')
    
    dict_unetl1 = loadUnet(l1_dict, 'l1' + flex, CLASS1_SIZE)
    dict_unetl2 = loadUnet(l2_dict, 'l2' + flex, CLASS1_SIZE)
    
    print('predict training imgs')
    valid_mask = predictImages(X_pre, test_ids, tile_size, model_c1, model_c2, dict_unetl1, dict_unetl2)
    
    save_obj(valid_mask, 'slice_mask')
    
    
    valid_mask = load_obj(mask_path)
    
    result = genRLE_base(valid_mask, test_ids, 0.5)
    result.to_csv('classic_tile{}.csv'.format(tile_size), index=False)
    
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def decode_one(maskcode):
    return rle_decode(maskcode.EncodedPixels, (maskcode.Width, maskcode.Height)).T
    
def genMaskFromSolution():
    from visualize import display_instances
    
    TAR_PATH = 'input/stage1_test/'
    stage1_solution = pd.read_csv('input/stage1_solution.csv')
    mask_counts = stage1_solution.ImageId.value_counts()
    
    for id in mask_counts.index:
        
        directory = os.path.join(TAR_PATH, id)
        if os.path.isdir(directory):
            dir_masks = os.path.join(directory, 'masks')
            if not os.path.exists(dir_masks):
                os.makedirs(dir_masks)
            
            cnt = mask_counts[id]
            print('{} {}'.format(id, cnt))
            sub_masks = stage1_solution.loc[stage1_solution.ImageId == id]
            first = sub_masks.iloc[0]
            final_masks = np.zeros((first.Height, first.Width, cnt)).astype(np.uint8)
            
            for i in range(len(sub_masks)):
                final_masks[...,i] = decode_one(sub_masks.iloc[i])
            
            image_path = os.path.join(TAR_PATH, id, 'images', '{}.png'.format(id))
            image = imread(image_path)
            display_instances(image, None, final_masks, np.ones(final_masks.shape[2]), 
                              ['BG', 'cell'], np.ones(final_masks.shape[2]), ax=get_ax())
            plt.show()
            
            
            for j in range(final_masks.shape[-1]):
                path = os.path.join(dir_masks, '{}.png'.format(j))
                imsave(path, final_masks[...,j])
        

    