# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 10:45:06 2018

@author: kowns
"""

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from config import Config
import utils
import model as modellib
import visualize
from tqdm import tqdm
from model import log
from utils import resize_image, resize_mask

from nucleus import scale_img_canals
from kaggle_utils.dill_helper import save_obj, load_obj
from keras.utils import Progbar
from skimage.io import imread, imshow, imread_collection, concatenate_images, imsave
from visualize import display_images
from keras import backend as K
from scipy import ndimage

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

TRAIN_PATH = 'input/stage1_train/'
#TRAIN_PATH = 'input/stage1_train_simple/'
VALID_PATH = 'input/stage1_train_valid/'
#TEST_PATH = 'input/stage1_test/'
#TEST_PATH = 'input/stage1_test_single/'

TEST_PATH = 'input/stage2_test_final/'
#TEST_PATH = 'input/stage2_test_final_single/'
DETAIL_PATH = 'input/detail'

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
total_size = len(train_ids)

IMG_SIZE = 512

"""
# 0.420
def processImg(img):
    from skimage.exposure import equalize_adapthist
    if img.mean() < 100:
        img = equalize_adapthist(img)
        img = scale_img_canals(img)

    return img
"""

def processImg(img):
    #tmp = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    if img.dtype == 'uint16':
        img = (img / 65535 * 255).astype(np.uint8)
    elif img.dtype == 'uint32':
        img = (img / 4294967295 * 255).astype(np.uint8)
    elif img.dtype == 'uint64':
        img = (img / 18446744073709551615 * 255).astype(np.uint8)
        
    if len(img.shape) >= 3:
        tmp = img[...,0]
        
        if tmp.mean() > 100:
            tmp = 255 - tmp
            
        for i in range(img.shape[-1]):
            img[...,i] = tmp
    else:
        if img.mean() > 100:
            img = 255 - img
            
        new_img = np.zeros((img.shape[0], img.shape[1], 3))
        for i in range(3):
            new_img[..., i] = img
        img = new_img
    
    return img

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths
    
class NucleusConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "massive"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_PADDING = True
    LEARNING_RATE = 0.002

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 1024
    DETECTION_MAX_INSTANCES = 512
    MAX_GT_INSTANCES = 512
    RPN_TRAIN_ANCHORS_PER_IMAGE = 512
    #BACKBONE = "resnet50"

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 0

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 10
    ARGUMENT = 0
    
class InferenceConfig(NucleusConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class CoCoNucluesConfig(NucleusConfig):
    NAME = "coco1024_"
    IMAGES_PER_GPU = 1
    TRAIN_FROM_COCO = 1
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    LEARNING_RATE = 0.002

    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_BBOX_STD_DEV = np.array([0.2, 0.2, 0.3, 0.3])
    BBOX_STD_DEV = np.array([0.2, 0.2, 0.3, 0.3])

class CoCoInferenceConfig(CoCoNucluesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    
class CoCoRevertNucluesConfig(CoCoNucluesConfig):
    NAME = "coco1024_gray_resnet50_all_"
    BACKBONE = 'resnet50'
    TRAIN_FROM_COCO = 1
    TRAIN_ROIS_PER_IMAGE = 896
    ROI_POSITIVE_RATIO = 0.5
    DETECTION_MAX_INSTANCES = 448
    MAX_GT_INSTANCES = 448
    RPN_TRAIN_ANCHORS_PER_IMAGE = 448
    MEAN_PIXEL = [117.4]
    IMAGE_CHANEL = 3
    LEARNING_RATE = 0.0005
    
    
class CoCoRevertInferenceConfig(CoCoRevertNucluesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

train_config = CoCoRevertNucluesConfig
test_config = CoCoRevertInferenceConfig

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    scale_ = 1.0
    padding_ = None

    def load_imgs(self, basepath):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "cell")
        
        ids = next(os.walk(basepath))[1]
        for id_ in ids:
            path = basepath + id_
            img_path = path + '/images/' + id_ + '.png'
            if not os.path.isfile(img_path):
                img_path = path + '/images/' + id_ + '.tif'
            
            shapes = []
            maskpath = path + '/masks/'
            if os.path.exists(maskpath):
                maskfiles = next(os.walk(maskpath))[2]
                for mask_file in maskfiles:
                    maskpath = path + '/masks/' + mask_file
                    shapes.append(("cell", maskpath))
            
            self.add_image("shapes", image_id=id_, path = img_path,
                           width=None, height=None, shapes=shapes)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        path = info['path']
        img = imread(path)
        img = processImg(img)

        channel = 3
        if len(img.shape) > 2 and img.shape[2] > channel:
            if channel > 1:
                img = img[:,:,:channel]
            else:
                img = img[:,:,0]
        img = img.astype(np.uint8)
        
        info['width'] = img.shape[0]
        info['height'] = img.shape[1]
        
        return img

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['width'], info['height'], count], dtype=np.uint8)
        for i, (shape, path) in enumerate(info['shapes']):
            mask_ = imread(path)
            if len(mask_.shape) > 2:
                mask_ = mask_[:,:,0]
            mask[:, :, i] = mask_.astype(np.uint8)
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask, class_ids.astype(np.int32)

def continueTraining(epochs = 50):
    config = NucleusConfig()
    config.display()

    dataset = ShapesDataset()
    dataset.load_imgs(TRAIN_PATH)
    dataset.prepare()
    
    print("Image Count: {}".format(len(dataset.image_ids)))
    print("Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))
        
    dataset_valid = ShapesDataset()
    dataset_valid.load_imgs(VALID_PATH)
    dataset_valid.prepare()
    
    model = modellib.MaskRCNN(mode="training", config=config,
                            model_dir=MODEL_DIR)
    
    model_path = model.find_last()[1]
    if model_path:
        print("Loading weights from " + model_path)
        model.load_weights(model_path, by_name=True)
    else:
        print('train from begin')
    
    model.train(dataset, dataset_valid, 
                learning_rate=config.LEARNING_RATE,
                epochs=epochs, 
                layers="all",
                argument = False)

def continueTrainingCoCo(epochs):
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "model", "mask_rcnn_coco.h5")
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    config = train_config()
    #config = CoCoNucluesConfig()
    config.display()

    dataset = ShapesDataset()
    dataset.load_imgs(TRAIN_PATH)
    dataset.prepare()
    
    print("Image Count: {}".format(len(dataset.image_ids)))
    print("Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))
        
    dataset_valid = ShapesDataset()
    dataset_valid.load_imgs(VALID_PATH)
    dataset_valid.prepare()
    
    model = modellib.MaskRCNN(mode="training", config=config,
                            model_dir=MODEL_DIR)
    
    model_path = model.find_last()[1]
    if model_path:
        print("Loading weights from " + model_path)
        model.load_weights(model_path, by_name=True)
    elif config.TRAIN_FROM_COCO == 1:
        print('train from CoCo')
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
    else:
        print('train from scrash')
    
    model.train(dataset, dataset_valid, 
                learning_rate=config.LEARNING_RATE,
                epochs=epochs, 
                #layers="heads",
                layers="all",
                argument = config.ARGUMENT)

def simpleValidation():
    class InferenceConfig(NucleusConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    
    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIR)
    
    model_path = model.find_last()[1]
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    
    dataset_valid = ShapesDataset()
    dataset_valid.load_imgs(VALID_PATH)
    dataset_valid.prepare()
    
    image_id = dataset_valid.image_ids[3]
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_valid, inference_config, 
                               image_id, use_mini_mask=True, augment=False)
    
    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
    
    results = model.detect([original_image], verbose=1)
    
    r = results[0]
    
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset_valid.class_names, r['scores'], ax=get_ax())
    
# Basic NMS on boxes, https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python
# Malisiewicz et al.
def non_max_suppression_fast(boxes, masks, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the integer data type
    #return boxes[pick].astype("int"), pick
    return pick

def non_max_suppression_mask_fast(masks, overlapThresh):
    if len(masks) == 0:
        print('no mask')
        return []  
    
    if masks.shape[2] > 50:
        print('too many masks, skip non_max')
        return list(range(masks.shape[2]))
                    
    pick = []
    area = []
    for i in range(masks.shape[2]):
        area.append(np.sum(masks[:,:,i]))
    idxs = np.argsort(area)
    
    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        idxs = idxs[:-1]

        # find the largest (x, y) coordinates for the start of the bounding box and the smallest (x, y) coordinates for the end of the bounding box
        remove_idx = []
        for j in range(len(idxs)):
            if idxs[j] == i:
                continue
            
            small_area = area[idxs[j]]
            intersection = np.sum(np.logical_and(masks[...,i], masks[...,idxs[j]]))
            overratio = intersection / small_area
            """
            if intersection > 0:
                print(i)
                print(intersection)
                plt.figure()
                plt.imshow(masks[...,i])
                plt.show()
                
                print(idxs[j])
                plt.figure()
                plt.imshow(masks[..., idxs[j]])
                plt.show()
            """
            #if overratio > 0:
            #    print(overratio)
            if overratio > overlapThresh:
                remove_idx.append(j)
        
        if len(remove_idx) > 0:
            idxs = [i for j, i in enumerate(idxs) if j not in remove_idx]

    return pick


# Compute NMS (i.e. select only one box when multiple boxes overlap) for across models.
def models_cv_masks_boxes_nms(models_cv_masks_boxes, masks, threshold=0.3):
    #boxes = np.concatenate(models_cv_masks_boxes).squeeze()
    pick = non_max_suppression_fast(models_cv_masks_boxes, threshold)
    return pick

def predictOne(model, dataset, config, id, nonmax=True, resize=True, plot = True):
    image_id = dataset.image_ids[id]
    scaled_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config, 
                               image_id, use_mini_mask=True, augment=False)
    """
    log("original_image", scaled_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)
    """
    
    file_id = dataset.image_info[image_id]['id']
    original_shape = (image_meta[1], image_meta[2])
    print('\n{}\t{}\t'.format(file_id, original_shape))
    
    results = model.detect([scaled_image], verbose=0)
    
    r = results[0]
    #r['masks'] = sorted(r['masks'], key = lambda x : np.sum(x > 0))
    """
    if plot: 
        if min(r['masks'].shape):
            visualize.display_instances(scaled_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], ax=get_ax())
            plt.show()
        else:
            print('no cell')
            plt.imshow(scaled_image)
            plt.show()
    """
    
    boxes = r['rois']
    masks = r['masks']

    if nonmax:
        pick = non_max_suppression_mask_fast(r['masks'], 0.4)
        print('non_max_suppression [{}] to [{}]'.format(boxes.shape[0], len(pick)))
    
    pick_rois = r['rois'][pick]
    pick_masks = r['masks'][:,:,pick]
    pick_class_ids = r['class_ids'][pick]
    pick_scores = r['scores'][pick]
    
    if plot: 
        if boxes.shape[0] != len(pick):
            ax = get_ax(rows=1, cols=2, size=5)
            visualize.display_instances(scaled_image, r['rois'], r['masks'], r['class_ids'], 
                                dataset.class_names, r['scores'], ax=ax[0])
            visualize.display_instances(scaled_image, pick_rois, pick_masks, pick_class_ids, 
                                dataset.class_names, pick_scores, ax=ax[1])
            plt.show()
        else:
            visualize.display_instances(scaled_image, pick_rois, pick_masks, pick_class_ids, 
                                dataset.class_names, pick_scores, ax=get_ax())
            plt.show()
    
        
    r['rois'] = pick_rois
    r['masks'] = pick_masks
    r['class_ids'] = pick_class_ids
    r['scores'] = pick_scores
    
    
    #from scipy.misc import imresize
    from skimage.transform import resize
    
    rles = []
    
    if resize:
        window = image_meta[4:8]
        original_img = scaled_image[window[0]:window[2], window[1]:window[3], :]
        original_img = resize(original_img, original_shape)
    
    if min(r['masks'].shape) > 0:
        submasks = np.zeros((image_meta[1], image_meta[2], r['masks'].shape[2]))
        mask_record = np.ones(original_shape)
        for i in range(r['masks'].shape[2]):
            submask = r['masks'][window[0]:window[2], window[1]:window[3], i]
            area = np.sum(submask)
            if area < 5:
                continue
            #print(area)
            
            submask = resize(submask, original_shape)
            submask = np.logical_and(submask, mask_record)
            
            if submask.shape != original_shape:
                print(submask.shape)
            mask_record = np.logical_and(mask_record, np.logical_not(submask))
            #plt.figure()
            #plt.imshow(mask_record)
            submasks[:,:,i] = submask
            rle = rle_encoding(submask)
            if len(rle) > 0:
                rles.append(rle)
            else:
                print('error')
    else:
        rles = [[0, 0]]
    
    
    return file_id, rles

def predictTests(model, config, validpath, plot):
    dataset_valid = ShapesDataset()
    dataset_valid.load_imgs(validpath)
    dataset_valid.prepare()

    #file_id, rle = predictOne(model, dataset_valid, inference_config, 1)

    new_test_ids = []
    rles = []
        
    for i in tqdm(range(len(dataset_valid.image_ids))):
        file_id, rle = predictOne(model, dataset_valid, config, i, 
                                  plot = plot)
        rles.extend(rle)
        new_test_ids.extend([file_id] * len(rle))
        
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
        
    sub.to_csv('{}.csv'.format(config.NAME), index=False)

def loadModelPredict():

    inference_config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIR)
    
    model_path = model.find_last()[1]
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    
    predictTests(model, inference_config, TEST_PATH)

def loadModelPredictCoCo(plot):
    
    inference_config = test_config()
    inference_config.display()
    #inference_config = CoCoInferenceConfig()
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIR)
    
    model_path = model.find_last()[1]
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    
    predictTests(model, inference_config, TEST_PATH, plot)
    
def checkHandyMask():
    dataset = ShapesDataset()
    dataset.load_imgs('input/stage2_new/')
    dataset.prepare()

    for i in range(len(dataset.image_ids)):
        id = dataset.image_ids[i]
        image = dataset.load_image(id)
        masks = dataset.load_mask(id)[0] > 0
        for i in range(masks.shape[-1]):
            plt.imshow(masks[...,i])
            plt.show()
    
"""
image_id = np.random.choice(dataset.image_ids, 1)[0]
image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
    dataset, config, image_id, augment=True, use_mini_mask=True)
log("mask", mask)
display_images([image]+[mask[:,:,i] for i in range(min(mask.shape[-1], 7))])
"""
if __name__ == "__main__":
    #continueTraining(30)
    #loadModelPredict()

    #continueTrainingCoCo(50)
    loadModelPredictCoCo(False)

    #checkHandyMask()
