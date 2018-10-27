
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
from model import log
from utils import resize_image, resize_mask
from skimage.transform import resize

from nucleus import scale_img_canals, class1Model, loadClassifierModel,\
     trainClass1Classifier, CLASS2_BLACK_INV, CLASS2_BLACK, CLASS1_INV, CLASS1
from nucleus_mrcnn import NucleusConfig, InferenceConfig, predictOne, ShapesDataset, \
    CoCoNucluesConfig, CoCoInferenceConfig, processImg, CoCoRevertNucluesConfig,\
    CoCoRevertInferenceConfig

from kaggle_utils.dill_helper import save_obj, load_obj
from keras.utils import Progbar
from skimage.io import imread, imshow, imread_collection, concatenate_images, imsave
from visualize import display_images
import math

from keras import backend as K

train_config = CoCoRevertNucluesConfig
predict_config = CoCoRevertInferenceConfig

# Root directory of the project
ROOT_DIR = os.getcwd()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs_classify")
BASEMODEL_DIR = os.path.join(ROOT_DIR, "logs")

#TEST_PATH = 'input/stage1_test/'
TEST_PATH = 'input/stage2_test_final/'
#TEST_PATH = 'input/stage2_test_final_single/'
DETAIL_PATH = 'input/detail'
DETAILCLASSIFY_PATH = 'input/detail_classify'
VALID_PATH = 'input/detail_simple'

IMG_SIZE = 512
TYPE_CLASS_SIZE = 64

CLASS1_ = {
    'black': 0,
    'cell': 1,
    'organ': 2,
    'white': 3,
}

CLASS2_BLACK_ = {
    'blackcommon' :1,
    'blackHuge' : 2,
    'blackStar' : 3,
}

"""
# 0.420
def processImg(img):
    from skimage.exposure import equalize_adapthist
    if img.mean() < 100:
        img = equalize_adapthist(img)
        img = scale_img_canals(img)

    return img
"""

def classifier_reader(path = './input/detail', flex = ''):
    l1_dict_path = 'l1_dict_mrcnn{}'.format(flex)
    l2_dict_path = 'l2_dict_mrcnn{}'.format(flex)
    
    if os.path.isfile('obj/{}.dl'.format(l1_dict_path)) and \
        os.path.isfile('obj/{}.dl'.format(l2_dict_path)):
        print('read from file')
        l1_dict = load_obj(l1_dict_path)
        l2_dict = load_obj(l2_dict_path)
        return l1_dict, l2_dict
    
    l1_dict = {

    }
    
    l2_dict = {
    }
    
    def fillclassdict(some_dict, img, cls, img_size):
        img = resize(img, (img_size, img_size), 
                   mode='constant', preserve_range=True).astype(np.uint8)
        if cls not in some_dict:
            some_dict[cls] = {'image':[]}
        some_dict[cls]['image'].append(img)

    for root, dirs, files in os.walk(path):
        if 'images' in root and len(files) == 1:
            f = files[0]
            path = os.path.join(root, f)
            img = imread(path)[:,:,:3].astype(np.uint8)

            #img = processImg(img).astype(np.uint8)
            
            spt = root.split('\\')
            ext = os.path.splitext(str(f))
            
            class1 = spt[1]
            class2 = spt[2]
            fillclassdict(l1_dict, img, class1, TYPE_CLASS_SIZE)
            
            if 'black' in class2:
                #print(class2)
                fillclassdict(l2_dict, img, class2, TYPE_CLASS_SIZE)
            
            print('{} {} {} {}'.format(class1, class2, img.max(), ext[0]))  
            
    save_obj(l1_dict, l1_dict_path)
    save_obj(l2_dict, l2_dict_path)

    return l1_dict, l2_dict

def loadClassifyMRCNN(level_dict, flex, imgsize):
    dict_unet = {}
    for key in level_dict.keys():
        inference_config = CoCoInferenceConfig()
        model = modellib.MaskRCNN(mode="inference", 
                                config=inference_config,
                                model_dir=MODEL_DIR)
        model_path = model.find_last()[1]
        if model_path:
            print("Loading weights from " + model_path)
            model.load_weights(model_path, by_name=True)
        else:
            print('no weights, skip')
            continue

        dict_unet[key] = model 
        
    return dict_unet



class ShapesDatasetDetail(ShapesDataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def load_imgs(self, basepath):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """

        # Add classes
        self.add_class("shapes", 1, "cell")
        
        inImages = True
        for root, dirs, files in os.walk(basepath):
            if 'images' in root and len(files) == 1:
                if not inImages:
                    inImages = True

                    self.add_image("shapes", image_id=imgfile, path = imgpath,
                           shapes=shapes)

                imgfile = files[0]
                imgpath = os.path.join(root, imgfile)

            elif 'masks' in root and len(files) > 0:
                inImages = False

                shapes = []
                for mask_file in (os.path.join(root, f) for f in files):
                    shapes.append(('cell', mask_file))
                    
        self.add_image("shapes", image_id=imgfile, path = imgpath,
                           shapes=shapes)

    
def trainSubMRCNN(typestr, modeldir, datadir, validdir):   
    dataset = ShapesDatasetDetail()
    dataset.load_imgs(datadir)
    dataset.prepare()
    
    print("Image Count: {}".format(len(dataset.image_ids)))
    print("Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))
        
    dataset_valid = ShapesDatasetDetail()
    dataset_valid.load_imgs(validdir)
    dataset_valid.prepare()
    
    config = train_config()

    config.STEPS_PER_EPOCH = 0
    config.LEARNING_RATE = 0.0005
    argument = False
    trainFromCommon = True
    epochs = 20

    if typestr == 'organ':
        argument = True
        #trainFromCommon = False
        epochs = 20
    elif typestr == 'cell':
        argument = True

    elif typestr == 'blackHuge':
        argument = True
        epochs = 20
    elif typestr == 'blackStar':
        argument = True
        epochs = 20
    elif typestr == 'blackcommon':
        #argument = True
        
        epochs = 10
    elif typestr == 'white':
        argument = True
        epochs = 10
    
    config.display()

    model = modellib.MaskRCNN(mode="training", config=config,
                            model_dir=modeldir)
    
    model_path = model.find_last()[1]
    if model_path:
        print("\nLoading weights from {}\n".format(model_path))
        model.load_weights(model_path, by_name=True)
    else:
        basemodel_path = model.find_last(BASEMODEL_DIR)[1]
        if trainFromCommon and basemodel_path:
            print('\ntrain from predicesor: {}\n'.format(basemodel_path))
            model.load_weights(basemodel_path, by_name=True)
        else:
            print('\nno basemodel no training\n')
            return
        

    print('\nstart traning ' + typestr)
    model.train(dataset, 
                dataset_valid, 
                learning_rate=config.LEARNING_RATE,
                epochs=epochs, 
                #layers="heads",
                layers="all",
                argument = argument)
    K.clear_session()
    
    """
    image_id = np.random.choice(dataset.image_ids, 1)[0]
    image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
        dataset, config, image_id, augment=True, use_mini_mask=True)
    log("mask", mask)
    display_images([image]+[mask[:,:,i] for i in range(min(mask.shape[-1], 7))])
    """
def classifiedTrain():
    root, dirs, files = next(os.walk(DETAIL_PATH))
    for typedir in dirs:
        subdir = os.path.join(root, typedir)
        subroot, subdirs, subfiles = next(os.walk(subdir))
        for subtype in subdirs:
            #if subtype in ['blackcommon']:
                #if subtype not in ['blackcommon', 'white', 'organ']:
            if subtype not in ['blackStar']:
                continue

            datadir = os.path.join(subroot, subtype)
            modeldir = os.path.join(MODEL_DIR, subtype)
            validdir = os.path.join(VALID_PATH, typedir, subtype)
            
            print(modeldir)
            if not os.path.exists(modeldir):
                os.makedirs(modeldir)
             
            trainSubMRCNN(subtype, modeldir, datadir, validdir)
            
def classify(model, img):
    if len(img.shape) < 4:
        img = img[np.newaxis, :, :, :]
    prob = model.predict(img, verbose=0)
    pred = prob.argmax(axis=-1)[0]
    return pred

def classify_dataset(model, ds, idx):
    image_id = ds.image_ids[idx]
    path = ds.image_info[image_id]['path']
    
    img = imread(path)[:,:,:3]
    #img = processImg(img)
    img = resize(img, (TYPE_CLASS_SIZE, TYPE_CLASS_SIZE), mode='constant', preserve_range=True).astype(np.uint8)
    
    plt.figure()
    plt.imshow(img)
    
    return classify(model, img)

def predictImages(validpath, model_c1, model_c2, dict_mrcnnl1, dict_mrcnnl2, plot):
    from tqdm import tqdm

    dataset_valid = ShapesDataset()
    dataset_valid.load_imgs(validpath)
    dataset_valid.prepare()

    inference_config = predict_config()
    inference_config.display()

    new_test_ids = []
    rles = []

    for i in range(len(dataset_valid.image_ids)):
        image_id = dataset_valid.image_ids[i]
        path = dataset_valid.image_info[image_id]['path']

        img = imread(path)
        if len(img.shape) < 3:
            new_img = np.zeros((img.shape[0], img.shape[1], 3))
            for i in range(3):
                new_img[..., i] = img
            img = new_img
        else:
            img = img[:,:,:3]
        
        img4d = resize(img, (TYPE_CLASS_SIZE, TYPE_CLASS_SIZE), mode='constant', preserve_range=True).astype(np.uint8)
        img4d = img4d[np.newaxis, :, :, :]
        prob = model_c1.predict(img4d, verbose=0)
        pred = prob.argmax(axis=-1)[0]
        
        mrcnn = None
        if pred == 0:
            prob2 = model_c2.predict(img4d, verbose=0)
            pred2 = prob2.argmax(axis=-1)[0]
            mrcnn = dict_mrcnnl2[CLASS2_BLACK_INV[pred2]]
            print('\n{} type1[{}] type2 [{}]'.format(image_id, CLASS1_INV[pred], CLASS2_BLACK_INV[pred2]))

        else:
            print('\n{} type1 [{}]'.format(image_id, CLASS1_INV[pred]))
            mrcnn = dict_mrcnnl1[CLASS1_INV[pred]]
            
        #continue
        file_id, rle = predictOne(mrcnn, dataset_valid, inference_config, i, plot = plot)
        rles.extend(rle)
        new_test_ids.extend([file_id] * len(rle))

        
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
        
    sub.to_csv('classify_mrcnn.csv', index=False)
    
def trainClassifier():
    l1_dict, l2_dict = classifier_reader(DETAILCLASSIFY_PATH, flex = '')
    trainClass1Classifier(l1_dict, 'mrcnn_class1', CLASS1, TYPE_CLASS_SIZE, 3, epochs = 100, arguratio = 10)
    trainClass1Classifier(l2_dict, 'mrcnn_class2', CLASS2_BLACK, TYPE_CLASS_SIZE, 3, epochs = 100, arguratio = 10)
            

def loadClassify_mrcnn(level_dict):
    dict_mrcnn = {}
    for key in level_dict.keys():
        if key in ['black']:
            continue
        
        inference_config = predict_config()
        modeldir = os.path.join(MODEL_DIR, key)
        print('load {} from {}'.format(inference_config.NAME, modeldir))
        
        model = modellib.MaskRCNN(mode="inference", 
                                  config=inference_config,
                                  model_dir=modeldir)
        
        model_path = model.find_last()[1]
        assert model_path != "", "Provide path to trained weights"
        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)
        
        dict_mrcnn[key] = model
    return dict_mrcnn

if __name__ == "__main__":    

    #trainClassifier()
    #classifiedTrain()
    
    model_c1 = loadClassifierModel('mrcnn_class1')
    model_c2 = loadClassifierModel('mrcnn_class2')
    
    dict_mrcnn_l1 = loadClassify_mrcnn(CLASS1_)
    dict_mrcnn_l2 = loadClassify_mrcnn(CLASS2_BLACK_)

    predictImages(TEST_PATH,  model_c1, model_c2, dict_mrcnn_l1, dict_mrcnn_l2, True)
    
    
