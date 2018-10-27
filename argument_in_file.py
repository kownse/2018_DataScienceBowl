import os
import numpy as np
import matplotlib.pyplot as plt
import random
import re

from scipy.ndimage import find_objects
from scipy.ndimage.interpolation import rotate
from skimage.io import imread, imshow, imread_collection, concatenate_images, imsave

from nucleus import plotOrigVsPred

def getImgAndMasks(subpath):
    img = None
    mask = None
    for subroot, subdir, subfiles in os.walk(subpath):
        if 'images' in subroot and len(subfiles) == 1:
            f = subfiles[0]
            path = os.path.join(subroot, f)
            img = imread(path)[:,:,:3]
            
        elif 'masks' in subroot and len(subfiles) > 0:
            mask = np.zeros((img.shape[0], img.shape[1], len(subfiles)), dtype=np.bool)
            for idx, mask_file in enumerate(os.path.join(subroot, f) for f in subfiles):
                mask_ = imread(mask_file)
                if len(mask_.shape) > 2:
                    mask_ = mask_[:,:,0].astype(np.uint8)
                mask[:,:,idx] = mask_
    return img, mask

def save_subimgmask(subimg, submask_valid, gendir, flex):
    name = '1{}{}'.format(flex, 1)
    
    dirs = next(os.walk(gendir))[1]
    if len(dirs) > 0:
        nums = [int(re.findall(r'\d+', d)[1]) for d in dirs]
        idx = max(nums) + 1
        name = '1{}{}'.format(flex, idx)
    
    newdir = os.path.join(gendir, name)
    if not os.path.exists(newdir):
        os.makedirs(newdir)
        
    imgdir = os.path.join(newdir, 'images')
    if not os.path.exists(imgdir):
        os.makedirs(imgdir)
        
    imgpath = '{}/{}.png'.format(imgdir, name)
    imsave(imgpath, subimg)
    
    maskdir = os.path.join(newdir, 'masks')
    if not os.path.exists(maskdir):
        os.makedirs(maskdir)
        
    #for i in range(submask_valid.shape[2]):
    for i in range(len(submask_valid)):
        maskpath = '{}/{}.png'.format(maskdir, i)
        mask = np.where(submask_valid[i] > 0, 255, 0)
        imsave(maskpath, mask)
        
    #print('saved to {}'.format(newdir))
        
def plotImgAndMask(img, mask):
    totsubmask = np.zeros((img.shape[0], img.shape[1]))
    for i in range(mask.shape[2]):
        totsubmask = np.maximum(totsubmask, mask[:,:,i])
    plotOrigVsPred([img, totsubmask])

def random_crop(img, mask, gendir, flex, subscale = 0.5):
    """
    if mask.shape[2] < 50:
        print('too many nucles({}), skip'.format(mask.shape[2]))
        return
    """
    if img.shape[0] < 400:
        print('too small({}), skip'.format(img.shape))
        return
    
     
    width = img.shape[0]
    height = img.shape[1]
    """
    rects = [
            (0, int(width * subscale), 0, int(height * subscale)),
            (int(width * subscale), int(width * 2 * subscale), 0, int(height * subscale)),
            (0, int(width * subscale), int(height * subscale), int(height * 2 * subscale)),
            (int(width * subscale), int(width * 2 * subscale), int(height * subscale), int(height * 2 * subscale)),
            (int(width * subscale * 0.5), int(width * subscale * 1.5), int(height * subscale * 0.5), int(height * subscale * 1.5))
            ]
    """
    rects = [(int(width * subscale * 0.5), int(width * subscale * 1.5), int(height * subscale * 0.5), int(height * subscale * 1.5))]
    cnt = int(1 / subscale)
    for i in range(cnt):
        for j in range(cnt):
            rects.append((int(width * subscale * i), 
                          int(width * subscale * (i + 1)), 
                          int(height * subscale * j),
                          int(width * subscale * (j + 1))))
    
    for (xstart, xend, ystart, yend) in rects:
        subimg = img[xstart:xend, ystart:yend, :]
        submasks = mask[xstart:xend, ystart:yend, :]
        
        if submasks.max() <= 0:
            print('no nucles, skip')
            continue
        
        submask_valid = []
        
        for i in range(submasks.shape[2]):
            if submasks[:,:,i].max() > 0:
                sm = np.where(submasks[:,:,i] > 0, 255, 0)
                submask_valid.append(sm)
        print('{} / {} valid'.format(len(submask_valid), submasks.shape[2]))
        if len(submask_valid) > 6:
            save_subimgmask(subimg, submask_valid, gendir, flex)
            
def horizontalStretch(img, mask, gendir, flex, subscale = 0.125, scalratio = 2.5):
    from scipy.misc import imresize
    
    width = img.shape[1]
    height = img.shape[0]
    
    rects = []
    cnt = int(1 / subscale)
    for i in range(cnt):
        rects.append((0, height, 
                      int(width * subscale * i), 
                      int(width * subscale * (i + 1))))
        
    for (xstart, xend, ystart, yend) in rects:
        subimg = img[xstart:xend, ystart:yend, :]
        submasks = mask[xstart:xend, ystart:yend, :]
        
        if submasks.max() <= 0:
            print('no nucles, skip')
            continue
        
        stretch_height = subimg.shape[0]
        stretch_width = int(subimg.shape[1] * scalratio)
        subimg = imresize(subimg, (stretch_height, stretch_width, subimg.shape[2]))
        
        valid_mask = []
        for j in range(submasks.shape[2]):
            if submasks[:,:,j].max() > 0:
                vm = np.where(submasks[:,:,j] > 0, 255, 0).astype(np.uint8)
                vm = imresize(vm, (stretch_height, stretch_width))
                valid_mask.append(vm)    
        
        save_masks = np.zeros((stretch_height, stretch_width, len(valid_mask))).astype(np.uint8)
        for j in range(len(valid_mask)):
            save_masks[:,:,j] = valid_mask[j]
        
        save_subimgmask(subimg, save_masks, gendir, flex)
            
def rotate(image, mask, gendir, flex):
    for rtype in range(3):
        if rtype == 0:
            subimage = np.fliplr(image)
            submask = np.fliplr(mask)
        elif rtype == 1:
            subimage = np.flipud(image)
            submask = np.flipud(mask)
        elif rtype == 2:
            subimage = np.rot90(image)
            submask = np.rot90(mask)
            
        save_subimgmask(subimage, submask, gendir, flex)


def generate(path, num_per_img, flex):

    gendir = os.path.join(path, 'generate')
    if not os.path.exists(gendir):
        os.makedirs(gendir)
            
    root, dirs, files = next(os.walk(path))
    for dir in dirs:
        if dir == 'generate':
            continue
        
        subpath = os.path.join(root, dir)
        #print(subpath)
        
        img, mask = getImgAndMasks(subpath)
        if img is not None and mask is not None:
            #if mask.shape[2] > 200:
            #    print(dir)
            random_crop(img, mask, gendir, flex, 0.25)
            #rotate(img, mask, gendir, flex)
            #horizontalStretch(img, mask, gendir, flex)
            
def checkGenerate(path, maxcnt):
    
    path = os.path.join(path, 'generate')
    for idx, dir in enumerate(next(os.walk(path))[1]):
        if idx > maxcnt:
            break
        
        subpath = os.path.join(path, dir)
        print(subpath)
        img, mask = getImgAndMasks(subpath)
        plotImgAndMask(img, mask)
        
        
        
if __name__ == "__main__": 
    generate('input/detail/organ/organ', 5, 'organ')
    #generate('input/detail/black/blackHuge', 5, 'blackHuge')
    #generate('input/detail/black/blackcommon', 5, 'blackMassive')
    #generate('input/detail/white/white', 5, 'white')
    #generate('input/detail/cell/cell', 5, 'cell')
    #generate('input/detail/black/blackblur', 5, 'blackblur')
    #generate('input/detail/black/many', 5, 'blackstretch')
    #checkGenerate('input/detail/black/blackHuge', 5)