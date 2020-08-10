from __future__ import print_function

import sys

import os
import glob
import numpy as np
from skimage.transform import resize
from skimage.io import imsave

from skimage.io import imread
from numba import cuda,jit

#data_path ='in'

image_rows = int(256)
image_cols = int(256)
image_depth = 16

def create_train_data():

    op=sys.argv[1]
    train_data_path=op+"/train/"
    mask_data_path=op+"/masks/"
    dirs = os.listdir(train_data_path)
    total = int(len(dirs)*16*2)

    imgs = np.ndarray((total, image_depth, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, image_depth, image_rows, image_cols), dtype=np.uint8)

    imgs_temp = np.ndarray((total, image_depth//2, image_rows, image_cols), dtype=np.uint8)
    imgs_mask_temp = np.ndarray((total, image_depth//2, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for dirr in sorted(os.listdir(train_data_path)):
        j = 0
        dirr=train_data_path+"/"+dirr
        images = sorted(os.listdir(dirr))
        count = total
        for image_name in images:
            img = imread(os.path.join(dirr, image_name), as_gray=True)
            img = img.astype(np.uint8)
            info = np.iinfo(np.uint16) # Get the information of the incoming image type
            # img = img / info.max # normalize the data to 0 - 1
            # img=255. * img
            img = np.array([img])
            imgs_temp[i,j] = img
            j += 1
            if j % (image_depth/2) == 0:
                j=0
                i += 1
                if (i % 100) == 0:
                    print('Done: {0}/{1} 3d images'.format(i, count))

    for x in range(0, imgs_temp.shape[0]-1):
        imgs[x]=np.append(imgs_temp[x], imgs_temp[x+1], axis=0)

    print('Loading of train data done.')

    i = 0
    for dirr in sorted(os.listdir(train_data_path)):
        j = 0
        dirr=mask_data_path+'/'+dirr
        print (dirr)
        images = sorted(file for file in os.listdir(dirr) if file.endswith('.png'))
        count = total
        for mask_name in images:
            img_mask= imread(dirr+'/'+mask_name,as_gray=True)
            
            info = np.iinfo(np.uint8) # Get the information of the incoming image type
            img_mask = img_mask.astype(np.uint16)
            # img_mask = img_mask / info.max # normalize the data to 0 - 1
            # img_mask = 255. * img_mask

            img_mask = np.array([img_mask])

            imgs_mask_temp[i,j] = img_mask

            j += 1
            if j % (image_depth/2) == 0:
                j = 0
                i += 1
                if (i % 100) == 0:
                    print('Done: {0}/{1} mask 3d images'.format(i, count))

    for x in range(0, imgs_mask_temp.shape[0]-1):
        imgs_mask[x]=np.append(imgs_mask_temp[x], imgs_mask_temp[x+1], axis=0)

    print('Loading of masks done.')


    imgs_mask = preprocess(imgs_mask)
    imgs = preprocess(imgs)

    print('Preprocessing of masks done.')

    np.save(op+'/imgs_train.npy', imgs)
    np.save(op+'/imgs_mask_train.npy', imgs_mask)

    imgs = preprocess_squeeze(imgs)
    imgs_mask = preprocess_squeeze(imgs_mask)

    count_processed = 0
    pred_dir = 'train_preprocessed'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, 30):
        for y in range(0, imgs.shape[1]):
            imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs[x][y])
            count_processed += 1
            if (count_processed % 100) == 0:
                print('Done: {0}/{1} train images'.format(count_processed, 500))

    count_processed = 0
    pred_dir = 'mask_preprocessed'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for x in range(0, 30):
        for y in range(0, imgs_mask.shape[1]):
            imsave(os.path.join(pred_dir, 'pre_processed_' + str(count_processed) + '.png'), imgs_mask[x][y])
            count_processed += 1
            if (count_processed % 100) == 0:
                print('Done: {0}/{1} train images'.format(count_processed, 500))


    print('Saving to .npy files :done.')


def preprocess(imgs):
    imgs = np.expand_dims(imgs, axis=4)
    print(' ---------------- preprocessed -----------------')
    return imgs

def preprocess_squeeze(imgs):
    imgs = np.squeeze(imgs, axis=4)
    print(' ---------------- preprocessed squeezed -----------------')
    return imgs


if __name__ == '__main__':
    create_train_data()
