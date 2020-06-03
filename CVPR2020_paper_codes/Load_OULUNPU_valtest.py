'''
Code of 'Searching Central Difference Convolutional Networks for Face Anti-Spoofing'
By Zitong Yu & Zhuo Su, 2019

If you use the code, please cite:
@inproceedings{yu2020searching,
    title={Searching Central Difference Convolutional Networks for Face Anti-Spoofing},
    author={Yu, Zitong and Zhao, Chenxu and Wang, Zezheng and Qin, Yunxiao and Su, Zhuo and Li, Xiaobai and Zhou, Feng and Zhao, Guoying},
    booktitle= {CVPR},
    year = {2020}
}

Only for research purpose, and commercial use is not allowed.

MIT License
Copyright (c) 2020
'''

from __future__ import print_function, division
import os
import torch
import pandas as pd
#from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os


frames_total = 8    # each video 8 uniform samples

face_scale = 1.3  #default for test and val
#face_scale = 1.1  #default for test and val

def crop_face_from_scene(image, scale, size):
    h_img, w_img = image.shape[0], image.shape[1]
    y_mid=(w_img)/2.0
    x_mid=(h_img)/2.0

    w_scale=scale*size
    h_scale=scale*size
    y1=y_mid-w_scale/2.0
    x1=x_mid-h_scale/2.0
    y2=y_mid+w_scale/2.0
    x2=x_mid+h_scale/2.0
    y1=max(math.floor(y1),0)
    x1=max(math.floor(x1),0)
    y2=min(math.floor(y2),w_img)
    x2=min(math.floor(x2),h_img)

    region=image[x1:x2,y1:y2]
    return region




class Normaliztion_valtest(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, val_map_x, spoofing_label = sample['image_x'],sample['val_map_x'] ,sample['spoofing_label']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        return {'image_x': new_image_x, 'val_map_x': val_map_x , 'spoofing_label': spoofing_label}


class ToTensor_valtest(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, val_map_x, spoofing_label = sample['image_x'],sample['val_map_x'] ,sample['spoofing_label']

        # swap color axis because    BGR2RGB
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W
        image_x = image_x[:,:,:,::-1].transpose((0, 3, 1, 2))
        image_x = np.array(image_x)

        val_map_x = np.array(val_map_x)

        spoofing_label_np = np.array([0],dtype=np.long)
        spoofing_label_np[0] = spoofing_label

        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'val_map_x': torch.from_numpy(val_map_x.astype(np.float)).float(),'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.long)).long()}


# /home/ztyu/FAS_dataset/OULU/Train_images/          6_3_20_5_121_scene.jpg        6_3_20_5_121_scene.dat
# /home/ztyu/FAS_dataset/OULU/IJCB_re/OULUtrain_images/        6_3_20_5_121_depth1D.jpg
class Spoofing_valtest(Dataset):

    def __init__(self, info_list, root_dir, val_map_dir,  transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=',', header=None)
        self.root_dir = root_dir
        self.val_map_dir = val_map_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)


    def __getitem__(self, idx):
        #print(self.landmarks_frame.iloc[idx, 0])
        videoname = str(self.landmarks_frame.iloc[idx, 1])
        image_path = os.path.join(self.root_dir, videoname)
        val_map_path = os.path.join(self.val_map_dir, videoname)

        image_x, val_map_x = self.get_single_image_x(image_path, val_map_path, videoname)

        spoofing_label = self.landmarks_frame.iloc[idx, 0]
        if spoofing_label == 1:
            spoofing_label = 1            # real
        else:
            spoofing_label = 0

        sample = {'image_x': image_x, 'val_map_x':val_map_x , 'spoofing_label': spoofing_label}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x(self, image_path, val_map_path, videoname):

        files_total = [name for name in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, name))]
        files_total = np.random.choice(files_total, max([len(files_total)//3,1]), replace=True)
        # interval = len(files_total)//10

        image_x = np.zeros((len(files_total), 256, 256, 3))
        val_map_x = np.ones((len(files_total), 32, 32))

        # random choose 1 frame
        for ii, image_name in enumerate(files_total):
            # RGB
            image_path2 = os.path.join(image_path, image_name)
            image_x_temp = cv2.imread(image_path2)



            # gray-map
            val_map_path_temp = os.path.join(val_map_path, image_name)
            val_map_x_temp = cv2.imread(val_map_path_temp, 0)

            # image_x[ii,:,:,:] = cv2.resize(crop_face_from_scene(image_x_temp, bbox_path, face_scale), (256, 256))
            image_x[ii,:,:,:] = cv2.resize(image_x_temp, (256, 256))
            # transform to binary mask --> threshold = 0
            # temp = cv2.resize(crop_face_from_scene(val_map_x_temp, bbox_path, face_scale), (32, 32))
            temp = cv2.resize(val_map_x_temp, (32, 32))
            np.where(temp < 1, temp, 1)
            val_map_x[ii,:,:] = temp


        return image_x, val_map_x
