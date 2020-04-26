#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:39:55 2020

@author: zhe
"""
from __future__ import print_function, division
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import json
from numpy import asarray
import torchvision.datasets.utils as dataset_utils
import torchvision.transforms as transforms

seed_num = 17
random.seed(seed_num)

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



  

def load_img(img_root, idx):
    im = Image.open(img_root + '/' + idx + '.png') 
    im.convert('RGB')
    im = asarray(im)
    im1 = Image.open(img_root + '/' + idx + '_2.png') 
    im1.convert('RGB')
    im1 = asarray(im1)
    return im, im1
    


class spot_and_diff_Dataset(Dataset):

    def __init__(self, csv_file, img_root, save_root, max_samples=-1,transform=None,testing=False):
       
        self.testing = testing
        self.transform = transform
        self.img_root = img_root
        self.data_root = csv_file
        self.spot_and_diff_dir = save_root  
        self.prepare_dataset()
        if testing in ['train', 'val', 'test']:
            self.data_label_tuples = torch.load(os.path.join(self.spot_and_diff_dir, testing) + '.pt')
        
        

    def __getitem__(self, index):
        """
        Args:
        index (int): Index
        
        Returns:
        tuple: (image, target) where target is index of the target class.
        """
        img_0, img_1, text = self.data_label_tuples[index]['img_0'], self.data_label_tuples[index]['img_1'], self.data_label_tuples[index]['sentences']

        if self.transform:
            img_0 = self.transform(img_0)
            img_1 = self.transform(img_1)
            text = text
            

        return img_0, img_1, text

    def __len__(self):
        return len(self.data_label_tuples)
    
    def prepare_dataset(self):
        
        if os.path.exists(os.path.join(self.spot_and_diff_dir, 'train.pt'))\
            and os.path.exists(os.path.join(self.spot_and_diff_dir, 'val.pt'))\
            and os.path.exists(os.path.join(self.spot_and_diff_dir, 'test.pt')):
            print('dataset already exists')
            return 
        
        
        
        if self.testing == 'train':
            if os.path.exists(os.path.join(self.spot_and_diff_dir, 'train.pt')):
                print('training data exists')
            else:
                train = []
                with open(self.data_root) as f:
                    data = json.load(f)
                    for i in range(len(data)):
                        idx = data[i]['img_id']
                        sentences = data[i]['sentences']
                        img_0, img_1 = load_img(self.img_root, idx)
                        sample = {'img_0':img_0, 'img_1': img_1, 'sentences': sentences}
                        train.append(sample)
                        print(i)
        elif self.testing == 'val':
             if os.path.exists(os.path.join(self.spot_and_diff_dir, 'val.pt')):
                 print('valid data exists')
             else:
                valid = []
                with open(self.data_root) as f:
                    data = json.load(f)
                    for i in range(len(data)):
                        idx = data[i]['img_id']
                        sentences = data[i]['sentences']
                        img_0, img_1 = load_img(self.img_root, idx)
                        sample = {'img_0':img_0, 'img_1': img_1, 'sentences': sentences}
                        valid.append(sample)
                        print(i)
                        
        elif self.testing == 'test':
             if os.path.exists(os.path.join(self.spot_and_diff_dir, 'test.pt')):
                 print('test data exists')
             else:
                test = []
                with open(self.data_root) as f:
                    data = json.load(f)
                    for i in range(len(data)):
                        idx = data[i]['img_id']
                        sentences = data[i]['sentences']
                        img_0, img_1 = load_img(self.img_root, idx)
                        sample = {'img_0':img_0, 'img_1': img_1, 'sentences': sentences}
                        test.append(sample)
                        print(i)
        
        
        dataset_utils.makedir_exist_ok(self.spot_and_diff_dir)
        if self.testing == 'train':
            torch.save(train, os.path.join(self.spot_and_diff_dir, 'train.pt'))
        elif self.testing == 'val':
            torch.save(valid, os.path.join(self.spot_and_diff_dir, 'val.pt'))
        elif self.testing == 'test':
            torch.save(test, os.path.join(self.spot_and_diff_dir, 'test.pt'))
                    
            
                
            
save_root = './datanew'    
data_root = './data/annotations'
img_root = './resized_images'
train_set = spot_and_diff_Dataset(csv_file=os.path.join(data_root,'train.json'),
                                    img_root=img_root,
                                    save_root = save_root,
                                    transform=transforms.ToTensor(),
                                    testing='train',
                                  )  

valid_set = spot_and_diff_Dataset(csv_file=os.path.join(data_root,'val.json'),
                                    img_root=img_root,
                                    save_root= save_root,
                                    transform=transforms.ToTensor(),
                                    testing='val',
                                  ) 

                                   
test_set = spot_and_diff_Dataset(csv_file=os.path.join(data_root,'test.json'),
                                    img_root=img_root,
                                    save_root = save_root,
                                    transform=transforms.ToTensor(),
                                    testing='test',
                                  ) 
a = valid_set[0][0]
b = valid_set[0][1]
c = valid_set[0][2]
train_data = DataLoader(valid_set, batch_size = 64, shuffle=True)    