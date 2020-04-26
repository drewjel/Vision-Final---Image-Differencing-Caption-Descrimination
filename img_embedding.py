#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 22:27:47 2020

@author: zhe
"""
import torch
import torch.nn as nn
import torchvision.models as models


class ImageEmbedding(nn.Module):
   def __init__(self, out_dim):
        super(ImageEmbedding, self).__init__()
        self.base_net = models.resnet18(pretrained=True)
        model_list = list(self.base_net.children())[:-1]
        model_list += [nn.Flatten(), nn.Linear(512, out_dim)]
        self.embed = nn.Sequential(*model_list)
        
   def forward(self, img_0, img_1):
       
       embed_0 = self.embed(img_0)
       embed_1 = self.embed(img_1)
       diff = embed_0 - embed_1
       return diff
        
