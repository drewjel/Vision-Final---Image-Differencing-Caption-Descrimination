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
    def __init__(self):
        super(ImageEmbedding, self).__init__()
        self.base_net = models.resnet50(pretrained=True)
        model_list = list(self.base_net.children())[:-1]
        model_list += [nn.Flatten()]
        self.embed = nn.Sequential(*model_list)

    def forward(self, img_0, img_1):
        self.base_net.eval()
        for param in self.base_net.parameters():
            param.requires_grad = False

        embed_0 = self.embed(img_0)
        embed_1 = self.embed(img_1)
        return torch.cat([embed_0, embed_1], dim=-1)
