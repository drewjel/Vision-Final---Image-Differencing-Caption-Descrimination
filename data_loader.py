#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:39:55 2020

@author: zhe
"""
from __future__ import print_function, division
import warnings
import os
import torch
import random
import json
import math

import numpy as np
from numpy import asarray

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.datasets.utils as dataset_utils
import torchvision.transforms as transforms

from transformers import DistilBertTokenizerFast


def img2tensor(img_dir, idx):
    im = Image.open(img_dir + '/' + idx + '.png')
    im.convert('RGB')
    im = asarray(im)
    im1 = Image.open(img_dir + '/' + idx + '_2.png')
    im1.convert('RGB')
    im1 = asarray(im1)
    return im, im1


def text2tensor(sentences, tokenizer):
    # SC: Currently, we concat all sentences into a long sentence for simplicity.
    return torch.tensor([tokenizer.encode(''.join(sentences), add_special_tokens=True)])


def pad_collate(batch):
    (img1s, img2s, sents, label) = zip(*batch)
    sents = [sent.squeeze(0) for sent in sents]
    sent_lens = [len(x) for x in sents]
    sents = pad_sequence(sents, batch_first=True, padding_value=0)
    return torch.stack(img1s), torch.stack(img2s), sents, sent_lens, torch.tensor(label)


class Spot_and_diff_dataset(Dataset):

    def __init__(self, csv_file, img_dir, save_dir, max_samples=-1, transform=transforms.ToTensor(), mode='test'):
        MODES = ['train', 'val', 'test']
        if mode not in MODES:
            raise ValueError('mode must be specified as one of ', MODES)
        self.mode = mode
        self.transform = transform
        self.img_dir = img_dir
        self.data_dir = csv_file
        self.spot_and_diff_dir = save_dir
        self.all_sentences = []
        self.prepare_dataset()
        self.data_label_tuples = torch.load(
            os.path.join(self.spot_and_diff_dir, mode) + '.pt')

    def __getitem__(self, index):
        """
        Args:
        index (int): Index

        Returns:
        tuple: (image, target) where target is index of the target class.
        """
        img_0, img_1, text, label = self.data_label_tuples[index]['img_0'], self.data_label_tuples[
            index]['img_1'], self.data_label_tuples[index]['sentences'], self.data_label_tuples[index]['label']

        if self.transform:
            img_0 = self.transform(img_0)
            img_1 = self.transform(img_1)

        return img_0, img_1, text, label

    def __len__(self):
        return len(self.data_label_tuples)

    def torch_bernoulli(self):
        return (torch.rand(1)).float()

    def augment_sample_sentences(self, sample, n_replacement = 1):

        sents = sample['sentences']
        n_sents = len(sents)
        replace_order = np.random.permutation(n_sents)

        for i in range(n_replacement):
            sent_to_replace = sents[replace_order[i]]
            while sent_to_replace in sents:
                sent_to_replace = self.all_sentences[np.random.choice(
                    len(self.all_sentences))]
            sents[replace_order[i]] = sent_to_replace

        return sample

    def prepare_dataset(self):
        saved_path = os.path.join(self.spot_and_diff_dir, self.mode + '.pt')
        if os.path.exists(saved_path):
            print(self.mode, 'data exists, read from', saved_path)
        else:
            raw_dataset = []
            neg_dataset = []
            tokenizer = DistilBertTokenizerFast.from_pretrained(
                'distilbert-base-cased')
            with open(self.data_dir) as f:
                data = json.load(f)
                for i in range(len(data)):
                    idx = data[i]['img_id']
                    sentences = data[i]['sentences']
                    self.all_sentences += sentences

                    img_0, img_1 = img2tensor(self.img_dir, idx)
                    sample = {'img_0': img_0, 'img_1': img_1,
                              'sentences': sentences, 'label': 1}
                    raw_dataset.append(sample)
                    if i % 100 == 99:
                        print(i, '/', len(data))
                for i in range(len(data)):
                    # Here, we do shallow copy to avoid dict-level in-place modification
                    current_sample = {**raw_dataset[i]}
                    current_sample['sentences'] = [] + current_sample['sentences']
                    current_sample['label'] = 0

                    if self.torch_bernoulli() > .5:
                        negative_sample = self.augment_sample_sentences(
                            current_sample, n_replacement=1)
                        neg_dataset.append(current_sample)
                    if i % 100 == 99:
                        print(i, '/', len(data))
            
            new_dataset = raw_dataset + neg_dataset
            for r in new_dataset:
                r['sentences'] = text2tensor(r['sentences'], tokenizer)

            os.makedirs(self.spot_and_diff_dir, exist_ok=True)
            torch.save(new_dataset, saved_path)
            print('Saved to', saved_path)
            print(len(raw_dataset), 'positive samples')
            print(len(neg_dataset), 'negative samples')


if __name__ == "__main__":
    seed_num = 17
    random.seed(seed_num)

    # Ignore warnings
    warnings.filterwarnings("ignore")
    save_dir = 'processed_data'
    data_dir = 'data/annotations'
    img_dir = 'data/resized_images'
    # train_set = Spot_and_diff_dataset(csv_file=os.path.join(data_dir, 'train.json'),
    #                                   img_dir=img_dir,
    #                                   save_dir=save_dir,
    #                                   transform=transforms.ToTensor(),
    #                                   mode='train',
    #                                   )
    # train_data = DataLoader(train_set, batch_size=64, shuffle=True)

    # valid_set = Spot_and_diff_dataset(csv_file=os.path.join(data_dir, 'val.json'),
    #                                 img_dir=img_dir,
    #                                 save_dir=save_dir,
    #                                 transform=transforms.ToTensor(),
    #                                 mode='val',
    #                                 )

    test_set = Spot_and_diff_dataset(csv_file=os.path.join(data_dir, 'test.json'),
                                     img_dir=img_dir,
                                     save_dir=save_dir,
                                     transform=transforms.ToTensor(),
                                     mode='test',
                                     )
