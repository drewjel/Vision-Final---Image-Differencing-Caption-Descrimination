from pathlib import Path
import time
import math

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

import os

from logger import Logger
from data_loader import Spot_and_diff_dataset, pad_collate
from model import DiffEval

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda:0')
if torch.cuda.is_available():
    torch.cuda.set_device(device)


def compute_loss_and_metrics(out, target):
    loss = torch.nn.functional.cross_entropy(out, target)
    _, predicted = torch.max(out.data, -1)
    acc = (predicted == target).sum().float() / target.size(0)
    return loss, acc

def train(model, train_data, val_data, bptt=16):
    lr = 0.003  # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epoch = 100
    logger = Logger(len(train_data), log_interval=10)
    for epoch in range(num_epoch):
        logger.reset()
        model.train()  # Turn on the train mode
        tot_loss = 0
        for batch, batch_data in enumerate(train_data):
            img1, img2, sents, sent_lens, label = batch_data
            img1 = img1.to(device)
            img2 = img2.to(device)
            sents = sents.to(device)
            label = label.to(device)

            out = model(img1, img2, sents, sent_lens)
            loss, acc = compute_loss_and_metrics(out, label)
            loss.backward()
            optimizer.step()
            logger.set_values(loss.item(), acc.item())
            logger.print_log(batch, epoch, lr)


if __name__ == "__main__":
    save_dir = 'processed_data'
    data_dir = 'data/annotations'
    img_dir = 'data/resized_images'
    test_set = Spot_and_diff_dataset(csv_file=os.path.join(data_dir, 'test.json'),
                                     img_dir=img_dir,
                                     save_dir=save_dir,
                                     mode='test',
                                     )
    bptt = 32
    test_data = DataLoader(test_set, batch_size=bptt,
                           shuffle=True, collate_fn=pad_collate)
    model = DiffEval()
    model.to(device)
    train(model, test_data, test_data, bptt=bptt)
