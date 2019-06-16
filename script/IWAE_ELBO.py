#!/home/xqding/apps/anaconda3/bin/python

# Created by Xinqiang Ding (xqding@umich.edu)
# at 2019/03/21 14:55:17

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from data_utils import *
from AIWAE_models import *
from sys import exit
import argparse
import time
import os

## parameter parser
parser = argparse.ArgumentParser(description="Importance Weighted Auto-Encoder")
parser.add_argument("--dataset", type = str,
                    required = True)
parser.add_argument("--num_samples", type = int,
                    required = True,
                    help = """num of samples used in Monte Carlo estimate of 
                              ELBO when using VAE; num of samples used in 
                              importance weighted ELBO when using IWAE.""")
parser.add_argument("--hidden_size", type = int,
                    required = True, default = 50)
parser.add_argument("--epoch", type = int,
                    required = True)
parser.add_argument("--repeat", type = int,
                    required = True)

## parse parameters
args = parser.parse_args()
num_samples = args.num_samples
epoch = args.epoch
hidden_size = args.hidden_size
repeat = args.repeat

## read data
if args.dataset == "MNIST":
    with open("./data/MNIST.pkl", 'rb') as file_handle:
        data = pickle.load(file_handle)
        train_image = data['train_image']
        test_image = data['test_image']
        
    train_data = MNIST_Dataset(train_image)
    test_data = MNIST_Dataset(test_image)
    
elif args.dataset == "Omniglot":
    with open("./data/Omniglot.pkl", 'rb') as file_handle:
        data = pickle.load(file_handle)
        train_image = data['train_image']
        test_image = data['test_image']
        
    train_data = OMNIGLOT_Dataset(train_image)
    test_data = OMNIGLOT_Dataset(test_image)
    
else:
    raise("Dataset is wrong!")
batch_size = 32

## IWAE models
input_size = train_image.shape[-1]
output_size = train_image.shape[-1]

aiwae = AIWAE(input_size, hidden_size)
aiwae = aiwae.cuda()

idx_repeat = args.repeat

## load the trained model
state_dict = torch.load("./output/model/IWAE_dataset_{}_num_samples_{}_epoch_{}_repeat_{}.pt".format(args.dataset,
    num_samples, epoch, idx_repeat))
aiwae.load_state_dict(state_dict['state_dict'])

test_data_loader = DataLoader(test_data,
                              batch_size = batch_size)
elbo_test = []
for idx_step, data in enumerate(test_data_loader):
    print(idx_repeat, idx_step, flush = True)    
    data = data.cuda()

    ###### calculate IWAE loss
    elbo = aiwae.calc_elbo(data, 5000)
    elbo_test += list(elbo.cpu().data.numpy())

train_data_loader = DataLoader(train_data,
                              batch_size = batch_size)
elbo_train = []
for idx_step, data in enumerate(train_data_loader):
    print(idx_repeat, idx_step, flush = True)
    data = data.cuda()

    ###### calculate IWAE loss
    elbo = aiwae.calc_elbo(data, 5000)
    elbo_train += list(elbo.cpu().data.numpy())

    
with open("./output/ELBO/IWAE_dataset_{}_num_samples_{}_epoch_{}_repeat_{}.pkl".format(args.dataset, num_samples, epoch, idx_repeat), 'wb') as file_handle:
          pickle.dump({'elbo_test': elbo_test,
                       'elbo_train': elbo_train}, file_handle)
