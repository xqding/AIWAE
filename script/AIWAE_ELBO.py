#!/home/xqding/apps/anaconda3/bin/python

# Created by Xinqiang Ding (xqding@umich.edu)
# at 2019/03/21 14:55:17

#SBATCH --job-name=ELBO
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
##SBATCH --exclude=gollum[003-045]
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH --output=./slurm_out/ELBO_%A_%a.out

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import sys
sys.path.append('/home/xqding/projects/AIWAE/MNIST/script/')
from data_utils import *
from AIWAE_models import *
from sys import exit
import argparse
import time
import os

## parameter parser
parser = argparse.ArgumentParser(description="Annealed Importance Weighted Auto-Encoder")
parser.add_argument("--dataset", type = str,
                    required = True)
parser.add_argument("--num_samples", type = int,
                    required = True,
                    help = """num of samples used in Monte Carlo estimate of 
                              ELBO when using VAE; num of samples used in 
                              importance weighted ELBO when using IWAE.""")
parser.add_argument("--num_beta", type = int,
                    required = True, default = 10)
parser.add_argument("--epoch", type = int,
                    required = True)

## parse parameters
args = parser.parse_args()
num_samples = args.num_samples
num_beta = args.num_beta
epoch = args.epoch

## read data
with open('./data/data.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)

train_image = data['train_image']
test_image = data['test_image']

batch_size = 32
if args.dataset == "MNIST":
    train_data = MNIST_Dataset(train_image)
    test_data = MNIST_Dataset(test_image)    
elif args.dataset == 'OMNIGLOT':
    train_data = OMNIGLOT_Dataset(train_image)
    test_data = OMNIGLOT_Dataset(test_image)
else:
    raise("Dataset is wrong")

## IWAE models
hidden_size = 50
input_size = train_image.shape[-1]
output_size = train_image.shape[-1]

aiwae = AIWAE(input_size, hidden_size)
aiwae = aiwae.cuda()

idx_repeat = int(os.environ['SLURM_ARRAY_TASK_ID'])

## load the trained model
# state_dict = torch.load("./output/model_from_turnip/AIWAE_num_samples_{}_num_beta_{}_epoch_{}.pt".format(
#     num_samples, num_beta, epoch))
# state_dict = torch.load("./output/model/AIWAE_num_samples_{}_num_beta_{}_epoch_{}.pt".format(
#     num_samples, num_beta, epoch))
state_dict = torch.load("./output/model/AIWAE_num_samples_{}_num_beta_{}_batch_size_128_epoch_{}_repeat_{}.pt".format(
    num_samples, num_beta, epoch, idx_repeat))
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

# assert(len(elbo_test) == 10000)
# assert(len(elbo_train) == 60000)

# assert(len(elbo_test) == 8070)
# assert(len(elbo_train) == 24345)
    
with open("./output/ELBO/AIWAE_num_samples_{}_num_beta_{}_batch_size_128_epoch_{}_repeat_{}.pt".format(num_samples, num_beta, epoch, idx_repeat), 'wb') as file_handle:
          pickle.dump({'elbo_test': elbo_test,
                       'elbo_train': elbo_train}, file_handle)

          
# elbo_test_repeat = np.array(elbo_test_repeat)
# elbo_train_repeat = np.array(elbo_train_repeat)

# elbo_test_mean = []
# elbo_train_mean = []

# for i in range(5):
#     elbo_test = np.copy(elbo_test_repeat[i,:])    
#     elbo_test = np.array(elbo_test)
#     elbo_test = elbo_test[elbo_test != -np.inf]
#     elbo_test_mean.append(np.mean(elbo_test))

#     elbo_train = np.copy(elbo_train_repeat[i,:])    
#     elbo_train = np.array(elbo_train)
#     elbo_train = elbo_train[elbo_train != -np.inf]
#     elbo_train_mean.append(np.mean(elbo_train))
# print("AIWAE: num_samples: {}, num_beta: {}".format(num_samples, num_beta))

# print("elbo_test: {:.2f} +- {:.2f}".format(np.mean(elbo_test_mean), np.std(elbo_test_mean)))
# print("elbo_train: {:.2f} +- {:.2f}".format(np.mean(elbo_train_mean), np.std(elbo_train_mean)))
