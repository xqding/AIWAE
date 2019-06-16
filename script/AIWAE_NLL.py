#!/home/xqding/apps/anaconda3/bin/python

# Created by Xinqiang Ding (xqding@umich.edu)
# at 2019/03/21 14:55:17

#SBATCH --job-name=AIS
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --exclude=gollum[003-045]
#SBATCH --gres=gpu:1
#SBATCH --array=0-199%50
#SBATCH --output=./slurm_out/AIS_%A_%a.out

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
parser.add_argument("--hidden_size", type = int,
                    required = True, default = 50)
parser.add_argument("--epoch", type = int,
                    required = True)
parser.add_argument("--repeat", type = int,
                    required = True)

## parse parameters
args = parser.parse_args()
num_samples = args.num_samples
num_beta = args.num_beta
epoch = args.epoch
hidden_size = args.hidden_size
idx_repeat = args.repeat

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

batch_size = 256

## IWAE models
input_size = train_image.shape[-1]
output_size = train_image.shape[-1]

aiwae = AIWAE(input_size, hidden_size)
aiwae = aiwae.cuda()

## load the trained model
state_dict = torch.load("./output/model/AIWAE_dataset_{}_num_samples_{}_num_beta_{}_batch_size_128_epoch_{}_repeat_{}.pt".format(args.dataset, num_samples, num_beta, epoch, idx_repeat))
aiwae.load_state_dict(state_dict['state_dict'])

## parameters for HMC
num_beta_hmc = 10000
betas = list(np.linspace(0, 1, num_beta_hmc))
epsilon = 0.3
epsilon_min = 0.02
epsilon_max = 0.3
epsilon_decrease_alpha = 0.998
epsilon_increase_alpha = 1.002
epsilon_target = 0.6
L = 10

## calculate NLL for test data
test_data_loader = DataLoader(test_data,
                              batch_size = batch_size)

NLL_test = []
log_w_test = []

for idx_step, data in enumerate(test_data_loader):
    print(idx_step)    
    data = data.cuda()

    ###### calculate IWAE loss
    x = data.expand(16, data.shape[0], 784)
    mu, sigma = aiwae.encoder(x)
    eps = torch.randn_like(mu)
    h = mu + sigma * eps
    
    log_w = 0
    epsilon = 0.1
    for i in range(num_beta_hmc-1):
        log_Pxh = aiwae.decoder.calc_logPxh(x, h)
        log_QhGx = aiwae.encoder.calc_logQhGx(x, h)
        
        log_w += (betas[i+1] - betas[i])*(log_Pxh.data - log_QhGx.data)

        flag_accept, h = aiwae.HMC(h.detach().clone(), x, epsilon, L, betas[i+1])

        accept_rate = flag_accept.float().mean().item()
        if accept_rate > epsilon_target:
            epsilon *= epsilon_increase_alpha
        else:
            epsilon *= epsilon_decrease_alpha
            
        print("beta_idx: {:>4d}, beta: {:.2f}, accept_rate: {:.3f}, epsilon: {:.3f}".format(i, betas[i+1], accept_rate, epsilon), flush = True)

    log_w_test.append(log_w.cpu().clone().numpy())
        
    log_w = log_w.double()
    log_w_min = log_w.min(0)[0]
    log_w = log_w - log_w_min
    w = torch.exp(log_w)
    nll = -(torch.log(torch.mean(w, 0)) + log_w_min)
    nll = list(nll.detach().cpu().data.numpy())
    NLL_test += nll


## calculate NLL for train data
train_data_loader = DataLoader(train_data,
                              batch_size = batch_size)

NLL_train = []
log_w_train = []

for idx_step, data in enumerate(train_data_loader):
    print(idx_step)    
    data = data.cuda()

    ###### calculate IWAE loss
    x = data.expand(16, data.shape[0], 784)
    mu, sigma = aiwae.encoder(x)
    eps = torch.randn_like(mu)
    h = mu + sigma * eps
    
    log_w = 0
    epsilon = 0.1
    for i in range(num_beta_hmc-1):
        log_Pxh = aiwae.decoder.calc_logPxh(x, h)
        log_QhGx = aiwae.encoder.calc_logQhGx(x, h)
        
        log_w += (betas[i+1] - betas[i])*(log_Pxh.data - log_QhGx.data)

        flag_accept, h = aiwae.HMC(h.detach().clone(), x, epsilon, L, betas[i+1])

        accept_rate = flag_accept.float().mean().item()
        if accept_rate > epsilon_target:
            epsilon *= epsilon_increase_alpha
        else:
            epsilon *= epsilon_decrease_alpha
            
        print("beta_idx: {:>4d}, beta: {:.2f}, accept_rate: {:.3f}, epsilon: {:.3f}".format(
            i, betas[i+1], accept_rate, epsilon), flush = True)

    log_w_train.append(log_w.cpu().clone().numpy())
        
    log_w = log_w.double()
    log_w_min = log_w.min(0)[0]
    log_w = log_w - log_w_min
    w = torch.exp(log_w)
    nll = -(torch.log(torch.mean(w, 0)) + log_w_min)
    nll = list(nll.detach().cpu().data.numpy())
    NLL_train += nll

with open("./output/NLL/AIWAE_NLL_dataset_{}_num_samples_{}_num_beta_{}_batch_size_128_epoch_{}_repeat_{}.pkl".format(args.dataset, num_samples, num_beta, epoch, idx_repeat), 'wb') as file_handle:
    
    pickle.dump({'log_w_test': log_w_test,
                 'NLL_test': NLL_test,
                 'log_w_train': log_w_train,
                 'NLL_train': NLL_train}, file_handle)
print("AIWAE, num_samples: {}, num_beta: {}, epoch: {}, NLL_train: {:.3f}, NLL_test: {:.3f}".format(
    num_samples, num_beta, epoch, np.mean(NLL_train), np.mean(NLL_test)))
