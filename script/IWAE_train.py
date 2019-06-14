import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import sys
from data_utils import *
from AIWAE_models import *
from sys import exit
import argparse
import time
import bisect

## parameter parser
parser = argparse.ArgumentParser(description="Annealed Importance Weighted Auto-Encoder")
parser.add_argument("--dataset", type = str,
                    required = True)
parser.add_argument("--hidden_size", type = int)
parser.add_argument("--num_samples", type = int,
                    required = True,
                    help = """num of samples used in Monte Carlo estimate of 
                              ELBO when using VAE; num of samples used in 
                              importance weighted ELBO when using IWAE.""")
parser.add_argument("--repeat", type = int)

## parse parameters
args = parser.parse_args()
hidden_size = args.hidden_size
num_samples = args.num_samples
repeat = args.repeat

## read data
with open('./data/data.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)

train_image = data['train_image']
test_image = data['test_image']

#batch_size = 256
batch_size = 20
if args.dataset == "MNIST":
    train_data = MNIST_Dataset(train_image)
    test_data = MNIST_Dataset(test_image)    
elif args.dataset == 'OMNIGLOT':
    train_data = OMNIGLOT_Dataset(train_image)
    test_data = OMNIGLOT_Dataset(test_image)
else:
    raise("Dataset is wrong")

train_data_loader = DataLoader(train_data,
                               batch_size = batch_size,
                               shuffle = True)
test_data_loader = DataLoader(test_data, batch_size = batch_size)

## IWAE models
hidden_size = args.hidden_size
input_size = train_image.shape[-1]
output_size = train_image.shape[-1]

aiwae = AIWAE(input_size, hidden_size)
aiwae = aiwae.cuda()

## optimizer
optimizer = optim.Adam(aiwae.parameters(), lr = 0.001, eps = 1e-4)
lambda_lr = lambda epoch : 10**(-epoch/7.0)
scheduler_lr = optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)

num_epoch = 3280
idx_epoch = 0

def calc_lr_idx(idx_epoch):
    count = [3**i for i in range(8)]
    count = np.cumsum(count)
    return bisect.bisect(count, idx_epoch)

while idx_epoch < num_epoch:
    lr_idx = calc_lr_idx(idx_epoch)
    scheduler_lr.step(lr_idx)

    for idx_step, data in enumerate(train_data_loader):
        data = data.float()
        x = data.cuda()
        
        ###### train decoder
        if num_samples == 1:
            loss = aiwae.encoder_loss(x)
        else:
            loss = aiwae.encoder_loss_multiple_samples(x, num_samples)
            
        loss = torch.mean(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elbo = aiwae.calc_elbo(x, num_samples)
        elbo = torch.mean(elbo)

        print("epoch: {:>3d}, step: {:>5d}, loss: {:.3f}, elbo: {:.3f}, lr: {:.5f}".format(
            idx_epoch, idx_step, loss.item(), elbo.item(), optimizer.param_groups[0]['lr']), flush = True)
        
        # if idx_step >= 19:
        #     print("time used: {:.2f}".format(time.time() - start_time))
        #     exit()

        
    if np.isnan(loss.item()):
        model_state_dict = torch.load("./output/model_hidden_size_2/IWAE_num_samples_{}_batch_size_{}_repeat_{}_restart.pt".format(num_samples, batch_size, repeat))
        aiwae.load_state_dict(model_state_dict['state_dict'])
        optimizer.load_state_dict(model_state_dict['optimizer_state_dict'])
        print("restart because of nan")
        continue
        
    torch.save({'state_dict': aiwae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args},
               "./output/model_hidden_size_2/IWAE_num_samples_{}_batch_size_{}_repeat_{}_restart.pt".format(num_samples, batch_size, repeat))
        
    if (idx_epoch + 1) % 80 == 0:
        torch.save({'state_dict': aiwae.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': args},
                   "./output/model_hidden_size_2/IWAE_num_samples_{}_batch_size_{}_epoch_{}_repeat_{}.pt".format(num_samples, batch_size, idx_epoch, repeat))    
    idx_epoch += 1

torch.save({'state_dict': aiwae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': args},
           "./output/model_hidden_size_2/IWAE_num_samples_{}_batch_size_{}_epoch_{}_repeat_{}.pt".format(num_samples, batch_size, idx_epoch, repeat))
exit()
