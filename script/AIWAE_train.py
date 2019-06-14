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
parser.add_argument("--dataset", type = str, required = True)
parser.add_argument("--hidden_size", type = int, required = True)
parser.add_argument("--num_HMC_steps", type = int, required = True)
parser.add_argument("--num_samples", type = int,
                    required = True,
                    help = """num of samples used in Monte Carlo estimate of 
                              ELBO when using VAE; num of samples used in 
                              importance weighted ELBO when using IWAE.""")
parser.add_argument("--num_beta", type = int, required = True)
parser.add_argument("--batch_size", type = int, required = True)
parser.add_argument("--repeat", type = int, required = True)

## parse parameters
args = parser.parse_args()
hidden_size = args.hidden_size
num_beta = args.num_beta
L = args.num_HMC_steps
num_samples = args.num_samples
batch_size = args.batch_size
repeat = args.repeat

## read data
with open('./data/data.pkl', 'rb') as file_handle:
    data = pickle.load(file_handle)

train_image = data['train_image']
test_image = data['test_image']

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
encoder_optimizer = optim.Adam(aiwae.encoder.parameters(),
                               lr = 0.001,
                               eps = 1e-4)
decoder_optimizer = optim.Adam(aiwae.decoder.parameters(),
                               lr = 0.001,
                               eps = 1e-4)
lambda_lr = lambda epoch : 10**(-epoch/7.0)

encoder_scheduler_lr = optim.lr_scheduler.LambdaLR(encoder_optimizer, lambda_lr)
decoder_scheduler_lr = optim.lr_scheduler.LambdaLR(decoder_optimizer, lambda_lr)

idx_epoch = 2799
restart_model = torch.load("./output/model_hidden_size_2/AIWAE_num_samples_{}_num_beta_{}_batch_size_{}_epoch_{}_repeat_{}.pt".format(num_samples, num_beta, batch_size, idx_epoch, repeat))
aiwae.load_state_dict(restart_model['state_dict'])
decoder_optimizer.load_state_dict(restart_model['decoder_optimizer_state_dict'])
encoder_optimizer.load_state_dict(restart_model['encoder_optimizer_state_dict'])

## parameters for HMC
#epsilons = [0.1 for i in range(num_beta)]

epsilons = [0.1] + list(np.linspace(0.017, 0.009, num_beta - 1))

epsilon_min = 0.001
epsilon_max = 0.3
epsilon_decrease_alpha = 0.998
epsilon_increase_alpha = 1.002
epsilon_target = 0.6
betas = list(np.linspace(0, 1, num_beta))

num_epoch = 3280
#idx_epoch = 0

def calc_lr_idx(idx_epoch):
    count = [3**i for i in range(8)]
    count = np.cumsum(count)
    return bisect.bisect(count, idx_epoch)

while idx_epoch < num_epoch:
    lr_idx = calc_lr_idx(idx_epoch)
    encoder_scheduler_lr.step(lr_idx)
    decoder_scheduler_lr.step(lr_idx)

    start_time = time.time()
    
    for idx_step, data in enumerate(train_data_loader):
        data = data.float()
        x = data.cuda()            
        ###### train decoder
        decoder_loss, accept_rate = aiwae.decoder_loss(x, num_samples, epsilons, L, betas)

        decoder_loss = torch.mean(decoder_loss)
        decoder_optimizer.zero_grad()
        decoder_loss.backward()
        decoder_optimizer.step()

        accept_rate = list(accept_rate.cpu().data.squeeze().numpy())
        for l in range(1, num_beta):
            if accept_rate[l] < epsilon_target:
                epsilons[l] *= epsilon_decrease_alpha
            else:
                epsilons[l] *= epsilon_increase_alpha

            if epsilons[l] < epsilon_min:
                epsilons[l] = epsilon_min
            if epsilons[l] > epsilon_max:
                epsilons[l] = epsilon_max

        #### train encoder
        encoder_loss = aiwae.encoder_loss(x)
        encoder_loss = torch.mean(encoder_loss)
        encoder_optimizer.zero_grad()
        encoder_loss.backward()
        encoder_optimizer.step()

        elbo = aiwae.calc_elbo(x, num_samples)
        elbo = torch.mean(elbo)

        print("epoch: {:>3d}, step: {:>5d}, decoder_loss: {:.3f}, encoder_loss: {:.3f}, elbo: {:.3f}, epsilon_start: {:.3f}, epsilon_end: {:.3f}, lr: {:.5f}".format(
            idx_epoch, idx_step, decoder_loss.item(), encoder_loss.item(), elbo.item(), epsilons[1], epsilons[-1], encoder_optimizer.param_groups[0]['lr']), flush = True)

        # if idx_step >= 19:
        #     print("time: {:.2f}".format(time.time() - start_time))
        #     exit()
        
    if np.isnan(encoder_loss.item()):
        #restart_model = torch.load("./output/model/AIWAE_num_samples_{}_num_beta_{}_restart.pt".format(num_samples, num_beta))
        restart_model = torch.load("./output/model_hidden_size_2/AIWAE_num_samples_{}_num_beta_{}_batch_size_{}_repeat_{}_restart.pt".format(num_samples, num_beta, batch_size, repeat))
        aiwae.load_state_dict(restart_model['state_dict'])
        decoder_optimizer.load_state_dict(restart_model['decoder_optimizer_state_dict'])
        encoder_optimizer.load_state_dict(restart_model['encoder_optimizer_state_dict'])
        print("restart because of nan")
        continue
    
    torch.save({'state_dict': aiwae.state_dict(),
                'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                'args': args},
               "./output/model_hidden_size_2/AIWAE_num_samples_{}_num_beta_{}_batch_size_{}_repeat_{}_restart.pt".format(num_samples, num_beta, batch_size, repeat))
        
    if (idx_epoch + 1) % 80 == 0:
        torch.save({'state_dict': aiwae.state_dict(),
                    'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                    'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                    'args': args},
                   "./output/model_hidden_size_2/AIWAE_num_samples_{}_num_beta_{}_batch_size_{}_epoch_{}_repeat_{}.pt".format(num_samples, num_beta, batch_size, idx_epoch, repeat))
    idx_epoch += 1

torch.save({'state_dict': aiwae.state_dict(),
            'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
            'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
            'args': args},
           "./output/model_hidden_size_2/AIWAE_num_samples_{}_num_beta_{}_batch_size_{}_epoch_{}_repeat_{}.pt".format(num_samples, num_beta, batch_size, idx_epoch, repeat))
exit()
