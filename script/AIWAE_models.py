import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.autograd as autograd

class AIWAE_Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(AIWAE_Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.transform = nn.Sequential(
            nn.Linear(hidden_size, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, output_size),
            nn.Sigmoid())
    
    def forward(self, h):
        p = self.transform(h)
        return p
    
    def calc_logPxh(self, x, h):
        log_Ph = torch.sum(-0.5*h**2 - 0.5*torch.log(2*h.new_tensor(np.pi)), -1)
        p = self.forward(h)
        log_PxGh = torch.sum(x*torch.log(p) + (1-x)*torch.log(1-p), -1)        
        logPxh = log_Ph + log_PxGh
        return logPxh
    
    

class AIWAE_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AIWAE_Encoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.transform = nn.Sequential(
            nn.Linear(input_size, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh())
        self.fc_mu = nn.Linear(200, hidden_size)
        self.fc_logsigma = nn.Linear(200, hidden_size)

    def forward(self, x):
        out = self.transform(x)
        mu = self.fc_mu(out)
        logsigma = self.fc_logsigma(out)
        sigma = torch.exp(logsigma)
        return mu, sigma
            
    def calc_logQhGx(self, x, h):
        mu, sigma = self.forward(x)
        log_QhGx = torch.sum(-0.5*((h-mu)/sigma)**2-0.5*torch.log(2*mu.new_tensor(np.pi)*sigma**2), -1)
        return log_QhGx
    

class AIWAE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AIWAE, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = AIWAE_Encoder(self.input_size, self.hidden_size)
        self.decoder = AIWAE_Decoder(self.hidden_size, self.input_size)


    def calc_energy(self, h, x, beta):
        """ Calculate energy U 
        """
        with torch.no_grad():
            log_QhGx = self.encoder.calc_logQhGx(x, h)
            log_Pxh = self.decoder.calc_logPxh(x,h)
            energy = - (beta * log_Pxh + (1-beta)*log_QhGx)    
        return energy.data, log_QhGx.data, log_Pxh.data

    def calc_energy_and_force(self, h, x, beta):
        """ Calculate energy U and gradient of U with respect to h
        """
        h = h.clone().detach()
        h.requires_grad_(True)
        log_QhGx = self.encoder.calc_logQhGx(x, h)
        log_Pxh = self.decoder.calc_logPxh(x,h)
        energy = - (beta * log_Pxh + (1-beta)*log_QhGx)
        h_grad = autograd.grad(outputs = energy,
                               inputs = h,
                               grad_outputs = torch.ones_like(energy))[0]
        return energy.data, h_grad.data, log_QhGx.data, log_Pxh.data

    def HMC(self, current_q, x, epsilon, L, beta):
        ''' Hamiltonian Monte Carlo (HMC) Algorithm    
        '''
        ## sample a new momentum
        current_q.requires_grad_(False)
        current_p = torch.randn_like(current_q)

        #### update momentum and position ####
        ## proposed (q,p)
        q = current_q.clone().detach()
        p = current_p.clone().detach()        

        ## propogate state (q,p) using Hamiltonian equation with
        ## leapfrog integration

        ## propagate momentum by a half step at the beginning
        U, grad_U, _, _ = self.calc_energy_and_force(q, x, beta)
        p = p - 0.5*epsilon*grad_U

        ## propagate position and momentum alternatively by a full step
        ## L is the number of steps
        for i in range(L):
            q = q + epsilon * p
            U, grad_U, _, _ = self.calc_energy_and_force(q, x, beta)
            if i != L-1:
                p = p - epsilon*grad_U

        ## propagate momentum by a half step at the end
        p = p - 0.5*epsilon*grad_U

        ## calculate Hamiltonian of current state and proposed state
        current_U, _, _ = self.calc_energy(current_q, x, beta)
        current_K = torch.sum(0.5*current_p**2, -1)
        current_E = current_U + current_K

        proposed_U, _, _ = self.calc_energy(q, x, beta)
        proposed_K = torch.sum(0.5*p**2, -1)
        proposed_E = proposed_U + proposed_K

        ## accept proposed state using Metropolis criterion
        flag_accept = torch.rand_like(proposed_E) <= torch.exp(-(proposed_E - current_E))
        current_q[flag_accept] = q[flag_accept]

        return flag_accept, current_q
    
    def decoder_loss(self, x, num_annealed_samples, epsilons, L, betas):
        x = x.expand(num_annealed_samples, x.shape[0], x.shape[1])
        accept_rate = x.new_ones(len(betas))
        
        #### annealed importance sampling
        ## sample from beta = 0
        with torch.no_grad():
            mu, sigma = self.encoder(x)        
            h = mu + sigma * torch.randn_like(mu)
            
        _, log_QhGx, log_Pxh = self.calc_energy(h, x, betas[0])
        log_w = (betas[1] - betas[0])*(log_Pxh - log_QhGx)
        
        accept_rate[0] = 1.0
        
        ## sample from beta > 0 and beta < 1
        for k in range(1, len(betas)-1):
            flag_accept, h = self.HMC(h, x, epsilons[k], L, betas[k])
            _, log_QhGx, log_Pxh = self.calc_energy(h, x, betas[k])
            log_w += (betas[k+1] - betas[k])*(log_Pxh - log_QhGx)

            accept_rate[k] = flag_accept.float().mean().item()
            #accept_rate.append(flag_accept.float().mean().item())
            
        ## sample from beta = 1
        flag_accept, h = self.HMC(h, x, epsilons[-1], L, betas[-1])
        accept_rate[-1] = flag_accept.float().mean().item()
        #accept_rate.append(flag_accept.float().mean().item())
            
        ## calculate annealed importance weights
        log_w = log_w - log_w.max(0)[0]
        w = torch.exp(log_w)
        w = w / w.sum(0)

        ## calculate decoder loss
        log_Pxh = self.decoder.calc_logPxh(x, h)        
        #loss = -torch.mean(torch.sum(w * log_Pxh, 0))
        #return loss, accept_rate
        loss = -torch.sum(w * log_Pxh, 0)

        accept_rate = accept_rate.reshape(1,-1)
        #accept_rate = torch.tensor(accept_rate).reshape(1,-1)
        return loss, accept_rate
        

    def encoder_loss_multiple_samples(self, x, num_samples):
        x = x.expand(num_samples, x.shape[0], x.shape[1])
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(mu)
        h = mu + sigma * eps
        
        log_Pxh = self.decoder.calc_logPxh(x, h)
        log_QhGx = torch.sum(-0.5*(eps)**2 -
                             0.5*torch.log(2*h.new_tensor(np.pi))
                             - torch.log(sigma), -1)
        log_weight = (log_Pxh - log_QhGx).detach().data
        log_weight = log_weight - torch.max(log_weight, 0)[0]
        weight = torch.exp(log_weight)
        weight = weight / torch.sum(weight, 0)
        #loss = -torch.mean(torch.sum(weight * (log_Pxh - log_QhGx), 0))
        
        loss = -torch.sum(weight * (log_Pxh - log_QhGx), 0)
        return loss

    def encoder_loss(self, x):
        mu, sigma = self.encoder(x)
        eps = torch.randn_like(mu)
        h = mu + sigma * eps
        p = self.decoder(h)        
        log_PxGh = torch.sum(x*torch.log(p) + (1-x)*torch.log(1-p), -1)
        DKL = 0.5 * torch.sum(mu**2 + sigma**2 - 1.0 - torch.log(sigma**2), -1)        
        loss = -log_PxGh + DKL
        
        return loss
    
    def forward(self, x, num_annealed_samples, epsilons, L, betas):
        decoder_loss, accept_rate = self.decoder_loss(x, num_annealed_samples, epsilons, L, betas)
        encoder_loss = self.encoder_loss(x)
        return decoder_loss, accept_rate, encoder_loss
        

    def calc_elbo(self, x, num_samples):
        with torch.no_grad():
            x = x.expand(num_samples, x.shape[0], x.shape[1])
            mu, sigma = self.encoder(x)
            eps = torch.randn_like(mu)
            h = mu + sigma * eps

            log_Pxh = self.decoder.calc_logPxh(x, h)
            log_QhGx = torch.sum(-0.5*(eps)**2 -
                                 0.5*torch.log(2*h.new_tensor(np.pi))
                                 - torch.log(sigma), -1)
            log_weight = (log_Pxh - log_QhGx).detach().data
            log_weight = log_weight.double()
            weight = torch.exp(log_weight)
            elbo = -torch.log(torch.mean(weight, 0))
            return elbo

    
