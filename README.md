# Annealed Importance Weighted Auto-Encoders (AIWAE)

This repository is part of the paper [Improving Importance Weighted Auto-Encoders with Annealed Importance Sampling](https://arxiv.org/abs/1906.04904) proposing AIWAE by Xinqiang Ding and David J. Freedman.

The repository includes both the datasets and source code that reproduce the results presented in the above paper.
It also includes code for calculating marginal probability density of samples using annealed importance sampling (AIS) and Hamiltonian Monte Carlo (HMC).

## Requirements
Python >= 3.5.0, PyTorch >= 1.0, Numpy, Pickle

## Dataset
Follow the [READEM.md](./data/README.md) in the `data` directory to download and process both MNIST and Omnilogt dataset.

## Train latent space models with IWAE/AIWAE
* With **IWAE**:
```bash
python ./script/AIWAE_train.py --dataset [MNIST,Omniglot]
                               --hidden_size 50
                               --num_samples 5    
                               --batch_size 20
                               --repeat 0 
```

* With **AIWAE**:
```bash
python ./script/AIWAE_train.py --dataset [MNIST,Omniglot]
                               --hidden_size 50
                               --num_HMC_steps 5 
                               --num_samples 5    
                               --num_beta 5   
                               --batch_size 128   
                               --repeat 0 
```

The parameter `hidden_size` is the dimension of the latent space. The parameter `num_samples` is the number of samples used in both IWAE and AIWAE. The parameter `num_beta` is the number of temperatures used in AIWAE. You can change values of these parameters to specify the model and the training algorithm.

## Evaluate models by calculating ELBO and NLL(negative log-likelihood)
* calculate ELBO for models trained with IWAE
```bash
python ./script/IWAE_ELBO.py --dataset [MNIST,Omniglot]
                             --num_samples 5 
                             --hidden_size 50 
                             --epoch 9 
                             --repeat 0
```

* calculate ELBO for models trained with AIWAE
```bash
python ./script/AIWAE_ELBO.py --dataset [MNIST,Omniglot]
                              --num_samples 5 
                              --hidden_size 50 
                              --num_beta 5
                              --epoch 9 
                              --repeat 0
```

* calculate NLL for models trained with IWAE
```bash
python ./script/IWAE_NLL.py --dataset [MNIST,Omniglot]
                            --num_samples 5 
                            --hidden_size 50 
                            --epoch 9 
                            --repeat 0
```

* calculate NLL for models trained with AIWAE
```bash
python ./script/AIWAE_NLL.py --dataset [MNIST,Omniglot]
                             --num_samples 5 
                             --hidden_size 50 
                             --num_beta 5
                             --epoch 9 
                             --repeat 0
```

Note that calculating NLL with annealed importance sampling (AIS) can be slow. Inside both `./script/IWAE_NLL.py` and `./script/AIWAE_NLL.py`, there is a variable called `num_beta_hmc` that specifies the number of temperatures used in AIS.
The accuracy of the calculated NLLs increases when increasing the value of this variable, which will slow down the calculation though. Therefore, you can change the value of the variable `num_beta_hmc` depends on how much computational resource you have.

