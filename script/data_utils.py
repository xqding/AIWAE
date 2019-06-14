import numpy as np
from torch.utils.data import Dataset, DataLoader

class MNIST_Dataset(Dataset):    
    def __init__(self, image):
        super(MNIST_Dataset).__init__()
        self.image = image
    def __len__(self):
        return self.image.shape[0]
    def __getitem__(self, idx):
        return np.random.binomial(1, self.image[idx, :]).astype('float32')

class OMNIGLOT_Dataset(Dataset):
    def __init__(self, image):
        super(OMNIGLOT_Dataset).__init__()
        self.image = image
    def __len__(self):
        return self.image.shape[0]
    def __getitem__(self, idx):
        return np.random.binomial(1, self.image[idx, :]).astype('float32')
