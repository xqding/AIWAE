__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/03/26 18:23:53"

import numpy as np
import scipy.io
import pickle

data = scipy.io.loadmat("./chardata.mat")
train_data = data['data'].T.astype('float32')
test_data = data['testdata'].T.astype('float32')

data = {'train_image': train_data,
        'test_image': test_data}

with open("./Omniglot.pkl", 'wb') as file_handle:
    pickle.dump(data, file_handle)
