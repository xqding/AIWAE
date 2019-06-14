## Download both MNIST and Omniglot dataset

First, change your current working directory into the directory ./data before running any following scripts.

To download the MNIST dataset, run the script `python ./download_MNIST.py`.
It will save the MNIST dataset as a pickle file MNIST.pkl in the current working directory.

The Omniglot dataset is from the importance weighted auto-encoders git repository https://github.com/yburda/iwae.
You need to first download the file chardata.mat via the address https://github.com/yburda/iwae/blob/master/datasets/OMNIGLOT/chardata.mat or the address https://drive.google.com/file/d/1Jb8FYVUOqHqK4eoN02fHpNhJvpB6qaYu/view?usp=sharing and put the file chardata.mat in the ./data directory.
Then running the scripy `python process_Omniglot_dataset.py`  will process the file chardata.mat and generate a pickle file Omniglot.pkl in the current working directory.

