# import os
# import numpy as np
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import random
# import time
# import copy
# import math
# import PIL
# import pandas as pd
#
# import torch
# import torch.nn as nn
#
# from torch.nn import CTCLoss
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torch.utils.data.sampler import SubsetRandomSampler
# import torch.optim.lr_scheduler as lrs
#
# import torchvision
# import torchvision.utils as utils
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# import torchvision.models as models
#
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# from torch.autograd import Variable
# from torchvision import datasets, transforms

vocab=['-','.'," ",' ',"'",'!',',']+[chr(ord('a')+i) for i in range(26)]+[chr(ord('A')+i) for i in range(26)]+[chr(ord('0')+i) for i in range(10)]
chrToindex={}
indexTochr={}
cnt=0
for c in vocab:
    chrToindex[c]=cnt
    indexTochr[cnt]=c
    cnt+=1


vocab=['-','.'," ",' ',"'",'!',',']+[chr(ord('a')+i) for i in range(26)]+[chr(ord('A')+i) for i in range(26)]+[chr(ord('0')+i) for i in range(10)]
vocab_size=cnt # uppercase and lowercase English characters and digits(26+26+10=62)


batch_size=16
sequence_len=28
RNN_input_dim=7168
RNN_hidden_dim=256
RNN_layer=2
RNN_type='LSTM'
RNN_dropout=0
use_VGG_extractor=False
learning_rate=(4e-3)*(0.8**0)
num_train = 2000

# print('Write your input ')
# x = input()
# print(x)
