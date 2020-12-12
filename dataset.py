import torch 
from torch.utils.data import Dataset
import json
import os 

class BagWordsDataset(Dataset):
    "output: image region, one-hot vocabulary vector"
    