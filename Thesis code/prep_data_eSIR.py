import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class eSIR_Dataset(Dataset):
    def __init__(self, sequence):
        super(eSIR_Dataset).__init__()
        self.sequence = sequence
    def __len__(self):
        return self.sequence.shape[0]
    def __getitem__(self, idx):
        sequence = np.copy(self.sequence[idx, :])
        
                  
        sequence = sequence.astype(np.float32)

        
        return sequence, idx
