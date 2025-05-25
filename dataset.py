import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TokenDataset(Dataset):

    def __init__(self, numpy_path, context_length, device=None, dtype=np.uint16):
        self.numpy = np.memmap(numpy_path, dtype=dtype, mode='r')
        self.token_count = len(self.numpy)
        self.context_length = context_length
        self.device = device

    def __len__(self):
        return self.token_count - self.context_length - 1

    def __getitem__(self, idx):
        input_seq = torch.from_numpy(self.numpy[idx:idx + self.context_length].astype(int)).to(self.device)
        target_seq = torch.from_numpy(self.numpy[idx + 1:idx + self.context_length + 1].astype(int)).to(self.device)
        return input_seq, target_seq