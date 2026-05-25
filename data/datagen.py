from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch
import numpy as np

class DataGenerator(ABC):
    def __init__(self, type, samples, data_seed):
        assert type in ["type2", "type1"]
        assert samples > 0
        torch.manual_seed(data_seed)
        np.random.seed(data_seed)
    @abstractmethod
    def generate(self)->Dataset:
        pass

class DatasetOperator(Dataset):

    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        a, b, c = self.a[idx], self.b[idx], self.c[idx]
        return a, b, c
