import numpy as np
import torch
from torch.utils.data import Dataset

from .datagen import DatasetOperator, DataGenerator


def get_cit_data(d=20, n=100, test='type1', seed=0, beta=1.0, alpha=0.1,
                 ca_dim_idx=0, cb_dim_idx=0, cr_dim_idx=0):
    """Generate data for the Sinusoidal CIT.
    :param d: dimension of the data
    :param n: number of samples
    :param test: type of the test
    :return: X, Y data
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    # c = np.random.RandomState(seed=seed*(n+1)).normal(0, 1, size=(n, d))
    c = np.random.normal(0, 1, size=(n, d))
    f = np.cos
    g = np.exp
    a_m = f(c[:, ca_dim_idx:ca_dim_idx+1])
    b_m = g(c[:, cb_dim_idx:cb_dim_idx+1])

    if test == 'type2':
        r = np.sin(beta * c[:, cr_dim_idx])
        a_r = np.zeros((n, 1))
        b_r = np.zeros((n, 1))
        for i in range(n):
            cov_matrix = [[1, r[i]], [r[i], 1]]
            # a_r[i, 0], b_r[i, 0] = np.random.RandomState(seed=seed*(n+1)+1+i).multivariate_normal([0, 0], cov_matrix)
            a_r[i, 0], b_r[i, 0] = np.random.multivariate_normal([0, 0], cov_matrix)
    elif test == 'type1':
        # a_r = np.random.RandomState(seed=seed*(n+1)+1).normal(0, 1, size=(n, 1))
        # b_r = np.random.RandomState(seed=seed*(n+1)+2).normal(0, 1, size=(n, 1))
        a_r = np.random.normal(0, 1, size=(n, 1))
        b_r = np.random.normal(0, 1, size=(n, 1))
    else:
        raise NotImplementedError(f'{test} has to be type1 or type2')
    a = a_m + alpha * a_r
    b = b_m + alpha * b_r

    return a, b, c, a_m, b_m



class SinCIT(DatasetOperator):
    """
    Sinusoidal Conditional Independence Test (CIT) dataset that extends the DatasetOperator.

    This class is responsible for creating a Sinusoidal CIT dataset.
    """

    def __init__(self, type, samples, d, seed, beta=1.0, alpha=0.1,
                 ca_dim_idx=0, cb_dim_idx=0, cr_dim_idx=0):
        """
        Initialize the SinCIT object.

        Args:
        - type (str): Specifies the type of dataset.
        - samples (int): Number of samples in the dataset.
        - seed (int): Random seed for reproducibility.
        - u (numpy.ndarray): A parameter for generating CIT data.
        - v (numpy.ndarray): Another parameter for generating CIT data.
        """

        # Retrieve data for Gaussian CIT
        a, b, c, a_m, b_m = get_cit_data( n=samples, d=d, test=type, seed=seed, beta=beta, alpha=alpha,
                                         ca_dim_idx=ca_dim_idx, cb_dim_idx=cb_dim_idx, cr_dim_idx=cr_dim_idx)
        self.a = torch.tensor(a, dtype=torch.float32)
        self.b = torch.tensor(b, dtype=torch.float32)
        self.c = torch.tensor(c, dtype=torch.float32)
        self.a_m = torch.tensor(a_m, dtype=torch.float32)
        self.b_m = torch.tensor(b_m, dtype=torch.float32)

    @classmethod
    def from_datasets(cls, datasets):
        """
        将多个 GaussianCIT dataset 合并成一个新的 dataset。
        datasets: list of GaussianCIT 实例
        """
        combined = cls.__new__(cls)  # 不调用 __init__
        combined.a = torch.cat([d.a for d in datasets], dim=0)
        combined.b = torch.cat([d.b for d in datasets], dim=0)
        combined.c = torch.cat([d.c for d in datasets], dim=0)
        return combined

class SinCITGen(DataGenerator):
    """
    Sinusoidal CIT Data Generator class that extends the DataGenerator.

    This class is responsible for generating datasets using the Sinusoidal CIT method.
    """

    def __init__(self, type, samples, d, data_seed, beta=1.0, alpha=0.1,
                 ca_dim_idx=0, cb_dim_idx=0, cr_dim_idx=0):
        """
        Initialize the SinCITGen object.

        Args:
        - type (str): Specifies the type of dataset.
        - samples (int): Number of samples to generate.
        - d (int): Dimension of the data.
        - data_seed (int): Seed for random number generation.
        - beta (float): Parameter for generating CIT data.
        - alpha (float): Parameter for generating CIT data.
        - ca_dim_idx (int): Index for the ca coordinate.
        - cb_dim_idx (int): Index for the cb coordinate.
        - cr_dim_idx (int): Index for the cr coordinate.
        """
        super().__init__(type, samples, data_seed)
        self.type, self.samples, self.data_seed = type, samples, data_seed
        self.d = d 
        self.beta = beta
        self.alpha = alpha
        self.ca_dim_idx = ca_dim_idx
        self.cb_dim_idx = cb_dim_idx
        self.cr_dim_idx = cr_dim_idx

    def generate(self, seed, samples=None) -> Dataset:
        """
        Generate data using the GaussianCIT method.

        Args:
        - seed (int): Seed for random number generation.

        Returns:
        - Dataset: A dataset generated using GaussianCIT.
        """
        # Use a modified seed value based on the provided seed and class's data_seed
        modified_seed = (self.data_seed + 1) * 1000 + seed # assme we use at most 1000 different seed for each data_seed
        samples = self.samples if samples is None else samples
        return SinCIT(self.type, samples, self.d, modified_seed, beta=self.beta, alpha=self.alpha,
                      ca_dim_idx=self.ca_dim_idx, cb_dim_idx=self.cb_dim_idx, cr_dim_idx=self.cr_dim_idx)
