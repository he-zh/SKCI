import numpy as np
import torch
from torch.utils.data import Dataset

from .datagen import DatasetOperator, DataGenerator


def get_cit_data(d=20, a=3, n=5000, test='type1', seed=0, u=None, v=None):
    """Generate data for the PCR test.
     Code from https://github.com/shaersh/ecrt/
    :param d: dimension of the data
    :param a: parameter for the type 2 test
    :param n: number of samples
    :param test: type of the test
    :return: X, Y, Z, X_mu, Y_mu_conditional - where X_mu and Y_mu_conditional are conditional means
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    Z_mu = np.zeros((d, 1)).ravel()
    Z_Sigma = np.eye(d)
    Z = np.random.multivariate_normal(Z_mu, Z_Sigma, n) # Z is n x (d)
    if v is None: v = np.random.normal(0, 1, (d, 1))
    X_mu = Z @ v  # E[X|Z]
    X = np.random.normal(X_mu, 1, (n, 1)) # X is n x 1
    if u is None: u = np.random.normal(0, 1, (d, 1))
    # beta = np.ones((d, 1))
    if test == 'type2':
        Y_mu = (Z @ u) ** 2 + a * X
        # E[Y|Z] under type2 depends on X, but we want E[Y|Z] marginally
        # For the conditional mean given only Z: E[Y|Z] = E[(Z@u)^2 + a*X | Z] = (Z@u)^2 + a*E[X|Z] = (Z@u)^2 + a*X_mu
        Y_mu_conditional = (Z @ u) ** 2 + a * X_mu
    elif test == 'type1':
        Y_mu = (Z @ u) ** 2  # E[Y|Z]
        Y_mu_conditional = Y_mu
        # beta[0] = 0
    Y = np.random.normal(Y_mu, 1, (n, 1)) # Y is n x 1
    # X = np.column_stack((X, Z)) # X is n x d
    return X, Y, Z, X_mu, Y_mu_conditional

def sample_X_given_Z(Z, X_mu):
    n, d = Z.shape
    X = np.random.normal(X_mu, 1, (n, 1))
    X = np.column_stack((X, Z))
    return X



class GaussianCIT(DatasetOperator):
    """
    Gaussian Conditional Independence Test (CIT) dataset that extends the DatasetOperator.

    This class is responsible for creating a Gaussian CIT dataset.
    """

    def __init__(self, type, samples, d, seed, u, v):
        """
        Initialize the GaussianCIT object.

        Args:
        - type (str): Specifies the type of dataset.
        - samples (int): Number of samples in the dataset.
        - seed (int): Random seed for reproducibility.
        - u (numpy.ndarray): A parameter for generating CIT data.
        - v (numpy.ndarray): Another parameter for generating CIT data.
        """

        # Retrieve data for Gaussian CIT
        X, Y, Z, X_mu, Y_mu = get_cit_data(d=d, u=u, v=v, n=samples, test=type, seed=seed)
        self.a = torch.tensor(X, dtype=torch.float32)
        self.b = torch.tensor(Y, dtype=torch.float32)
        self.c = torch.tensor(Z, dtype=torch.float32)
        # Conditional means E[a|c] and E[b|c] for oracle mode
        self.a_m = torch.tensor(X_mu, dtype=torch.float32)
        self.b_m = torch.tensor(Y_mu, dtype=torch.float32)

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

class GaussianCITGen(DataGenerator):
    """
    Gaussian CIT Data Generator class that extends the DataGenerator.

    This class is responsible for generating datasets using the GaussianCIT method.
    """

    def __init__(self, type, samples, d, data_seed):
        """
        Initialize the GaussianCITGen object.

        Args:
        - type (str): Specifies the type of dataset.
        - samples (int): Number of samples to generate.
        - d (int): Dimension of the data.
        - data_seed (int): Seed for random number generation.
        """
        super().__init__(type, samples, data_seed)
        self.type, self.samples, self.data_seed = type, samples, data_seed
        self.d = d 
        # Generate random vectors for u and v
        torch.manual_seed(data_seed)
        np.random.seed(data_seed)
        self.v = np.random.normal(0, 1, (self.d, 1))
        self.u = np.random.normal(0, 1, (self.d, 1))

    def generate(self, seed, samples=None) -> Dataset:
        """
        Generate data using the GaussianCIT method.

        Args:
        - seed (int): Seed for random number generation.

        Returns:
        - Dataset: A dataset generated using GaussianCIT.
        """
        # Use a modified seed value based on the provided seed and class's data_seed
        modified_seed = (self.data_seed + 1) * 1000 + seed
        samples = self.samples if samples is None else samples
        return GaussianCIT(self.type, samples, self.d, modified_seed, self.u, self.v)