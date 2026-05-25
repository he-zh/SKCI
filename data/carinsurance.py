"""
Car Insurance Dataset for Conditional Independence Testing.

Adapted from the kernel-ci-testing repository:
https://github.com/romanpogodin/kernel-ci-testing

The car insurance data tests whether insurance premiums (Y) are conditionally 
independent of minority status (X) given state risk (Z). This is a fairness 
testing scenario.

Data source: https://github.com/felipemaiapolo/cit/tree/main (MIT license)
"""

import numpy as np
import pandas as pd
import torch
import copy
import os
from torch.utils.data import Dataset
import scipy.stats as stats

from .datagen import DatasetOperator, DataGenerator

def data_normalize(data):
    data = stats.zscore(data, ddof=1, axis=0)
    data[np.isnan(data)] = 0.
    return data


def find_nearest(array, value):
    """Find the nearest value in an array."""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_available_states(data_path):
    """
    Get list of available states from the data directory.
    
    Args:
        data_path: Path to the directory containing CSV files
    
    Returns:
        List of state codes (e.g., ['ca', 'il', 'mo', 'tx'])
    """
    states = []
    for f in os.listdir(data_path):
        if f.endswith('-per-zip.csv'):
            state = f.replace('-per-zip.csv', '')
            states.append(state)
    return sorted(states)


def get_companies_for_state(data_path, state):
    """
    Get list of unique insurance companies for a given state.
    
    Args:
        data_path: Path to the directory containing CSV files
        state: State code ('ca', 'il', 'mo', 'tx')
    
    Returns:
        List of company names
    """
    csv_path = os.path.join(data_path, f'{state}-per-zip.csv')
    data = pd.read_csv(csv_path)
    data = data.loc[:, ['state_risk', 'combined_premium', 'minority', 'companies_name']].dropna()
    companies = sorted(data['companies_name'].unique().tolist())
    return companies


def get_company_sample_size(data_path, state, company):
    """
    Get the number of samples available for a specific company in a state.
    
    Args:
        data_path: Path to the directory containing CSV files
        state: State code
        company: Company name
    
    Returns:
        Number of samples for that company
    """
    csv_path = os.path.join(data_path, f'{state}-per-zip.csv')
    data = pd.read_csv(csv_path)
    data = data.loc[:, ['state_risk', 'combined_premium', 'minority', 'companies_name']].dropna()
    company_data = data.loc[data.companies_name == company]
    return len(company_data)


def get_company_by_index(data_path, state, company_idx):
    """
    Get company name by 1-based index for a given state.
    
    Args:
        data_path: Path to the directory containing CSV files
        state: State code ('ca', 'il', 'mo', 'tx')
        company_idx: 1-based index of the company (1, 2, 3, ...)
    
    Returns:
        Company name string
    
    Raises:
        ValueError: If company_idx is out of range
    """
    companies = get_companies_for_state(data_path, state)
    if company_idx < 0 or company_idx >= len(companies):
        raise ValueError(f"company_idx {company_idx} is out of range. "
                        f"State {state} has {len(companies)} companies (0-{len(companies)-1})")
    return companies[company_idx]  # Convert to 0-based index


def get_num_companies(data_path, state):
    """
    Get the total number of companies for a state.
    
    Args:
        data_path: Path to the directory containing CSV files
        state: State code ('ca', 'il', 'mo', 'tx')
    
    Returns:
        Number of companies in that state
    """
    return len(get_companies_for_state(data_path, state))


def load_car_insurance_full(data_path, state='ca', n_vals=20, test='type1', 
                            data_seed=0, company=None):
    """
    Load and process the full car insurance data for a given data_seed.
    
    Args:
        data_path: Path to the directory containing CSV files
        state: State code ('ca', 'il', 'mo', 'tx')
        n_vals: Number of bins for discretizing state_risk
        test: 'type1' for H0 (simulated independence), 'type2' for real data (H1)
        data_seed: Seed for shuffling Y within Z bins (defines the dataset)
        company: Optional company name filter
    
    Returns:
        a: Combined premium (Y) - what we're testing
        b: Minority status (X) - binary
        c: State risk (Z) - conditioning variable
    """
    # Load data
    torch.manual_seed(data_seed)
    np.random.seed(data_seed)
    csv_path = os.path.join(data_path, f'{state}-per-zip.csv')
    data = pd.read_csv(csv_path)
    data = data.loc[:, ['state_risk', 'combined_premium', 'minority', 'companies_name']].dropna()
    # show all company names
    print(f"Available companies in {state}: {data['companies_name'].unique().tolist()}")
    if company is not None:
        data = data.loc[data.companies_name == company]
    
    Z = np.array(data.state_risk).reshape((-1, 1))
    Y = np.array(data.combined_premium).reshape((-1, 1))
    X = (1 * np.array(data.minority)).reshape((-1, 1))  # Binary minority status
    
    if test == 'type1':
        # Simulated H0: shuffle Y within Z bins to break X-Y dependence given Z
        bins = np.linspace(np.min(Z), np.max(Z), n_vals + 2)
        bins = bins[1:-1]
        Y_ci = copy.deepcopy(Y)
        Z_bin = np.array([find_nearest(bins, z) for z in Z.squeeze()]).reshape(Z.shape)
        
        # Use data_seed to define the shuffling (this defines the full dataset)
        for val in np.unique(Z_bin):
            ind = Z_bin == val
            # rng = np.random.RandomState(data_seed)
            ind2 = np.random.choice(np.sum(ind), np.sum(ind), replace=False)
            Y_ci[ind] = Y_ci[ind][ind2]
        
        # Use binned Z for conditioning
        c = Z_bin
        a = Y_ci
        b = X
    else:
        # Real data (type2) - test actual conditional independence
        c = Z
        a = Y
        b = X
    
    return a, b, c


class CarInsuranceCIT(DatasetOperator):
    """
    Car Insurance Conditional Independence Test dataset.
    
    Tests: Y (premium) ⊥ X (minority) | Z (state risk)
    """

    def __init__(self, a, b, c):
        """
        Initialize the CarInsuranceCIT object from pre-loaded data arrays.

        Args:
            a: Premium tensor
            b: Minority status tensor
            c: State risk tensor
        """
        self.a = a
        self.b = b
        self.c = c
        # For car insurance, we don't have noiseless conditional means
        self.a_m = self.a
        self.b_m = self.b

    @classmethod
    def from_datasets(cls, datasets):
        """Combine multiple CarInsuranceCIT datasets."""
        combined = cls.__new__(cls)
        combined.a = torch.cat([d.a for d in datasets], dim=0)
        combined.b = torch.cat([d.b for d in datasets], dim=0)
        combined.c = torch.cat([d.c for d in datasets], dim=0)
        combined.a_m = combined.a
        combined.b_m = combined.b
        return combined


class CarInsuranceCITGen(DataGenerator):
    """
    Car Insurance CIT Data Generator.
    
    Generates datasets for testing conditional independence in car insurance data.
    Loads the full dataset for the given data_seed and samples without
    replacement across sequences.
    """

    def __init__(self, type, samples, data_seed, data_path, state='ca', n_vals=20, 
                 company=None, company_idx=None):
        """
        Initialize the CarInsuranceCITGen object.

        Args:
            type: 'type1' for simulated H0, 'type2' for real data
            samples: Number of samples per batch
            data_seed: Seed for defining the shuffled dataset (Y permuted within Z bins)
            data_path: Path to data directory containing state CSV files
            state: State code ('ca', 'il', 'mo', 'tx')
            n_vals: Number of bins for discretizing state_risk
            company: Optional company name filter (use this OR company_idx)
            company_idx: Optional 1-based company index (1, 2, 3, ...). 
                        Use get_num_companies(data_path, state) to get total count.
        """
        # Don't call super().__init__ as it has type assertions
        self.type = type
        self.samples = samples
        self.data_seed = data_seed
        self.data_path = data_path
        self.state = state
        self.n_vals = n_vals
        
        # Resolve company from company_idx if provided
        if company_idx is not None:
            company = get_company_by_index(data_path, state, company_idx)
        self.company = company
        
        # Load the full dataset for this data_seed
        a_np, b_np, c_np = load_car_insurance_full(
            data_path=data_path,
            state=state,
            n_vals=n_vals,
            test=type,
            data_seed=data_seed,
            company=company
        )


        # Store full data as tensors
        self.full_a = torch.tensor(a_np, dtype=torch.float32)
        self.full_b = torch.tensor(b_np, dtype=torch.float32)
        self.full_c = torch.tensor(c_np, dtype=torch.float32)
        self.max_n_points = len(self.full_a)

        # Initialize available indices for non-replacement sampling
        self.available_indices = list(range(self.max_n_points))
        
        # Set seeds
        torch.manual_seed(data_seed)
        np.random.seed(data_seed)
        
        # Shuffle the available indices once based on data_seed
        self.rng = np.random.default_rng(seed=data_seed)
        self.rng.shuffle(self.available_indices)
        self.current_idx = 0  # Pointer to track where we are in the shuffled indices

    def generate(self, seed, samples=None) -> CarInsuranceCIT:
        """
        Generate data by sampling without replacement from the loaded dataset.

        Args:
            seed: Not used for sampling (kept for API compatibility)
            samples: Optional override for number of samples

        Returns:
            Dataset: A CarInsuranceCIT dataset
        """
        samples = self.samples if samples is None else samples
        
        # Check if we have enough samples left
        if self.current_idx + samples > self.max_n_points:
            raise ValueError(
                f"Not enough samples left for non-replacement sampling. "
                f"Requested {samples}, but only {self.max_n_points - self.current_idx} remaining. "
                f"Consider using fewer sequences or smaller batch sizes."
            )
        
        # Get the next batch of indices (without replacement)
        batch_indices = self.available_indices[self.current_idx:self.current_idx + samples]
        self.current_idx += samples
        
        # Extract data for these indices
        a = self.full_a[batch_indices]
        b = self.full_b[batch_indices]
        c = self.full_c[batch_indices]
        
        return CarInsuranceCIT(a, b, c)
    