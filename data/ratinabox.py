"""
RatInABox (Rat in a Box) Dataset for Conditional Independence Testing.

Adapted from the kernel-ci-testing repository:
https://github.com/romanpogodin/kernel-ci-testing

This dataset simulates neural recordings from a rat navigating a maze.
It uses head direction cells, grid cells, and their combinations to test
conditional independence.

The setup:
- A: Head direction cell population (dependent on actual head direction under H1)
- B: Combined head direction + grid cell population  
- C: Position and head direction (conditioning variable)

Under H0: A uses independent noise, making A ⊥ B | C
Under H1: A uses actual head direction, so A ⊥̸ B | C

Requires pre-generated data files from RatInABox simulations.
See: https://github.com/RatInABox-Lab/RatInABox/
"""

import numpy as np
import torch
import os
from torch.utils.data import Dataset

from .datagen import DatasetOperator, DataGenerator


def load_rat_data_full(data_path, seed, ground_truth, n_cells=100, noise_std=0.1, max_n_points=3000):
    """
    Load the full pre-generated rat neural recording data file.
    
    Args:
        data_path: Path to directory containing .npy data files
        seed: Seed index (determines which data file to load)
        ground_truth: 'H0' for null hypothesis, 'H1' for alternative
        n_cells: Number of cells in the simulation
        noise_std: Noise standard deviation used in generation
        max_n_points: Maximum number of points in the data file
    
    Returns:
        a: Head direction cell responses (max_n_points, n_cells)
        b: Combined head+grid cell responses (max_n_points, n_cells)  
        c: Position and head direction (max_n_points, 4) - [x, y, head_dir]
    """
    filename = f'{n_cells}_cells_{max_n_points}_points_noise_{noise_std}_seed_{seed}.npy'
    path = os.path.join(data_path, filename)
    
    data = np.load(path, allow_pickle=True).item()
    
    if ground_truth == 'H0':
        # Under H0: use independent head direction cells (not dependent on actual head direction)
        a = np.maximum(data['head_dir_ind_rate'], 0)
    else:  # H1
        # Under H1: use actual head direction cells (dependent on head direction)
        a = np.maximum(data['head_dir_rate'], 0)
    
    # B: Combined head direction + grid cells
    b_1 = np.maximum(data['head_dir_rate'], 0)
    b_2 = data['grid_rate']
    b = np.maximum(b_1 + b_2 - 1, 0)
    
    # C: Position (x, y) and head direction
    c_pos = data['pos']  # (max_n_points, 2)
    c_hd = data['head_direction']  # (max_n_points, 2) or (max_n_points,)
    if c_hd.ndim == 1:
        c_hd = c_hd.reshape(-1, 1)
    c = np.concatenate([c_pos, c_hd], axis=1)  # (max_n_points, 4)
    
    return a, b, c


class RatInABoxCIT(DatasetOperator):
    """
    RatInABox Conditional Independence Test dataset.
    
    Tests: A (head direction cells) ⊥ B (head+grid cells) | C (position, head direction)
    """

    def __init__(self, a, b, c):
        """
        Initialize the RatInABoxCIT object from pre-loaded data arrays.

        Args:
            a: Head direction cell responses tensor
            b: Combined head+grid cell responses tensor
            c: Position and head direction tensor
        """
        self.a = a
        self.b = b
        self.c = c
        # No noiseless conditional means for this real data
        self.a_m = self.a
        self.b_m = self.b

    @classmethod
    def from_datasets(cls, datasets):
        """Combine multiple RatInABoxCIT datasets."""
        combined = cls.__new__(cls)
        combined.a = torch.cat([d.a for d in datasets], dim=0)
        combined.b = torch.cat([d.b for d in datasets], dim=0)
        combined.c = torch.cat([d.c for d in datasets], dim=0)
        combined.a_m = combined.a
        combined.b_m = combined.b
        return combined


class RatInABoxCITGen(DataGenerator):
    """
    RatInABox CIT Data Generator.
    
    Generates datasets from pre-computed rat neural simulation data.
    Loads the full dataset for the given data_seed and samples without
    replacement across sequences.
    """

    def __init__(self, type, samples, data_seed, data_path, dim=100, n_cells=100, noise_std=0.1, max_n_points=3000):
        """
        Initialize the RatInABoxCITGen object.

        Args:
            type: 'type1' for H0, 'type2' for H1
            samples: Number of samples per batch
            data_seed: Seed determining which data file to load
            data_path: Path to directory with pre-generated .npy files
            dim: Dimension of neural responses (number of cells)
            n_cells: Number of cells in the simulation
            noise_std: Noise standard deviation used in data generation
            max_n_points: Maximum number of points in the data file
        """
        # Don't call super().__init__ as it has type assertions
        self.type = type
        self.samples = samples
        self.data_seed = data_seed
        self.data_path = data_path
        self.dim = dim
        self.n_cells = n_cells
        self.noise_std = noise_std
        self.max_n_points = max_n_points
        
        # Load the full dataset for this data_seed
        ground_truth = 'H0' if type == 'type1' else 'H1'
        a_np, b_np, c_np = load_rat_data_full(
            data_path=data_path,
            seed=data_seed,
            ground_truth=ground_truth,
            n_cells=n_cells,
            noise_std=noise_std,
            max_n_points=max_n_points
        )
        
        # Store full data as tensors
        self.full_a = torch.tensor(a_np, dtype=torch.float32)
        self.full_b = torch.tensor(b_np, dtype=torch.float32)
        self.full_c = torch.tensor(c_np, dtype=torch.float32)
        
        # Initialize available indices for non-replacement sampling
        self.available_indices = list(range(max_n_points))
        
        # Set seeds
        torch.manual_seed(data_seed)
        np.random.seed(data_seed)
        
        # Shuffle the available indices once based on data_seed
        self.rng = np.random.default_rng(seed=data_seed)
        self.rng.shuffle(self.available_indices)
        self.current_idx = 0  # Pointer to track where we are in the shuffled indices

    def generate(self, seed, samples=None) -> RatInABoxCIT:
        """
        Generate data by sampling without replacement from the loaded dataset.

        Args:
            seed: Not used for sampling (kept for API compatibility), 
                  but can be used for other randomness if needed
            samples: Optional override for number of samples

        Returns:
            Dataset: A RatInABoxCIT dataset
        """
        samples = self.samples if samples is None else samples
        
        # Check if we have enough samples left
        if self.current_idx + samples > self.max_n_points:
            raise ValueError(
                f"Not enough samples left for non-replacement sampling. "
                f"Requested {samples}, but only {self.max_n_points - self.current_idx} remaining. "
                f"Consider using a larger data file or fewer sequences."
            )
        
        # Get the next batch of indices (without replacement)
        batch_indices = self.available_indices[self.current_idx:self.current_idx + samples]
        self.current_idx += samples
        
        # Extract data for these indices
        a = self.full_a[batch_indices]
        b = self.full_b[batch_indices]
        c = self.full_c[batch_indices]
        
        return RatInABoxCIT(a, b, c)

