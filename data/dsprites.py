import numpy as np
import torch
from torch.utils.data import Dataset

from .datagen import DatasetOperator, DataGenerator


def get_tight_bbox(img: np.ndarray):
    """Finds the exact minimum bounding box of the non-zero pixels."""
    y_idx, x_idx = np.where(img > 0)
    if len(y_idx) == 0:
        return 0, 0, 0, 0
    return y_idx.min(), y_idx.max(), x_idx.min(), x_idx.max()


def process_images_to_cit(
    B: np.ndarray,
    A: np.ndarray,
    test: str,
    seed: int,
    crop: int = 28,
    noise_std: float = 0.0,
    b_type: str = "image",
):
    """
    Process pre-loaded images to generate C (crops) with noise.
    
    Args:
        B: Array of images (N, 64, 64) as float32
        A: Array of shape labels (N,) as int64
        test: 'type1' for CI (centered crop) or 'type2' for non-CI (shifted crop)
        seed: Random seed for noise and shifts
        crop: Size of the crop canvas
        noise_std: Standard deviation of Gaussian noise to add
        b_type: 'image' for whole 64x64 image, 'bbox' for bounding box (xmin, ymin, h, w)
    
    Returns:
        A, B, C, bboxes - processed arrays with noise applied
    """
    rng = np.random.default_rng(seed)
    random_indices = rng.permutation(len(A))
    A = A[random_indices]
    B = B[random_indices]
    B_original = B.copy()  # Keep original for bbox detection without noise
    # Add Gaussian noise to B
    if noise_std > 0:
        noise = rng.normal(0, noise_std, B.shape).astype(np.float32)
        B = np.clip(B + noise, 0, 1)

    C_list = []
    B_bbox_list = []
    bboxes = [] 
    pad_size = 64

    for i in range(len(A)):
        ymin, ymax, xmin, xmax = get_tight_bbox(B_original[i])
        h = ymax - ymin + 1
        w = xmax - xmin + 1
        
        # Store bbox coordinates for b_type='bbox'
        B_bbox_list.append([xmin, ymin, h, w])

        if test == "type1":
            dx, dy = 0, 0
            y0 = ymin + pad_size
            y1 = ymax + 1 + pad_size
            x0 = xmin + pad_size
            x1 = xmax + 1 + pad_size
        else:
            shift_x = rng.integers(int(w * 0.3), int(w * 0.6))
            shift_y = rng.integers(int(h * 0.3), int(h * 0.6))
            dx = int(rng.choice([-1, 1]) * shift_x)
            dy = int(rng.choice([-1, 1]) * shift_y)
            h, w = (h + w) // 2, (h + w) // 2
            y0 = ymin + pad_size + dy
            y1 = ymin + h + pad_size + dy
            x0 = xmin + pad_size + dx
            x1 = xmin + w + pad_size + dx
            
        if noise_std > 0:
            pad_noise = rng.normal(0, noise_std, (64 + 2*pad_size, 64 + 2*pad_size)).astype(np.float32)
            pad_noise = np.clip(pad_noise, 0, 1)
            B_padded = pad_noise.copy()
            B_padded[pad_size:pad_size+64, pad_size:pad_size+64] = B[i]
        else:
            B_padded = np.pad(B_original[i], pad_size, mode='constant')

        c_exact = B_padded[y0:y1, x0:x1]

        if noise_std > 0:
            c_canvas = rng.normal(0, noise_std, (crop, crop)).astype(np.float32)
            c_canvas = np.clip(c_canvas, 0, 1)
        else:
            c_canvas = np.zeros((crop, crop), dtype=np.float32)
        start_y = max(0, (crop - h) // 2)
        start_x = max(0, (crop - w) // 2)
        
        ch, cw = min(h, crop), min(w, crop)
        c_canvas[start_y:start_y+ch, start_x:start_x+cw] = c_exact[:ch, :cw]

        C_list.append(c_canvas)
        bboxes.append((xmin + dx, ymin + dy, w, h))

    C = np.stack(C_list, axis=0).astype(np.float32)
    
    if b_type == "bbox":
        B = np.array(B_bbox_list, dtype=np.float32)  # (N, 4): xmin, ymin, h, w
    
    return A, B, C, bboxes


class DspritesCIT(DatasetOperator):
    """
    dSprites Conditional Independence/Dependence dataset.

    This class mirrors the DatasetOperator interface used by SinCIT/GaussianCIT.
    """

    def __init__(self, a, b, c, normalize=False, norm_stats=None):
        """
        Args:
            a, b, c: Data arrays
            normalize: Whether to normalize image data
            norm_stats: Dict with pre-computed stats {'b_mean', 'b_std', 'c_mean', 'c_std'}
                       If None and normalize=True, stats are computed from this data.
                       If provided, uses these stats (for val/test sets).
        """
        self.a = torch.tensor(a, dtype=torch.float32).reshape(-1, 1)  # Shape labels as (N, 1)
        if b.ndim == 3: # If b is images (N, H, W)
            self.b = torch.tensor(b, dtype=torch.float32).reshape(-1, 1, b.shape[-2], b.shape[-1])  # (N, 1, H, W)
        else: # If b is bounding boxes (N, 4)
            self.b = torch.tensor(b, dtype=torch.float32)
        self.c = torch.tensor(c, dtype=torch.float32).reshape(-1, 1, c.shape[-2], c.shape[-1])  # (N, 1, H, W)
        
        # Apply normalization to image data (zero mean, unit variance)
        if normalize and b.ndim == 2: # Only normalize if b is bbox
            if norm_stats is not None:
                self.b_mean, self.b_std = norm_stats['b_mean'], norm_stats['b_std']
                self.b = (self.b - self.b_mean) / self.b_std
            else:
                self.b, self.b_mean, self.b_std = self._normalize(self.b)
        else:
            self.b_mean, self.b_std = None, None
        
        # For dsprites dataset, we don't have noiseless conditional means
        self.a_m = self.a
        self.b_m = self.b
    
    def get_norm_stats(self):
        """Return normalization stats for use with val/test sets."""
        if self.b_mean is None:
            return None
        return {
            'b_mean': self.b_mean,
            'b_std': self.b_std,
        }
    
    @staticmethod
    def _normalize(x):
        """Normalize tensor to zero mean and unit variance."""
        mean = x.mean()
        std = x.std()
        std = std if std > 1e-6 else 1.0  # Avoid division by zero
        return (x - mean) / std, mean, std

    @classmethod
    def from_datasets(cls, datasets):
        """Combine multiple DspritesCIT datasets."""
        combined = cls.__new__(cls)
        combined.a = torch.cat([d.a for d in datasets], dim=0)
        combined.b = torch.cat([d.b for d in datasets], dim=0)
        combined.c = torch.cat([d.c for d in datasets], dim=0)
        combined.a_m = combined.a
        combined.b_m = combined.b
        return combined


class DspritesCITGen(DataGenerator):
    """
    dSprites CIT Data Generator class that extends the DataGenerator.
    
    Loads the full dataset during initialization and samples sequentially
    without replacement across generate() calls.
    """

    def __init__(
        self,
        type,
        samples,
        data_seed,
        crop=28,
        y_band=None,
        x_bands=None,
        data_path=None,
        scale=2,
        noise_std=0.0,
        b_type="image",
        normalize=True,
    ):
        super().__init__(type, samples, data_seed)
        self.type, self.samples, self.data_seed = type, samples, data_seed
        self.crop = crop
        self.y_band = y_band
        self.x_bands = x_bands
        self.data_path = data_path
        self.scale = scale
        self.noise_std = noise_std
        self.b_type = b_type  # 'image' for 64x64 image, 'bbox' for bounding box coords
        self.normalize = normalize  # Whether to normalize image data
        self._norm_stats = None  # Stores normalization stats computed from first batch (training)
        
        # Load the full dataset once during initialization
        # Use a large n_per_shape to get all available data
        self._load_full_dataset()
    
    def _load_full_dataset(self):
        """Load and preprocess the full dataset."""
        rng = np.random.default_rng(self.data_seed)
        
        if self.x_bands is None:
            x_bands = {0: (0, 9), 1: (11, 20), 2: (22, 31)} 
        else:
            x_bands = self.x_bands
        if self.y_band is None:
            y_band = (0, 31)
        else:
            y_band = self.y_band
        
        try:
            dataset = np.load(self.data_path, encoding='latin1', allow_pickle=True)
            all_imgs = dataset['imgs']
            latents_classes = dataset['latents_classes'] 
        except FileNotFoundError:
            raise FileNotFoundError(f"dSprites dataset not found at {self.data_path}")
        
        # Collect all valid indices
        valid_indices = []
        
        for idx in range(len(all_imgs)):
            lat = latents_classes[idx].astype(np.int64) 

            if int(lat[2]) != self.scale:
                continue

            shape_class = int(lat[1]) 
            if shape_class not in (0, 1, 2):
                continue

            posX, posY = int(lat[4]), int(lat[5])
            x_lo, x_hi = x_bands[shape_class]
            y_lo, y_hi = y_band
            if not (x_lo <= posX <= x_hi and y_lo <= posY <= y_hi):
                continue

            valid_indices.append((idx, shape_class))
        
        # Shuffle all indices together
        rng.shuffle(valid_indices)
        
        # Store all data (shuffled)
        imgs = np.stack([all_imgs[i].astype(np.float32) for i, _ in valid_indices], axis=0)
        labels = np.array([sc for _, sc in valid_indices], dtype=np.int64)
        
        self.full_B = imgs
        self.full_A = labels
        self.total_samples = len(labels)
        self.current_idx = 0

    def generate(self, seed, samples=None) -> Dataset:
        """
        Generate data by sampling sequentially without replacement.
        
        Samples randomly across all shape classes.
        """
        samples = self.samples if samples is None else samples
        
        # Check if we have enough samples left
        remaining = self.total_samples - self.current_idx
        if remaining < samples:
            raise ValueError(
                f"Not enough samples left. "
                f"Requested {samples}, but only {remaining} remaining. "
                f"Consider using fewer sequences or resetting the generator."
            )
        
        # Get the next batch
        start = self.current_idx
        end = start + samples
        self.current_idx = end
        
        B = self.full_B[start:end].copy()
        A = self.full_A[start:end].copy()
        
        # Use the shared processing function with seed for noise/shifts
        A, B, C, _ = process_images_to_cit(
            B, A, self.type, seed, self.crop, self.noise_std, self.b_type
        )
        
        # Create dataset with normalization
        # First call (training): compute stats; subsequent calls: reuse stats
        dataset = DspritesCIT(A, B, C, normalize=self.normalize, norm_stats=self._norm_stats)
        
        # Store stats from first normalized dataset (training set)
        if self.normalize and self._norm_stats is None:
            self._norm_stats = dataset.get_norm_stats()
        
        return dataset
    
    def reset(self, reset_norm_stats=False):
        """Reset the sampling pointer to allow reuse of the dataset.
        
        Args:
            reset_norm_stats: If True, also reset normalization stats
                             (use when starting a new training run)
        """
        self.current_idx = 0
        if reset_norm_stats:
            self._norm_stats = None
    
    def compute_norm_stats_from_data(self, data):
        """
        Compute and store normalization stats from given dataset.
        
        Call this with training data to set stats that will be reused
        for all subsequent generate() calls.
        
        Args:
            data: A DspritesCIT dataset to compute stats from
        """
        if not self.normalize:
            return
        
        # Compute stats from the provided data (before it was normalized)
        # Note: data.b and data.c are already normalized if normalize=True
        # So we use the stored mean/std from the dataset
        if hasattr(data, 'b_mean') and data.b_mean is not None:
            self._norm_stats = data.get_norm_stats()
        else:
            # If data wasn't normalized, compute stats now
            b_mean = data.b.mean()
            b_std = data.b.std()
            b_std = b_std if b_std > 1e-6 else 1.0
            c_mean = data.c.mean()
            c_std = data.c.std()
            c_std = c_std if c_std > 1e-6 else 1.0
            self._norm_stats = {
                'b_mean': b_mean,
                'b_std': b_std,
                'c_mean': c_mean,
                'c_std': c_std,
            }
    
    def set_norm_stats(self, norm_stats):
        """Directly set normalization stats (e.g., from another dataset)."""
        self._norm_stats = norm_stats


if __name__ == "__main__":
    # Test the classes
    print("\nTesting DspritesCITGen...")
    gen = DspritesCITGen(type="type2", samples=300, data_seed=1, b_type="bbox",
                         data_path="/scratch0/zhhe/data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
    dataset = gen.generate(seed=0)
    print(f"Total samples per shape: {gen.samples_per_shape}")
    print(f"A shape: {dataset.a.shape}, B shape: {dataset.b.shape}, C shape: {dataset.c.shape}")
