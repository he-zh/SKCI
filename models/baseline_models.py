import logging

import torch
import torch.nn as nn
from omegaconf import ListConfig
from utils.criterions import LeaveOneOutKRR, SquareLossKRR
from utils.matrix_processing import add_diag

class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) with ReLU activation function and optional layer normalization and dropout.
    """
    def __init__(self, input_size, hidden_layer_size, output_size, layer_norm=True, drop_out=True, drop_out_p=0.3, bias=True):
        super(MLP, self).__init__()
        layers = []
        in_features = input_size

        if isinstance(hidden_layer_size, (list, ListConfig)):
            for out_features in hidden_layer_size:
                layers.append(nn.Linear(in_features, out_features, bias=bias))
                if layer_norm:
                    layers.append(nn.LayerNorm(out_features))
                layers.append(nn.ReLU())
                if drop_out:
                    layers.append(nn.Dropout(drop_out_p))
                in_features = out_features

        layers.append(nn.Linear(in_features, output_size, bias=bias))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1)
        return self.model(x)


class CNN(nn.Module):
    """
    A convolutional neural network for image data with optional layer normalization and dropout.
    Suitable for image inputs like 28x28 grayscale or RGB images.
    """
    def __init__(self, input_channels=1, input_dim=28, hidden_channels=[32, 64],
                 fc_hidden_size=[128], output_size=64, layer_norm=True,
                 drop_out=True, drop_out_p=0.3, bias=True):
        super(CNN, self).__init__()
        self.input_channels = input_channels
        self.input_dim = input_dim

        # Build convolutional layers
        conv_layers = []
        in_channels = input_channels
        current_dim = input_dim

        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]

        for out_channels in hidden_channels:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias))
            if layer_norm:
                conv_layers.append(nn.GroupNorm(1, out_channels))  # LayerNorm equivalent for CNNs
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(2, 2))  # Halves spatial dimensions
            if drop_out:
                conv_layers.append(nn.Dropout2d(drop_out_p))
            in_channels = out_channels
            current_dim = current_dim // 2

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate flattened dimension after conv layers
        flatten_dim = in_channels * current_dim * current_dim

        # Build fully connected layers
        fc_layers = [nn.Flatten()]
        prev_size = flatten_dim

        if isinstance(fc_hidden_size, int):
            fc_hidden_size = [fc_hidden_size]

        for h_size in fc_hidden_size:
            fc_layers.append(nn.Linear(prev_size, h_size, bias=bias))
            if layer_norm:
                fc_layers.append(nn.LayerNorm(h_size))
            fc_layers.append(nn.ReLU())
            if drop_out:
                fc_layers.append(nn.Dropout(drop_out_p))
            prev_size = h_size

        fc_layers.append(nn.Linear(prev_size, output_size, bias=bias))
        self.fc_layers = nn.Sequential(*fc_layers)

    def _reshape_to_image(self, x):
        """Reshape flattened input to image format (N, C, H, W)."""
        if x.dim() == 2:
            # Flattened input: (N, C*H*W) -> (N, C, H, W)
            batch_size = x.shape[0]
            return x.view(batch_size, self.input_channels, self.input_dim, self.input_dim)
        elif x.dim() == 3:
            # (N, H, W) -> (N, 1, H, W)
            return x.unsqueeze(1)
        elif x.dim() == 4:
            # Already in image format (N, C, H, W)
            return x
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

    def forward(self, x):
        x = self._reshape_to_image(x)
        x = self.conv_layers(x)
        return self.fc_layers(x)


class CNNDecoder(nn.Module):
    """
    A convolutional decoder network that takes a latent vector and produces an image.
    Uses transposed convolutions for upsampling.
    """
    def __init__(self, latent_dim=128, output_channels=1, output_dim=28,
                 hidden_channels=[64, 32], layer_norm=True,
                 drop_out=True, drop_out_p=0.3, bias=True):
        super(CNNDecoder, self).__init__()
        self.output_channels = output_channels
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        # Calculate the initial spatial size after FC layers
        # We'll start from a small spatial size and upsample
        num_upsample = len(hidden_channels)
        self.init_dim = output_dim // (2 ** num_upsample)
        if self.init_dim < 1:
            self.init_dim = 1

        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]

        # First hidden channel is used for initial projection
        init_channels = hidden_channels[0] if hidden_channels else 64

        # FC layer to project latent to spatial feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, init_channels * self.init_dim * self.init_dim, bias=bias),
            nn.LayerNorm(init_channels * self.init_dim * self.init_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(drop_out_p) if drop_out else nn.Identity()
        )
        self.init_channels = init_channels

        # Build transposed convolutional layers for upsampling
        deconv_layers = []
        in_channels = init_channels

        for i, out_channels in enumerate(hidden_channels[1:] + [output_channels]):
            is_last = (i == len(hidden_channels) - 1)
            deconv_layers.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=bias)
            )
            if not is_last:
                if layer_norm:
                    deconv_layers.append(nn.GroupNorm(1, out_channels))
                deconv_layers.append(nn.ReLU())
                if drop_out:
                    deconv_layers.append(nn.Dropout2d(drop_out_p))
            in_channels = out_channels

        self.deconv_layers = nn.Sequential(*deconv_layers)

    def forward(self, z):
        """
        Args:
            z: Latent vector of shape (batch_size, latent_dim)

        Returns:
            Image of shape (batch_size, output_channels, output_dim, output_dim)
        """
        if z.dim() > 2:
            z = torch.flatten(z, start_dim=1)

        x = self.fc(z)
        x = x.view(-1, self.init_channels, self.init_dim, self.init_dim)
        x = self.deconv_layers(x)

        # Ensure output is exactly the target size
        if x.shape[-1] != self.output_dim or x.shape[-2] != self.output_dim:
            x = nn.functional.interpolate(x, size=(self.output_dim, self.output_dim), mode='bilinear', align_corners=False)

        return x


class ImageRegressor(nn.Module):
    """
    Regressor for high-dimensional image outputs.
    Takes (a, c) as input where c can be an image (partial image), and outputs b (full image).

    Architecture:
    - If c is an image: CNN encoder extracts features from c
    - Concatenate encoded c with a
    - Decode to produce output image b

    Args:
        a_dim: Dimension of input a (target/label, typically low-dimensional)
        c_input_type: Type of input c ('image' or 'vector')
        c_dim: Dimension of c if vector, or image size if image (assumes square)
        c_channels: Number of channels in c if image
        b_dim: Output image size (assumes square)
        b_channels: Number of output channels
        encoder_hidden_channels: Hidden channels for CNN encoder
        encoder_fc_hidden_size: FC hidden sizes for encoder
        latent_dim: Dimension of latent space (encoder output)
        decoder_hidden_channels: Hidden channels for CNN decoder
        layer_norm: Whether to use layer normalization
        drop_out: Whether to use dropout
        drop_out_p: Dropout probability
    """
    def __init__(self, a_dim=1, c_input_type='image', c_dim=28, c_channels=1,
                 b_dim=28, b_channels=1,
                 encoder_hidden_channels=[32, 64], encoder_fc_hidden_size=[128],
                 latent_dim=64, decoder_hidden_channels=[64, 32],
                 layer_norm=True, drop_out=True, drop_out_p=0.3):
        super(ImageRegressor, self).__init__()
        self.a_dim = a_dim
        self.c_input_type = c_input_type.lower()
        self.c_dim = c_dim
        self.c_channels = c_channels
        self.b_dim = b_dim
        self.b_channels = b_channels
        self.latent_dim = latent_dim

        # Encoder for c
        if self.c_input_type == 'image':
            self.c_encoder = CNN(
                input_channels=c_channels,
                input_dim=c_dim,
                hidden_channels=encoder_hidden_channels,
                fc_hidden_size=encoder_fc_hidden_size,
                output_size=latent_dim,
                layer_norm=layer_norm,
                drop_out=drop_out,
                drop_out_p=drop_out_p
            )
            c_encoded_dim = latent_dim
        else:  # vector
            self.c_encoder = MLP(
                input_size=c_dim,
                hidden_layer_size=encoder_fc_hidden_size,
                output_size=latent_dim,
                layer_norm=layer_norm,
                drop_out=drop_out,
                drop_out_p=drop_out_p
            )
            c_encoded_dim = latent_dim

        # Fusion layer: combine encoded c with a
        fusion_input_dim = c_encoded_dim + a_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, latent_dim),
            nn.LayerNorm(latent_dim) if layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(drop_out_p) if drop_out else nn.Identity()
        )

        # Decoder to produce output image
        self.decoder = CNNDecoder(
            latent_dim=latent_dim,
            output_channels=b_channels,
            output_dim=b_dim,
            hidden_channels=decoder_hidden_channels,
            layer_norm=layer_norm,
            drop_out=drop_out,
            drop_out_p=drop_out_p
        )

    def forward(self, a, c):
        """
        Forward pass.

        Args:
            a: Target/label tensor of shape (batch_size, a_dim)
            c: Conditioning variable - image (batch_size, c_channels, c_dim, c_dim)
               or flattened (batch_size, c_channels * c_dim * c_dim)
               or vector (batch_size, c_dim)

        Returns:
            b: Output image of shape (batch_size, b_channels, b_dim, b_dim)
        """
        # Ensure a is 2D
        if a.dim() == 1:
            a = a.unsqueeze(-1)

        # Encode c
        c_encoded = self.c_encoder(c)  # (batch_size, latent_dim)

        # Concatenate a and encoded c
        ac = torch.cat([a, c_encoded], dim=-1)

        # Fuse
        fused = self.fusion(ac)  # (batch_size, latent_dim)

        # Decode to image
        b_pred = self.decoder(fused)  # (batch_size, b_channels, b_dim, b_dim)

        return b_pred

    def forward_concat(self, ac):
        """
        Forward pass with pre-concatenated (a, c) input.
        This is for compatibility with the existing ECRT trainer interface.

        Args:
            ac: Concatenated (a, c) tensor. For image c, expects
                (batch_size, a_dim + c_channels * c_dim * c_dim)

        Returns:
            b: Output image of shape (batch_size, b_channels * b_dim * b_dim) (flattened)
        """
        # Split ac into a and c
        a = ac[:, :self.a_dim]

        if self.c_input_type == 'image':
            c_flat = ac[:, self.a_dim:]
            c = c_flat.view(-1, self.c_channels, self.c_dim, self.c_dim)
        else:
            c = ac[:, self.a_dim:]

        # Forward through the model
        b_image = self.forward(a, c)

        # Flatten output for MSE loss compatibility
        return b_image.view(b_image.shape[0], -1)


class mu_X_Given_Z_Estimator(nn.Module):
    """Estimates conditional mean E[X|Z].

    Three modes controlled by ``use_krr`` and ``feature_extractor_type``:

    * ``use_krr=False`` (default): plain MLP/CNN trained end-to-end with MSE.
    * ``use_krr=True, feature_extractor_type=[...]``: MLP/CNN feature extractor + RBF KRR.
      The feature extractor is trained jointly with KRR hyperparameters via the LOO loss.
    * ``use_krr=True, feature_extractor_type=None``: pure RBF KRR on raw input features
      (no feature extractor). Only ``log_gamma`` and ``ridge_lambda`` are trainable.

    Args:
        input_dim: Input dimension (for MLP) or image size (for CNN, assumes square).
        fc_hidden_size: Hidden layer sizes for CNN's or MLP's fully connected layers.
        output_size: Output dimension.
        layer_norm: Whether to use layer normalization.
        drop_out: Whether to use dropout.
        drop_out_p: Dropout probability.
        use_krr: Whether to use kernel ridge regression on top of features.
        gamma_init: Initial value for RBF kernel bandwidth.
        feature_extractor_type: Type of feature extractor ('mlp' or 'cnn').
        input_channels: Number of input channels for CNN (1 for grayscale, 3 for RGB).
        hidden_channels: Hidden channel sizes for CNN convolutional layers.
    """
    def __init__(self, input_dim=19, fc_hidden_size=[128], output_size=1,
                 layer_norm=True, drop_out=True, drop_out_p=0.3,
                 use_krr=False, gamma_init=1.0, feature_extractor_type='mlp',
                 input_channels=1, hidden_channels=[32, 64]):
        super().__init__()
        self.use_krr = use_krr
        self.gamma_init = gamma_init
        self.feature_extractor_type = feature_extractor_type.lower()

        if self.feature_extractor_type in ['mlp', 'cnn']:
            if self.feature_extractor_type == 'cnn':
                self.feature_extractor = CNN(
                    input_channels=input_channels,
                    input_dim=input_dim,
                    hidden_channels=hidden_channels,
                    fc_hidden_size=fc_hidden_size,
                    output_size=output_size,
                    layer_norm=layer_norm,
                    drop_out=drop_out,
                    drop_out_p=drop_out_p
                )
            else:  # default to 'mlp'
                self.feature_extractor = MLP(
                    input_dim, fc_hidden_size, output_size,
                    layer_norm, drop_out, drop_out_p
                )
        else:
            self.feature_extractor = None

        # Trainable KRR hyperparameters (only when use_krr=True)
        if self.use_krr:
            # KRR state
            self.krr_fitted = False
            self._train_features = None
            self._train_targets = None
            self._krr_weights = None
            gamma_dim = output_size if self.feature_extractor else input_dim
            self.log_gamma = nn.Parameter(torch.log(torch.tensor([self.gamma_init] * gamma_dim)), requires_grad=True)  # log bandwidth for each feature dimension
            self.ridge_lambda = nn.Parameter(torch.tensor(1e-4), requires_grad=True)  # log ridge parameter

        if self.feature_extractor is None and not self.use_krr:
            raise ValueError("mu_X_Given_Z_Estimator must have either a feature extractor (hidden_size) or use_krr=True.")
        
    def extract_features(self, z):
        """Extract features: feature extractor output when available, raw input otherwise."""
        if self.feature_extractor is not None:
            return self.feature_extractor(z)
        return z

    def compute_loo_loss(self, z, x):
        """Compute KRR Leave-One-Out loss for joint training.

        The LOO criterion has a closed form for KRR:
            LOO_error_i = (A⁻¹ y)_i / (A⁻¹)_{ii}
        where A = K + ridge_lambda * I.  Gradients flow through the MLP features,
        the RBF bandwidth gamma, and the ridge parameter ridge_lambda.

        Args:
            z: Input conditioning variables (n, input_dim)
            x: Target values (n, target_size)

        Returns:
            Scalar LOO loss (mean squared LOO error)
        """
        features = self.extract_features(z)  # Feature extractor output or raw z
        self._train_features = features.clone()  # Store for later KRR fitting and inference
        self._train_targets = x.clone()

        K_ZZ = self._compute_rbf_kernel(features, features, self.log_gamma)  # (n, n)
        K_XX = x @ x.T  # (n, n)
        loo_loss = LeaveOneOutKRR()
        return loo_loss(K_ZZ, K_XX, self.ridge_lambda)

    def compute_val_loss(self, z, x):
        """Compute validation loss for KRR model (after fitting).

        This is the standard MSE loss of KRR predictions on the validation set.
        It does not use the LOO criterion, since we want to evaluate actual
        predictive performance after fitting.

        Args:
            z: Input conditioning variables (n, input_dim)
            x: Target values (n, target_size)

        Returns:
            Scalar MSE loss of KRR predictions
        """
        if self._train_features is None:
            raise ValueError("KRR model is not fitted yet. Call fit_krr() first.")

        features = self.extract_features(z)  # Feature extractor output or raw z

        K_xX = self._compute_rbf_kernel(features, self._train_features, self.log_gamma)  # (n, n_train)
        K_XX = self._compute_rbf_kernel(self._train_features, self._train_features, self.log_gamma)  # (n_train, n_train)
        K_yy = x @ x.T  # (n, n)
        K_yY = x @ self._train_targets.T  # (n, output_size) @ (output_size, n_train) -> (n, n_train)
        K_YY = self._train_targets @ self._train_targets.T  # (output_size, n_train) @ (n_train, output_size) -> (n_train, n_train)
        mse_loss = SquareLossKRR()
        return mse_loss(K_xX, K_XX, K_yy, K_yY, K_YY, self.ridge_lambda)

    @staticmethod
    def _compute_rbf_kernel(X, Y, log_gamma):
        """Compute RBF kernel matrix K(X, Y) = exp(-gamma * ||x_i - y_j||^2).

        Args:
            X: (n, d) tensor
            Y: (m, d) tensor
            gamma: RBF bandwidth parameter

        Returns:
            K: (n, m) kernel matrix
        """
        inv_length_scale = torch.exp(-log_gamma / 2)
        X_scaled = X * inv_length_scale
        Y_scaled = Y * inv_length_scale
        dists_sq = torch.cdist(X_scaled, Y_scaled, p=2) ** 2
        return torch.exp(-dists_sq / 2)

    def fit_krr(self, z, x):
        """Store KRR solution using the trained gamma and ridge_lambda for inference.

        Call after training to prepare for inference via forward().
        No cross-validation is needed because gamma and ridge_lambda are jointly trained
        with the MLP backbone via the LOO loss.

        Args:
            z: Input conditioning variables (n, input_dim)
            x: Target values (n, output_size)
        """
        self.eval()
        with torch.no_grad():
            features = self.extract_features(z)  # Feature extractor output or raw z
            K = self._compute_rbf_kernel(features, features, self.log_gamma)
            n = K.shape[0]
            self._krr_weights = torch.linalg.solve(
                add_diag(K, n * self.ridge_lambda),
                x
            )
            self._train_features = features.clone()
            self.krr_fitted = True

            # logging.info(f"KRR fitted: gamma={torch.exp(self.log_gamma).item():.4e}, ridge_lambda={self.ridge_lambda.item():.4e}, n_support={n}")

    def reset_krr(self):
        """Reset KRR state. Call before retraining MLP features."""
        self.krr_fitted = False
        self._train_features = None
        self._krr_weights = None

    def forward(self, z):
        if self.use_krr and self.krr_fitted and self._train_features is not None:
            features = self.extract_features(z)  # Feature extractor output or raw z
            K = self._compute_rbf_kernel(features, self._train_features, self.log_gamma)
            return K @ self._krr_weights
        elif self.feature_extractor is not None:
            return self.feature_extractor(z)
        else:
            raise RuntimeError(
                "mu_X_Given_Z_Estimator has no feature extractor (hidden_size=None) and KRR "
                "is not fitted. Call fit_krr() first or provide hidden layers."
            )


class GMMN_Estimator(nn.Module):
    """
    Generative Moment Matching Network for estimating P(X|Z).
    Maps noise eta ~ N(0, I) and conditioning variable z to samples from P(X|Z=z).
    """
    def __init__(self, input_dim=19, hidden_size=128, output_size=1,
                 drop_out=False, drop_out_p=0.3, noise_dim=16, layer_norm=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_size = output_size
        self.noise_dim = noise_dim

        if isinstance(hidden_size, int):
            hidden_sizes = [hidden_size]
        else:
            hidden_sizes = list(hidden_size)

        layers = []
        prev_size = noise_dim + input_dim
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            if layer_norm:
                layers.append(nn.LayerNorm(h_size))
            layers.append(nn.ReLU())
            prev_size = h_size

        layers.append(nn.Linear(prev_size, output_size))
        self.generator = nn.Sequential(*layers)

    def forward(self, eta, z):
        return self.generator(torch.cat([eta, z], dim=-1))

    def sample(self, z, n_samples=1):
        batch_size = z.shape[0]
        device = z.device

        if n_samples == 1:
            eta = torch.randn(batch_size, self.noise_dim, device=device)
            return self.forward(eta, z)
        else:
            return self.sample_multiple(z, n_samples)

    def sample_multiple(self, z, M):
        batch_size = z.shape[0]
        device = z.device
        z_repeated = z.repeat_interleave(M, dim=0)
        eta = torch.randn(batch_size * M, self.noise_dim, device=device)
        samples = self.forward(eta, z_repeated)
        return samples.view(batch_size, M, self.output_size)


class NormalizedGMMN_Estimator(nn.Module):
    """
    Wrapper around GMMN_Estimator that applies z-score normalization.

    This class handles normalization of inputs (Z) and outputs (X) automatically:
    - During training: learns on normalized data (mean=0, std=1)
    - During sampling: normalizes inputs and denormalizes outputs

    Usage:
        model = NormalizedGMMN_Estimator(input_dim=19, output_size=1)
        model.fit_normalization(Z_data, X_data)  # Compute normalization stats

        # Training (use normalized data internally)
        x_fake = model.sample_multiple(z, M=5)

        # Sampling (automatically normalizes/denormalizes)
        x_sample = model.sample(z, n_samples=1)
    """
    def __init__(self, input_dim=19, hidden_size=128, output_size=1,
                 drop_out=False, drop_out_p=0.3, noise_dim=16, layer_norm=True):
        super().__init__()

        # Core GMMN model
        self.gmmn = GMMN_Estimator(
            input_dim=input_dim,
            hidden_size=hidden_size,
            output_size=output_size,
            drop_out=drop_out,
            drop_out_p=drop_out_p,
            noise_dim=noise_dim,
            layer_norm=layer_norm
        )

        # Normalization statistics (will be computed from data)
        self.register_buffer('x_mean', None)
        self.register_buffer('x_std', None)
        self.register_buffer('z_mean', None)
        self.register_buffer('z_std', None)
        self._normalization_fitted = False

        # Expose these for compatibility
        self.input_dim = input_dim
        self.output_size = output_size
        self.noise_dim = noise_dim

    def fit_normalization(self, z_data, x_data, eps=1e-8):
        """
        Compute normalization statistics from data.

        Args:
            z_data: Conditioning variables (n, input_dim)
            x_data: Target variables (n, output_size)
            eps: Small constant to avoid division by zero
        """
        device = next(self.parameters()).device
        z_data = z_data.to(device)
        x_data = x_data.to(device)

        # Compute mean and std
        self.x_mean = x_data.mean(dim=0, keepdim=True)
        self.x_std = x_data.std(dim=0, keepdim=True) + eps
        self.z_mean = z_data.mean(dim=0, keepdim=True)
        self.z_std = z_data.std(dim=0, keepdim=True) + eps

        self._normalization_fitted = True

    def normalize_x(self, x):
        """Normalize X using z-score."""
        if not self._normalization_fitted:
            return x
        return (x - self.x_mean) / self.x_std

    def denormalize_x(self, x_normalized):
        """Denormalize X from z-score."""
        if not self._normalization_fitted:
            return x_normalized
        return x_normalized * self.x_std + self.x_mean

    def normalize_z(self, z):
        """Normalize Z using z-score."""
        if not self._normalization_fitted:
            return z
        return (z - self.z_mean) / self.z_std

    def denormalize_z(self, z_normalized):
        """Denormalize Z from z-score."""
        if not self._normalization_fitted:
            return z_normalized
        return z_normalized * self.z_std + self.z_mean

    def forward(self, eta, z):
        """
        Forward pass through GMMN.
        Assumes z is already normalized if normalization is fitted.
        """
        return self.gmmn.forward(eta, z)

    def sample(self, z, n_samples=1):
        """
        Sample from P(X|Z).
        Automatically normalizes input z and denormalizes output x.

        Args:
            z: Conditioning variables (batch_size, input_dim)
            n_samples: Number of samples per z

        Returns:
            x_samples: Samples in original (denormalized) space
        """
        z_normalized = self.normalize_z(z)
        x_normalized = self.gmmn.sample(z_normalized, n_samples)
        return self.denormalize_x(x_normalized)

    def sample_multiple(self, z, M):
        """
        Sample M samples from P(X|Z) for each z.
        Automatically normalizes input z and denormalizes output x.

        Args:
            z: Conditioning variables (batch_size, input_dim)
            M: Number of samples per z

        Returns:
            x_samples: Samples in original space (batch_size, M, output_size)
        """
        z_normalized = self.normalize_z(z)
        x_normalized = self.gmmn.sample_multiple(z_normalized, M)
        return self.denormalize_x(x_normalized)

    def sample_normalized(self, z_normalized, n_samples=1):
        """
        Sample from normalized space (for training).
        Input and output are both in normalized space.

        Args:
            z_normalized: Pre-normalized conditioning variables
            n_samples: Number of samples

        Returns:
            x_normalized: Samples in normalized space
        """
        return self.gmmn.sample(z_normalized, n_samples)

    def sample_multiple_normalized(self, z_normalized, M):
        """
        Sample M samples in normalized space (for training).

        Args:
            z_normalized: Pre-normalized conditioning variables
            M: Number of samples per z

        Returns:
            x_normalized: Samples in normalized space (batch_size, M, output_size)
        """
        return self.gmmn.sample_multiple(z_normalized, M)


class MMDEMLP(MLP):
    """
    MMDEMLP is an extension of the base MLP (Multi-Layer Perceptron) for DAVT.

    This class implements a custom forward operation that compares two inputs
    and computes log(1 + tanh(g(x) - g(y))) for the DAVT e-value calculation.
    """

    def __init__(self, input_size, hidden_layer_size, output_size, return_logits=False, layer_norm=True,
                 drop_out=True, drop_out_p=0.3, bias=True, flatten=True):
        """
        Initialize the MMDEMLP model.

        Args:
        - input_size (int): Size of input layer
        - hidden_layer_size (int or list): Size(s) of hidden layer(s)
        - output_size (int): Size of output layer
        - return_logits (bool): Whether to return logits instead of applying log(1 + tanh(...)) transformation
        - layer_norm (bool): Whether to apply layer normalization
        - drop_out (bool): Whether to apply dropout
        - drop_out_p (float): Dropout probability
        - bias (bool): Whether to use bias in linear layers
        - flatten (bool): Whether to flatten input tensors
        """
        super(MMDEMLP, self).__init__(
            input_size, hidden_layer_size, output_size,
            layer_norm, drop_out, drop_out_p, bias
        )
        self.sigma = torch.nn.Tanh()
        self.flatten = flatten
        self.return_logits = return_logits

    def forward(self, x, y) -> torch.Tensor:
        """
        Forward pass for the MMDEMLP model.

        Args:
        - x (torch.Tensor): First input tensor (z = [a, b, c])
        - y (torch.Tensor): Second input tensor (tau_z = [tilde_a, b, c])

        Returns:
        - torch.Tensor: log(1 + tanh(g(x) - g(y)))
        """
        if len(x.shape) > 2 or len(y.shape) > 2:
            if self.flatten:
                x = torch.flatten(x, start_dim=1)
                y = torch.flatten(y, start_dim=1)
                g_x = self.model(x)
                g_y = self.model(y)
            else:
                num_samples = x.shape[-1]
                g_x, g_y = 0, 0
                for i in range(num_samples):
                    g_x += self.model(torch.flatten(x[..., i], start_dim=1)) / num_samples
                    g_y += self.model(torch.flatten(y[..., i], start_dim=1)) / num_samples
        else:
            g_x = self.model(x)
            g_y = self.model(y)

        if self.return_logits:
            return g_x, g_y
        else:
            output = torch.log(1 + self.sigma(g_x - g_y))
            return output


class MMDECNN(nn.Module):
    """
    MMDECNN for DAVT with mixed inputs: scalar a, image b, and image c.

    This handles the case where inputs are concatenated as [a, b_flat, c_flat]:
    - a: 1D vector (e.g., dimension 1)
    - b: image (e.g., 28x28, flattened to 784)
    - c: image (e.g., 64x64, flattened to 4096)

    Architecture:
    - Separate CNN encoders for b and c (different image sizes)
    - Concatenate a with encoded features from b and c
    - FC layers to produce final output for DAVT e-value calculation
    """

    def __init__(self, a_dim=1, b_dim=28, c_dim=64,
                 b_channels=1, c_channels=1,
                 b_hidden_channels=[32, 64], c_hidden_channels=[32, 64, 128],
                 b_latent_dim=64, c_latent_dim=128,
                 fc_hidden_size=[128], output_size=1,
                 return_logits=False,
                 layer_norm=True, drop_out=True, drop_out_p=0.3, bias=True):
        """
        Initialize the MMDECNN model for mixed inputs.

        Args:
        - a_dim (int): Dimension of scalar input a
        - b_dim (int): Image dimension for b (assumes square, e.g., 28 for 28x28)
        - c_dim (int): Image dimension for c (assumes square, e.g., 64 for 64x64)
        - b_channels (int): Number of channels for image b (1 for grayscale)
        - c_channels (int): Number of channels for image c (1 for grayscale)
        - b_hidden_channels (list): Hidden channels for b's CNN encoder
        - c_hidden_channels (list): Hidden channels for c's CNN encoder
        - b_latent_dim (int): Output dimension of b's CNN encoder
        - c_latent_dim (int): Output dimension of c's CNN encoder
        - fc_hidden_size (list): Size(s) of final fully connected hidden layer(s)
        - output_size (int): Size of output layer
        - layer_norm (bool): Whether to apply layer normalization
        - drop_out (bool): Whether to apply dropout
        - drop_out_p (float): Dropout probability
        - bias (bool): Whether to use bias in layers
        - return_logits (bool): Whether to return logits instead of applying log(1 + tanh(...)) transformation
        """
        super(MMDECNN, self).__init__()

        self.a_dim = a_dim
        self.b_dim = b_dim
        self.c_dim = c_dim
        self.b_channels = b_channels
        self.c_channels = c_channels
        self.b_flat_dim = b_channels * b_dim * b_dim
        self.c_flat_dim = c_channels * c_dim * c_dim
        self.return_logits = return_logits

        # CNN encoder for b (e.g., 28x28 image)
        self.b_encoder = CNN(
            input_channels=b_channels,
            input_dim=b_dim,
            hidden_channels=b_hidden_channels,
            fc_hidden_size=[],  # No FC layers in encoder, just conv
            output_size=b_latent_dim,
            layer_norm=layer_norm,
            drop_out=drop_out,
            drop_out_p=drop_out_p,
            bias=bias
        )

        # CNN encoder for c (e.g., 64x64 image)
        self.c_encoder = CNN(
            input_channels=c_channels,
            input_dim=c_dim,
            hidden_channels=c_hidden_channels,
            fc_hidden_size=[],  # No FC layers in encoder, just conv
            output_size=c_latent_dim,
            layer_norm=layer_norm,
            drop_out=drop_out,
            drop_out_p=drop_out_p,
            bias=bias
        )

        # Final FC layers: takes a + b_features + c_features
        fusion_dim = a_dim + b_latent_dim + c_latent_dim
        fc_layers = []
        prev_size = fusion_dim

        if isinstance(fc_hidden_size, int):
            fc_hidden_size = [fc_hidden_size]

        for h_size in fc_hidden_size:
            fc_layers.append(nn.Linear(prev_size, h_size, bias=bias))
            if layer_norm:
                fc_layers.append(nn.LayerNorm(h_size))
            fc_layers.append(nn.ReLU())
            if drop_out:
                fc_layers.append(nn.Dropout(drop_out_p))
            prev_size = h_size

        fc_layers.append(nn.Linear(prev_size, output_size, bias=bias))
        self.fc_layers = nn.Sequential(*fc_layers)

        self.sigma = torch.nn.Tanh()

    def _split_input(self, z):
        """
        Split concatenated input [a, b_flat, c_flat] into components.

        Args:
        - z (torch.Tensor): Concatenated input of shape (batch_size, a_dim + b_flat_dim + c_flat_dim)

        Returns:
        - a: (batch_size, a_dim)
        - b: (batch_size, b_channels, b_dim, b_dim)
        - c: (batch_size, c_channels, c_dim, c_dim)
        """
        a = z[:, :self.a_dim]
        b_flat = z[:, self.a_dim:self.a_dim + self.b_flat_dim]
        c_flat = z[:, self.a_dim + self.b_flat_dim:]

        b = b_flat.view(-1, self.b_channels, self.b_dim, self.b_dim)
        c = c_flat.view(-1, self.c_channels, self.c_dim, self.c_dim)

        return a, b, c

    def _extract_features(self, z):
        """
        Extract features from concatenated input [a, b, c].

        Args:
        - z (torch.Tensor): Concatenated input

        Returns:
        - torch.Tensor: Fused features
        """
        a, b, c = self._split_input(z)

        # Encode images
        b_features = self.b_encoder(b)  # (batch_size, b_latent_dim)
        c_features = self.c_encoder(c)  # (batch_size, c_latent_dim)

        # Concatenate a with image features
        fused = torch.cat([a, b_features, c_features], dim=-1)

        # Pass through FC layers
        return self.fc_layers(fused)

    def forward(self, x, y) -> torch.Tensor:
        """
        Forward pass for the MMDECNN model.

        Args:
        - x (torch.Tensor): First input tensor (z = [a, b, c] concatenated)
        - y (torch.Tensor): Second input tensor (tau_z = [tilde_a, b, c] concatenated)

        Returns:
        - torch.Tensor: log(1 + tanh(g(x) - g(y)))
        """
        # Flatten if needed (for batched inputs)
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        if y.dim() > 2:
            y = torch.flatten(y, start_dim=1)

        g_x = self._extract_features(x)
        g_y = self._extract_features(y)
        if self.return_logits:
            return g_x, g_y
        else:
            output = torch.log(1 + self.sigma(g_x - g_y))
            return output
