import torch
import torch.nn as nn
from utils.matrix_processing import compute_pdist_sq

class Kernel(nn.Module):
    def __init__(self, kernel_type='linear', gamma=1.0, degree=3., coef0=1., is_trainable=False):
        super(Kernel, self).__init__()
        self.kernel_type = kernel_type
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(gamma, dtype=torch.float32)), requires_grad=is_trainable)
        self.log_degree = nn.Parameter(torch.log(torch.tensor(degree, dtype=torch.float32)), requires_grad=is_trainable)
        self.coef0 = nn.Parameter(torch.tensor(coef0, dtype=torch.float32), requires_grad=is_trainable)
        

    def forward(self, X1, X2=None):
        if X2 is None:
            X2 = X1 # remove .clone()

        # Ensure inputs are 2D (required for cdist and matmul)
        if X1.dim() == 1:
            X1 = X1.unsqueeze(1)
        else:
            X1 = X1.view(X1.shape[0], -1)  # flatten if more than 2D
        if X2.dim() == 1:
            X2 = X2.unsqueeze(1)
        else:
            X2 = X2.view(X2.shape[0], -1)  # flatten if more than 2D

        if self.kernel_type == 'linear':
            return torch.matmul(X1, X2.T)
        elif self.kernel_type == 'rbf':
            # exp(-1/(2*gamma) * ||x - y||^2)
            # return torch.exp(-compute_pdist_sq(X1 / torch.exp(self.log_gamma/2), X2 / torch.exp(self.log_gamma/2)) / 2)
            inv_length_scale = torch.exp(-self.log_gamma / 2)
            X1_scaled = X1 * inv_length_scale
            X2_scaled = X2 * inv_length_scale
            dists_sq = torch.cdist(X1_scaled, X2_scaled, p=2) ** 2
            return torch.exp(-dists_sq / 2)
        elif self.kernel_type == 'polynomial':
            return (torch.exp(self.log_gamma) * torch.matmul(X1, X2.T) + self.coef0) ** torch.exp(self.log_degree)
        elif self.kernel_type == 'sigmoid':
            return torch.tanh(torch.exp(self.log_gamma) * torch.matmul(X1, X2.T) + self.coef0)
        elif self.kernel_type == 'kronecker':
            # Kronecker delta kernel: returns 1 if x == y, 0 otherwise
            # Useful for discrete/categorical variables
            return (X1[:, None, :] == X2[None, :, :]).all(dim=-1).float()
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")



class BaseModel(nn.Module):
    """
    Base model class for kernel functions with feature extraction.

    Args:
        kernel_type (str): Type of kernel function to use.
        gamma (float): Parameter for the RBF, polynomial, exponential chi2
        and sigmoid kernels. Interpretation of the default value is left to
        the kernel. Ignored by other kernels.
        is_trainable (bool): Whether to learn the kernel parameters
            from the data.
        degree : Degree of the polynomial kernel. Ignored by other kernels.
        coef0 : Zero coefficient for polynomial and sigmoid kernels. Ignored by other kernels.
        feature_extractor_parameters (dict): Parameters for the feature extractor.

    """
    def __init__(self, kernel_type='linear', gamma=1.0, gamma_dim=1, degree=3., coef0=1., ridge_lambda=1e-4,
                 feature_extractor=None, is_trainable=False, gamma_init_method=None, **kwargs):
        super(BaseModel, self).__init__()
        gamma = gamma if gamma_dim == 1 else [[gamma]*gamma_dim]
        self.ridge_lambda = nn.Parameter(torch.tensor(ridge_lambda, dtype=torch.float32), requires_grad=is_trainable)
        # gamma_init_method: how to initialize RBF kernel bandwidth from data
        # None = use config gamma value, 'variance' = use data variance, 'median' = median heuristic
        self.gamma_init_method = gamma_init_method
        # self.log_ridge_lambda = nn.Parameter(torch.log(torch.tensor(ridge_lambda, dtype=torch.float32)), requires_grad=is_trainable)
        self.kernel = Kernel(kernel_type=kernel_type, gamma=gamma, degree=degree, coef0=coef0, 
                             is_trainable=is_trainable)
        self.feature_extractor = feature_extractor if feature_extractor is not None \
                                    else nn.Identity()
        self.is_trainable = is_trainable
        self._train_feature = None
        self._kernel_matrix = None

    @property
    def train_feature(self):
        if self._train_feature is None:
            raise ValueError("Training features have not been set. Call `set_kernel_matrix` first.")
        return self._train_feature

    @property
    def kernel_matrix(self):
        if self._kernel_matrix is None:
            raise ValueError("Kernel matrix has not been computed. Call `set_kernel_matrix` first.")
        return self._kernel_matrix

    def set_kernel_matrix(self, train_X):
        """
        Sets the training features and kernel matrix.
        For trainable models, this function should be called after training.
        """
        # if train_X.shape[0] > 2000:
        #     indices = torch.randperm(train_X.shape[0])[:2000]
        #     train_X = train_X[indices]
        
        self._train_feature = self.feature_extractor(train_X).detach()
        if self.gamma_init_method is not None and self.kernel.kernel_type == 'rbf' and self.is_trainable == False:
            self.set_gamma_from_data(self._train_feature.clone().detach())
        self._kernel_matrix = self.kernel(self._train_feature).detach()

    def set_gamma_from_data(self, features):
        """
        Set the RBF kernel bandwidth (gamma) based on data statistics.
        Sets gamma per dimension for multidimensional data.
        
        Args:
            data: Input tensor of shape (n, d) to compute statistics from
            method: Method to compute gamma:
                - 'variance': gamma = var(data) per dimension
                - 'median': gamma = median of pairwise distances per dimension
                - 'std': gamma = std(data) per dimension
        """
        if self.kernel.kernel_type != 'rbf':
            return  # Only applicable for RBF kernel
        
        with torch.no_grad():
            
            # Ensure features is 2D
            if features.dim() == 1:
                features = features.unsqueeze(1)
            
            n_dims = features.shape[1]
            
            if self.gamma_init_method == 'variance':
                # Set gamma to the variance of each dimension
                gamma = features.var(dim=0)
                # Avoid zero variance
                gamma = torch.clamp(gamma, min=1e-6)
            elif self.gamma_init_method == 'std':
                # Set gamma to the standard deviation of each dimension
                gamma = features.std(dim=0)
                gamma = torch.clamp(gamma, min=1e-6)
            elif self.gamma_init_method == 'median':
                # Median heuristic per dimension: gamma_d = median of |x_d - y_d|^2
                gamma_list = []
                for d in range(n_dims):
                    col = features[:, d:d+1]  # (n, 1)
                    dists = torch.cdist(col, col, p=2)  # (n, n)
                    # Get upper triangular (excluding diagonal)
                    triu_indices = torch.triu_indices(dists.shape[0], dists.shape[1], offset=1)
                    pairwise_dists = dists[triu_indices[0], triu_indices[1]]
                    median_dist = torch.median(pairwise_dists)
                    gamma_list.append(median_dist ** 2)
                gamma = torch.stack(gamma_list)
                gamma = torch.clamp(gamma, min=1e-6)
            else:
                raise ValueError(f"Unknown method: {self.gamma_init_method}. Use 'variance', 'std', or 'median'.")
            
            # Update log_gamma parameter - ensure shape matches
            self.kernel.log_gamma.data = torch.log(gamma)


    def forward(self, X1, X2=None):
        """
        Compute the kernel matrix for the input data
        """
        features_X1 = self.feature_extractor(X1)
        if X2 is not None:
            # K(X1, X2)
            features_X2 = self.feature_extractor(X2)
        else:
            if self._train_feature is not None:
                # K(X1, train_X)
                features_X2 = self.train_feature
            else:
                raise ValueError("X2 is None and training features are not set. Cannot compute kernel matrix.")

        return self.kernel(features_X1, features_X2)

class LinearModel(BaseModel):
    def __init__(self, ridge_lambda=1e-4, is_trainable=False, **kwargs):
        super(LinearModel, self).__init__(kernel_type='linear', 
                                         ridge_lambda=ridge_lambda, 
                                         feature_extractor=None, 
                                         is_trainable=is_trainable)

class KroneckerModel(BaseModel):
    def __init__(self, ridge_lambda=1e-4, is_trainable=False, **kwargs):
        super(KroneckerModel, self).__init__(kernel_type='kronecker', 
                                         ridge_lambda=ridge_lambda, 
                                         feature_extractor=None, 
                                         is_trainable=is_trainable)
class RBFModel(BaseModel):
    def __init__(self, input_dim, gamma=1.0, ridge_lambda=1e-4, is_trainable=False, gamma_init_method=None, **kwargs):
        super(RBFModel, self).__init__(kernel_type='rbf', gamma=gamma, gamma_dim=input_dim, 
                                      ridge_lambda=ridge_lambda, 
                                      feature_extractor=None, 
                                      is_trainable=is_trainable,
                                      gamma_init_method=gamma_init_method)

class FCModel(BaseModel):
    def __init__(self, kernel_type, input_dim, hidden_dim, gamma=1.0, degree=3., coef0=1., ridge_lambda=1e-4,
                 is_trainable=False, **kwargs):
        feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
        )
        super(FCModel, self).__init__(kernel_type, gamma, hidden_dim, degree, coef0, ridge_lambda, feature_extractor, 
                                      is_trainable)


class MLPModel(BaseModel):
    def __init__(
        self,
        kernel_type, input_dim, hidden_dim, output_dim, gamma=1.0, degree=3., coef0=1.,
        ridge_lambda=1e-4, dropout=0.5, is_trainable=False, **kwargs):
        layers = []
        prev_dim = input_dim
        
        # hidden layers
        for h in hidden_dim:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        
        # output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        # optionally:
        # layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        feature_extractor = nn.Sequential(*layers)
        
        # Freeze feature extractor if not trainable
        if not is_trainable:
            for param in feature_extractor.parameters():
                param.requires_grad = False

        super().__init__(
            kernel_type, gamma, output_dim, degree, coef0,
            ridge_lambda, feature_extractor, is_trainable
        )


class CNNModel(BaseModel):
    """
    CNN model for image data (e.g., 28x28 or similar).
    
    Args:
        kernel_type: Type of kernel function to use after feature extraction.
        input_channels: Number of input channels (1 for grayscale, 3 for RGB).
        input_dim: Size of input image (assumes square, e.g., 28 for 28x28).
        hidden_channels: List of channel sizes for convolutional layers.
        fc_hidden_size: List of hidden sizes for fully connected layers.
        output_dim: Dimension of the output feature vector.
        gamma: RBF kernel bandwidth.
        ridge_lambda: Ridge regularization parameter.
        layer_norm: Whether to apply layer normalization.
        dropout: Dropout rate (0 to disable).
        is_trainable: Whether to train the CNN feature extractor.
    """
    def __init__(
        self,
        kernel_type,
        input_channels=1,
        input_dim=28,
        hidden_channels=[32, 64],
        fc_hidden_size=[128],
        output_dim=64,
        gamma=1.0,
        degree=3.,
        coef0=1.,
        ridge_lambda=1e-4,
        layer_norm=False,
        dropout=0.3,
        is_trainable=False,
        gamma_init_method=None,
        **kwargs
    ):
        # Store image reshape info for forward pass
        self.input_channels = input_channels
        self.input_dim = input_dim
        
        # Ensure hidden_channels and fc_hidden_size are lists
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]
        if isinstance(fc_hidden_size, int):
            fc_hidden_size = [fc_hidden_size]
        
        # Build convolutional layers
        conv_layers = []
        in_channels = input_channels
        current_dim = input_dim
        
        for out_channels in hidden_channels:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if layer_norm:
                conv_layers.append(nn.GroupNorm(1, out_channels))  # LayerNorm equivalent for CNNs
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(2, 2))  # Halves spatial dimensions
            if dropout > 0:
                conv_layers.append(nn.Dropout2d(dropout))
            in_channels = out_channels
            current_dim = current_dim // 2
        
        cnn_layers = nn.Sequential(*conv_layers)
        
        # Calculate flattened size after conv layers
        flatten_dim = in_channels * current_dim * current_dim
        
        # Build fully connected layers
        fc_layers = [nn.Flatten()]
        prev_size = flatten_dim
        
        for h_size in fc_hidden_size:
            fc_layers.append(nn.Linear(prev_size, h_size))
            if layer_norm:
                fc_layers.append(nn.LayerNorm(h_size))
            fc_layers.append(nn.ReLU())
            if dropout > 0:
                fc_layers.append(nn.Dropout(dropout))
            prev_size = h_size
        
        fc_layers.append(nn.Linear(prev_size, output_dim))
        fc_layers_seq = nn.Sequential(*fc_layers)
        
        # Combine CNN and FC
        feature_extractor = nn.Sequential(cnn_layers, fc_layers_seq)
        
        # Freeze feature extractor if not trainable
        if not is_trainable:
            for param in feature_extractor.parameters():
                param.requires_grad = False

        super().__init__(
            kernel_type, gamma, output_dim, degree, coef0,
            ridge_lambda, feature_extractor, is_trainable,
            gamma_init_method=gamma_init_method
        )
    
    def _reshape_to_image(self, X):
        """Reshape flattened input to image format (N, C, H, W)."""
        if X.dim() == 2:
            # Flattened input: (N, C*H*W) -> (N, C, H, W)
            batch_size = X.shape[0]
            return X.view(batch_size, self.input_channels, self.input_dim, self.input_dim)
        elif X.dim() == 3:
            # If input is (N, H, W), reshape to (N, 1, H, W)
            assert X.shape[1] == self.input_dim and X.shape[2] == self.input_dim, \
                f"Expected input shape (N, {self.input_dim}, {self.input_dim}), got {X.shape}"
            batch_size = X.shape[0]
            return X.view(batch_size, self.input_channels, self.input_dim, self.input_dim)
        elif X.dim() == 4:
            # Already in image format
            return X
        else:
            raise ValueError(f"Unexpected input shape: {X.shape}")
    
    def forward(self, X1, X2=None):
        """
        Compute the kernel matrix for the input data.
        Reshapes flattened input to image format before feature extraction.
        """
        X1_img = self._reshape_to_image(X1)
        features_X1 = self.feature_extractor(X1_img)
        
        if X2 is not None:
            X2_img = self._reshape_to_image(X2)
            features_X2 = self.feature_extractor(X2_img)
        else:
            if self._train_feature is not None:
                features_X2 = self.train_feature
            else:
                raise ValueError("X2 is None and training features are not set.")

        return self.kernel(features_X1, features_X2)
    
    def set_kernel_matrix(self, train_X):
        """
        Sets the training features and kernel matrix.
        Reshapes flattened input to image format before feature extraction.
        """
        train_X_img = self._reshape_to_image(train_X)
        self._train_feature = self.feature_extractor(train_X_img).detach()
        if self.gamma_init_method is not None and self.kernel.kernel_type == 'rbf' and not self.is_trainable:
            self.set_gamma_from_data(self._train_feature.clone().detach())
        self._kernel_matrix = self.kernel(self._train_feature).detach()


class ImageAutoencoder(nn.Module):
    """
    Autoencoder for image data (e.g., 64x64 dSprites images).
    
    Can be used to reduce dimensionality of image data before applying
    kernel methods. The encoder outputs latent features, and the decoder
    reconstructs the original image for training.
    
    Args:
        input_channels: Number of input channels (1 for grayscale).
        input_dim: Size of input image (assumes square, e.g., 64 for 64x64).
        latent_dim: Dimension of the latent space.
        hidden_channels: List of channel sizes for encoder conv layers.
    """
    def __init__(
        self,
        input_channels=1,
        input_dim=64,
        latent_dim=32,
        hidden_channels=[32, 64, 128],
    ):
        super().__init__()
        self.input_channels = input_channels
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder
        encoder_layers = []
        in_ch = input_channels
        for out_ch in hidden_channels:
            encoder_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        self.encoder_conv = nn.Sequential(*encoder_layers)
        
        # Calculate size after convolutions
        # Each conv with stride=2 halves the spatial dimension
        self.conv_output_dim = input_dim // (2 ** len(hidden_channels))
        self.flatten_dim = hidden_channels[-1] * self.conv_output_dim * self.conv_output_dim
        
        # Latent space projection
        self.encoder_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, latent_dim),
        )
        
        # Build decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, self.flatten_dim),
            nn.ReLU(inplace=True),
        )
        
        # Transpose convolutions for decoder
        decoder_layers = []
        reversed_channels = list(reversed(hidden_channels))
        for i, (in_ch, out_ch) in enumerate(zip(reversed_channels[:-1], reversed_channels[1:])):
            decoder_layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
        # Final layer to reconstruct original channels
        decoder_layers.extend([
            nn.ConvTranspose2d(reversed_channels[-1], input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Output in [0, 1] range for images
        ])
        self.decoder_conv = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input images to latent representation."""
        x = self._reshape_to_image(x)
        h = self.encoder_conv(x)
        z = self.encoder_fc(h)
        return z
    
    def decode(self, z):
        """Decode latent representation to reconstructed images."""
        h = self.decoder_fc(z)
        h = h.view(-1, self.flatten_dim // (self.conv_output_dim * self.conv_output_dim), 
                   self.conv_output_dim, self.conv_output_dim)
        x_recon = self.decoder_conv(h)
        return x_recon
    
    def forward(self, x):
        """Full forward pass: encode then decode."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def _reshape_to_image(self, X):
        """Reshape input to image format (N, C, H, W)."""
        if X.dim() == 2:
            # Flattened input: (N, C*H*W) -> (N, C, H, W)
            batch_size = X.shape[0]
            return X.view(batch_size, self.input_channels, self.input_dim, self.input_dim)
        elif X.dim() == 3:
            # If input is (N, H, W), reshape to (N, 1, H, W)
            batch_size = X.shape[0]
            return X.view(batch_size, self.input_channels, self.input_dim, self.input_dim)
        elif X.dim() == 4:
            # Already in image format
            return X
        else:
            raise ValueError(f"Unexpected input shape: {X.shape}")


class AutoencoderModel(BaseModel):
    """
    Kernel model that uses a pretrained autoencoder's encoder as feature extractor.
    
    The autoencoder should be trained separately on the image data.
    This model uses the encoder part to extract latent features and then
    applies a kernel on those features.
    
    Optionally supports LDA (Linear Discriminant Analysis) for supervised 
    dimensionality reduction or PCA for unsupervised reduction of the latent space.
    
    Args:
        kernel_type: Type of kernel function to use on latent features.
        input_channels: Number of input image channels.
        input_dim: Width of input image (assumes square).
        latent_dim: Dimension of the autoencoder's latent space.
        hidden_channels: List of channel sizes for encoder conv layers.
        gamma: RBF kernel bandwidth.
        ridge_lambda: Ridge regularization parameter.
        is_trainable: Whether to train the autoencoder jointly.
        pretrained_path: Optional path to pretrained autoencoder weights.
        use_lda: Whether to apply LDA after autoencoder encoding (supervised).
        use_pca: Whether to apply PCA after autoencoder encoding (unsupervised).
        lda_n_components: Number of LDA components (max is n_classes - 1).
        pca_n_components: Number of PCA components.
    """
    def __init__(
        self,
        kernel_type='rbf',
        input_channels=1,
        input_dim=64,
        latent_dim=32,
        hidden_channels=[32, 64, 128],
        gamma=1.0,
        degree=3.,
        coef0=1.,
        ridge_lambda=1e-4,
        is_trainable=False,
        gamma_init_method=None,
        pretrained_path=None,
        use_lda=False,
        use_pca=False,
        lda_n_components=2,
        pca_n_components=2,
        **kwargs
    ):
        # Create autoencoder first (but don't assign to self yet)
        autoencoder = ImageAutoencoder(
            input_channels=input_channels,
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
        )
        
        # Load pretrained weights if provided
        if pretrained_path is not None:
            autoencoder.load_state_dict(torch.load(pretrained_path))
        
        # Create dimensionality reducer if requested (LDA takes priority over PCA)
        if use_lda:
            reducer = LDAReducer(n_components=lda_n_components)
        elif use_pca:
            reducer = PCAReducer(n_components=pca_n_components)
        else:
            reducer = None
        
        # Determine output dimension for kernel
        # If reduction is used, output dim will be n_components after fitting
        # Before fitting, we use latent_dim
        output_dim = latent_dim
        
        # Use encoder (with optional reduction) as feature extractor
        feature_extractor = EncoderReducerWrapper(autoencoder, reducer)
        
        # Freeze autoencoder if not trainable
        if not is_trainable:
            for param in autoencoder.parameters():
                param.requires_grad = False

        # Call parent __init__ first before assigning any nn.Module attributes
        super().__init__(
            kernel_type, gamma, output_dim, degree, coef0,
            ridge_lambda, feature_extractor, is_trainable,
            gamma_init_method=gamma_init_method
        )
        
        # Now we can safely assign nn.Module attributes
        self.autoencoder = autoencoder
        self.reducer = reducer
        self.use_lda = use_lda
        self.use_pca = use_pca
        self.lda_n_components = lda_n_components
        self.pca_n_components = pca_n_components
        
        # Backward compatibility alias
        self.lda_reducer = reducer if use_lda else None
        self.pca_reducer = reducer if use_pca else None
        
        # Store image reshape info
        self.input_channels = input_channels
        self.input_dim = input_dim
        self.latent_dim = latent_dim
    
    def get_autoencoder(self):
        """Return the autoencoder for separate training."""
        return self.autoencoder
    
    def set_autoencoder_trainable(self, trainable):
        """Enable or disable autoencoder training."""
        for param in self.autoencoder.parameters():
            param.requires_grad = trainable
    
    def get_lda_reducer(self):
        """Return the LDA reducer for inspection or manual fitting."""
        return self.lda_reducer
    
    def get_pca_reducer(self):
        """Return the PCA reducer for inspection or manual fitting."""
        return self.pca_reducer
    
    def get_reducer(self):
        """Return the active reducer (LDA or PCA) for inspection."""
        return self.reducer
    
    def fit_lda(self, X, y):
        """
        Fit LDA on labeled image data for supervised dimensionality reduction.
        
        This extracts latent representations from images using the autoencoder,
        then fits LDA on those representations using the provided labels.
        
        Args:
            X: Input images of shape (n_samples, C, H, W) or flattened.
            y: Class labels of shape (n_samples,) or (n_samples, 1).
        
        Returns:
            self
        """
        if not self.use_lda or self.lda_reducer is None:
            raise RuntimeError("LDA is not enabled for this model. Set use_lda=True.")
        
        # Get latent representations from autoencoder (without LDA)
        self.autoencoder.eval()
        with torch.no_grad():
            latents = self.autoencoder.encode(X)
        
        # Fit LDA on latent representations
        self.lda_reducer.fit(latents, y)
        
        return self
    
    def fit_pca(self, X):
        """
        Fit PCA on image data for unsupervised dimensionality reduction.
        
        This extracts latent representations from images using the autoencoder,
        then fits PCA on those representations.
        
        Args:
            X: Input images of shape (n_samples, C, H, W) or flattened.
        
        Returns:
            self
        """
        if not self.use_pca or self.pca_reducer is None:
            raise RuntimeError("PCA is not enabled for this model. Set use_pca=True.")
        
        # Get latent representations from autoencoder (without PCA)
        self.autoencoder.eval()
        with torch.no_grad():
            latents = self.autoencoder.encode(X)
        
        # Fit PCA on latent representations
        self.pca_reducer.fit(latents)
        
        return self
    
    def fit_reducer(self, X, y=None):
        """
        Fit the active reducer (LDA or PCA) on image data.
        
        Args:
            X: Input images of shape (n_samples, C, H, W) or flattened.
            y: Class labels (required for LDA, ignored for PCA).
        
        Returns:
            self
        """
        if self.use_lda:
            return self.fit_lda(X, y)
        elif self.use_pca:
            return self.fit_pca(X)
        else:
            raise RuntimeError("No reducer is enabled. Set use_lda=True or use_pca=True.")
    
    def is_lda_fitted(self):
        """Check if LDA has been fitted."""
        if self.lda_reducer is None:
            return False
        return self.lda_reducer.is_fitted
    
    def is_pca_fitted(self):
        """Check if PCA has been fitted."""
        if self.pca_reducer is None:
            return False
        return self.pca_reducer.is_fitted
    
    def is_reducer_fitted(self):
        """Check if the active reducer (LDA or PCA) has been fitted."""
        if self.reducer is None:
            return False
        return self.reducer.is_fitted


class PCAReducer(nn.Module):
    """
    Principal Component Analysis (PCA) for unsupervised dimensionality reduction.
    
    Implemented in PyTorch for GPU compatibility and integration with neural networks.
    Reduces latent representations to n_components dimensions that maximize variance.
    
    Args:
        n_components: Target dimensionality.
    """
    def __init__(self, n_components=2):
        super().__init__()
        self.n_components = n_components
        self.projection = None  # Will be set after fit
        self.mean = None
        self._fitted = False
    
    def fit(self, X, y=None):
        """
        Fit PCA on data (y is ignored, kept for API consistency with LDA).
        
        Args:
            X: Input features of shape (n_samples, n_features).
            y: Ignored. Present for API consistency.
        
        Returns:
            self
        """
        device = X.device
        dtype = X.dtype
        
        n_samples, n_features = X.shape
        
        # Limit n_components to min of n_samples and n_features
        self.n_components = min(self.n_components, n_samples, n_features)
        
        # Compute mean and center data
        self.mean = X.mean(dim=0)
        X_centered = X - self.mean
        
        # Compute covariance matrix
        cov = (X_centered.T @ X_centered) / (n_samples - 1)
        
        try:
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            
            # Sort by eigenvalue magnitude (descending)
            # eigh returns in ascending order, so reverse
            sorted_idx = torch.argsort(eigenvalues, descending=True)
            
            # Select top n_components eigenvectors
            self.projection = eigenvectors[:, sorted_idx[:self.n_components]]
            self._fitted = True
            
        except Exception as e:
            # Fallback: use identity projection
            print(f"PCA failed ({e}), falling back to identity projection")
            self.projection = torch.eye(n_features, self.n_components, device=device, dtype=dtype)
            self._fitted = True
        
        return self
    
    def transform(self, X):
        """
        Transform data using the fitted PCA projection.
        
        Args:
            X: Input features of shape (n_samples, n_features).
        
        Returns:
            Transformed features of shape (n_samples, n_components).
        """
        if not self._fitted:
            raise RuntimeError("PCA has not been fitted. Call fit() first.")
        
        # Center and project
        X_centered = X - self.mean.to(X.device)
        return X_centered @ self.projection.to(X.device)
    
    def fit_transform(self, X, y=None):
        """Fit PCA and transform data."""
        self.fit(X, y)
        return self.transform(X)
    
    @property
    def is_fitted(self):
        return self._fitted


class LDAReducer(nn.Module):
    """
    Linear Discriminant Analysis (LDA) for supervised dimensionality reduction.
    
    Implemented in PyTorch for GPU compatibility and integration with neural networks.
    Reduces latent representations to n_components dimensions that maximize class separability.
    
    Args:
        n_components: Target dimensionality (max is n_classes - 1).
        reg: Regularization parameter for within-class scatter matrix.
    """
    def __init__(self, n_components=2, reg=1e-6):
        super().__init__()
        self.n_components = n_components
        self.reg = reg
        self.projection = None  # Will be set after fit
        self.mean = None
        self._fitted = False
    
    def fit(self, X, y):
        """
        Fit LDA on labeled data.
        
        Args:
            X: Input features of shape (n_samples, n_features).
            y: Class labels of shape (n_samples,).
        
        Returns:
            self
        """
        device = X.device
        dtype = X.dtype
        
        # Ensure y is 1D
        if y.dim() > 1:
            y = y.squeeze()
        
        n_samples, n_features = X.shape
        classes = torch.unique(y)
        n_classes = len(classes)
        
        # Limit n_components to n_classes - 1
        self.n_components = min(self.n_components, n_classes - 1, n_features)
        
        # Compute overall mean
        self.mean = X.mean(dim=0)
        
        # Compute class means and class priors
        class_means = torch.zeros(n_classes, n_features, device=device, dtype=dtype)
        class_counts = torch.zeros(n_classes, device=device, dtype=dtype)
        
        for i, c in enumerate(classes):
            mask = (y == c)
            class_counts[i] = mask.sum()
            class_means[i] = X[mask].mean(dim=0)
        
        # Compute between-class scatter matrix S_B
        # S_B = sum_c n_c * (mu_c - mu)(mu_c - mu)^T
        mean_diff = class_means - self.mean
        S_B = torch.zeros(n_features, n_features, device=device, dtype=dtype)
        for i in range(n_classes):
            diff = mean_diff[i:i+1].T  # (n_features, 1)
            S_B += class_counts[i] * (diff @ diff.T)
        
        # Compute within-class scatter matrix S_W
        # S_W = sum_c sum_{x in c} (x - mu_c)(x - mu_c)^T
        S_W = torch.zeros(n_features, n_features, device=device, dtype=dtype)
        for i, c in enumerate(classes):
            mask = (y == c)
            X_c = X[mask] - class_means[i]
            S_W += X_c.T @ X_c
        
        # Add regularization to S_W
        S_W += self.reg * torch.eye(n_features, device=device, dtype=dtype)
        
        # Solve generalized eigenvalue problem: S_B @ v = lambda * S_W @ v
        # Equivalent to: inv(S_W) @ S_B @ v = lambda * v
        try:
            S_W_inv = torch.linalg.inv(S_W)
            M = S_W_inv @ S_B
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = torch.linalg.eig(M)
            
            # Take real part (eigenvalues should be real for symmetric matrices)
            eigenvalues = eigenvalues.real
            eigenvectors = eigenvectors.real
            
            # Sort by eigenvalue magnitude (descending)
            sorted_idx = torch.argsort(eigenvalues, descending=True)
            
            # Select top n_components eigenvectors
            self.projection = eigenvectors[:, sorted_idx[:self.n_components]]
            self._fitted = True
            
        except Exception as e:
            # Fallback: use PCA if LDA fails
            print(f"LDA failed ({e}), falling back to identity projection")
            self.projection = torch.eye(n_features, self.n_components, device=device, dtype=dtype)
            self._fitted = True
        
        return self
    
    def transform(self, X):
        """
        Transform data using the fitted LDA projection.
        
        Args:
            X: Input features of shape (n_samples, n_features).
        
        Returns:
            Transformed features of shape (n_samples, n_components).
        """
        if not self._fitted:
            raise RuntimeError("LDA has not been fitted. Call fit() first.")
        
        # Center and project
        X_centered = X - self.mean.to(X.device)
        return X_centered @ self.projection.to(X.device)
    
    def fit_transform(self, X, y):
        """Fit LDA and transform data."""
        self.fit(X, y)
        return self.transform(X)
    
    @property
    def is_fitted(self):
        return self._fitted


class EncoderReducerWrapper(nn.Module):
    """
    Wrapper that applies autoencoder encoding followed by optional dimensionality reduction.
    
    Supports both LDA (supervised) and PCA (unsupervised) reduction methods.
    
    Args:
        autoencoder: The ImageAutoencoder instance.
        reducer: Optional LDAReducer or PCAReducer instance. If None, no reduction is applied.
    """
    def __init__(self, autoencoder, reducer=None):
        super().__init__()
        self.autoencoder = autoencoder
        self.reducer = reducer
    
    def forward(self, x):
        # Get latent representation from autoencoder
        z = self.autoencoder.encode(x)
        
        # Apply dimensionality reduction if fitted
        if self.reducer is not None and self.reducer.is_fitted:
            z = self.reducer.transform(z)
        
        return z



class EncoderWrapper(nn.Module):
    """Wrapper to use autoencoder's encode method as a feature extractor."""
    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder
    
    def forward(self, x):
        return self.autoencoder.encode(x)
