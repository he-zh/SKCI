import torch
import torch.nn as nn
import numpy as np
import logging
import wandb
from omegaconf import ListConfig
from models import EarlyStopper
from utils.data_merging import CombinedDataset
from models.baseline_models import MLP, mu_X_Given_Z_Estimator, NormalizedGMMN_Estimator, ImageRegressor
from utils.gmmn_training import mmd_loss_conditional


class ECRTTrainer:
    """
    ECRT (e-value based Conditional Randomization Test) Trainer for sequential testing.

    This trainer implements the ECRT baseline method with two models:
    1. model_a: Estimates the distribution of a given c. Three modes:
       - 'oracle': Distribution is completely known (uses dataset's conditional mean)
       - 'pretrained': Pretrained on auxiliary data
       - 'online': Trained on streaming data

    2. ecrt_model: Regresses (a, c) -> b

    For each sequence k, the ecrt_model model is trained on all previous train_data.
    For each test batch, K tilde_a samples are generated for each (a_i, b_i, c_i).
    Wealth is updated based on the sign of MSE differences.
    """

    def __init__(self, cfg, datagen, device):
        """
        Initialize the ECRT Trainer.

        Args:
        - cfg: Configuration object containing trainer settings
        - datagen: Data generator object
        - device: Device (CPU/GPU) for training
        """
        # Extract configurations from cfg (consistent with trainer.py)
        self.seed = cfg.seed
        self.epochs = cfg.epochs
        self.seqs = cfg.seqs
        self.patience = cfg.earlystopping.patience
        self.delta = cfg.earlystopping.delta
        self.alpha = cfg.alpha
        self.T = cfg.T
        self.bs = cfg.get('batch_size', 100)
        self.pretrain_samples = cfg.get('pretrain_samples', 1000)

        self.model_x_mode = cfg.get('model_x_mode', 'online')  # 'online', 'pretrained', 'oracle'

        # Model-x specific parameters
        self.noise_std = cfg.get('noise_std', 1.0)  # Std for Gaussian sampling of tilde_a

        # Online/Pseudo-model-x specific parameters
        self.model_a_type = cfg.get('model_a_type', 'mean_estimator')  # 'mean_estimator' or 'gmmn'
        self.feature_extractor_type = cfg.get('model_a_feature_extractor_type', 'mlp')  # 'mlp' or 'cnn' (only for mean_estimator)
        self.use_krr = cfg.get('use_krr', False)  # Whether to use MLP+RBF KRR (only for mean_estimator)
        self.gamma_init = cfg.get('gamma_init', 1.0)  # Initial gamma for RBF kernel in KRR
        self.model_a_input_channels = cfg.get('model_a_input_channels', 1)  # Only for CNN feature extractor
        self.model_a_hidden_channel_dims = cfg.get('model_a_hidden_channels', [32, 64, 128])
        self.model_a_hidden_fc_dims = cfg.get('model_a_hidden_fc_dims', [128, 64, 32])
        self.model_a_output_size = cfg.get('model_a_output_size', 1)  # If None, defaults to 1
        self.model_a_lr = cfg.get('model_a_lr', 0.01)
        self.model_a_weight_decay = cfg.get('model_a_weight_decay', 1e-4)
        self.model_a_layer_norm = cfg.get('model_a_layer_norm', True)
        self.model_a_dropout = cfg.get('model_a_dropout', 0.1)
        self.noise_dim = cfg.get('noise_dim', 16)  # for GMMN

        # ECRT specific parameters
        self.K = cfg.get('K', 5)  # Number of tilde_a samples per test point
        self.ecrt_model_lr = cfg.get('ecrt_model_lr', 0.01)
        self.ecrt_model_weight_decay = cfg.get('ecrt_model_weight_decay', 1e-4)
        self.ecrt_model_dropout = cfg.get('ecrt_model_dropout', 0.1)
        self.ecrt_model_layer_norm = cfg.get('ecrt_model_layer_norm', True)
        # ECRT model type: 'mlp' for low-dim output, 'image_regressor' for high-dim image output
        self.ecrt_model_type = cfg.get('ecrt_model_type', 'mlp')

        # MLP specific parameters (for low-dimensional outputs)
        self.ecrt_model_hidden_dims = cfg.get('ecrt_model_hidden_dims', [64, 32])  # Only for MLP ecrt_model

        # ImageRegressor specific parameters (for high-dimensional image outputs)
        self.c_input_type = cfg.get('c_input_type', 'image')  # 'image' or 'vector'
        self.c_channels = cfg.get('c_channels', 1)  # channels for image input c
        self.b_channels = cfg.get('b_channels', 1)  # channels for image output b
        self.ecrt_model_encoder_hidden_channels = cfg.get('ecrt_model_encoder_hidden_channels', [32, 64, 128])
        self.ecrt_model_encoder_fc_hidden_size = cfg.get('ecrt_model_encoder_fc_hidden_size', [128])
        self.ecrt_model_latent_dim = cfg.get('ecrt_model_latent_dim', 64)
        self.ecrt_model_decoder_hidden_channels = cfg.get('ecrt_model_decoder_hidden_channels', [128, 64, 32])


        # Data generator and device
        self.datagen = datagen
        self.device = device

        # Will be initialized after seeing first batch of data
        self.a_dim = None
        self.b_dim = None
        self.c_dim = None
        self.model_a = None
        self.ecrt_model = None

        # Optimizers (will be initialized after models)
        self.optimizer_model_a = None
        self.optimizer_ecrt_model = None

        # Early stopper
        self.early_stopper = EarlyStopper(patience=self.patience, min_delta=self.delta)

        # Tracking variables
        self.current_seq = 0
        self.current_epoch = 0

        # Wealth tracking (martingale) - following source ECRT implementation
        # Uses 1000 λ values for integration over betting fraction
        self.integral_vector = np.linspace(0, 1, 1001, endpoint=False)[1:]  # 1000 λ values in (0,1)
        self.St_v = np.ones((1000,))  # Martingale values for each λ
        self.St = 1.0  # Current integrated wealth (mean of St_v)

    def _build_model_a(self, input_dim, output_dim):
        if self.model_a_type == 'gmmn':
            return NormalizedGMMN_Estimator(
                input_dim=input_dim,
                hidden_size=self.model_a_hidden_fc_dims,
                output_size=output_dim,
                noise_dim=self.noise_dim,
                layer_norm=self.model_a_layer_norm,
                drop_out=self.model_a_dropout > 0,
                drop_out_p=self.model_a_dropout
            ).to(self.device)

        return mu_X_Given_Z_Estimator(
            input_dim=input_dim,
            feature_extractor_type=self.feature_extractor_type,
            input_channels=self.model_a_input_channels,
            hidden_channels=self.model_a_hidden_channel_dims,
            fc_hidden_size=self.model_a_hidden_fc_dims,
            output_size=self.model_a_output_size,
            layer_norm=self.model_a_layer_norm,
            drop_out=self.model_a_dropout > 0,
            drop_out_p=self.model_a_dropout,
            use_krr=self.use_krr,
            gamma_init=self.gamma_init
        ).to(self.device)

    def _initialize_model_a(self, c_dim, a_dim):
        if self.model_x_mode not in ['online', 'pretrained']:
            self.model_a = None
            return

        logging.info(f"Initializing model_a of type {self.model_a_type} for c -> a estimation...")
        self.model_a = self._build_model_a(c_dim, a_dim)

    def _initialize_model_a_optimizer(self):
        if self.model_x_mode in ['online', 'pretrained']:
            self.optimizer_model_a = torch.optim.Adam(
                self.model_a.parameters(),
                lr=self.model_a_lr,
                weight_decay=self.model_a_weight_decay
            )
        else:
            self.optimizer_model_a = None

    def _initialize_models(self, a_dim, b_dim, c_dim):
        """Initialize models after seeing data dimensions."""
        self.a_dim = a_dim
        self.b_dim = b_dim
        self.c_dim = c_dim

        # Model A: c -> a (for sampling tilde_a)
        self._initialize_model_a(c_dim, a_dim)

        # ecrt_model model: (a, c) -> b
        if self.ecrt_model_type == 'image_regressor':
            # For high-dimensional image outputs

            self.ecrt_model = ImageRegressor(
                a_dim=a_dim,
                c_input_type=self.c_input_type,
                c_dim=c_dim,
                c_channels=self.c_channels,
                b_dim=b_dim,
                b_channels=self.b_channels,
                encoder_hidden_channels=self.ecrt_model_encoder_hidden_channels,
                encoder_fc_hidden_size=self.ecrt_model_encoder_fc_hidden_size,
                latent_dim=self.ecrt_model_latent_dim,
                decoder_hidden_channels=self.ecrt_model_decoder_hidden_channels,
                layer_norm=self.ecrt_model_layer_norm,
                drop_out=self.ecrt_model_dropout > 0,
                drop_out_p=self.ecrt_model_dropout
            ).to(self.device)
            logging.info(f"Initialized ImageRegressor: a_dim={a_dim}, c_dim={c_dim}, b_dim={b_dim}")
        else:
            # Default MLP for low-dimensional outputs
            self.ecrt_model = MLP(
                input_size=a_dim + c_dim,
                hidden_layer_size=self.ecrt_model_hidden_dims,
                output_size=b_dim,
                layer_norm=self.ecrt_model_layer_norm,
                drop_out=self.ecrt_model_dropout > 0,
                drop_out_p=self.ecrt_model_dropout
            ).to(self.device)

        # Initialize optimizers with separate hyperparameters
        self._initialize_model_a_optimizer()
        self.optimizer_ecrt_model = torch.optim.Adam(
            self.ecrt_model.parameters(),
            lr=self.ecrt_model_lr,
            weight_decay=self.ecrt_model_weight_decay
        )

        logging.info(f"Initialized models: a_dim={a_dim}, b_dim={b_dim}, c_dim={c_dim}, model_a_type={self.model_a_type}")

    def _fit_gmmn_normalization(self, data):
        """
        Fit normalization statistics for NormalizedGMMN_Estimator.
        Only used when model_a_type == 'gmmn'.

        Args:
        - data: Dataset with attributes a, c
        """
        if self.model_a_type != 'gmmn':
            return

        a = data.a.to(self.device)
        c = data.c.to(self.device)

        self.model_a.fit_normalization(c, a)
        logging.info(f"GMMN normalization fitted: x_mean={self.model_a.x_mean.mean().item():.3f}, x_std={self.model_a.x_std.mean().item():.3f}, z_mean={self.model_a.z_mean.mean().item():.3f}, z_std={self.model_a.z_std.mean().item():.3f}")

    def log(self, logs):
        """Log metrics to wandb and logging (consistent with trainer.py)."""
        for key, value in logs.items():
            wandb.log({key: value}, step=self.current_seq)
            logging.info(f"Seq: {self.current_seq}, {key}: {value}")

    def load_data(self, seed, samples=None):
        """
        Load data using the datagen object and return dataset.
        (consistent with trainer.py)
        """
        data = self.datagen.generate(seed, samples=samples)
        return data

    def _pretrain_model_a(self, train_data, val_data):
        """
        Pretrain model_a (c -> a) on auxiliary data.
        (similar structure to trainer.py's _pretrain_regressor)

        Args:
        - train_data: Training dataset with attributes a, b, c
        - val_data: Validation dataset with attributes a, b, c
        """
        logging.info(f"Pre-training model_a with {len(train_data)} samples...")

        # Fit normalization statistics for GMMN before training
        if self.model_a_type == 'gmmn':
            self._fit_gmmn_normalization(train_data)

        for t in range(self.epochs*10):  # More epochs for pretraining
            self.current_epoch = t
            train_loss = self._train_evaluate_model_a_epoch(train_data, mode='train')
            val_loss = self._train_evaluate_model_a_epoch(val_data, mode='val')

            # Early stopping check
            if self.early_stopper.early_stop(val_loss, model=self.model_a) or (t + 1) == self.epochs*10:
                self.early_stopper.restore_best(self.model_a)
                self.early_stopper.reset()
                break

        # Fit KRR on MLP features for mean_estimator (if enabled)
        if self.model_a_type == 'mean_estimator' and self.use_krr:
            c = train_data.c.to(self.device)
            a = train_data.a.to(self.device)
            self.model_a.fit_krr(c, a)

        logging.info(f"Model_a pre-training complete. Final val loss: {val_loss:.4e}")

    def _train_evaluate_model_a_epoch(self, data, mode="train"):
        """
        Train or evaluate model_a for one epoch.
        (similar structure to trainer.py's train_evaluate_regressor_epoch)

        Args:
        - data: Dataset with attributes a, b, c
        - mode: 'train' or 'val'

        Returns:
        - Average loss over the dataset
        """
        c = data.c.to(self.device)
        a = data.a.to(self.device)

        if self.model_a_type == 'gmmn':
            # GMMN: Use CMMD loss with normalized inputs/outputs
            M_train = 5

            # Normalize inputs for GMMN (using the model's built-in normalization)
            c_normalized = self.model_a.normalize_z(c)
            a_normalized = self.model_a.normalize_x(a)

            if mode == "train":
                self.model_a.train()
                self.optimizer_model_a.zero_grad()

                # Generate M samples per conditioning variable (in normalized space)
                a_fake_normalized = self.model_a.sample_multiple_normalized(c_normalized, M_train)
                a_real_normalized = a_normalized.view(a_normalized.shape[0], -1)
                c_real_normalized = c_normalized.view(c_normalized.shape[0], -1)

                loss = mmd_loss_conditional(a_real_normalized, a_fake_normalized, c_real_normalized,
                                           kernel_type='rbf', M=M_train)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_a.parameters(), max_norm=0.5)
                self.optimizer_model_a.step()
            else:
                self.model_a.eval()
                with torch.no_grad():
                    a_fake_normalized = self.model_a.sample_multiple_normalized(c_normalized, M_train)
                    a_real_normalized = a_normalized.view(a_normalized.shape[0], -1)
                    c_real_normalized = c_normalized.view(c_normalized.shape[0], -1)
                    loss = mmd_loss_conditional(a_real_normalized, a_fake_normalized, c_real_normalized,
                                               kernel_type='rbf', M=M_train)
        else:
            # Mean estimator
            if self.use_krr:
                # LOO loss for joint MLP + KRR hyperparameter optimization
                if mode == "train":
                    self.model_a.train()
                    self.optimizer_model_a.zero_grad()
                    loss = self.model_a.compute_loo_loss(c, a)
                    loss.backward()
                    self.optimizer_model_a.step()
                    self.model_a.ridge_lambda.data.clamp_(min=1e-5)
                # For validation, we compute the MSE loss
                else:
                    self.model_a.eval()
                    with torch.no_grad():
                        loss = self.model_a.compute_val_loss(c, a)
            else:
                # Plain MSE loss
                if mode == "train":
                    self.model_a.train()
                    self.optimizer_model_a.zero_grad()
                    a_pred = self.model_a(c)
                    loss = nn.MSELoss()(a_pred, a)
                    loss.backward()
                    self.optimizer_model_a.step()
                else:
                    self.model_a.eval()
                    with torch.no_grad():
                        a_pred = self.model_a(c)
                        loss = nn.MSELoss()(a_pred, a)

        logging.info(f"Seq: {self.current_seq}, Epoch: {self.current_epoch}, Model_a {mode} loss: {loss.item():.4e}")
        return loss.item()

    def _train_evaluate_ecrt_model_epoch(self, data, mode="train"):
        """
        Train or evaluate ecrt_model model for one epoch.
        (similar structure to trainer.py's train_evaluate_regressor_epoch)

        Args:
        - data: Dataset with attributes a, b, c
        - mode: 'train' or 'val'

        Returns:
        - Average loss over the dataset
        """
        a = data.a.to(self.device)
        b = data.b.to(self.device)
        c = data.c.to(self.device)

        # Flatten b for MSE loss if it's an image
        b_flat = b.view(b.shape[0], -1) if b.dim() > 2 else b

        if self.ecrt_model_type == 'image_regressor':
            # ImageRegressor uses forward(a, c) interface
            if mode == "train":
                self.ecrt_model.train()
                self.optimizer_ecrt_model.zero_grad()
                b_pred = self.ecrt_model(a, c)
                b_pred_flat = b_pred.view(b_pred.shape[0], -1)
                loss = nn.MSELoss()(b_pred_flat, b_flat)
                loss.backward()
                self.optimizer_ecrt_model.step()
            else:
                self.ecrt_model.eval()
                with torch.no_grad():
                    b_pred = self.ecrt_model(a, c)
                    b_pred_flat = b_pred.view(b_pred.shape[0], -1)
                    loss = nn.MSELoss()(b_pred_flat, b_flat)
        else:
            # MLP uses concatenated (a, c) as input
            ac = torch.cat([a, c], dim=-1)

            if mode == "train":
                self.ecrt_model.train()
                self.optimizer_ecrt_model.zero_grad()
                b_pred = self.ecrt_model(ac)
                loss = nn.MSELoss()(b_pred, b_flat)
                loss.backward()
                self.optimizer_ecrt_model.step()
            else:
                self.ecrt_model.eval()
                with torch.no_grad():
                    b_pred = self.ecrt_model(ac)
                    loss = nn.MSELoss()(b_pred, b_flat)

        logging.info(f"Seq: {self.current_seq}, Epoch: {self.current_epoch}, ecrt_model {mode} loss: {loss.item():.4e}")
        return loss.item()

    def sample_tilde_a_single(self, c, a, a_m=None):
        """
        Sample a single tilde_a given c (one sample per data point).

        Args:
        - c: Conditioning variable of shape (batch_size, c_dim)
        - a: Original a values (used for estimating noise_std in mean_estimator mode)
        - a_m: True conditional mean E[a|c] for oracle mode (optional).
               If provided, uses this directly instead of model_a.

        Returns:
        - tilde_a: Sampled values of shape (batch_size, a_dim)
        """
        if self.model_x_mode == 'oracle' and a_m is not None:
            # Use the true conditional mean directly (no model_a needed)
            noise = torch.randn_like(a_m) * self.noise_std
            tilde_a = a_m + noise
        elif self.model_x_mode == 'oracle' and a_m is None:
            raise ValueError("In oracle mode, a_m (true conditional mean) must be provided.")
        elif self.model_a_type == 'gmmn':
            # GMMN: sample directly using the generator
            # NormalizedGMMN_Estimator handles normalization/denormalization automatically
            with torch.no_grad():
                self.model_a.eval()
                tilde_a = self.model_a.sample(c, n_samples=1)
        else:
            # mean_estimator: estimate mean and add noise based on residuals
            with torch.no_grad():
                self.model_a.eval()
                a_mean = self.model_a(c)  # (batch_size, a_dim)
                # estimate noise_std from training residuals
                noise_std = torch.std(a - a_mean, dim=0)
                noise = torch.randn_like(a_mean) * noise_std
                tilde_a = a_mean + noise
        return tilde_a

    def compute_wealth_update(self, test_data):
        """
        Compute wealth update for a test batch using ECRT method.

        Following the source ECRT implementation:
        1. Compute MSE for original (a, c) -> b once
        2. For K iterations, sample tilde_a, compute tilde MSE, accumulate betting scores
        3. Update all 1000 λ-martingales with average betting score
        4. Return mean of all martingales (integrated wealth)

        Args:
        - test_data: Test dataset with attributes a, b, c (and a_m for oracle mode)

        Returns:
        - Current integrated wealth (self.St)
        """
        a = test_data.a.to(self.device)
        b = test_data.b.to(self.device)
        c = test_data.c.to(self.device)

        # For oracle mode, get the true conditional mean from the dataset
        a_m = None
        if self.model_x_mode == 'oracle' and hasattr(test_data, 'a_m'):
            a_m = test_data.a_m.to(self.device)
        elif self.model_x_mode == 'oracle' and not hasattr(test_data, 'a_m'):
            raise ValueError("In oracle mode, test_data must have attribute a_m (true conditional mean).")

        self.ecrt_model.eval()
        with torch.no_grad():
            # Flatten b for MSE computation if it's an image
            b_flat = b.view(b.shape[0], -1) if b.dim() > 2 else b

            # Compute MSE for original features once
            if self.ecrt_model_type == 'image_regressor':
                b_pred_original = self.ecrt_model(a, c)
                b_pred_original_flat = b_pred_original.view(b_pred_original.shape[0], -1)
            else:
                ac = torch.cat([a, c], dim=-1)
                b_pred_original_flat = self.ecrt_model(ac)

            q = ((b_pred_original_flat - b_flat) ** 2).mean().item()  # MSE for original

            # Accumulate betting scores over K dummy samples
            total_betting_score = 0.0
            for k in range(self.K):
                # Sample tilde_a for this iteration
                # For oracle mode, uses true conditional mean a_m
                tilde_a_k = self.sample_tilde_a_single(c, a, a_m=a_m)  # (batch_size, a_dim)

                if self.ecrt_model_type == 'image_regressor':
                    b_pred_tilde = self.ecrt_model(tilde_a_k, c)
                    b_pred_tilde_flat = b_pred_tilde.view(b_pred_tilde.shape[0], -1)
                else:
                    tilde_ac_k = torch.cat([tilde_a_k, c], dim=-1)
                    b_pred_tilde_flat = self.ecrt_model(tilde_ac_k)

                q_tilde = ((b_pred_tilde_flat - b_flat) ** 2).mean().item()  # MSE for tilde

                # g_func = sign (antisymmetric betting function)
                # sign(q - q_tilde): positive if original is better (supports H1)
                total_betting_score += np.sign(q_tilde - q)

            # Average betting score
            avg_betting_score = total_betting_score / self.K

            # Update all λ-martingales: St_v = St_v * (1 + λ * avg_score)
            self.St_v = self.St_v * (1 + self.integral_vector * avg_betting_score)

            # Integrated wealth = mean over all λ values
            self.St = np.mean(self.St_v)

        return self.St

    def train(self):
        """
        Main training loop for ECRT method.
        (consistent structure with trainer.py)
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Load initial data (consistent with trainer.py)
        train_data = self.load_data(self.seed)
        val_data = self.load_data(self.seed + 1)

        # Initialize models based on data dimensions
        if self.a_dim is None:
            a_sample = train_data.a[0]
            b_sample = train_data.b[0]
            c_sample = train_data.c[0]

            a_dim = a_sample.shape[-1] if a_sample.dim() > 0 else 1
            b_dim = b_sample.shape[-1] if b_sample.dim() > 0 else 1
            c_dim = c_sample.shape[-1] if c_sample.dim() > 0 else 1

        self._initialize_models(a_dim, b_dim, c_dim)

        # Handle pretraining for model_a (only needed for pretrained mode)
        # For oracle mode: we use the true conditional distribution directly from dataset
        # For online mode: model_a is trained on streaming data in the main loop
        if self.model_x_mode == 'pretrained':
            # Pretrain with noisy samples
            pretrain_len = int(self.pretrain_samples * 0.8)
            pretrain_data = self.load_data(seed=999999 + self.seed, samples=pretrain_len)
            val_len = self.pretrain_samples - pretrain_len
            pretrain_val_data = self.load_data(seed=888888 + self.seed, samples=val_len)
            self._pretrain_model_a(pretrain_data, pretrain_val_data)
        elif self.model_x_mode == 'oracle':
            # No pretraining needed - we use the true conditional mean a_m from dataset
            logging.info("oracle mode: using true conditional distribution from dataset (no model_a training)")

        # Main sequential loop (consistent with trainer.py)
        # Wealth is tracked in self.St (integrated martingale)
        reject_null = 0.0

        for k in range(self.seqs):
            self.current_seq = k
            test_data = self.load_data(self.seed + k + 2)

            # Train model_a on streaming data if in 'online' mode
            # (consistent with trainer.py's handling of online mode)
            if self.model_x_mode == 'online':
                logging.info(f"Training model_a on sequence {k}...")

                # Reset KRR before retraining MLP features
                if self.model_a_type == 'mean_estimator' and self.use_krr:
                    self.model_a.reset_krr()

                # Fit normalization statistics for GMMN (only on first iteration)
                if self.model_a_type == 'gmmn' and k == 0:
                    self._fit_gmmn_normalization(train_data)

                for t in range(self.epochs):
                    self.current_epoch = t
                    train_loss = self._train_evaluate_model_a_epoch(train_data, mode='train')
                    val_loss = self._train_evaluate_model_a_epoch(val_data, mode='val')
                
                    if self.early_stopper.early_stop(val_loss, model=self.model_a) or (t + 1) == self.epochs:
                        self.early_stopper.restore_best(self.model_a)
                        self.early_stopper.reset()
                        break

                self.log({'model_a_train_loss': train_loss})
                self.log({'model_a_val_loss': val_loss})

                # Fit KRR on MLP features for mean_estimator (if enabled)
                if self.model_a_type == 'mean_estimator' and self.use_krr:
                    c_all = train_data.c.to(self.device)
                    a_all = train_data.a.to(self.device)
                    self.model_a.fit_krr(c_all, a_all)

            # Train ecrt_model model on all previous data
            # For each k, train on all previous train_data
            logging.info(f"Training ecrt_model model on sequence {k}...")
            for t in range(self.epochs):
                self.current_epoch = t
                train_loss = self._train_evaluate_ecrt_model_epoch(train_data, mode='train')
                val_loss = self._train_evaluate_ecrt_model_epoch(val_data, mode='val')

                if self.early_stopper.early_stop(val_loss, model=self.ecrt_model) or (t + 1) == self.epochs:
                    self.early_stopper.restore_best(self.ecrt_model)
                    self.early_stopper.reset()
                    break

            self.log({'ecrt_model_train_loss': train_loss})
            self.log({'ecrt_model_val_loss': val_loss})

            # Compute wealth update on test data if k >= T
            # (consistent with trainer.py's wealth update logic)
            if k >= self.T:
                # compute_wealth_update updates self.St_v and self.St internally
                current_wealth = self.compute_wealth_update(test_data)
                self.log({'wealth': current_wealth})

                # Check for rejection (consistent with trainer.py)
                if self.St > (1.0 / self.alpha):
                    reject_null = 1.0
                    logging.info(f"Reject null at sequence {k}, wealth: {self.St}")
                    self.log({'stopping_time': k})
                    self.log({'reject_null': reject_null})
                else:
                    self.log({'reject_null': reject_null})

            # Combine train and val, move test to val (consistent with trainer.py)
            train_data = CombinedDataset([train_data, val_data])
            val_data = test_data

            self.log({'historical_sample_nums': len(train_data)})
            self.log({'all_sample_nums': len(train_data) + len(test_data)})

        logging.info(f"Training complete. Final wealth: {self.St}")
