import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import wandb
from models import EarlyStopper, ImageAutoencoder, AutoencoderModel
from utils.criterions import LeaveOneOutKRR, SquareLossKRR, compute_unbiased_kci
from utils.matrix_processing import get_centered_kernel_matrix
from utils.data_merging import CombinedDataset

class Trainer:

    def __init__(self, cfg, kernel_a, kernel_b, kernel_c, kernel_ca, kernel_cb, datagen, device):
        """
        Initializes the Trainer object with the provided configurations and parameters.

        Args:
        - cfg (Config): Configuration object containing trainer settings.
        - net (nn.Module): The neural network model to train.
        - tau1 (float): Operator 1.
        - tau2 (float): Operator 2.
        - datagen (DataGenerator): Object to generate data.
        - device (torch.device): The device (CPU/GPU) where training should take place.
        """
        # Extract configurations from the cfg object
        self.seed = cfg.seed
        self.lr = cfg.lr
        self.epochs = cfg.epochs
        self.seqs = cfg.seqs
        self.patience = cfg.earlystopping.patience
        self.delta = cfg.earlystopping.delta
        self.alpha = cfg.alpha
        # self.scale = cfg.scale
        self.gamma = torch.tensor([0.0], dtype=torch.float32, device=device)
        self.eps = torch.tensor([cfg.eps], dtype=torch.float32, device=device)
        self.T = cfg.T
        self.bs = cfg.batch_size
        self.pretrain_samples = cfg.get('pretrain_samples', 1000)
        # model_x_mode can be: 'online', 'pretrained', 'oracle'
        # 'online' = learn regression on streaming data
        # 'pretrained' = pretrain with noisy samples
        # 'oracle' = pretrain with conditional means on a grid 
        self.model_x_mode = cfg.get('model_x_mode', 'online')

        # Model, data generator, and device assignment
        self.kernel_a = kernel_a
        self.kernel_b = kernel_b
        self.kernel_c = kernel_c
        self.kernel_ca = kernel_ca
        self.kernel_cb = kernel_cb
        # self.betting_fraction = nn.Parameter(torch.tensor([0.1])).to(device)
        self.betting_fraction_trainable = cfg.get('betting_fraction_trainable', True)
        self.betting_fraction = nn.Parameter(torch.tensor([cfg.get('betting_fraction', 0.0)], dtype=torch.float32, device=device),
                                             requires_grad=self.betting_fraction_trainable)
        self.datagen = datagen
        self.device = device

        # L1 and L2 regularization parameters
        self.weight_decay = cfg.l2_lambda

        # Initialize the optimizer
        if self.betting_fraction_trainable:
            self.optimizer_betting = torch.optim.Adam([self.betting_fraction],
                                                  lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer_betting = None
        if self.kernel_c.is_trainable:
            self.optimizer_c = torch.optim.Adam(self.kernel_c.parameters(), 
                                                lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer_c = None
        # self.optimizer_c_cp = torch.optim.Adam(self.kernel_c_cp.parameters(), 
        #                                           lr=self.lr, weight_decay=self.weight_decay)
        if kernel_ca.is_trainable:
            self.optimizer_ca = torch.optim.Adam(self.kernel_ca.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer_ca = None
        if kernel_cb.is_trainable:
            self.optimizer_cb = torch.optim.Adam(self.kernel_cb.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer_cb = None
        self.early_stopper = EarlyStopper(patience=self.patience, min_delta=self.delta)

        # Autoencoder settings for image dimension reduction
        self.autoencoder_lr = cfg.get('autoencoder_lr', 1e-3)
        self.use_autoencoder_for_b = cfg.get('use_autoencoder_for_b', False)
        
        # Check if kernel_b is an AutoencoderModel and setup autoencoder training
        self.has_autoencoder = isinstance(kernel_b, AutoencoderModel)
        if self.has_autoencoder and self.use_autoencoder_for_b:
            self.autoencoder = kernel_b.get_autoencoder()
            self.autoencoder_optimizer = torch.optim.Adam(
                self.autoencoder.parameters(), 
                lr=self.autoencoder_lr, 
                weight_decay=self.weight_decay
            )
        else:
            self.autoencoder = None
            self.autoencoder_optimizer = None

        # Variables to keep track of the current sequence and epoch
        self.current_seq = 0
        self.current_epoch = 0

    def log(self, logs):
        """
        Log metrics for visualization and monitoring.

        Args:
        - logs (dict): Dictionary containing metrics to be logged.
        """
        for key, value in logs.items():
            wandb.log({key: value}, step=self.current_seq)
            logging.info(f"Seq: {self.current_seq}, {key}: {value}")

    def load_data(self, seed, samples=None):
        """
        Load data using the datagen object and return a DataLoader object.

        Args:
        - seed (int): Seed for generating data.
        - samples (int, optional): Number of samples to generate. If None, uses default.

        Returns:
        - tuple: Generated data and corresponding DataLoader object.
        """
        data = self.datagen.generate(seed, samples=samples)

        return data

    def get_Ckci_kernel_matrix(self, data):
        a = data.a.to(self.device)
        b = data.b.to(self.device)
        c = data.c.to(self.device)
        K_aa_centered = get_centered_kernel_matrix(c, a, self.kernel_ca, self.kernel_a)
        K_bb_centered = get_centered_kernel_matrix(c, b, self.kernel_cb, self.kernel_b)
        K_cc = self.kernel_c(c, c)
        
        return K_aa_centered, K_bb_centered, K_cc

    def _train_autoencoder(self, train_data, val_data):
        """
        Train the autoencoder on image data (b) for dimensionality reduction.
        
        Uses reconstruction loss (MSE) to train the autoencoder.
        
        Args:
            train_data: Training dataset containing images in b
            val_data: Validation dataset for early stopping
        """
        if self.autoencoder is None:
            logging.warning("No autoencoder to train.")
            return
        
        logging.info(f"Training autoencoder with {len(train_data)} samples...")
        
        # Enable gradient computation for autoencoder
        self.autoencoder.train()
        for param in self.autoencoder.parameters():
            param.requires_grad = True
        
        # Get training images
        train_b = train_data.b.to(self.device)
        val_b = val_data.b.to(self.device)
        
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training step
            self.autoencoder_optimizer.zero_grad()
            recon_b, latent = self.autoencoder(train_b)
            train_loss = F.mse_loss(recon_b, train_b)
            train_loss.backward()
            self.autoencoder_optimizer.step()
            
            # Validation step
            self.autoencoder.eval()
            with torch.no_grad():
                recon_val, _ = self.autoencoder(val_b)
                val_loss = F.mse_loss(recon_val, val_b)
            self.autoencoder.train()
            
            # Early stopping check
            if val_loss < best_val_loss - self.delta:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.autoencoder.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                logging.info(f"Autoencoder Epoch {epoch}: train_loss={train_loss.item():.6f}, val_loss={val_loss.item():.6f}")
            
            if patience_counter >= self.patience:
                logging.info(f"Early stopping autoencoder at epoch {epoch}")
                break
        
        # Restore best model
        if best_state is not None:
            self.autoencoder.load_state_dict(best_state)
        
        # Freeze autoencoder after pretraining (unless jointly trainable)
        if not self.kernel_b.is_trainable:
            for param in self.autoencoder.parameters():
                param.requires_grad = False
        
        self.autoencoder.eval()
        logging.info(f"Autoencoder training complete. Final val_loss: {best_val_loss:.6f}")
        

    def generate_noiseless_data(self, seed, samples, type="ca"):
        """
        Generate noiseless data for oracle mode where the target is the 
        conditional mean (no random noise added).
        
        Args:
        - seed: Random seed for reproducibility
        - samples: Number of samples to generate
        - type: Either 'ca' (for a) or 'cb' (for b)
        
        Returns:
        - c: Conditioning variable values
        - target_mean: Conditional means E[a|c] or E[b|c] (noiseless)
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate data using datagen but extract only the conditional mean
        data = self.datagen.generate(seed, samples=samples)
        c = data.c
        
        if type == "ca":
            # For Gaussian data, this is X_mu; for Sin data, this is a_m
            target_mean = data.a_m if hasattr(data, 'a_m') else data.a
        else:  # type == "cb"
            # For Gaussian data, this is Y_mu; for Sin data, this is b_m
            target_mean = data.b_m if hasattr(data, 'b_m') else data.b
        
        return c, target_mean

    def _pretrain_regressor(self, X, x, Y, y, kernel_x, kernel_y, optimizer, type="ca"):
        """
        Performs one-time training for kernel_ca or kernel_cb using a fixed,
        large number of samples (e.g., 1000).
        Args:
        - val_x: validation input data for early stopping
        - val_y: validation output data for early stopping
        - kernel_x: kernel_ca or kernel_cb
        - kernel_y: kernel_a or kernel_b
        - optimizer: optimizer for kernel_x
        """
        logging.info(f"Pre-training regressor {type} with {len(X)} samples...")

        # --- Pre-train kernel_ca/cb ---
        # estimate ca/cb regressior with noise-free inputs if under oracle setting

        # Set the kernel matrix for the pre-trained labels
        kernel_y.set_kernel_matrix(Y.to(self.device))
        if kernel_x.is_trainable:
            logging.info(f"Pre-training kernel_{type}...")
            for t in range(self.epochs):
                self.current_epoch = t
                # Train for kernel_ca
                regressor_train_loss = self.train_evaluate_regressor_epoch(
                    X, None, Y, None, 
                    kernel_x, kernel_y, 
                    optimizer=optimizer, mode='train', type=type
                )
                regressor_val_loss = self.train_evaluate_regressor_epoch(X, x, Y, y, kernel_x, kernel_y, 
                                                                        mode='val', type=type)

                if self.early_stopper.early_stop(regressor_val_loss, model=kernel_x) or (t + 1) == self.epochs:
                    self.early_stopper.restore_best(kernel_x)
                    self.early_stopper.reset()
                    break

        # Set the kernel matrix for the pre-trained inputs
        kernel_x.set_kernel_matrix(X.to(self.device))

        logging.info(f"Regressor {type} pre-training complete.")

    def train_evaluate_regressor_epoch(self, X, x, Y, y, model_x, model_y, optimizer=None, mode="train", type="ca"):
        """
        Train or evaluate the model for one epoch and log the results.

        Args:
        - X: Training input data.
        - x: Validation/Test input data.
        - Y: Training output data.
        - y: Validation/Test output data.
        - model_x: Regressor model (kernel_ca or kernel_cb).
        - model_y: Target model (kernel_a or kernel_b).
        - optimizer: Optimizer for the regressor model.
        - mode (str): Either "train", "val", or "test". Determines how to run the model.
        - type (str): Type of regressor, either "ca" or "cb".

        Returns:
        - tuple: Aggregated loss and davt for the current epoch.
        """
        X = X.to(self.device)
        if x is not None:
            x = x.to(self.device)
        Y = Y.to(self.device)
        if y is not None:
            y = y.to(self.device)

        K_YY = model_y.kernel_matrix
        if mode == "train" and model_x.is_trainable:
            # leave-one-out cross-validation for training
            criterion = LeaveOneOutKRR()
            optimizer.zero_grad()
            model_x.train()
            K_XX = model_x(X, X)
            loss = criterion(K_XX, K_YY, model_x.ridge_lambda)
            loss.backward()
            optimizer.step()
            model_x.ridge_lambda.data.clamp_(min=1e-5)
        else:
            # full batch for validation and testing
            criterion = SquareLossKRR()
            model_x.eval()
            with torch.no_grad():
                K_xX = model_x(x, X)
                K_XX = model_x(X, X)
                K_yy = model_y(y, y)
                K_yY = model_y(y, Y)
                loss = criterion(K_xX, K_XX, K_yy, K_yY, K_YY, model_x.ridge_lambda)

        logging.info(f"Seq: {self.current_seq}, Epoch: {self.current_epoch}, Regression ({type}) {mode} loss: {loss.item():.4e}")
        return loss.item()

    def train_evaluate_kernel_c_epoch(self, train_data, val_data=None, mode="train"):
        """
        Train or evaluate the model for one epoch and log the results.

        Args:
        - loader (DataLoader): DataLoader object to iterate through data.
        - mode (str): Either "train", "val", or "test". Determines how to run the model.

        Returns:
        - tuple: Aggregated loss and davt for the current epoch.
        wealth = ave / ||Ckci||_HS
        """
        # For training, maximize kci
        if mode == "train":
            self.kernel_c.train()
            K_aa_centered, K_bb_centered, K_cc = self.get_Ckci_kernel_matrix(train_data)
            # Optimize kernel_c by maximizing kci
            # could also consider merge into one optimizer with betting fraction
            kci = compute_unbiased_kci(K_aa_centered, K_bb_centered, K_cc)
            loss = - kci
            self.optimizer_c.zero_grad()
            loss.backward()
            self.optimizer_c.step()

            logging.info(f"Seq: {self.current_seq}, Epoch: {self.current_epoch}, Bandwidth(kernel C) = {torch.exp(self.kernel_c.kernel.log_gamma.data.reshape(-1)[0]):.4f}")
        else:
            self.kernel_c.eval()
            with torch.no_grad():
                train_val_data = CombinedDataset([train_data, val_data])
                K_aa_centered, K_bb_centered, K_cc = self.get_Ckci_kernel_matrix(train_val_data)
                kci = compute_unbiased_kci(K_aa_centered, K_bb_centered, K_cc)
                loss = - kci

        logging.info(f"Seq: {self.current_seq}, Epoch: {self.current_epoch}, Kernel C {mode} loss: {loss.item():.4e}")
        return loss.item()

    def get_gamma(self, train_data, val_data, mode="train"):
        """
        Estimate gamma using wild bootstrap

        Args:
            train_data: Training dataset.
            val_data: Validation dataset.
            mode: "train" or "val" - determines how to use the data for estimation

        Returns:
            gamma (tensor float): Estimated gamma value.
        """

        if val_data is None and mode == "val":
            raise ValueError("Validation data must be provided to estimate gamma in val mode.")
        with torch.no_grad():
            if mode == "train":
                train_nums = len(train_data)
                # For training, use training data with a wild bootstrap approach to estimate variance
                K_aa_centered, K_bb_centered, K_cc = self.get_Ckci_kernel_matrix(train_data)
                K_prod = (K_aa_centered * K_bb_centered * K_cc).detach()
                ckci2 = K_prod.mean()  # V statistics estimator of E[K_prod]

                K_mean_vector = K_prod.mean(dim=0) 

                variance = (1 / self.bs) * torch.mean((K_mean_vector / (ckci2 + self.eps)) ** 2)
                
            elif mode == "val":
                # combine datasets
                train_val_data = CombinedDataset([train_data, val_data])
                train_nums = len(train_data)
                val_nums = len(val_data)

                train_idx = torch.arange(train_nums)
                val_idx = torch.arange(len(train_val_data))[-val_nums:]

                # kernel matrices
                K_aa_centered, K_bb_centered, K_cc = self.get_Ckci_kernel_matrix(train_val_data)

                # product kernel
                K_prod = (K_aa_centered * K_bb_centered * K_cc).detach()

                # training statistic
                K_prod_train = K_prod[train_idx][:, train_idx]
                ckci2 = K_prod_train.mean()

                # cross (train–val) block
                K_prod_cross = K_prod[train_idx][:, val_idx]
                
                K_mean_vector = K_prod_cross.mean(dim=0) 
                variance = (1.0 / val_nums) * torch.mean((K_mean_vector / (ckci2 + self.eps)) ** 2)
            else:
                raise ValueError("Mode must be either 'train' or 'val' for gamma estimation.")
            sigma = torch.sqrt(variance)
            sigma_val = float(sigma.item())

            max_gamma = max(2.0, 8.0 * sigma_val + 1.0)
            grid = torch.linspace(0.0, max_gamma, 20000, device=K_mean_vector.device)
            normal = torch.distributions.Normal(loc=0.0, scale=1.0)
            rhs = sigma * normal.cdf((grid - 1.0) / sigma)
            residual = torch.abs(grid - rhs)
            idx = torch.argmin(residual)
            self.gamma = grid[idx]

        logging.info(f"Seq: {self.current_seq}, Epoch: {self.current_epoch}, Estimated gamma: {self.gamma:.4f}")
        # return gamma.item()
    
    def compute_V_mfold(self, K_aa_centered, K_bb_centered_cc, folds, mode="train"):
        """
        Compute W under m-fold cross validation.

        Args:
            K_aa_centered: (n,n) tensor
            K_bb_centered_cc: (n,n) tensor
            folds: list of index sets (each a list/array of indices for one fold)
            mode: "train" or "val"
        Returns:
            V_list: V over folds
        """
        n = K_aa_centered.shape[0]

        V_list = []
        K_prod = K_aa_centered * K_bb_centered_cc
        if mode == "train":
            # For training, each fold use training all data to estimate biased ckci2
            # while only using its own fold to estimate unbiased E<ckci, \phi(a,b,c)>
            # which is \sum_{i=1}^n \sum_{j in fold and j \neq i} K_prod_{i,j} / (bs*(n-1))
            ckci2 = K_prod.mean() 
            ckci = torch.sqrt(ckci2)

            _K_prod = K_prod - torch.diagflat(torch.diag(K_prod))
            K_prod_row_sum = _K_prod.sum(dim=1)
            for fold_idx, val_idx in enumerate(folds):
                numerator = K_prod_row_sum[val_idx].sum() / (len(val_idx)*(n - 1))

                V_fold = numerator / (ckci2+self.eps) - self.gamma
                V_fold = torch.clamp(V_fold, min=-1)

                V_list.append(V_fold)
        else:
            # For evaluation, use training set to estimate biased ckci2
            # and use validation set to estimate unbiased E<ckci, \phi(a,b,c)>
            # which is \sum_{i=1}^n \sum_{j=n+1}^{n+m} K_prod_{i,j} / (m*n)
            assert len(folds) == 1, "For evaluation, only one fold (the validation set) should be provided."
            val_idx = folds[0]
            train_idx = torch.arange(n)[~np.isin(np.arange(n), val_idx)]
            K_prod_train = K_prod[train_idx][:, train_idx]
            ckci2 = K_prod_train.mean() 
            ckci = torch.sqrt(ckci2)
            numerator = K_prod[train_idx][:, val_idx].mean()

            V_fold = numerator / (ckci2+self.eps) - self.gamma
            V_fold = torch.clamp(V_fold, min=-1)

            V_list.append(V_fold)


        V_list = torch.stack(V_list)
        return V_list

        
    def train_evaluate_epoch(self, train_data, val_data=None, mode="train"):
        """
        Train or evaluate the betting fraction for one epoch and log the results.

        Args:
        - loader (DataLoader): DataLoader object to iterate through data.
        - mode (str): Either "train", "val", or "test". Determines how to run the model.

        Returns:
        - tuple: Aggregated loss and davt for the current epoch.
        """

        if mode == "train":
            # For training, split train_data into folds with batch size self.bs
            # and compute V for each fold
            self.kernel_c.train()
            K_aa_centered, K_bb_centered, K_cc = self.get_Ckci_kernel_matrix(train_data)
            m = K_aa_centered.shape[0]
            cv_folds = [torch.arange(i * self.bs, min((i + 1) * self.bs, m))
                        for i in range(0, (m + self.bs - 1) // self.bs)]
            v_list = self.compute_V_mfold(K_aa_centered.detach(), (K_bb_centered.detach() * K_cc), folds=cv_folds, mode="train")
            betting = torch.sigmoid(self.betting_fraction)
            loss = - torch.log(1 + betting * v_list).sum()
            if self.optimizer_betting is not None:
                self.optimizer_betting.zero_grad()
            if self.optimizer_c is not None:
                self.optimizer_c.zero_grad()
            loss.backward()
            if self.optimizer_betting is not None:
                self.optimizer_betting.step()
            if self.optimizer_c is not None:
                self.optimizer_c.step()
            self.get_gamma(train_data=train_data, val_data=val_data, mode="train")
        else:
            # For evaluation, use val_data as the validation set
            # and compute V with train_data as training set
            self.kernel_c.eval()
            with torch.no_grad():
                train_val_data = CombinedDataset([train_data, val_data])
                val_nums = len(val_data)
                val_idx = [torch.arange(len(train_val_data))[-val_nums:]]
                K_aa_centered, K_bb_centered, K_cc = self.get_Ckci_kernel_matrix(train_val_data)
                v_list = self.compute_V_mfold(K_aa_centered.detach(), (K_bb_centered * K_cc).detach(), folds=val_idx, mode="val")
                betting = torch.sigmoid(self.betting_fraction)
                wealth_update = (1 + betting * v_list[0])
                loss = - wealth_update
        logging.info(f"Seq: {self.current_seq}, Epoch: {self.current_epoch}, Bandwidth(kernel C) = {torch.exp(self.kernel_c.kernel.log_gamma.data.reshape(-1)[0]):.4f}")
        logging.info(f"Seq: {self.current_seq}, Epoch: {self.current_epoch}, Betting fraction = {betting.item():.4f}")
        logging.info(f"Seq: {self.current_seq}, Epoch: {self.current_epoch}, Betting {mode} loss: {loss.item():.4e}")
        return loss.item(), v_list[0].item()

    def train(self):
        """
        Train the model for a specified number of sequences and epochs, and apply early stopping if required.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # For pretrained mode, load pretrain_data FIRST so normalization stats
        # are computed from pretrain_data (not train_data)
        pretrain_data = None
        pretrain_val_data = None
        if self.model_x_mode == 'pretrained':
            pretrain_len = int(self.pretrain_samples * 0.8)
            val_len = self.pretrain_samples - pretrain_len
            # Load pretrain data first - this computes norm_stats if normalize=True
            pretrain_data = self.load_data(seed=999999+self.seed, samples=pretrain_len)
            pretrain_val_data = self.load_data(seed=888888+self.seed, samples=val_len)
            # Explicitly set norm_stats from pretrain_data for subsequent data
            if hasattr(self.datagen, 'compute_norm_stats_from_data'):
                self.datagen.compute_norm_stats_from_data(pretrain_data)
        
        # Now load train/val data - will reuse norm_stats computed from pretrain_data
        # (or compute norm_stats here if in online mode)
        train_data = self.load_data(self.seed)
        val_data = self.load_data(self.seed + 1)
        
        # In online mode, first compute norm_stats from initial train_data
        if self.model_x_mode == 'online':
            if hasattr(self.datagen, 'compute_norm_stats_from_data'):
                self.datagen.compute_norm_stats_from_data(train_data)
        
        wealth = 1.0
        reject_null = 0.0

        # Handle model_x_mode for regression of c -> a
        if self.model_x_mode == 'pretrained':
            # Pretrain with noisy samples (old model_x=True behavior)
            # pretrain_data already loaded above
            self._pretrain_regressor(pretrain_data.c, pretrain_val_data.c, pretrain_data.a, pretrain_val_data.a, 
                                     self.kernel_ca, self.kernel_a, self.optimizer_ca, type="ca")
        elif self.model_x_mode == 'oracle':
            # Pretrain with conditional means (new mode)
            # c is sampled, target is E[a|c] (noiseless)
            pretrain_len = int(self.pretrain_samples * 0.8)
            val_len = self.pretrain_samples - pretrain_len
            c_train, a_mean_train = self.generate_noiseless_data(seed=999999+self.seed, samples=pretrain_len, type="ca")
            c_val, a_mean_val = self.generate_noiseless_data(seed=888888+self.seed, samples=val_len, type="ca")
            self._pretrain_regressor(c_train, c_val, a_mean_train, a_mean_val, 
                                     self.kernel_ca, self.kernel_a, self.optimizer_ca, type="ca")
        # If 'online', do nothing here - regression will be learned during streaming

        # Pretrain autoencoder for image dimension reduction (if kernel_b is AutoencoderModel)
        # Only pretrain here if in 'pretrained' mode; 'online' mode trains at each step
        if self.has_autoencoder and self.use_autoencoder_for_b and self.model_x_mode == 'pretrained':
            # Use the same pretrain_data as regressors
            self._train_autoencoder(pretrain_data, pretrain_val_data)
           

        for k in range(self.seqs):
            self.current_seq = k
            test_data = self.load_data(self.seed + k + 2)
            
            # Online autoencoder training: train at each step with current train_data
            if self.has_autoencoder and self.use_autoencoder_for_b and self.model_x_mode == 'online':
                self._train_autoencoder(train_data, val_data)
            
            if self.model_x_mode == 'online':
                # Storing training labels for regressor ca
                self.kernel_a.set_kernel_matrix(train_data.a.to(self.device))
                if self.kernel_ca.is_trainable:
                    # Train for kernel_ca
                    for t in range(self.epochs):
                        self.current_epoch = t
                        regressor_ca_train_loss = self.train_evaluate_regressor_epoch(train_data.c, None, train_data.a, None, 
                                                            self.kernel_ca, self.kernel_a, 
                                                            optimizer=self.optimizer_ca, mode='train',
                                                            type='ca')
                        regressor_ca_val_loss = self.train_evaluate_regressor_epoch(train_data.c, val_data.c, 
                                                                                    train_data.a, val_data.a, 
                                                                                    self.kernel_ca, self.kernel_a, 
                                                                                    mode='val', type='ca')

                        if self.early_stopper.early_stop(regressor_ca_val_loss, model=self.kernel_ca) or (t + 1) == self.epochs:
                            self.early_stopper.restore_best(self.kernel_ca)
                            regressor_ca_test_loss = self.train_evaluate_regressor_epoch(train_data.c, test_data.c, train_data.a, test_data.a, 
                                                                self.kernel_ca, self.kernel_a, mode='test', type='ca')
                            break
                    self.log({f"ridge_lambda_ca": self.kernel_ca.ridge_lambda.item()})
                    self.log({f"regression_ca_train_loss": regressor_ca_train_loss})
                    self.log({f"regression_ca_val_loss": regressor_ca_val_loss})
                    self.log({f"regression_ca_test_loss": regressor_ca_test_loss})
                    
                    # Reset the early stopper for the next sequence  
                    self.early_stopper.reset()
                # Storing training inputs for regressor ca
                self.kernel_ca.set_kernel_matrix(train_data.c.to(self.device))
            
            # For regressor cb, we always train it with online mode
            # Storing training labels for regressor cb
            self.kernel_b.set_kernel_matrix(train_data.b.to(self.device))
            if self.kernel_cb.is_trainable:
                # Train for kernel_cb
                for t in range(self.epochs):
                    self.current_epoch = t
                    regressor_cb_train_loss = self.train_evaluate_regressor_epoch(train_data.c, None, train_data.b, None, 
                                                        self.kernel_cb, self.kernel_b, 
                                                        optimizer=self.optimizer_cb, mode='train', type='cb')
                    regressor_cb_val_loss = self.train_evaluate_regressor_epoch(train_data.c, val_data.c, 
                                                                                train_data.b, val_data.b, 
                                                                                self.kernel_cb, self.kernel_b, 
                                                                                mode='val', type='cb')

                    if self.early_stopper.early_stop(regressor_cb_val_loss, model=self.kernel_cb) or (t + 1) == self.epochs:
                        self.early_stopper.restore_best(self.kernel_cb)
                        regressor_cb_test_loss = self.train_evaluate_regressor_epoch(train_data.c, test_data.c, train_data.b, test_data.b,
                                                            self.kernel_cb, self.kernel_b, mode='test', type='cb')
                        break
                self.log({f"ridge_lambda_cb": self.kernel_cb.ridge_lambda.item()})
                self.log({f"regression_cb_train_loss": regressor_cb_train_loss})
                self.log({f"regression_cb_val_loss": regressor_cb_val_loss})
                self.log({f"regression_cb_test_loss": regressor_cb_test_loss})

                # Reset the early stopper for the next sequence  
                self.early_stopper.reset()
            # Storing training inputs for regressor cb
            self.kernel_cb.set_kernel_matrix(train_data.c.to(self.device))

            # Update lambda and wealth
            if k >= self.T:
                if (self.kernel_c.is_trainable or self.betting_fraction_trainable):
                    for t in range(self.epochs):
                        self.current_epoch = t
                        self.train_evaluate_epoch(train_data=train_data, val_data=val_data, mode='train')
                        loss_val, _ = self.train_evaluate_epoch(train_data=train_data, val_data=val_data, mode='val')
                        # Check for early stopping or end of epochs
                        betting_model = {"betting_fraction": self.betting_fraction,
                                        "kernel_c": self.kernel_c,
                                        "gamma": self.gamma}
                        if self.early_stopper.early_stop(loss_val, model=betting_model) or (t + 1) == self.epochs:
                            self.early_stopper.restore_best(betting_model)
                            self.get_gamma(train_data=train_data, val_data=val_data, mode='val')
                            loss_test, vt = self.train_evaluate_epoch(train_data=train_data, val_data=test_data, mode='test')
                            wealth_update = - loss_test
                            if wealth_update < 0:
                                raise ValueError(f'Wealth update is negative: {wealth_update}')
                            wealth *= wealth_update
                            break
                else: 
                    self.get_gamma(train_data=train_data, val_data=val_data, mode='val')
                    loss_val, _ = self.train_evaluate_epoch(train_data=train_data, val_data=val_data, mode='val')
                    loss_test, vt = self.train_evaluate_epoch(train_data=train_data, val_data=test_data, mode='test')
                    wealth_update = - loss_test
                    if wealth_update < 0:
                        raise ValueError(f'Wealth update is negative: {wealth_update}')
                    wealth *= wealth_update
                    
                self.log({"wealth_update_val": - loss_val})
                self.log({"wealth_update_test": wealth_update})
                self.log({"gamma": self.gamma})
                self.log({"betting_fraction": torch.sigmoid(self.betting_fraction).item()})
                self.log({"test_Vt": vt})
                self.log({"wealth": wealth})

                # Reset the early stopper for the next sequence
                self.early_stopper.reset()

            train_data = CombinedDataset([train_data, val_data])
            val_data = test_data 
            self.log({"historical_sample_nums": len(train_data)})
            self.log({"all_sample_nums": len(train_data) + len(test_data)})
            # Log information if davt exceeds the threshold
            if wealth > (1. / self.alpha):
                reject_null = 1.0
                logging.info("Reject null at %f", wealth)
                self.log({"stopping_time": k})
                self.log({"reject_null": reject_null})
            else:
                self.log({"reject_null": reject_null})

        
