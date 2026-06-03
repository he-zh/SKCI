import torch
import torch.nn as nn
import numpy as np
import logging
import wandb
from models import MMDEMLP
from models.baseline_models import MMDECNN
from utils.data_merging import CombinedDataset
from utils.gmmn_training import mmd_loss_conditional
from trainer.ecrt_trainer import ECRTTrainer

class EC2STTrainer(ECRTTrainer):
    """
    EC2ST (E-Value Computation for Sequential Testing) Trainer for sequential testing.

    Uses MMDEMLP model to compute e-values for sequential hypothesis testing.
    The model compares z=(a,b,c) with tau_z=(tilde_a,b,c) where tilde_a is sampled
    from the null distribution.
    """

    def __init__(self, cfg, datagen, device):
        """
        Initialize the EC2ST Trainer.

        Args:
        - cfg: Configuration object containing trainer settings
        - datagen: Data generator object
        - device: Device (CPU/GPU) for training
        """
        super().__init__(cfg, datagen, device)

        # EC2ST specific parameters
        self.ec2st_model_type = cfg.get('ec2st_model_type', 'mmdemlp')
        self.ec2st_model_lr = cfg.get('ec2st_model_lr', 0.01)
        self.ec2st_model_weight_decay = cfg.get('ec2st_model_weight_decay', 1e-4)

        # MMDEMLP specific parameters
        self.ec2st_model_hidden_dims = cfg.get('ec2st_model_hidden_dims', [128])

        # MMDECNN specific parameters (for image data with mixed inputs)
        self.ec2st_b_channels = cfg.get('ec2st_b_channels', 1)  # Channels for image b
        self.ec2st_c_channels = cfg.get('ec2st_c_channels', 1)  # Channels for image c
        self.ec2st_b_hidden_channels = cfg.get('ec2st_b_hidden_channels', [32, 64])  # CNN channels for b encoder
        self.ec2st_c_hidden_channels = cfg.get('ec2st_c_hidden_channels', [32, 64, 128])  # CNN channels for c encoder
        self.ec2st_b_latent_dim = cfg.get('ec2st_b_latent_dim', 64)  # Latent dim for b encoder
        self.ec2st_c_latent_dim = cfg.get('ec2st_c_latent_dim', 128)  # Latent dim for c encoder
        self.ec2st_model_fc_hidden_dims = cfg.get('ec2st_model_fc_hidden_dims', [128])

        self.ec2st_model_dropout = cfg.get('ec2st_model_dropout', 0.1)
        self.ec2st_model_layer_norm = cfg.get('ec2st_model_layer_norm', True)

        # Models (will be initialized when data is loaded)
        self.ec2st_model = None

        # Optimizers (will be initialized after models)
        self.optimizer_ec2st_model = None

    def l1_regularization(self):
        """Compute L1 regularization loss."""
        l1_loss = torch.tensor(0., requires_grad=True, device=self.device)
        for name, param in self.ec2st_model.named_parameters():
            if 'bias' not in name:
                l1_loss = l1_loss + torch.norm(param, p=1)
        return l1_loss

    def _initialize_models(self, a_dim, b_dim, c_dim):
        """
        Initialize models based on data dimensions.

        Args:
        - data: Dataset to infer dimensions from
        """
        self.a_dim = a_dim
        self.b_dim = b_dim
        self.c_dim = c_dim
        abc_dim = a_dim + b_dim + c_dim

        # Create model_a for sampling tilde_a
        self._initialize_model_a(c_dim, a_dim)

        # Create EC2ST MMDEMLP model
        if self.ec2st_model_type == 'mmdemlp':
            self.ec2st_model = MMDEMLP(
                input_size=abc_dim,
                hidden_layer_size=self.ec2st_model_hidden_dims,
                output_size=1,
                layer_norm=self.ec2st_model_layer_norm,
                drop_out=self.ec2st_model_dropout > 0,
                drop_out_p=self.ec2st_model_dropout,
                flatten=True,
                return_logits=True
            ).to(self.device)
        elif self.ec2st_model_type == 'cnn':
            self.ec2st_model = MMDECNN(
                a_dim=a_dim,
                b_dim=b_dim,
                c_dim=c_dim,
                b_channels=self.ec2st_b_channels,
                c_channels=self.ec2st_c_channels,
                b_hidden_channels=self.ec2st_b_hidden_channels,
                c_hidden_channels=self.ec2st_c_hidden_channels,
                b_latent_dim=self.ec2st_b_latent_dim,
                c_latent_dim=self.ec2st_c_latent_dim,
                fc_hidden_size=self.ec2st_model_fc_hidden_dims,
                output_size=1,
                layer_norm=self.ec2st_model_layer_norm,
                drop_out=self.ec2st_model_dropout > 0,
                drop_out_p=self.ec2st_model_dropout,
                return_logits=True
            ).to(self.device)

        # Initialize optimizers
        self._initialize_model_a_optimizer()

        self.optimizer_ec2st_model = torch.optim.Adam(
            self.ec2st_model.parameters(),
            lr=self.ec2st_model_lr,
            weight_decay=self.ec2st_model_weight_decay
        )

        logging.info(f"Initialized models with dimensions: a={self.a_dim}, b={self.b_dim}, c={self.c_dim}")

    
    def _train_evaluate_ec2st_model_epoch(self, data, mode="train"):
        """
        Train or evaluate the EC2ST model for one epoch using minibatch training.
        The utilization of minibatch is for consistency with the original implementation,
        although EC2ST does not strictly require it.

        Args:
        - data: Dataset
        - mode: "train" or "val"

        Returns:
        - loss: Average loss value across all batches
        - ec2st: E-value for this epoch (product of batch e-values)
        """
        n_samples = len(data)
        n_batches = (n_samples + self.bs - 1) // self.bs

        total_loss = 0.0
        log_ec2st_sum = 0.0  # Track log sum for numerical stability

        if mode == "train":
            self.ec2st_model.train()
        else:
            self.ec2st_model.eval()

        # Process data in minibatches
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.bs
            end_idx = min((batch_idx + 1) * self.bs, n_samples)

            # Get batch data
            a = data.a[start_idx:end_idx].to(self.device)
            b = data.b[start_idx:end_idx].to(self.device)
            c = data.c[start_idx:end_idx].to(self.device)

            # Get true conditional mean if available (for oracle mode)
            a_m = None
            if self.model_x_mode == 'oracle' and hasattr(data, 'a_m'):
                a_m = data.a_m[start_idx:end_idx].to(self.device)
            elif self.model_x_mode == 'oracle' and not hasattr(data, 'a_m'):
                raise ValueError("In oracle mode, data must have attribute a_m (true conditional mean).")

            # Flatten images if needed (for concatenation)
            a_flat = a.view(a.shape[0], -1) if a.dim() > 2 else a
            b_flat = b.view(b.shape[0], -1) if b.dim() > 2 else b
            c_flat = c.view(c.shape[0], -1) if c.dim() > 2 else c

            # Ensure a is 2D
            if a_flat.dim() == 1:
                a_flat = a_flat.unsqueeze(-1)

            z = torch.cat([a_flat, b_flat, c_flat], dim=-1)

            if mode == "train":
                self.optimizer_ec2st_model.zero_grad()

                # Sample tilde_a (uses a_m if available for oracle mode)
                tilde_a = self.sample_tilde_a_single(c, a, a_m)
                tilde_a_flat = tilde_a.view(tilde_a.shape[0], -1) if tilde_a.dim() > 2 else tilde_a
                if tilde_a_flat.dim() == 1:
                    tilde_a_flat = tilde_a_flat.unsqueeze(-1)
                tau_z = torch.cat([tilde_a_flat, b_flat, c_flat], dim=-1)

                # ec2st forward: g(z), g(tau_z)
                g_z, g_tau_z = self.ec2st_model(z, tau_z)
                all_outputs = torch.cat([g_z, g_tau_z], dim=0)
                all_labels = torch.cat([torch.ones_like(g_z), torch.zeros_like(g_tau_z)], dim=0).to(self.device)
                # Compute loss (binary cross-entropy with logits)
                loss = nn.functional.binary_cross_entropy_with_logits(all_outputs, all_labels, reduction='sum')

                loss.backward()
                self.optimizer_ec2st_model.step()

                total_loss += loss.item()
            else:
                with torch.no_grad():
                    tilde_a = self.sample_tilde_a_single(c, a, a_m)
                    tilde_a_flat = tilde_a.view(tilde_a.shape[0], -1) if tilde_a.dim() > 2 else tilde_a
                    if tilde_a_flat.dim() == 1:
                        tilde_a_flat = tilde_a_flat.unsqueeze(-1)
                    tau_z = torch.cat([tilde_a_flat, b_flat, c_flat], dim=-1)
                    g_z, g_tau_z = self.ec2st_model(z, tau_z)
                    all_outputs = torch.cat([g_z, g_tau_z], dim=0)
                    all_labels = torch.cat([torch.ones_like(g_z), torch.zeros_like(g_tau_z)], dim=0).to(self.device)
                    # Compute loss (binary cross-entropy with logits)
                    loss = nn.functional.binary_cross_entropy_with_logits(all_outputs, all_labels, reduction='sum')

                    total_loss += loss.item()
                    log_ec2st_sum += self.log_e_c2st(all_outputs, all_labels)

        # Compute average loss and overall e-value
        avg_loss = total_loss / n_samples
        ec2st = torch.exp(torch.tensor(log_ec2st_sum)).item()

        return avg_loss, ec2st

    def log_e_c2st(self, outputs, labels):
        """
        Compute e-value for a batch based on model outputs g(z) and g(tau_z).

        Args:
        - g_z: Model output for true samples z
        - g_tau_z: Model output for null samples tau_z

        Returns:
        - batch_ec2st: E-value for this batch
        """ 
        n = labels.shape[0]
        n_h1 = labels.sum().item()  # Number of samples from true distribution (z)
        n_h0 = n - n_h1  # Number of samples from null distribution (tau_z)
        emp_freq_class0 = n_h0 / n
        emp_freq_class1 = n_h1 / n

        # H1: Probabilities under empirical model (using train data)
        f = torch.nn.Sigmoid()  # Convert logits to probabilities
        
        p = f(outputs)
        ratio1 = p.detach() / emp_freq_class1
        ratio0 = (1-p.detach()) / emp_freq_class0
        log_eval = torch.sum(
            labels * torch.log(ratio1) + (1 - labels) * torch.log(ratio0))

        return log_eval
    
    def train(self):
        """
        Train the EC2ST model for sequential testing across multiple sequences.
        """
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Load initial data
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

        # Initialize e-value tracking
        ec2sts = []
        aggregated_ec2st = 1.0
        reject_null = 0.0

        for k in range(self.seqs):
            self.current_seq = k
            test_data = self.load_data(self.seed + k + 2)

            # Train model_a in online mode
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

            # Train EC2ST model
            logging.info(f"Training ec2st_model model on sequence {k}...")
            for t in range(self.epochs):
                self.current_epoch = t
                train_loss, _ = self._train_evaluate_ec2st_model_epoch(train_data, mode='train')
                val_loss, _ = self._train_evaluate_ec2st_model_epoch(val_data, mode='val')

                # Early stopping
                if self.early_stopper.early_stop(val_loss, model=self.ec2st_model) or (t + 1) == self.epochs:
                    self.early_stopper.restore_best(self.ec2st_model)
                    self.early_stopper.reset()
                    break

            self.log({'ec2st_train_loss': train_loss})
            self.log({'ec2st_val_loss': val_loss})

            # Compute e-value on test data
            _, conditional_ec2st = self._train_evaluate_ec2st_model_epoch(test_data, mode='test')

            ec2sts.append(conditional_ec2st)

            # Aggregate e-values (product from position T onwards)
            if k >= self.T:
                aggregated_ec2st = np.prod(np.array(ec2sts[self.T:]))
                self.log({"wealth": aggregated_ec2st})

                # Check for rejection
                if aggregated_ec2st > (1. / self.alpha):
                    reject_null = 1.0
                    logging.info(f"Reject null at sequence {k}, e-value: {aggregated_ec2st}")
                    self.log({"stopping_time": k})
                    self.log({"reject_null": reject_null})
                else:
                    self.log({"reject_null": reject_null})

            # Update datasets for next sequence
            train_data = CombinedDataset([train_data, val_data])
            val_data = test_data

            self.log({'historical_sample_nums': len(train_data)})
            self.log({'all_sample_nums': len(train_data) + len(test_data)})

        logging.info(f"Training complete. Final wealth: {aggregated_ec2st}")
