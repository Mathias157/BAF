
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pybalmorel import IncFile, Balmorel, MainResults
from GeneralHelperFunctions import get_combined_obj_value
import random  
import os

# ignore warnings 
import warnings
warnings.filterwarnings("ignore")


class ScenarioGenerator(nn.Module):
    def __init__(self, X_tensor=None, input_shape=(24, 72), latent_dim=32, device='cpu', lr=0.0005, seed=42, scale=True, selection_method="kcenter", scaler_type="standard",
                 const_tol_abs=1e-8, const_tol_rel=1e-6, logger=None):
        """
        Non-conditional AE Scenario Generator.

        - Detect features constant across ALL samples & timesteps; drop them from AE.
        - Scale ONLY dynamic features (StandardScaler or MinMaxScaler).
        - Loss is computed ONLY on scaled dynamic features.
        - On reconstruction & generation: inverse-scale dynamic and reinsert constants
          at their original indices with exact original values.
        """
        super().__init__()

        self.seq_len, self.input_dim = input_shape
        self.latent_dim = latent_dim
        self.lr = lr
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") \
                      if device == 'cpu' else torch.device(device)

        # logs
        self.total_loss_history = []
        self.pretrain_loss_history = []
        self.obj_values = []
        self.recon_loss_feedback = []
        self.reconstruction_loss_total = None

        # data holders
        self.new_scenarios = None
        self.new_scenarios_list = []
        self.X_original = None
        self.reconstruction_scenarios = None
        self.empty_df = None

        # config
        self.scale = scale
        self.selection_method = selection_method.lower()  # "kcenter" | "dpp" | "kmeans"
        self.scaler_type = scaler_type.lower()            # "standard" | "minmax"
        self.const_tol_abs = const_tol_abs
        self.const_tol_rel = const_tol_rel

        # constants/dynamic bookkeeping
        self.const_idx = []            # list[int]
        self.dyn_idx = []              # list[int]
        self.const_values = None       # np.ndarray (D_const,)
        self.dynamic_dim = 0
        self.flat_dyn = 0
        self.x_dtype = np.float32      # will be overwritten from data

        # scaler for dynamic only
        self.scaler = None

        # training tensor (scaled dynamic only)
        self.X_tensor_dyn_scaled = None
        self.X_tensor = X_tensor  # kept for API compatibility

        # seed
        self.seed = seed
        torch.manual_seed(self.seed)
        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        np.random.seed(self.seed)
        random.seed(self.seed)

        # placeholders; real layers built after dynamic_dim known
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()
        self.optimizer_ = None  # <- defer optimizer creation until layers are built
        
        self.logger = logger
        self.log = (self.logger.info if self.logger else print)

        self.to(self.device)
        
        
    # ---------------------- build model per dynamic size ----------------------
    def _build_models_for_dynamic(self):
        """(Re)build encoder/decoder sized to dynamic features and (re)create optimizer."""
        if self.dynamic_dim == 0:
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()
            self.flat_dyn = 0
        else:
            self.flat_dyn = self.seq_len * self.dynamic_dim
            self.encoder = nn.Sequential(
                nn.Linear(self.flat_dyn, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, self.latent_dim),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, self.flat_dyn),
            )
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        # (Re)create optimizer NOW that parameters exist
        self.optimizer_ = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr
        )

    # ---------------------- forward on dynamic (scaled) ----------------------
    def forward(self, x_dyn_scaled):
        """
        x_dyn_scaled: (B, T, D_dyn)  -- scaled dynamic features ONLY
        returns:
            recon_dyn_scaled: (B, T, D_dyn) scaled
            z: (B, latent_dim)
        """
        B = x_dyn_scaled.size(0)
        if self.dynamic_dim == 0:
            z = torch.zeros(B, self.latent_dim, device=self.device)
            return x_dyn_scaled, z

        x_flat = x_dyn_scaled.reshape(B, self.flat_dyn)
        z = self.encoder(x_flat)
        recon_dyn_scaled = self.decoder(z).reshape(B, self.seq_len, self.dynamic_dim)
        return recon_dyn_scaled, z

    # ---------------------- loss (scaled dynamic only) ----------------------
    def cae_loss(self, recon_dyn_scaled, x_dyn_scaled):
        if self.dynamic_dim == 0:
            return torch.zeros((), device=self.device, requires_grad=True)
        return nn.MSELoss()(recon_dyn_scaled, x_dyn_scaled)

    # ---------------------- selection utils (as in your code) ----------------------
    def kcenter_farthest_points(self, Z, k):
        rng = np.random.default_rng()
        N = Z.shape[0]
        first = rng.integers(N)
        selected = [first]
        d2_min = np.sum((Z - Z[first])**2, axis=1)
        for _ in range(1, k):
            nxt = int(np.argmax(d2_min))
            selected.append(nxt)
            d2_new = np.sum((Z - Z[nxt])**2, axis=1)
            d2_min = np.minimum(d2_min, d2_new)
        return np.array(selected, dtype=int)

    def rbf_kernel(self, Z, sigma=None):
        G = Z @ Z.T
        H = np.diag(G)
        D2 = H[:,None] + H[None,:] - 2*G
        if sigma is None:
            med = np.median(D2[np.triu_indices_from(D2, k=1)])
            sigma = np.sqrt(0.5*med + 1e-12)
        K = np.exp(-D2 / (2*sigma**2 + 1e-12))
        return K

    def dpp_greedy(self, K, k, rng=None):
        rng = np.random.RandomState(0) if rng is None else rng
        N = K.shape[0]
        selected = []
        gains = np.diag(K).copy()
        C = np.zeros((k, N))
        for i in range(k):
            j = np.argmax(gains)
            selected.append(j)
            if i == k-1:
                break
            if i == 0:
                C[i] = K[j] / np.sqrt(K[j, j] + 1e-12)
            else:
                proj = C[:i] @ K[j]
                norm2 = K[j, j] - np.sum(proj**2)
                norm2 = max(norm2, 1e-12)
                C[i] = (K[j] - C[:i].T @ proj) / np.sqrt(norm2)
            gains = np.diag(K) - np.sum(C[:i+1]**2, axis=0)
            gains[selected] = -np.inf
        return np.array(selected)

    # ---------------------- data prep ----------------------
    def generate_kday_blocks(self, df, k_days=1):
        df = df.copy()
        df.fillna(0, inplace=True)

        df['year'] = df['WY'].astype(int)
        df['week'] = df['SSS'].str.extract(r'S(\d+)').astype(int)
        df['hour'] = df['TTT'].str.extract(r'T(\d+)').astype(int)

        df['day_in_week'] = ((df['hour'] - 1) // 24) + 1
        df['block_in_week'] = ((df['day_in_week'] - 1) // k_days) + 1

        group = df.groupby(['year', 'week', 'block_in_week'])
        df = group.filter(lambda x: len(x) == 24 * k_days)

        feature_cols = df.columns.difference(['WY', 'SSS', 'TTT', 'year', 'week', 'hour', 'day_in_week', 'block_in_week'])
        if len(feature_cols) == 0:
            raise ValueError("No valid feature columns found. Ensure there are non-constant features.")

        self.log(f"Using {len(feature_cols)} features for {k_days}-day blocks.")
        self.empty_df = pd.DataFrame(columns=feature_cols)

        samples = []
        for (_, _, _), grp in df.groupby(['year', 'week', 'block_in_week']):
            X_block = grp[feature_cols].values
            samples.append(X_block)

        X = np.stack(samples)  # (N_blocks, k*24, num_features)
        return X

    def _detect_constants_and_scale_dynamic(self, X):
        """
        Detect global constants; fit scaler on dynamic; build AE; create training tensor.
        """
        N, T, D = X.shape
        self.x_dtype = X.dtype  # remember original dtype for exact reinsertion

        flat = X.reshape(-1, D)
        min_v = flat.min(axis=0)
        max_v = flat.max(axis=0)

        # tolerance-based constant detection (handles tiny jitter)
        ptp = max_v - min_v
        scale = np.maximum(np.abs(max_v), np.abs(min_v))
        thr = self.const_tol_abs + self.const_tol_rel * scale
        const_mask = ptp <= thr

        self.const_idx = np.where(const_mask)[0].tolist()
        self.dyn_idx = np.where(~const_mask)[0].tolist()
        self.dynamic_dim = len(self.dyn_idx)

        # Save exact constant values from the original tensor (first sample/time)
        if len(self.const_idx) > 0:
            self.const_values = X[0, 0, self.const_idx].astype(self.x_dtype, copy=True)
        else:
            self.const_values = np.array([], dtype=self.x_dtype)

        # Build AE for dynamic dims and (re)create optimizer
        self._build_models_for_dynamic()

        # Prepare scaled dynamic training tensor
        if self.dynamic_dim == 0:
            dyn_scaled = np.zeros((N, T, 0), dtype=np.float32)
            self.scaler = None
        else:
            X_dyn = X[:, :, self.dyn_idx]  # (N,T,D_dyn)
            if self.scale:
                if self.scaler_type == "standard":
                    self.scaler = StandardScaler()
                elif self.scaler_type == "minmax":
                    self.scaler = MinMaxScaler()
                else:
                    raise ValueError("scaler_type must be 'standard' or 'minmax'")
                X_dyn_scaled = self.scaler.fit_transform(
                    X_dyn.reshape(-1, self.dynamic_dim)
                ).reshape(N, T, self.dynamic_dim)
            else:
                self.scaler = None
                X_dyn_scaled = X_dyn.astype(np.float32)
            dyn_scaled = X_dyn_scaled.astype(np.float32)

        self.X_tensor_dyn_scaled = torch.tensor(dyn_scaled, dtype=torch.float32).to(self.device)
        self.X_tensor = self.X_tensor_dyn_scaled  # for compatibility with your loops

    def load_and_process_data(self, file_path, k_days=1):
        self.log("-" * 121)
        self.log("Loading and processing data from file:", file_path)

        df = pd.read_csv(file_path)
        X = self.generate_kday_blocks(df, k_days=k_days)
        self.X_original = X.copy()  # ORIGINAL units, full D

        # detect constants & prepare scaled dynamic training tensor
        self._detect_constants_and_scale_dynamic(X)

        # Report counts
        self.log(f"Original data shape: {self.X_original.shape}")
        self.log(f"Detected {len(self.const_idx)} constant features and {len(self.dyn_idx)} dynamic features.")
        if len(self.const_idx) > 0:
            self.log(f"Constant feature indices: {self.const_idx}")

    # ---------------------- training ----------------------
    def pretrain(self, epochs=20, batch_size=32):
        if self.X_tensor_dyn_scaled is None:
            raise ValueError("Data not prepared. Call load_and_process_data() first.")
        if self.optimizer_ is None:
            raise RuntimeError("Optimizer not initialized. Did _build_models_for_dynamic run?")

        dataset = TensorDataset(self.X_tensor_dyn_scaled)  # scaled dynamic only
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train()
        self.log("-" * 121)
        self.log("Pretraining the model...")
        self.log("*" * 74)
        self.log(f"Pretraining on {len(loader)} batches with batch size {batch_size} for {epochs} epochs.")

        for epoch in range(epochs):
            total_loss = 0.0
            for (x_dyn_batch,) in loader:
                x_dyn_batch = x_dyn_batch.to(self.device)
                self.optimizer_.zero_grad()
                recon_dyn, _ = self(x_dyn_batch)
                loss = self.cae_loss(recon_dyn, x_dyn_batch)  # loss on scaled dynamic only
                loss.backward()
                self.optimizer_.step()
                total_loss += loss.item()
            self.pretrain_loss_history.append(total_loss)
            self.log(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.6f}")

    # ---------------------- utils for reconstruction ----------------------
    def _inverse_scale_dynamic_np(self, dyn_scaled_np):
        """Inverse transform dynamic scaled -> original units (numpy)."""
        if self.scaler is None or self.dynamic_dim == 0:
            return dyn_scaled_np
        flat = dyn_scaled_np.reshape(-1, self.dynamic_dim)
        inv = self.scaler.inverse_transform(flat).reshape(dyn_scaled_np.shape)
        return inv

    def _reinsert_constants_np(self, dyn_np):
        """Given dynamic (original units) -> full (original units) with constants exactly restored."""
        N, T, _ = dyn_np.shape
        full = np.empty((N, T, self.input_dim), dtype=self.x_dtype)
        if self.dynamic_dim > 0:
            full[:, :, self.dyn_idx] = dyn_np.astype(self.x_dtype, copy=False)
        if len(self.const_idx) > 0:
            full[:, :, self.const_idx] = self.const_values.reshape(1, 1, -1).astype(self.x_dtype, copy=False)
        return full

    def _force_constants_exact(self, arr: np.ndarray) -> np.ndarray:
        """Safety net: overwrite constant columns with saved values."""
        if len(self.const_idx) > 0 and arr is not None:
            arr[:, :, self.const_idx] = self.const_values.reshape(1, 1, -1).astype(self.x_dtype, copy=False)
        return arr

    # ---------------------- scenario generation ----------------------
    def generate_scenario(self, batch_size=32, n_scenarios=10):
        if self.X_tensor_dyn_scaled is None:
            raise ValueError("Data not prepared. Call load_and_process_data() first.")
        if self.optimizer_ is None:
            raise RuntimeError("Optimizer not initialized. Did _build_models_for_dynamic run?")

        dataset = TensorDataset(self.X_tensor_dyn_scaled)  # scaled dynamic only
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train()
        if self.empty_df is not None:
            self.empty_df.drop(self.empty_df.index, inplace=True)
            self.empty_df.drop(columns=['scenario_id', 'time_step'], inplace=True, errors='ignore')

        all_z_batches = []
        all_recon_dyn_scaled_batches = []
        recon_loss_total = []

        for (x_dyn_batch,) in loader:
            x_dyn_batch = x_dyn_batch.to(self.device)
            self.optimizer_.zero_grad()

            recon_dyn_scaled, z = self(x_dyn_batch)
            recon_loss = self.cae_loss(recon_dyn_scaled, x_dyn_batch)  # dynamic scaled
            recon_loss_total.append(recon_loss)

            all_z_batches.append(z.detach().cpu().numpy())
            all_recon_dyn_scaled_batches.append(recon_dyn_scaled.detach().cpu().numpy())

        self.reconstruction_loss_total = torch.stack(recon_loss_total).sum()

        all_z_batches = np.concatenate(all_z_batches, axis=0)
        all_recon_dyn_scaled = np.concatenate(all_recon_dyn_scaled_batches, axis=0)  # (N,T,D_dyn)

        # Inverse-scale to original dynamic, then reinsert constants to get full D
        recon_dyn_inv = self._inverse_scale_dynamic_np(all_recon_dyn_scaled)
        full_recon = self._reinsert_constants_np(recon_dyn_inv)

        # Optional clamp ONLY dynamic dims (constants untouched)
        if self.dynamic_dim > 0:
            dyn = full_recon[:, :, self.dyn_idx]
            dyn[dyn < 0] = 0
            full_recon[:, :, self.dyn_idx] = dyn

        # Safety net: enforce constants exactly
        self.reconstruction_scenarios = self._force_constants_exact(full_recon)

        # ---- Select representative latents ----
        method = (self.selection_method or "kcenter").lower()

        if self.dynamic_dim == 0:
            # degenerate case: everything constant
            idx = np.arange(min(n_scenarios, len(all_z_batches)))
            centroids_z = torch.zeros((len(idx), self.latent_dim), dtype=torch.float32, device=self.device)
        elif method == "kmeans":
            kmeans_z = KMeans(n_clusters=n_scenarios, random_state=self.seed)
            kmeans_z.fit(all_z_batches)
            centroids_z = torch.tensor(kmeans_z.cluster_centers_, dtype=torch.float32).to(self.device)
        elif method == "kcenter":
            idx = self.kcenter_farthest_points(all_z_batches, k=n_scenarios)
            centroids_z = torch.tensor(all_z_batches[idx], dtype=torch.float32).to(self.device)
        elif method == "dpp":
            K = self.rbf_kernel(all_z_batches, sigma=None)
            idx = self.dpp_greedy(K, k=n_scenarios, rng=np.random.RandomState(self.seed))
            centroids_z = torch.tensor(all_z_batches[idx], dtype=torch.float32).to(self.device)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}. Choose from 'kmeans', 'kcenter', or 'dpp'.")

        # ---- Decode new scenarios (dynamic scaled) -> inverse-scale -> reinsert constants ----
        if self.dynamic_dim == 0:
            new_full = np.empty((n_scenarios, self.seq_len, self.input_dim), dtype=self.x_dtype)
            if len(self.const_idx) > 0:
                new_full[:, :, self.const_idx] = self.const_values.reshape(1, 1, -1).astype(self.x_dtype, copy=False)
        else:
            new_dyn_scaled = self.decoder(centroids_z).view(-1, self.seq_len, self.dynamic_dim).cpu().detach().numpy()
            new_dyn_inv = self._inverse_scale_dynamic_np(new_dyn_scaled).astype(self.x_dtype, copy=False)
            new_full = self._reinsert_constants_np(new_dyn_inv)

            # optional clamp ONLY dynamic dims
            dyn = new_full[:, :, self.dyn_idx]
            dyn[dyn < 0] = 0
            new_full[:, :, self.dyn_idx] = dyn

        # Safety net: enforce constants exactly
        self.new_scenarios = self._force_constants_exact(new_full)
        self.new_scenarios_list.append(self.new_scenarios)

        # Tidy DataFrame (if empty_df was initialized in generate_kday_blocks)
        if self.empty_df is not None:
            new_rows = []
            for i in range(self.new_scenarios.shape[0]):
                for j in range(self.new_scenarios.shape[1]):
                    new_row = pd.Series(self.new_scenarios[i, j, :], index=self.empty_df.columns)
                    new_row['scenario_id'] = i + 1
                    new_row['time_step'] = j + 1
                    new_rows.append(new_row)
            new_df = pd.DataFrame(new_rows)
            self.empty_df = pd.concat([self.empty_df, new_df], ignore_index=True)
            self.empty_df['scenario_id'] = self.empty_df['scenario_id'].astype(int)
            self.empty_df['time_step'] = self.empty_df['time_step'].astype(int)
            cols = ['scenario_id', 'time_step'] + [c for c in self.empty_df.columns if c not in ['scenario_id', 'time_step']]
            self.empty_df = self.empty_df[cols]

        return self.new_scenarios, (self.empty_df if self.empty_df is not None else None)

    # ---------------------- feedback update ----------------------
    def update(self, obj_value, epoch=None):
        if epoch == 0:
            self.log("-" * 121)
            self.log("Updating model with the feedback...")
            self.log("*" * 74)

        self.obj_values.append(obj_value)
        self.recon_loss_feedback.append(self.reconstruction_loss_total.item() if self.reconstruction_loss_total is not None else np.nan)

        policy_loss = obj_value
        total_batch_loss = (self.reconstruction_loss_total if self.reconstruction_loss_total is not None else 0.0) + policy_loss

        self.optimizer_.zero_grad()
        total_batch_loss.backward()
        self.optimizer_.step()

        self.total_loss_history.append(total_batch_loss.item() if hasattr(total_batch_loss, "item") else float(total_batch_loss))
        prefix = f'Epoch {epoch+1} - ' if epoch is not None else ''
        rec = self.reconstruction_loss_total.item() if self.reconstruction_loss_total is not None else float('nan')
        pol = policy_loss.item() if hasattr(policy_loss, "item") else float(policy_loss)
        tot = self.total_loss_history[-1]
        self.log(f"{prefix}Total Loss: {tot:.6f} | Reconstruction Loss: {rec:.6f} | Policy Loss: {pol:.6f}")

    # ---------------------- plots ----------------------
    def plot_loss_history(self):
        plt.plot(self.pretrain_loss_history, label='Pretrain Loss')
        plt.title("Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.recon_loss_feedback, label='Reconstruction Loss')
        plt.title("Reconstruction History in Feedback Loop")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.plot(self.total_loss_history, label='Total Loss')
        plt.plot(self.obj_values, label='Objective Values', linestyle='--')
        plt.title("Total Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_scenarios(self, selecting_feature_index=0):
        if self.new_scenarios is None:
            raise ValueError("No new scenarios generated. Please call generate_scenario() first.")

        plt.figure(figsize=(12, 6))
        for i in range(self.X_original.shape[0]):
            plt.plot(self.X_original[i, :, selecting_feature_index], alpha=0.1)

        for i in range(self.new_scenarios.shape[0]):
            plt.plot(self.new_scenarios[i, :, selecting_feature_index], alpha=0.7)

        plt.title("Generated Scenario vs Original Scenario")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # ---------------------- save / load ----------------------
    def save_model(self, file_path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'encoder_state_dict': self.encoder.state_dict() if not isinstance(self.encoder, nn.Identity) else None,
            'decoder_state_dict': self.decoder.state_dict() if not isinstance(self.decoder, nn.Identity) else None,
            'total_loss_history': self.total_loss_history,
            'pretrain_loss_history': self.pretrain_loss_history,
            'obj_values': self.obj_values,
            'new_scenarios_list': self.new_scenarios_list,
            'X_original': self.X_original,
            'X_tensor_dyn_scaled': self.X_tensor_dyn_scaled,
            'reconstruction_loss_total': self.reconstruction_loss_total,
            'empty_df': self.empty_df,
            'reconstruction_scenarios': self.reconstruction_scenarios,
            'recon_loss_feedback': self.recon_loss_feedback,
            # constants/dynamic info
            'const_idx': self.const_idx,
            'dyn_idx': self.dyn_idx,
            'const_values': self.const_values,
            'dynamic_dim': self.dynamic_dim,
            'x_dtype': str(self.x_dtype),
            # scaler & config
            'scaler': self.scaler,
            'scaler_type': self.scaler_type,
            'scale': self.scale,
            'selection_method': self.selection_method,
            'input_shape': (self.seq_len, self.input_dim),
            'latent_dim': self.latent_dim,
            'lr': self.lr,
            'const_tol_abs': self.const_tol_abs,
            'const_tol_rel': self.const_tol_rel,
        }, file_path)

        print("-" * 121)
        print(f'Model and additional attributes saved to {file_path}')

    def load_model(self, file_path):
        checkpoint = torch.load(file_path, weights_only=False, map_location=self.device)

        # restore config
        self.scale = checkpoint.get('scale', self.scale)
        self.scaler_type = checkpoint.get('scaler_type', self.scaler_type)
        self.selection_method = checkpoint.get('selection_method', self.selection_method)
        self.lr = checkpoint.get('lr', self.lr)
        self.const_tol_abs = checkpoint.get('const_tol_abs', self.const_tol_abs)
        self.const_tol_rel = checkpoint.get('const_tol_rel', self.const_tol_rel)

        # restore histories
        self.total_loss_history = checkpoint.get('total_loss_history', [])
        self.pretrain_loss_history = checkpoint.get('pretrain_loss_history', [])
        self.obj_values = checkpoint.get('obj_values', [])
        self.new_scenarios_list = checkpoint.get('new_scenarios_list', [])
        self.X_original = checkpoint.get('X_original', None)
        self.X_tensor_dyn_scaled = checkpoint.get('X_tensor_dyn_scaled', None)
        if isinstance(self.X_tensor_dyn_scaled, torch.Tensor):
            self.X_tensor_dyn_scaled = self.X_tensor_dyn_scaled.to(self.device)
        self.reconstruction_loss_total = checkpoint.get('reconstruction_loss_total', None)
        self.empty_df = checkpoint.get('empty_df', None)
        self.reconstruction_scenarios = checkpoint.get('reconstruction_scenarios', None)
        self.recon_loss_feedback = checkpoint.get('recon_loss_feedback', [])

        # restore constants/dynamics
        self.const_idx = checkpoint.get('const_idx', [])
        self.dyn_idx = checkpoint.get('dyn_idx', [])
        self.const_values = checkpoint.get('const_values', None)
        self.dynamic_dim = checkpoint.get('dynamic_dim', len(self.dyn_idx))
        x_dtype_str = checkpoint.get('x_dtype', 'float32')
        self.x_dtype = np.dtype(x_dtype_str)

        # scaler
        self.scaler = checkpoint.get('scaler', None)

        # rebuild encoder/decoder for dynamic dims and create optimizer
        self._build_models_for_dynamic()

        # load encoder/decoder weights if present
        enc_sd = checkpoint.get('encoder_state_dict', None)
        dec_sd = checkpoint.get('decoder_state_dict', None)
        if enc_sd is not None and not isinstance(self.encoder, nn.Identity):
            self.encoder.load_state_dict(enc_sd, strict=False)
        if dec_sd is not None and not isinstance(self.decoder, nn.Identity):
            self.decoder.load_state_dict(dec_sd, strict=False)

        self.to(self.device)
        print(f'Model and additional attributes loaded from {file_path}')


    def plot_reconstruction_scenarios(self, features_to_plot, sample_index=0):
        num_features = len(features_to_plot)
        num_cols = 3
        num_rows = (num_features + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
        axes = axes.flatten()
        for i, feature in enumerate(features_to_plot):
            if i >= len(axes):
                break
            axes[i].plot(self.X_original[sample_index, :, feature], label='Original', alpha=0.5)
            axes[i].plot(self.reconstruction_scenarios[sample_index, :, feature], label='Reconstructed', alpha=0.7)
            axes[i].set_title(f'Feature {feature}')
            axes[i].set_xlabel('Time Step')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True)

        plt.tight_layout()
        plt.show()
        print("Reconstruction scenarios plotted for sample index:", sample_index)
        print("Features plotted:", features_to_plot)
        print("Original data shape:", self.X_original.shape)
        print("Reconstruction scenarios shape:", self.reconstruction_scenarios.shape)
        print("Sample index:", sample_index)
        print("Number of features:", num_features)
        
def convert_to_incfiles(new_scenarios_df: pd.DataFrame,
                        scenario: str,
                        scenario_folder: str,
                        balmorel_model_path: str = 'Balmorel',
                        gams_system_directory: str | None = None,
                        overwrite_data_load: bool = False):
    
    
    # Get parameters and temporal resolution
    parameters = new_scenarios_df.columns[2:] # pop scenario id and time_step
    balmorel_season_time_index = [f"S{row['scenario_id']:02d} . T{row['time_step']:03d}" for i,row  in new_scenarios_df.loc[:, ['scenario_id', 'time_step']].iterrows()]
    balmorel_season_index = [f"S{season:02d}" for season in new_scenarios_df.loc[:, 'scenario_id'].values]
    balmorel_term_index = [f"T{term:03d}" for term in new_scenarios_df.loc[:, 'time_step'].values]
    
    # Load Balmorel input data for descriptions and set names
    model = Balmorel(balmorel_model_path, gams_system_directory=gams_system_directory)
    model.load_incfiles(scenario, overwrite=overwrite_data_load)
    
    placeholder_parameter = ''
    for parameter in parameters:

        # Get elements and parameter_name
        elements = parameter.split('|')
        parameter_name = elements.pop(0)
        
        # Skip if we already processed this parameter
        if placeholder_parameter == parameter_name:
            continue
        else:
            placeholder_parameter = parameter_name
        
        # Find all values of this parameter
        idx=new_scenarios_df.columns.str.find(parameter_name + '|') == 0
        
        # Get sets, values and explanation
        sets = model.input_data[scenario][parameter_name].domains_as_strings
        text = model.input_data[scenario][parameter_name].text
        table_or_param = 'PARAMETER' if (len(sets) == 1) else 'TABLE'
        if table_or_param == 'PARAMETER':
            prefix = f"""{table_or_param} {parameter_name}({','.join(sets)}) "{text}" \n/\n"""
            suffix = '\n/;'
        else:
            prefix = f"""{table_or_param} {parameter_name}({','.join(sets)}) "{text}" \n"""
            suffix = '\n;'
        
        df = new_scenarios_df.loc[:, idx]
        df.columns = df.columns.str.replace(parameter_name+'|', '', n=1).str.replace('|', ' . ')

        if 'SSS' in sets and 'TTT' in sets:
            df.index = balmorel_season_time_index   
            df.index.name = 'ST'
            df = df.pivot_table(index='ST', aggfunc='mean')
            df.index.name = ''
            df = df.T
            
        elif 'SSS' in sets:
            df.index = balmorel_season_index
            df.index.name = 'S'
            df = df.pivot_table(index='S', aggfunc='mean')
            df.index.name = ''
            df = df.T
            
        else:
            df = df.mean().T
            if table_or_param != 'PARAMETER':
                prefix = prefix.replace('TABLE', 'PARAMETER')
                prefix += "/\n"
                suffix = "\n/;"
        
        if parameter_name != 'HYDROGEN_DH2':
            file = IncFile(
                name=parameter_name,
                prefix=prefix,
                body=df.to_string(),
                suffix=suffix,
                path=balmorel_model_path + f'/{scenario_folder}/capexp_data'
            )
        else:
            file = IncFile(
                name='HYDROGEN_DH2',
                prefix='',
                suffix='',
                body="\n".join([f"HYDROGEN_DH2('{year}', '{region}') = HYDROGEN_DH2('{year}', '{region}') + {df.loc[f'{year} . {region}']};" for year, region in df.index.str.split(' . ', expand=True)]),
                path=balmorel_model_path + f'/{scenario_folder}/capexp_data'
            )
            
        if parameter_name == 'DE_VAR_T':
            file.suffix += "\n* Flat profiles for industry and datacenter"
            file.suffix += "\nDE_VAR_T(RRR,'PII',SSS,TTT)=1;"
            file.suffix += "\nDE_VAR_T(RRR,'DATACENTER',SSS,TTT)=1;"
            
        file.save()
        
    # Define temporal resolution
    IncFile(
        name='S',
        prefix="SET S(SSS)  'Seasons in the simulation'\n/\n",
        body=', '.join(np.unique(balmorel_season_index)),
        suffix='\n/;',
        path=balmorel_model_path + f'/{scenario_folder}/capexp_data'
    ).save()
    IncFile(
        name='T',
        prefix="SET T(TTT)  'Time periods within a season in the simulation'\n/\n",
        body=','.join(np.unique(balmorel_term_index)),
        suffix='\n/;',
        path=balmorel_model_path + f'/{scenario_folder}/capexp_data'
    ).save()

        
def pretrain(epochs: int, days: int = 1, n_scenarios: int = 4, latent_dim: int = 64, batch_size: int = 256, learning_rate: float = 5e-4, seed: int = 42, n_features=72, logger=None, scenario_folder: str = 'operun'):
    """
    Pretrain the scenario generator.
    
    Parameters
    ----------
    epochs : int
        Number of pretraining epochs.
    days : int, default 1
        Length of each block in days (input_shape uses days*24).
    n_scenarios : int, default 4
        How many scenarios to generate after pretraining.
    latent_dim : int, default 64
        Latent space dimensionality for the ScenarioGenerator.
    batch_size : int, default 256
        Batch size used in pretraining and generation.
    learning_rate : float, default 1e-3
        Optimizer learning rate for the ScenarioGenerator.
    seed : int, default 42
        Random seed for reproducibility.
    n_features : int, default 72
        Amount of features in training data, 72 for all weather years for small-system, 33 for one weather year and no yearly data.
    logger : bool, default None
        Whether to log or print (None = False)
    scenario_folder : str, default operun
        Which scenario folder to run
    """
    
    log = (logger.info if logger else print)


    model = ScenarioGenerator(input_shape=(days*24, n_features), latent_dim=latent_dim, lr=learning_rate, seed=seed, logger=logger, selection_method="kmeans", scaler_type="standard")
    model.load_and_process_data('Pre-Processing/Output/genmodel_input.csv', k_days=days)
    
    log(f"Pretraining for {epochs} epochs, batch_size={batch_size}, lr={learning_rate}, seed={seed}")
    
    model.pretrain(epochs=epochs, batch_size=batch_size)
    
    log("Pretraining done; generating initial scenarios")
    
    # create new incfiles
    new_scenarios, new_scenarios_df = model.generate_scenario(batch_size=batch_size, n_scenarios=n_scenarios)
    
    log("Converting scenarios to INC files")
    
    convert_to_incfiles(new_scenarios_df, 'base', scenario_folder, gams_system_directory=os.getenv('GAMS_DIR','/appl/gams/47.6.0'), overwrite_data_load=True)

    # model.save_model(f'Pre-Processing/Output/{scenario}_model.pth')

    return model

def train(model: ScenarioGenerator, scenario: str, epoch_string: str, n_scenarios: int=4, batch_size: int = 256, logger=None, scenario_folder: str = 'operun'):
    
    # Get the objective value
    results = MainResults([f'MainResults_{scenario}_{runtype}_E{epoch_string}.gdx' for runtype in ['capacity', 'dispatch']],
                          paths=[f'Balmorel/{scenario_folder}/model'])
    obj_value, capital_costs, operational_costs = get_combined_obj_value(results, return_capex_opex_dfs=True)

    log = (logger.info if logger else print)
    
    log("read objective values from capacity and dispatch runs")
    log(f"Capital costs:\n%s"%capital_costs.to_string())
    log(f"Operational costs:\n%s"%operational_costs.to_string())
    log(f'Loss value: {obj_value} M€')

    # update the model with the objective value
    model.update(obj_value, epoch=int(epoch_string))
    
    # create new incfiles
    new_scenarios, new_scenarios_df = model.generate_scenario(batch_size=batch_size, n_scenarios=n_scenarios)
    
    log("running convert_to_incfiles")
    convert_to_incfiles(new_scenarios_df, 'base', scenario_folder, gams_system_directory=os.getenv('GAMS_DIR','/appl/gams/47.6.0'))
    
    return model