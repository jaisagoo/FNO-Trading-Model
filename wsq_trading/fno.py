"""Fourier Neural Operator for H(t) prediction, with its training loop and dataset."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from wsq_trading import config
from wsq_trading.utils import get_logger, timer

log = get_logger(__name__)


class SpectralConv1d(nn.Module):
    """1-D Fourier integral operator layer (core FNO primitive).

    Applies a learnable linear map in the truncated Fourier domain:

        (W * x_hat)[:n_modes] then zero-padded back to the original length.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    out_channels : int
        Number of output feature channels.
    n_modes : int
        Number of Fourier modes to retain (must be <= seq_len // 2 + 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int,
    ) -> None:
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.n_modes      = n_modes

        # Complex weight tensor: shape (in_channels, out_channels, n_modes)
        # Initialised with scale 1/sqrt(in_channels * out_channels) following
        # the original FNO paper (Li et al. 2020).
        scale = 1.0 / math.sqrt(in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, n_modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution.

        Parameters
        ----------
        x : Tensor, shape (B, C_in, L)
        """
        B, C_in, L = x.shape
        n_freq = L // 2 + 1  # length of rfft output

        # Forward real FFT along last dimension
        x_ft = torch.fft.rfft(x, n=L, dim=-1)   # (B, C_in, n_freq)

        # Truncate to n_modes (or all freq if n_modes >= n_freq)
        n_modes = min(self.n_modes, n_freq)

        # Complex multiply: einsum over in_channels
        # x_ft[:, :, :n_modes]  (B, C_in, n_modes)
        # self.weights            (C_in, C_out, n_modes)
        out_ft = torch.einsum(
            "bin,ion->bon",
            x_ft[:, :, :n_modes],   # (B, C_in, n_modes)
            self.weights[:, :, :n_modes],  # (C_in, C_out, n_modes)
        )  # (B, C_out, n_modes)

        # Zero-pad back to full rfft length
        out_ft_full = torch.zeros(B, self.out_channels, n_freq,
                                  dtype=torch.cfloat, device=x.device)
        out_ft_full[:, :, :n_modes] = out_ft

        # Inverse real FFT
        return torch.fft.irfft(out_ft_full, n=L, dim=-1)  # (B, C_out, L)


# FNO block (one layer of the operator)

class FNOBlock(nn.Module):
    """Single FNO layer: spectral path + local bypass + activation.

    Parameters
    ----------
    width : int
        Number of channels in and out (both paths share the same width).
    n_modes : int
        Number of Fourier modes for the spectral convolution.
    """

    def __init__(self, width: int, n_modes: int) -> None:
        super().__init__()
        self.spectral = SpectralConv1d(width, width, n_modes)
        self.bypass   = nn.Conv1d(width, width, kernel_size=1)
        self.norm     = nn.InstanceNorm1d(width, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Parameters
        ----------
        x : Tensor, shape (B, width, L)
        """
        return F.gelu(self.norm(self.spectral(x) + self.bypass(x)))


# Full FNO1d model

class FNO1d(nn.Module):
    """Fourier Neural Operator for Hurst exponent prediction.

    Takes a window of 1-D return features and predicts a scalar H in (0, 1).

    Parameters
    ----------
    in_channels : int
        Number of input feature channels per time step.
        Minimum 1 (log-returns only); more channels (e.g. realised vol)
        give the model richer input.
    width : int
        Hidden channel width.  Default: ``config.FNO_WIDTH``.
    n_modes : int
        Fourier modes per spectral layer.  Default: ``config.FNO_MODES``.
    n_layers : int
        Number of FNO blocks.  Default: ``config.FNO_LAYERS``.
    mlp_hidden : int
        Hidden size of the projection MLP head.  Default: ``width * 2``.
    dropout : float
        Dropout applied inside the projection MLP.  Default: 0.1.
    """

    def __init__(
        self,
        in_channels: int = 1,
        width: Optional[int] = None,
        n_modes: Optional[int] = None,
        n_layers: Optional[int] = None,
        mlp_hidden: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.width     = width     or config.FNO_WIDTH
        self.n_modes   = n_modes   or config.FNO_MODES
        self.n_layers  = n_layers  or config.FNO_LAYERS
        mlp_hidden     = mlp_hidden or self.width * 2

        # 1. Lifting layer: pointwise linear C_in -> width
        self.lifting = nn.Conv1d(in_channels, self.width, kernel_size=1)

        # 2. FNO blocks
        self.blocks = nn.ModuleList(
            [FNOBlock(self.width, self.n_modes) for _ in range(self.n_layers)]
        )

        # 3. Projection MLP: width -> mlp_hidden -> 1 + Sigmoid
        self.proj = nn.Sequential(
            nn.Linear(self.width, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
            nn.Sigmoid(),   # H is in (0, 1)
        )

        # Log architecture summary
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(
            "FNO1d: in_channels=%d  width=%d  n_modes=%d  n_layers=%d  "
            "mlp_hidden=%d  params=%d",
            in_channels, self.width, self.n_modes, self.n_layers, mlp_hidden,
            n_params,
        )

    # Forward pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict H from a batch of input windows.

        Parameters
        ----------
        x : Tensor, shape (B, C_in, L)
            Batch of input feature windows.

        Returns
        -------
        Tensor, shape (B, 1)
            Predicted Hurst exponents in (0, 1).
        """
        # Lift
        x = self.lifting(x)           # (B, width, L)

        # FNO blocks
        for block in self.blocks:
            x = block(x)              # (B, width, L)

        # Global mean pool over sequence dimension
        x = x.mean(dim=-1)            # (B, width)

        # Project to scalar H
        return self.proj(x)           # (B, 1)

    # Convenience: single-window prediction

    def predict_H(self, window: "np.ndarray") -> float:  # type: ignore[name-defined]
        """Predict H for a single feature window (numpy-friendly).

        Parameters
        ----------
        window : np.ndarray, shape (C_in, L) or (L,)
            A single window of features.  If 1-D, it is treated as a single
            channel (shape ``(1, L)``).

        Returns
        -------
        float
            Predicted Hurst exponent in (0, 1).
        """
        import numpy as np  # local import: keeps torch as the only hard dep

        arr = np.asarray(window, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]       # (1, L)
        t = torch.from_numpy(arr).unsqueeze(0)  # (1, C_in, L)
        self.eval()
        with torch.no_grad():
            return float(self.forward(t).squeeze())


class HurstDataset(Dataset):
    """Sliding-window dataset built from SPDE simulation output.

    Each sample is a pair (window_tensor, h_label) where:
        window_tensor  -- float32 Tensor of shape (n_features, window_size)
        h_label        -- float32 scalar Tensor, the true Hurst exponent

    Parameters
    ----------
    windows : np.ndarray, shape (N, C, L)
        Pre-extracted feature windows.
    labels : np.ndarray, shape (N,)
        Hurst exponent label for each window.
    """

    def __init__(self, windows: np.ndarray, labels: np.ndarray) -> None:
        if windows.shape[0] != labels.shape[0]:
            raise ValueError(
                f"windows ({windows.shape[0]}) and labels ({labels.shape[0]}) "
                "must have the same number of samples."
            )
        self.windows = torch.from_numpy(windows.astype(np.float32))
        self.labels  = torch.from_numpy(labels.astype(np.float32))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.windows[idx], self.labels[idx].unsqueeze(0)


# TrainingHistory

@dataclass
class TrainingHistory:
    """Records of training progress returned by :meth:`FNOTrainer.train`.

    Attributes
    ----------
    train_losses : list[float]
        Mean MSE loss per epoch on the training set.
    val_losses : list[float]
        Mean MSE loss per epoch on the validation set.
    val_maes : list[float]
        Mean absolute error per epoch on the validation set.
    best_epoch : int
        Epoch index (0-based) with the lowest validation loss.
    best_val_loss : float
        Best (lowest) validation loss achieved.
    checkpoint_path : Path or None
        Path where the best checkpoint was saved.
    """

    train_losses:    list[float] = field(default_factory=list)
    val_losses:      list[float] = field(default_factory=list)
    val_maes:        list[float] = field(default_factory=list)
    best_epoch:      int = 0
    best_val_loss:   float = float("inf")
    checkpoint_path: Optional[Path] = None


# Trainer

class FNOTrainer:
    """End-to-end training pipeline for :class:`~wsq_trading.fno.FNO1d`.

    Parameters
    ----------
    model : FNO1d, optional
        Pre-built model.  A default ``FNO1d(in_channels=2)`` is created if
        omitted (2 channels = log_return + variance).
    feature_cols : list[str]
        DataFrame columns to use as input channels.
        Default: ``["log_return", "variance"]``.
    window_size : int
        Sliding-window length in bars.  Default: ``config.WINDOW_SIZE``.
    stride : int
        Step between consecutive windows.  Default: ``window_size // 4``
        (75% overlap) - gives plentiful but partially correlated samples.
    val_frac : float
        Fraction of paths held out for validation.  Default: 0.15.
    learning_rate : float
        Initial Adam learning rate.  Default: ``config.LEARNING_RATE``.
    batch_size : int
        Loader batch size.  Default: ``config.BATCH_SIZE``.
    num_epochs : int
        Maximum training epochs.  Default: ``config.NUM_EPOCHS``.
    patience : int
        Early-stopping patience (epochs with no val improvement).
        Default: ``config.EARLY_STOPPING_PATIENCE``.
    grad_clip : float
        Max gradient norm for clipping.  Default: ``config.GRADIENT_CLIP_NORM``.
    checkpoint_dir : Path, optional
        Directory for saving checkpoints.  Default: ``config.CHECKPOINTS_DIR``.
    device : str
        ``"cuda"``, ``"cpu"``, or ``"auto"`` (default: auto-detect).
    """

    def __init__(
        self,
        model: Optional[FNO1d] = None,
        feature_cols: Optional[list[str]] = None,
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
        val_frac: float = 0.15,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        num_epochs: Optional[int] = None,
        patience: Optional[int] = None,
        grad_clip: Optional[float] = None,
        checkpoint_dir: Optional[Path] = None,
        device: str = "auto",
    ) -> None:
        self.feature_cols    = feature_cols or ["log_return", "variance"]
        self.window_size     = window_size  or config.WINDOW_SIZE
        self.stride          = stride       or max(1, self.window_size // 4)
        self.val_frac        = val_frac
        self.lr              = learning_rate or config.LEARNING_RATE
        self.batch_size      = batch_size   or config.BATCH_SIZE
        self.num_epochs      = num_epochs   or config.NUM_EPOCHS
        self.patience        = patience     or config.EARLY_STOPPING_PATIENCE
        self.grad_clip       = grad_clip    or config.GRADIENT_CLIP_NORM
        self.checkpoint_dir  = Path(checkpoint_dir or config.CHECKPOINTS_DIR)

        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model
        n_features = len(self.feature_cols)
        self.model = model or FNO1d(in_channels=n_features)
        self.model.to(self.device)

        log.info(
            "FNOTrainer: device=%s  features=%s  window=%d  stride=%d  "
            "lr=%.1e  batch=%d  epochs=%d  patience=%d",
            self.device, self.feature_cols, self.window_size, self.stride,
            self.lr, self.batch_size, self.num_epochs, self.patience,
        )

    # Internal helpers

    def _extract_windows(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slide a window over each path and collect (features, H) pairs."""
        for col in self.feature_cols + ["path_id", "H"]:
            if col not in df.columns:
                raise ValueError(f"Dataset missing required column: '{col}'")

        all_windows: list[np.ndarray] = []
        all_labels:  list[float]      = []
        W = self.window_size
        S = self.stride

        for path_id, group in df.groupby("path_id", sort=False):
            feats = group[self.feature_cols].to_numpy(dtype=np.float32)  # (T, C)
            H_val = float(group["H"].iloc[0])
            T = len(feats)

            for start in range(0, T - W + 1, S):
                window = feats[start: start + W]  # (W, C)
                # Normalise each channel: subtract mean, divide by std
                mu  = window.mean(axis=0, keepdims=True)
                std = window.std(axis=0, keepdims=True) + 1e-8
                window = (window - mu) / std
                all_windows.append(window.T)  # (C, W)
                all_labels.append(H_val)

        windows = np.stack(all_windows)      # (N, C, W)
        labels  = np.array(all_labels)       # (N,)
        log.info("Extracted %d windows from %d paths.", len(labels),
                 df["path_id"].nunique())
        return windows, labels

    def _build_loaders(
        self, dataset: HurstDataset
    ) -> tuple[DataLoader, DataLoader]:
        """Split dataset into train / val and wrap in DataLoaders."""
        n_val   = max(1, int(len(dataset) * self.val_frac))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(config.RANDOM_SEED),
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size,
            shuffle=True, drop_last=True, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size * 2,
            shuffle=False, num_workers=0,
        )
        log.info("DataLoaders: train=%d  val=%d  batch=%d",
                 n_train, n_val, self.batch_size)
        return train_loader, val_loader

    # Training

    def train(self, dataset_df: pd.DataFrame) -> TrainingHistory:
        """Train the FNO on SPDE-generated data.

        Parameters
        ----------
        dataset_df : pd.DataFrame
            Output of :meth:`SPDESimulator.generate_dataset` or any DataFrame
            with columns ``path_id``, ``H``, and all ``self.feature_cols``.

        Returns
        -------
        TrainingHistory
            Per-epoch metrics and best checkpoint location.
        """
        with timer("FNO training: window extraction"):
            windows, labels = self._extract_windows(dataset_df)

        dataset = HurstDataset(windows, labels)
        train_loader, val_loader = self._build_loaders(dataset)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
        )

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.checkpoint_dir / "fno_best.pt"

        history = TrainingHistory(checkpoint_path=ckpt_path)
        no_improve = 0

        for epoch in range(self.num_epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(train_loader.dataset)

            # Validate
            val_loss, val_mae = self._evaluate(val_loader, criterion)
            scheduler.step(val_loss)

            history.train_losses.append(train_loss)
            history.val_losses.append(val_loss)
            history.val_maes.append(val_mae)

            # Checkpoint & early stopping
            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss
                history.best_epoch    = epoch
                no_improve = 0
                self._save_checkpoint(ckpt_path, epoch, val_loss)
            else:
                no_improve += 1

            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                log.info(
                    "Epoch %3d/%d  train=%.5f  val=%.5f  mae=%.4f  "
                    "lr=%.2e  no_improve=%d",
                    epoch + 1, self.num_epochs, train_loss, val_loss, val_mae,
                    optimizer.param_groups[0]["lr"], no_improve,
                )

            if no_improve >= self.patience:
                log.info(
                    "Early stopping at epoch %d (patience=%d).",
                    epoch + 1, self.patience,
                )
                break

        log.info(
            "Training complete. Best epoch=%d  val_loss=%.5f  val_mae=%.4f",
            history.best_epoch + 1, history.best_val_loss,
            history.val_maes[history.best_epoch],
        )
        return history

    # Evaluation

    def _evaluate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> tuple[float, float]:
        """Run one pass over *loader* and return (mse_loss, mae)."""
        self.model.eval()
        total_loss = 0.0
        total_mae  = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                total_loss += criterion(pred, yb).item() * len(xb)
                total_mae  += (pred - yb).abs().sum().item()
                n += len(xb)
        return total_loss / n, total_mae / n

    # Checkpointing

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_loss: float,
    ) -> None:
        torch.save(
            {
                "epoch":      epoch,
                "val_loss":   val_loss,
                "model_state": self.model.state_dict(),
                "model_kwargs": {
                    "in_channels": self.model.lifting.in_channels,
                    "width":       self.model.width,
                    "n_modes":     self.model.n_modes,
                    "n_layers":    self.model.n_layers,
                },
            },
            path,
        )
        log.debug("Checkpoint saved: %s (epoch=%d  val_loss=%.5f)",
                  path, epoch + 1, val_loss)

    def load_checkpoint(self, path: Optional[Path] = None) -> None:
        """Load model weights from a checkpoint file.

        Parameters
        ----------
        path : Path, optional
            Path to the ``.pt`` file.  Defaults to the trainer's own
            ``checkpoint_dir / fno_best.pt``.
        """
        ckpt_path = Path(path or (self.checkpoint_dir / "fno_best.pt"))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        log.info(
            "Loaded checkpoint from %s  (epoch=%d  val_loss=%.5f)",
            ckpt_path, ckpt["epoch"] + 1, ckpt["val_loss"],
        )

    # Inference

    def predict_rolling(
        self,
        returns: pd.Series,
        variance: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Apply the trained FNO with a rolling window to a live return series.

        Parameters
        ----------
        returns : pd.Series
            Log-return series.  Length must be >= ``window_size``.
        variance : pd.Series, optional
            Realised or model variance series aligned with *returns*.
            Required when ``feature_cols`` includes ``"variance"``.
            If omitted and ``"variance"`` is in ``feature_cols``,
            a proxy is computed via EWM variance with span=21.

        Returns
        -------
        pd.Series
            Rolling H estimates, same index as *returns*.
            Values are NaN during the warm-up period (first window_size-1 bars).
        """
        n = len(returns)
        W = self.window_size

        if n < W:
            raise ValueError(
                f"Series length {n} < window_size {W}."
            )

        # Build feature array (T, C)
        col_arrays = []
        for col in self.feature_cols:
            if col == "log_return":
                col_arrays.append(returns.to_numpy(dtype=np.float32))
            elif col == "variance":
                if variance is not None:
                    col_arrays.append(variance.to_numpy(dtype=np.float32))
                else:
                    # EWM variance (span=21) is far less noisy than r² and
                    # much closer to the simulated variance series the FNO
                    # was trained on.  r² has ~90% noise-to-signal at daily
                    # frequency and degrades H prediction quality.
                    log.debug(
                        "variance not provided; using EWM variance (span=21) as proxy."
                    )
                    ewm_var = (
                        pd.Series(returns.to_numpy(), dtype=float)
                        .ewm(span=21, min_periods=5)
                        .var()
                        .to_numpy(dtype=np.float32)
                    )
                    col_arrays.append(ewm_var)
            else:
                raise ValueError(
                    f"Feature column '{col}' not available for live inference. "
                    "Provide it via the 'variance' argument or subclass predict_rolling."
                )

        feats = np.stack(col_arrays, axis=1)  # (T, C)

        self.model.eval()
        h_values = np.full(n, np.nan)

        with torch.no_grad():
            for i in range(W, n + 1):
                window = feats[i - W: i].copy()      # (W, C)
                mu  = window.mean(axis=0, keepdims=True)
                std = window.std(axis=0, keepdims=True) + 1e-8
                window = (window - mu) / std
                t = torch.from_numpy(window.T[np.newaxis]).to(self.device)  # (1,C,W)
                h_values[i - 1] = float(self.model(t).squeeze().cpu())

        return pd.Series(h_values, index=returns.index, name="H_fno")

