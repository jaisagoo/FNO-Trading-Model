"""Tests for the Fourier neural operator and its trainer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from wsq_trading.fno import FNO1d, FNOBlock, SpectralConv1d

# Helpers

def _rand(shape, seed=0):
    torch.manual_seed(seed)
    return torch.randn(*shape)


# SpectralConv1d

class TestSpectralConv1d:
    def test_output_shape_matches_input(self):
        layer = SpectralConv1d(in_channels=4, out_channels=8, n_modes=8)
        x = _rand((2, 4, 64))
        y = layer(x)
        assert y.shape == (2, 8, 64)

    def test_output_shape_odd_length(self):
        layer = SpectralConv1d(in_channels=4, out_channels=4, n_modes=8)
        x = _rand((3, 4, 63))
        y = layer(x)
        assert y.shape == (3, 4, 63)

    def test_output_is_real(self):
        layer = SpectralConv1d(in_channels=2, out_channels=2, n_modes=8)
        x = _rand((2, 2, 32))
        y = layer(x)
        assert y.dtype == torch.float32

    def test_n_modes_exceeds_half_seq_no_crash(self):
        """n_modes > seq_len // 2 + 1 should not raise."""
        layer = SpectralConv1d(in_channels=2, out_channels=2, n_modes=100)
        x = _rand((1, 2, 16))
        y = layer(x)
        assert y.shape == (1, 2, 16)

    def test_gradient_flows(self):
        layer = SpectralConv1d(in_channels=2, out_channels=2, n_modes=8)
        x = _rand((2, 2, 32)).requires_grad_(True)
        y = layer(x).sum()
        y.backward()
        assert x.grad is not None
        assert layer.weights.grad is not None

    def test_different_batch_sizes_consistent(self):
        layer = SpectralConv1d(in_channels=1, out_channels=1, n_modes=8)
        x1 = _rand((1, 1, 32))
        x4 = x1.expand(4, -1, -1)
        y1 = layer(x1)
        y4 = layer(x4)
        assert torch.allclose(y1.expand(4, -1, -1), y4, atol=1e-5)


# FNOBlock

class TestFNOBlock:
    def test_output_shape_preserved(self):
        block = FNOBlock(width=32, n_modes=8)
        x = _rand((4, 32, 63))
        y = block(x)
        assert y.shape == x.shape

    def test_output_is_float32(self):
        block = FNOBlock(width=16, n_modes=4)
        x = _rand((2, 16, 32))
        y = block(x)
        assert y.dtype == torch.float32

    def test_gradient_flows_through_block(self):
        block = FNOBlock(width=16, n_modes=4)
        x = _rand((2, 16, 32)).requires_grad_(True)
        block(x).sum().backward()
        assert x.grad is not None

    def test_not_equal_to_input(self):
        """Block should actually transform the input."""
        block = FNOBlock(width=16, n_modes=4)
        x = _rand((2, 16, 32))
        y = block(x)
        assert not torch.allclose(x, y)


# FNO1d

class TestFNO1dShape:
    """Shape contract: (B, C_in, L) -> (B, 1)."""

    @pytest.fixture()
    def model(self):
        return FNO1d(in_channels=1, width=16, n_modes=8, n_layers=2, mlp_hidden=32)

    def test_output_shape_standard(self, model):
        x = _rand((8, 1, 63))
        y = model(x)
        assert y.shape == (8, 1)

    def test_output_shape_batch_one(self, model):
        x = _rand((1, 1, 63))
        y = model(x)
        assert y.shape == (1, 1)

    def test_output_shape_multi_channel(self):
        model = FNO1d(in_channels=3, width=16, n_modes=8, n_layers=2, mlp_hidden=32)
        x = _rand((4, 3, 63))
        y = model(x)
        assert y.shape == (4, 1)

    def test_output_shape_short_sequence(self, model):
        """Should handle sequences much shorter than n_modes without crashing."""
        x = _rand((2, 1, 20))
        y = model(x)
        assert y.shape == (2, 1)

    def test_output_shape_long_sequence(self, model):
        x = _rand((2, 1, 252))
        y = model(x)
        assert y.shape == (2, 1)


class TestFNO1dOutput:
    """Output value constraints."""

    @pytest.fixture()
    def model(self):
        return FNO1d(in_channels=1, width=16, n_modes=8, n_layers=2, mlp_hidden=32)

    def test_output_bounded_zero_one(self, model):
        x = _rand((32, 1, 63))
        y = model(x)
        assert (y > 0).all() and (y < 1).all()

    def test_output_is_float32(self, model):
        x = _rand((4, 1, 63))
        assert model(x).dtype == torch.float32

    def test_different_inputs_give_different_outputs(self, model):
        x1 = _rand((4, 1, 63), seed=0)
        x2 = _rand((4, 1, 63), seed=1)
        assert not torch.allclose(model(x1), model(x2))

    def test_eval_mode_disables_dropout(self, model):
        """In eval mode two identical forward passes should be identical."""
        model.eval()
        x = _rand((4, 1, 63))
        y1 = model(x)
        y2 = model(x)
        assert torch.allclose(y1, y2)

    def test_train_mode_dropout_varies(self):
        """In train mode with high dropout, two passes may differ."""
        model = FNO1d(in_channels=1, width=16, n_modes=8, n_layers=2,
                      mlp_hidden=32, dropout=0.5)
        model.train()
        x = _rand((64, 1, 63))
        y1 = model(x)
        y2 = model(x)
        # Not identical due to stochastic dropout
        assert not torch.allclose(y1, y2)


class TestFNO1dGradients:
    """Gradient flow checks for training."""

    @pytest.fixture()
    def model(self):
        return FNO1d(in_channels=1, width=16, n_modes=8, n_layers=2, mlp_hidden=32)

    def test_backward_pass_does_not_crash(self, model):
        x = _rand((4, 1, 63))
        loss = model(x).sum()
        loss.backward()

    def test_all_parameters_receive_gradients(self, model):
        x = _rand((4, 1, 63))
        model(x).sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"

    def test_mse_loss_decreases_over_steps(self):
        """A few SGD steps on a fixed batch should reduce MSE loss."""
        torch.manual_seed(99)
        model = FNO1d(in_channels=1, width=16, n_modes=8, n_layers=2, mlp_hidden=32)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = _rand((16, 1, 63))
        target = torch.full((16, 1), 0.3)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x), target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss did not decrease over 10 steps"


class TestFNO1dPredictH:
    """predict_H convenience wrapper."""

    @pytest.fixture()
    def model(self):
        model = FNO1d(in_channels=1, width=16, n_modes=8, n_layers=2, mlp_hidden=32)
        model.eval()
        return model

    def test_returns_float(self, model):
        window = np.random.default_rng(0).normal(0, 0.01, 63).astype(np.float32)
        h = model.predict_H(window)
        assert isinstance(h, float)

    def test_output_in_zero_one(self, model):
        window = np.random.default_rng(0).normal(0, 0.01, 63).astype(np.float32)
        h = model.predict_H(window)
        assert 0.0 < h < 1.0

    def test_2d_input_accepted(self, model):
        """(C_in, L) input should work without error."""
        window = np.random.default_rng(0).normal(0, 0.01, (1, 63)).astype(np.float32)
        h = model.predict_H(window)
        assert isinstance(h, float)

    def test_consistent_with_forward(self, model):
        """predict_H should agree with forward on the same data."""
        rng = np.random.default_rng(7)
        window = rng.normal(0, 0.01, 63).astype(np.float32)
        h_convenience = model.predict_H(window)
        t = torch.from_numpy(window[np.newaxis, np.newaxis, :])
        h_forward = float(model(t).squeeze())
        assert abs(h_convenience - h_forward) < 1e-6


# Config-driven defaults

class TestFNO1dDefaults:
    def test_default_constructor_uses_config(self):
        from wsq_trading import config
        model = FNO1d(in_channels=1)
        assert model.width   == config.FNO_WIDTH
        assert model.n_modes == config.FNO_MODES
        assert model.n_layers == config.FNO_LAYERS

    def test_parameter_count_reasonable(self):
        """Model should have a non-trivial but not gigantic number of params."""
        model = FNO1d(in_channels=1)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 10_000 < n_params < 10_000_000, f"Unexpected param count: {n_params}"


from wsq_trading.fno import FNOTrainer, HurstDataset, TrainingHistory

# Helpers

def _make_spde_df(
    n_paths: int = 6,
    n_steps: int = 100,
    H_values: list[float] | None = None,
) -> pd.DataFrame:
    """Minimal SPDE-like DataFrame with path_id, step, log_return, variance, H."""
    if H_values is None:
        H_values = [0.20, 0.50, 0.80]
    rng = np.random.default_rng(42)
    rows = []
    path_id = 0
    for H in H_values:
        for _ in range(n_paths // len(H_values)):
            for step in range(n_steps):
                rows.append({
                    "path_id":    path_id,
                    "step":       step,
                    "log_return": rng.normal(0.0, 0.01),
                    "variance":   abs(rng.normal(0.04, 0.005)),
                    "H":          H,
                })
            path_id += 1
    return pd.DataFrame(rows)


def _small_trainer(tmp_path: Path, num_epochs: int = 3, patience: int = 10, **kwargs) -> FNOTrainer:
    """Trainer with tiny model and fast hyperparams for tests."""
    model = FNO1d(in_channels=2, width=8, n_modes=4, n_layers=1, mlp_hidden=16)
    return FNOTrainer(
        model=model,
        window_size=20,
        stride=10,
        batch_size=8,
        num_epochs=num_epochs,
        patience=patience,
        checkpoint_dir=tmp_path,
        **kwargs,
    )


# HurstDataset

class TestHurstDataset:
    def test_len_matches_n_samples(self):
        windows = np.random.default_rng(0).normal(size=(50, 2, 20)).astype(np.float32)
        labels  = np.random.default_rng(0).uniform(0.1, 0.9, 50).astype(np.float32)
        ds = HurstDataset(windows, labels)
        assert len(ds) == 50

    def test_getitem_shapes(self):
        windows = np.ones((10, 2, 20), dtype=np.float32)
        labels  = np.full(10, 0.3, dtype=np.float32)
        ds = HurstDataset(windows, labels)
        x, y = ds[0]
        assert x.shape == (2, 20)
        assert y.shape == (1,)

    def test_getitem_dtypes_float32(self):
        windows = np.ones((4, 1, 10), dtype=np.float64)
        labels  = np.full(4, 0.5, dtype=np.float64)
        ds = HurstDataset(windows, labels)
        x, y = ds[0]
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_mismatched_lengths_raise(self):
        windows = np.ones((10, 2, 20), dtype=np.float32)
        labels  = np.ones(5, dtype=np.float32)
        with pytest.raises(ValueError, match="same number"):
            HurstDataset(windows, labels)

    def test_label_value_preserved(self):
        windows = np.zeros((3, 1, 10), dtype=np.float32)
        labels  = np.array([0.2, 0.5, 0.8], dtype=np.float32)
        ds = HurstDataset(windows, labels)
        _, y = ds[1]
        assert abs(float(y) - 0.5) < 1e-6


# FNOTrainer._extract_windows

class TestExtractWindows:
    def test_returns_correct_shapes(self, tmp_path):
        trainer = _small_trainer(tmp_path)
        df = _make_spde_df(n_paths=6, n_steps=100)
        windows, labels = trainer._extract_windows(df)
        assert windows.ndim == 3
        assert windows.shape[1] == 2    # 2 feature cols
        assert windows.shape[2] == 20   # window_size
        assert labels.ndim == 1
        assert windows.shape[0] == labels.shape[0]

    def test_window_count_per_path(self, tmp_path):
        trainer = _small_trainer(tmp_path)
        df = _make_spde_df(n_paths=3, n_steps=60, H_values=[0.20, 0.50, 0.80])
        windows, labels = trainer._extract_windows(df)
        # Per path: floor((60 - 20) / 10) + 1 = 5 windows; 3 paths = 15 total
        assert windows.shape[0] == 15

    def test_labels_are_valid_h_values(self, tmp_path):
        trainer = _small_trainer(tmp_path)
        df = _make_spde_df(n_paths=6, n_steps=100)
        _, labels = trainer._extract_windows(df)
        assert (labels >= 0.0).all() and (labels <= 1.0).all()

    def test_missing_column_raises(self, tmp_path):
        trainer = _small_trainer(tmp_path)
        df = _make_spde_df(n_paths=2, n_steps=50, H_values=[0.20])
        df = df.drop(columns=["variance"])
        with pytest.raises(ValueError, match="variance"):
            trainer._extract_windows(df)

    def test_windows_are_normalised(self, tmp_path):
        """Each window should have ~zero mean and ~unit std per channel."""
        trainer = _small_trainer(tmp_path)
        df = _make_spde_df(n_paths=3, n_steps=60, H_values=[0.30])
        windows, _ = trainer._extract_windows(df)
        for i in range(windows.shape[0]):
            for c in range(windows.shape[1]):
                w = windows[i, c]
                assert abs(w.mean()) < 0.1, "Window mean too large after normalisation"
                assert 0.5 < w.std() < 2.0, "Window std too far from 1"


# FNOTrainer._build_loaders

class TestBuildLoaders:
    def test_loaders_partition_dataset(self, tmp_path):
        trainer = _small_trainer(tmp_path)
        windows = np.zeros((100, 2, 20), dtype=np.float32)
        labels  = np.zeros(100, dtype=np.float32)
        ds = HurstDataset(windows, labels)
        train_loader, val_loader = trainer._build_loaders(ds)
        n_train = len(train_loader.dataset)
        n_val   = len(val_loader.dataset)
        assert n_train + n_val == 100

    def test_val_fraction_respected(self, tmp_path):
        trainer = _small_trainer(tmp_path, val_frac=0.20)
        windows = np.zeros((200, 2, 20), dtype=np.float32)
        labels  = np.zeros(200, dtype=np.float32)
        ds = HurstDataset(windows, labels)
        _, val_loader = trainer._build_loaders(ds)
        assert len(val_loader.dataset) == 40  # 20% of 200

    def test_train_loader_batches_correct_shape(self, tmp_path):
        trainer = _small_trainer(tmp_path)
        windows = np.zeros((50, 2, 20), dtype=np.float32)
        labels  = np.zeros(50, dtype=np.float32)
        ds = HurstDataset(windows, labels)
        train_loader, _ = trainer._build_loaders(ds)
        xb, yb = next(iter(train_loader))
        assert xb.shape == (min(8, len(train_loader.dataset)), 2, 20)
        assert yb.shape[1] == 1


# FNOTrainer.train

class TestTrain:
    def test_returns_training_history(self, tmp_path):
        trainer = _small_trainer(tmp_path)
        df = _make_spde_df(n_paths=6, n_steps=80)
        history = trainer.train(df)
        assert isinstance(history, TrainingHistory)

    def test_history_has_correct_length(self, tmp_path):
        trainer = _small_trainer(tmp_path, num_epochs=3)
        df = _make_spde_df(n_paths=6, n_steps=80)
        history = trainer.train(df)
        assert len(history.train_losses) == 3
        assert len(history.val_losses)   == 3
        assert len(history.val_maes)     == 3

    def test_train_losses_are_positive(self, tmp_path):
        trainer = _small_trainer(tmp_path)
        df = _make_spde_df(n_paths=6, n_steps=80)
        history = trainer.train(df)
        assert all(l > 0 for l in history.train_losses)

    def test_checkpoint_saved(self, tmp_path):
        trainer = _small_trainer(tmp_path)
        df = _make_spde_df(n_paths=6, n_steps=80)
        history = trainer.train(df)
        assert history.checkpoint_path is not None
        assert history.checkpoint_path.exists()

    def test_best_val_loss_is_minimum(self, tmp_path):
        trainer = _small_trainer(tmp_path)
        df = _make_spde_df(n_paths=6, n_steps=80)
        history = trainer.train(df)
        assert history.best_val_loss == pytest.approx(min(history.val_losses), rel=1e-4)

    def test_early_stopping_fires(self, tmp_path):
        trainer = _small_trainer(tmp_path, num_epochs=20, patience=2)
        df = _make_spde_df(n_paths=6, n_steps=80)
        history = trainer.train(df)
        # Should stop before 20 epochs if val loss stops improving
        assert len(history.train_losses) <= 20


# FNOTrainer checkpoint round-trip

class TestCheckpoint:
    def test_load_checkpoint_restores_weights(self, tmp_path):
        trainer = _small_trainer(tmp_path)
        df = _make_spde_df(n_paths=6, n_steps=80)
        history = trainer.train(df)

        # Load checkpoint weights into a reference copy
        ref_state = torch.load(history.checkpoint_path, map_location="cpu")["model_state"]

        # Corrupt model weights
        with torch.no_grad():
            for p in trainer.model.parameters():
                p.fill_(0.0)

        # Reload from checkpoint
        trainer.load_checkpoint()

        # Weights should match the checkpoint (not necessarily post-training weights)
        for k, v in trainer.model.state_dict().items():
            assert torch.allclose(v, ref_state[k].to(v.device)), f"Weight mismatch for {k}"

    def test_load_checkpoint_missing_file_raises(self, tmp_path):
        trainer = _small_trainer(tmp_path)
        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint(tmp_path / "nonexistent.pt")


# FNOTrainer.predict_rolling

class TestPredictRolling:
    def _trained_trainer(self, tmp_path):
        trainer = _small_trainer(tmp_path)
        df = _make_spde_df(n_paths=6, n_steps=80)
        trainer.train(df)
        return trainer

    def test_returns_series_with_correct_length(self, tmp_path):
        trainer = self._trained_trainer(tmp_path)
        returns = pd.Series(
            np.random.default_rng(0).normal(0, 0.01, 100),
            index=pd.bdate_range("2020-01-01", periods=100),
        )
        h_series = trainer.predict_rolling(returns)
        assert len(h_series) == len(returns)

    def test_index_matches_returns(self, tmp_path):
        trainer = self._trained_trainer(tmp_path)
        returns = pd.Series(
            np.random.default_rng(0).normal(0, 0.01, 80),
            index=pd.bdate_range("2020-01-01", periods=80),
        )
        h_series = trainer.predict_rolling(returns)
        assert h_series.index.equals(returns.index)

    def test_warmup_bars_are_nan(self, tmp_path):
        trainer = self._trained_trainer(tmp_path)
        returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, 80))
        h_series = trainer.predict_rolling(returns)
        W = trainer.window_size
        assert h_series.iloc[:W - 1].isna().all()

    def test_valid_values_in_zero_one(self, tmp_path):
        trainer = self._trained_trainer(tmp_path)
        returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, 80))
        h_series = trainer.predict_rolling(returns)
        valid = h_series.dropna()
        assert (valid > 0).all() and (valid < 1).all()

    def test_too_short_raises(self, tmp_path):
        trainer = self._trained_trainer(tmp_path)
        short = pd.Series(np.zeros(5))
        with pytest.raises(ValueError, match="window_size"):
            trainer.predict_rolling(short)

    def test_variance_proxy_when_not_provided(self, tmp_path):
        """When variance column is in feature_cols but not passed, uses r^2."""
        trainer = self._trained_trainer(tmp_path)
        returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, 80))
        # Should not raise even without variance argument
        h_series = trainer.predict_rolling(returns)
        assert not h_series.dropna().empty

