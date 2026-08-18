"""
Learnable Factor Attribution for CD-DSD.

Trains a small MLP to classify which domain factor is responsible
for the shift between a test image and the source distribution.

Input:  feature difference  φ(x_test) - μ_CheXpert  (1024-dim)
Output: probability over 4 factors (brightness, noise, scanner, structure)

Training data is generated synthetically from CheXpert with known labels.
This makes attribution genuinely learnable — optimised by cross-entropy loss.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

from cd_dsd.datasets import CheXpertDataset, get_eval_transform, collate_fn

logger = logging.getLogger(__name__)

FACTOR_NAMES = [
    "brightness_contrast",
    "noise_texture",
    "scanner_artefact",
    "global_structure",
]

# ---------------------------------------------------------------------------
# Corruption functions
# ---------------------------------------------------------------------------

def corrupt_brightness(x, strength=None):
    s = strength or (torch.rand(1).item() * 1.5 + 0.5)
    return (x + s).clamp(-3, 3)

def corrupt_darkness(x, strength=None):
    s = strength or (torch.rand(1).item() * 1.5 + 0.5)
    return (x - s).clamp(-3, 3)

def corrupt_noise(x, sigma=None):
    s = sigma or (torch.rand(1).item() * 0.6 + 0.2)
    return x + torch.randn_like(x) * s

def corrupt_blur(x, kernel_size=9, sigma=None):
    s      = sigma or (torch.rand(1).item() * 2.5 + 1.0)
    k      = kernel_size
    coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
    g      = torch.exp(-0.5 * (coords / s) ** 2)
    g      = g / g.sum()
    kernel = (g[:, None] * g[None, :])
    kernel = kernel[None, None].expand(x.shape[1], 1, -1, -1)
    return F.conv2d(x, kernel, padding=k // 2, groups=x.shape[1])

def corrupt_contrast(x, factor=None):
    f    = factor or (torch.rand(1).item() * 2.0 + 1.5)
    mean = x.mean(dim=(-2,-1), keepdim=True)
    return ((x - mean) * f + mean).clamp(-3, 3)

def corrupt_structure(x):
    B = x.shape[0]
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    theta[:, 0, 2] = (torch.rand(B, device=x.device) - 0.5) * 0.4
    theta[:, 1, 2] = (torch.rand(B, device=x.device) - 0.5) * 0.4
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False)

# Map factor index to list of corruption functions
# Multiple corruptions per factor increases training diversity
FACTOR_CORRUPTIONS = {
    0: [corrupt_brightness, corrupt_darkness, corrupt_contrast],   # brightness_contrast
    1: [corrupt_noise],                           # noise_texture
    2: [corrupt_blur],                            # scanner_artefact
    3: [corrupt_structure],                       # global_structure
}


# ---------------------------------------------------------------------------
# Learnable Factor MLP
# ---------------------------------------------------------------------------

class FactorMLP(nn.Module):
    """
    Small MLP trained to identify dominant domain shift factor.

    Input:  normalised feature difference (1024-dim)
    Output: logits over 4 factors
    """
    def __init__(self, feat_dim=1024, hidden_dim=256, n_factors=4,
                 dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_factors),
        )

    def forward(self, x):
        return self.net(x)   # (B, n_factors) logits


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_training_data(diagnoser, cfg, n_samples_per_factor=500):
    """
    Generate synthetic training data:
      - For each factor, apply its corruption to CheXpert images
      - Extract feature difference φ(x_corrupted) - μ_CheXpert
      - Label = factor index

    Returns
    -------
    X : (N, 1024) normalised feature differences
    y : (N,)      factor labels  0-3
    """

    logger.info("Building factor classifier training data...")
    eval_tf = get_eval_transform(
        cfg.IMAGE_SIZE, cfg.MEAN, cfg.STD)
    ds = CheXpertDataset(
        csv_path   = cfg.CHEXPERT_TRAIN_CSV,
        image_root = cfg.CHEXPERT_IMAGE_ROOT,
        label_cols = cfg.LABEL_COLS,
        transform  = eval_tf,
    )
    loader = DataLoader(ds, batch_size=16, shuffle=True,
                        collate_fn=collate_fn, num_workers=2)

    # Collect enough clean images
    n_total = n_samples_per_factor * 4
    clean_batches = []
    for imgs, _, _ in loader:
        clean_batches.append(imgs.to(diagnoser.device))
        if sum(len(b) for b in clean_batches) >= n_total:
            break
    clean_imgs = torch.cat(clean_batches)[:n_total]
    logger.info(f"  Collected {len(clean_imgs)} clean images")

    all_X = []
    all_y = []

    for factor_idx, corrupt_fns in FACTOR_CORRUPTIONS.items():
        factor_name = FACTOR_NAMES[factor_idx]
        logger.info(f"  Generating samples for: {factor_name}")

        # Take a slice of clean images for this factor
        start = factor_idx * n_samples_per_factor
        end   = start + n_samples_per_factor
        imgs  = clean_imgs[start:end]

        # Apply corruptions (round-robin if multiple)
        corrupted_list = []
        for i, img in enumerate(imgs):
            fn = corrupt_fns[i % len(corrupt_fns)]
            corrupted_list.append(fn(img.unsqueeze(0)).squeeze(0))
        corrupted = torch.stack(corrupted_list)   # (N, 3, H, W)

        # Extract features in batches
        feat_diffs = []
        batch_size = 16
        for i in range(0, len(corrupted), batch_size):
            batch = corrupted[i:i + batch_size]
            feats = diagnoser._extract_features(batch)          # (B, 1024)
            # Normalised feature difference from source distribution
            diff  = ((feats - diagnoser.chexpert_mean)
                     / diagnoser.chexpert_std)                  # (B, 1024)
            feat_diffs.append(diff.cpu())

        feat_diffs = torch.cat(feat_diffs)                      # (N, 1024)
        labels     = torch.full((len(feat_diffs),), factor_idx,
                                dtype=torch.long)

        all_X.append(feat_diffs)
        all_y.append(labels)

        logger.info(f"    {factor_name}: {len(feat_diffs)} samples")

    X = torch.cat(all_X)   # (4N, 1024)
    y = torch.cat(all_y)   # (4N,)
    logger.info(f"Training data built: {len(X)} total samples, "
                f"{len(FACTOR_NAMES)} classes")
    return X, y


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class FactorClassifierTrainer:
    """
    Trains the FactorMLP on synthetic corrupted CheXpert data.
    """

    def __init__(self, diagnoser, cfg):
        self.diagnoser = diagnoser
        self.cfg       = cfg
        self.device    = diagnoser.device
        self.model     = FactorMLP(
            feat_dim   = 1024,
            hidden_dim = 256,
            n_factors  = len(FACTOR_NAMES),
        ).to(self.device)

    def train(self,
              n_samples_per_factor = 500,
              epochs               = 30,
              lr                   = 1e-3,
              batch_size           = 64,
              val_split            = 0.15):
        """
        Full training loop.

        Returns trained FactorMLP.
        """
        # Build dataset
        X, y = build_training_data(
            self.diagnoser, self.cfg,
            n_samples_per_factor=n_samples_per_factor)

        # Train / val split
        n_val   = int(len(X) * val_split)
        idx     = torch.randperm(len(X))
        X_train, y_train = X[idx[n_val:]], y[idx[n_val:]]
        X_val,   y_val   = X[idx[:n_val]], y[idx[:n_val]]

        train_ds = TensorDataset(X_train, y_train)
        val_ds   = TensorDataset(X_val,   y_val)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,
                                     weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)

        best_val_acc = 0.0
        best_state   = None

        logger.info(f"Training FactorMLP for {epochs} epochs...")
        logger.info(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

        for epoch in range(1, epochs + 1):
            # ---- Train ----
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss   = F.cross_entropy(logits, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(train_ds)

            # ---- Validate ----
            self.model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb  = xb.to(self.device), yb.to(self.device)
                    preds   = self.model(xb).argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total   += len(yb)
            val_acc = correct / total

            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state   = {k: v.clone()
                                for k, v in self.model.state_dict().items()}

            if epoch % 5 == 0:
                logger.info(f"  Epoch {epoch:3d}/{epochs}  "
                            f"loss={train_loss:.4f}  "
                            f"val_acc={val_acc:.1%}")

        # Restore best
        self.model.load_state_dict(best_state)
        logger.info(f"Training complete. Best val accuracy: "
                    f"{best_val_acc:.1%}")
        return self.model, best_val_acc

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "factor_names": FACTOR_NAMES,
        }, path)
        logger.info(f"Factor classifier saved → {path}")


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

class FactorAttributor:
    """
    Uses trained FactorMLP to attribute domain uncertainty to factors.
    Drop-in replacement for the heuristic direction-based attribution.
    """

    def __init__(self, diagnoser, cfg):
        self.diagnoser = diagnoser
        self.cfg       = cfg
        self.device    = diagnoser.device
        self.model     = FactorMLP(
            feat_dim   = 1024,
            hidden_dim = 256,
            n_factors  = len(FACTOR_NAMES),
        ).to(self.device)
        self.model.eval()

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        logger.info(f"Factor classifier loaded from {path}")
        return self

    @torch.no_grad()
    def attribute(self, x_test):
        """
        Returns per-factor attribution probabilities.

        Returns
        -------
        dict  factor_name → (B,) probability tensor
        """
        feats     = self.diagnoser._extract_features(x_test)
        feat_diff = ((feats - self.diagnoser.chexpert_mean)
                     / self.diagnoser.chexpert_std)

        logits = self.model(feat_diff)              # (B, 4)
        probs  = F.softmax(logits, dim=1)           # (B, 4)

        return {
            FACTOR_NAMES[i]: probs[:, i]
            for i in range(len(FACTOR_NAMES))
        }

    @torch.no_grad()
    def attribute_with_uncertainty(self, x_test, n_samples=10):
        """
        MC Dropout on the factor classifier for attribution uncertainty.
        Returns mean ± std per factor per image.
        """
        # Enable dropout for uncertainty
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        feats     = self.diagnoser._extract_features(x_test)
        feat_diff = ((feats - self.diagnoser.chexpert_mean)
                     / self.diagnoser.chexpert_std)

        all_probs = []
        for _ in range(n_samples):
            logits = self.model(feat_diff)
            all_probs.append(F.softmax(logits, dim=1))

        self.model.eval()

        probs_stack = torch.stack(all_probs)        # (n_samples, B, 4)
        mean_probs  = probs_stack.mean(dim=0)       # (B, 4)
        std_probs   = probs_stack.std(dim=0)        # (B, 4)

        mean_attrs = {FACTOR_NAMES[i]: mean_probs[:, i]
                      for i in range(len(FACTOR_NAMES))}
        std_attrs  = {FACTOR_NAMES[i]: std_probs[:, i]
                      for i in range(len(FACTOR_NAMES))}

        return mean_attrs, std_attrs