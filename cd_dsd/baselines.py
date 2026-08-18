"""
Baseline out-of-distribution / domain-shift detection methods, for comparing
against CD-DSD's U_domain on the same images. All baselines reuse the SAME
frozen classifier + diffusion model (no extra training beyond a lightweight
calibration pass), so the comparison isolates the *scoring mechanism*, not
model capacity.

References, in publication-year order:

  [2017] MSP          Hendrycks & Gimpel. "A Baseline for Detecting Misclassified
                       and Out-of-Distribution Examples." ICLR 2017.
  [2018] Mahalanobis  Lee et al. "A Simple Unified Framework for Detecting OOD
                       Samples and Adversarial Attacks." NeurIPS 2018.
  [2020] Energy        Liu et al. "Energy-based Out-of-distribution Detection."
                       NeurIPS 2020.
  [2021] ReAct         Sun et al. "ReAct: Out-of-distribution Detection With
                       Rectified Activations." NeurIPS 2021.
  [2022] ViM           Wang et al. "ViM: Out-Of-Distribution with Virtual-logit
                       Matching." CVPR 2022.
  [2022] KNN           Sun et al. "Out-of-Distribution Detection with Deep
                       Nearest Neighbors." ICML 2022.
  [2022] DICE          Sun & Li. "DICE: Leveraging Sparsification for OOD
                       Detection." ECCV 2022.
  [2023] GEN           Liu et al. "GEN: Pushing the Limits of Softmax-Based
                       OOD Detection." CVPR 2023.
  [2023] ASH           Djurisic et al. "Extremely Simple Activation Shaping for
                       OOD Detection." ICLR 2023.
  [2023] DDA           Gao et al. "Back to the Source: Diffusion-Driven
                       Test-Time Adaptation." CVPR 2023.
  [2024] DiffPath*     Bounoua et al. "OOD Detection with a Single Unconditional
                       Diffusion Model." NeurIPS 2024 (arXiv:2405.11881).
  [2026] EigenScore*   Shoushtari et al. "EigenScore: OOD Detection using
                       Posterior Covariance in Diffusion Models." ICLR 2026
                       (arXiv:2510.07206).

  * DiffPath and EigenScore are re-implemented IN SPIRIT using Monte-Carlo
    approximations. Report these as simplified reproductions, not exact
    replications of the published method.

  NOTE ON MULTI-LABEL ADAPTATION: Methods designed for single-label softmax
  classification (MSP, Energy, GEN, ReAct, DICE, ASH, ViM) apply softmax over
  the 8 raw logits for a comparable probability simplex. State this adaptation
  explicitly if reported in a paper.
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from cd_dsd.datasets import CheXpertDataset, get_eval_transform, collate_fn


logger = logging.getLogger(__name__)


class BaselineSuite:
    """
    Fits on a clean, in-domain (CheXpert) calibration set, then scores new
    batches with each baseline. All `score_*` methods return a (B,) tensor
    where HIGHER = more anomalous / more domain-shifted, so every method is
    directly comparable via AUROC against the same clean-vs-corrupted labels.
    """

    def __init__(self, diagnoser, cfg):
        self.diagnoser = diagnoser
        self.cfg       = cfg
        self.device    = diagnoser.device
        self.fitted    = False

    # ------------------------------------------------------------------
    # Fit on clean CheXpert calibration data
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit(self, n_samples=1000, knn_k=10, vim_var_explained=0.9,
            react_percentile=90, dice_percentile=70):
        logger.info(f"Fitting BaselineSuite on {n_samples} clean CheXpert images...")
        eval_tf = get_eval_transform(self.cfg.IMAGE_SIZE, self.cfg.MEAN, self.cfg.STD)
        ds = CheXpertDataset(
            csv_path   = self.cfg.CHEXPERT_TRAIN_CSV,
            image_root = self.cfg.CHEXPERT_IMAGE_ROOT,
            label_cols = self.cfg.LABEL_COLS,
            transform  = eval_tf,
        )
        loader = DataLoader(ds, batch_size=32, shuffle=True,
                            collate_fn=collate_fn, num_workers=2)

        feats_all, logits_all = [], []
        n_collected = 0
        for imgs, _, _ in loader:
            imgs = imgs.to(self.device)
            feats_all.append(self.diagnoser._extract_features(imgs))
            logits_all.append(self.diagnoser.classifier(imgs))
            n_collected += imgs.shape[0]
            if n_collected >= n_samples:
                break
        feats  = torch.cat(feats_all)[:n_samples]    # (N, 1024)
        logits = torch.cat(logits_all)[:n_samples]   # (N, 8)

        # ---- Mahalanobis (reuse diagnoser's already-calibrated stats) ----
        # diagnoser.chexpert_mean / chexpert_std computed in CDDSDDiagnoser._calibrate()

        # ---- KNN (Sun et al. ICML 2022): bank of L2-normalised features ----
        self.knn_bank = F.normalize(feats, dim=1)     # (N, 1024)
        self.knn_k    = knn_k

        # ---- ViM (Wang et al. CVPR 2022): principal-subspace residual ----
        feat_mean = feats.mean(0)
        centered  = feats - feat_mean
        cov       = (centered.T @ centered) / (len(centered) - 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)             # ascending order
        eigvals = eigvals.flip(0); eigvecs = eigvecs.flip(1)  # descending
        explained = torch.cumsum(eigvals, 0) / eigvals.sum()
        k = int((explained < vim_var_explained).sum().item()) + 1
        k = max(1, min(k, feats.shape[1] - 1))
        self.vim_mean           = feat_mean
        self.vim_residual_basis = eigvecs[:, k:]              # (1024, 1024-k) null space
        residual_norm_bank = (centered @ self.vim_residual_basis).norm(dim=1)
        energy_bank = torch.logsumexp(logits, dim=1)
        self.vim_alpha = (energy_bank.mean() / residual_norm_bank.mean().clamp(min=1e-6)).item()
        logger.info(f"  ViM: kept {k}/{feats.shape[1]} principal dims, alpha={self.vim_alpha:.4f}")

        # ---- ReAct (Sun et al. NeurIPS 2021): clip threshold at p-th pct ----
        self.react_threshold = torch.quantile(
            feats.flatten(), react_percentile / 100.0).item()
        logger.info(f"  ReAct: threshold={self.react_threshold:.4f} (p={react_percentile})")

        # ---- DICE (Sun & Li, ECCV 2022): sparse FC via contribution masking ----
        fc_weight    = self.diagnoser.classifier.fc.weight     # (8, 1024)
        mean_act     = feats.mean(0)                           # (1024,)
        contribution = (fc_weight.abs() * mean_act.unsqueeze(0)).mean(0)  # (1024,)
        dice_thresh  = torch.quantile(contribution, dice_percentile / 100.0)
        self.dice_mask = (contribution >= dice_thresh).float() # (1024,)
        logger.info(f"  DICE: keeping {self.dice_mask.sum().int().item()}/1024 features "
                    f"(p={dice_percentile})")

        self.fitted = True
        logger.info("BaselineSuite fit complete.")

    # ------------------------------------------------------------------
    # Individual baseline scores  (higher = more anomalous)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_msp(self, x):
        """[2017] Hendrycks & Gimpel. 1 - max class probability."""
        probs = torch.sigmoid(self.diagnoser.classifier(x))
        return 1.0 - probs.max(dim=1).values

    @torch.no_grad()
    def score_mahalanobis(self, x):
        """[2018] Lee et al. Diagonal Mahalanobis distance from in-domain feature mean."""
        feats = self.diagnoser._extract_features(x)
        diff  = (feats - self.diagnoser.chexpert_mean) / self.diagnoser.chexpert_std
        return diff.pow(2).mean(dim=1)

    @torch.no_grad()
    def score_energy(self, x):
        """[2020] Liu et al. Negative log-sum-exp of logits ("free energy")."""
        logits = self.diagnoser.classifier(x)
        return -torch.logsumexp(logits, dim=1)

    @torch.no_grad()
    def score_vim(self, x):
        """[2022] Wang et al. Virtual-logit matching: residual-subspace norm vs. logit energy."""
        assert self.fitted, "Call .fit() first"
        feats    = self.diagnoser._extract_features(x)
        centered = feats - self.vim_mean
        residual = (centered @ self.vim_residual_basis).norm(dim=1)
        vlogit   = self.vim_alpha * residual
        logits   = self.diagnoser.classifier(x)
        energy   = torch.logsumexp(logits, dim=1)
        return vlogit - energy

    @torch.no_grad()
    def score_knn(self, x):
        """[2022] Sun et al. Euclidean distance to k-th nearest neighbour in feature bank."""
        assert self.fitted, "Call .fit() first"
        feats = F.normalize(self.diagnoser._extract_features(x), dim=1)
        dists = torch.cdist(feats, self.knn_bank)         # (B, N)
        kth, _ = dists.topk(self.knn_k, dim=1, largest=False)
        return kth[:, -1]

    @torch.no_grad()
    def score_gen(self, x, gamma=0.1):
        """[2023] Liu et al. Generalized entropy over the (softmax-adapted) class distribution."""
        logits = self.diagnoser.classifier(x)
        probs  = F.softmax(logits, dim=1).clamp(min=1e-12)
        pg     = probs.pow(gamma)
        return -(pg * pg.clamp(min=1e-12).log()).sum(dim=1)

    @torch.no_grad()
    def score_dda(self, x, t_star=None):
        """
        [2023] Gao et al. "Back to the Source" -- project x toward the source
        domain via the diffusion model, score = shift in predicted probabilities.
        """
        t_star = t_star or self.cfg.T_STAR_FULL
        x_corrected = self.diagnoser.diffusion.partial_correct(
            x, t_star, num_steps=self.cfg.DDIM_STEPS, eta=self.cfg.DDIM_ETA
        )
        p_orig = torch.sigmoid(self.diagnoser.classifier(x))
        p_corr = torch.sigmoid(self.diagnoser.classifier(x_corrected))
        return (p_orig - p_corr).abs().mean(dim=1)

    @torch.no_grad()
    def score_diffpath(self, x, timesteps=(200, 400, 600)):
        """
        [2024] Bounoua et al. (simplified). Curvature of the predicted-x0
        path across three timesteps, sharing one noise draw per image so the
        three points lie on a single forward trajectory.
        """
        diff = self.diagnoser.diffusion
        eps  = torch.randn_like(x)
        pred_x0s = []
        for t in timesteps:
            ab_t = diff.alpha_bar[t]
            x_t  = ab_t.sqrt() * x + (1 - ab_t).sqrt() * eps
            t_batch  = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            eps_pred = diff.unet(x_t, t_batch)
            pred_x0  = (x_t - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
            pred_x0s.append(pred_x0)
        curvature = pred_x0s[2] - 2 * pred_x0s[1] + pred_x0s[0]
        return curvature.pow(2).mean(dim=(1, 2, 3))

    @torch.no_grad()
    def score_eigenscore(self, x, t=400, n_repeats=5):
        """
        [2026] Shoushtari et al. (simplified). Monte-Carlo trace of the
        denoiser's posterior covariance: variance of predicted-x0 across
        repeated noise draws at a fixed timestep.
        """
        diff = self.diagnoser.diffusion
        ab_t = diff.alpha_bar[t]
        pred_x0s = []
        for _ in range(n_repeats):
            eps      = torch.randn_like(x)
            x_t      = ab_t.sqrt() * x + (1 - ab_t).sqrt() * eps
            t_batch  = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            eps_pred = diff.unet(x_t, t_batch)
            pred_x0  = (x_t - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
            pred_x0s.append(pred_x0)
        stacked = torch.stack(pred_x0s)                  # (R, B, C, H, W)
        trace_approx = stacked.var(dim=0).mean(dim=(1, 2, 3))   # (B,)
        return trace_approx

    @torch.no_grad()
    def score_react(self, x):
        """[2021] Sun et al. ReAct: rectify (clip) pre-FC activations, then energy."""
        assert self.fitted, "Call .fit() first"
        feats  = self.diagnoser._extract_features(x).clamp(max=self.react_threshold)
        fc     = self.diagnoser.classifier.fc
        logits = feats @ fc.weight.T + fc.bias
        return -torch.logsumexp(logits, dim=1)

    @torch.no_grad()
    def score_dice(self, x):
        """[2022] Sun & Li. DICE: sparsified FC weight → energy score."""
        assert self.fitted, "Call .fit() first"
        feats      = self.diagnoser._extract_features(x)
        fc         = self.diagnoser.classifier.fc
        sparse_w   = fc.weight * self.dice_mask.unsqueeze(0)
        logits     = feats @ sparse_w.T + fc.bias
        return -torch.logsumexp(logits, dim=1)

    @torch.no_grad()
    def score_ash(self, x, p=65):
        """[2023] Djurisic et al. ASH-S: remove bottom p% activations, scale remainder."""
        feats  = self.diagnoser._extract_features(x).clone()
        s1     = feats.sum(dim=1, keepdim=True)
        thresh = torch.quantile(feats, p / 100.0, dim=1, keepdim=True)
        feats  = feats * (feats >= thresh).float()
        s2     = feats.sum(dim=1, keepdim=True).clamp(min=1e-12)
        feats  = feats * (s1 / s2)
        fc     = self.diagnoser.classifier.fc
        logits = feats @ fc.weight.T + fc.bias
        return -torch.logsumexp(logits, dim=1)

    @torch.no_grad()
    def score_cd_dsd(self, x):
        """CD-DSD's U_domain: diffusion denoising-error + HFER (calibrated)."""
        from cd_dsd.diagnoser import hf_energy_score
        domain_raw = self.diagnoser._domain_score_raw(x)
        domain_z   = ((domain_raw - self.diagnoser.domain_baseline_mean)
                      / self.diagnoser.domain_baseline_std).clamp(min=0.0)
        hfer_raw   = hf_energy_score(x, self.cfg.HFER_RADIUS_FRAC)
        hfer_z     = (-(hfer_raw - self.diagnoser.hfer_mean)
                      / self.diagnoser.hfer_std).clamp(min=0.0)
        combined_z = domain_z + self.cfg.HFER_WEIGHT * hfer_z
        return self.diagnoser.domain_weight * combined_z

    # ------------------------------------------------------------------

    def all_methods(self):
        return {
            "MSP (2017)":         self.score_msp,
            "Mahalanobis (2018)": self.score_mahalanobis,
            "Energy (2020)":      self.score_energy,
            "ReAct (2021)":       self.score_react,
            "ViM (2022)":         self.score_vim,
            "KNN (2022)":         self.score_knn,
            "DICE (2022)":        self.score_dice,
            "GEN (2023)":         self.score_gen,
            "ASH (2023)":         self.score_ash,
            "DDA (2023)":         self.score_dda,
            "DiffPath (2024)*":   self.score_diffpath,
            "EigenScore (2026)*": self.score_eigenscore,
            "CD-DSD (ours)":      self.score_cd_dsd,
        }