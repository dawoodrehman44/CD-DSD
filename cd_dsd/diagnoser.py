import os
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader

from cd_dsd.unet import UNet
from cd_dsd.diffusion import DiffusionModel
from cd_dsd.classifier import MCDropoutClassifier
from cd_dsd.factor_mlp import FactorMLP, FactorAttributor, FactorClassifierTrainer, FACTOR_NAMES
from cd_dsd.datasets import CheXpertDataset, get_eval_transform, collate_fn

logger = logging.getLogger(__name__)

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denorm(t: torch.Tensor) -> np.ndarray:
    img = (t.cpu().float() * _STD + _MEAN).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def hf_energy_score(x: torch.Tensor, hf_radius_frac: float = 0.25) -> torch.Tensor:
    """FFT High-Frequency Energy Ratio (HFER).

    Returns a (B,) tensor. Clean images score HIGH (lots of HF content);
    blurred / low-resolution images score LOW. Invert before using as an
    anomaly score so that lower HFER → higher anomaly score.
    """
    B, C, H, W = x.shape
    gray = x.mean(dim=1, keepdim=True)
    fft  = torch.fft.fft2(gray)
    fft  = torch.fft.fftshift(fft, dim=(-2, -1))
    mag  = fft.abs().pow(2)
    cy, cx = H // 2, W // 2
    ys = torch.arange(H, device=x.device).float() - cy
    xs = torch.arange(W, device=x.device).float() - cx
    r  = torch.sqrt(ys[:, None].pow(2) + xs[None, :].pow(2))
    max_r    = min(cy, cx)
    hf_mask  = (r > hf_radius_frac * max_r).float()
    hf_power    = (mag * hf_mask).sum(dim=(-3, -2, -1))
    total_power = mag.sum(dim=(-3, -2, -1)).clamp(min=1e-12)
    return (hf_power / total_power).squeeze()


class CDDSDDiagnoser:

    def __init__(self, cfg):
        self.cfg    = cfg
        self.device = torch.device(cfg.DEVICE)
        self.classifier = self._load_classifier()
        self.diffusion  = self._load_diffusion()
        self._calibrate()
        self.factor_attributor = self._load_factor_attributor()

    def _load_classifier(self):
        ckpt_path = os.path.join(self.cfg.CHECKPOINT_DIR, self.cfg.CLS_CKPT_NAME)
        model = MCDropoutClassifier(
            num_classes  = self.cfg.NUM_CLASSES,
            dropout_rate = self.cfg.CLS_DROPOUT,
            pretrained   = False,
        ).to(self.device)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(ckpt["model_state"])
            logger.info(f"Classifier loaded from {ckpt_path}  "
                        f"(best AUC={ckpt.get('val_auc', '?')})")
        else:
            raise FileNotFoundError(
                f"Classifier checkpoint NOT found at {ckpt_path}. "
                f"Check cfg.CLS_CKPT_NAME / cfg.CHECKPOINT_DIR."
            )
        return model

    def _load_diffusion(self):
        ckpt_path = os.path.join(self.cfg.CHECKPOINT_DIR, "diffusion_best.pt")
        unet = UNet(
            in_channels   = 3,
            base_channels = self.cfg.DIFF_BASE_CHANNELS,
            channel_mults = self.cfg.DIFF_CHANNEL_MULTS,
            dropout       = self.cfg.DIFF_DROPOUT,
            attn_depths   = self.cfg.DIFF_ATTN_DEPTHS,
        )
        model = DiffusionModel(
            unet       = unet,
            T          = self.cfg.DIFF_T,
            beta_start = self.cfg.DIFF_BETA_START,
            beta_end   = self.cfg.DIFF_BETA_END,
        ).to(self.device)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device)
            model.load_state_dict(ckpt["model_state"])
            logger.info(f"Diffusion model loaded from {ckpt_path}")
        else:
            raise FileNotFoundError(f"Diffusion checkpoint NOT found at {ckpt_path}.")
        return model

    @torch.no_grad()
    def _calibrate(self, n_samples=200):
        logger.info("Calibrating on clean CheXpert reference set...")
        eval_tf = get_eval_transform(self.cfg.IMAGE_SIZE, self.cfg.MEAN, self.cfg.STD)
        ds = CheXpertDataset(
            csv_path   = self.cfg.CHEXPERT_TRAIN_CSV,
            image_root = self.cfg.CHEXPERT_IMAGE_ROOT,
            label_cols = self.cfg.LABEL_COLS,
            transform  = eval_tf,
        )
        loader = DataLoader(ds, batch_size=16, shuffle=True,
                            collate_fn=collate_fn, num_workers=2)

        all_feats, all_domain_raw, all_semantic, all_hfer = [], [], [], []
        n_collected = 0
        for imgs, _, _ in loader:
            imgs = imgs.to(self.device)
            all_feats.append(self._extract_features(imgs))
            all_domain_raw.append(self._domain_score_raw(imgs))
            all_semantic.append(self._uncertainty(imgs))
            all_hfer.append(hf_energy_score(imgs, self.cfg.HFER_RADIUS_FRAC))
            n_collected += imgs.shape[0]
            if n_collected >= n_samples:
                break

        feats = torch.cat(all_feats)[:n_samples]
        self.chexpert_mean = feats.mean(0)
        self.chexpert_std  = feats.std(0).clamp(min=1e-6)

        domain_raw = torch.cat(all_domain_raw)[:n_samples]
        self.domain_baseline_mean = domain_raw.mean()
        self.domain_baseline_std  = domain_raw.std().clamp(min=1e-6)

        hfer_raw = torch.cat(all_hfer)[:n_samples]
        self.hfer_mean = hfer_raw.mean()
        self.hfer_std  = hfer_raw.std().clamp(min=1e-6)

        semantic = torch.cat(all_semantic)[:n_samples]
        self.domain_weight = semantic.std().clamp(min=1e-6)

        logger.info(
            f"Calibration done. domain_baseline_mean={self.domain_baseline_mean.item():.5f}  "
            f"domain_baseline_std={self.domain_baseline_std.item():.5f}  "
            f"hfer_mean={self.hfer_mean.item():.5f}  hfer_std={self.hfer_std.item():.5f}  "
            f"domain_weight={self.domain_weight.item():.5f}"
        )

    def _load_factor_attributor(self):
        ckpt_path  = os.path.join(self.cfg.CHECKPOINT_DIR, "factor_classifier.pt")
        attributor = FactorAttributor(self, self.cfg)
        if os.path.exists(ckpt_path):
            attributor.load(ckpt_path)
        else:
            logger.info("Factor classifier not found — training now...")
            trainer = FactorClassifierTrainer(self, self.cfg)
            trainer.train(n_samples_per_factor=500, epochs=30)
            trainer.save(ckpt_path)
            attributor.load(ckpt_path)
        return attributor

    @torch.no_grad()
    def _extract_features(self, x):
        self.classifier.eval()
        feats = self.classifier.features(x)
        feats = torch.nn.functional.relu(feats, inplace=False)
        feats = self.classifier.pool(feats).flatten(1)
        return feats

    @torch.no_grad()
    def _domain_score_raw(self, x):
        scores = [
            self.diffusion.domain_score(x, n_timesteps=self.cfg.DOMAIN_SCORE_TIMESTEPS)
            for _ in range(self.cfg.DOMAIN_SCORE_REPEATS)
        ]
        return torch.stack(scores).mean(dim=0)

    def _uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier.uncertainty_scalar(x, n_samples=self.cfg.MC_SAMPLES)

    @torch.no_grad()
    def diagnose_batch(self, x_test, domain="unknown", save_vis=True,
                       save_dir=None, sample_ids=None):
        x_test   = x_test.to(self.device)
        B        = x_test.shape[0]
        save_dir = save_dir or self.cfg.RESULTS_DIR
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        if sample_ids is None:
            sample_ids = list(range(B))

        u_semantic = self._uncertainty(x_test)

        domain_raw    = self._domain_score_raw(x_test)
        domain_z      = ((domain_raw - self.domain_baseline_mean)
                          / self.domain_baseline_std).clamp(min=0.0)

        # HFER: lower ratio = more blur/resolution shift = higher anomaly score
        hfer_raw = hf_energy_score(x_test, self.cfg.HFER_RADIUS_FRAC)
        hfer_z   = (-(hfer_raw - self.hfer_mean) / self.hfer_std).clamp(min=0.0)

        combined_z    = domain_z + self.cfg.HFER_WEIGHT * hfer_z
        u_domain      = self.domain_weight * combined_z
        u_total       = u_semantic + u_domain
        domain_fraction = u_domain / (u_total + 1e-7)

        x_star = self.diffusion.partial_correct(
            x_test, self.cfg.T_STAR_FULL,
            num_steps=self.cfg.DDIM_STEPS, eta=self.cfg.DDIM_ETA
        )

        level_names = sorted(self.cfg.T_STAR_LEVELS,
                             key=lambda k: self.cfg.T_STAR_LEVELS[k])
        level_data  = {}
        for factor_name in level_names:
            t_k = self.cfg.T_STAR_LEVELS[factor_name]
            if factor_name == "brightness_contrast":
                x_k = self.diffusion.brightness_correct(x_test)
            else:
                x_k = self.diffusion.partial_correct(
                    x_test, t_k,
                    num_steps = self.cfg.DDIM_STEPS,
                    eta       = self.cfg.DDIM_ETA,
                )
            level_data[factor_name] = dict(x=x_k)

        mean_attrs, std_attrs = self.factor_attributor.attribute_with_uncertainty(
            x_test, n_samples=10)

        factor_attributions_batch = [{} for _ in range(B)]
        for factor_name in level_names:
            attr_k = mean_attrs[factor_name]
            for b in range(B):
                factor_attributions_batch[b][factor_name] = attr_k[b].item()

        pred_orig = self.classifier.predict(x_test)
        pred_corr = self.classifier.predict(x_star)

        results = []
        for b in range(B):
            result = dict(
                sample_id           = sample_ids[b],
                domain              = domain,
                u_total             = u_total[b].item(),
                u_domain            = u_domain[b].item(),
                u_domain_raw        = domain_raw[b].item(),
                u_hfer_raw          = hfer_raw[b].item(),
                u_hfer_z            = hfer_z[b].item(),
                u_semantic          = u_semantic[b].item(),
                domain_fraction     = domain_fraction[b].item(),
                factor_attributions = factor_attributions_batch[b],
                pred_original       = pred_orig[b].cpu().tolist(),
                pred_corrected      = pred_corr[b].cpu().tolist(),
                label_cols          = self.cfg.LABEL_COLS,
            )
            if save_vis:
                vis_path = os.path.join(
                    save_dir, f"{domain}_sample{sample_ids[b]}_diagnosis.png")
                self._save_visualisation(
                    x_orig       = x_test[b],
                    x_corrected  = x_star[b],
                    level_images = {k: v["x"][b] for k, v in level_data.items()},
                    result       = result,
                    save_path    = vis_path,
                )
                result["vis_path"] = vis_path
            results.append(result)

        return results

    def _save_visualisation(self, x_orig, x_corrected, level_images,
                            result, save_path):
        n_levels   = len(level_images)
        total_imgs = 2 + n_levels
        fig_w      = 3 * total_imgs + 2
        fig        = plt.figure(figsize=(fig_w, 7))
        gs         = gridspec.GridSpec(2, total_imgs, figure=fig,
                                       hspace=0.4, wspace=0.3)

        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(denorm(x_orig), cmap="gray")
        ax.set_title("Original\n(test)", fontsize=9)
        ax.axis("off")

        for i, (name, x_k) in enumerate(level_images.items()):
            ax = fig.add_subplot(gs[0, i + 1])
            ax.imshow(denorm(x_k), cmap="gray")
            ax.set_title(f"{name.replace('_', chr(10))}\ncorrection (illustrative)", fontsize=8)
            ax.axis("off")

        ax = fig.add_subplot(gs[0, -1])
        ax.imshow(denorm(x_corrected), cmap="gray")
        ax.set_title("Full\ncorrection\n(x*, illustrative)", fontsize=9)
        ax.axis("off")

        ax_unc = fig.add_subplot(gs[1, :total_imgs // 2])
        ax_unc.bar(
            ["Domain\nUncertainty", "Semantic\nUncertainty"],
            [result["u_domain"], result["u_semantic"]],
            color=["#E74C3C", "#3498DB"], edgecolor="k", width=0.5
        )
        ax_unc.set_title(
            f"Total U={result['u_total']:.3f}  "
            f"Domain fraction={result['domain_fraction']:.1%}",
            fontsize=9)
        ax_unc.set_ylabel("Uncertainty")

        ax_attr = fig.add_subplot(gs[1, total_imgs // 2:])
        attrs = result["factor_attributions"]
        ax_attr.barh(
            [k.replace("_", "\n") for k in attrs],
            list(attrs.values()),
            color="#2ECC71", edgecolor="k"
        )
        ax_attr.set_title("Factor Attribution\n(FactorMLP probability)", fontsize=9)
        ax_attr.set_xlabel("P(factor)")

        fig.suptitle(
            f"CD-DSD  |  domain={result['domain']}  |  sample={result['sample_id']}",
            fontsize=10, fontweight="bold")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)


def evaluate_domain(diagnoser, dataset, domain_name, max_samples, batch_size=8):
    from cd_dsd.datasets import collate_fn
    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 4,
        pin_memory  = True,
    )

    all_results = []
    n_done      = 0

    for imgs, _labels, indices in loader:
        if n_done >= max_samples:
            break
        cap     = min(imgs.shape[0], max_samples - n_done)
        imgs    = imgs[:cap]
        ids     = list(indices[:cap])
        results = diagnoser.diagnose_batch(
            imgs, domain=domain_name, save_vis=True, sample_ids=ids)
        all_results.extend(results)
        n_done += cap
        logger.info(f"  {domain_name}: {n_done}/{max_samples} diagnosed")

    return all_results
