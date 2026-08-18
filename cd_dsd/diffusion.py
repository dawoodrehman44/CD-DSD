"""
DDPM noise schedule + DDIM sampler.

Key functions used by CD-DSD:
  - DiffusionModel.train_step()       : standard DDPM loss
  - DiffusionModel.domain_score()     : denoising-error domain-shift score
                                         used directly as U_domain in CD-DSD
  - DiffusionModel.partial_correct()  : SDEdit-style correction, used ONLY
                                         for the illustrative visualization
                                         panel (brightness/noise/scanner/
                                         structure images) — not part of
                                         any reported score
  - DiffusionModel.ddim_sample()      : full DDIM denoising from noise
  - DiffusionModel.ddim_invert()      : deterministic encode x0 → x_T,
                                         kept as a utility, currently unused
                                         by CD-DSD
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Noise schedule helpers
# ---------------------------------------------------------------------------

def linear_beta_schedule(T: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    f = torch.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.9999)


def precompute_schedule(betas: torch.Tensor) -> dict:
    alphas           = 1.0 - betas
    alpha_bar        = torch.cumprod(alphas, dim=0)
    alpha_bar_prev   = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

    sqrt_ab          = alpha_bar.sqrt()
    sqrt_one_m_ab    = (1.0 - alpha_bar).sqrt()
    sqrt_recip_ab    = (1.0 / alpha_bar).sqrt()
    sqrt_recip_m1_ab = (1.0 / alpha_bar - 1.0).sqrt()

    post_var         = betas * (1 - alpha_bar_prev) / (1 - alpha_bar)
    post_mean_c1     = betas * alpha_bar_prev.sqrt() / (1 - alpha_bar)
    post_mean_c2     = (1 - alpha_bar_prev) * alphas.sqrt() / (1 - alpha_bar)

    return dict(
        betas            = betas,
        alphas           = alphas,
        alpha_bar        = alpha_bar,
        alpha_bar_prev   = alpha_bar_prev,
        sqrt_ab          = sqrt_ab,
        sqrt_one_m_ab    = sqrt_one_m_ab,
        sqrt_recip_ab    = sqrt_recip_ab,
        sqrt_recip_m1_ab = sqrt_recip_m1_ab,
        post_var         = post_var,
        post_mean_c1     = post_mean_c1,
        post_mean_c2     = post_mean_c2,
    )


# ---------------------------------------------------------------------------
# DiffusionModel
# ---------------------------------------------------------------------------

class DiffusionModel(nn.Module):

    def __init__(
        self,
        unet,
        T:          int   = 1000,
        beta_start: float = 1e-4,
        beta_end:   float = 0.02,
        schedule:   str   = "linear",
    ):
        super().__init__()
        self.unet = unet
        self.T    = T

        if schedule == "cosine":
            betas = cosine_beta_schedule(T)
        else:
            betas = linear_beta_schedule(T, beta_start, beta_end)

        sched = precompute_schedule(betas)
        for k, v in sched.items():
            self.register_buffer(k, v.float())

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def train_step(self, x0: torch.Tensor) -> torch.Tensor:
        B      = x0.shape[0]
        device = x0.device
        t      = torch.randint(0, self.T, (B,), device=device)
        eps    = torch.randn_like(x0)
        x_t    = (
            self.sqrt_ab[t, None, None, None] * x0
            + self.sqrt_one_m_ab[t, None, None, None] * eps
        )
        eps_pred = self.unet(x_t, t)
        return F.mse_loss(eps_pred, eps)

    # ------------------------------------------------------------------
    # Internal: single DDIM reverse step  x_t → x_{t_prev}
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _ddim_step(self, x_t, t, t_prev, eta=0.0):
        device   = x_t.device
        t_batch  = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)

        eps_pred = self.unet(x_t, t_batch)
        ab_t     = self.alpha_bar[t]
        ab_tp    = self.alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)

        pred_x0  = (x_t - self.sqrt_one_m_ab[t] * eps_pred) / self.sqrt_ab[t]
        pred_x0  = pred_x0.clamp(-4, 4)

        sigma_t  = eta * ((1 - ab_tp) / (1 - ab_t) * (1 - ab_t / ab_tp)).sqrt()
        dir_xt   = (1 - ab_tp - sigma_t ** 2).sqrt() * eps_pred
        noise    = sigma_t * torch.randn_like(x_t) if eta > 0 else 0

        return ab_tp.sqrt() * pred_x0 + dir_xt + noise

    # ------------------------------------------------------------------
    # Internal: single DDIM FORWARD step  x_t → x_{t_next}  (inversion)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _ddim_invert_step(self, x_t, t, t_next):
        """
        One deterministic forward step used during DDIM inversion.
        Goes x_t → x_{t_next} where t_next > t (moving toward more noise).
        """
        device  = x_t.device
        t_batch = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)

        eps_pred = self.unet(x_t, t_batch)
        ab_t     = self.alpha_bar[t]
        ab_next  = self.alpha_bar[t_next]

        # DDIM inversion formula (deterministic, eta=0)
        pred_x0 = (x_t - (1 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()
        pred_x0 = pred_x0.clamp(-4, 4)

        x_next  = ab_next.sqrt() * pred_x0 + (1 - ab_next).sqrt() * eps_pred
        return x_next

    # ------------------------------------------------------------------
    # Full DDIM sampling  noise → image
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(self, x_T, num_steps=50, eta=0.0, verbose=False):
        timesteps = torch.linspace(self.T - 1, 0, num_steps + 1).long().tolist()
        x     = x_T
        pairs = list(zip(timesteps[:-1], timesteps[1:]))
        for t, t_prev in (tqdm(pairs, desc="DDIM sample") if verbose else pairs):
            x = self._ddim_step(x, int(t), int(t_prev), eta)
        return x

    # ------------------------------------------------------------------
    # DDIM Inversion  x0 → x_T  (deterministic encoding)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_invert(self, x0, num_steps=50):
        """
        Deterministically encode x0 → x_T using DDIM inversion.

        In-domain images (CheXpert) encode cleanly because the diffusion
        model knows their distribution well.
        Out-of-domain images (MIMIC, NIH) encode poorly — the model's
        noise predictions are inaccurate, so the encoding drifts.
        This drift is what makes the reconstruction distance meaningful.
        """
        timesteps = torch.linspace(0, self.T - 1, num_steps + 1).long().tolist()
        x = x0
        for t, t_next in zip(timesteps[:-1], timesteps[1:]):
            x = self._ddim_invert_step(x, int(t), int(t_next))
        return x   # x_T

    # ------------------------------------------------------------------
    # Invert + Reconstruct  ← Core domain correction for CD-DSD
    # ------------------------------------------------------------------

    @torch.no_grad()
    def domain_score(self, x0, n_timesteps=10):
        """
        Compute domain shift score as mean noise prediction error
        across multiple timesteps.
        
        Low score  → image looks like training data (CheXpert)
        High score → image is out-of-domain (MIMIC, NIH)
        """
        device = x0.device
        B      = x0.shape[0]
        errors = []

        # Sample n_timesteps evenly spread between t=100 and t=700
        # (avoid very low t where all images look similar,
        #  avoid very high t where everything is pure noise)
        timesteps = torch.linspace(100, 700, n_timesteps).long().to(device)

        for t_val in timesteps:
            t_batch = t_val.expand(B)
            eps     = torch.randn_like(x0)

            # Corrupt image to timestep t
            x_t = (
                self.sqrt_ab[t_val] * x0
                + self.sqrt_one_m_ab[t_val] * eps
            )

            # Model tries to predict the noise
            eps_pred = self.unet(x_t, t_batch)

            # MSE between true noise and predicted noise, per image
            mse = (eps_pred - eps).pow(2).mean(dim=(1, 2, 3))  # (B,)
            errors.append(mse)

        # Average across all timesteps → single score per image
        return torch.stack(errors).mean(dim=0)   # (B,)

    @torch.no_grad()
    def brightness_correct(self, x):
        """
        Simple intensity normalisation — shift mean and std 
        back to training distribution before diffusion correction.
        This handles brightness and contrast shifts explicitly.
        """
        # Per-image normalisation to zero mean unit std
        mean = x.mean(dim=(1,2,3), keepdim=True)
        std  = x.std(dim=(1,2,3),  keepdim=True).clamp(min=1e-6)
        x_norm = (x - mean) / std
        
        # Then scale to match training distribution statistics
        # Use CheXpert training mean/std (your cfg.MEAN, cfg.STD)
        target_mean = torch.tensor(0.0, device=x.device)  
        target_std  = torch.tensor(1.0, device=x.device)
        return x_norm * target_std + target_mean
    # ------------------------------------------------------------------
    # SDEdit-style partial correction  ← used for factor attribution only
    # ------------------------------------------------------------------

    @torch.no_grad()
    def partial_correct(self, x_test, t_star, num_steps=50, eta=0.0):
        """
        SDEdit partial correction at noise level t_star.
        Used ONLY for factor attribution (brightness/noise/scanner/structure).
        NOT used for U_domain computation (use ddim_invert_and_reconstruct instead).

        Lower t_star → corrects only low-level factors (brightness, contrast)
        Higher t_star → corrects deeper factors (scanner artefacts, structure)
        """
        device   = x_test.device
        noise    = torch.randn_like(x_test)
        ab_t     = self.alpha_bar[t_star]
        x_noised = ab_t.sqrt() * x_test + (1 - ab_t).sqrt() * noise

        timesteps = torch.linspace(t_star, 0, num_steps + 1).long().clamp(0, self.T - 1).tolist()
        x = x_noised
        for t, t_prev in zip(timesteps[:-1], timesteps[1:]):
            x = self._ddim_step(x, int(t), int(t_prev), eta)
        return x

    # ------------------------------------------------------------------
    # Reconstruction score (auxiliary OOD signal)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reconstruction_score(self, x_test, t_star, n_samples=5):
        errors = []
        for _ in range(n_samples):
            x_star = self.partial_correct(x_test, t_star, num_steps=50, eta=0.2)
            errors.append(F.mse_loss(x_star, x_test, reduction='none').mean(dim=(1, 2, 3)))
        return torch.stack(errors).mean(dim=0)