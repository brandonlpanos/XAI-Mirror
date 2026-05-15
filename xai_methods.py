"""
xai_methods.py — XAI attribution methods for ViT-based models.

All functions accept:
  • model       : EmotionClassifier instance
  • pixel_values: (1, 3, 224, 224) tensor on the appropriate device
  • target_class: int

All functions return:
  • (H_patches, W_patches) or (224, 224) float32 numpy array (un-normalised)
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

import config


# ─────────────────────────────────────────────────────────────────────────────
#  Grad-CAM (ViT adaptation)
# ─────────────────────────────────────────────────────────────────────────────
def vit_grad_cam(emotion_model, pixel_values, target_class: int) -> np.ndarray:
    """
    Class-discriminative saliency for ViT.

    In ViT the classifier reads only the CLS token (position 0), so
    d_score/d_last_hidden[patch_k] == 0 for all k > 0 — the last-layer patch
    hidden states are not in the score's computation graph.  We instead
    differentiate w.r.t. the input pixels (always non-zero), then pool
    pixel-level gradients to patch resolution.

    Returns: (H_patches, W_patches) float32 numpy array
    """
    pv = pixel_values.detach().requires_grad_(True)

    with torch.enable_grad():
        out = emotion_model.model(pixel_values=pv)
        score = out.logits[0, target_class]
        grad = torch.autograd.grad(score, pv)[0]   # (1, 3, 224, 224)

    # L2 norm over colour channels → spatial sensitivity map
    saliency = grad[0].norm(dim=0)                 # (224, 224)

    # Average-pool to patch grid so the output matches the attention map scale
    n = config.OBJ_PATCHES_PER_SIDE
    p = config.OBJ_PATCH_SIZE
    cam = saliency.reshape(n, p, n, p).mean(dim=(1, 3))   # (n, n)

    return cam.cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Integrated Gradients
# ─────────────────────────────────────────────────────────────────────────────
def integrated_gradients(emotion_model, pixel_values, target_class: int,
                          n_steps: int = config.IG_STEPS_DEFAULT) -> np.ndarray:
    """
    Integrated Gradients (Sundararajan et al. 2017).
    Baseline: zero tensor (black image).

    Returns: (224, 224) float32 numpy array (channel-magnitude summed)
    """
    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        raise RuntimeError("captum is required for Integrated Gradients. "
                           "Install with: pip install captum")

    def _forward(x):
        return emotion_model.model(pixel_values=x).logits

    ig = IntegratedGradients(_forward)
    baseline = torch.zeros_like(pixel_values)

    attrs = ig.attribute(
        pixel_values,
        baselines=baseline,
        target=target_class,
        n_steps=n_steps,
        internal_batch_size=1,
    )
    # attrs: (1, 3, 224, 224)
    # Reduce channels by L2 magnitude; smooth to remove patch-grid blockiness
    raw = attrs[0].norm(dim=0).detach().cpu().numpy().astype(np.float32)
    return gaussian_filter(raw, sigma=8.0)


# ─────────────────────────────────────────────────────────────────────────────
#  SmoothGrad
# ─────────────────────────────────────────────────────────────────────────────
def smoothgrad(emotion_model, pixel_values, target_class: int,
               n_samples: int = config.SG_SAMPLES_DEFAULT,
               noise_frac: float = config.SG_NOISE_FRAC) -> np.ndarray:
    """
    SmoothGrad (Smilkov et al. 2017) via captum NoiseTunnel + Saliency.

    Returns: (224, 224) float32 numpy array
    """
    try:
        from captum.attr import Saliency, NoiseTunnel
    except ImportError:
        raise RuntimeError("captum is required for SmoothGrad. "
                           "Install with: pip install captum")

    def _forward(x):
        return emotion_model.model(pixel_values=x).logits

    saliency = Saliency(_forward)
    nt = NoiseTunnel(saliency)

    stdev = float(pixel_values.std()) * noise_frac

    attrs = nt.attribute(
        pixel_values,
        target=target_class,
        nt_type="smoothgrad",
        nt_samples=n_samples,
        stdevs=stdev,
    )
    raw = attrs[0].abs().mean(dim=0).detach().cpu().numpy().astype(np.float32)
    return gaussian_filter(raw, sigma=8.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Gradient × Input  (cheap but informative sanity check)
# ─────────────────────────────────────────────────────────────────────────────
def gradient_x_input(emotion_model, pixel_values, target_class: int) -> np.ndarray:
    """
    Element-wise product of gradient and input (fast, no external deps).

    Returns: (224, 224) float32 numpy array
    """
    pv = pixel_values.detach().requires_grad_(True)
    emotion_model.model.zero_grad()

    out = emotion_model.model(pixel_values=pv)
    score = out.logits[0, target_class]
    score.backward()

    grad = pv.grad[0]                  # (3, 224, 224)
    saliency = (grad * pv[0]).abs().mean(dim=0)   # (224, 224)
    raw = saliency.detach().cpu().numpy().astype(np.float32)
    return gaussian_filter(raw, sigma=8.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Guided Backpropagation
# ─────────────────────────────────────────────────────────────────────────────
def guided_backprop(model_wrapper, pixel_values, target_class: int) -> np.ndarray:
    """
    Guided Backpropagation (Springenberg et al. 2015).
    Zeros out negative gradients at each ReLU during the backward pass,
    producing sharper attributions than vanilla gradients.
    ViT uses GELU (no ReLU), so this is equivalent to plain gradient saliency
    for the standard ViT-B/16 — but the output is still valid and meaningful.

    Returns: (224, 224) float32 numpy array
    """
    try:
        from captum.attr import GuidedBackprop
    except ImportError:
        raise RuntimeError("captum is required. pip install captum")

    def _forward(x):
        return model_wrapper.model(pixel_values=x).logits

    gb = GuidedBackprop(_forward)
    attrs = gb.attribute(pixel_values, target=target_class)
    raw = attrs[0].abs().mean(dim=0).detach().cpu().numpy().astype(np.float32)
    return gaussian_filter(raw, sigma=6.0)
