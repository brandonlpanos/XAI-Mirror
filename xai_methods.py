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

import config


# ─────────────────────────────────────────────────────────────────────────────
#  Grad-CAM (ViT adaptation)
# ─────────────────────────────────────────────────────────────────────────────
def vit_grad_cam(emotion_model, pixel_values, target_class: int) -> np.ndarray:
    """
    Class-discriminative saliency via gradient of the target logit w.r.t. the
    last encoder block's hidden states (Selvaraju et al. adapted for ViT).

    Returns: (H_patches, W_patches) float32 numpy array
    """
    _activations = {}

    def _fwd_hook(module, inp, out):
        # ViTLayer output is a tuple: (hidden_states, [attn_weights])
        h = out[0] if isinstance(out, tuple) else out
        _activations["feat"] = h
        h.retain_grad()

    last_block = emotion_model.model.vit.encoder.layer[-1]
    handle = last_block.register_forward_hook(_fwd_hook)

    # Enable grad on input so the graph is built
    pv = pixel_values.detach().requires_grad_(True)
    emotion_model.model.zero_grad()

    try:
        out = emotion_model.model(pixel_values=pv)
        score = out.logits[0, target_class]
        score.backward()
    finally:
        handle.remove()

    feat = _activations["feat"]          # (1, seq_len, hidden)
    grad = feat.grad                     # (1, seq_len, hidden)

    if grad is None:
        # Gradient didn't reach here; return blank map
        n = config.EMOTION_PATCHES_PER_SIDE
        return np.zeros((n, n), dtype=np.float32)

    # Skip CLS token (index 0)
    feat_patches = feat[0, 1:].detach()  # (num_patches, hidden)
    grad_patches = grad[0, 1:].detach()  # (num_patches, hidden)

    # Global average pool over hidden dim → importance weight per patch
    weights = grad_patches.mean(dim=-1)  # (num_patches,)
    cam = torch.relu((weights.unsqueeze(-1) * feat_patches).sum(dim=-1))

    n = config.EMOTION_PATCHES_PER_SIDE
    return cam.reshape(n, n).cpu().numpy().astype(np.float32)


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
    # Reduce channels by L2 magnitude for a cleaner map
    return attrs[0].norm(dim=0).detach().cpu().numpy().astype(np.float32)


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
    return attrs[0].abs().mean(dim=0).detach().cpu().numpy().astype(np.float32)


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
    return saliency.detach().cpu().numpy().astype(np.float32)
