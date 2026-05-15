"""
pipeline.py — orchestrates the full per-frame XAI computation.

Takes a raw RGB frame and a set of feature flags, returns a dict of
arrays ready for visualization.
"""

from __future__ import annotations
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter as _gf

import config
from models import DinoExtractor, ObjectClassifier
import xai_methods as xai


class XAIPipeline:
    """
    Holds the loaded models and exposes a single `run()` call.
    Designed to be instantiated once (via @st.cache_resource) and
    called on every frame.
    """

    def __init__(self):
        self.dino    = DinoExtractor()
        self.obj_clf = ObjectClassifier()

    # ── public API ───────────────────────────────────────────────────────────
    def run(self,
            frame_rgb: np.ndarray,
            *,
            want_dino_heads: bool = True,
            want_rollout: bool = True,
            want_obj_attn: bool = True,
            want_gradcam: bool = True,
            want_ig: bool = True,
            want_sg: bool = False,
            want_gxi: bool = False,
            want_gbp: bool = False,
            ig_steps: int = config.IG_STEPS_DEFAULT,
            sg_samples: int = config.SG_SAMPLES_DEFAULT,
            ) -> dict:
        """
        Args:
            frame_rgb : (H, W, 3) uint8 RGB numpy array

        Returns:
            dict with keys:
              crop         : (224, 224, 3) uint8 RGB — resized input frame
              labels       : list[str] — top-K ImageNet class names
              probs        : (OBJ_TOP_K,) float32 — top-K probabilities
              active_class : int — top-1 index in the full 1000-class space
                             (used by gradient-based XAI methods)
              dino_heads   : (6, 16, 16) float32  — if want_dino_heads
              rollout      : (16, 16) float32      — if want_rollout
              obj_attn     : (12, 14, 14) float32  — if want_obj_attn
              gradcam      : (14, 14) float32       — if want_gradcam
              ig           : (224, 224) float32     — if want_ig
              sg           : (224, 224) float32     — if want_sg
        """

        # ── 1. Center-crop to square, then resize ────────────────────────
        H, W = frame_rgb.shape[:2]
        side = min(H, W)
        y0 = (H - side) // 2
        x0 = (W - side) // 2
        pil_frame = Image.fromarray(frame_rgb[y0:y0+side, x0:x0+side]).resize(
            (config.IMG_SIZE, config.IMG_SIZE), Image.LANCZOS)
        crop_224 = np.array(pil_frame)

        # ── 2. Object prediction ──────────────────────────────────────────
        obj_pv = self.obj_clf.preprocess(pil_frame)
        top_labels, top_probs, top_cls = self.obj_clf.predict(obj_pv)

        result: dict = {
            "crop":         crop_224,
            "labels":       top_labels,
            "probs":        top_probs,
            "active_class": top_cls,
        }

        # ── 3. DINO preprocessing ─────────────────────────────────────────
        dino_pv = self.dino.preprocess(pil_frame)

        # ── 4. DINO attention — average of heads 0-2 ─────────────────────
        # Head 3 (index 3) in DINOv2-small fires on background/unstructured
        # regions; heads 0-2 are more semantically consistent.
        if want_dino_heads:
            raw = self.dino.get_last_attention(dino_pv).numpy()  # (6, 16, 16)
            smoothed = np.stack([_gf(h, sigma=1.0) for h in raw[:3]])
            result["dino_avg"] = smoothed.mean(axis=0)           # (16, 16)

        # ── 5. Attention rollout ──────────────────────────────────────────
        if want_rollout:
            result["rollout"] = _gf(
                self.dino.attention_rollout(dino_pv).numpy(), sigma=1.0)

        # ── 6. Object ViT attention ───────────────────────────────────────
        if want_obj_attn:
            raw = self.obj_clf.get_attention(obj_pv).numpy()  # (12, 14, 14)
            result["obj_attn"] = np.stack([_gf(h, sigma=1.0) for h in raw])

        # ── 7. Grad-CAM ───────────────────────────────────────────────────
        if want_gradcam:
            result["gradcam"] = xai.vit_grad_cam(self.obj_clf, obj_pv, top_cls)

        # ── 8. Integrated Gradients ───────────────────────────────────────
        if want_ig:
            result["ig"] = xai.integrated_gradients(
                self.obj_clf, obj_pv, top_cls, n_steps=ig_steps)

        # ── 9. SmoothGrad ─────────────────────────────────────────────────
        if want_sg:
            result["sg"] = xai.smoothgrad(
                self.obj_clf, obj_pv, top_cls, n_samples=sg_samples)

        # ── 10. Gradient × Input ──────────────────────────────────────────
        if want_gxi:
            result["gxi"] = xai.gradient_x_input(self.obj_clf, obj_pv, top_cls)

        # ── 11. Guided Backpropagation ────────────────────────────────────
        if want_gbp:
            result["gbp"] = xai.guided_backprop(self.obj_clf, obj_pv, top_cls)

        return result
