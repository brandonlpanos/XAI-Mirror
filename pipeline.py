"""
pipeline.py — orchestrates the full per-frame XAI computation.

Takes a raw RGB frame and a set of feature flags, returns a dict of
arrays ready for visualization.
"""

from __future__ import annotations
import numpy as np
from PIL import Image

import config
from models import DinoExtractor, EmotionClassifier, FaceDetector
import xai_methods as xai


class XAIPipeline:
    """
    Holds the loaded models and exposes a single `run()` call.
    Designed to be instantiated once (via @st.cache_resource) and
    called on every frame.
    """

    def __init__(self):
        self.face_det = FaceDetector()
        self.dino     = DinoExtractor()
        self.emotion  = EmotionClassifier()

    # ── public API ───────────────────────────────────────────────────────────
    def run(self,
            frame_rgb: np.ndarray,
            *,
            target_class: int | None = None,
            auto_select: bool = True,
            want_dino_heads: bool = True,
            want_rollout: bool = True,
            want_emotion_attn: bool = True,
            want_gradcam: bool = True,
            want_ig: bool = True,
            want_sg: bool = False,
            ig_steps: int = config.IG_STEPS_DEFAULT,
            sg_samples: int = config.SG_SAMPLES_DEFAULT,
            ) -> dict | None:
        """
        Args:
            frame_rgb : (H, W, 3) uint8 RGB numpy array

        Returns:
            None if no face is detected.
            Otherwise a dict with keys:
              crop         : (224, 224, 3) uint8 RGB — resized face crop
              probs        : (num_classes,) float32 probability array
              labels       : list[str]
              active_class : int — class used for gradient-based methods
              bbox         : (x1, y1, x2, y2) or None
              dino_heads   : (6, 16, 16) float32  — if want_dino_heads
              rollout      : (16, 16) float32      — if want_rollout
              emotion_attn : (12, 14, 14) float32  — if want_emotion_attn
              gradcam      : (14, 14) float32       — if want_gradcam
              ig           : (224, 224) float32     — if want_ig
              sg           : (224, 224) float32     — if want_sg
        """

        # ── 1. Face detection ─────────────────────────────────────────────
        crop_rgb, bbox = self.face_det.detect_and_crop(frame_rgb)
        if crop_rgb is None or crop_rgb.size == 0:
            return None

        pil_crop = Image.fromarray(crop_rgb).resize(
            (config.IMG_SIZE, config.IMG_SIZE), Image.LANCZOS)
        crop_224 = np.array(pil_crop)

        # ── 2. Emotion prediction ─────────────────────────────────────────
        em_pv   = self.emotion.preprocess(pil_crop)
        probs   = self.emotion.predict(em_pv)                   # (C,)
        top_cls = int(np.argmax(probs))
        active  = top_cls if auto_select else (target_class or top_cls)

        result: dict = {
            "crop":         crop_224,
            "probs":        probs,
            "labels":       self.emotion.labels,
            "active_class": active,
            "bbox":         bbox,
        }

        # ── 3. DINO preprocessing ─────────────────────────────────────────
        dino_pv = self.dino.preprocess(pil_crop)

        # ── 4. DINO per-head attention ────────────────────────────────────
        if want_dino_heads:
            result["dino_heads"] = self.dino.get_last_attention(dino_pv).numpy()
            # shape (6, 16, 16)

        # ── 5. Attention rollout ──────────────────────────────────────────
        if want_rollout:
            result["rollout"] = self.dino.attention_rollout(dino_pv).numpy()
            # shape (16, 16)

        # ── 6. Supervised emotion ViT attention ───────────────────────────
        if want_emotion_attn:
            result["emotion_attn"] = self.emotion.get_attention(em_pv).numpy()
            # shape (12, 14, 14)

        # ── 7. Grad-CAM ───────────────────────────────────────────────────
        if want_gradcam:
            result["gradcam"] = xai.vit_grad_cam(self.emotion, em_pv, active)
            # shape (14, 14)

        # ── 8. Integrated Gradients ───────────────────────────────────────
        if want_ig:
            result["ig"] = xai.integrated_gradients(
                self.emotion, em_pv, active, n_steps=ig_steps)
            # shape (224, 224)

        # ── 9. SmoothGrad ─────────────────────────────────────────────────
        if want_sg:
            result["sg"] = xai.smoothgrad(
                self.emotion, em_pv, active, n_samples=sg_samples)
            # shape (224, 224)

        return result
