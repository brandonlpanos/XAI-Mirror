"""
models.py — model wrappers for:
  • DinoExtractor  : DINOv2-small, self-supervised attention
  • EmotionClassifier : ViT-B/16 fine-tuned for 7 facial emotions
  • FaceDetector   : MediaPipe face crop utility
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoImageProcessor, AutoModelForImageClassification

import config

# ── Device selection ─────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()


# ─────────────────────────────────────────────────────────────────────────────
#  DINOv2 wrapper
# ─────────────────────────────────────────────────────────────────────────────
class DinoExtractor(nn.Module):
    """
    DINOv2-small backbone.

    Provides:
      • get_last_attention(pixel_values) → (num_heads, H_p, W_p)
      • attention_rollout(pixel_values) → (H_p, W_p)
    """

    def __init__(self):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(config.DINO_MODEL_ID)
        self.model = AutoModel.from_pretrained(
            config.DINO_MODEL_ID,
            add_pooling_layer=False,
        ).to(DEVICE)
        self.model.eval()
        self._n = config.DINO_PATCHES_PER_SIDE

    def preprocess(self, pil_image):
        """PIL Image → pixel_values tensor on DEVICE."""
        inputs = self.processor(images=pil_image, return_tensors="pt")
        return inputs["pixel_values"].to(DEVICE)

    @torch.no_grad()
    def get_last_attention(self, pixel_values):
        """
        Last transformer block, CLS→patch attention per head.
        Returns: float32 tensor (num_heads, H_patches, W_patches)
        """
        outputs = self.model(pixel_values=pixel_values, output_attentions=True)
        # outputs.attentions: tuple of (1, heads, seq, seq) per layer
        last = outputs.attentions[-1]            # (1, 6, 257, 257)
        cls_attn = last[0, :, 0, 1:]             # (6, 256)  — skip CLS token
        return cls_attn.reshape(-1, self._n, self._n).cpu()

    @torch.no_grad()
    def attention_rollout(self, pixel_values,
                          discard_ratio=config.ROLLOUT_DISCARD_RATIO):
        """
        Attention rollout (Abnar & Zuidema 2020).
        Propagates attention recursively through all layers, adding residual
        connections and discarding the lowest-weight entries.
        Returns: float32 tensor (H_patches, W_patches)
        """
        outputs = self.model(pixel_values=pixel_values, output_attentions=True)
        all_attns = outputs.attentions          # list of (1, heads, seq, seq)
        seq = all_attns[0].shape[-1]

        result = torch.eye(seq, device=pixel_values.device)

        for attn in all_attns:
            avg = attn[0].mean(dim=0)           # (seq, seq) mean over heads

            # Zero out the bottom `discard_ratio` fraction of weights
            flat = avg.reshape(-1)
            k = int(flat.numel() * discard_ratio)
            if k > 0:
                threshold = flat.kthvalue(k).values.item()
                avg = avg.clamp(min=threshold)

            # Add residual and re-normalise row-wise
            avg = avg + torch.eye(seq, device=pixel_values.device)
            avg = avg / (avg.sum(dim=-1, keepdim=True) + 1e-9)

            result = avg @ result

        mask = result[0, 1:]                    # CLS row, skip CLS itself
        return mask.reshape(self._n, self._n).cpu()


# ─────────────────────────────────────────────────────────────────────────────
#  Emotion ViT classifier
# ─────────────────────────────────────────────────────────────────────────────
class EmotionClassifier(nn.Module):
    """
    ViT-B/16 fine-tuned on FER+ for 7 facial emotion classes.

    Provides:
      • predict(pixel_values) → probabilities (num_classes,) numpy array
      • get_attention(pixel_values) → (num_heads, H_p, W_p)
      • labels : list[str]
    """

    def __init__(self):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(config.EMOTION_MODEL_ID)
        self.model = AutoModelForImageClassification.from_pretrained(
            config.EMOTION_MODEL_ID,
        ).to(DEVICE)
        self.model.eval()

        id2label = self.model.config.id2label
        self.labels = [id2label[i] for i in range(len(id2label))]
        self._n = config.EMOTION_PATCHES_PER_SIDE

    def preprocess(self, pil_image):
        inputs = self.processor(images=pil_image, return_tensors="pt")
        return inputs["pixel_values"].to(DEVICE)

    @torch.no_grad()
    def predict(self, pixel_values):
        """Returns softmax probabilities as (num_classes,) numpy array."""
        out = self.model(pixel_values=pixel_values)
        return torch.softmax(out.logits[0], dim=-1).cpu().numpy()

    @torch.no_grad()
    def get_attention(self, pixel_values):
        """
        Last-layer CLS attention from the supervised emotion ViT.
        Returns: float32 tensor (num_heads, H_patches, W_patches)
        """
        out = self.model(pixel_values=pixel_values, output_attentions=True)
        last = out.attentions[-1]               # (1, 12, 197, 197)
        cls_attn = last[0, :, 0, 1:]            # (12, 196)
        return cls_attn.reshape(-1, self._n, self._n).cpu()


# ─────────────────────────────────────────────────────────────────────────────
#  Face detector / cropper
# ─────────────────────────────────────────────────────────────────────────────
class FaceDetector:
    """
    MediaPipe face detection with padded crop.
    Falls back to centre-crop if no face is found.
    """

    def __init__(self):
        try:
            import mediapipe as mp
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=config.FACE_CONFIDENCE,
            )
            self._use_mp = True
        except Exception:
            self._use_mp = False

    def detect_and_crop(self, frame_rgb: np.ndarray):
        """
        Args:
            frame_rgb: (H, W, 3) uint8 RGB array

        Returns:
            crop_rgb  : (crop_H, crop_W, 3) uint8 RGB, or center crop
            bbox      : (x1, y1, x2, y2) absolute ints, or None
        """
        H, W = frame_rgb.shape[:2]

        if self._use_mp:
            results = self._detector.process(frame_rgb)
            if results.detections:
                det = results.detections[0]
                bb = det.location_data.relative_bounding_box
                pad = config.FACE_PADDING
                x1 = int((bb.xmin - pad * bb.width) * W)
                y1 = int((bb.ymin - pad * bb.height) * H)
                x2 = int((bb.xmin + (1 + pad) * bb.width) * W)
                y2 = int((bb.ymin + (1 + pad) * bb.height) * H)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                return frame_rgb[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

        # Fallback: centre square crop
        side = min(H, W)
        y0, x0 = (H - side) // 2, (W - side) // 2
        return frame_rgb[y0:y0+side, x0:x0+side].copy(), None
