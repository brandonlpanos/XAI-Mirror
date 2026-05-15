"""
models.py — model wrappers for:
  • DinoExtractor    : DINOv2-small, self-supervised attention
  • ObjectClassifier : ViT-B/16 fine-tuned on ImageNet-1k (1000 classes)
"""

import numpy as np
import torch
import torch.nn as nn
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
            attn_implementation="eager",
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
        last = outputs.attentions[-1]            # (1, 6, 257, 257)
        cls_attn = last[0, :, 0, 1:]             # (6, 256) — skip CLS token
        return cls_attn.reshape(-1, self._n, self._n).cpu()

    @torch.no_grad()
    def attention_rollout(self, pixel_values,
                          discard_ratio=config.ROLLOUT_DISCARD_RATIO):
        """
        Attention rollout (Abnar & Zuidema 2020).
        Returns: float32 tensor (H_patches, W_patches)
        """
        outputs = self.model(pixel_values=pixel_values, output_attentions=True)
        all_attns = outputs.attentions          # list of (1, heads, seq, seq)
        seq = all_attns[0].shape[-1]

        result = torch.eye(seq, device=pixel_values.device)

        for attn in all_attns:
            avg = attn[0].mean(dim=0)           # (seq, seq)

            flat = avg.reshape(-1)
            k = int(flat.numel() * discard_ratio)
            if k > 0:
                threshold = flat.kthvalue(k).values.item()
                avg = avg.clamp(min=threshold)

            avg = avg + torch.eye(seq, device=pixel_values.device)
            avg = avg / (avg.sum(dim=-1, keepdim=True) + 1e-9)

            result = avg @ result

        mask = result[0, 1:]                    # CLS row, skip CLS itself
        return mask.reshape(self._n, self._n).cpu()


# ─────────────────────────────────────────────────────────────────────────────
#  Object classifier (ImageNet ViT-B/16)
# ─────────────────────────────────────────────────────────────────────────────
class ObjectClassifier(nn.Module):
    """
    ViT-B/16 fine-tuned on ImageNet-1k (1000 classes).

    Provides:
      • predict(pixel_values) → (top_labels, top_probs, top1_global_idx)
      • get_attention(pixel_values) → (num_heads, H_p, W_p)
    """

    def __init__(self):
        super().__init__()
        self.processor = AutoImageProcessor.from_pretrained(config.OBJECT_MODEL_ID)
        self.model = AutoModelForImageClassification.from_pretrained(
            config.OBJECT_MODEL_ID,
            attn_implementation="eager",
        ).to(DEVICE)
        self.model.eval()
        self._n = config.OBJ_PATCHES_PER_SIDE

    def preprocess(self, pil_image):
        inputs = self.processor(images=pil_image, return_tensors="pt")
        return inputs["pixel_values"].to(DEVICE)

    @torch.no_grad()
    def predict(self, pixel_values):
        """
        Returns top-K predictions.
          top_labels        : list[str] of length OBJ_TOP_K
          top_probs         : float32 array of length OBJ_TOP_K
          top1_global_idx   : int, index into the full 1000-class logit vector
                              (used by gradient-based XAI methods)
        """
        out = self.model(pixel_values=pixel_values)
        probs_all = torch.softmax(out.logits[0], dim=-1)
        top = torch.topk(probs_all, config.OBJ_TOP_K)
        indices = top.indices.cpu().tolist()
        values  = top.values.cpu().numpy().astype(np.float32)
        labels  = [self.model.config.id2label[i] for i in indices]
        return labels, values, indices[0]

    @torch.no_grad()
    def get_attention(self, pixel_values):
        """
        Last-layer CLS attention from the object ViT.
        Returns: float32 tensor (num_heads, H_patches, W_patches)
        """
        out = self.model(pixel_values=pixel_values, output_attentions=True)
        last = out.attentions[-1]               # (1, 12, 197, 197)
        cls_attn = last[0, :, 0, 1:]            # (12, 196)
        return cls_attn.reshape(-1, self._n, self._n).cpu()
