"""
config.py — centralised constants for XAI Face Lab.
Edit here to swap models or tune pipeline parameters.
"""

# ── Model identifiers ────────────────────────────────────────────────────────
DINO_MODEL_ID    = "facebook/dinov2-small"       # 21 M params, patch-14, 384-dim
EMOTION_MODEL_ID = "dima806/facial_emotions_image_detection"  # ViT-B/16, 7 classes

# ── Image / patch geometry ───────────────────────────────────────────────────
IMG_SIZE = 224

# DINOv2-small: patch_size=14  →  224/14 = 16 patches per side
DINO_PATCH_SIZE           = 14
DINO_PATCHES_PER_SIDE     = IMG_SIZE // DINO_PATCH_SIZE   # 16
DINO_NUM_PATCHES          = DINO_PATCHES_PER_SIDE ** 2    # 256
DINO_NUM_HEADS            = 6

# Emotion ViT (ViT-B/16): patch_size=16  →  224/16 = 14 patches per side
EMOTION_PATCH_SIZE        = 16
EMOTION_PATCHES_PER_SIDE  = IMG_SIZE // EMOTION_PATCH_SIZE   # 14
EMOTION_NUM_PATCHES       = EMOTION_PATCHES_PER_SIDE ** 2   # 196
EMOTION_NUM_HEADS         = 12

# ── XAI method parameters ────────────────────────────────────────────────────
IG_STEPS_DEFAULT          = 20    # integrated gradients: steps (5–50 tradeoff)
SG_SAMPLES_DEFAULT        = 12    # smoothgrad: noise samples
SG_NOISE_FRAC             = 0.15  # noise std as fraction of input std

ROLLOUT_DISCARD_RATIO     = 0.9   # fraction of lowest attention weights to zero
OVERLAY_ALPHA_DEFAULT     = 0.55  # heatmap opacity over face crop

# ── Face detection ───────────────────────────────────────────────────────────
FACE_CONFIDENCE           = 0.5
FACE_PADDING              = 0.30  # fractional padding around bbox on each side

# ── Display ──────────────────────────────────────────────────────────────────
PANEL_PX      = 224          # each panel is PANEL_PX × PANEL_PX
PANEL_GAP     = 6
LABEL_H       = 28           # pixel height of title bar under each panel
GRID_COLS     = 4
COLORMAP      = "inferno"    # options: inferno, jet, plasma, magma, hot

# ── Streaming ────────────────────────────────────────────────────────────────
PROCESS_EVERY_N_FRAMES = 4   # run heavy XAI pipeline every N camera frames
