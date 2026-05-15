"""
config.py — centralised constants for XAI Object Lab.
Edit here to swap models or tune pipeline parameters.
"""

# ── Model identifiers ────────────────────────────────────────────────────────
DINO_MODEL_ID   = "facebook/dinov2-small"         # 21 M params, patch-14, 384-dim
OBJECT_MODEL_ID = "google/vit-base-patch16-224"   # ViT-B/16, ImageNet-1k, 1000 classes

# ── Image / patch geometry ───────────────────────────────────────────────────
IMG_SIZE = 224

# DINOv2-small: patch_size=14  →  224/14 = 16 patches per side
DINO_PATCH_SIZE       = 14
DINO_PATCHES_PER_SIDE = IMG_SIZE // DINO_PATCH_SIZE   # 16
DINO_NUM_PATCHES      = DINO_PATCHES_PER_SIDE ** 2    # 256
DINO_NUM_HEADS        = 6

# Object ViT (ViT-B/16): patch_size=16  →  224/16 = 14 patches per side
OBJ_PATCH_SIZE        = 16
OBJ_PATCHES_PER_SIDE  = IMG_SIZE // OBJ_PATCH_SIZE   # 14
OBJ_NUM_PATCHES       = OBJ_PATCHES_PER_SIDE ** 2    # 196
OBJ_NUM_HEADS         = 12
OBJ_TOP_K             = 5   # top-K predictions shown in the probability chart

# ── XAI method parameters ────────────────────────────────────────────────────
IG_STEPS_DEFAULT      = 20    # integrated gradients: steps (5–50 tradeoff)
SG_SAMPLES_DEFAULT    = 12    # smoothgrad: noise samples
SG_NOISE_FRAC         = 0.15  # noise std as fraction of input std

ROLLOUT_DISCARD_RATIO = 0.5   # fraction of lowest attention weights to zero
OVERLAY_ALPHA_DEFAULT = 0.55  # heatmap opacity over image

# ── Display ──────────────────────────────────────────────────────────────────
PANEL_PX    = 448          # each panel is PANEL_PX × PANEL_PX (2× for crisp rendering)
PANEL_GAP   = 10
LABEL_H     = 44           # pixel height of title bar under each panel
GRID_COLS   = 2
COLORMAP    = "gray"       # options: gray, inferno, jet, plasma, magma, hot, binary, binary_r

# ── Streaming ────────────────────────────────────────────────────────────────
PROCESS_EVERY_N_FRAMES = 4   # run heavy XAI pipeline every N camera frames
