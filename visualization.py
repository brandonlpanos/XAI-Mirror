"""
visualization.py — panel assembly and rendering utilities.

All functions work with numpy uint8 RGB arrays.
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm

import config


# ─────────────────────────────────────────────────────────────────────────────
#  Font helper — uses Pillow 10+ load_default(size) with graceful fallback
# ─────────────────────────────────────────────────────────────────────────────
def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


# ─────────────────────────────────────────────────────────────────────────────
#  Core helpers
# ─────────────────────────────────────────────────────────────────────────────
def _normalize(x: np.ndarray) -> np.ndarray:
    """Normalize float map to [0, 1]."""
    mn, mx = x.min(), x.max()
    if mx - mn > 1e-9:
        return (x - mn) / (mx - mn)
    return np.zeros_like(x)


def _resize_map(x: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Bilinearly resize a 2-D float map to (H, W)."""
    return cv2.resize(x.astype(np.float32), (hw[1], hw[0]),
                      interpolation=cv2.INTER_CUBIC)


def apply_colormap(x: np.ndarray,
                   cmap_name: str | None = None) -> np.ndarray:
    """
    Apply a matplotlib colormap to a 2-D float map.
    Returns (H, W, 3) uint8 RGB.
    """
    if cmap_name is None:
        cmap_name = config.COLORMAP
    normed = _normalize(x)
    cmap = cm.get_cmap(cmap_name)
    rgb = (cmap(normed)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def overlay_heatmap(base_rgb: np.ndarray,
                    heatmap_2d: np.ndarray,
                    alpha: float | None = None,
                    cmap_name: str | None = None) -> np.ndarray:
    """
    Alpha-blend a heatmap over a base RGB image.
    heatmap_2d is resized to match base_rgb's dimensions.
    Returns (H, W, 3) uint8 RGB.
    """
    if alpha is None:
        alpha = config.OVERLAY_ALPHA_DEFAULT
    if cmap_name is None:
        cmap_name = config.COLORMAP
    h, w = base_rgb.shape[:2]
    resized = _resize_map(heatmap_2d, (h, w))
    colored = apply_colormap(resized, cmap_name)

    blended = ((1 - alpha) * base_rgb.astype(np.float32) +
               alpha * colored.astype(np.float32))
    return np.clip(blended, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  Individual panel builder
# ─────────────────────────────────────────────────────────────────────────────
_TITLE_BG  = (255, 255, 255)
_TITLE_FG  = (20, 20, 20)
_BORDER_C  = (220, 220, 220)
_PANEL_PX  = config.PANEL_PX
_LABEL_H   = config.LABEL_H


def make_panel(face_crop_rgb: np.ndarray,
               heatmap_2d: np.ndarray | None,
               title: str,
               size: int = _PANEL_PX,
               alpha: float | None = None) -> np.ndarray:
    """
    Build a single labelled panel: face crop (+ optional heatmap) + title bar.
    Returns (size + LABEL_H, size, 3) uint8 RGB.
    """
    if alpha is None:
        alpha = config.OVERLAY_ALPHA_DEFAULT
    img = cv2.resize(face_crop_rgb, (size, size),
                     interpolation=cv2.INTER_LANCZOS4)

    if heatmap_2d is not None:
        img = overlay_heatmap(img, heatmap_2d, alpha=alpha)

    # Title bar
    bar = np.full((_LABEL_H, size, 3), _TITLE_BG, dtype=np.uint8)
    panel = np.vstack([img, bar])

    pil = Image.fromarray(panel)
    draw = ImageDraw.Draw(pil)
    draw.text((10, size + 10), title, fill=_TITLE_FG, font=_font(20))

    return np.array(pil)


# ─────────────────────────────────────────────────────────────────────────────
#  Probability bar chart
# ─────────────────────────────────────────────────────────────────────────────
_BAR_BG     = (255, 255, 255)
_BAR_NORM   = (210, 210, 210)
_BAR_ACTIVE = (20, 20, 20)
_TEXT_DIM   = (170, 170, 170)
_TEXT_BRIGHT = (20, 20, 20)


def make_prob_chart(labels: list[str],
                    probs: np.ndarray,
                    active_idx: int | None = None,
                    width: int = 360,
                    height: int = 320) -> np.ndarray:
    """
    Render a horizontal probability bar chart as a numpy RGB image.
    Internally renders at 3× the requested dimensions for crisp text.
    """
    S = 3  # render scale — downscaled by browser, never upscaled
    n = len(labels)
    W, H     = width * S, height * S
    label_w  = 100 * S
    pct_w    = 52 * S
    bar_zone = W - label_w - pct_w

    img = np.full((H, W, 3), _BAR_BG, dtype=np.uint8)

    usable_h = H - 20 * S
    row_h = max(10 * S, usable_h // n)
    bar_h = max(8 * S, row_h - 10 * S)

    font = _font(14 * S)

    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)

    for i, (lbl, p) in enumerate(zip(labels, probs)):
        y_top = 10 * S + i * row_h
        bar_w = int(bar_zone * float(p))
        color = _BAR_ACTIVE if i == active_idx else _BAR_NORM
        txt_col = _TEXT_BRIGHT if i == active_idx else _TEXT_DIM

        # Background track
        draw.rectangle([label_w, y_top, label_w + bar_zone, y_top + bar_h],
                       fill=(230, 230, 230))
        # Value bar
        if bar_w > 0:
            draw.rectangle([label_w, y_top, label_w + bar_w, y_top + bar_h],
                           fill=color)

        # Label
        draw.text((2 * S, y_top + S), lbl[:14], fill=txt_col, font=font)

        # Probability as percentage
        draw.text((label_w + bar_zone + 4 * S, y_top + S),
                  f"{p:.1%}", fill=txt_col, font=font)

    return np.array(pil)


# ─────────────────────────────────────────────────────────────────────────────
#  Grid assembler
# ─────────────────────────────────────────────────────────────────────────────
_GRID_BG = (240, 240, 240)


def assemble_grid(panels: list[np.ndarray],
                  ncols: int = config.GRID_COLS,
                  gap: int = config.PANEL_GAP) -> np.ndarray:
    """
    Arrange a flat list of same-shape panels into a grid.
    Pads the last row with dark filler if needed.
    """
    if not panels:
        return np.full((config.PANEL_PX + _LABEL_H, config.PANEL_PX, 3),
                       _GRID_BG, dtype=np.uint8)

    ph, pw = panels[0].shape[:2]
    bg_tile = np.full((ph, pw, 3), _GRID_BG, dtype=np.uint8)
    v_gap = np.full((gap, pw * ncols + gap * (ncols - 1), 3),
                    _GRID_BG, dtype=np.uint8)
    h_gap = np.full((ph, gap, 3), _GRID_BG, dtype=np.uint8)

    rows = []
    for i in range(0, len(panels), ncols):
        row_tiles = panels[i:i + ncols]
        while len(row_tiles) < ncols:
            row_tiles.append(bg_tile.copy())
        row = row_tiles[0]
        for tile in row_tiles[1:]:
            row = np.hstack([row, h_gap, tile])
        rows.append(row)

    grid = rows[0]
    for row in rows[1:]:
        grid = np.vstack([grid, v_gap, row])

    return grid


# ─────────────────────────────────────────────────────────────────────────────
#  Live-feed annotation
# ─────────────────────────────────────────────────────────────────────────────
def annotate_feed(frame_rgb: np.ndarray,
                  bbox: tuple | None,
                  label: str,
                  confidence: float) -> np.ndarray:
    """
    Draw bounding box and label on the live camera frame.
    Returns a copy of the frame (RGB).
    """
    out = frame_rgb.copy()
    if bbox is None:
        return out
    x1, y1, x2, y2 = bbox
    col = (30, 220, 140)
    cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
    txt = f"{label}  {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 8, y1), col, -1)
    cv2.putText(out, txt, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 20, 30), 2)
    return out
