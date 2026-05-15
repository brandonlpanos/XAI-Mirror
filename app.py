"""
app.py — XAI Face Lab Streamlit dashboard.

Run with:
    streamlit run app.py

Two modes:
  • Snapshot  : uses st.camera_input — works everywhere, no network config needed
  • Stream    : uses streamlit-webrtc — true real-time, requires WebRTC-compatible network
"""

import queue
import time
import threading

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, RTCConfiguration, webrtc_streamer

import config
from pipeline import XAIPipeline
from visualization import (
    assemble_grid,
    make_panel,
    make_prob_chart,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Page setup and global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI Object Lab",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
}

.stApp { background-color: #ffffff; color: #111111; }

section[data-testid="stSidebar"] {
    background-color: #f7f7f7;
    border-right: 1px solid #e2e2e2;
}

h1 { color: #000 !important; font-size: 1.05rem !important; font-weight: 700 !important;
     letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0 !important; }
h2, h4 { color: #000 !important; font-weight: 500 !important; font-size: 0.85rem !important; }
h3 { color: #999 !important; font-size: 0.68rem !important; font-weight: 400 !important;
     text-transform: uppercase; letter-spacing: 0.12em; margin: 1.2em 0 0.4em 0 !important; }

label { font-size: 0.74rem !important; color: #444 !important; }
.stCaption p { color: #aaa !important; font-size: 0.68rem !important; }

[data-testid="stMetricValue"] { color: #000 !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #999 !important; font-size: 0.68rem !important;
                                text-transform: uppercase; letter-spacing: 0.08em; }

hr { border: none !important; border-top: 1px solid #e2e2e2 !important; margin: 1em 0 !important; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Model loading (cached across reruns)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models (first run only)…")
def load_pipeline():
    return XAIPipeline()

pipeline = load_pipeline()

# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## XAI Object Lab")
    st.markdown('<hr>', unsafe_allow_html=True)

    demo_mode = st.radio("Mode", ["Snapshot", "Stream"], horizontal=True)

    st.markdown("### Active methods")
    want_dino   = st.checkbox("DINO attn avg (h1-3)",    value=True)
    st.markdown("### Extra methods")
    want_oattn  = st.checkbox("Object ViT attention",    value=False)
    want_gxi    = st.checkbox("Gradient x Input",        value=False)
    want_gcam   = st.checkbox("Grad-CAM",                value=False)
    want_ig     = st.checkbox("Integrated Gradients",    value=False)
    want_sg     = st.checkbox("SmoothGrad",              value=False)
    want_roll   = st.checkbox("DINO rollout",            value=False)
    want_gbp    = st.checkbox("Guided Backprop",         value=False)

    st.markdown("### Method parameters")
    ig_steps   = st.slider("IG steps", 5, 50, config.IG_STEPS_DEFAULT, 5)
    sg_samples = st.slider("SmoothGrad samples", 5, 30,
                           config.SG_SAMPLES_DEFAULT, 5)
    alpha      = st.slider("Overlay α", 0.25, 0.80,
                           config.OVERLAY_ALPHA_DEFAULT, 0.05)
    cmap_name  = st.selectbox("Colormap",
                              ["inferno", "plasma", "magma", "hot", "jet",
                               "binary", "binary_r"],
                              index=0)

    # Patch the global default colormap so visualization module picks it up
    config.COLORMAP = cmap_name
    config.OVERLAY_ALPHA_DEFAULT = alpha

    st.markdown('<hr>', unsafe_allow_html=True)
    st.caption(f"DINOv2-S · ViT-B/16 · ImageNet-1k top-{config.OBJ_TOP_K}")

# ─────────────────────────────────────────────────────────────────────────────
#  Layout placeholders
# ─────────────────────────────────────────────────────────────────────────────
grid_ph = st.empty()

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("#### Prediction")
col_chart, col_metrics = st.columns([2, 1])
with col_chart:
    prob_ph = st.empty()
with col_metrics:
    metric_ph = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
#  Shared kwargs for pipeline.run()
# ─────────────────────────────────────────────────────────────────────────────
def _pipeline_kwargs():
    return dict(
        want_dino_heads = want_dino,
        want_rollout    = want_roll,
        want_obj_attn   = want_oattn,
        want_gradcam    = want_gcam,
        want_ig         = want_ig,
        want_sg         = want_sg,
        want_gxi        = want_gxi,
        want_gbp        = want_gbp,
        ig_steps        = ig_steps,
        sg_samples      = sg_samples,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Render results → Streamlit placeholders
# ─────────────────────────────────────────────────────────────────────────────
def render(results: dict, raw_frame: np.ndarray | None = None):
    crop   = results["crop"]
    probs  = results["probs"]
    labels = results["labels"]

    # ── Probability chart (right column) ─────────────────────────────────
    chart = make_prob_chart(labels, probs, active_idx=0,
                            width=360, height=36 * len(labels) + 20)
    prob_ph.image(chart, channels="RGB", use_container_width=True)

    with metric_ph.container():
        col1, col2 = st.columns(2)
        col1.metric("Top object", labels[0].upper())
        col2.metric("Confidence", f"{probs[0]:.1%}")

    # ── Build panels: camera first, then XAI maps (all same size) ────────
    panels = []

    # Camera panel — use the raw frame if available (stream), else the crop
    cam_img = raw_frame if raw_frame is not None else crop
    panels.append(make_panel(cam_img, None, "Camera"))

    if "dino_avg" in results:
        panels.append(make_panel(crop, results["dino_avg"], "DINO attn avg"))
    if "obj_attn" in results:
        panels.append(make_panel(crop, results["obj_attn"].mean(axis=0),
                                 "Object ViT attn"))
    if "gxi" in results:
        panels.append(make_panel(crop, results["gxi"], "Grad x Input"))
    if "gradcam" in results:
        panels.append(make_panel(crop, results["gradcam"],
                                 f"Grad-CAM [{labels[0]}]"))

    # Extra methods
    if "ig" in results:
        panels.append(make_panel(crop, results["ig"], "Integrated Grads"))
    if "sg" in results:
        panels.append(make_panel(crop, results["sg"], "SmoothGrad"))
    if "rollout" in results:
        panels.append(make_panel(crop, results["rollout"], "DINO rollout"))
    if "gbp" in results:
        panels.append(make_panel(crop, results["gbp"], "Guided Backprop"))

    grid = assemble_grid(panels, ncols=config.GRID_COLS)
    grid_ph.image(grid, channels="RGB", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Snapshot mode
# ─────────────────────────────────────────────────────────────────────────────
if demo_mode == "Snapshot":
    st.caption("Take a photo to run the explanation pipeline.")
    img_file = st.camera_input("Take photo", label_visibility="collapsed")

    if img_file:
        from PIL import Image as PILImage
        pil = PILImage.open(img_file).convert("RGB")
        frame = np.array(pil)

        t0 = time.perf_counter()
        with st.spinner("Running XAI pipeline…"):
            results = pipeline.run(frame, **_pipeline_kwargs())
        elapsed = time.perf_counter() - t0

        render(results, raw_frame=frame)
        if results:
            st.caption(f"{elapsed:.2f} s")

# ─────────────────────────────────────────────────────────────────────────────
#  Stream mode (streamlit-webrtc)
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.caption("WebRTC stream — if it doesn't start, switch to Snapshot mode.")

    result_q: queue.Queue[dict] = queue.Queue(maxsize=2)
    lock = threading.Lock()
    _frame_counter = [0]
    _last_results: dict | None = None
    _last_annotated: np.ndarray | None = None

    def video_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = img_bgr[:, :, ::-1].copy()

        with lock:
            _frame_counter[0] += 1
            run_xai = (_frame_counter[0] % config.PROCESS_EVERY_N_FRAMES == 0)

        if run_xai:
            results = pipeline.run(img_rgb, **_pipeline_kwargs())
            try:
                result_q.put_nowait(results)
            except queue.Full:
                pass  # Drop if consumer is slow


        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

    ctx = webrtc_streamer(
        key="xai-lab-stream",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        video_frame_callback=video_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.state.playing:
        st.caption(f"Stream active — updates every {config.PROCESS_EVERY_N_FRAMES} frames.")
        while ctx.state.playing:
            try:
                results = result_q.get(timeout=0.5)
                _last_results = results
                render(results)
            except queue.Empty:
                pass
            time.sleep(0.05)
