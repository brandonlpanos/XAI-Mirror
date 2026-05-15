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
    annotate_feed,
    assemble_grid,
    make_panel,
    make_prob_chart,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Page setup and global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI Face Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Dark background */
.stApp { background-color: #080c14; color: #c8d8f0; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1220;
    border-right: 1px solid #1e2d4a;
}

/* Headers */
h1, h2, h3 {
    font-family: 'Space Mono', monospace !important;
    letter-spacing: -0.02em;
}
h1 { color: #5bd4ff; font-size: 1.6rem !important; }
h2 { color: #8ab8e0; font-size: 1.1rem !important; }
h3 { color: #6890b8; font-size: 0.9rem !important; }

/* Info / divider */
hr { border-color: #1a2540; }

/* Selectbox, slider labels */
label { font-family: 'Space Mono', monospace !important;
        font-size: 0.75rem !important; color: #6080a0 !important; }

/* Streamlit image captions */
.stImage { border-radius: 4px; overflow: hidden; }

/* Metric boxes */
[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    color: #30e8a0 !important;
}

/* Custom banner */
.xai-banner {
    background: linear-gradient(135deg, #0a1525 0%, #0f1e38 100%);
    border: 1px solid #1e3560;
    border-radius: 8px;
    padding: 12px 20px;
    margin-bottom: 16px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: #4a8ab8;
    letter-spacing: 0.06em;
}
.xai-banner span { color: #5bd4ff; }
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
    st.markdown("## 🔬 XAI Face Lab")
    st.markdown('<hr>', unsafe_allow_html=True)

    demo_mode = st.radio("Camera mode", ["📸 Snapshot", "🎥 Stream"],
                         horizontal=True)

    st.markdown("### Target class")
    auto_select = st.checkbox("Auto (top prediction)", value=True)
    manual_class = st.selectbox(
        "Manual override",
        pipeline.emotion.labels,
        disabled=auto_select,
    )
    manual_idx = pipeline.emotion.labels.index(manual_class)

    st.markdown("### Active methods")
    want_dino   = st.checkbox("DINO self-attn heads",   value=True)
    want_roll   = st.checkbox("Attention rollout (DINO)", value=True)
    want_eattn  = st.checkbox("Emotion ViT attention",  value=True)
    want_gcam   = st.checkbox("Grad-CAM",               value=True)
    want_ig     = st.checkbox("Integrated Gradients",   value=True)
    want_sg     = st.checkbox("SmoothGrad",             value=False)

    st.markdown("### Method parameters")
    ig_steps   = st.slider("IG steps", 5, 50, config.IG_STEPS_DEFAULT, 5)
    sg_samples = st.slider("SmoothGrad samples", 5, 30,
                           config.SG_SAMPLES_DEFAULT, 5)
    alpha      = st.slider("Overlay α", 0.25, 0.80,
                           config.OVERLAY_ALPHA_DEFAULT, 0.05)
    cmap_name  = st.selectbox("Colormap",
                              ["inferno", "plasma", "magma", "hot", "jet"],
                              index=0)

    # Patch the global default colormap so visualization module picks it up
    config.COLORMAP = cmap_name
    config.OVERLAY_ALPHA_DEFAULT = alpha

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(
        "<small style='color:#304060;font-family:Space Mono,monospace'>"
        "FHNW · I4DS · XAI Lab<br>"
        f"DINOv2-S  ·  ViT-B/16 · {pipeline.emotion.labels}</small>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='xai-banner'>
  <span>MULTI-METHOD XAI DISPLAY</span> &nbsp;·&nbsp;
  DINOv2 self-supervised attention &nbsp;·&nbsp; Grad-CAM &nbsp;·&nbsp;
  Integrated Gradients &nbsp;·&nbsp; SmoothGrad &nbsp;·&nbsp;
  Attention Rollout
</div>
""", unsafe_allow_html=True)

col_feed, col_probs = st.columns([3, 2])
with col_feed:
    st.markdown("#### Camera feed")
    feed_ph = st.empty()
with col_probs:
    st.markdown("#### Emotion distribution")
    prob_ph  = st.empty()
    metric_ph = st.empty()

st.markdown("---")
st.markdown("#### Explanation panels")
grid_ph = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
#  Shared kwargs for pipeline.run()
# ─────────────────────────────────────────────────────────────────────────────
def _pipeline_kwargs():
    return dict(
        auto_select    = auto_select,
        target_class   = manual_idx,
        want_dino_heads = want_dino,
        want_rollout   = want_roll,
        want_emotion_attn = want_eattn,
        want_gradcam   = want_gcam,
        want_ig        = want_ig,
        want_sg        = want_sg,
        ig_steps       = ig_steps,
        sg_samples     = sg_samples,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Render results → Streamlit placeholders
# ─────────────────────────────────────────────────────────────────────────────
def render(results: dict, raw_frame: np.ndarray | None = None):
    if results is None:
        feed_ph.info("👤 No face detected — position yourself in front of the camera.")
        return

    crop    = results["crop"]
    probs   = results["probs"]
    labels  = results["labels"]
    active  = results["active_class"]

    # ── Annotated feed ────────────────────────────────────────────────────
    display_frame = raw_frame if raw_frame is not None else crop
    if results.get("bbox") and raw_frame is not None:
        display_frame = annotate_feed(
            raw_frame, results["bbox"],
            labels[active], float(probs[active]))
    feed_ph.image(display_frame, channels="RGB", use_container_width=True)

    # ── Probability chart ─────────────────────────────────────────────────
    chart = make_prob_chart(labels, probs, active_idx=active,
                            width=300, height=36 * len(labels) + 20)
    prob_ph.image(chart, channels="RGB", use_container_width=True)

    with metric_ph.container():
        col1, col2 = st.columns(2)
        col1.metric("Prediction", labels[active].upper())
        col2.metric("Confidence", f"{probs[active]:.1%}")

    # ── Build explanation panels ──────────────────────────────────────────
    panels = []

    # 1. Raw input (no overlay)
    panels.append(make_panel(crop, None, "Input face", alpha=alpha))

    # 2. DINO per-head attention (first 4 heads)
    if "dino_heads" in results:
        for i, head in enumerate(results["dino_heads"][:4]):
            panels.append(make_panel(crop, head, f"DINO head {i+1}", alpha=alpha))

    # 3. DINO attention rollout
    if "rollout" in results:
        panels.append(make_panel(crop, results["rollout"],
                                 "DINO rollout", alpha=alpha))

    # 4. Supervised emotion ViT attention (average over heads)
    if "emotion_attn" in results:
        avg_attn = results["emotion_attn"].mean(axis=0)   # (14, 14)
        panels.append(make_panel(crop, avg_attn,
                                 f"Emotion ViT attn", alpha=alpha))

    # 5. Grad-CAM
    if "gradcam" in results:
        panels.append(make_panel(crop, results["gradcam"],
                                 f"Grad-CAM  [{labels[active]}]", alpha=alpha))

    # 6. Integrated Gradients
    if "ig" in results:
        panels.append(make_panel(crop, results["ig"],
                                 "Integrated Grads", alpha=alpha))

    # 7. SmoothGrad
    if "sg" in results:
        panels.append(make_panel(crop, results["sg"],
                                 "SmoothGrad", alpha=alpha))

    grid = assemble_grid(panels, ncols=config.GRID_COLS)
    grid_ph.image(grid, channels="RGB", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Snapshot mode
# ─────────────────────────────────────────────────────────────────────────────
if "📸" in demo_mode:
    st.caption("Click **Take photo** below, then wait a moment for all explanations to compute.")
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
            st.caption(f"Pipeline time: {elapsed:.2f} s  ·  "
                       f"Methods: Grad-CAM + IG({ig_steps} steps) + rollout")

# ─────────────────────────────────────────────────────────────────────────────
#  Stream mode (streamlit-webrtc)
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.caption(
        "Real-time streaming via WebRTC. "
        "If the video doesn't start, your network may block WebRTC — switch to Snapshot mode."
    )

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
            if results is not None:
                try:
                    result_q.put_nowait(results)
                except queue.Full:
                    pass  # Drop if consumer is slow

        # Annotate live frame with last known prediction
        nonlocal _last_results
        if _last_results and _last_results.get("bbox"):
            lbl    = _last_results["labels"][_last_results["active_class"]]
            conf   = float(_last_results["probs"][_last_results["active_class"]])
            x1, y1, x2, y2 = _last_results["bbox"]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (30, 220, 140), 2)
            cv2.putText(img_bgr, f"{lbl}  {conf:.0%}",
                        (x1 + 4, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 220, 140), 2)

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

    ctx = webrtc_streamer(
        key="xai-lab-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        video_frame_callback=video_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.state.playing:
        st.success("📡 Stream active — XAI panels update every "
                   f"{config.PROCESS_EVERY_N_FRAMES} frames.")
        while ctx.state.playing:
            try:
                results = result_q.get(timeout=0.5)
                _last_results = results
                render(results)
            except queue.Empty:
                pass
            time.sleep(0.05)
