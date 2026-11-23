"""PixelForge - a computer vision darkroom (Streamlit entry point).

Run with:  streamlit run app.py

Modes
  Lab           one algorithm at a time on the original image
  Forge Chain   stack algorithms; every stage is a frame on a film strip
  Arena         run competing algorithms on the same image
  From Scratch  NumPy implementations checked against OpenCV
  Codex         the maths behind every operation
"""
from __future__ import annotations

import inspect
import re
import time

import cv2
import numpy as np
import streamlit as st

from core import scratch, utils
from core.registry import (ALL_OP_LABELS, CATEGORIES, LABEL_TO_OP, OP_NUMBER, OPS, PRESETS,
                           apply_op, ops_in, params_summary)
from ui import components as C
from ui import theme

st.set_page_config(page_title="PixelForge", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

MODES = ["Lab", "Forge Chain", "Arena", "From Scratch", "Codex"]
ICONS = dict(CATEGORIES)


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
def init_state() -> None:
    defaults = {"orig": None, "preview": None, "name": "", "last_upload": None, "upload_error": "",
                "nonce": 0, "chain": [], "next_id": 1}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def set_image(img: np.ndarray, name: str) -> None:
    ss = st.session_state
    ss.orig = img                                   # never modified after this point
    ss.preview = utils.make_preview(img)            # all live processing uses this copy
    ss.name = name


def reset_all() -> None:
    st.session_state.nonce += 1                     # fresh widget keys -> default parameters
    st.session_state.chain = []


def timed(op, img, params):
    t0 = time.perf_counter()
    out, note = apply_op(op, img, params)
    return out, note, (time.perf_counter() - t0) * 1000


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# --------------------------------------------------------------------------- #
# Sidebar and landing
# --------------------------------------------------------------------------- #
def sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        st.markdown(theme.logo_html(), unsafe_allow_html=True)
        upload = st.file_uploader("Upload a JPG or PNG", type=["jpg", "jpeg", "png"])
        if upload is None:
            ss.last_upload, ss.upload_error = None, ""
        else:
            sig = f"{upload.name}:{upload.size}"
            if sig != ss.last_upload:
                ss.last_upload = sig
                try:
                    set_image(utils.decode_image(upload.getvalue()), upload.name)
                    ss.upload_error = ""
                except ValueError as exc:
                    ss.upload_error = str(exc)
        if ss.upload_error:
            st.error(ss.upload_error)

        c1, c2 = st.columns(2)
        if c1.button("Load demo"):
            set_image(utils.demo_image(), "demo_scene.png")
        c2.button("Reset all", on_click=reset_all)

        if ss.orig is not None:
            st.markdown(theme.props_html(utils.image_info(ss.orig, ss.name)), unsafe_allow_html=True)
            st.caption(f"Live preview runs at {utils.PREVIEW_WIDTH}px wide. The original is never modified, "
                       "and exports can use full resolution.")


@st.cache_data(show_spinner=False)
def hero_strip() -> str:
    """A real filmstrip of the demo image going through four algorithms."""
    img = utils.demo_image()
    stages = [("Original", img, "720 x 480")]
    chain = [("Grayscale", "Grayscale", {}),
             ("Gaussian Blur", "Gaussian blur", {"ksize": 7}),
             ("Canny", "Canny edges", {"low": 40, "high": 120}),
             ("Morphology", "Closing", {"operation": "Closing", "ksize": 5, "binarize": "None"})]
    cur = img
    for op_name, label, params in chain:
        cur, _ = apply_op(OPS[op_name], cur, params)
        stages.append((label, cur, op_name))
    packed = [(label, utils.to_data_uri(utils.make_preview(im, 300), 88), sub) for label, im, sub in stages]
    return theme.frames_html(packed, "strip")


def landing() -> None:
    st.markdown("<div class='hero-line'>Drop in a photo, stack algorithms, and watch every stage develop.</div>",
                unsafe_allow_html=True)
    st.markdown(hero_strip(), unsafe_allow_html=True)
    st.caption("That strip was made by PixelForge: original, grayscale, Gaussian blur, Canny edges, morphological closing.")
    if st.button("Load the demo scene"):
        set_image(utils.demo_image(), "demo_scene.png")
        st.rerun()
    st.markdown("<div class='note'>Or upload your own JPG or PNG from the sidebar. Everything runs on your machine, "
                "with no internet, database or GPU needed.</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Mode 1: Lab
# --------------------------------------------------------------------------- #
def lab() -> None:
    ss = st.session_state
    cat = st.radio("Category", [c for c, _ in CATEGORIES], horizontal=True, label_visibility="collapsed",
                   key="lab_cat", format_func=lambda c: f"{ICONS[c]} {c}")
    left, right = st.columns([1, 2.1], gap="large")

    with left:
        st.markdown("<div class='label'>Algorithm</div>", unsafe_allow_html=True)
        op = OPS[st.selectbox("Algorithm", [o.name for o in ops_in(cat)], key=f"lab_op_{cat}",
                              label_visibility="collapsed")]
        st.caption(op.blurb)
        params = C.render_params(op, f"lab{ss.nonce}_{op.name}")
        st.markdown("<div class='label'>View</div>", unsafe_allow_html=True)
        view = st.radio("View", ["Reveal", "Side by side", "Delta map"], horizontal=True,
                        label_visibility="collapsed", key="lab_view")
        with st.expander("Under the hood"):
            st.markdown(op.theory)

    try:
        out, note, ms = timed(op, ss.preview, params)
    except Exception as exc:                        # keep the app alive on any bad input
        st.error(f"{op.name} failed: {exc}")
        return

    with right:
        C.compare(ss.preview, out, view)
        oh, ow = out.shape[:2]
        fields = [("op", op.name), ("out", f"{ow}x{oh}"), ("time", f"{ms:.1f} ms")]
        fields.append(("result", note) if note else ("params", params_summary(op, params)))
        st.markdown(theme.edge_html(f"{OP_NUMBER[op.name]:02d}A", fields), unsafe_allow_html=True)
        if note:
            st.caption(f"Parameters: {params_summary(op, params)}")

        if st.toggle("Show histograms", key="lab_hist"):
            C.histogram_panel(ss.preview, out)

        full = st.toggle("Export at full resolution", key="lab_full",
                         help="Pixel-based parameters such as kernel size are applied as-is at full size.")
        if full:
            with st.spinner("Processing the full-resolution image..."):
                data = utils.encode_png(apply_op(op, ss.orig, params)[0])
        else:
            data = utils.encode_png(out)
        st.download_button("Download PNG", data, file_name=f"pixelforge_{slug(op.name)}.png", mime="image/png")


# --------------------------------------------------------------------------- #
# Mode 2: Forge Chain
# --------------------------------------------------------------------------- #
def add_stage(op_name: str) -> None:
    ss = st.session_state
    ss.chain.append({"id": ss.next_id, "op": op_name})
    ss.next_id += 1


def add_selected() -> None:
    add_stage(LABEL_TO_OP[st.session_state.chain_pick])


def move_stage(i: int, delta: int) -> None:
    chain = st.session_state.chain
    j = i + delta
    if 0 <= j < len(chain):
        chain[i], chain[j] = chain[j], chain[i]


def remove_stage(i: int) -> None:
    st.session_state.chain.pop(i)


def load_preset() -> None:
    ss = st.session_state
    ss.chain = []
    for op_name, params in PRESETS[ss.chain_preset]:
        stage_id = ss.next_id
        ss.next_id += 1
        ss.chain.append({"id": stage_id, "op": op_name})
        for key, value in params.items():
            ss[f"ch{stage_id}_{key}"] = value


def clear_chain() -> None:
    st.session_state.chain = []


def chain_tab() -> None:
    ss = st.session_state
    c1, c2, c3, c4, c5 = st.columns([3, 1, 3, 1, 1])
    c1.selectbox("Add a stage", ALL_OP_LABELS, key="chain_pick", label_visibility="collapsed")
    c2.button("Add stage", on_click=add_selected)
    c3.selectbox("Preset", list(PRESETS), key="chain_preset", label_visibility="collapsed")
    c4.button("Load preset", on_click=load_preset)
    c5.button("Clear", on_click=clear_chain)

    film = st.container()
    if not ss.chain:
        with film:
            st.markdown("<div class='note'>The chain is empty. Add a stage, or load a preset such as "
                        "<b>Edge hunter</b> or <b>Document scanner</b>. Each stage receives the output of the "
                        "stage before it.</div>", unsafe_allow_html=True)
        return

    left, right = st.columns([1, 2], gap="large")
    stages = [("Original", ss.preview, "input", 0.0, "")]
    steps = []
    cur = ss.preview

    with left:
        for i, stage in enumerate(ss.chain):
            op = OPS[stage["op"]]
            with st.container(border=True):
                h1, h2, h3, h4 = st.columns([5, 1, 1, 1])
                h1.markdown(f"**{i + 1}. {op.name}**")
                h2.button("▲", key=f"up{stage['id']}", on_click=move_stage, args=(i, -1), disabled=i == 0)
                h3.button("▼", key=f"dn{stage['id']}", on_click=move_stage, args=(i, 1),
                          disabled=i == len(ss.chain) - 1)
                h4.button("✕", key=f"rm{stage['id']}", on_click=remove_stage, args=(i,))
                params = C.render_params(op, f"ch{stage['id']}")
            steps.append((op, params))
            try:
                cur, note, ms = timed(op, cur, params)
            except Exception as exc:
                st.error(f"{op.name} failed: {exc}")
                ms, note = 0.0, "failed"
            stages.append((f"{i + 1}. {op.name}", cur, f"{ms:.1f} ms", ms, note))

    with film:
        C.frames([(label, img, sub) for label, img, sub, *_ in stages], layout="strip", thumb=300)

    with right:
        labels = [s[0] for s in stages]
        selected = st.select_slider("Inspect stage", options=labels, value=labels[-1], key="chain_sel")
        idx = labels.index(selected)
        baseline = st.radio("Compare against", ["Original", "Previous stage"], horizontal=True, key="chain_base")
        view = st.radio("View", ["Reveal", "Side by side", "Delta map"], horizontal=True, key="chain_view")
        base_img = stages[0][1] if (baseline == "Original" or idx == 0) else stages[idx - 1][1]
        C.compare(base_img, stages[idx][1], view)

        total_ms = sum(s[3] for s in stages[1:])
        fh, fw = stages[-1][1].shape[:2]
        st.markdown(theme.edge_html(f"{len(ss.chain):02d}A", [
            ("stages", len(ss.chain)), ("total", f"{total_ms:.1f} ms"), ("out", f"{fw}x{fh}"),
            ("inspecting", selected)]), unsafe_allow_html=True)

        with st.expander("Recipe (for your report)"):
            st.code("\n".join(f"{i + 1}. {op.name}: {params_summary(op, p)}" for i, (op, p) in enumerate(steps)),
                    language="text")

        full = st.toggle("Export at full resolution", key="chain_full")
        if full:
            with st.spinner("Running the chain on the full-resolution image..."):
                result = ss.orig
                for op, p in steps:
                    result = apply_op(op, result, p)[0]
            data = utils.encode_png(result)
        else:
            data = utils.encode_png(stages[-1][1])
        st.download_button("Download final PNG", data, file_name="pixelforge_chain.png", mime="image/png")


# --------------------------------------------------------------------------- #
# Mode 3: Arena
# --------------------------------------------------------------------------- #
def psnr_text(clean: np.ndarray, result: np.ndarray) -> tuple[float, str]:
    value = float(cv2.PSNR(clean, result))
    return value, ("identical" if np.isinf(value) else f"PSNR {value:.1f} dB")


def arena() -> None:
    ss = st.session_state
    src = utils.make_preview(ss.orig, 520)
    kind = st.radio("Shootout", ["Filters", "Edges", "Thresholds", "Morphology"], horizontal=True,
                    label_visibility="collapsed", key="arena_kind")

    if kind == "Filters":
        c1, c2, c3, c4 = st.columns(4)
        inject = c1.toggle("Inject noise", value=True, key="ar_inject")
        noise_kind = c2.selectbox("Noise type", ["Salt & pepper", "Gaussian"], key="ar_nk")
        amount = c3.slider("Noise amount", 1.0, 30.0, 6.0, key="ar_amt")
        k = c4.slider("Kernel size", 3, 15, 5, 2, key="ar_k")
        noisy = apply_op(OPS["Add Noise"], src, {"kind": noise_kind, "amount": amount})[0] if inject else src
        runs = [("Gaussian", "Gaussian Blur", {"ksize": k, "sigma": 0.0}),
                ("Median", "Median Blur", {"ksize": k}),
                ("Bilateral", "Bilateral Filter", {"d": k, "sigma_color": 75, "sigma_space": 75}),
                ("Average", "Average Blur", {"ksize": k})]
        items, scores = [("Clean source" if not inject else "Noisy input", noisy,
                          "no noise" if not inject else f"{noise_kind} {amount:g}")], []
        for label, op_name, params in runs:
            out, _, ms = timed(OPS[op_name], noisy, params)
            score, text = psnr_text(src, out)
            scores.append(score)
            items.append((label, out, f"{ms:.0f} ms  {text}"))
        C.frames(items, layout="sheet", thumb=420, best=1 + int(np.argmax(scores)))
        st.caption("PSNR compares each result with the clean source: higher means closer. Median usually wins on "
                   "salt-and-pepper noise, while bilateral keeps edges sharper than Gaussian or average.")

    elif kind == "Edges":
        c1, c2, c3, c4 = st.columns(4)
        lo = c1.slider("Canny lower", 0, 300, 70, key="ar_lo")
        hi = c2.slider("Canny upper", 0, 400, 150, key="ar_hi")
        sk = c3.slider("Sobel kernel", 1, 7, 3, 2, key="ar_sk")
        lk = c4.slider("Laplacian kernel", 1, 7, 3, 2, key="ar_lk")
        runs = [("Canny", "Canny", {"low": lo, "high": hi}), ("Sobel", "Sobel", {"ksize": sk}),
                ("Laplacian", "Laplacian", {"ksize": lk})]
        items = [("Original", src, "input")]
        for label, op_name, params in runs:
            out, _, ms = timed(OPS[op_name], src, params)
            items.append((label, out, f"{ms:.0f} ms  edge pixels {(out > 64).mean() * 100:.1f}%"))
        C.frames(items, layout="sheet", thumb=420)
        st.caption("Canny gives thin, connected edges. Sobel shows gradient strength, thick and soft. "
                   "Laplacian is sharp but picks up noise.")

    elif kind == "Thresholds":
        c1, c2, c3 = st.columns(3)
        t = c1.slider("Global threshold", 0, 255, 127, key="ar_t")
        block = c2.slider("Adaptive block size", 3, 51, 21, 2, key="ar_blk")
        cc = c3.slider("Adaptive constant C", -10, 20, 5, key="ar_c")
        runs = [("Binary", "Threshold", {"thresh": t, "mode": "Binary"}),
                ("Otsu", "Threshold", {"mode": "Otsu (auto)"}),
                ("Adaptive mean", "Adaptive Threshold", {"method": "Mean", "block": block, "c": cc}),
                ("Adaptive Gaussian", "Adaptive Threshold", {"method": "Gaussian", "block": block, "c": cc})]
        items = [("Original", src, "input")]
        for label, op_name, params in runs:
            out, note, ms = timed(OPS[op_name], src, params)
            items.append((label, out, f"{ms:.0f} ms  white {(out == 255).mean() * 100:.0f}%" + (f"  {note}" if note else "")))
        C.frames(items, layout="sheet", thumb=420)
        st.caption("Global thresholds struggle with uneven lighting. Adaptive methods decide per neighbourhood.")

    else:
        c1, c2 = st.columns(2)
        k = c1.slider("Structuring element size", 3, 21, 5, 2, key="ar_mk")
        shape = c2.selectbox("Shape", ["Rectangle", "Ellipse", "Cross"], key="ar_ms")
        base = apply_op(OPS["Threshold"], src, {"mode": "Otsu (auto)"})[0]
        items = [("Binary base (Otsu)", base, "input")]
        for name in ("Erosion", "Dilation", "Opening", "Closing", "Gradient"):
            out, _, ms = timed(OPS["Morphology"], base, {"operation": name, "ksize": k, "shape": shape,
                                                         "binarize": "None"})
            items.append((name, out, f"{ms:.0f} ms  white {(out == 255).mean() * 100:.0f}%"))
        C.frames(items, layout="sheet", thumb=420)
        st.caption("Opening removes small white specks, closing fills small holes, and the gradient outlines shapes.")


# --------------------------------------------------------------------------- #
# Mode 4: From Scratch
# --------------------------------------------------------------------------- #
def scratch_tab() -> None:
    ss = st.session_state
    gray = utils.to_gray(utils.make_preview(ss.orig, 480))
    which = st.radio("Experiment", ["Convolution", "Histogram equalization", "Otsu threshold"], horizontal=True,
                     label_visibility="collapsed", key="scratch_kind")
    st.markdown("<div class='note'>These functions use only NumPy. Each one is run next to OpenCV on your image, "
                "so you can show an examiner that you know what <code>cv2</code> does underneath.</div>",
                unsafe_allow_html=True)

    if which == "Convolution":
        c1, c2, c3 = st.columns(3)
        name = c1.selectbox("Kernel", scratch.KERNEL_NAMES, key="sc_kernel")
        k = c2.slider("Size (Gaussian and box)", 3, 15, 5, 2, key="sc_k")
        sigma = c3.slider("Sigma (Gaussian)", 0.3, 5.0, 1.2, 0.1, key="sc_sigma")
        kernel = scratch.named_kernel(name, k, sigma)
        t0 = time.perf_counter()
        mine = scratch.conv2d(gray, kernel)
        t_mine = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        ref = cv2.filter2D(gray, cv2.CV_64F, kernel, borderType=cv2.BORDER_REFLECT_101)
        t_cv = (time.perf_counter() - t0) * 1000
        diff = np.abs(mine - ref)
        items = [("From scratch", scratch.to_display(mine, kernel), f"{t_mine:.1f} ms"),
                 ("OpenCV filter2D", scratch.to_display(ref, kernel), f"{t_cv:.1f} ms"),
                 ("Difference x50", np.clip(diff * 50, 0, 255).astype(np.uint8), f"max {diff.max():.1e}")]
        C.frames(items, layout="sheet", thumb=420)
        verdict = "identical to OpenCV" if diff.max() < 1e-6 else "differs from OpenCV"
        st.markdown(theme.edge_html("01A", [("kernel", f"{kernel.shape[0]}x{kernel.shape[1]} {name}"),
                                            ("max difference", f"{diff.max():.2e}"), ("verdict", verdict),
                                            ("speed", f"OpenCV is {t_mine / max(t_cv, 1e-6):.0f}x faster")]),
                    unsafe_allow_html=True)
        if kernel.shape[0] <= 7:
            st.dataframe(np.round(kernel, 4))
        with st.expander("Source: conv2d"):
            st.code(inspect.getsource(scratch.conv2d), language="python")

    elif which == "Histogram equalization":
        lut = scratch.equalize_lut(gray)
        mine = lut[gray]
        ref = cv2.equalizeHist(gray)
        diff = np.abs(mine.astype(int) - ref.astype(int))
        C.frames([("Input", gray, "gray"), ("From scratch", mine, "cdf mapping"),
                  ("OpenCV equalizeHist", ref, f"max level difference {diff.max()}")], layout="sheet", thumb=420)
        c1, c2 = st.columns(2)
        with c1:
            C.lut_figure(lut)
        with c2:
            C.histogram_panel(gray, mine)
        with st.expander("Source: equalize_lut"):
            st.code(inspect.getsource(scratch.equalize_lut), language="python")

    else:
        t_mine, sigma_b = scratch.otsu_threshold(gray)
        t_cv, ref = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mine = scratch.threshold(gray, t_mine)
        mismatch = (mine != ref).mean() * 100
        C.frames([("Input", gray, "gray"), ("From scratch", mine, f"T = {t_mine}"),
                  ("OpenCV Otsu", ref, f"T = {t_cv:.0f}")], layout="sheet", thumb=420)
        C.otsu_figure(gray, sigma_b, t_mine, int(t_cv))
        st.caption(f"Pixels that differ between the two results: {mismatch:.3f}%")
        with st.expander("Source: otsu_threshold"):
            st.code(inspect.getsource(scratch.otsu_threshold), language="python")


# --------------------------------------------------------------------------- #
# Mode 5: Codex
# --------------------------------------------------------------------------- #
def codex() -> None:
    st.markdown("<div class='note'>Everything in the app, with its parameters and the maths behind it. "
                "Use it as viva revision.</div>", unsafe_allow_html=True)
    for cat, icon in CATEGORIES:
        st.markdown(f"### {icon} {cat}")
        for op in ops_in(cat):
            with st.expander(op.name):
                st.write(op.blurb)
                if op.params:
                    rows = ["| parameter | range | default |", "|---|---|---|"]
                    for p in op.params:
                        rng = ", ".join(p.options) if p.kind == "select" else ("on / off" if p.kind == "bool" else f"{p.min} to {p.max}")
                        rows.append(f"| {p.label} | {rng} | {p.default} |")
                    st.markdown("\n".join(rows))
                st.markdown(op.theory)


# --------------------------------------------------------------------------- #
def main() -> None:
    init_state()
    theme.inject()
    sidebar()
    st.markdown(theme.header_html(), unsafe_allow_html=True)
    if st.session_state.preview is None:
        landing()
        return
    mode = st.radio("Mode", MODES, horizontal=True, label_visibility="collapsed", key="mode")
    {"Lab": lab, "Forge Chain": chain_tab, "Arena": arena, "From Scratch": scratch_tab, "Codex": codex}[mode]()


main()
