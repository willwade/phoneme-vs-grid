# streamlit_app.py
# Phoneme vs Whole-word AAC: interactive explorer (with visual search/cognitive load)
# -------------------------------------------------------------------------------
# This Streamlit app compares predicted time-per-word for:
#   1) Phoneme input (fixed inventory, e.g., 47 keys)
#   2) Whole-word AAC with paged grids (serial realistic)
#   3) Whole-word AAC with paged grids (idealized upper bound direct indexing)
#   4) Whole-word AAC with frequency-weighted access (core vs fringe)
#
# The “indexed” model is hypothetical — no current AAC gives true direct indexing at large vocab.
# We keep it as a **theoretical lower bound** for context.
#
# The frequency-weighted model is closer to real symbol-based systems, where:
#   • Top ~100 words are on the home page (1 tap)
#   • Next ~900 words take ~2 taps
#   • Remaining words take ~3 taps
# These are weighted by usage frequency (Zipfian-like), making high-frequency words faster.
#
# Vocabulary development & usage anchors:
#   • Age ~1: ~50 recognized words
#   • Age ~3: ~1,000 recognized words
#   • Age ~5: ~10,000 recognized words
#   • First 25 words ≈ 33% of everyday writing
#   • First 100 words ≈ 50% of writing
#   • First 1,000 words ≈ 89% of writing
#
# Run locally:
#   pip install streamlit matplotlib numpy
#   streamlit run streamlit_app.py
# -------------------------------------------------------------------------------

import math
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------
# Core timing functions
# -------------------------

def hick_time(N: int, T0: float, c: float) -> float:
    return T0 + c * math.log2(N + 1)

def visual_search_time(M: int, VS0: float, M_ref: int, alpha: float) -> float:
    M = max(1, M)
    return VS0 * (M / max(1, M_ref)) ** alpha

def time_phoneme(avg_phonemes: float, N_ph: int, T0: float, c: float,
                 VS0: float, M_ref: int, alpha: float) -> float:
    per_selection = hick_time(N_ph, T0, c) + visual_search_time(N_ph, VS0, M_ref, alpha)
    return avg_phonemes * per_selection

def time_word_indexed(V: int, B: int, T0: float, c: float,
                      VS0: float, M_ref: int, alpha: float,
                      page_index_capacity: int) -> float:
    P = max(math.ceil(V / B), 1)
    M_page = min(P, max(1, page_index_capacity))
    page_select = hick_time(P, T0, c) + visual_search_time(M_page, VS0, M_ref, alpha)
    word_select = hick_time(min(B, V), T0, c) + visual_search_time(min(B, V), VS0, M_ref, alpha)
    return page_select + word_select

def time_word_serial(V: int, B: int, T_flip: float, T0: float, c: float,
                     VS0: float, M_ref: int, alpha: float) -> float:
    P = max(math.ceil(V / B), 1)
    flips = max(P - 1, 0) / 2.0
    on_page_choice = hick_time(min(B, V), T0, c) + visual_search_time(min(B, V), VS0, M_ref, alpha)
    return flips * T_flip + on_page_choice

# Frequency-weighted word model: 25=33%, 100=50%, 1000=89%
def time_word_frequency_weighted(V: int, T0: float, c: float,
                                 VS0: float, M_ref: int, alpha: float) -> float:
    # weights: top 100 (0.5 prob, 1 tap), next 900 (0.39 prob, 2 taps), rest (0.11 prob, 3 taps)
    if V <= 25:
        return hick_time(V, T0, c) + visual_search_time(V, VS0, M_ref, alpha)
    # approximate distribution
    p_core100 = 0.50
    p_next900 = 0.39
    p_rest = 0.11
    t_core = hick_time(100, T0, c) + visual_search_time(100, VS0, M_ref, alpha)
    t_mid = 2 * (hick_time(60, T0, c) + visual_search_time(60, VS0, M_ref, alpha))
    t_tail = 3 * (hick_time(60, T0, c) + visual_search_time(60, VS0, M_ref, alpha))
    return p_core100 * t_core + p_next900 * t_mid + p_rest * t_tail

def find_breakeven(Vs, phoneme_times, word_times) -> Optional[int]:
    for V, p, w in zip(Vs, phoneme_times, word_times):
        if p <= w:
            return int(V)
    return None

# -------------------------
# UI
# -------------------------

st.set_page_config(page_title="Phoneme vs Word AAC Explorer", layout="wide")
st.title("Phoneme vs Whole-word AAC: Time-per-Word Explorer")

with st.sidebar:
    st.header("Assumptions")
    V_min, V_max, V_step = 25, 10000, 25
    V = st.slider("Vocabulary size V", min_value=V_min, max_value=V_max, value=1500, step=V_step)

    B = st.number_input("Buttons per page (B)", min_value=6, max_value=120, value=60, step=6)
    N_ph = st.number_input("Phoneme keys (N_ph)", min_value=20, max_value=80, value=47, step=1)
    avg_phonemes = st.number_input("Average phonemes per word", min_value=1.5, max_value=5.0, value=2.5, step=0.1)

    T0 = st.number_input("T0 (base time per choice, s)", 0.0, 1.0, 0.30, 0.05)
    c = st.number_input("c (s per bit)", 0.05, 0.5, 0.15, 0.01)
    T_flip = st.number_input("T_flip (serial page flip, s)", 0.0, 1.0, 0.35, 0.05)

    VS0 = st.number_input("VS0 (base visual search, s)", 0.0, 2.0, 0.50, 0.05)
    M_ref = st.number_input("Reference density M_ref", 10, 100, 47, 1)
    alpha = st.number_input("Density exponent α", 0.1, 1.0, 0.5, 0.05)
    page_index_capacity = st.number_input("Page-index buttons visible", 4, 60, 12, 1)

    modes = st.multiselect(
        "Compare models",
        ["Phoneme (flat)", "Word (serial realistic)", "Word (indexed upper bound)", "Word (frequency-weighted)"],
        default=["Phoneme (flat)", "Word (serial realistic)", "Word (indexed upper bound)", "Word (frequency-weighted)"]
    )
    show_breakeven = st.checkbox("Show break-even markers", True)
    show_age_markers = st.checkbox("Show age vocabulary markers", True)
    show_frequency_markers = st.checkbox("Show frequency usage markers", True)

# -------------------------
# Compute curves
# -------------------------

Vs = np.arange(V_min, V_max + V_step, V_step)
phoneme_curve = np.array([time_phoneme(avg_phonemes, N_ph, T0, c, VS0, M_ref, alpha) for _ in Vs])
word_serial_curve = np.array([time_word_serial(v, B, T_flip, T0, c, VS0, M_ref, alpha) for v in Vs])
word_indexed_curve = np.array([time_word_indexed(v, B, T0, c, VS0, M_ref, alpha, page_index_capacity) for v in Vs])
word_freq_curve = np.array([time_word_frequency_weighted(v, T0, c, VS0, M_ref, alpha) for v in Vs])

Tp = time_phoneme(avg_phonemes, N_ph, T0, c, VS0, M_ref, alpha)
Tws = time_word_serial(V, B, T_flip, T0, c, VS0, M_ref, alpha)
Twi = time_word_indexed(V, B, T0, c, VS0, M_ref, alpha, page_index_capacity)
Twf = time_word_frequency_weighted(V, T0, c, VS0, M_ref, alpha)

# -------------------------
# Plot
# -------------------------

fig = plt.figure(figsize=(9,5))
if "Phoneme (flat)" in modes:
    plt.plot(Vs, phoneme_curve, label="Phoneme (flat)")
if "Word (serial realistic)" in modes:
    plt.plot(Vs, word_serial_curve, label="Word (serial realistic)")
if "Word (indexed upper bound)" in modes:
    plt.plot(Vs, word_indexed_curve, label="Word (indexed upper bound)")
if "Word (frequency-weighted)" in modes:
    plt.plot(Vs, word_freq_curve, label="Word (frequency-weighted)")

plt.xlabel("Vocabulary size V (unique words)")
plt.ylabel("Predicted time per word (seconds)")
plt.title("Phoneme vs Whole-word AAC: Models with Visual Search & Frequency Weighting")

if show_age_markers:
    for x, lbl in [(50, "Age ~1 (~50)"), (1000, "Age ~3 (~1k)"), (10000, "Age ~5 (~10k)")]:
        if V_min <= x <= V_max:
            plt.axvline(x, linestyle=":")
            plt.text(x, plt.ylim()[1]*0.9, lbl, rotation=90, va="top")

if show_frequency_markers:
    for x, lbl in [(25, "Top 25 ≈33%"), (100, "Top 100 ≈50%"), (1000, "Top 1k ≈89%")]:
        if V_min <= x <= V_max:
            plt.axvline(x, linestyle="--")
            plt.text(x, plt.ylim()[1]*0.8, lbl, rotation=90, va="top")

plt.legend()
plt.tight_layout()
st.pyplot(fig, clear_figure=True)

# -------------------------
# Break-even markers
# -------------------------

if show_breakeven:
    be_serial = find_breakeven(Vs, phoneme_curve, word_serial_curve)
    be_indexed = find_breakeven(Vs, phoneme_curve, word_indexed_curve)
    be_freq = find_breakeven(Vs, phoneme_curve, word_freq_curve)
    st.markdown("### Break-even (Phoneme ≤ Word)")
    st.write({
        "Serial realistic": be_serial,
        "Indexed upper bound": be_indexed,
        "Frequency-weighted": be_freq,
    })
