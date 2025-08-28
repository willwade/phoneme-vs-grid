# streamlit_app.py
# Phoneme vs Whole-word AAC: interactive explorer (with visual search/cognitive load)
# -------------------------------------------------------------------------------
# This Streamlit app compares predicted time-per-word for:
#   1) Phoneme input (fixed inventory, e.g., 47 keys)
#   2) Whole-word AAC with paged grids (either idealized direct indexing or serial paging)
#
# IMPORTANT REALITY CHECK
# -----------------------
# No commercial AAC system (to our knowledge) provides *true direct indexing* to any page at
# arbitrary vocab sizes without scrolling/paging. The "indexed" model below is therefore an
# *idealized upper bound* on whole-word performance: two choices only (page, then word).
# Real-world whole-word navigation typically behaves closer to the serial model due to paging,
# tabs, category depth, and visual search overhead.
#
# The model combines:
#   • Hick's law per discrete choice:      T_choice(N) = T0 + c * log2(N + 1)
#   • Visual search / cognitive load term: T_search(M) = VS0 * (M / M_ref)**alpha
#       where M = number of buttons visible on the screen for the current selection.
#   • Serial page flips: expected flips ~ (P-1)/2 at T_flip each (pessimistic UI)
#
# Vocabulary development & usage anchors (rough heuristics for markers):
#   • Age ~1: ~50 recognized words
#   • Age ~3: ~1,000 recognized words
#   • Age ~5: ~10,000 recognized words
#   • First 25 words ≈ 33% of everyday writing
#   • First 100 words ≈ 50% of adult/student writing
#   • First 1,000 words ≈ 89% of everyday writing
# These anchors help explain where whole-word systems can feel efficient (very small, very frequent sets),
# and where scaling pressure pushes towards phoneme-based input.
#
# You can tune: V, B, N_ph, avg phonemes, Hick parameters, visual search base & density exponent,
# page indexing capacity, and serial flip cost. You can also toggle age/frequency markers on the plot.
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
    """Hick's law component for a single discrete choice among N options."""
    return T0 + c * math.log2(N + 1)


def visual_search_time(M: int, VS0: float, M_ref: int, alpha: float) -> float:
    """Visual search / cognitive load term for M buttons visible on screen.
    VS0 is base time per search at reference density M_ref. As density increases
    (M grows), search time scales by (M/M_ref)**alpha.
    """
    M = max(1, M)
    return VS0 * (M / max(1, M_ref)) ** alpha


def time_phoneme(avg_phonemes: float, N_ph: int, T0: float, c: float,
                 VS0: float, M_ref: int, alpha: float) -> float:
    # Each phoneme selection entails a discrete choice among N_ph + a visual search across N_ph
    per_selection = hick_time(N_ph, T0, c) + visual_search_time(N_ph, VS0, M_ref, alpha)
    return avg_phonemes * per_selection


def time_word_indexed(V: int, B: int, T0: float, c: float,
                      VS0: float, M_ref: int, alpha: float,
                      page_index_capacity: int) -> float:
    # Choose a page among P, then a word among B. Each step has a Hick term + visual search term.
    P = max(math.ceil(V / B), 1)

    # Page selection UI shows up to 'page_index_capacity' page buttons at once (e.g., 12).
    # Visual search operates over min(P, page_index_capacity) visible items.
    M_page = min(P, max(1, page_index_capacity))
    page_select = hick_time(P, T0, c) + visual_search_time(M_page, VS0, M_ref, alpha)

    # Word selection on its page: choose among B buttons
    word_select = hick_time(min(B, V), T0, c) + visual_search_time(min(B, V), VS0, M_ref, alpha)

    return page_select + word_select


def time_word_serial(V: int, B: int, T_flip: float, T0: float, c: float,
                     VS0: float, M_ref: int, alpha: float) -> float:
    # Serial flips to reach the page, then choose among B on that page
    P = max(math.ceil(V / B), 1)
    flips = max(P - 1, 0) / 2.0
    on_page_choice = hick_time(min(B, V), T0, c) + visual_search_time(min(B, V), VS0, M_ref, alpha)
    return flips * T_flip + on_page_choice


def find_breakeven(Vs: np.ndarray, phoneme_times: np.ndarray, word_times: np.ndarray) -> Optional[int]:
    for V, p, w in zip(Vs, phoneme_times, word_times):
        if p <= w:
            return int(V)
    return None

# -------------------------
# UI
# -------------------------

st.set_page_config(page_title="Phoneme vs Word AAC Explorer", layout="wide")
st.title("Phoneme vs Whole-word AAC: Time-per-Word Explorer")
st.caption(
    "Tune assumptions and compare predicted time per produced word. "
    "We combine a Hick's law choice-time with a visual-search/cognitive-load term that grows with on-screen button density."
)

with st.expander("How the model works (short)", expanded=False):
    st.markdown(
        """
        **Per selection time** = Hick's law (choice among N options) + Visual search (scan M visible buttons).
        
        - Hick's law: T_choice(N) = T0 + c · log2(N + 1)
        - Visual search: T_search(M) = VS0 · (M / M_ref)^α
          
          • VS0 is base search time at reference density M_ref (e.g., 47).  
          • α controls how sharply search grows with density (try 0.3–0.7).
        
        Phoneme layout: ~avg_phonemes selections on a flat page of N_ph buttons.  
        Word (indexed, **idealized upper bound**): pick a page among P=ceil(V/B) (with a page index UI that shows up to page_index_capacity buttons at once), then pick a word among B on that page.  
        Word (serial, **more realistic**): flip pages (expected (P−1)/2 flips at T_flip each), then pick among B.
        
        **Reality check:** current AAC products don’t provide true direct indexing at large V. Treat the indexed curve as an *upper bound*.
        """
    )

with st.sidebar:
    st.header("Assumptions")
    V_min, V_max, V_step = 50, 10000, 50
    V_default = 1500
    V = st.slider("Vocabulary size V (unique words)", min_value=V_min, max_value=V_max, value=V_default, step=V_step)

    B = st.number_input("Buttons per page (B)", min_value=6, max_value=120, value=60, step=6)
    N_ph = st.number_input("Phoneme keys (N_ph)", min_value=20, max_value=80, value=47, step=1)
    avg_phonemes = st.number_input("Average phonemes per word", min_value=1.5, max_value=5.0, value=2.5, step=0.1)

    st.markdown("---")
    st.subheader("Timing parameters")
    T0 = st.number_input("T0 (base time per choice, s)", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
    c = st.number_input("c (s per bit)", min_value=0.05, max_value=0.5, value=0.15, step=0.01)
    T_flip = st.number_input("T_flip (serial page flip, s)", min_value=0.0, max_value=1.0, value=0.35, step=0.05)

    st.markdown("---")
    st.subheader("Visual search / cognitive load")
    VS0 = st.number_input("VS0 (base visual search per selection, s)", min_value=0.0, max_value=2.0, value=0.50, step=0.05)
    M_ref = st.number_input("Reference density M_ref (buttons)", min_value=10, max_value=100, value=47, step=1)
    alpha = st.number_input("Density exponent α", min_value=0.1, max_value=1.0, value=0.5, step=0.05)
    page_index_capacity = st.number_input("Page-index buttons visible at once", min_value=4, max_value=60, value=12, step=1)

    st.markdown("---")
    st.subheader("Plot options")
    modes: List[str] = st.multiselect(
        "Compare models",
        options=["Phoneme (flat)", "Word (indexed — upper bound)", "Word (serial — realistic)"],
        default=["Phoneme (flat)", "Word (indexed — upper bound)", "Word (serial — realistic)"]
    )

    show_breakeven = st.checkbox("Show break-even markers (vs phoneme)", value=True)
    show_age_markers = st.checkbox("Show age vocabulary markers (≈50, 1k, 10k)", value=True)
    show_frequency_markers = st.checkbox("Show frequency markers (25, 100, 1k common words)", value=True)

# -------------------------
# Compute curves
# -------------------------

Vs = np.arange(V_min, V_max + V_step, V_step)
phoneme_curve = np.array([
    time_phoneme(avg_phonemes, N_ph, T0, c, VS0, M_ref, alpha) for _ in Vs
])
word_indexed_curve = np.array([
    time_word_indexed(v, B, T0, c, VS0, M_ref, alpha, page_index_capacity) for v in Vs
])
word_serial_curve = np.array([
    time_word_serial(v, B, T_flip, T0, c, VS0, M_ref, alpha) for v in Vs
])

# Per-selected V snapshot
Tp = time_phoneme(avg_phonemes, N_ph, T0, c, VS0, M_ref, alpha)
Twi = time_word_indexed(V, B, T0, c, VS0, M_ref, alpha, page_index_capacity)
Tws = time_word_serial(V, B, T_flip, T0, c, VS0, M_ref, alpha)
P_pages = max(math.ceil(V / B), 1)

# -------------------------
# Headline metrics
# -------------------------

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Pages (ceil(V/B))", P_pages)
with col2:
    st.metric("Phoneme (s/word)", f"{Tp:.3f}")
with col3:
    st.metric("Word Indexed (s/word)", f"{Twi:.3f}")
with col4:
    st.metric("Word Serial (s/word)", f"{Tws:.3f}")

# -------------------------
# Plot
# -------------------------

fig = plt.figure(figsize=(9, 5.5))
if "Phoneme (flat)" in modes:
    plt.plot(Vs, phoneme_curve, label="Phoneme (flat)")
if "Word (indexed — upper bound)" in modes:
    plt.plot(Vs, word_indexed_curve, label="Word (indexed — upper bound)")
if "Word (serial — realistic)" in modes:
    plt.plot(Vs, word_serial_curve, label="Word (serial — realistic)")

plt.xlabel("Vocabulary size V (unique words)")
plt.ylabel("Predicted time per produced word (seconds)")
plt.title("Predicted Time per Word vs Vocabulary Size (with visual search)")

# Age & frequency markers
if show_age_markers:
    for x, lbl in [(50, "~Age 1: ~50"), (1000, "~Age 3: ~1k"), (10000, "~Age 5: ~10k")]:
        if V_min <= x <= V_max:
            plt.axvline(x=x, linestyle=":")
            plt.text(x, plt.ylim()[1]*0.92, lbl, rotation=90, va="top")

if show_frequency_markers:
    for x, lbl in [(25, "Top 25 ≈33%"), (100, "Top 100 ≈50%"), (1000, "Top 1k ≈89%")]:
        if V_min <= x <= V_max:
            plt.axvline(x=x, linestyle="--")
            plt.text(x, plt.ylim()[1]*0.78, lbl, rotation=90, va="top")

plt.legend()
plt.tight_layout()
st.pyplot(fig, clear_figure=True)

# -------------------------
# Break-even markers
# -------------------------

if show_breakeven:
    be_idx = find_breakeven(Vs, phoneme_curve, word_indexed_curve)
    be_ser = find_breakeven(Vs, phoneme_curve, word_serial_curve)

    st.markdown("### Break-even (Phoneme ≤ Word)")
    st.write({
        "Indexed (upper bound)": be_idx,
        "Serial (realistic)": be_ser,
    })

# -------------------------
# Table of selected checkpoints
# -------------------------

check_Vs = [25, 50, 100, 240, 600, 1000, 1200, 2400, 3600]
rows = []
for v in check_Vs:
    rows.append({
        "V": v,
        "Pages": max(math.ceil(v / B), 1),
        "Phoneme (s)": round(time_phoneme(avg_phonemes, N_ph, T0, c, VS0, M_ref, alpha), 3),
        "Word Indexed (s)": round(time_word_indexed(v, B, T0, c, VS0, M_ref, alpha, page_index_capacity), 3),
        "Word Serial (s)": round(time_word_serial(v, B, T_flip, T0, c, VS0, M_ref, alpha), 3),
    })

st.markdown("### Examples table")
st.dataframe(rows, use_container_width=True)

# -------------------------
# Notes
# -------------------------

st.markdown(
    """
    **Notes**
    - The "Word (indexed — upper bound)" curve is *not* achievable in current AAC products at large V; it bounds what would be possible with perfect direct indexing.
    - Small, very frequent vocabularies (e.g., top 25/100/1k words) can make whole-word layouts feel faster because selection is one-and-done and search sets are tiny.
    - As vocabularies expand and real interfaces require paging/scrolling, phoneme layouts benefit from fixed targets, larger buttons, and zero navigation, improving speed and consistency.
    - Use VS0/α to approximate cognitive load: denser screens → higher visual search time. Calibrate with your user testing.
    - This model doesn’t include prediction, error correction, or motor learning; adding prediction typically reduces average phonemes per word and further helps phoneme input.
    """
)
