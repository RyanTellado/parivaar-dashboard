"""
app.py
------
Parivaar Probabilistic Clinical Dashboard — Streamlit UI

Two modules:
  1. Differential Diagnosis (Naive Bayes)
  2. Treatment Success Estimator (Beta-Bernoulli)

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from models import (
    NaiveBayesClassifier,
    BetaBernoulliModel,
    DIAGNOSES,
    SYMPTOMS,
    TREATMENTS,
    MALNUT_BOOST_SYMPTOMS,
    AGE_LOG_PRIOR_ADJUSTMENTS,
    TEAL, AMBER, BG_COLOR,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Parivaar Clinical Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>

/* ─────────────────────────────────────────────────────────────────────────
   ROOT
───────────────────────────────────────────────────────────────────────── */
html, body, .stApp {
    background-color: #f2f6f6 !important;
    font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
}

/* ─────────────────────────────────────────────────────────────────────────
   SIDEBAR
───────────────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0b2e2e !important;
    border-right: 2px solid #145252 !important;
}

/* Logo image — blend white background into dark sidebar */
[data-testid="stSidebar"] [data-testid="stImage"] img {
    mix-blend-mode: lighten;
    border-radius: 6px;
    display: block;
}

/* Sidebar markdown text — scoped tightly, no bleed */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #9ac8c8 !important;
    font-size: 0.90rem !important;
    line-height: 1.65 !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
    color: #cce8e8 !important;
}

/* Sidebar widget labels (Age, District, etc.) */
[data-testid="stSidebar"] label {
    color: #b8d8d8 !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
}

/* Sidebar selectbox dropdown text (the chosen value shown in the box) */
[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
    color: #1a2e2e !important;
    font-size: 0.93rem !important;
    text-transform: none !important;
    font-weight: 500 !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: #1e4444 !important;
    margin: 14px 0 !important;
}

/* Malnourished checkbox — amber card so it stands out on dark background */
[data-testid="stSidebar"] div[data-testid="stCheckbox"] {
    background: rgba(232, 168, 56, 0.10) !important;
    border: 1.5px solid rgba(232, 168, 56, 0.50) !important;
    border-radius: 8px !important;
    padding: 0 !important;
    margin: 4px 0 !important;
    width: 100% !important;
}
[data-testid="stSidebar"] div[data-testid="stCheckbox"] > label {
    display: flex !important;
    align-items: center !important;
    min-height: 44px !important;
    padding: 10px 14px !important;
    width: 100% !important;
    cursor: pointer !important;
    box-sizing: border-box !important;
}
[data-testid="stSidebar"] div[data-testid="stCheckbox"]:has(input:checked) {
    background: rgba(232, 168, 56, 0.22) !important;
    border-color: #e8a838 !important;
}
[data-testid="stSidebar"] div[data-testid="stCheckbox"] p {
    color: #f0d898 !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    margin: 0 !important;
}

/* ─────────────────────────────────────────────────────────────────────────
   MAIN CONTENT AREA
───────────────────────────────────────────────────────────────────────── */
[data-testid="stMainBlockContainer"],
[data-testid="block-container"] {
    background-color: #f2f6f6 !important;
    padding-top: 1.2rem !important;
}

/* ─────────────────────────────────────────────────────────────────────────
   PAGE HEADER BANNER
───────────────────────────────────────────────────────────────────────── */
.dash-header {
    background: linear-gradient(120deg, #0b2e2e 0%, #0d5c5c 60%, #0f7070 100%);
    border-radius: 10px;
    padding: 28px 36px 24px 36px;
    margin-bottom: 22px;
    border: 1px solid #0d5c5c;
}
.dash-header h1 {
    margin: 0 0 6px 0;
    font-size: 1.55rem;
    font-weight: 700;
    color: #ffffff !important;
    letter-spacing: -0.4px;
    line-height: 1.2;
}
.dash-header .subtitle {
    color: rgba(255,255,255,0.78);
    font-size: 0.88rem;
    margin: 0 0 14px 0;
    line-height: 1.5;
}
.dash-badge {
    display: inline-block;
    background: rgba(232,168,56,0.20);
    color: #f0c860;
    border: 1px solid rgba(232,168,56,0.45);
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 0.70rem;
    font-weight: 700;
    letter-spacing: 0.09em;
}

/* ─────────────────────────────────────────────────────────────────────────
   TABS
───────────────────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    background-color: #dce8e8 !important;
    border-radius: 8px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: none !important;
    margin-bottom: 4px !important;
}
[data-baseweb="tab"] {
    background-color: transparent !important;
    border-radius: 6px !important;
    color: #2a5050 !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 8px 22px !important;
    border: none !important;
    letter-spacing: 0.01em !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    background-color: #0d6e6e !important;
    color: #ffffff !important;
}

/* ─────────────────────────────────────────────────────────────────────────
   SECTION LABELS (module sub-headers)
───────────────────────────────────────────────────────────────────────── */
.section-label {
    font-size: 0.70rem;
    font-weight: 700;
    color: #5a8888;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    margin: 0 0 6px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #ccdede;
}
.section-title {
    font-size: 1.08rem;
    font-weight: 700;
    color: #0b2e2e;
    margin: 0 0 4px 0;
    letter-spacing: -0.2px;
}
.section-desc {
    font-size: 0.86rem;
    color: #4a6e6e;
    margin: 0 0 16px 0;
    line-height: 1.5;
}

/* ─────────────────────────────────────────────────────────────────────────
   SYMPTOM CHECKBOXES — full-width, consistent height, clean card style
───────────────────────────────────────────────────────────────────────── */

/* Outer container: fill full column width */
div[data-testid="stCheckbox"] {
    width: 100% !important;
    display: block !important;
    box-sizing: border-box !important;
    background: #ffffff !important;
    border: 1.5px solid #c8dede !important;
    border-radius: 8px !important;
    padding: 0 !important;
    margin: 4px 0 !important;
    overflow: hidden !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
div[data-testid="stCheckbox"]:hover {
    border-color: #0d6e6e !important;
    box-shadow: 0 2px 8px rgba(13, 110, 110, 0.10) !important;
}
/* Checked state — teal tint */
div[data-testid="stCheckbox"]:has(input:checked) {
    border-color: #0d6e6e !important;
    background: #eef7f7 !important;
}

/* Inner label — the actual click target, fills the card */
div[data-testid="stCheckbox"] > label {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    min-height: 52px !important;
    padding: 12px 18px !important;
    cursor: pointer !important;
    box-sizing: border-box !important;
    gap: 10px !important;
}

/* Checkbox label text */
div[data-testid="stCheckbox"] p {
    font-size: 0.97rem !important;
    font-weight: 500 !important;
    color: #1a3535 !important;
    margin: 0 !important;
    line-height: 1.3 !important;
}

/* ─────────────────────────────────────────────────────────────────────────
   OVERRIDE: sidebar checkboxes must keep amber style, not teal
   (order matters — these rules come after the general checkbox rules)
───────────────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] div[data-testid="stCheckbox"]:has(input:checked) {
    border-color: #e8a838 !important;
    background: rgba(232, 168, 56, 0.22) !important;
}
[data-testid="stSidebar"] div[data-testid="stCheckbox"]:hover {
    border-color: #e8a838 !important;
    box-shadow: 0 2px 8px rgba(232, 168, 56, 0.15) !important;
}

/* ─────────────────────────────────────────────────────────────────────────
   METRIC CARDS
───────────────────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid #daeaea !important;
    border-top: 3px solid #0d6e6e !important;
    border-radius: 8px !important;
    padding: 16px 18px !important;
    box-shadow: 0 1px 4px rgba(11, 46, 46, 0.06) !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color: #5a8080 !important;
    font-size: 0.73rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0b2e2e !important;
    font-size: 1.45rem !important;
    font-weight: 700 !important;
}

/* ─────────────────────────────────────────────────────────────────────────
   EXPANDER
───────────────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #d0e4e4 !important;
    border-radius: 8px !important;
    margin-top: 8px !important;
}
[data-testid="stExpander"] summary {
    color: #0b2e2e !important;
    font-weight: 600 !important;
    font-size: 0.86rem !important;
    padding: 12px 16px !important;
}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
    color: #2a4444 !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
}

/* ─────────────────────────────────────────────────────────────────────────
   SELECTBOX
───────────────────────────────────────────────────────────────────────── */
[data-testid="stMainBlockContainer"] [data-baseweb="select"] {
    background-color: #ffffff !important;
    border-radius: 7px !important;
}
[data-testid="stMainBlockContainer"] label {
    color: #4a6e6e !important;
    font-size: 0.80rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ─────────────────────────────────────────────────────────────────────────
   TOGGLE
───────────────────────────────────────────────────────────────────────── */
[data-testid="stToggle"] p {
    color: #2a4444 !important;
    font-weight: 500 !important;
    font-size: 0.90rem !important;
}

/* ─────────────────────────────────────────────────────────────────────────
   DIVIDER
───────────────────────────────────────────────────────────────────────── */
hr {
    border-color: #ccdede !important;
    margin: 14px 0 !important;
}

/* ─────────────────────────────────────────────────────────────────────────
   ALERT / INFO BOXES
───────────────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 0.87rem !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data and model loading (cached — only runs once per session)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("data/patients.csv")


@st.cache_resource
def load_nb_model(df: pd.DataFrame) -> NaiveBayesClassifier:
    nb = NaiveBayesClassifier()
    nb.fit(df)
    return nb


df       = load_data()
nb_model = load_nb_model(df)
bb_model = BetaBernoulliModel()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:

    # Logo + wordmark
    st.image("Parivaarlogo.png", width=80)
    st.markdown(
        "<div style='padding:6px 0 14px 0'>"
        "  <div style='color:#e8a838;font-size:1.35rem;font-weight:800;"
        "letter-spacing:0.05em;line-height:1.2'>PARIVAAR</div>"
        "  <div style='color:#5a9898;font-size:0.72rem;font-weight:600;"
        "letter-spacing:0.14em;margin-top:3px'>CLINICAL DASHBOARD</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Patient profile section
    st.markdown(
        "<div style='color:#5a9898;font-size:0.74rem;font-weight:700;"
        "letter-spacing:0.11em;text-transform:uppercase;margin-bottom:14px'>"
        "Patient Profile</div>",
        unsafe_allow_html=True,
    )

    patient_age  = st.slider("Age (years)", min_value=0, max_value=90, value=28)

    district_options = ["All Districts"] + [f"District {i}" for i in range(1, 18)]
    patient_district = st.selectbox("District", district_options, index=0)
    # None = no district filter (all districts); otherwise extract integer
    patient_district_num = (
        None if patient_district == "All Districts"
        else int(patient_district.split()[1])
    )

    st.markdown(
        "<div style='color:#5a9898;font-size:0.74rem;font-weight:700;"
        "letter-spacing:0.11em;text-transform:uppercase;"
        "margin-top:16px;margin-bottom:6px'>Nutritional Status</div>",
        unsafe_allow_html=True,
    )
    patient_malnourished = st.checkbox("Malnourished", value=False)

    st.divider()

    # About section
    st.markdown(
        "<div style='color:#5a9898;font-size:0.74rem;font-weight:700;"
        "letter-spacing:0.11em;text-transform:uppercase;margin-bottom:10px'>"
        "About</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Bayesian decision support for health workers serving "
        "**500K+ patients** in rural India and South Sudan."
    )
    st.markdown(
        "<div style='color:#3a6060;font-size:0.76rem;margin-top:10px'>"
        "Stanford CS109 · Probability Challenge · 2026</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='dash-header'>"
    "  <h1>Parivaar Probabilistic Clinical Dashboard</h1>"
    "  <p class='subtitle'>"
    "    Bayesian decision support for rural health workers — "
    "    all probability estimates derived from patient outcomes data."
    "  </p>"
    "  <span class='dash-badge'>STANFORD CS109 · PROBABILITY CHALLENGE</span>"
    "</div>",
    unsafe_allow_html=True,
)

# Session state: carry top diagnosis from Tab 1 → Tab 2
if "top_diagnosis" not in st.session_state:
    st.session_state.top_diagnosis = DIAGNOSES[0]

# Compute age group once — used in both Module 1 (prior adjustment)
# and Module 2 (cohort filter).
if patient_age < 18:
    patient_age_group  = "child"
    age_group_label    = "Children (0–17)"
elif patient_age <= 60:
    patient_age_group  = "adult"
    age_group_label    = "Adults (18–60)"
else:
    patient_age_group  = "elderly"
    age_group_label    = "Elderly (61+)"


# ===========================================================================
# Tabs
# ===========================================================================
tab1, tab2 = st.tabs([
    "Module 1 — Differential Diagnosis",
    "Module 2 — Treatment Estimator",
])


# ===========================================================================
# TAB 1: Differential Diagnosis (Naive Bayes)
# ===========================================================================
with tab1:

    st.markdown(
        "<p class='section-label'>Naive Bayes Classifier</p>"
        "<p class='section-title'>Presenting Symptoms</p>"
        "<p class='section-desc'>"
        "Check the symptoms observed in this patient. "
        "The posterior probability distribution over all six diagnoses "
        "recalculates in real time using Bayes' theorem."
        "</p>",
        unsafe_allow_html=True,
    )

    # Symptom labels — no emojis, clean text
    symptom_labels = {
        "fever":               "Fever",
        "chills":              "Chills",
        "fatigue":             "Fatigue",
        "cough":               "Cough",
        "night_sweats":        "Night Sweats",
        "headache":            "Headache",
        "nausea":              "Nausea",
        "joint_pain":          "Joint Pain",
        "pale_complexion":     "Pale Complexion",
        "shortness_of_breath": "Shortness of Breath",
        "abdominal_pain":      "Abdominal Pain",
        "weight_loss":         "Weight Loss",
    }

    # Active-modifier banner — always shown so the user sees what's in effect
    age_adj    = AGE_LOG_PRIOR_ADJUSTMENTS[patient_age_group]
    # Identify which diagnoses are notably boosted / penalised for this age group
    boosted    = [DIAGNOSES[i] for i, v in enumerate(age_adj) if v >= 0.20]
    penalised  = [DIAGNOSES[i] for i, v in enumerate(age_adj) if v <= -0.20]

    modifier_parts = []
    if boosted or penalised:
        age_note = f"Age {patient_age} ({age_group_label})"
        if boosted:
            age_note += f" — prior boosted for {', '.join(boosted)}"
        if penalised:
            age_note += f"; reduced for {', '.join(penalised)}"
        modifier_parts.append(age_note)

    if patient_malnourished:
        modifier_parts.append(
            "Malnourished — using malnourished-stratum likelihoods "
            "(fatigue, pale complexion, weight loss elevated +15%)"
        )

    if modifier_parts:
        items_html = "".join(
            f"<li style='margin-bottom:3px'>{p}</li>" for p in modifier_parts
        )
        st.markdown(
            "<div style='background:#fffbf0;border:1px solid #e8c870;"
            "border-left:4px solid #e8a838;border-radius:8px;"
            "padding:11px 18px;margin-bottom:14px;font-size:0.85rem;color:#5a3e00'>"
            "<strong style='color:#7a5200'>Active modifiers:</strong>"
            f"<ul style='margin:6px 0 0 0;padding-left:18px;line-height:1.7'>"
            f"{items_html}</ul></div>",
            unsafe_allow_html=True,
        )

    # 2-column symptom grid — all checkboxes fill full column width via CSS
    col_a, col_b = st.columns(2, gap="medium")
    symptom_values = {}
    symptom_keys   = list(symptom_labels.keys())

    for i, sym_key in enumerate(symptom_keys):
        col = col_a if i % 2 == 0 else col_b
        base_label = symptom_labels[sym_key]
        # Append malnutrition marker for the three affected symptoms
        if patient_malnourished and sym_key in MALNUT_BOOST_SYMPTOMS:
            display_label = base_label + "  + M"
        else:
            display_label = base_label
        with col:
            symptom_values[sym_key] = int(
                st.checkbox(display_label, key=f"sym_{sym_key}")
            )

    st.divider()

    # --- Compute posterior ---
    # Conditions on: age-adjusted priors + malnutrition-stratum likelihoods
    posteriors = nb_model.predict_proba(
        symptom_values,
        malnourished=bool(patient_malnourished),
        age_group=patient_age_group,
    )
    sorted_diags = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
    top_diag, top_prob = sorted_diags[0]
    st.session_state.top_diagnosis = top_diag

    # --- Results header ---
    st.markdown(
        "<p class='section-label'>Naive Bayes Posterior</p>"
        "<p class='section-title'>Diagnostic Probability Distribution</p>",
        unsafe_allow_html=True,
    )

    diag_names = [d for d, _ in sorted_diags]
    diag_probs = [p for _, p in sorted_diags]
    bar_colors = ["#e8a838" if d == top_diag else "#0d6e6e" for d in diag_names]

    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    bars = ax.barh(
        diag_names[::-1], diag_probs[::-1],
        color=bar_colors[::-1],
        edgecolor="none",
        height=0.55,
    )

    # Probability labels — white inside bar if wide enough, dark outside if narrow
    for bar, prob in zip(bars, diag_probs[::-1]):
        w = bar.get_width()
        if w > 0.12:
            ax.text(w - 0.015, bar.get_y() + bar.get_height() / 2,
                    f"{prob:.1%}", va="center", ha="right",
                    fontsize=9.5, fontweight="700", color="white")
        else:
            ax.text(w + 0.012, bar.get_y() + bar.get_height() / 2,
                    f"{prob:.1%}", va="center", ha="left",
                    fontsize=9.5, fontweight="700", color="#2a4444")

    ax.set_xlim(0, 1.04)
    ax.set_xlabel("P(diagnosis | symptoms)", fontsize=9, color="#6a8888", labelpad=8)
    ax.tick_params(axis="y", labelsize=10.5, colors="#1a3535", pad=8)
    ax.tick_params(axis="x", labelsize=8, colors="#aabcbc")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color="#eef2f2", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    top_p  = mpatches.Patch(facecolor="#e8a838", label="Top diagnosis",   edgecolor="none")
    rest_p = mpatches.Patch(facecolor="#0d6e6e", label="Other diagnoses", edgecolor="none")
    ax.legend(handles=[top_p, rest_p], fontsize=8.5, loc="lower right",
              framealpha=0.9, edgecolor="#dde8e8", frameon=True)

    fig.tight_layout(pad=1.2)
    st.pyplot(fig)
    plt.close(fig)

    # --- Confidence callout ---
    n_checked = sum(symptom_values.values())
    if top_prob >= 0.60:
        bg, border, badge_bg, badge_color = "#edf7ee", "#4caf50", "#d4edda", "#1d6b2a"
        confidence_label = "High confidence"
    elif top_prob >= 0.35:
        bg, border, badge_bg, badge_color = "#fffbf0", "#e8a838", "#fff3cd", "#7a5200"
        confidence_label = "Moderate confidence"
    else:
        bg, border, badge_bg, badge_color = "#fdf3f3", "#d9534f", "#fde8e8", "#8b1a1a"
        confidence_label = "Low confidence — consider additional testing"

    st.markdown(
        f"<div style='background:{bg};border:1px solid {border};"
        f"border-left:4px solid {border};"
        f"border-radius:8px;padding:16px 20px;margin-top:4px'>"
        f"  <div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap'>"
        f"    <div style='font-size:1.15rem;font-weight:800;color:#0b2e2e'>{top_diag}</div>"
        f"    <div style='background:{badge_bg};color:{badge_color};"
        f"border-radius:4px;padding:2px 10px;font-size:0.72rem;"
        f"font-weight:700;letter-spacing:0.06em'>{confidence_label.upper()}</div>"
        f"  </div>"
        f"  <div style='color:#4a6060;font-size:0.86rem;margin-top:6px'>"
        f"    Posterior probability: <strong style='color:#0b2e2e'>{top_prob:.1%}</strong>"
        f"    &nbsp;·&nbsp; {n_checked} symptom(s) observed"
        f"    &nbsp;·&nbsp; {age_group_label}"
        + ("" if n_checked > 0 else " &nbsp;·&nbsp; <em>Showing prior distribution</em>")
        + f"  </div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # --- Math expander ---
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    with st.expander("How this works — Naive Bayes probability math"):
        st.markdown(r"""
**Bayes' Theorem** — the core of this classifier:

$$P(D \mid \mathbf{s}, m, a) \propto P(D, a) \cdot \prod_{i=1}^{12} P(s_i \mid D, m)$$

where $m$ = malnourished status, $a$ = age group.

**Age-adjusted priors** — domain-knowledge correction applied before inference:

$$\log P(D, a) = \log \hat{P}(D) + \delta(D, a)$$

$\hat{P}(D)$ is the MLE prior from the training data; $\delta(D, a)$ is an additive log-space adjustment encoding known age-disease relationships (e.g. ARI and Malaria are more prevalent in children; TB is more prevalent in the elderly). The adjusted priors are renormalized to sum to 1.

**Malnutrition-stratified likelihoods** — two separate likelihood tables are fit from the training data, one for $m=0$ and one for $m=1$:

$$\hat{P}(s_i = 1 \mid D, m) = \frac{\text{count}(s_i{=}1, D, m) + 1}{\text{count}(D, m) + 2}$$

**Log-space computation** prevents floating-point underflow:

$$\log P(D \mid \mathbf{s}, m, a) \propto \underbrace{\log \hat{P}(D) + \delta(D,a)}_{\text{age-adjusted prior}} + \sum_{i=1}^{12} \log \hat{P}(s_i \mid D, m)$$
""")
        st.caption(
            f"Training data: {len(df):,} patients · "
            f"{len(df[df.malnourished==1]):,} malnourished "
            f"({len(df[df.malnourished==1])/len(df):.0%}) · "
            f"{len(DIAGNOSES)} diagnoses · 12 symptoms"
        )


# ===========================================================================
# TAB 2: Treatment Success Estimator (Beta-Bernoulli)
# ===========================================================================
with tab2:

    st.markdown(
        "<p class='section-label'>Beta-Bernoulli Conjugate Model</p>"
        "<p class='section-title'>Treatment Success Estimator</p>"
        "<p class='section-desc'>"
        "Select a diagnosis and treatment to see the Bayesian posterior over "
        "treatment success probability θ, estimated from similar patients in the dataset."
        "</p>",
        unsafe_allow_html=True,
    )

    # --- Controls row ---
    ctrl1, ctrl2 = st.columns(2, gap="medium")
    with ctrl1:
        selected_diagnosis = st.selectbox(
            "Diagnosis",
            options=DIAGNOSES,
            index=DIAGNOSES.index(st.session_state.top_diagnosis),
            help="Pre-filled from the top result in Module 1.",
        )
    with ctrl2:
        selected_treatment = st.selectbox(
            "Treatment", options=TREATMENTS[selected_diagnosis]
        )

    # All three sidebar filters are always applied:
    #   age_group  — from the Age slider
    #   malnourished — from the Malnourished checkbox
    #   district   — from the District dropdown (None = All Districts)
    malnut_filter = True if patient_malnourished else None

    # --- Compute posteriors at each filter level for the funnel display ---
    # Unfiltered baseline
    alpha_all, beta_all, n_all = bb_model.get_posterior_params(
        df, selected_diagnosis, selected_treatment,
    )
    # After age filter
    _, _, n_age = bb_model.get_posterior_params(
        df, selected_diagnosis, selected_treatment,
        age_group=patient_age_group,
    )
    # After age + malnutrition
    _, _, n_age_malnut = bb_model.get_posterior_params(
        df, selected_diagnosis, selected_treatment,
        age_group=patient_age_group,
        malnourished=malnut_filter,
    )
    # Final: age + malnutrition + district (primary curve)
    alpha, beta_param, n_cohort = bb_model.get_posterior_params(
        df, selected_diagnosis, selected_treatment,
        age_group=patient_age_group,
        malnourished=malnut_filter,
        district=patient_district_num,
    )
    show_comparison = True  # always compare against unfiltered baseline

    mean_estimate = bb_model.posterior_mean(alpha, beta_param)
    variance_est  = bb_model.posterior_variance(alpha, beta_param)
    ci_lo, ci_hi  = bb_model.credible_interval(alpha, beta_param)
    ci_width      = ci_hi - ci_lo

    st.divider()

    # --- Main layout: chart (left) | stats (right) ---
    plot_col, stats_col = st.columns([11, 7], gap="large")

    with plot_col:
        st.markdown(
            "<p class='section-label'>Posterior Distribution</p>",
            unsafe_allow_html=True,
        )
        fig = bb_model.plot_posterior(
            alpha, beta_param, n_cohort,
            selected_diagnosis, selected_treatment,
            age_group=age_group_label,
            malnourished=malnut_filter,
            comparison_params=(alpha_all, beta_all, n_all),
            comparison_label=f"All patients (n={n_all})",
        )
        st.pyplot(fig)
        plt.close(fig)

    with stats_col:
        st.markdown(
            "<p class='section-label'>Posterior Estimates</p>",
            unsafe_allow_html=True,
        )

        m1, m2 = st.columns(2)
        with m1:
            st.metric("Success Rate",  f"{mean_estimate:.1%}",
                      help="Posterior mean: α / (α + β)")
        with m2:
            st.metric("Sample Size",  f"{n_cohort:,}",
                      help="Patients with this diagnosis + treatment.")

        m3, m4 = st.columns(2)
        with m3:
            st.metric("95% CI Lower", f"{ci_lo:.1%}",
                      help="Lower bound of equal-tailed 95% credible interval.")
        with m4:
            st.metric("95% CI Upper", f"{ci_hi:.1%}",
                      help="Upper bound of equal-tailed 95% credible interval.")

        # Beta parameter card
        st.markdown(
            f"<div style='background:#ffffff;border:1px solid #cddede;"
            f"border-radius:8px;padding:16px 18px;margin-top:10px'>"
            f"  <div style='font-size:0.70rem;font-weight:700;color:#5a8080;"
            f"letter-spacing:0.10em;text-transform:uppercase;"
            f"margin-bottom:10px'>Posterior Parameters</div>"
            f"  <div style='font-size:1.05rem;font-weight:700;color:#0b2e2e;"
            f"font-family:monospace;margin-bottom:4px'>"
            f"Beta( α={alpha:.0f},  β={beta_param:.0f} )</div>"
            f"  <div style='font-size:0.81rem;color:#6a8888;line-height:1.7'>"
            f"Prior Beta(2, 2)<br>"
            f"+ {alpha-2:.0f} successes &nbsp;+&nbsp; {beta_param-2:.0f} failures<br>"
            f"Variance: <strong style='color:#2a4444'>{variance_est:.5f}</strong>"
            f"<span style='color:#9abcbc'> (→ 0 as n → ∞)</span>"
            f"  </div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # --- Cohort funnel: show how each filter reduces n ---
        mean_all = bb_model.posterior_mean(alpha_all, beta_all)
        shift    = mean_estimate - mean_all
        arrow    = "↓" if shift < 0 else "↑"

        def funnel_row(label, n, n_prev, is_last=False):
            pct  = f"{n/n_prev:.0%} of previous" if n_prev != n_all else ""
            bold = "font-weight:700;" if is_last else ""
            return (
                f"<tr>"
                f"<td style='padding:3px 8px 3px 0;color:#7a5200;{bold}'>{label}</td>"
                f"<td style='padding:3px 8px;text-align:right;color:#0b2e2e;{bold}font-family:monospace'>{n:,}</td>"
                f"<td style='padding:3px 0;color:#9a8060;font-size:0.78rem'>{pct}</td>"
                f"</tr>"
            )

        rows = funnel_row("All patients", n_all, n_all)
        rows += funnel_row(f"After age filter ({age_group_label})", n_age, n_all)
        if malnut_filter is not None:
            rows += funnel_row("After malnutrition filter", n_age_malnut, n_age)
        if patient_district_num is not None:
            rows += funnel_row(f"After {patient_district} filter", n_cohort,
                               n_age_malnut if malnut_filter else n_age, is_last=True)
        else:
            # Re-mark last row as bold
            rows = rows.replace(
                funnel_row(f"After age filter ({age_group_label})", n_age, n_all),
                funnel_row(f"After age filter ({age_group_label})", n_age, n_all, is_last=True)
                if malnut_filter is None else
                funnel_row(f"After age filter ({age_group_label})", n_age, n_all),
            )

        st.markdown(
            f"<div style='background:#fffbf0;border:1px solid #e8c870;"
            f"border-left:4px solid #e8a838;border-radius:8px;"
            f"padding:14px 16px;margin-top:10px;font-size:0.84rem;color:#5a3e00'>"
            f"  <div style='font-size:0.68rem;font-weight:700;color:#9a7040;"
            f"letter-spacing:0.10em;text-transform:uppercase;margin-bottom:8px'>"
            f"Cohort Filters — Sample Funnel</div>"
            f"  <table style='border-collapse:collapse;width:100%'>{rows}</table>"
            f"  <div style='margin-top:8px;padding-top:8px;border-top:1px solid #e8d098;"
            f"color:#7a5200;font-size:0.82rem'>"
            f"  Final cohort: <strong>n = {n_cohort:,}</strong> &nbsp;·&nbsp; "
            f"  Success rate: <strong>{mean_estimate:.1%}</strong> "
            f"  ({arrow}&nbsp;{abs(shift):.1%} vs. population {mean_all:.1%})"
            f"  </div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Math expander
    st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
    with st.expander("How this works — Beta-Bernoulli probability math"):
        st.markdown(r"""
**Generative model:** each treatment outcome is a Bernoulli trial with unknown success probability $\theta$.

$$\text{outcome}_i \mid \theta \sim \text{Bernoulli}(\theta)$$

**Prior** — Beta is the conjugate prior for Bernoulli. We start with a weak, symmetric belief:

$$\theta \sim \text{Beta}(2,\ 2) \qquad \text{mean} = 0.5,\quad \text{var} = 0.05$$

**Posterior update** — conjugacy gives a closed-form result with no numerical integration:

$$\theta \mid \text{data} \sim \text{Beta}(\alpha_0 + s,\ \beta_0 + f)$$

where $s$ = successes, $f$ = failures observed in the cohort, and the prior is $\alpha_0 = \beta_0 = 2$.

**Posterior mean and variance:**

$$\text{Mean} = \frac{\alpha}{\alpha + \beta} \qquad \text{Variance} = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$$

**95% Credible Interval** via `scipy.stats.beta.ppf` (percent-point function):

$$P(L \leq \theta \leq U \mid \text{data}) = 0.95$$

As $n$ grows, $\alpha + \beta$ grows, variance shrinks, and the curve concentrates around the true $\theta$.
Applying any filter (district, age group, malnutrition) reduces $n$ and widens the posterior — correctly expressing more uncertainty.
""")
        st.caption(
            f"Prior: Beta(2, 2)  ·  "
            f"Posterior: Beta({alpha:.0f}, {beta_param:.0f})  ·  "
            f"mean = {mean_estimate:.3f}  ·  var = {variance_est:.5f}  ·  "
            f"cohort: {patient_district} · n = {n_cohort:,}"
        )
