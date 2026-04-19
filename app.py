"""
StatTest Suite — Automated Parametric & Non-Parametric Hypothesis Testing
SPSS-equivalent formulas | High-contrast UI | Full downloadable outputs
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.lines as mlines
from scipy import stats
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus import Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="StatTest Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# CSS — HIGH CONTRAST
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;600;700&family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,300;1,400&display=swap');

html, body, [class*="css"] { font-family:'Source Serif 4',Georgia,serif; color:#F0EDE8; }
.stApp { background:#0D0D0D; }
section[data-testid="stSidebar"] { background:#111; border-right:1px solid #2A2A2A; }
section[data-testid="stSidebar"] * { color:#E8E4DF!important; }
h1,h2,h3,h4,h5 { font-family:'Source Code Pro',monospace!important; color:#FFFFFF!important; }

.app-header {
    background:linear-gradient(135deg,#1A1A1A 0%,#0D0D0D 100%);
    border-bottom:2px solid #E8B84B;
    padding:1.8rem 2rem 1.4rem; margin:-1rem -1rem 2rem -1rem;
}
.app-title { font-family:'Source Code Pro',monospace; font-size:2.2rem; font-weight:700;
    color:#FFFFFF; letter-spacing:-0.03em; line-height:1.1; margin:0; }
.app-subtitle { font-family:'Source Serif 4',serif; font-style:italic; font-size:1rem;
    color:#A89878; margin-top:.35rem; }
.app-badge { display:inline-block; background:#E8B84B; color:#0D0D0D;
    font-family:'Source Code Pro',monospace; font-size:.62rem; font-weight:700;
    letter-spacing:.12em; text-transform:uppercase; padding:.2rem .6rem;
    border-radius:2px; margin-right:.4rem; margin-top:.7rem; }

.sec-head { font-family:'Source Code Pro',monospace; font-size:.66rem; font-weight:700;
    text-transform:uppercase; letter-spacing:.18em; color:#E8B84B;
    border-bottom:1px solid #2A2A2A; padding-bottom:.4rem; margin:1.6rem 0 .9rem; }

.metric-card { background:#181818; border:1px solid #2E2E2E; border-top:3px solid #E8B84B;
    padding:.9rem 1.1rem; border-radius:3px; margin:.3rem 0; }
.metric-card.red  { border-top-color:#E85454; }
.metric-card.green{ border-top-color:#4EBF85; }
.metric-card.blue { border-top-color:#5A9FD4; }
.metric-label { font-family:'Source Code Pro',monospace; font-size:.6rem; font-weight:700;
    text-transform:uppercase; letter-spacing:.12em; color:#8A8070; margin-bottom:.3rem; }
.metric-val { font-family:'Source Code Pro',monospace; font-size:1.4rem; font-weight:700;
    color:#FFFFFF; line-height:1.1; }
.metric-sub { font-size:.73rem; color:#6A6058; margin-top:.2rem; }

.verdict-banner { padding:1.3rem 2rem; border-radius:3px; margin:1rem 0; text-align:center; }
.verdict-reject { background:#1F0E0E; border:1px solid #6A2020; border-left:5px solid #E85454; }
.verdict-fail   { background:#0E1F14; border:1px solid #206A38; border-left:5px solid #4EBF85; }
.verdict-label  { font-family:'Source Code Pro',monospace; font-size:.63rem;
    text-transform:uppercase; letter-spacing:.18em; color:#8A8070; margin-bottom:.45rem; }
.verdict-text-reject { font-family:'Source Code Pro',monospace; font-size:1.4rem;
    font-weight:700; color:#E85454; }
.verdict-text-fail { font-family:'Source Code Pro',monospace; font-size:1.4rem;
    font-weight:700; color:#4EBF85; }
.verdict-sub { font-size:.83rem; color:#A09080; margin-top:.35rem; }

.info-box { background:#141820; border:1px solid #232B38; border-left:3px solid #5A9FD4;
    padding:.85rem 1rem; border-radius:3px; font-size:.87rem; color:#C0CEDD;
    line-height:1.7; margin:.6rem 0; }
.warn-box { background:#1C1810; border:1px solid #3A3010; border-left:3px solid #E8B84B;
    padding:.85rem 1rem; border-radius:3px; font-size:.87rem; color:#D4C090;
    line-height:1.7; margin:.6rem 0; }
.formula-box { background:#0A0A0A; border:1px solid #222; border-left:3px solid #E8B84B;
    padding:.9rem 1.3rem; border-radius:3px; font-family:'Source Code Pro',monospace;
    font-size:.8rem; color:#E8B84B; line-height:2; margin:.7rem 0; white-space:pre-wrap; }

.badge-pass { background:#0E2018; color:#4EBF85; font-family:'Source Code Pro',monospace;
    font-size:.6rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
    padding:.18rem .5rem; border-radius:2px; border:1px solid #1A4028; }
.badge-fail-b { background:#200E0E; color:#E85454; font-family:'Source Code Pro',monospace;
    font-size:.6rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase;
    padding:.18rem .5rem; border-radius:2px; border:1px solid #401A1A; }

.spss-table { width:100%; border-collapse:collapse; font-size:.84rem; margin:.7rem 0; }
.spss-table th { background:#1E1E1E; color:#E8B84B; font-family:'Source Code Pro',monospace;
    font-size:.7rem; text-transform:uppercase; letter-spacing:.1em; padding:.6rem .85rem;
    border-bottom:2px solid #E8B84B; text-align:left; }
.spss-table td { padding:.52rem .85rem; border-bottom:1px solid #1E1E1E; color:#F0EDE8; }
.spss-table tr:nth-child(even) td { background:#121212; }
.spss-table tr:hover td { background:#1A1A1A; }
.spss-table td.num { font-family:'Source Code Pro',monospace; text-align:right; color:#FFFFFF; }

.stButton>button { background:#E8B84B!important; color:#0D0D0D!important;
    font-family:'Source Code Pro',monospace!important; font-weight:700!important;
    font-size:.73rem!important; letter-spacing:.1em!important; text-transform:uppercase!important;
    border:none!important; border-radius:2px!important; padding:.55rem 1.6rem!important; }
.stButton>button:hover { background:#F5CA6A!important; }
.stDownloadButton>button { background:transparent!important; color:#E8B84B!important;
    border:1.5px solid #E8B84B!important; font-family:'Source Code Pro',monospace!important;
    font-size:.7rem!important; letter-spacing:.08em!important; font-weight:600!important;
    text-transform:uppercase!important; border-radius:2px!important; }
.stDownloadButton>button:hover { background:#E8B84B!important; color:#0D0D0D!important; }

div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stFileUploader"] label {
    font-family:'Source Code Pro',monospace!important; font-size:.68rem!important;
    color:#8A8070!important; text-transform:uppercase!important; letter-spacing:.12em!important; }
.stTabs [data-baseweb="tab-list"] { background:#111; gap:0; border-bottom:1px solid #2A2A2A; }
.stTabs [data-baseweb="tab"] { font-family:'Source Code Pro',monospace; font-size:.66rem;
    text-transform:uppercase; letter-spacing:.12em; color:#6A6058;
    background:transparent; border-radius:0; padding:.75rem 1.3rem; }
.stTabs [aria-selected="true"] { color:#E8B84B!important; border-bottom:2px solid #E8B84B!important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PLOT COLOUR CONSTANTS
# ══════════════════════════════════════════════════════════════
GOLD  = "#E8B84B"
RED   = "#E85454"
GREEN = "#4EBF85"
BLUE  = "#5A9FD4"
WHITE = "#F0EDE8"
GREY  = "#8A8070"
BG    = "#0D0D0D"
AXBG  = "#141414"

def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor":   BG,
        "axes.facecolor":     AXBG,
        "axes.edgecolor":     "#2A2A2A",
        "axes.labelcolor":    GREY,
        "axes.titlecolor":    WHITE,
        "axes.titlesize":     10.5,
        "axes.labelsize":     8.5,
        "xtick.color":        GREY,
        "ytick.color":        GREY,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "text.color":         WHITE,
        "grid.color":         "#1E1E1E",
        "grid.linewidth":     0.7,
        "lines.linewidth":    2,
        "font.family":        "monospace",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "legend.facecolor":   "#1A1A1A",
        "legend.edgecolor":   "#2A2A2A",
        "legend.fontsize":    8,
        "figure.dpi":         120,
    })

set_plot_style()

def fig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()

def draw_sep(ax, y, color="#2A2A2A"):
    ax.plot([0.04, 0.96], [y, y], color=color, linewidth=0.6,
            transform=ax.transAxes, clip_on=False)

def result_panel(ax, result, alpha_val):
    ax.set_facecolor("#0F0F0F")
    ax.axis("off")
    clr = RED if result["reject"] else GREEN
    verdict = "REJECT H₀" if result["reject"] else "FAIL TO REJECT H₀"
    border = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                            boxstyle="round,pad=0.01", lw=1.5,
                            edgecolor=clr, facecolor="#0F0F0F",
                            transform=ax.transAxes, clip_on=False)
    ax.add_patch(border)
    strip = FancyBboxPatch((0.02, 0.86), 0.96, 0.12,
                           boxstyle="round,pad=0.01", lw=0,
                           facecolor=clr + "33",
                           transform=ax.transAxes, clip_on=False)
    ax.add_patch(strip)
    ax.text(0.5, 0.91, verdict, ha="center", va="center",
            fontsize=10.5, color=clr, fontweight="bold",
            fontfamily="monospace", transform=ax.transAxes)
    rows = [
        ("Test Statistic", f"{result['statistic']:.4f}"),
        ("p-value",        f"{result['p_value']:.4f}" if result['p_value'] >= 0.0001 else "< 0.0001"),
        (result["effect_label"], f"{result['effect_size']:.4f}"),
        ("Effect Size",    interpret_effect(result["effect_size"], result["effect_label"])),
    ]
    if "df" in result:
        rows.insert(1, ("df", f"{result['df']:.4f}"))
    y = 0.78
    for lbl, val in rows:
        ax.text(0.08, y, lbl, ha="left", va="top", fontsize=7.5,
                color=GREY, fontfamily="monospace", transform=ax.transAxes)
        ax.text(0.92, y, val, ha="right", va="top", fontsize=8.5,
                color=WHITE, fontfamily="monospace", fontweight="bold",
                transform=ax.transAxes)
        y -= 0.12
        draw_sep(ax, y + 0.05)
    ax.text(0.5, 0.06, f"α = {alpha_val}", ha="center", va="bottom",
            fontsize=7.5, color=GREY, fontfamily="monospace",
            transform=ax.transAxes)


# ══════════════════════════════════════════════════════════════
# SPSS-EXACT STATISTICAL FORMULAS
# ══════════════════════════════════════════════════════════════

def normality_test(data, alpha=0.05):
    n = len(data)
    if n < 3:
        return {"test": "N/A", "stat_label": "—", "statistic": np.nan,
                "p_value": np.nan, "normal": True,
                "interpretation": "Too few observations", "n": n}
    if n <= 50:
        stat, p = stats.shapiro(data)
        test_name, stat_label = "Shapiro-Wilk", "W"
    else:
        stat, p = stats.kstest(data, "norm",
                               args=(np.mean(data), np.std(data, ddof=1)))
        test_name, stat_label = "Kolmogorov-Smirnov", "D"
    return {"test": test_name, "stat_label": stat_label,
            "statistic": float(stat), "p_value": float(p),
            "normal": p > alpha,
            "interpretation": "Normal distribution" if p > alpha else "Non-normal distribution",
            "n": n}

def levene_test(g1, g2, alpha=0.05):
    stat, p = stats.levene(g1, g2, center="mean")
    return {"test": "Levene's Test (based on Mean)",
            "stat_label": "F", "statistic": float(stat),
            "df1": 1, "df2": int(len(g1)+len(g2)-2),
            "p_value": float(p),
            "equal_variance": p > alpha,
            "interpretation": ("Equal variances assumed"
                                if p > alpha else "Equal variances not assumed")}

def _p(t, df, alternative):
    if alternative == "two-sided": return float(2 * stats.t.sf(abs(t), df))
    if alternative == "greater":   return float(stats.t.sf(t, df))
    return float(stats.t.cdf(t, df))

# ── One-Sample t-Test ──────────────────────────────────────
def one_sample_t(data, mu0, alpha=0.05, alternative="two-sided"):
    arr  = np.asarray(data, float)
    n    = len(arr)
    xbar = arr.mean()
    s    = arr.std(ddof=1)
    se   = s / np.sqrt(n)
    t    = (xbar - mu0) / se
    df   = n - 1
    pv   = _p(t, df, alternative)
    tc   = stats.t.ppf(1 - alpha/2, df)
    diff = xbar - mu0
    d    = diff / s
    return {"test": "One-Sample t-Test",
            "statistic": float(t), "df": float(df), "p_value": pv,
            "reject": pv < alpha,
            "n": n, "mean": float(xbar), "std": float(s), "se": float(se),
            "mu0": mu0, "mean_diff": float(diff),
            "ci": (float(diff - tc*se), float(diff + tc*se)),
            "mean_ci": (float(xbar - tc*se), float(xbar + tc*se)),
            "effect_size": float(d), "effect_label": "Cohen's d",
            "alternative": alternative}

# ── Wilcoxon Signed-Rank One-Sample ───────────────────────
def wilcoxon_one_sample(data, mu0, alpha=0.05, alternative="two-sided"):
    arr  = np.asarray(data, float)
    diff = arr - mu0
    diff = diff[diff != 0]
    n    = len(diff)
    if n == 0:
        raise ValueError("All differences are zero.")
    ranks  = stats.rankdata(np.abs(diff))
    T_plus = float(np.sum(ranks[diff > 0]))
    T_minus= float(np.sum(ranks[diff < 0]))
    W      = min(T_plus, T_minus)
    mu_T   = n * (n + 1) / 4
    uniq, cnts = np.unique(ranks, return_counts=True)
    tie_c  = np.sum(cnts**3 - cnts) / 48
    sig_T  = np.sqrt(max(n*(n+1)*(2*n+1)/24 - tie_c, 0))
    Z      = (W - mu_T + 0.5) / sig_T if sig_T > 0 else 0.0
    if alternative == "two-sided": pv = float(2 * stats.norm.sf(abs(Z)))
    elif alternative == "greater": pv = float(stats.norm.sf(Z))
    else:                          pv = float(stats.norm.cdf(Z))
    r = abs(Z) / np.sqrt(n)
    return {"test": "Wilcoxon Signed-Rank Test (One-Sample)",
            "statistic": float(W), "Z": float(Z), "p_value": pv,
            "reject": pv < alpha,
            "n": len(data), "n_effective": n,
            "T_plus": T_plus, "T_minus": T_minus,
            "mu0": mu0, "median": float(np.median(data)),
            "effect_size": float(r), "effect_label": "r (rank-biserial)",
            "alternative": alternative}

# ── Paired t-Test ──────────────────────────────────────────
def paired_t(g1, g2, alpha=0.05, alternative="two-sided"):
    a1, a2 = np.asarray(g1, float), np.asarray(g2, float)
    d      = a1 - a2
    n      = len(d)
    dbar   = d.mean()
    sd     = d.std(ddof=1)
    se     = sd / np.sqrt(n)
    t      = dbar / se
    df     = n - 1
    pv     = _p(t, df, alternative)
    tc     = stats.t.ppf(1 - alpha/2, df)
    dz     = dbar / sd
    rc, rp = stats.pearsonr(a1, a2)
    return {"test": "Paired-Samples t-Test",
            "statistic": float(t), "df": float(df), "p_value": pv,
            "reject": pv < alpha,
            "n": n, "mean_diff": float(dbar), "std_diff": float(sd), "se": float(se),
            "ci": (float(dbar - tc*se), float(dbar + tc*se)),
            "pearson_r": float(rc), "pearson_p": float(rp),
            "effect_size": float(dz), "effect_label": "Cohen's d_z",
            "alternative": alternative, "differences": d,
            "mean1": float(a1.mean()), "mean2": float(a2.mean()),
            "std1": float(a1.std(ddof=1)), "std2": float(a2.std(ddof=1)),
            "n1": n, "n2": n}

# ── Wilcoxon Signed-Rank Paired ────────────────────────────
def wilcoxon_paired(g1, g2, alpha=0.05, alternative="two-sided"):
    a1, a2 = np.asarray(g1, float), np.asarray(g2, float)
    d      = a1 - a2
    base   = wilcoxon_one_sample(d, 0.0, alpha, alternative)
    base.update({"test": "Wilcoxon Signed-Rank Test (Paired)",
                 "differences": d,
                 "median_diff": float(np.median(d)),
                 "mean1": float(a1.mean()), "mean2": float(a2.mean()),
                 "std1": float(a1.std(ddof=1)), "std2": float(a2.std(ddof=1)),
                 "n1": len(g1), "n2": len(g2)})
    return base

# ── Independent-Samples t-Test ─────────────────────────────
def independent_t(g1, g2, alpha=0.05, alternative="two-sided"):
    a1, a2 = np.asarray(g1, float), np.asarray(g2, float)
    n1, n2 = len(a1), len(a2)
    m1, m2 = a1.mean(), a2.mean()
    s1, s2 = a1.std(ddof=1), a2.std(ddof=1)
    diff   = m1 - m2
    # Student's
    sp2    = ((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2)
    sp     = np.sqrt(sp2)
    se_eq  = sp * np.sqrt(1/n1 + 1/n2)
    df_eq  = float(n1 + n2 - 2)
    t_eq   = diff / se_eq
    p_eq   = _p(t_eq, df_eq, alternative)
    tc_eq  = stats.t.ppf(1 - alpha/2, df_eq)
    ci_eq  = (diff - tc_eq*se_eq, diff + tc_eq*se_eq)
    # Welch's
    se_w   = np.sqrt(s1**2/n1 + s2**2/n2)
    num_df = (s1**2/n1 + s2**2/n2)**2
    den_df = (s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)
    df_w   = float(num_df / den_df) if den_df > 0 else df_eq
    t_w    = diff / se_w
    p_w    = _p(t_w, df_w, alternative)
    tc_w   = stats.t.ppf(1 - alpha/2, df_w)
    ci_w   = (diff - tc_w*se_w, diff + tc_w*se_w)
    # Levene
    lev    = levene_test(a1, a2, alpha)
    use_eq = lev["equal_variance"]
    # Cohen's d (pooled)
    d      = diff / sp
    chosen_t  = t_eq  if use_eq else t_w
    chosen_df = df_eq if use_eq else df_w
    chosen_p  = p_eq  if use_eq else p_w
    chosen_ci = ci_eq if use_eq else ci_w
    chosen_se = se_eq if use_eq else se_w
    test_name = ("Independent-Samples t-Test (Student's)"
                 if use_eq else "Independent-Samples t-Test (Welch's)")
    return {"test": test_name,
            "statistic": float(chosen_t), "df": float(chosen_df),
            "p_value": float(chosen_p), "reject": chosen_p < alpha,
            "se": float(chosen_se), "ci": (float(chosen_ci[0]), float(chosen_ci[1])),
            "n1": n1, "n2": n2,
            "mean1": float(m1), "mean2": float(m2),
            "std1": float(s1), "std2": float(s2),
            "mean_diff": float(diff),
            "effect_size": float(d), "effect_label": "Cohen's d",
            "alternative": alternative,
            "t_equal": float(t_eq), "df_equal": float(df_eq),
            "p_equal": float(p_eq),
            "ci_equal": (float(ci_eq[0]), float(ci_eq[1])),
            "t_welch": float(t_w), "df_welch": float(df_w),
            "p_welch": float(p_w),
            "ci_welch": (float(ci_w[0]), float(ci_w[1])),
            "levene": lev, "use_equal": use_eq}

# ── Mann-Whitney U ─────────────────────────────────────────
def mann_whitney_u(g1, g2, alpha=0.05, alternative="two-sided"):
    a1, a2 = np.asarray(g1, float), np.asarray(g2, float)
    n1, n2 = len(a1), len(a2)
    N      = n1 + n2
    combined = np.concatenate([a1, a2])
    ranks  = stats.rankdata(combined)
    R1     = float(ranks[:n1].sum())
    R2     = float(ranks[n1:].sum())
    U1     = n1*n2 + n1*(n1+1)/2 - R1
    U2     = n1*n2 - U1
    U      = min(U1, U2)
    mu_U   = n1*n2/2
    uniq, cnts = np.unique(ranks, return_counts=True)
    tie_c  = np.sum(cnts**3 - cnts) / (N*(N-1)) if N > 1 else 0
    sig_U  = np.sqrt(n1*n2/12 * ((N+1) - tie_c))
    Z      = (U - mu_U + 0.5) / sig_U if sig_U > 0 else 0.0
    if alternative == "two-sided": pv = float(2 * stats.norm.sf(abs(Z)))
    elif alternative == "greater": pv = float(stats.norm.sf(Z))
    else:                          pv = float(stats.norm.cdf(Z))
    r = abs(Z) / np.sqrt(N)
    return {"test": "Mann-Whitney U Test",
            "statistic": float(U), "U1": float(U1), "U2": float(U2),
            "Z": float(Z), "p_value": pv, "reject": pv < alpha,
            "n1": n1, "n2": n2,
            "mean1": float(a1.mean()), "mean2": float(a2.mean()),
            "median1": float(np.median(a1)), "median2": float(np.median(a2)),
            "std1": float(a1.std(ddof=1)), "std2": float(a2.std(ddof=1)),
            "R1": R1, "R2": R2,
            "effect_size": float(r), "effect_label": "r (rank-biserial)",
            "alternative": alternative}


# ══════════════════════════════════════════════════════════════
# EFFECT SIZE & DESCRIPTIVES
# ══════════════════════════════════════════════════════════════
def interpret_effect(val, label):
    v = abs(val)
    if "Cohen" in label:
        if v < 0.20: return "Negligible"
        if v < 0.50: return "Small"
        if v < 0.80: return "Medium"
        return "Large"
    else:
        if v < 0.10: return "Negligible"
        if v < 0.30: return "Small"
        if v < 0.50: return "Medium"
        return "Large"

def descriptives(arr):
    a = np.asarray(arr, float)
    n = len(a); m = a.mean(); s = a.std(ddof=1)
    return {"N": n, "Mean": round(float(m),4),
            "Std. Deviation": round(float(s),4),
            "Std. Error of Mean": round(float(s/np.sqrt(n)),4),
            "Median": round(float(np.median(a)),4),
            "Variance": round(float(s**2),4),
            "Minimum": round(float(a.min()),4),
            "Maximum": round(float(a.max()),4),
            "Range": round(float(a.max()-a.min()),4),
            "IQR": round(float(np.percentile(a,75)-np.percentile(a,25)),4),
            "25th Percentile": round(float(np.percentile(a,25)),4),
            "75th Percentile": round(float(np.percentile(a,75)),4),
            "Skewness": round(float(stats.skew(a)),4),
            "Kurtosis": round(float(stats.kurtosis(a,fisher=True)),4)}


# ══════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════
def plot_normality(data, col_name, norm_res):
    set_plot_style()
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.suptitle(f"Normality Diagnostics — {col_name}",
                 color=WHITE, fontsize=13, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)
    clr = GREEN if norm_res["normal"] else RED
    mu  = float(np.mean(data)); sig = float(np.std(data, ddof=1))
    x_r = np.linspace(mu - 4*sig, mu + 4*sig, 400)

    # 1 Histogram + KDE
    ax1 = fig.add_subplot(gs[0, :2]); ax1.set_facecolor(AXBG)
    ax1.hist(data, bins="auto", density=True, color="#1E3A5A",
             edgecolor="#2A5A8A", alpha=0.85, zorder=2, label="Sample")
    if len(data) >= 3:
        kde = stats.gaussian_kde(data, bw_method="scott")
        ax1.plot(x_r, kde(x_r), color=BLUE, lw=2.2, label="KDE")
    ax1.plot(x_r, stats.norm.pdf(x_r, mu, sig),
             color=GOLD, lw=2, ls="--", label="Normal fit")
    ax1.axvline(mu, color=WHITE, lw=1.2, ls=":", alpha=0.7, label=f"Mean={mu:.3f}")
    ax1.set_title("Histogram + KDE vs. Normal Fit", color=WHITE, pad=8)
    ax1.set_xlabel(col_name, color=GREY); ax1.set_ylabel("Density", color=GREY)
    ax1.legend(); ax1.grid(True, alpha=0.2)

    # 2 Normality result box
    ax2 = fig.add_subplot(gs[0, 2]); ax2.set_facecolor(AXBG); ax2.axis("off")
    border = FancyBboxPatch((0.05,0.05),0.90,0.90,boxstyle="round,pad=0.02",
                            lw=2, edgecolor=clr, facecolor="#111",
                            transform=ax2.transAxes, clip_on=False)
    ax2.add_patch(border)
    slbl = norm_res.get("stat_label","W")
    ax2.text(0.5,0.88,norm_res["test"],ha="center",va="top",fontsize=9,
             color=GREY,fontfamily="monospace",transform=ax2.transAxes)
    ax2.text(0.5,0.72,f"{slbl} = {norm_res['statistic']:.5f}",ha="center",
             va="top",fontsize=13,color=GOLD,fontweight="bold",
             fontfamily="monospace",transform=ax2.transAxes)
    ax2.text(0.5,0.55,f"p = {norm_res['p_value']:.5f}",ha="center",va="top",
             fontsize=12,color=WHITE,fontfamily="monospace",transform=ax2.transAxes)
    ax2.text(0.5,0.37,norm_res["interpretation"].upper(),ha="center",va="top",
             fontsize=10,color=clr,fontweight="bold",fontfamily="monospace",
             transform=ax2.transAxes)
    ax2.text(0.5,0.22,f"n = {norm_res['n']}",ha="center",va="top",fontsize=9,
             color=GREY,fontfamily="monospace",transform=ax2.transAxes)
    ax2.text(0.5,0.11,"→ Parametric" if norm_res["normal"] else "→ Non-Parametric",
             ha="center",va="top",fontsize=9,color=clr,fontfamily="monospace",
             transform=ax2.transAxes)

    # 3 Q-Q
    ax3 = fig.add_subplot(gs[1,0]); ax3.set_facecolor(AXBG)
    (osm,osr),(slope,intercept,_) = stats.probplot(data,dist="norm")
    ax3.scatter(osm,osr,color=BLUE,s=22,alpha=0.85,zorder=3,label="Observed")
    ax3.plot(osm,slope*np.array(osm)+intercept,color=GOLD,lw=2,label="Expected")
    ax3.set_title("Normal Q-Q Plot",color=WHITE,pad=8)
    ax3.set_xlabel("Theoretical Quantiles",color=GREY)
    ax3.set_ylabel("Sample Quantiles",color=GREY)
    ax3.legend(); ax3.grid(True,alpha=0.2)

    # 4 Box plot
    ax4 = fig.add_subplot(gs[1,1]); ax4.set_facecolor(AXBG)
    bp = ax4.boxplot(data,patch_artist=True,widths=0.45,
                     medianprops=dict(color=GOLD,lw=2.5),
                     boxprops=dict(facecolor="#1E3A5A",edgecolor=BLUE,lw=1.5),
                     whiskerprops=dict(color=BLUE,lw=1.5,ls="--"),
                     capprops=dict(color=BLUE,lw=2),
                     flierprops=dict(marker="o",color=RED,ms=5,alpha=0.8))
    ax4.scatter([1],[mu],color=RED,s=60,zorder=5,marker="D",label="Mean")
    ax4.set_title("Box-and-Whisker Plot",color=WHITE,pad=8)
    ax4.set_ylabel(col_name,color=GREY); ax4.set_xticks([])
    ax4.legend(); ax4.grid(True,alpha=0.2,axis="y")

    # 5 ECDF
    ax5 = fig.add_subplot(gs[1,2]); ax5.set_facecolor(AXBG)
    sd2 = np.sort(data); ey = np.arange(1,len(data)+1)/len(data)
    ax5.step(sd2,ey,color=BLUE,lw=2,label="ECDF")
    ax5.plot(x_r,stats.norm.cdf(x_r,mu,sig),color=GOLD,lw=2,ls="--",label="Normal CDF")
    ax5.set_title("ECDF vs. Normal CDF",color=WHITE,pad=8)
    ax5.set_xlabel(col_name,color=GREY); ax5.set_ylabel("Cumulative Prob.",color=GREY)
    ax5.legend(); ax5.grid(True,alpha=0.2)

    plt.tight_layout(rect=[0,0,1,0.96])
    return fig


def plot_one_sample(data, result, mu0, alpha_val):
    set_plot_style()
    fig = plt.figure(figsize=(16,10),facecolor=BG)
    fig.suptitle(f"Test Results — {result['test']}",
                 color=WHITE,fontsize=12,fontweight="bold",y=0.98)
    gs  = gridspec.GridSpec(2,3,figure=fig,hspace=0.48,wspace=0.38)
    arr = np.asarray(data,float); mu = arr.mean(); sig = arr.std(ddof=1)
    x_r = np.linspace(mu-4*sig,mu+4*sig,400)

    # 1 Distribution
    ax1 = fig.add_subplot(gs[0,:2]); ax1.set_facecolor(AXBG)
    if len(arr)>=3:
        kde = stats.gaussian_kde(arr); ax1.fill_between(x_r,kde(x_r),alpha=0.3,color=BLUE)
        ax1.plot(x_r,kde(x_r),color=BLUE,lw=2.2,label="Sample KDE")
    ax1.axvline(mu,color=GOLD,lw=2.5,label=f"X̄ = {mu:.4f}")
    ax1.axvline(mu0,color=RED,lw=2.5,ls="--",label=f"μ₀ = {mu0}")
    if "mean_ci" in result:
        ax1.axvspan(result["mean_ci"][0],result["mean_ci"][1],
                    alpha=0.15,color=GOLD,label=f"{int((1-alpha_val)*100)}% CI of Mean")
    ax1.set_title("Sample Distribution vs. Hypothesised Value",color=WHITE,pad=8)
    ax1.set_xlabel("Value",color=GREY); ax1.set_ylabel("Density",color=GREY)
    ax1.legend(); ax1.grid(True,alpha=0.2)

    # 2 Result panel
    ax2 = fig.add_subplot(gs[0,2]); result_panel(ax2,result,alpha_val)

    # 3 t / Rank distribution
    ax3 = fig.add_subplot(gs[1,0:2]); ax3.set_facecolor(AXBG)
    if "t-Test" in result["test"]:
        df_val = result["df"]
        xt = np.linspace(-5,5,600); yt = stats.t.pdf(xt,df_val)
        ax3.plot(xt,yt,color=WHITE,lw=2)
        tc = stats.t.ppf(1-alpha_val/2,df_val)
        ax3.fill_between(xt,yt,where=(xt>=tc),color=RED,alpha=0.4,label="Rejection region")
        ax3.fill_between(xt,yt,where=(xt<=-tc),color=RED,alpha=0.4)
        ax3.fill_between(xt,yt,where=((xt>=-tc)&(xt<=tc)),color=GREEN,alpha=0.12,label="Retention region")
        ts = result["statistic"]
        ax3.axvline(ts,color=GOLD,lw=2.5,label=f"t={ts:.4f}  p={result['p_value']:.4f}")
        ax3.set_xlim(-5,5)
        ax3.set_title(f"t-Distribution (df={df_val:.2f}) with Rejection Regions",color=WHITE,pad=8)
        ax3.set_xlabel("t",color=GREY); ax3.set_ylabel("Density",color=GREY)
    else:
        d2 = arr - mu0; d2nz = d2[d2!=0]
        rk = stats.rankdata(np.abs(d2nz))
        pr = rk[d2nz>0]; nr = rk[d2nz<0]
        if len(pr)>0: ax3.bar(range(1,len(pr)+1),np.sort(pr)[::-1],color=GREEN,alpha=0.75,label=f"Positive ranks  T+={result.get('T_plus',0):.1f}")
        if len(nr)>0: ax3.bar(range(1,len(nr)+1),-np.sort(nr)[::-1],color=RED,alpha=0.75,label=f"Negative ranks  T−={result.get('T_minus',0):.1f}")
        ax3.axhline(0,color=WHITE,lw=1)
        ax3.set_title("Wilcoxon: Positive vs. Negative Rank Sums",color=WHITE,pad=8)
        ax3.set_xlabel("Index",color=GREY); ax3.set_ylabel("Signed Rank",color=GREY)
    ax3.legend(); ax3.grid(True,alpha=0.2)

    # 4 Descriptive panel
    ax4 = fig.add_subplot(gs[1,2]); ax4.set_facecolor(AXBG); ax4.axis("off")
    ditems = [("N",str(len(arr))),("Mean",f"{mu:.4f}"),
              ("Std Dev",f"{arr.std(ddof=1):.4f}"),
              ("Std Error",f"{arr.std(ddof=1)/np.sqrt(len(arr)):.4f}"),
              ("Median",f"{np.median(arr):.4f}"),
              ("Min",f"{arr.min():.4f}"),("Max",f"{arr.max():.4f}"),
              ("Skewness",f"{stats.skew(arr):.4f}"),
              ("Kurtosis",f"{stats.kurtosis(arr,fisher=True):.4f}")]
    ax4.text(0.5,0.97,"DESCRIPTIVE STATISTICS",ha="center",va="top",
             fontsize=8,color=GOLD,fontfamily="monospace",fontweight="bold",
             transform=ax4.transAxes)
    y=0.87
    for lbl,val in ditems:
        ax4.text(0.05,y,lbl,ha="left",va="top",fontsize=8,color=GREY,
                 fontfamily="monospace",transform=ax4.transAxes)
        ax4.text(0.95,y,val,ha="right",va="top",fontsize=8.5,color=WHITE,
                 fontfamily="monospace",fontweight="bold",transform=ax4.transAxes)
        y-=0.10; draw_sep(ax4,y+0.04)
    plt.tight_layout(rect=[0,0,1,0.96]); return fig


def plot_paired(g1,g2,result,lbl1,lbl2,alpha_val):
    set_plot_style()
    fig = plt.figure(figsize=(16,10),facecolor=BG)
    fig.suptitle(f"Test Results — {result['test']}",
                 color=WHITE,fontsize=12,fontweight="bold",y=0.98)
    gs  = gridspec.GridSpec(2,3,figure=fig,hspace=0.48,wspace=0.38)
    a1,a2 = np.asarray(g1,float),np.asarray(g2,float)
    d = result["differences"]

    # 1 Before-after
    ax1 = fig.add_subplot(gs[0,:2]); ax1.set_facecolor(AXBG)
    for i in range(len(a1)):
        c = RED if a1[i]>a2[i] else GREEN
        ax1.plot([0,1],[a1[i],a2[i]],color=c,alpha=0.35,lw=1)
    ax1.scatter([0]*len(a1),a1,color=BLUE,s=40,zorder=4,alpha=0.9,label=lbl1)
    ax1.scatter([1]*len(a2),a2,color=GOLD,s=40,zorder=4,alpha=0.9,label=lbl2)
    ax1.plot([0,1],[a1.mean(),a2.mean()],color=WHITE,lw=3,zorder=6,marker="D",ms=8,label="Means")
    ax1.set_xticks([0,1]); ax1.set_xticklabels([lbl1,lbl2],color=WHITE,fontsize=10)
    ax1.set_title("Individual Observations (Before–After)",color=WHITE,pad=8)
    ax1.set_ylabel("Value",color=GREY); ax1.legend(); ax1.grid(True,alpha=0.2,axis="y")

    # 2 Result panel
    ax2 = fig.add_subplot(gs[0,2]); result_panel(ax2,result,alpha_val)

    # 3 Diff histogram
    ax3 = fig.add_subplot(gs[1,0]); ax3.set_facecolor(AXBG)
    ax3.hist(d,bins="auto",color="#1E3A5A",edgecolor=BLUE,alpha=0.85,density=True)
    if len(d)>=3:
        kdx=np.linspace(d.min()-d.std(),d.max()+d.std(),300)
        ax3.plot(kdx,stats.gaussian_kde(d)(kdx),color=BLUE,lw=2)
    ax3.axvline(0,color=RED,lw=2,ls="--",label="Zero (H₀)")
    ax3.axvline(d.mean(),color=GOLD,lw=2,label=f"Mean diff={d.mean():.4f}")
    if "ci" in result:
        ax3.axvspan(result["ci"][0],result["ci"][1],alpha=0.15,color=GOLD,
                    label=f"{int((1-alpha_val)*100)}% CI")
    ax3.set_title("Distribution of Differences",color=WHITE,pad=8)
    ax3.set_xlabel(f"{lbl1} − {lbl2}",color=GREY)
    ax3.set_ylabel("Density",color=GREY); ax3.legend(); ax3.grid(True,alpha=0.2)

    # 4 Box plots
    ax4 = fig.add_subplot(gs[1,1]); ax4.set_facecolor(AXBG)
    bp = ax4.boxplot([a1,a2],patch_artist=True,widths=0.4,
                     medianprops=dict(color=GOLD,lw=2.5),
                     boxprops=dict(facecolor="#1E3A5A",edgecolor=BLUE,lw=1.5),
                     whiskerprops=dict(color=BLUE,lw=1.5,ls="--"),
                     capprops=dict(color=BLUE,lw=2),
                     flierprops=dict(marker="o",color=RED,ms=5,alpha=0.8))
    bp["boxes"][0].set_facecolor("#1A3050"); bp["boxes"][1].set_facecolor("#2A3818")
    ax4.scatter([1,2],[a1.mean(),a2.mean()],color=RED,s=60,zorder=6,marker="D",label="Mean")
    ax4.set_xticks([1,2]); ax4.set_xticklabels([lbl1,lbl2],color=WHITE)
    ax4.set_title("Box Plots",color=WHITE,pad=8)
    ax4.legend(); ax4.grid(True,alpha=0.2,axis="y")

    # 5 Q-Q of differences
    ax5 = fig.add_subplot(gs[1,2]); ax5.set_facecolor(AXBG)
    (osm,osr),(sl,ic,_) = stats.probplot(d,dist="norm")
    ax5.scatter(osm,osr,color=BLUE,s=22,alpha=0.85)
    ax5.plot(osm,sl*np.array(osm)+ic,color=GOLD,lw=2)
    ax5.set_title("Q-Q Plot of Differences",color=WHITE,pad=8)
    ax5.set_xlabel("Theoretical Quantiles",color=GREY)
    ax5.set_ylabel("Sample Quantiles",color=GREY)
    ax5.grid(True,alpha=0.2)
    plt.tight_layout(rect=[0,0,1,0.96]); return fig


def plot_independent(g1,g2,result,lbl1,lbl2,alpha_val):
    set_plot_style()
    fig = plt.figure(figsize=(16,10),facecolor=BG)
    fig.suptitle(f"Test Results — {result['test']}",
                 color=WHITE,fontsize=12,fontweight="bold",y=0.98)
    gs  = gridspec.GridSpec(2,3,figure=fig,hspace=0.48,wspace=0.38)
    a1,a2 = np.asarray(g1,float),np.asarray(g2,float)
    all_d = np.concatenate([a1,a2])
    x_r   = np.linspace(all_d.min()-all_d.std(),all_d.max()+all_d.std(),400)

    # 1 KDE comparison
    ax1 = fig.add_subplot(gs[0,:2]); ax1.set_facecolor(AXBG)
    if len(a1)>=3:
        k1=stats.gaussian_kde(a1); ax1.fill_between(x_r,k1(x_r),alpha=0.35,color=BLUE)
        ax1.plot(x_r,k1(x_r),color=BLUE,lw=2.2,label=lbl1)
    if len(a2)>=3:
        k2=stats.gaussian_kde(a2); ax1.fill_between(x_r,k2(x_r),alpha=0.35,color=GOLD)
        ax1.plot(x_r,k2(x_r),color=GOLD,lw=2.2,label=lbl2)
    ax1.axvline(a1.mean(),color=BLUE,lw=1.8,ls="--",alpha=0.9,label=f"X̄₁={a1.mean():.3f}")
    ax1.axvline(a2.mean(),color=GOLD,lw=1.8,ls="--",alpha=0.9,label=f"X̄₂={a2.mean():.3f}")
    ax1.set_title("Group Distributions (KDE)",color=WHITE,pad=8)
    ax1.set_xlabel("Value",color=GREY); ax1.set_ylabel("Density",color=GREY)
    ax1.legend(); ax1.grid(True,alpha=0.2)

    # 2 Result panel
    ax2 = fig.add_subplot(gs[0,2]); result_panel(ax2,result,alpha_val)

    # 3 t-distribution
    ax3 = fig.add_subplot(gs[1,0]); ax3.set_facecolor(AXBG)
    if "t-Test" in result["test"]:
        df_v=result["df"]; xt=np.linspace(-5,5,600); yt=stats.t.pdf(xt,df_v)
        ax3.plot(xt,yt,color=WHITE,lw=2)
        tc=stats.t.ppf(1-alpha_val/2,df_v)
        ax3.fill_between(xt,yt,where=(xt>=tc),color=RED,alpha=0.4,label="Rejection")
        ax3.fill_between(xt,yt,where=(xt<=-tc),color=RED,alpha=0.4)
        ax3.fill_between(xt,yt,where=((xt>=-tc)&(xt<=tc)),color=GREEN,alpha=0.12,label="Retention")
        ts=result["statistic"]
        ax3.axvline(ts,color=GOLD,lw=2.5,label=f"t={ts:.4f}")
        ax3.set_xlim(-5,5)
        ax3.set_title(f"t-Distribution (df={df_v:.2f})",color=WHITE,pad=8)
        ax3.set_xlabel("t",color=GREY); ax3.set_ylabel("Density",color=GREY)
    else:
        ax3.bar([lbl1,lbl2],[np.median(a1),np.median(a2)],color=[BLUE,GOLD],alpha=0.8)
        ax3.set_title("Group Medians",color=WHITE,pad=8)
        ax3.set_ylabel("Median",color=GREY)
    ax3.legend(); ax3.grid(True,alpha=0.2)

    # 4 Box + Violin
    ax4 = fig.add_subplot(gs[1,1]); ax4.set_facecolor(AXBG)
    bp = ax4.boxplot([a1,a2],patch_artist=True,widths=0.3,positions=[0.85,2.15],
                     medianprops=dict(color=GOLD,lw=2.5),
                     boxprops=dict(facecolor="#1E3A5A",edgecolor=BLUE,lw=1.5),
                     whiskerprops=dict(color=BLUE,lw=1.5,ls="--"),
                     capprops=dict(color=BLUE,lw=2),
                     flierprops=dict(marker="o",color=RED,ms=5,alpha=0.7))
    bp["boxes"][0].set_facecolor("#1A3050"); bp["boxes"][1].set_facecolor("#2A3818")
    parts = ax4.violinplot([a1,a2],positions=[1.15,1.85],widths=0.3,
                           showmedians=False,showextrema=False)
    for i,pc in enumerate(parts["bodies"]):
        pc.set_facecolor([BLUE,GOLD][i]); pc.set_alpha(0.3)
    ax4.scatter([1,2],[a1.mean(),a2.mean()],color=RED,s=70,zorder=6,marker="D",label="Mean")
    ax4.set_xticks([1,2]); ax4.set_xticklabels([lbl1,lbl2],color=WHITE)
    ax4.set_title("Box + Violin Plots",color=WHITE,pad=8)
    ax4.legend(); ax4.grid(True,alpha=0.2,axis="y")

    # 5 Effect size panel
    ax5 = fig.add_subplot(gs[1,2]); ax5.set_facecolor(AXBG); ax5.axis("off")
    d_val=result["effect_size"]; interp=interpret_effect(d_val,result["effect_label"])
    clr_e={"Negligible":GREY,"Small":BLUE,"Medium":GOLD,"Large":RED}.get(interp,WHITE)
    ax5.text(0.5,0.90,"EFFECT SIZE",ha="center",va="top",fontsize=8,color=GREY,
             fontfamily="monospace",fontweight="bold",transform=ax5.transAxes)
    ax5.text(0.5,0.78,result["effect_label"],ha="center",va="top",fontsize=8.5,
             color=GREY,fontfamily="monospace",transform=ax5.transAxes)
    ax5.text(0.5,0.62,f"{d_val:.4f}",ha="center",va="top",fontsize=26,
             color=clr_e,fontfamily="monospace",fontweight="bold",transform=ax5.transAxes)
    ax5.text(0.5,0.42,interp.upper(),ha="center",va="top",fontsize=13,
             color=clr_e,fontfamily="monospace",fontweight="bold",transform=ax5.transAxes)
    levels=[0,0.2,0.5,0.8,1.5]; lc=[GREY,BLUE,GOLD,RED]
    for i in range(4):
        x0=levels[i]/1.5*0.80+0.10; x1=levels[i+1]/1.5*0.80+0.10
        r2=plt.Rectangle((x0,0.22),x1-x0,0.06,facecolor=lc[i],alpha=0.65,
                          transform=ax5.transAxes,clip_on=False)
        ax5.add_patch(r2)
    mx=min(abs(d_val)/1.5,1.0)*0.80+0.10
    ax5.plot([mx],[0.22+0.03],"v",color=WHITE,ms=9,transform=ax5.transAxes,clip_on=False)
    for i,lbl in enumerate(["Neg","Small","Med","Large"]):
        ax5.text(0.10+i*0.20,0.17,lbl,ha="center",va="top",fontsize=6,color=GREY,
                 fontfamily="monospace",transform=ax5.transAxes)
    plt.tight_layout(rect=[0,0,1,0.96]); return fig


# ══════════════════════════════════════════════════════════════
# HTML TABLE HELPER
# ══════════════════════════════════════════════════════════════
def spss_table(rows):
    hdr = rows[0]; body = rows[1:]
    ths = "".join(f"<th>{h}</th>" for h in hdr)
    trs = ""
    for row in body:
        tds = ""
        for i,cell in enumerate(row):
            cls = "num" if i > 0 else ""
            tds += f'<td class="{cls}">{cell}</td>'
        trs += f"<tr>{tds}</tr>"
    return f'<table class="spss-table"><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table>'


# ══════════════════════════════════════════════════════════════
# CSV EXPORT
# ══════════════════════════════════════════════════════════════
def export_csv(test_result, assump_dict, desc_dict, alpha, alternative):
    rows = []
    rows.append(["=== DESCRIPTIVE STATISTICS ===",""])
    if isinstance(desc_dict,dict):
        first = list(desc_dict.values())[0]
        if isinstance(first,dict):
            for g,d in desc_dict.items():
                rows.append([f"--- {g} ---",""])
                for k,v in d.items(): rows.append([k,v])
        else:
            for k,v in desc_dict.items(): rows.append([k,v])
    rows.append(["",""])
    rows.append(["=== ASSUMPTION TESTS ===",""])
    for aname,ares in assump_dict.items():
        rows.append([f"--- {aname} ---",""])
        for k,v in ares.items():
            if not callable(v): rows.append([k,v])
    rows.append(["",""])
    rows.append(["=== HYPOTHESIS TEST RESULTS ===",""])
    rows.append(["Test",test_result["test"]])
    rows.append(["Alpha",alpha]); rows.append(["Alternative",alternative])
    for k,v in test_result.items():
        if k in ("differences","levene"): continue
        if k=="ci": rows.append(["CI Lower",round(v[0],6)]); rows.append(["CI Upper",round(v[1],6)])
        elif k in ("ci_equal","ci_welch"):
            rows.append([f"{k} Lower",round(v[0],6)]); rows.append([f"{k} Upper",round(v[1],6)])
        elif isinstance(v,(int,float,bool,str)): rows.append([k,v])
    rows.append(["Effect Size Magnitude",
                 interpret_effect(test_result["effect_size"],test_result.get("effect_label",""))])
    if "levene" in test_result:
        rows.append(["",""]); rows.append(["=== LEVENE TEST ===",""])
        for k,v in test_result["levene"].items(): rows.append([k,v])
    dfout = pd.DataFrame(rows,columns=["Parameter","Value"])
    buf = io.StringIO(); dfout.to_csv(buf,index=False); return buf.getvalue()


# ══════════════════════════════════════════════════════════════
# PDF REPORT
# ══════════════════════════════════════════════════════════════
def build_pdf(test_type,assump_dict,test_result,desc_dict,
              norm_bytes_list,fig_test_bytes,alpha,alternative,col_names=None):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf,pagesize=A4,
                            leftMargin=2*cm,rightMargin=2*cm,
                            topMargin=2.2*cm,bottomMargin=2*cm)
    CG=colors.HexColor("#E8B84B"); CW=colors.HexColor("#F0EDE8")
    CX=colors.HexColor("#8A8070"); CR=colors.HexColor("#E85454")
    CN=colors.HexColor("#4EBF85"); CD=colors.HexColor("#1A1A1A")
    CL=colors.HexColor("#2A2A2A"); CB=colors.HexColor("#111111")

    def ts():
        return TableStyle([
            ("BACKGROUND",(0,0),(-1,0),CD),("TEXTCOLOR",(0,0),(-1,0),CG),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,0),8),
            ("LINEBELOW",(0,0),(-1,0),1.5,CG),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[CD,CB]),
            ("TEXTCOLOR",(0,1),(-1,-1),CW),("FONTNAME",(0,1),(-1,-1),"Helvetica"),
            ("FONTSIZE",(0,1),(-1,-1),8.5),("FONTNAME",(1,1),(-1,-1),"Courier"),
            ("GRID",(0,0),(-1,-1),0.4,CL),("ALIGN",(1,0),(-1,-1),"RIGHT"),
            ("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING",(0,0),(0,-1),6),
        ])

    T_s  = ParagraphStyle("T",fontName="Helvetica-Bold",fontSize=20,textColor=CW,alignment=TA_CENTER,spaceAfter=4)
    S_s  = ParagraphStyle("S",fontName="Helvetica-Oblique",fontSize=10,textColor=CX,alignment=TA_CENTER,spaceAfter=14)
    H2_s = ParagraphStyle("H2",fontName="Helvetica-Bold",fontSize=12,textColor=CG,spaceBefore=16,spaceAfter=6)
    H3_s = ParagraphStyle("H3",fontName="Helvetica-Bold",fontSize=10,textColor=CW,spaceBefore=10,spaceAfter=4)
    B_s  = ParagraphStyle("B",fontName="Helvetica",fontSize=9.5,textColor=CW,spaceAfter=6,leading=15,alignment=TA_JUSTIFY)
    F_s  = ParagraphStyle("F",fontName="Helvetica-Oblique",fontSize=7.5,textColor=CX,alignment=TA_CENTER)
    RF_s = ParagraphStyle("RF",fontName="Helvetica",fontSize=8.5,textColor=CX,spaceAfter=5,leading=13,leftIndent=10,alignment=TA_JUSTIFY)

    story=[]
    story.append(Spacer(1,0.6*cm))
    story.append(Paragraph("Statistical Hypothesis Testing Report",T_s))
    story.append(Paragraph("SPSS-Equivalent Output | Automated Parametric &amp; Non-Parametric Analysis",S_s))
    meta=[["Test Type",test_type],["Significance Level (α)",str(alpha)],
          ["Alternative Hypothesis",alternative],
          ["Variables / Groups",", ".join(col_names) if col_names else "—"]]
    mt=Table(meta,colWidths=[7*cm,9*cm]); mt.setStyle(ts()); story.append(mt)
    story.append(HRFlowable(width="100%",thickness=1.5,color=CG,spaceAfter=10))

    # 1. Descriptives
    story.append(Paragraph("1. Descriptive Statistics",H2_s))
    if desc_dict:
        first=list(desc_dict.values())[0]
        if isinstance(first,dict):
            groups=list(desc_dict.keys()); sk=list(list(desc_dict.values())[0].keys())
            rows=[["Statistic"]+groups]+[[s]+[str(desc_dict[g].get(s,"—")) for g in groups] for s in sk]
            widths=[5*cm]+[5.5*cm]*len(groups)
        else:
            rows=[["Statistic","Value"]]+[[k,str(v)] for k,v in desc_dict.items()]
            widths=[8*cm,8*cm]
        dt=Table(rows,colWidths=widths); dt.setStyle(ts()); story.append(dt)
    story.append(Spacer(1,0.3*cm))

    # 2. Assumptions
    story.append(Paragraph("2. Assumption Tests",H2_s))
    for i,(aname,ares) in enumerate(assump_dict.items()):
        story.append(Paragraph(f"2.{i+1}  {aname}",H3_s))
        arows=[["Parameter","Value"]]
        for k,v in ares.items():
            if k in ("test","stat_label","statistic","p_value","interpretation","n",
                     "normal","equal_variance","df1","df2"):
                if k=="normal": v2="Yes — parametric applicable" if v else "No — non-parametric used"
                elif k=="equal_variance": v2="Yes — Student's t" if v else "No — Welch's t"
                elif isinstance(v,float): v2=f"{v:.6f}"
                else: v2=str(v)
                arows.append([k.replace("_"," ").title(),v2])
        at=Table(arows,colWidths=[8*cm,8*cm]); at.setStyle(ts()); story.append(at)
        story.append(Spacer(1,0.2*cm))

    # 3. Normality plots
    if norm_bytes_list:
        story.append(PageBreak())
        story.append(Paragraph("3. Normality Diagnostic Plots",H2_s))
        for i,nb in enumerate(norm_bytes_list):
            story.append(Paragraph(f"Figure 3.{i+1} — Normality Diagnostics",H3_s))
            story.append(RLImage(io.BytesIO(nb),width=16.5*cm,height=10.3*cm))
            story.append(Spacer(1,0.4*cm))

    # 4. Hypothesis test output
    story.append(PageBreak())
    story.append(Paragraph("4. Hypothesis Test Output (SPSS-Style)",H2_s))
    r=test_result
    vc=CR if r["reject"] else CN; vt="REJECT H₀" if r["reject"] else "FAIL TO REJECT H₀"
    vd_s=ParagraphStyle("VD",fontName="Helvetica-Bold",fontSize=14,textColor=vc,
                         alignment=TA_CENTER,spaceBefore=8,spaceAfter=8)
    story.append(Paragraph(f"Decision: {vt}",vd_s))

    # Levene + both t rows for independent
    if "t_equal" in r:
        story.append(Paragraph("4a. Levene's Test for Equality of Variances",H3_s))
        lv=r["levene"]
        lr=[["","F","df1","df2","Sig."],
            ["Levene's Test",f"{lv['statistic']:.4f}",str(lv['df1']),
             str(lv['df2']),f"{lv['p_value']:.4f}"]]
        lt=Table(lr,colWidths=[5*cm,3*cm,2*cm,2*cm,4*cm]); lt.setStyle(ts()); story.append(lt)
        story.append(Spacer(1,0.2*cm))
        story.append(Paragraph("4b. t-Test for Equality of Means",H3_s))
        ci_p=int((1-alpha)*100)
        br=[["","t","df","Sig.(2-tailed)",f"{ci_p}% CI Lower",f"{ci_p}% CI Upper","Selected"],
            ["Equal variances assumed",f"{r['t_equal']:.4f}",f"{r['df_equal']:.2f}",
             f"{r['p_equal']:.4f}",f"{r['ci_equal'][0]:.4f}",f"{r['ci_equal'][1]:.4f}",
             "✓" if r["use_equal"] else ""],
            ["Equal variances NOT assumed (Welch's)",f"{r['t_welch']:.4f}",
             f"{r['df_welch']:.4f}",f"{r['p_welch']:.4f}",
             f"{r['ci_welch'][0]:.4f}",f"{r['ci_welch'][1]:.4f}",
             "" if r["use_equal"] else "✓"]]
        bt=Table(br,colWidths=[5.5*cm,1.8*cm,1.8*cm,2.2*cm,1.8*cm,1.8*cm,1.6*cm])
        bt.setStyle(ts())
        cr=1 if r["use_equal"] else 2
        bt.setStyle(TableStyle([("BACKGROUND",(0,cr),(-1,cr),colors.HexColor("#1A2A10")),
                                ("TEXTCOLOR",(0,cr),(-1,cr),CN)]))
        story.append(bt)
    else:
        # General results table
        rrows=[["Parameter","Value"]]
        dkeys={"statistic":"Test Statistic","df":"df","p_value":"Sig.",
               "mean_diff":"Mean Difference","std_diff":"Std. Error of Difference",
               "n":"N","n_effective":"N (effective)",
               "T_plus":"T+ (Sum Positive Ranks)","T_minus":"T− (Sum Negative Ranks)",
               "Z":"Z (asymptotic)","U1":"U₁","U2":"U₂",
               "pearson_r":"Pearson r","pearson_p":"r Sig.",
               "mean1":"Mean (Group 1)","mean2":"Mean (Group 2)",
               "median1":"Median (Group 1)","median2":"Median (Group 2)"}
        for k,lbl in dkeys.items():
            if k in r and r[k] is not None and not isinstance(r[k],bool):
                v=r[k]; rrows.append([lbl,f"{v:.6f}" if isinstance(v,float) else str(v)])
        if "ci" in r:
            rrows.append([f"{int((1-alpha)*100)}% CI Lower",f"{r['ci'][0]:.6f}"])
            rrows.append([f"{int((1-alpha)*100)}% CI Upper",f"{r['ci'][1]:.6f}"])
        rrows.append(["Effect Size Magnitude",
                      interpret_effect(r["effect_size"],r.get("effect_label",""))])
        rt=Table(rrows,colWidths=[9.5*cm,6.5*cm]); rt.setStyle(ts()); story.append(rt)

    # 5. Test figure
    if fig_test_bytes:
        story.append(PageBreak())
        story.append(Paragraph("5. Test Result Visualisation",H2_s))
        story.append(RLImage(io.BytesIO(fig_test_bytes),width=16.5*cm,height=10.3*cm))
        story.append(Spacer(1,0.4*cm))

    # 6. Interpretation
    story.append(Paragraph("6. Statistical Interpretation",H2_s))
    ei=interpret_effect(r["effect_size"],r.get("effect_label",""))
    pstr=f"p = {r['p_value']:.4f}" if r['p_value']>=0.0001 else "p &lt; 0.0001"
    sl2="t" if "t-Test" in r["test"] else ("W" if "Wilcoxon" in r["test"] else "U")
    df_str=f", df = {r['df']:.2f}" if "df" in r else ""
    ci_str=""
    if "ci" in r:
        ci_str=(f" The {int((1-alpha)*100)}% confidence interval for the mean difference "
                f"is [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}].")
    story.append(Paragraph(
        f"The <b>{r['test']}</b> yielded a test statistic of "
        f"<i>{sl2}</i> = {r['statistic']:.4f}{df_str}, {pstr}. "
        f"At α = {alpha}, the null hypothesis is "
        f"<b>{'rejected' if r['reject'] else 'not rejected'}</b>. "
        f"The effect size ({r.get('effect_label','')} = {r['effect_size']:.4f}) "
        f"is <b>{ei.lower()}</b> (Cohen, 1988; Field, 2009).{ci_str}", B_s))

    # 7. References
    story.append(Paragraph("7. References",H2_s))
    for ref in [
        "Cohen, J. (1988). <i>Statistical Power Analysis for the Behavioral Sciences</i> (2nd ed.). Lawrence Erlbaum.",
        "Field, A. (2018). <i>Discovering Statistics Using IBM SPSS Statistics</i> (5th ed.). SAGE.",
        "IBM Corp. (2023). <i>IBM SPSS Statistics for Windows, Version 29.0</i>. IBM Corp.",
        "Lakens, D. (2013). Calculating and reporting effect sizes. <i>Frontiers in Psychology, 4</i>, 863.",
        "Levene, H. (1960). Robust tests for equality of variances. <i>Contributions to Probability and Statistics</i> (pp. 278–292). Stanford.",
        "Satterthwaite, F. E. (1946). An approximate distribution of variance estimates. <i>Biometrics Bulletin, 2</i>(6), 110–114.",
        "Shapiro, S. S., &amp; Wilk, M. B. (1965). An analysis of variance test for normality. <i>Biometrika, 52</i>, 591–611.",
        "Welch, B. L. (1947). The generalization of Student's problem. <i>Biometrika, 34</i>, 28–35.",
        "Wilcoxon, F. (1945). Individual comparisons by ranking methods. <i>Biometrics Bulletin, 1</i>(6), 80–83.",
    ]:
        story.append(Paragraph(f"• {ref}",RF_s))

    story.append(Spacer(1,0.5*cm))
    story.append(HRFlowable(width="100%",thickness=0.5,color=CL))
    story.append(Spacer(1,0.2*cm))
    story.append(Paragraph("Generated by StatTest Suite — SPSS-Equivalent Automated Testing",F_s))
    doc.build(story); buf.seek(0); return buf.read()


# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════
def main():
    st.markdown("""
<div class="app-header">
  <div class="app-title">StatTest Suite</div>
  <div class="app-subtitle">Automated Parametric &amp; Non-Parametric Hypothesis Testing</div>
  <span class="app-badge">SPSS-Equivalent Formulas</span>
  <span class="app-badge">Auto Test Selection</span>
  <span class="app-badge">Full Output</span>
</div>""", unsafe_allow_html=True)

    # ── SIDEBAR ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sec-head">Test Configuration</div>', unsafe_allow_html=True)
        test_type   = st.selectbox("Test Design",
                                   ["One-Sample","Paired (Dependent) Samples","Independent Samples"])
        alpha       = st.selectbox("Significance Level (α)",[0.05,0.01,0.10],index=0)
        alternative = st.selectbox("Alternative Hypothesis (H₁)",["two-sided","greater","less"],
                                   format_func=lambda x:{"two-sided":"Two-tailed (≠)",
                                                          "greater":"One-tailed (>)",
                                                          "less":"One-tailed (<)"}[x])
        st.markdown('<div class="sec-head">Decision Logic</div>', unsafe_allow_html=True)
        st.markdown("""
<div style="font-size:.78rem;color:#A09080;line-height:1.9">
<b style="color:#E8B84B">Normality:</b><br>
&nbsp;n ≤ 50 → Shapiro-Wilk<br>
&nbsp;n > 50 → Kolmogorov-Smirnov<br><br>
<b style="color:#E8B84B">If normal:</b><br>
&nbsp;→ One-sample t-test<br>
&nbsp;→ Paired-samples t-test<br>
&nbsp;→ Independent t-test<br>
&nbsp;&nbsp;&nbsp;(Levene → Student's/Welch's)<br><br>
<b style="color:#E8B84B">If non-normal:</b><br>
&nbsp;→ Wilcoxon (one-sample)<br>
&nbsp;→ Wilcoxon (paired)<br>
&nbsp;→ Mann-Whitney U
</div>""", unsafe_allow_html=True)
        st.markdown('<div class="sec-head">CSV Format Guide</div>', unsafe_allow_html=True)
        with st.expander("View format examples"):
            st.markdown("""
**One-Sample:** one numeric column
```
value
23.4
25.1
22.8
```
**Paired:** two numeric columns
```
pre,post
23.4,21.1
25.1,22.5
```
**Independent:** value + group columns
```
value,group
23.4,Control
25.1,Treatment
```""")

    # ── TABS ─────────────────────────────────────────────────
    t1,t2,t3,t4 = st.tabs([
        "📂 Data Input","🔬 Assumption Tests","📊 Test Results","📚 Statistical Guide"])

    # ── TAB 1: DATA INPUT ────────────────────────────────────
    with t1:
        st.markdown('<div class="sec-head">Upload CSV File</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        col_up, col_cfg = st.columns([3,2])

        with col_up:
            st.markdown('<div class="sec-head">Demo Datasets</div>', unsafe_allow_html=True)
            demo = st.selectbox("Select demo",
                ["One-Sample (normal)","One-Sample (non-normal)",
                 "Paired (normal)","Paired (non-normal)",
                 "Independent (normal, equal var)","Independent (normal, unequal var)",
                 "Independent (non-normal)"])
            if st.button("▶  Load Demo & Run"):
                np.random.seed(42)
                dmu0 = 0.0
                if demo=="One-Sample (normal)":
                    ddf=pd.DataFrame({"value":np.random.normal(52,8,30)}); dmu0=50.0
                elif demo=="One-Sample (non-normal)":
                    ddf=pd.DataFrame({"value":np.random.exponential(5,25)}); dmu0=4.0
                elif demo=="Paired (normal)":
                    p=np.random.normal(70,10,25)
                    ddf=pd.DataFrame({"pre":p,"post":p-np.random.normal(5,3,25)})
                elif demo=="Paired (non-normal)":
                    p=np.random.exponential(10,20)
                    ddf=pd.DataFrame({"pre":p,"post":p*np.random.uniform(0.7,0.95,20)})
                elif demo=="Independent (normal, equal var)":
                    ddf=pd.DataFrame({"value":np.concatenate([np.random.normal(65,8,25),np.random.normal(72,8,25)]),"group":["A"]*25+["B"]*25})
                elif demo=="Independent (normal, unequal var)":
                    ddf=pd.DataFrame({"value":np.concatenate([np.random.normal(65,5,20),np.random.normal(72,15,20)]),"group":["A"]*20+["B"]*20})
                else:
                    ddf=pd.DataFrame({"value":np.concatenate([np.random.exponential(5,20),np.random.exponential(8,20)]),"group":["A"]*20+["B"]*20})
                ttype_map={"One-Sample (normal)":"One-Sample","One-Sample (non-normal)":"One-Sample",
                           "Paired (normal)":"Paired (Dependent) Samples","Paired (non-normal)":"Paired (Dependent) Samples",
                           "Independent (normal, equal var)":"Independent Samples",
                           "Independent (normal, unequal var)":"Independent Samples","Independent (non-normal)":"Independent Samples"}
                st.session_state.update({"df_input":ddf,"run_auto":True,
                                         "test_type_sel":ttype_map[demo],
                                         "alpha_sel":alpha,"alt_sel":alternative,
                                         "demo_mu0":dmu0,"cfg":{}})
                st.rerun()

        if uploaded:
            df_input=pd.read_csv(uploaded)
            st.session_state["df_input"]=df_input

        if "df_input" in st.session_state:
            df_input=st.session_state["df_input"]
            with col_up:
                st.markdown('<div class="sec-head">Data Preview</div>', unsafe_allow_html=True)
                st.dataframe(df_input.head(30),use_container_width=True)
                st.markdown(f'<div class="info-box">📌 <b>{len(df_input)}</b> rows × <b>{len(df_input.columns)}</b> columns loaded.</div>',
                            unsafe_allow_html=True)

            with col_cfg:
                st.markdown('<div class="sec-head">Column Mapping</div>', unsafe_allow_html=True)
                num_cols=df_input.select_dtypes(include=[np.number]).columns.tolist()
                all_cols=df_input.columns.tolist()
                ttype=st.session_state.get("test_type_sel",test_type)
                alph =st.session_state.get("alpha_sel",alpha)
                alt  =st.session_state.get("alt_sel",alternative)
                cfg={}
                if "One-Sample" in ttype:
                    cfg["value_col"]=st.selectbox("Value Column",num_cols,key="vc")
                    cfg["mu0"]=st.number_input("Hypothesised Mean (μ₀)",
                                               value=st.session_state.get("demo_mu0",0.0),
                                               step=0.01,format="%.4f")
                elif "Paired" in ttype:
                    cfg["pre_col"] =st.selectbox("Pre / Group 1 Column",num_cols,key="pc1")
                    rem=[c for c in num_cols if c!=cfg.get("pre_col")]
                    cfg["post_col"]=st.selectbox("Post / Group 2 Column",
                                                  rem if rem else num_cols,key="pc2")
                else:
                    cfg["value_col"]=st.selectbox("Value Column",num_cols,key="vc2")
                    cfg["group_col"]=st.selectbox("Group Column",all_cols,key="gc")
                if st.button("▶  Run Analysis",key="run_btn"):
                    st.session_state.update({"cfg":cfg,"test_type_sel":ttype,
                                             "alpha_sel":alph,"alt_sel":alt,
                                             "run_auto":True})
                    st.rerun()
        else:
            st.markdown('<div class="info-box">⬆ Upload a CSV file or load a demo dataset above.</div>',
                        unsafe_allow_html=True)

    # ── ANALYSIS ENGINE ──────────────────────────────────────
    if st.session_state.get("run_auto") and "df_input" in st.session_state:
        df   = st.session_state["df_input"]
        cfg  = st.session_state.get("cfg",{})
        ttype= st.session_state.get("test_type_sel","One-Sample")
        alph = st.session_state.get("alpha_sel",0.05)
        alt  = st.session_state.get("alt_sel","two-sided")

        # Auto-build cfg for demos
        if not cfg:
            nc=df.select_dtypes(include=[np.number]).columns
            if "One-Sample" in ttype:
                cfg={"value_col":nc[0],"mu0":st.session_state.get("demo_mu0",0.0)}
            elif "Paired" in ttype:
                cfg={"pre_col":nc[0],"post_col":nc[1] if len(nc)>1 else nc[0]}
            else:
                cfg={"value_col":nc[0],"group_col":df.columns[-1]}
            st.session_state["cfg"]=cfg

        try:
            assump_dict={}; norm_figs=[]; norm_bytes_list=[]
            test_res=None; test_fig=None; desc_out={}; col_names=[]

            if "One-Sample" in ttype:
                data = df[cfg["value_col"]].dropna().values
                mu0  = float(cfg.get("mu0",0.0))
                col_names=[cfg["value_col"],f"μ₀={mu0}"]
                desc_out=descriptives(data)
                norm=normality_test(data,alph); assump_dict["Normality Test"]=norm
                nf=plot_normality(data,cfg["value_col"],norm)
                norm_figs.append(nf); norm_bytes_list.append(fig_bytes(nf))
                test_res=(one_sample_t(data,mu0,alph,alt)
                          if norm["normal"] else wilcoxon_one_sample(data,mu0,alph,alt))
                test_fig=plot_one_sample(data,test_res,mu0,alph)

            elif "Paired" in ttype:
                g1=df[cfg["pre_col"]].dropna().values
                g2=df[cfg["post_col"]].dropna().values
                n_min=min(len(g1),len(g2)); g1,g2=g1[:n_min],g2[:n_min]
                lbl1,lbl2=cfg["pre_col"],cfg["post_col"]
                col_names=[lbl1,lbl2]
                d=g1-g2
                desc_out={lbl1:descriptives(g1),lbl2:descriptives(g2),"Differences":descriptives(d)}
                norm=normality_test(d,alph); assump_dict["Normality of Differences"]=norm
                nf=plot_normality(d,f"{lbl1} − {lbl2}",norm)
                norm_figs.append(nf); norm_bytes_list.append(fig_bytes(nf))
                test_res=(paired_t(g1,g2,alph,alt)
                          if norm["normal"] else wilcoxon_paired(g1,g2,alph,alt))
                test_fig=plot_paired(g1,g2,test_res,lbl1,lbl2,alph)

            else:
                vc=cfg["value_col"]; gc=cfg["group_col"]
                grps=df[gc].dropna().unique()
                if len(grps)<2: st.error("Need at least 2 groups."); st.stop()
                if len(grps)>2: st.warning(f"Using first 2 groups: {grps[0]}, {grps[1]}")
                g1=df[df[gc]==grps[0]][vc].dropna().values
                g2=df[df[gc]==grps[1]][vc].dropna().values
                lbl1,lbl2=str(grps[0]),str(grps[1]); col_names=[lbl1,lbl2]
                desc_out={lbl1:descriptives(g1),lbl2:descriptives(g2)}
                nm1=normality_test(g1,alph); nm2=normality_test(g2,alph)
                assump_dict[f"Normality — {lbl1}"]=nm1
                assump_dict[f"Normality — {lbl2}"]=nm2
                nf1=plot_normality(g1,f"{vc} [{lbl1}]",nm1)
                nf2=plot_normality(g2,f"{vc} [{lbl2}]",nm2)
                norm_figs+=[nf1,nf2]; norm_bytes_list+=[fig_bytes(nf1),fig_bytes(nf2)]
                if nm1["normal"] and nm2["normal"]:
                    lev=levene_test(g1,g2,alph)
                    assump_dict["Levene's Test (Equality of Variances)"]=lev
                    test_res=independent_t(g1,g2,alph,alt)
                else:
                    test_res=mann_whitney_u(g1,g2,alph,alt)
                test_fig=plot_independent(g1,g2,test_res,lbl1,lbl2,alph)

            st.session_state.update({
                "assump_dict":assump_dict,"test_res":test_res,
                "test_fig":test_fig,"test_fig_bytes":fig_bytes(test_fig),
                "norm_figs":norm_figs,"norm_bytes":norm_bytes_list,
                "desc_out":desc_out,"col_names":col_names,"run_auto":False})

        except Exception as ex:
            import traceback
            st.error(f"Analysis error: {ex}")
            st.code(traceback.format_exc())
            st.session_state["run_auto"]=False

    # ── TAB 2: ASSUMPTIONS ───────────────────────────────────
    with t2:
        if "assump_dict" not in st.session_state:
            st.markdown('<div class="info-box">Run an analysis from the <b>Data Input</b> tab first.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="sec-head">Prerequisite Assumption Checks</div>',
                        unsafe_allow_html=True)
            for aname,ares in st.session_state["assump_dict"].items():
                is_ok=ares.get("normal",ares.get("equal_variance",True))
                badge=('<span class="badge-pass">✓ PASSED</span>' if is_ok
                       else '<span class="badge-fail-b">✗ FAILED</span>')
                with st.expander(f"{aname}  —  {badge}", expanded=True):
                    c1,c2,c3,c4=st.columns(4)
                    slbl=ares.get("stat_label","Stat")
                    with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Test</div><div class="metric-val" style="font-size:.95rem">{ares["test"]}</div></div>',unsafe_allow_html=True)
                    with c2: st.markdown(f'<div class="metric-card"><div class="metric-label">{slbl} Statistic</div><div class="metric-val">{ares["statistic"]:.5f}</div></div>',unsafe_allow_html=True)
                    with c3: st.markdown(f'<div class="metric-card"><div class="metric-label">p-value</div><div class="metric-val">{ares["p_value"]:.5f}</div></div>',unsafe_allow_html=True)
                    with c4:
                        clsc="green" if is_ok else "red"
                        st.markdown(f'<div class="metric-card {clsc}"><div class="metric-label">Result</div><div class="metric-val" style="font-size:.9rem">{ares["interpretation"]}</div></div>',unsafe_allow_html=True)
                    msg=("✅ <b>Parametric test applicable.</b> Normality assumption satisfied." if is_ok
                         else "⚠️ <b>Non-parametric equivalent applied.</b> Normality assumption violated.")
                    box="info-box" if is_ok else "warn-box"
                    st.markdown(f'<div class="{box}">{msg}</div>',unsafe_allow_html=True)

            st.markdown('<div class="sec-head">Normality Diagnostic Plots</div>', unsafe_allow_html=True)
            for nf in st.session_state["norm_figs"]:
                st.pyplot(nf,use_container_width=True); plt.close(nf)

    # ── TAB 3: RESULTS ───────────────────────────────────────
    with t3:
        if "test_res" not in st.session_state:
            st.markdown('<div class="info-box">Run an analysis from the <b>Data Input</b> tab first.</div>',
                        unsafe_allow_html=True)
        else:
            r    =st.session_state["test_res"]
            alph =st.session_state.get("alpha_sel",0.05)

            # Verdict
            if r["reject"]:
                st.markdown(f'<div class="verdict-banner verdict-reject"><div class="verdict-label">Decision at α = {alph}</div><div class="verdict-text-reject">✗ REJECT H₀</div><div class="verdict-sub">Sufficient statistical evidence to reject the null hypothesis.</div></div>',unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-banner verdict-fail"><div class="verdict-label">Decision at α = {alph}</div><div class="verdict-text-fail">✓ FAIL TO REJECT H₀</div><div class="verdict-sub">Insufficient statistical evidence to reject the null hypothesis.</div></div>',unsafe_allow_html=True)

            # Key metrics
            sl=("t" if "t-Test" in r["test"] else "W" if "Wilcoxon" in r["test"] else "U")
            pd_=f"{r['p_value']:.4f}" if r['p_value']>=0.0001 else "< 0.0001"
            ei=interpret_effect(r["effect_size"],r.get("effect_label",""))
            c1,c2,c3,c4=st.columns(4)
            with c1: st.markdown(f'<div class="metric-card"><div class="metric-label">Test Statistic ({sl})</div><div class="metric-val">{r["statistic"]:.4f}</div></div>',unsafe_allow_html=True)
            with c2:
                cc="red" if r["reject"] else "green"
                st.markdown(f'<div class="metric-card {cc}"><div class="metric-label">p-value</div><div class="metric-val">{pd_}</div></div>',unsafe_allow_html=True)
            with c3:
                if "df" in r: st.markdown(f'<div class="metric-card blue"><div class="metric-label">df</div><div class="metric-val">{r["df"]:.4f}</div></div>',unsafe_allow_html=True)
                else:
                    ne=r.get("n_effective",r.get("n","—"))
                    st.markdown(f'<div class="metric-card blue"><div class="metric-label">Effective N</div><div class="metric-val">{ne}</div></div>',unsafe_allow_html=True)
            with c4: st.markdown(f'<div class="metric-card"><div class="metric-label">{r.get("effect_label","Effect")}</div><div class="metric-val">{r["effect_size"]:.4f}</div><div class="metric-sub">{ei}</div></div>',unsafe_allow_html=True)

            if "ci" in r:
                st.markdown(f'<div class="info-box">📐 {int((1-alph)*100)}% Confidence Interval of the Difference: <b>[{r["ci"][0]:.4f}, {r["ci"][1]:.4f}]</b></div>',unsafe_allow_html=True)

            # SPSS-style tables
            st.markdown('<div class="sec-head">SPSS-Style Output Tables</div>', unsafe_allow_html=True)
            desc=st.session_state.get("desc_out",{})
            if desc:
                st.markdown("**Descriptive Statistics**")
                first=list(desc.values())[0]
                if isinstance(first,dict):
                    grps=list(desc.keys()); sk=list(list(desc.values())[0].keys())
                    rows=[["Statistic"]+grps]+[[s]+[str(desc[g][s]) for g in grps] for s in sk]
                else:
                    rows=[["Statistic","Value"]]+[[k,str(v)] for k,v in desc.items()]
                st.markdown(spss_table(rows),unsafe_allow_html=True)

            if "t_equal" in r:
                st.markdown("**Levene's Test for Equality of Variances**")
                lv=r["levene"]
                st.markdown(spss_table([["","F","df1","df2","Sig."],
                    ["Levene's Test",f"{lv['statistic']:.4f}",str(lv['df1']),
                     str(lv['df2']),f"{lv['p_value']:.4f}"]]),unsafe_allow_html=True)
                cp=int((1-alph)*100)
                st.markdown("**t-Test for Equality of Means**")
                st.markdown(spss_table([
                    ["","t","df","Sig.(2-tailed)",f"{cp}% CI Lower",f"{cp}% CI Upper","Used"],
                    ["Equal variances assumed",
                     f"{r['t_equal']:.4f}",f"{r['df_equal']:.2f}",f"{r['p_equal']:.4f}",
                     f"{r['ci_equal'][0]:.4f}",f"{r['ci_equal'][1]:.4f}",
                     "✓" if r["use_equal"] else ""],
                    ["Equal variances NOT assumed (Welch's)",
                     f"{r['t_welch']:.4f}",f"{r['df_welch']:.4f}",f"{r['p_welch']:.4f}",
                     f"{r['ci_welch'][0]:.4f}",f"{r['ci_welch'][1]:.4f}",
                     "" if r["use_equal"] else "✓"]]),unsafe_allow_html=True)
            else:
                st.markdown(f"**{r['test']}**")
                srows=[["Parameter","Value"]]
                dmap={"statistic":sl,"df":"df","p_value":"Sig.",
                      "mean_diff":"Mean Difference","std_diff":"Std. Dev. of Differences",
                      "n":"N","n_effective":"N (effective)",
                      "T_plus":"T+ (Positive Ranks)","T_minus":"T− (Negative Ranks)",
                      "Z":"Z (asymptotic)","U1":"U₁","U2":"U₂",
                      "pearson_r":"Pearson r","pearson_p":"r Sig.",
                      "mean1":"Mean (Group 1)","mean2":"Mean (Group 2)",
                      "median1":"Median (Group 1)","median2":"Median (Group 2)"}
                for k,lbl in dmap.items():
                    if k in r and r[k] is not None and not isinstance(r[k],bool):
                        v=r[k]; srows.append([lbl,f"{v:.4f}" if isinstance(v,float) else str(v)])
                if "ci" in r:
                    srows.append([f"{int((1-alph)*100)}% CI Lower",f"{r['ci'][0]:.4f}"])
                    srows.append([f"{int((1-alph)*100)}% CI Upper",f"{r['ci'][1]:.4f}"])
                srows.append(["Effect Magnitude",ei])
                st.markdown(spss_table(srows),unsafe_allow_html=True)

            if "pearson_r" in r and "t-Test" in r["test"]:
                st.markdown("**Paired Samples Correlations**")
                st.markdown(spss_table([["Pair","N","Pearson r","Sig."],
                    ["Pair 1",str(r["n"]),f"{r['pearson_r']:.4f}",f"{r['pearson_p']:.4f}"]]),
                    unsafe_allow_html=True)

            # Plot
            st.markdown('<div class="sec-head">Visualisation</div>', unsafe_allow_html=True)
            st.pyplot(st.session_state["test_fig"],use_container_width=True)
            try: plt.close(st.session_state["test_fig"])
            except: pass

            # Downloads
            st.markdown('<div class="sec-head">Download Results</div>', unsafe_allow_html=True)
            dl1,dl2,dl3=st.columns(3)
            with dl1:
                st.download_button("⬇ Download Plot (PNG)",
                    data=st.session_state['fig_test_bytes'],
                    file_name="test_results.png",
                    mime="image/png"
                )
            
            # CSV download
            with dl2:
                csv_buf = io.StringIO()
                full_df.to_csv(csv_buf, index=False)
                st.download_button(
                    "⬇  Download Results (CSV)",
                    data=csv_buf.getvalue(),
                    file_name="test_results.csv",
                    mime="text/csv"
                )
            
            # PDF download
            with dl3:
                pdf_bytes = generate_pdf_report(
                    test_type=result['test'],
                    assumption_results=st.session_state['assumption_results'],
                    test_result=result,
                    fig_norm_bytes_list=st.session_state['fig_norm_bytes_list'],
                    fig_test_bytes=st.session_state['fig_test_bytes'],
                    descriptives=st.session_state['descriptives'],
                    alpha=st.session_state.get('alpha', 0.05),
                    alternative=st.session_state.get('alternative', 'two-sided')
                )
                st.download_button(
                    "⬇  Download Report (PDF)",
                    data=pdf_bytes,
                    file_name="statistical_report.pdf",
                    mime="application/pdf"
                )

    # ── TAB 4: USER GUIDE ────────────────────
    with tab4:
        st.markdown("## User Guide — StatTest Suite")
        
        st.markdown("""
<div class="info-box">
This application automates the selection and execution of parametric and non-parametric hypothesis tests, 
including prerequisite assumption checking. All decisions are made algorithmically based on your data.
</div>
""", unsafe_allow_html=True)
        
        st.markdown("### 1. Conceptual Overview")
        st.markdown("""
Statistical hypothesis testing is a formal procedure for evaluating evidence against a null hypothesis (H₀). 
The **parametric** approach assumes the data follow a specific probability distribution (typically Gaussian/Normal). 
When this assumption is violated, **non-parametric** methods — which rely on rank-ordering rather than distributional 
assumptions — are employed as robust alternatives.
""")
        
        st.markdown("### 2. Decision Logic")
        st.markdown("""
<div class="formula-box">
STEP 1: Upload CSV → select test type and configuration
STEP 2: System runs normality test automatically
         • n < 50  → Shapiro-Wilk (W statistic)
         • n ≥ 50  → Kolmogorov-Smirnov (D statistic)
STEP 3: Decision branch
         • Normally distributed → Parametric test
         • Non-normal          → Non-parametric equivalent
STEP 4: For independent samples, if normal:
         • Run Levene's test for equality of variances
         • Equal variances     → Student's t-test (pooled)
         • Unequal variances   → Welch's t-test (Satterthwaite df)
</div>
""", unsafe_allow_html=True)
        
        st.markdown("### 3. Hypothesis Statements")
        
        with st.expander("One-Sample Tests", expanded=True):
            st.markdown("""
**Parametric: One-Sample t-Test**

H₀: μ = μ₀  
H₁: μ ≠ μ₀ (two-sided) | μ > μ₀ (greater) | μ < μ₀ (less)

Formula:
""")
            st.markdown("""
<div class="formula-box">
t = (x̄ − μ₀) / (s / √n)

where:
  x̄  = sample mean
  μ₀ = hypothesised population mean
  s  = sample standard deviation (ddof = 1)
  n  = sample size
  df = n − 1

Cohen's d = (x̄ − μ₀) / s
</div>
""", unsafe_allow_html=True)
            st.markdown("""
**Non-Parametric: Wilcoxon Signed-Rank Test (One-Sample)**

H₀: Population median = μ₀  
Procedure: Compute differences dᵢ = xᵢ − μ₀, rank |dᵢ|, compute W⁺ and W⁻.
Effect size r = Z / √N
""")
        
        with st.expander("Paired (Dependent) Samples Tests", expanded=True):
            st.markdown("""
**Parametric: Paired-Samples t-Test**

H₀: μ_d = 0 (no mean difference)  
H₁: μ_d ≠ 0 (two-sided) | μ_d > 0 | μ_d < 0

Formula:
""")
            st.markdown("""
<div class="formula-box">
t = d̄ / (s_d / √n)

where:
  d̄  = mean of paired differences (dᵢ = x₁ᵢ − x₂ᵢ)
  s_d = standard deviation of differences
  n   = number of pairs
  df  = n − 1

Cohen's d_z = d̄ / s_d
</div>
""", unsafe_allow_html=True)
            st.markdown("""
**Non-Parametric: Wilcoxon Signed-Rank Test (Paired)**

H₀: Distribution of differences is symmetric about zero  
Same procedure as one-sample variant, applied to paired differences.
""")
        
        with st.expander("Independent Samples Tests", expanded=True):
            st.markdown("""
**Parametric: Student's t-Test (equal variances)**
""")
            st.markdown("""
<div class="formula-box">
t = (x̄₁ − x̄₂) / (s_p · √(1/n₁ + 1/n₂))

s_p² = [(n₁−1)s₁² + (n₂−1)s₂²] / (n₁ + n₂ − 2)
df = n₁ + n₂ − 2
</div>
""", unsafe_allow_html=True)
            st.markdown("""
**Parametric: Welch's t-Test (unequal variances)**
""")
            st.markdown("""
<div class="formula-box">
t = (x̄₁ − x̄₂) / √(s₁²/n₁ + s₂²/n₂)

df (Satterthwaite) = (s₁²/n₁ + s₂²/n₂)² / 
                     [(s₁²/n₁)²/(n₁−1) + (s₂²/n₂)²/(n₂−1)]

Cohen's d = (x̄₁ − x̄₂) / s_pooled
</div>
""", unsafe_allow_html=True)
            st.markdown("""
**Non-Parametric: Mann-Whitney U Test**

H₀: The two populations have the same distribution  
Effect size r = Z / √(n₁ + n₂)
""")
        
        st.markdown("### 4. Effect Size Interpretation")
        st.markdown("""
| Magnitude   | Cohen's d | Rank-biserial r |
|-------------|-----------|-----------------|
| Negligible  | < 0.20    | < 0.10          |
| Small       | 0.20–0.49 | 0.10–0.29       |
| Medium      | 0.50–0.79 | 0.30–0.49       |
| Large       | ≥ 0.80    | ≥ 0.50          |

*Based on Cohen (1988) and Rosenthal (1991) conventions.*
""")
        
        st.markdown("### 5. Assumptions & Limitations")
        st.markdown("""
- **Independence**: All tests assume observations are independent within groups.
- **Scale**: t-tests and Wilcoxon tests assume interval-level data; Mann-Whitney U requires at least ordinal data.
- **Sample size**: Very small samples (n < 5) may yield unreliable results regardless of test choice.
- **Ties**: Wilcoxon and Mann-Whitney procedures handle ties via midrank correction.
- **Multiple testing**: This suite performs single comparisons only. Apply appropriate corrections (e.g., Bonferroni) when conducting multiple tests.
""")
        
        st.markdown("### 6. References")
        st.markdown("""
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum.
- Field, A. (2018). *Discovering Statistics Using IBM SPSS Statistics* (5th ed.). SAGE.
- Hollander, M., Wolfe, D.A., & Chicken, E. (2013). *Nonparametric Statistical Methods* (3rd ed.). Wiley.
- Shapiro, S.S., & Wilk, M.B. (1965). An analysis of variance test for normality. *Biometrika*, 52(3–4), 591–611.
- Welch, B.L. (1947). The generalization of "Student's" problem when several different population variances are involved. *Biometrika*, 34(1–2), 28–35.
""")


if __name__ == "__main__":
    main()
