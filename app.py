"""
Statistical Hypothesis Testing Suite
Automated Parametric & Non-Parametric Testing
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import io
import base64
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether, PageBreak
)
from reportlab.platypus import Image as RLImage
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StatTest Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'Libre Baskerville', Georgia, serif;
}

.stApp {
    background: #0f0f0f;
    color: #e8e0d0;
}

h1, h2, h3, h4 {
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: -0.02em;
}

.main-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: #f0e6c8;
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-family: 'Libre Baskerville', serif;
    font-style: italic;
    color: #8a7f6a;
    font-size: 1rem;
    margin-bottom: 2rem;
}

.stat-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-left: 3px solid #c9a84c;
    padding: 1.2rem 1.4rem;
    border-radius: 2px;
    margin: 0.6rem 0;
}

.stat-card h4 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #c9a84c;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}

.stat-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #f0e6c8;
}

.stat-card .label {
    font-size: 0.82rem;
    color: #6a6055;
    margin-top: 0.2rem;
}

.verdict-box {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    padding: 1.6rem;
    border-radius: 2px;
    margin: 1rem 0;
    text-align: center;
}

.verdict-reject {
    border-top: 4px solid #d4574a;
}

.verdict-fail {
    border-top: 4px solid #5a9e6f;
}

.verdict-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #8a7f6a;
    margin-bottom: 0.5rem;
}

.verdict-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
}

.reject-color { color: #d4574a; }
.fail-color { color: #5a9e6f; }

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #c9a84c;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

.assumption-row {
    display: flex;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #1e1e1e;
    font-size: 0.88rem;
}

.badge-pass {
    background: #1e3328;
    color: #5a9e6f;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    padding: 0.2rem 0.5rem;
    border-radius: 1px;
    border: 1px solid #2a4a36;
    letter-spacing: 0.08em;
}

.badge-fail {
    background: #3a1e1e;
    color: #d4574a;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    padding: 0.2rem 0.5rem;
    border-radius: 1px;
    border: 1px solid #5a2a2a;
    letter-spacing: 0.08em;
}

.info-box {
    background: #141414;
    border: 1px solid #252525;
    border-left: 2px solid #5a7fa0;
    padding: 1rem 1.2rem;
    border-radius: 2px;
    font-size: 0.86rem;
    color: #a09080;
    margin: 0.8rem 0;
    line-height: 1.7;
}

.formula-box {
    background: #111;
    border: 1px solid #222;
    padding: 1rem 1.4rem;
    border-radius: 2px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #c9a84c;
    margin: 0.8rem 0;
    line-height: 2;
}

.stButton > button {
    background: #c9a84c !important;
    color: #0f0f0f !important;
    border: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.8rem !important;
    border-radius: 1px !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: #dbb85c !important;
}

.stDownloadButton > button {
    background: transparent !important;
    color: #c9a84c !important;
    border: 1px solid #c9a84c !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

.stSelectbox label, .stNumberInput label, .stRadio label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #8a7f6a !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

[data-testid="stSidebar"] {
    background: #0a0a0a;
    border-right: 1px solid #1e1e1e;
}

.stDataFrame {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
}

hr {
    border-color: #2a2a2a !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #111;
    border-bottom: 1px solid #2a2a2a;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6a6055;
    background: transparent;
    border-radius: 0;
    padding: 0.8rem 1.2rem;
}

.stTabs [aria-selected="true"] {
    color: #c9a84c !important;
    border-bottom: 2px solid #c9a84c !important;
    background: transparent !important;
}

.warning-box {
    background: #1e1a10;
    border: 1px solid #3a3020;
    border-left: 3px solid #c9a84c;
    padding: 0.8rem 1rem;
    border-radius: 2px;
    font-size: 0.84rem;
    color: #a09060;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MATPLOTLIB THEME
# ─────────────────────────────────────────────
def apply_dark_theme():
    plt.rcParams.update({
        'figure.facecolor': '#111111',
        'axes.facecolor': '#161616',
        'axes.edgecolor': '#2a2a2a',
        'axes.labelcolor': '#a09080',
        'axes.titlecolor': '#e0d8c8',
        'xtick.color': '#6a6055',
        'ytick.color': '#6a6055',
        'text.color': '#a09080',
        'grid.color': '#1e1e1e',
        'grid.linewidth': 0.8,
        'lines.linewidth': 1.8,
        'font.family': 'serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

apply_dark_theme()


# ─────────────────────────────────────────────
# STATISTICAL FUNCTIONS
# ─────────────────────────────────────────────

def shapiro_wilk_test(data, alpha=0.05):
    """Shapiro-Wilk normality test. Best for n < 50."""
    stat, p = stats.shapiro(data)
    return {
        'test': 'Shapiro-Wilk',
        'statistic': stat,
        'p_value': p,
        'normal': p > alpha,
        'interpretation': 'Normal' if p > alpha else 'Non-normal'
    }

def kolmogorov_smirnov_test(data, alpha=0.05):
    """Kolmogorov-Smirnov normality test. Used for n >= 50."""
    stat, p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
    return {
        'test': 'Kolmogorov-Smirnov',
        'statistic': stat,
        'p_value': p,
        'normal': p > alpha,
        'interpretation': 'Normal' if p > alpha else 'Non-normal'
    }

def normality_test(data, alpha=0.05):
    """Automatically select normality test based on sample size."""
    n = len(data)
    if n < 50:
        result = shapiro_wilk_test(data, alpha)
    else:
        result = kolmogorov_smirnov_test(data, alpha)
    result['n'] = n
    return result

def levene_test(group1, group2, alpha=0.05):
    """Levene's test for equality of variances."""
    stat, p = stats.levene(group1, group2)
    return {
        'test': "Levene's Test",
        'statistic': stat,
        'p_value': p,
        'equal_variance': p > alpha,
        'interpretation': 'Equal variances' if p > alpha else 'Unequal variances'
    }

# ── One-Sample Tests ──────────────────────────

def one_sample_t_test(data, mu0, alpha=0.05, alternative='two-sided'):
    """One-sample t-test: H0: μ = μ0"""
    n = len(data)
    x_bar = np.mean(data)
    s = np.std(data, ddof=1)
    se = s / np.sqrt(n)
    t_stat = (x_bar - mu0) / se
    df = n - 1
    if alternative == 'two-sided':
        p_val = 2 * stats.t.sf(abs(t_stat), df)
    elif alternative == 'greater':
        p_val = stats.t.sf(t_stat, df)
    else:
        p_val = stats.t.cdf(t_stat, df)
    # Confidence interval
    t_crit = stats.t.ppf(1 - alpha/2, df)
    ci_lower = x_bar - t_crit * se
    ci_upper = x_bar + t_crit * se
    # Cohen's d
    d = (x_bar - mu0) / s
    return {
        'test': 'One-Sample t-Test',
        'statistic': t_stat,
        'df': df,
        'p_value': p_val,
        'reject': p_val < alpha,
        'mean': x_bar,
        'std': s,
        'se': se,
        'mu0': mu0,
        'n': n,
        'ci': (ci_lower, ci_upper),
        'effect_size': d,
        'effect_label': 'Cohen\'s d',
        'alternative': alternative
    }

def wilcoxon_signed_rank_one_sample(data, mu0, alpha=0.05, alternative='two-sided'):
    """Wilcoxon Signed-Rank Test (one-sample): non-parametric equivalent of one-sample t-test."""
    differences = np.array(data) - mu0
    # Remove zeros
    differences = differences[differences != 0]
    n = len(differences)
    abs_diff = np.abs(differences)
    ranks = stats.rankdata(abs_diff)
    W_plus = np.sum(ranks[differences > 0])
    W_minus = np.sum(ranks[differences < 0])
    W = min(W_plus, W_minus)
    # Use scipy implementation for p-value
    stat, p_val = stats.wilcoxon(np.array(data) - mu0, alternative=alternative)
    # Effect size r = Z / sqrt(N)
    mean_W = n * (n + 1) / 4
    std_W = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    Z = (W - mean_W) / std_W if std_W > 0 else 0
    r = abs(Z) / np.sqrt(n)
    return {
        'test': 'Wilcoxon Signed-Rank Test (One-Sample)',
        'statistic': stat,
        'W_plus': W_plus,
        'W_minus': W_minus,
        'p_value': p_val,
        'reject': p_val < alpha,
        'n_effective': n,
        'mu0': mu0,
        'median': np.median(data),
        'effect_size': r,
        'effect_label': 'r (rank-biserial)',
        'alternative': alternative
    }

# ── Paired/Dependent Samples Tests ────────────

def paired_t_test(group1, group2, alpha=0.05, alternative='two-sided'):
    """Paired-samples t-test: H0: μ_d = 0"""
    d = np.array(group1) - np.array(group2)
    n = len(d)
    d_bar = np.mean(d)
    s_d = np.std(d, ddof=1)
    se = s_d / np.sqrt(n)
    t_stat = d_bar / se
    df = n - 1
    if alternative == 'two-sided':
        p_val = 2 * stats.t.sf(abs(t_stat), df)
    elif alternative == 'greater':
        p_val = stats.t.sf(t_stat, df)
    else:
        p_val = stats.t.cdf(t_stat, df)
    t_crit = stats.t.ppf(1 - alpha/2, df)
    ci_lower = d_bar - t_crit * se
    ci_upper = d_bar + t_crit * se
    d_effect = d_bar / s_d
    return {
        'test': 'Paired-Samples t-Test',
        'statistic': t_stat,
        'df': df,
        'p_value': p_val,
        'reject': p_val < alpha,
        'mean_diff': d_bar,
        'std_diff': s_d,
        'se': se,
        'n': n,
        'ci': (ci_lower, ci_upper),
        'effect_size': d_effect,
        'effect_label': 'Cohen\'s d_z',
        'alternative': alternative,
        'differences': d
    }

def wilcoxon_signed_rank_paired(group1, group2, alpha=0.05, alternative='two-sided'):
    """Wilcoxon Signed-Rank Test (paired): non-parametric equivalent of paired t-test."""
    d = np.array(group1) - np.array(group2)
    stat, p_val = stats.wilcoxon(group1, group2, alternative=alternative)
    d_nz = d[d != 0]
    n_eff = len(d_nz)
    abs_d = np.abs(d_nz)
    ranks = stats.rankdata(abs_d)
    W_plus = np.sum(ranks[d_nz > 0])
    mean_W = n_eff * (n_eff + 1) / 4
    std_W = np.sqrt(n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24)
    Z = (W_plus - mean_W) / std_W if std_W > 0 else 0
    r = abs(Z) / np.sqrt(n_eff)
    return {
        'test': 'Wilcoxon Signed-Rank Test (Paired)',
        'statistic': stat,
        'p_value': p_val,
        'reject': p_val < alpha,
        'n': len(group1),
        'n_effective': n_eff,
        'median_diff': np.median(d),
        'effect_size': r,
        'effect_label': 'r (rank-biserial)',
        'alternative': alternative,
        'differences': d
    }

# ── Independent Samples Tests ─────────────────

def independent_t_test(group1, group2, alpha=0.05, alternative='two-sided', equal_var=True):
    """Independent-samples t-test (Student's or Welch's)."""
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    if equal_var:
        # Student's t-test (pooled variance)
        sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
        sp = np.sqrt(sp2)
        se = sp * np.sqrt(1/n1 + 1/n2)
        df = n1 + n2 - 2
        test_name = "Independent-Samples t-Test (Student's)"
    else:
        # Welch's t-test
        se = np.sqrt(s1**2/n1 + s2**2/n2)
        df = (s1**2/n1 + s2**2/n2)**2 / ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1))
        sp = None
        test_name = "Independent-Samples t-Test (Welch's)"
    t_stat = (m1 - m2) / se
    if alternative == 'two-sided':
        p_val = 2 * stats.t.sf(abs(t_stat), df)
    elif alternative == 'greater':
        p_val = stats.t.sf(t_stat, df)
    else:
        p_val = stats.t.cdf(t_stat, df)
    t_crit = stats.t.ppf(1 - alpha/2, df)
    ci_lower = (m1 - m2) - t_crit * se
    ci_upper = (m1 - m2) + t_crit * se
    # Cohen's d (pooled)
    sp_eff = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    d = (m1 - m2) / sp_eff
    return {
        'test': test_name,
        'statistic': t_stat,
        'df': df,
        'p_value': p_val,
        'reject': p_val < alpha,
        'mean1': m1, 'mean2': m2,
        'std1': s1, 'std2': s2,
        'n1': n1, 'n2': n2,
        'se': se,
        'ci': (ci_lower, ci_upper),
        'effect_size': d,
        'effect_label': "Cohen's d",
        'equal_var': equal_var,
        'alternative': alternative
    }

def mann_whitney_u_test(group1, group2, alpha=0.05, alternative='two-sided'):
    """Mann-Whitney U test: non-parametric equivalent of independent-samples t-test."""
    stat, p_val = stats.mannwhitneyu(group1, group2, alternative=alternative)
    n1, n2 = len(group1), len(group2)
    # Effect size r
    U1 = stat
    mean_U = n1 * n2 / 2
    std_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    Z = (U1 - mean_U) / std_U if std_U > 0 else 0
    r = abs(Z) / np.sqrt(n1 + n2)
    return {
        'test': 'Mann-Whitney U Test',
        'statistic': stat,
        'p_value': p_val,
        'reject': p_val < alpha,
        'n1': n1, 'n2': n2,
        'median1': np.median(group1),
        'median2': np.median(group2),
        'effect_size': r,
        'effect_label': 'r (rank-biserial)',
        'alternative': alternative
    }


# ─────────────────────────────────────────────
# EFFECT SIZE INTERPRETATION
# ─────────────────────────────────────────────

def interpret_effect_size(d, label):
    """Interpret effect size magnitude."""
    if 'Cohen' in label:
        d = abs(d)
        if d < 0.2: return 'Negligible'
        elif d < 0.5: return 'Small'
        elif d < 0.8: return 'Medium'
        else: return 'Large'
    else:  # r
        d = abs(d)
        if d < 0.1: return 'Negligible'
        elif d < 0.3: return 'Small'
        elif d < 0.5: return 'Medium'
        else: return 'Large'


# ─────────────────────────────────────────────
# VISUALIZATION FUNCTIONS
# ─────────────────────────────────────────────

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()

def plot_normality_diagnostics(data, col_name, norm_result):
    apply_dark_theme()
    fig = plt.figure(figsize=(14, 10), facecolor='#111111')
    fig.suptitle(f'Normality Diagnostics — {col_name}', 
                 color='#e0d8c8', fontsize=13, fontweight='bold',
                 y=0.98, fontfamily='monospace')
    
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    gold = '#c9a84c'
    red = '#d4574a'
    green = '#5a9e6f'
    
    # 1. Histogram with KDE
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor('#161616')
    x = np.linspace(min(data) - 0.5*np.std(data), max(data) + 0.5*np.std(data), 300)
    ax1.hist(data, bins='auto', density=True, color='#2a3a4a', edgecolor='#1a2a3a',
             alpha=0.85, zorder=2)
    kde = stats.gaussian_kde(data)
    ax1.plot(x, kde(x), color=gold, lw=2, label='KDE', zorder=3)
    mu, sigma = np.mean(data), np.std(data, ddof=1)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), color=green, lw=1.5,
             linestyle='--', label='Normal fit', zorder=3)
    ax1.set_title('Histogram + KDE', color='#e0d8c8', fontsize=10, pad=8)
    ax1.set_xlabel(col_name, color='#8a7f6a', fontsize=9)
    ax1.set_ylabel('Density', color='#8a7f6a', fontsize=9)
    ax1.legend(fontsize=8, framealpha=0.2, facecolor='#1a1a1a', edgecolor='#2a2a2a')
    ax1.axvline(mu, color='#5a7fa0', lw=1.2, linestyle=':', alpha=0.8)
    ax1.grid(True, alpha=0.15)

    # 2. Normality test annotation
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor('#161616')
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.axis('off')
    clr = green if norm_result['normal'] else red
    ax2.text(0.5, 0.88, norm_result['test'], ha='center', va='top',
             fontsize=9, color='#8a7f6a', fontfamily='monospace',
             transform=ax2.transAxes)
    ax2.text(0.5, 0.68, f"W = {norm_result['statistic']:.4f}",
             ha='center', va='top', fontsize=12, color=gold, fontfamily='monospace',
             fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.5, 0.50, f"p = {norm_result['p_value']:.4f}",
             ha='center', va='top', fontsize=11, color='#e0d8c8', fontfamily='monospace',
             transform=ax2.transAxes)
    ax2.text(0.5, 0.30, norm_result['interpretation'].upper(),
             ha='center', va='top', fontsize=14, color=clr, fontfamily='monospace',
             fontweight='bold', transform=ax2.transAxes)
    ax2.text(0.5, 0.12, f"n = {norm_result['n']}", ha='center', va='top',
             fontsize=9, color='#6a6055', fontfamily='monospace',
             transform=ax2.transAxes)
    rect = FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.02",
                          linewidth=1, edgecolor=clr, facecolor='none',
                          transform=ax2.transAxes)
    ax2.add_patch(rect)

    # 3. Q-Q Plot
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#161616')
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist='norm')
    ax3.scatter(osm, osr, color='#3a5a7a', s=18, alpha=0.8, zorder=3)
    ax3.plot(osm, slope * np.array(osm) + intercept, color=gold, lw=1.5, zorder=4)
    ax3.set_title('Q-Q Plot', color='#e0d8c8', fontsize=10, pad=8)
    ax3.set_xlabel('Theoretical Quantiles', color='#8a7f6a', fontsize=8)
    ax3.set_ylabel('Sample Quantiles', color='#8a7f6a', fontsize=8)
    ax3.grid(True, alpha=0.15)

    # 4. Box Plot
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#161616')
    bp = ax4.boxplot(data, patch_artist=True, widths=0.4,
                     medianprops=dict(color=gold, lw=2),
                     boxprops=dict(facecolor='#2a3a4a', edgecolor='#3a5a7a'),
                     whiskerprops=dict(color='#3a5a7a', lw=1.2),
                     capprops=dict(color='#3a5a7a', lw=1.5),
                     flierprops=dict(marker='o', color=red, ms=4, alpha=0.7))
    ax4.set_title('Box Plot', color='#e0d8c8', fontsize=10, pad=8)
    ax4.set_ylabel(col_name, color='#8a7f6a', fontsize=8)
    ax4.set_xticks([])
    ax4.grid(True, alpha=0.15, axis='y')

    # 5. ECDF vs Normal CDF
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor('#161616')
    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(data) + 1) / len(data)
    x_range = np.linspace(min(data), max(data), 300)
    ax5.step(sorted_data, ecdf, color='#3a5a7a', lw=1.5, label='ECDF', zorder=3)
    ax5.plot(x_range, stats.norm.cdf(x_range, mu, sigma), color=gold,
             lw=1.5, linestyle='--', label='Normal CDF', zorder=4)
    ax5.set_title('ECDF vs Normal CDF', color='#e0d8c8', fontsize=10, pad=8)
    ax5.set_xlabel(col_name, color='#8a7f6a', fontsize=8)
    ax5.set_ylabel('Cumulative Probability', color='#8a7f6a', fontsize=8)
    ax5.legend(fontsize=7, framealpha=0.2, facecolor='#1a1a1a', edgecolor='#2a2a2a')
    ax5.grid(True, alpha=0.15)

    return fig


def plot_one_sample_results(data, result, mu0):
    apply_dark_theme()
    fig = plt.figure(figsize=(14, 9), facecolor='#111111')
    fig.suptitle(f'One-Sample Test Results — {result["test"]}',
                 color='#e0d8c8', fontsize=12, fontweight='bold', y=0.98,
                 fontfamily='monospace')
    
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)
    gold = '#c9a84c'
    red = '#d4574a'
    green = '#5a9e6f'
    blue = '#5a7fa0'
    
    # 1. Distribution with test value
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor('#161616')
    x = np.linspace(min(data) - 2*np.std(data), max(data) + 2*np.std(data), 300)
    kde = stats.gaussian_kde(data)
    ax1.fill_between(x, kde(x), color='#2a3a4a', alpha=0.7, zorder=2)
    ax1.plot(x, kde(x), color=blue, lw=2, zorder=3)
    ax1.axvline(np.mean(data), color=gold, lw=2, label=f'Sample mean = {np.mean(data):.3f}', zorder=4)
    ax1.axvline(mu0, color=red, lw=2, linestyle='--', label=f'H₀: μ = {mu0}', zorder=4)
    if 'ci' in result:
        ax1.axvspan(result['ci'][0], result['ci'][1], alpha=0.12, color=gold, label='95% CI')
    ax1.set_title('Sample Distribution vs. Hypothesised Value', color='#e0d8c8', fontsize=10, pad=8)
    ax1.set_xlabel('Value', color='#8a7f6a', fontsize=9)
    ax1.set_ylabel('Density', color='#8a7f6a', fontsize=9)
    ax1.legend(fontsize=8, framealpha=0.2, facecolor='#1a1a1a', edgecolor='#2a2a2a')
    ax1.grid(True, alpha=0.15)

    # 2. Result panel
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor('#161616'); ax2.axis('off')
    clr = red if result['reject'] else green
    verdict = 'REJECT H₀' if result['reject'] else 'FAIL TO REJECT H₀'
    items = [
        ('Test Statistic', f"{result['statistic']:.4f}"),
        ('p-value', f"{result['p_value']:.4f}"),
        (result['effect_label'], f"{result['effect_size']:.4f}"),
    ]
    y_pos = 0.92
    ax2.text(0.5, y_pos, verdict, ha='center', va='top', fontsize=13,
             color=clr, fontweight='bold', fontfamily='monospace',
             transform=ax2.transAxes)
    y_pos -= 0.16
    for lbl, val in items:
        ax2.text(0.08, y_pos, lbl, ha='left', va='top', fontsize=8,
                 color='#6a6055', fontfamily='monospace', transform=ax2.transAxes)
        ax2.text(0.92, y_pos, val, ha='right', va='top', fontsize=9,
                 color=gold, fontfamily='monospace', fontweight='bold',
                 transform=ax2.transAxes)
        y_pos -= 0.13
        ax2.plot([0.05, 0.95], [y_pos + 0.03, y_pos + 0.03],
                 color='#2a2a2a', linewidth=0.5, transform=ax2.transAxes)
    rect = FancyBboxPatch((0.03, 0.03), 0.94, 0.94, boxstyle="round,pad=0.02",
                          lw=1, edgecolor=clr, facecolor='none',
                          transform=ax2.transAxes)
    ax2.add_patch(rect)

    # 3. t/Z distribution (for t-test)
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax3.set_facecolor('#161616')
    if 't-Test' in result['test']:
        df = result.get('df', 30)
        x_t = np.linspace(-5, 5, 500)
        y_t = stats.t.pdf(x_t, df)
        ax3.plot(x_t, y_t, color=blue, lw=2, zorder=3)
        alpha = 0.05
        t_crit = stats.t.ppf(0.975, df)
        ax3.fill_between(x_t, y_t, where=x_t >= t_crit, color=red, alpha=0.35, label='Rejection region')
        ax3.fill_between(x_t, y_t, where=x_t <= -t_crit, color=red, alpha=0.35)
        ax3.axvline(result['statistic'], color=gold, lw=2, label=f't = {result["statistic"]:.3f}')
        ax3.set_title(f't-Distribution (df = {df:.1f})', color='#e0d8c8', fontsize=10, pad=8)
        ax3.set_xlabel('t-statistic', color='#8a7f6a', fontsize=9)
    else:
        # For Wilcoxon: show rank distribution approximation
        sorted_d = np.sort(np.abs(np.array(data) - mu0))
        ax3.bar(range(1, len(sorted_d)+1), sorted_d, color='#2a3a4a',
                edgecolor='#3a5a7a', alpha=0.8)
        ax3.set_title('Absolute Differences (Ranked)', color='#e0d8c8', fontsize=10, pad=8)
        ax3.set_xlabel('Rank', color='#8a7f6a', fontsize=9)
        ax3.set_ylabel('|x - μ₀|', color='#8a7f6a', fontsize=9)
    ax3.legend(fontsize=8, framealpha=0.2, facecolor='#1a1a1a', edgecolor='#2a2a2a')
    ax3.grid(True, alpha=0.15)

    # 4. Descriptive stats
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor('#161616'); ax4.axis('off')
    desc = [
        ('n', str(len(data))),
        ('Mean', f'{np.mean(data):.4f}'),
        ('Median', f'{np.median(data):.4f}'),
        ('Std Dev', f'{np.std(data, ddof=1):.4f}'),
        ('Min', f'{np.min(data):.4f}'),
        ('Max', f'{np.max(data):.4f}'),
    ]
    ax4.text(0.5, 0.96, 'DESCRIPTIVES', ha='center', va='top', fontsize=8,
             color='#8a7f6a', fontfamily='monospace', transform=ax4.transAxes,
             fontweight='bold')
    y_pos = 0.84
    for lbl, val in desc:
        ax4.text(0.1, y_pos, lbl, ha='left', va='top', fontsize=8.5,
                 color='#6a6055', fontfamily='monospace', transform=ax4.transAxes)
        ax4.text(0.92, y_pos, val, ha='right', va='top', fontsize=8.5,
                 color='#e0d8c8', fontfamily='monospace', transform=ax4.transAxes)
        y_pos -= 0.12
    return fig


def plot_paired_results(group1, group2, result, label1='Group 1', label2='Group 2'):
    apply_dark_theme()
    fig = plt.figure(figsize=(14, 9), facecolor='#111111')
    fig.suptitle(f'Paired-Samples Test Results — {result["test"]}',
                 color='#e0d8c8', fontsize=12, fontweight='bold', y=0.98,
                 fontfamily='monospace')
    
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)
    gold = '#c9a84c'; red = '#d4574a'; green = '#5a9e6f'; blue = '#5a7fa0'
    
    # 1. Before-after plot
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor('#161616')
    n = len(group1)
    x_vals = [0, 1]
    for i in range(n):
        ax1.plot(x_vals, [group1[i], group2[i]],
                 color=red if group1[i] > group2[i] else green, alpha=0.35, lw=0.9)
    ax1.scatter([0]*n, group1, color=blue, s=30, zorder=4, alpha=0.8)
    ax1.scatter([1]*n, group2, color=gold, s=30, zorder=4, alpha=0.8)
    ax1.plot(x_vals, [np.mean(group1), np.mean(group2)], color='white', lw=2.5, zorder=5,
             marker='D', ms=6, label='Group means')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels([label1, label2], fontsize=10)
    ax1.set_title('Individual Observations — Before & After', color='#e0d8c8', fontsize=10, pad=8)
    ax1.set_ylabel('Value', color='#8a7f6a', fontsize=9)
    ax1.legend(fontsize=8, framealpha=0.2, facecolor='#1a1a1a', edgecolor='#2a2a2a')
    ax1.grid(True, alpha=0.15, axis='y')

    # 2. Result panel
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor('#161616'); ax2.axis('off')
    clr = red if result['reject'] else green
    verdict = 'REJECT H₀' if result['reject'] else 'FAIL TO REJECT H₀'
    stat_key = 'statistic'
    items = [
        ('Test Statistic', f"{result[stat_key]:.4f}"),
        ('p-value', f"{result['p_value']:.4f}"),
        (result['effect_label'], f"{result['effect_size']:.4f}"),
    ]
    y_pos = 0.92
    ax2.text(0.5, y_pos, verdict, ha='center', va='top', fontsize=12,
             color=clr, fontweight='bold', fontfamily='monospace', transform=ax2.transAxes)
    y_pos -= 0.16
    for lbl, val in items:
        ax2.text(0.08, y_pos, lbl, ha='left', va='top', fontsize=8,
                 color='#6a6055', fontfamily='monospace', transform=ax2.transAxes)
        ax2.text(0.92, y_pos, val, ha='right', va='top', fontsize=9,
                 color=gold, fontfamily='monospace', fontweight='bold', transform=ax2.transAxes)
        y_pos -= 0.13
        ax2.plot([0.05, 0.95], [y_pos + 0.03, y_pos + 0.03],
                 color='#2a2a2a', linewidth=0.5, transform=ax2.transAxes)
    rect = FancyBboxPatch((0.03, 0.03), 0.94, 0.94, boxstyle="round,pad=0.02",
                          lw=1, edgecolor=clr, facecolor='none', transform=ax2.transAxes)
    ax2.add_patch(rect)

    # 3. Differences histogram
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#161616')
    diffs = result['differences']
    ax3.hist(diffs, bins='auto', color='#2a3a4a', edgecolor='#3a5a7a', alpha=0.85)
    ax3.axvline(0, color=red, lw=1.5, linestyle='--', label='Zero difference')
    ax3.axvline(np.mean(diffs), color=gold, lw=1.5, label=f'Mean diff = {np.mean(diffs):.3f}')
    ax3.set_title('Distribution of Differences', color='#e0d8c8', fontsize=10, pad=8)
    ax3.set_xlabel('Difference (Pre − Post)', color='#8a7f6a', fontsize=8)
    ax3.legend(fontsize=7.5, framealpha=0.2, facecolor='#1a1a1a', edgecolor='#2a2a2a')
    ax3.grid(True, alpha=0.15)

    # 4. Box plots
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#161616')
    bdata = [group1, group2]
    bp = ax4.boxplot(bdata, patch_artist=True, widths=0.4,
                     medianprops=dict(color=gold, lw=2),
                     boxprops=dict(facecolor='#2a3a4a', edgecolor='#3a5a7a'),
                     whiskerprops=dict(color='#3a5a7a'), capprops=dict(color='#3a5a7a'),
                     flierprops=dict(marker='o', color=red, ms=4, alpha=0.7))
    ax4.set_xticks([1, 2])
    ax4.set_xticklabels([label1, label2])
    ax4.set_title('Box Plots', color='#e0d8c8', fontsize=10, pad=8)
    ax4.grid(True, alpha=0.15, axis='y')

    # 5. CI for mean difference
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor('#161616'); ax5.axis('off')
    md = result.get('mean_diff', result.get('median_diff', 0))
    ax5.text(0.5, 0.92, 'MEAN DIFFERENCE', ha='center', va='top', fontsize=7.5,
             color='#8a7f6a', fontfamily='monospace', transform=ax5.transAxes, fontweight='bold')
    ax5.text(0.5, 0.76, f'{md:.4f}', ha='center', va='top', fontsize=18,
             color=gold, fontfamily='monospace', fontweight='bold', transform=ax5.transAxes)
    if 'ci' in result:
        ax5.text(0.5, 0.58, '95% Confidence Interval', ha='center', va='top',
                 fontsize=8, color='#6a6055', fontfamily='monospace', transform=ax5.transAxes)
        ax5.text(0.5, 0.44, f'[{result["ci"][0]:.4f},  {result["ci"][1]:.4f}]',
                 ha='center', va='top', fontsize=9, color='#e0d8c8',
                 fontfamily='monospace', transform=ax5.transAxes)
    ax5.text(0.5, 0.24, interpret_effect_size(result['effect_size'], result['effect_label']).upper() + ' EFFECT',
             ha='center', va='top', fontsize=10, color=clr, fontfamily='monospace',
             fontweight='bold', transform=ax5.transAxes)
    return fig


def plot_independent_results(group1, group2, result, label1='Group 1', label2='Group 2'):
    apply_dark_theme()
    fig = plt.figure(figsize=(14, 9), facecolor='#111111')
    fig.suptitle(f'Independent-Samples Test Results — {result["test"]}',
                 color='#e0d8c8', fontsize=12, fontweight='bold', y=0.98,
                 fontfamily='monospace')
    
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)
    gold = '#c9a84c'; red = '#d4574a'; green = '#5a9e6f'; blue = '#5a7fa0'
    
    # 1. Overlapping KDEs
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor('#161616')
    all_data = list(group1) + list(group2)
    x_range = np.linspace(min(all_data) - np.std(all_data), max(all_data) + np.std(all_data), 300)
    kde1 = stats.gaussian_kde(group1)
    kde2 = stats.gaussian_kde(group2)
    ax1.fill_between(x_range, kde1(x_range), alpha=0.4, color=blue, label=label1)
    ax1.fill_between(x_range, kde2(x_range), alpha=0.4, color=gold, label=label2)
    ax1.plot(x_range, kde1(x_range), color=blue, lw=2)
    ax1.plot(x_range, kde2(x_range), color=gold, lw=2)
    ax1.axvline(np.mean(group1), color=blue, lw=1.5, linestyle='--', alpha=0.8)
    ax1.axvline(np.mean(group2) if 't-Test' in result['test'] else np.median(group2),
                color=gold, lw=1.5, linestyle='--', alpha=0.8)
    ax1.set_title('Group Distributions', color='#e0d8c8', fontsize=10, pad=8)
    ax1.set_xlabel('Value', color='#8a7f6a', fontsize=9)
    ax1.set_ylabel('Density', color='#8a7f6a', fontsize=9)
    ax1.legend(fontsize=8.5, framealpha=0.2, facecolor='#1a1a1a', edgecolor='#2a2a2a')
    ax1.grid(True, alpha=0.15)

    # 2. Result panel
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor('#161616'); ax2.axis('off')
    clr = red if result['reject'] else green
    verdict = 'REJECT H₀' if result['reject'] else 'FAIL TO REJECT H₀'
    items = [
        ('Test Statistic', f"{result['statistic']:.4f}"),
        ('p-value', f"{result['p_value']:.4f}"),
        (result['effect_label'], f"{result['effect_size']:.4f}"),
    ]
    y_pos = 0.92
    ax2.text(0.5, y_pos, verdict, ha='center', va='top', fontsize=12,
             color=clr, fontweight='bold', fontfamily='monospace', transform=ax2.transAxes)
    y_pos -= 0.16
    for lbl, val in items:
        ax2.text(0.08, y_pos, lbl, ha='left', va='top', fontsize=8,
                 color='#6a6055', fontfamily='monospace', transform=ax2.transAxes)
        ax2.text(0.92, y_pos, val, ha='right', va='top', fontsize=9,
                 color=gold, fontfamily='monospace', fontweight='bold', transform=ax2.transAxes)
        y_pos -= 0.13
        ax2.plot([0.05, 0.95], [y_pos + 0.03, y_pos + 0.03],
                 color='#2a2a2a', linewidth=0.5, transform=ax2.transAxes)
    rect = FancyBboxPatch((0.03, 0.03), 0.94, 0.94, boxstyle="round,pad=0.02",
                          lw=1, edgecolor=clr, facecolor='none', transform=ax2.transAxes)
    ax2.add_patch(rect)

    # 3. Box plots
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#161616')
    bp = ax3.boxplot([group1, group2], patch_artist=True, widths=0.4,
                     medianprops=dict(color='white', lw=2),
                     boxprops=dict(facecolor='#2a3a4a', edgecolor='#3a5a7a'),
                     whiskerprops=dict(color='#3a5a7a'), capprops=dict(color='#3a5a7a'),
                     flierprops=dict(marker='o', color=red, ms=4, alpha=0.7))
    bp['boxes'][0].set_facecolor('#1a2a4a')
    bp['boxes'][1].set_facecolor('#3a2a10')
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels([label1, label2])
    ax3.set_title('Box Plots', color='#e0d8c8', fontsize=10, pad=8)
    ax3.grid(True, alpha=0.15, axis='y')

    # 4. Violin plots
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#161616')
    parts = ax4.violinplot([group1, group2], positions=[1, 2], showmedians=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([blue, gold][i])
        pc.set_alpha(0.5)
    parts['cmedians'].set_color('white')
    parts['cbars'].set_color('#3a5a7a')
    parts['cmins'].set_color('#3a5a7a')
    parts['cmaxes'].set_color('#3a5a7a')
    ax4.set_xticks([1, 2])
    ax4.set_xticklabels([label1, label2])
    ax4.set_title('Violin Plots', color='#e0d8c8', fontsize=10, pad=8)
    ax4.grid(True, alpha=0.15, axis='y')

    # 5. Summary stats
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor('#161616'); ax5.axis('off')
    use_mean = 't-Test' in result['test']
    lbl_central = 'Mean' if use_mean else 'Median'
    c1 = np.mean(group1) if use_mean else np.median(group1)
    c2 = np.mean(group2) if use_mean else np.median(group2)
    rows = [
        ('', label1, label2),
        ('n', str(result['n1']), str(result['n2'])),
        (lbl_central, f'{c1:.3f}', f'{c2:.3f}'),
        ('Std Dev', f'{np.std(group1, ddof=1):.3f}', f'{np.std(group2, ddof=1):.3f}'),
        ('Min', f'{np.min(group1):.3f}', f'{np.min(group2):.3f}'),
        ('Max', f'{np.max(group1):.3f}', f'{np.max(group2):.3f}'),
    ]
    y = 0.95
    for i, row in enumerate(rows):
        fc = blue if i == 0 else ('none')
        ax5.text(0.04, y, row[0], ha='left', va='top', fontsize=7.5,
                 color='#8a7f6a', fontfamily='monospace', transform=ax5.transAxes,
                 fontweight='bold' if i == 0 else 'normal')
        ax5.text(0.45, y, row[1], ha='center', va='top', fontsize=7.5,
                 color=blue if i > 0 else blue, fontfamily='monospace',
                 transform=ax5.transAxes, fontweight='bold' if i == 0 else 'normal')
        ax5.text(0.85, y, row[2], ha='center', va='top', fontsize=7.5,
                 color=gold if i > 0 else gold, fontfamily='monospace',
                 transform=ax5.transAxes, fontweight='bold' if i == 0 else 'normal')
        y -= 0.12
    return fig


# ─────────────────────────────────────────────
# PDF REPORT GENERATION
# ─────────────────────────────────────────────

def generate_pdf_report(test_type, assumption_results, test_result,
                        fig_norm_bytes_list, fig_test_bytes,
                        descriptives, alpha, alternative):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2.2*cm, bottomMargin=2*cm)
    
    styles = getSampleStyleSheet()
    dark = colors.HexColor('#1a1a1a')
    gold = colors.HexColor('#c9a84c')
    light = colors.HexColor('#e8e0d0')
    muted = colors.HexColor('#8a7f6a')
    red_c = colors.HexColor('#d4574a')
    green_c = colors.HexColor('#5a9e6f')
    
    title_style = ParagraphStyle('Title', fontName='Helvetica-Bold', fontSize=20,
                                 textColor=light, spaceAfter=4, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle('Sub', fontName='Helvetica-Oblique', fontSize=10,
                                    textColor=muted, spaceAfter=20, alignment=TA_CENTER)
    h2_style = ParagraphStyle('H2', fontName='Helvetica-Bold', fontSize=12,
                               textColor=gold, spaceBefore=16, spaceAfter=6)
    h3_style = ParagraphStyle('H3', fontName='Helvetica-Bold', fontSize=10,
                               textColor=light, spaceBefore=10, spaceAfter=4)
    body_style = ParagraphStyle('Body', fontName='Helvetica', fontSize=9,
                                textColor=light, spaceAfter=6, leading=14,
                                alignment=TA_JUSTIFY)
    mono_style = ParagraphStyle('Mono', fontName='Courier', fontSize=8.5,
                                textColor=gold, spaceAfter=4)
    
    story = []
    
    # Header
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Statistical Hypothesis Testing Report", title_style))
    story.append(Paragraph(f"Test Type: {test_type} | α = {alpha} | Alternative: {alternative}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=gold))
    story.append(Spacer(1, 0.3*cm))
    
    # Descriptive Statistics
    story.append(Paragraph("1. Descriptive Statistics", h2_style))
    desc_data = [['Statistic'] + list(descriptives.keys())]
    row_vals = ['Value']
    for v in descriptives.values():
        row_vals.append(str(v))
    desc_data.append(row_vals)
    # Transpose
    transposed = [[descriptives_label for descriptives_label in ['Statistic', 'Value']]]
    for k, v in descriptives.items():
        transposed.append([k, str(v)])
    
    t_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2a2a2a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), gold),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#1a1a1a'), colors.HexColor('#141414')]),
        ('TEXTCOLOR', (0, 1), (-1, -1), light),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#2a2a2a')),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('FONTNAME', (1, 1), (-1, -1), 'Courier'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ])
    tbl = Table(transposed, colWidths=[8*cm, 8*cm])
    tbl.setStyle(t_style)
    story.append(tbl)
    story.append(Spacer(1, 0.3*cm))
    
    # Assumption Testing
    story.append(Paragraph("2. Assumption Testing", h2_style))
    for name, res in assumption_results.items():
        story.append(Paragraph(f"2.{list(assumption_results.keys()).index(name)+1} {name}", h3_style))
        for k, v in res.items():
            if k not in ('test', 'interpretation', 'normal', 'equal_variance'):
                story.append(Paragraph(f"&nbsp;&nbsp;{k}: {v}", mono_style))
        status = res.get('interpretation', res.get('interpretation', ''))
        story.append(Paragraph(f"<b>Result: {status}</b>", body_style))
    
    # Normality figures
    if fig_norm_bytes_list:
        story.append(Paragraph("3. Normality Diagnostic Plots", h2_style))
        for img_bytes in fig_norm_bytes_list:
            img_buf = io.BytesIO(img_bytes)
            story.append(RLImage(img_buf, width=16*cm, height=11.4*cm))
            story.append(Spacer(1, 0.3*cm))
    
    # Main test
    story.append(PageBreak())
    story.append(Paragraph("4. Hypothesis Test Results", h2_style))
    
    # Test info table
    test_rows = [['Parameter', 'Value']]
    for k, v in test_result.items():
        if k in ('test', 'statistic', 'p_value', 'df', 'effect_size',
                 'effect_label', 'reject', 'ci', 'alternative', 'n', 'n1', 'n2',
                 'mean', 'mean1', 'mean2', 'mean_diff', 'median_diff',
                 'std', 'std1', 'std2'):
            if k == 'ci':
                test_rows.append([k, f'[{v[0]:.4f}, {v[1]:.4f}]'])
            elif k == 'reject':
                test_rows.append([k, 'REJECT H₀' if v else 'FAIL TO REJECT H₀'])
            elif isinstance(v, float):
                test_rows.append([k, f'{v:.6f}'])
            else:
                test_rows.append([k, str(v)])
    
    t2_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2a2a2a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), gold),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#1a1a1a'), colors.HexColor('#141414')]),
        ('TEXTCOLOR', (0, 1), (-1, -1), light),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#2a2a2a')),
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
        ('FONTNAME', (1, 1), (-1, -1), 'Courier'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ])
    tbl2 = Table(test_rows, colWidths=[8*cm, 8*cm])
    tbl2.setStyle(t2_style)
    story.append(tbl2)
    story.append(Spacer(1, 0.4*cm))
    
    # Verdict box
    verdict_color = red_c if test_result['reject'] else green_c
    verdict_text = 'REJECT H₀' if test_result['reject'] else 'FAIL TO REJECT H₀'
    verdict_style = ParagraphStyle('Verdict', fontName='Helvetica-Bold', fontSize=14,
                                   textColor=verdict_color, alignment=TA_CENTER,
                                   spaceBefore=10, spaceAfter=10)
    story.append(Paragraph(f"Decision: {verdict_text}", verdict_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#2a2a2a')))
    story.append(Spacer(1, 0.3*cm))
    
    # Test result figure
    if fig_test_bytes:
        story.append(Paragraph("5. Test Result Visualisation", h2_style))
        img_buf2 = io.BytesIO(fig_test_bytes)
        story.append(RLImage(img_buf2, width=16*cm, height=10.3*cm))
        story.append(Spacer(1, 0.3*cm))
    
    # Interpretation
    story.append(Paragraph("6. Statistical Interpretation", h2_style))
    eff_interp = interpret_effect_size(test_result['effect_size'], test_result['effect_label'])
    p_str = f"p = {test_result['p_value']:.4f}"
    interp_text = (
        f"The {test_result['test']} yielded a test statistic of "
        f"{test_result['statistic']:.4f} ({p_str}). "
        f"At the significance level α = {alpha}, the null hypothesis is "
        f"{'rejected' if test_result['reject'] else 'not rejected'}. "
        f"The effect size ({test_result['effect_label']} = {test_result['effect_size']:.4f}) "
        f"is characterised as {eff_interp.lower()} according to conventional benchmarks."
    )
    story.append(Paragraph(interp_text, body_style))
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#2a2a2a')))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph("Generated by StatTest Suite — Automated Parametric & Non-Parametric Testing",
                            ParagraphStyle('Footer', fontName='Helvetica-Oblique', fontSize=7.5,
                                           textColor=muted, alignment=TA_CENTER)))
    
    doc.build(story)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────

def main():
    # Header
    st.markdown('<div class="main-title">StatTest Suite</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Automated Parametric & Non-Parametric Hypothesis Testing</div>',
                unsafe_allow_html=True)

    # ── SIDEBAR ──────────────────────────────
    with st.sidebar:
        st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)
        
        test_type = st.selectbox(
            "Test Type",
            ["One-Sample", "Paired (Dependent) Samples", "Independent Samples"]
        )
        
        alpha = st.selectbox("Significance Level (α)", [0.05, 0.01, 0.10], index=0)
        
        alternative = st.selectbox(
            "Alternative Hypothesis",
            ["two-sided", "greater", "less"]
        )
        
        st.markdown('<div class="section-header">Guide</div>', unsafe_allow_html=True)
        with st.expander("📋 CSV Format Guide", expanded=False):
            st.markdown("""
**One-Sample**
```
value
23.4
25.1
22.8
...
```
One column: `value`

**Paired Samples**
```
pre, post
23.4, 21.1
25.1, 22.5
...
```
Two columns: `pre`, `post`

**Independent Samples**
```
value, group
23.4, A
25.1, B
22.8, A
...
```
Two columns: `value`, `group`
""")
        with st.expander("🔬 Statistical Methods", expanded=False):
            st.markdown("""
**Normality Tests**
- n < 50: Shapiro-Wilk
- n ≥ 50: Kolmogorov-Smirnov

**Parametric Tests**
- One-sample t-test
- Paired-samples t-test
- Independent-samples t-test (Student's or Welch's)

**Non-Parametric Equivalents**
- Wilcoxon Signed-Rank (one-sample)
- Wilcoxon Signed-Rank (paired)
- Mann-Whitney U test

**Effect Size Metrics**
- Cohen's d (parametric)
- Rank-biserial r (non-parametric)
""")

    # ── TABS ─────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📂 Data Input",
        "📋 Assumptions",
        "📊 Test Results",
        "📚 User Guide"
    ])

    with tab1:
        st.markdown('<div class="section-header">Upload & Configure Data</div>', unsafe_allow_html=True)
        
        uploaded = st.file_uploader("Upload CSV File", type=['csv'])
        
        col1, col2 = st.columns([2, 1])
        
        if uploaded:
            df = pd.read_csv(uploaded)
            with col1:
                st.markdown('<div class="section-header">Preview</div>', unsafe_allow_html=True)
                st.dataframe(df.head(20), use_container_width=True)
                st.markdown(f'<div class="info-box">Dataset: <b>{len(df)}</b> rows × <b>{len(df.columns)}</b> columns</div>',
                            unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="section-header">Column Mapping</div>', unsafe_allow_html=True)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                all_cols = df.columns.tolist()
                
                config = {}
                
                if test_type == "One-Sample":
                    config['value_col'] = st.selectbox("Value Column", numeric_cols)
                    config['mu0'] = st.number_input("Hypothesised Mean (μ₀)", value=0.0, step=0.1)
                    
                elif test_type == "Paired (Dependent) Samples":
                    config['pre_col'] = st.selectbox("Pre / Group 1 Column", numeric_cols)
                    config['post_col'] = st.selectbox("Post / Group 2 Column",
                                                       [c for c in numeric_cols if c != config.get('pre_col', '')],
                                                       index=min(1, len(numeric_cols)-1))
                    
                else:  # Independent
                    config['value_col'] = st.selectbox("Value Column", numeric_cols)
                    config['group_col'] = st.selectbox("Group Column", all_cols)
                
                st.session_state['df'] = df
                st.session_state['config'] = config
                st.session_state['test_type'] = test_type
                st.session_state['alpha'] = alpha
                st.session_state['alternative'] = alternative
                
                if st.button("▶  Run Analysis"):
                    st.session_state['run'] = True
                    st.rerun()
        else:
            st.markdown("""
<div class="info-box">
<b>No file uploaded.</b> Please upload a CSV file using the widget above. 
Refer to the <i>CSV Format Guide</i> in the sidebar for the expected structure.
</div>
""", unsafe_allow_html=True)
            
            # Demo data
            st.markdown('<div class="section-header">Or Use Demo Data</div>', unsafe_allow_html=True)
            demo_type = st.selectbox("Demo Dataset", ["One-Sample", "Paired Samples", "Independent Samples"])
            if st.button("Load Demo"):
                np.random.seed(42)
                if demo_type == "One-Sample":
                    demo_df = pd.DataFrame({'value': np.random.normal(52, 8, 30)})
                elif demo_type == "Paired Samples":
                    pre = np.random.normal(70, 10, 25)
                    demo_df = pd.DataFrame({'pre': pre, 'post': pre - np.random.normal(5, 3, 25)})
                else:
                    demo_df = pd.DataFrame({
                        'value': np.concatenate([np.random.normal(65, 8, 20), np.random.normal(72, 9, 20)]),
                        'group': ['A']*20 + ['B']*20
                    })
                st.session_state['df'] = demo_df
                st.session_state['config'] = {
                    'value_col': 'value' if 'value' in demo_df.columns else None,
                    'pre_col': 'pre' if 'pre' in demo_df.columns else None,
                    'post_col': 'post' if 'post' in demo_df.columns else None,
                    'group_col': 'group' if 'group' in demo_df.columns else None,
                    'mu0': 50.0
                }
                st.session_state['test_type'] = demo_type.replace(" Samples", " (Dependent) Samples") if demo_type == "Paired Samples" else demo_type
                st.session_state['alpha'] = alpha
                st.session_state['alternative'] = alternative
                st.session_state['run'] = True
                st.rerun()

    # ── RUN ANALYSIS ─────────────────────────
    if st.session_state.get('run') and 'df' in st.session_state:
        df = st.session_state['df']
        config = st.session_state['config']
        ttype = st.session_state['test_type']
        alph = st.session_state['alpha']
        alt = st.session_state['alternative']
        
        assumption_results = {}
        fig_norm_bytes_list = []
        test_result = None
        fig_test = None
        descriptives = {}
        norm_figs = []

        try:
            # ── ONE SAMPLE ──────────────────────
            if "One-Sample" in ttype:
                data = df[config['value_col']].dropna().values
                mu0 = config.get('mu0', 0.0)
                
                norm = normality_test(data, alph)
                assumption_results['Normality Test'] = norm
                
                fig_norm = plot_normality_diagnostics(data, config['value_col'], norm)
                norm_figs.append(fig_norm)
                fig_norm_bytes_list.append(fig_to_bytes(fig_norm))
                
                is_normal = norm['normal']
                
                if is_normal:
                    test_result = one_sample_t_test(data, mu0, alph, alt)
                else:
                    test_result = wilcoxon_signed_rank_one_sample(data, mu0, alph, alt)
                
                fig_test = plot_one_sample_results(data, test_result, mu0)
                
                descriptives = {
                    'n': len(data),
                    'Mean': f'{np.mean(data):.4f}',
                    'Std Dev': f'{np.std(data, ddof=1):.4f}',
                    'Median': f'{np.median(data):.4f}',
                    'Min': f'{np.min(data):.4f}',
                    'Max': f'{np.max(data):.4f}',
                    'Skewness': f'{stats.skew(data):.4f}',
                    'Kurtosis': f'{stats.kurtosis(data):.4f}',
                }

            # ── PAIRED ──────────────────────────
            elif "Paired" in ttype:
                g1 = df[config['pre_col']].dropna().values
                g2 = df[config['post_col']].dropna().values
                n_min = min(len(g1), len(g2))
                g1, g2 = g1[:n_min], g2[:n_min]
                diffs = g1 - g2
                
                norm = normality_test(diffs, alph)
                assumption_results['Normality of Differences'] = norm
                
                fig_norm = plot_normality_diagnostics(diffs, 'Differences (Pre − Post)', norm)
                norm_figs.append(fig_norm)
                fig_norm_bytes_list.append(fig_to_bytes(fig_norm))
                
                is_normal = norm['normal']
                
                if is_normal:
                    test_result = paired_t_test(g1, g2, alph, alt)
                else:
                    test_result = wilcoxon_signed_rank_paired(g1, g2, alph, alt)
                
                fig_test = plot_paired_results(g1, g2, test_result,
                                               config['pre_col'], config['post_col'])
                
                descriptives = {
                    'n (pairs)': n_min,
                    f'Mean ({config["pre_col"]})': f'{np.mean(g1):.4f}',
                    f'Mean ({config["post_col"]})': f'{np.mean(g2):.4f}',
                    'Mean Difference': f'{np.mean(diffs):.4f}',
                    'Std Diff': f'{np.std(diffs, ddof=1):.4f}',
                    'Median Diff': f'{np.median(diffs):.4f}',
                }

            # ── INDEPENDENT ─────────────────────
            else:
                value_col = config['value_col']
                group_col = config['group_col']
                groups = df[group_col].unique()
                
                if len(groups) != 2:
                    st.error(f"Independent samples test requires exactly 2 groups. Found: {len(groups)}")
                    st.stop()
                
                g1 = df[df[group_col] == groups[0]][value_col].dropna().values
                g2 = df[df[group_col] == groups[1]][value_col].dropna().values
                label1, label2 = str(groups[0]), str(groups[1])
                
                norm1 = normality_test(g1, alph)
                norm2 = normality_test(g2, alph)
                assumption_results[f'Normality — {label1}'] = norm1
                assumption_results[f'Normality — {label2}'] = norm2
                
                fig_n1 = plot_normality_diagnostics(g1, f'{value_col} [{label1}]', norm1)
                fig_n2 = plot_normality_diagnostics(g2, f'{value_col} [{label2}]', norm2)
                norm_figs.extend([fig_n1, fig_n2])
                fig_norm_bytes_list.extend([fig_to_bytes(fig_n1), fig_to_bytes(fig_n2)])
                
                both_normal = norm1['normal'] and norm2['normal']
                
                if both_normal:
                    lev = levene_test(g1, g2, alph)
                    assumption_results["Levene's Test (Equality of Variances)"] = lev
                    equal_var = lev['equal_variance']
                    test_result = independent_t_test(g1, g2, alph, alt, equal_var)
                else:
                    test_result = mann_whitney_u_test(g1, g2, alph, alt)
                
                fig_test = plot_independent_results(g1, g2, test_result, label1, label2)
                
                descriptives = {
                    f'n ({label1})': len(g1),
                    f'n ({label2})': len(g2),
                    f'Mean ({label1})': f'{np.mean(g1):.4f}',
                    f'Mean ({label2})': f'{np.mean(g2):.4f}',
                    f'Std ({label1})': f'{np.std(g1, ddof=1):.4f}',
                    f'Std ({label2})': f'{np.std(g2, ddof=1):.4f}',
                }

            st.session_state['assumption_results'] = assumption_results
            st.session_state['test_result'] = test_result
            st.session_state['norm_figs'] = norm_figs
            st.session_state['fig_test'] = fig_test
            st.session_state['fig_test_bytes'] = fig_to_bytes(fig_test)
            st.session_state['fig_norm_bytes_list'] = fig_norm_bytes_list
            st.session_state['descriptives'] = descriptives

        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    # ── TAB 2: ASSUMPTIONS ───────────────────
    with tab2:
        if 'assumption_results' not in st.session_state:
            st.markdown('<div class="info-box">Run the analysis from the Data Input tab first.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-header">Prerequisite Assumption Tests</div>',
                        unsafe_allow_html=True)
            
            for name, res in st.session_state['assumption_results'].items():
                with st.expander(f"**{name}**", expanded=True):
                    is_ok = res.get('normal', res.get('equal_variance', True))
                    badge = f'<span class="badge-pass">PASSED</span>' if is_ok else f'<span class="badge-fail">FAILED</span>'
                    
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f'<div class="stat-card"><h4>Test</h4><div class="value" style="font-size:1rem">{res["test"]}</div></div>',
                                    unsafe_allow_html=True)
                    with c2:
                        stat_key = 'statistic'
                        st.markdown(f'<div class="stat-card"><h4>Statistic</h4><div class="value">{res[stat_key]:.4f}</div></div>',
                                    unsafe_allow_html=True)
                    with c3:
                        st.markdown(f'<div class="stat-card"><h4>p-value</h4><div class="value">{res["p_value"]:.4f}</div><div class="label">Result: {res.get("interpretation", "")}</div></div>',
                                    unsafe_allow_html=True)
                    
                    st.markdown(f'<div class="info-box">{badge} &nbsp;&nbsp;{res.get("interpretation", "")} — {"Parametric test applicable" if is_ok else "Non-parametric equivalent will be applied"}</div>',
                                unsafe_allow_html=True)
            
            # Normality plots
            if 'norm_figs' in st.session_state:
                st.markdown('<div class="section-header">Normality Diagnostic Plots</div>',
                            unsafe_allow_html=True)
                for fig in st.session_state['norm_figs']:
                    st.pyplot(fig, use_container_width=True)

    # ── TAB 3: RESULTS ───────────────────────
    with tab3:
        if 'test_result' not in st.session_state:
            st.markdown('<div class="info-box">Run the analysis from the Data Input tab first.</div>',
                        unsafe_allow_html=True)
        else:
            result = st.session_state['test_result']
            
            st.markdown(f'<div class="section-header">{result["test"]}</div>', unsafe_allow_html=True)
            
            # Decision
            reject = result['reject']
            verdict_class = 'verdict-reject' if reject else 'verdict-fail'
            verdict_text = 'REJECT H₀' if reject else 'FAIL TO REJECT H₀'
            verdict_color = 'reject-color' if reject else 'fail-color'
            st.markdown(f"""
<div class="verdict-box {verdict_class}">
    <div class="verdict-label">Decision at α = {st.session_state.get('alpha', 0.05)}</div>
    <div class="verdict-text <{verdict_color}">{verdict_text}</div>
</div>""", unsafe_allow_html=True)
            
            # Key metrics
            c1, c2, c3, c4 = st.columns(4)
            stat_label = 't' if 't-Test' in result['test'] else ('W' if 'Wilcoxon' in result['test'] else 'U')
            with c1:
                st.markdown(f'<div class="stat-card"><h4>Test Statistic ({stat_label})</h4><div class="value">{result["statistic"]:.4f}</div></div>',
                            unsafe_allow_html=True)
            with c2:
                p_display = f'{result["p_value"]:.4f}' if result['p_value'] >= 0.0001 else '< 0.0001'
                st.markdown(f'<div class="stat-card"><h4>p-value</h4><div class="value">{p_display}</div></div>',
                            unsafe_allow_html=True)
            with c3:
                if 'df' in result:
                    st.markdown(f'<div class="stat-card"><h4>Degrees of Freedom</h4><div class="value">{result["df"]:.2f}</div></div>',
                                unsafe_allow_html=True)
                else:
                    n_eff = result.get('n_effective', result.get('n', '—'))
                    st.markdown(f'<div class="stat-card"><h4>Effective n</h4><div class="value">{n_eff}</div></div>',
                                unsafe_allow_html=True)
            with c4:
                eff_interp = interpret_effect_size(result['effect_size'], result['effect_label'])
                st.markdown(f'<div class="stat-card"><h4>{result["effect_label"]}</h4><div class="value">{result["effect_size"]:.4f}</div><div class="label">{eff_interp}</div></div>',
                            unsafe_allow_html=True)
            
            # CI if available
            if 'ci' in result:
                st.markdown(f'<div class="info-box">95% Confidence Interval: <b>[{result["ci"][0]:.4f}, {result["ci"][1]:.4f}]</b></div>',
                            unsafe_allow_html=True)
            
            # Test figure
            st.markdown('<div class="section-header">Visualisation</div>', unsafe_allow_html=True)
            st.pyplot(st.session_state['fig_test'], use_container_width=True)
            
            # Descriptive stats table
            st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
            desc = st.session_state['descriptives']
            desc_df = pd.DataFrame(list(desc.items()), columns=['Statistic', 'Value'])
            st.dataframe(desc_df, use_container_width=True, hide_index=True)
            
            # Full result table
            st.markdown('<div class="section-header">Complete Test Output</div>', unsafe_allow_html=True)
            out_rows = {}
            for k, v in result.items():
                if k == 'differences':
                    continue
                if k == 'ci':
                    out_rows['CI Lower'] = f'{v[0]:.6f}'
                    out_rows['CI Upper'] = f'{v[1]:.6f}'
                elif isinstance(v, bool):
                    out_rows[k] = str(v)
                elif isinstance(v, float):
                    out_rows[k] = f'{v:.6f}'
                else:
                    out_rows[k] = str(v)
            full_df = pd.DataFrame(list(out_rows.items()), columns=['Parameter', 'Value'])
            st.dataframe(full_df, use_container_width=True, hide_index=True)
            
            # Downloads
            st.markdown('<div class="section-header">Download Results</div>', unsafe_allow_html=True)
            dl1, dl2, dl3 = st.columns(3)
            
            # PNG download
            with dl1:
                st.download_button(
                    "⬇  Download Plot (PNG)",
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
