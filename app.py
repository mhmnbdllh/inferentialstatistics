"""
Inferential Statistics App — SPSS-Equivalent Output
====================================================
Parametric & Non-parametric tests with automatic selection via normality testing.
Covers: One-Sample, Paired-Sample, Independent-Sample T-Tests
         + Wilcoxon Signed-Rank, Mann-Whitney U (non-parametric equivalents)
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from io import BytesIO
import io
import warnings
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                 Paragraph, Spacer, HRFlowable, Image, PageBreak)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Inferential Statistics", page_icon="📐",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
.main-hdr{background:linear-gradient(120deg,#0a0a0a 0%,#1a1a2e 40%,#16213e 100%);
  padding:2rem 2.4rem;border-radius:14px;margin-bottom:1.8rem;
  border-left:5px solid #e94560;box-shadow:0 8px 32px rgba(233,69,96,.15);}
.main-hdr h1{color:#fff;font-size:2rem;font-weight:700;margin:0 0 .4rem 0;letter-spacing:-.5px;}
.main-hdr p{color:#94a3b8;margin:0;font-size:.92rem;}
.main-hdr .badge{display:inline-block;background:#e94560;color:#fff;font-size:.7rem;
  padding:2px 8px;border-radius:20px;margin-left:8px;font-weight:600;vertical-align:middle;}
.sec-title{background:linear-gradient(90deg,#1a1a2e,#16213e);color:#e2e8f0;
  padding:9px 16px;border-radius:6px 6px 0 0;font-weight:600;font-size:.84rem;
  letter-spacing:.6px;margin-top:1.4rem;font-family:'DM Mono',monospace;
  border-bottom:2px solid #e94560;}
.spss-wrap{overflow-x:auto;margin-bottom:.4rem;}
.spss-tbl{font-family:'DM Mono',monospace;font-size:.77rem;border-collapse:collapse;width:100%;min-width:400px;}
.spss-tbl th{background:#1a1a2e;color:#e2e8f0;padding:7px 12px;text-align:center;
  font-weight:600;border:1px solid #334155;font-size:.74rem;white-space:nowrap;}
.spss-tbl td{padding:5px 12px;border:1px solid #e2e8f0;text-align:right;
  color:#1e293b;white-space:nowrap;background:#fff;}
.spss-tbl tr:nth-child(even) td{background:#f8fafc;}
.spss-tbl td:first-child,.spss-tbl td:nth-child(2){text-align:left;
  font-weight:500;background:#f1f5f9!important;}
.test-card{background:linear-gradient(135deg,#f0f9ff,#e0f2fe);
  border:1px solid #bae6fd;border-radius:12px;padding:1.2rem 1.4rem;margin:.8rem 0;
  border-left:4px solid #0284c7;}
.test-card.parametric{background:linear-gradient(135deg,#f0fdf4,#dcfce7);
  border-color:#bbf7d0;border-left-color:#16a34a;}
.test-card.nonparametric{background:linear-gradient(135deg,#fff7ed,#ffedd5);
  border-color:#fed7aa;border-left-color:#ea580c;}
.test-card.auto-select{background:linear-gradient(135deg,#faf5ff,#ede9fe);
  border-color:#ddd6fe;border-left-color:#7c3aed;}
.test-card h4{margin:0 0 .4rem 0;font-size:.95rem;font-weight:700;}
.test-card p{margin:0;font-size:.84rem;color:#475569;line-height:1.5;}
.interp-box{background:linear-gradient(135deg,#f8fafc,#f1f5f9);
  border-left:4px solid #0284c7;padding:1rem 1.2rem;border-radius:0 10px 10px 0;
  margin:.7rem 0;font-size:.87rem;line-height:1.8;color:#1e293b;}
.interp-box b{color:#0284c7;}
.interp-box.sig{border-left-color:#16a34a;background:linear-gradient(135deg,#f0fdf4,#dcfce7);}
.interp-box.sig b{color:#16a34a;}
.interp-box.nonsig{border-left-color:#dc2626;background:linear-gradient(135deg,#fef2f2,#fee2e2);}
.interp-box.nonsig b{color:#dc2626;}
.metric-card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;
  padding:.9rem;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.05);}
.metric-val{font-size:1.45rem;font-weight:700;color:#1a1a2e;font-family:'DM Mono',monospace;}
.metric-lbl{font-size:.7rem;color:#64748b;margin-top:.25rem;
  text-transform:uppercase;letter-spacing:.6px;}
.pass{color:#16a34a;font-weight:700;} .fail{color:#dc2626;font-weight:700;}
.warn-box{background:#fffbeb;border-left:4px solid #f59e0b;padding:.7rem 1rem;
  border-radius:0 6px 6px 0;font-size:.82rem;color:#92400e;margin:.4rem 0;}
.info-box{background:#eff6ff;border-left:4px solid #3b82f6;padding:.7rem 1rem;
  border-radius:0 6px 6px 0;font-size:.82rem;color:#1e40af;margin:.4rem 0;}
.note-txt{font-size:.74rem;color:#64748b;font-style:italic;margin-top:.3rem;}
.decision-banner{padding:1rem 1.4rem;border-radius:10px;margin:1rem 0;font-size:.9rem;font-weight:600;}
.decision-banner.use-param{background:#dcfce7;color:#14532d;border:1px solid #86efac;}
.decision-banner.use-nonparam{background:#ffedd5;color:#7c2d12;border:1px solid #fdba74;}
</style>
""", unsafe_allow_html=True)

# ── Sample data templates ──────────────────────────────────────────────────────
SAMPLES = {
    "One-Sample T-Test": {
        "csv": """subject_id,score
1,78
2,85
3,72
4,90
5,68
6,88
7,75
8,82
9,79
10,93
11,71
12,84
13,76
14,89
15,80""",
        "desc": "Test whether the mean **score** differs from a known population value (μ₀). Select the `score` column as variable.",
        "cols": ["subject_id", "score"],
        "note": "One numeric column required. Set your test value (μ₀) in the sidebar."
    },
    "Paired-Sample T-Test": {
        "csv": """subject_id,pre_score,post_score
1,65,72
2,70,78
3,58,65
4,75,80
5,62,70
6,68,74
7,72,79
8,60,68
9,74,81
10,66,73
11,71,76
12,63,69
13,69,75
14,73,80
15,67,72""",
        "desc": "Test whether the mean difference between **pre_score** and **post_score** is zero (before vs. after).",
        "cols": ["subject_id", "pre_score", "post_score"],
        "note": "Two numeric columns required: select Variable 1 (pre) and Variable 2 (post)."
    },
    "Independent-Sample T-Test": {
        "csv": """subject_id,group,score
1,Control,58
2,Control,62
3,Control,55
4,Control,67
5,Control,60
6,Control,63
7,Control,57
8,Control,65
9,Treatment,72
10,Treatment,78
11,Treatment,70
12,Treatment,81
13,Treatment,75
14,Treatment,77
15,Treatment,73
16,Treatment,79""",
        "desc": "Test whether **score** means differ between **Control** and **Treatment** groups.",
        "cols": ["subject_id", "group", "score"],
        "note": "One grouping column (categorical) and one numeric outcome column required."
    }
}

# ── Utility helpers ────────────────────────────────────────────────────────────
def _f(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "."
    return f"{v:.{d}f}"

def _p(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "."
    return ".000" if v < .001 else f"{v:.3f}"

def _p2(v):
    """Two-tailed p formatted SPSS style (show < .001 as .000)"""
    return _p(v)

def cohens_d_onesample(data, mu0):
    return (np.mean(data) - mu0) / np.std(data, ddof=1)

def cohens_d_paired(d):
    return np.mean(d) / np.std(d, ddof=1)

def cohens_d_independent(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1-1)*np.var(g1,ddof=1) + (n2-1)*np.var(g2,ddof=1)) / (n1+n2-2))
    return (np.mean(g1) - np.mean(g2)) / pooled if pooled else np.nan

def effect_label(d):
    ad = abs(d)
    if ad < .2: return "negligible"
    if ad < .5: return "small"
    if ad < .8: return "medium"
    return "large"

def r_from_wilcoxon(W, n):
    """Effect size r for Wilcoxon from z-score approximation"""
    mu_w = n*(n+1)/4
    sigma_w = np.sqrt(n*(n+1)*(2*n+1)/24)
    z = (W - mu_w) / sigma_w
    return abs(z) / np.sqrt(n)

def r_from_mannwhitney(U, n1, n2):
    """Effect size r for Mann-Whitney"""
    mu_u = n1*n2/2
    sigma_u = np.sqrt(n1*n2*(n1+n2+1)/12)
    z = (U - mu_u) / sigma_u
    return abs(z) / np.sqrt(n1+n2)

def r_label(r):
    if r < .1: return "negligible"
    if r < .3: return "small"
    if r < .5: return "medium"
    return "large"

def ci_mean(data, alpha=0.05):
    n = len(data); m = np.mean(data); se = stats.sem(data)
    t_c = stats.t.ppf(1-alpha/2, n-1)
    return m - t_c*se, m + t_c*se

def normality_decision(p_val, alpha=0.05):
    return p_val > alpha

# ── Normality test ─────────────────────────────────────────────────────────────
def test_normality(data, label=""):
    """Run Shapiro-Wilk. Return dict with stat, p, pass."""
    if len(data) < 3:
        return {"label": label, "W": np.nan, "p": np.nan, "pass": False, "n": len(data)}
    W, p = stats.shapiro(data)
    return {"label": label, "W": float(W), "p": float(p),
            "pass": float(p) > 0.05, "n": len(data)}

# ══════════════════════════════════════════════════════════════════════════════
# ONE-SAMPLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def run_one_sample(data, mu0, alpha=0.05):
    R = {}
    n = len(data); m = np.mean(data); sd = np.std(data, ddof=1); se = sd/np.sqrt(n)

    # Descriptives
    ci_l, ci_u = ci_mean(data, alpha)
    R["desc"] = pd.DataFrame([{
        "N": n, "Mean": m, "Std. Deviation": sd, "Std. Error Mean": se,
        f"{int((1-alpha)*100)}% CI Lower": ci_l,
        f"{int((1-alpha)*100)}% CI Upper": ci_u,
        "Minimum": np.min(data), "Maximum": np.max(data),
        "Skewness": float(stats.skew(data)), "Kurtosis": float(stats.kurtosis(data))
    }])

    # Normality
    norm = test_normality(data, "Variable")
    R["normality"] = [norm]
    use_param = norm["pass"]
    R["use_param"] = use_param

    # ── Parametric: One-sample t-test ─────────────────────────────────────────
    t_stat, p_two = stats.ttest_1samp(data, mu0)
    p_one_l = stats.t.cdf(t_stat, n-1)
    p_one_u = 1 - stats.t.cdf(t_stat, n-1)
    diff_mean = m - mu0
    t_crit = stats.t.ppf(1-alpha/2, n-1)
    ci_diff_l = diff_mean - t_crit*se
    ci_diff_u = diff_mean + t_crit*se
    d = cohens_d_onesample(data, mu0)
    R["parametric"] = {
        "test": "One-Sample T-Test",
        "t": float(t_stat), "df": n-1,
        "p_two": float(p_two), "p_one_l": float(p_one_l), "p_one_u": float(p_one_u),
        "mean_diff": diff_mean,
        "ci_lower": ci_diff_l, "ci_upper": ci_diff_u,
        "cohens_d": d, "mu0": mu0
    }

    # ── Non-parametric: Wilcoxon Signed-Rank ──────────────────────────────────
    diffs = np.array(data) - mu0
    diffs_nonzero = diffs[diffs != 0]
    if len(diffs_nonzero) >= 1:
        W_stat, p_wilc = stats.wilcoxon(diffs_nonzero)
        r_eff = r_from_wilcoxon(W_stat, len(diffs_nonzero))
    else:
        W_stat, p_wilc, r_eff = np.nan, np.nan, np.nan
    R["nonparametric"] = {
        "test": "Wilcoxon Signed-Rank Test",
        "W": float(W_stat) if not np.isnan(W_stat) else np.nan,
        "n_nonzero": len(diffs_nonzero),
        "p": float(p_wilc) if not np.isnan(p_wilc) else np.nan,
        "r_effect": r_eff, "mu0": mu0
    }
    return R

# ══════════════════════════════════════════════════════════════════════════════
# PAIRED-SAMPLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def run_paired(data1, data2, label1="Var1", label2="Var2", alpha=0.05):
    R = {}
    n = len(data1)
    diff = np.array(data1) - np.array(data2)
    m1, m2 = np.mean(data1), np.mean(data2)
    sd1, sd2 = np.std(data1,ddof=1), np.std(data2,ddof=1)
    se1, se2 = sd1/np.sqrt(n), sd2/np.sqrt(n)
    m_diff = np.mean(diff); sd_diff = np.std(diff,ddof=1); se_diff = sd_diff/np.sqrt(n)

    # Correlation between variables
    r_corr, p_corr = stats.pearsonr(data1, data2)

    # Descriptives for each variable
    ci1_l, ci1_u = ci_mean(data1, alpha)
    ci2_l, ci2_u = ci_mean(data2, alpha)
    R["desc"] = pd.DataFrame([
        {"Variable": label1, "N": n, "Mean": m1, "Std. Deviation": sd1,
         "Std. Error Mean": se1,
         f"{int((1-alpha)*100)}% CI Lower": ci1_l, f"{int((1-alpha)*100)}% CI Upper": ci1_u},
        {"Variable": label2, "N": n, "Mean": m2, "Std. Deviation": sd2,
         "Std. Error Mean": se2,
         f"{int((1-alpha)*100)}% CI Lower": ci2_l, f"{int((1-alpha)*100)}% CI Upper": ci2_u},
    ])

    # Paired correlation table (SPSS style)
    R["correlation"] = pd.DataFrame([{
        "Variable 1": label1, "Variable 2": label2,
        "N": n, "Pearson r": r_corr, "Sig. (2-tailed)": p_corr
    }])

    # Normality on differences
    norm = test_normality(diff, "Differences")
    R["normality"] = [norm]
    R["use_param"] = norm["pass"]

    # ── Parametric: Paired t-test ──────────────────────────────────────────────
    t_stat, p_two = stats.ttest_rel(data1, data2)
    t_crit = stats.t.ppf(1-alpha/2, n-1)
    ci_diff_l = m_diff - t_crit*se_diff
    ci_diff_u = m_diff + t_crit*se_diff
    d = cohens_d_paired(diff)
    R["parametric"] = {
        "test": "Paired Samples T-Test",
        "label1": label1, "label2": label2,
        "mean_diff": m_diff, "sd_diff": sd_diff, "se_diff": se_diff,
        "t": float(t_stat), "df": n-1,
        "p_two": float(p_two),
        "p_one_l": float(stats.t.cdf(t_stat, n-1)),
        "p_one_u": float(1-stats.t.cdf(t_stat, n-1)),
        "ci_lower": ci_diff_l, "ci_upper": ci_diff_u,
        "cohens_d": d
    }

    # ── Non-parametric: Wilcoxon Signed-Rank ──────────────────────────────────
    diff_nonzero = diff[diff != 0]
    if len(diff_nonzero) >= 1:
        W_stat, p_wilc = stats.wilcoxon(diff_nonzero)
        r_eff = r_from_wilcoxon(W_stat, len(diff_nonzero))
        pos_ranks = np.sum(diff_nonzero > 0)
        neg_ranks = np.sum(diff_nonzero < 0)
    else:
        W_stat=p_wilc=r_eff=pos_ranks=neg_ranks=np.nan
    R["nonparametric"] = {
        "test": "Wilcoxon Signed-Rank Test",
        "W": float(W_stat) if not np.isnan(W_stat) else np.nan,
        "n": n, "n_nonzero": len(diff_nonzero),
        "pos_ranks": pos_ranks, "neg_ranks": neg_ranks,
        "p": float(p_wilc) if not np.isnan(p_wilc) else np.nan,
        "r_effect": r_eff,
        "label1": label1, "label2": label2
    }
    return R

# ══════════════════════════════════════════════════════════════════════════════
# INDEPENDENT-SAMPLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def run_independent(g1, g2, label1="Group 1", label2="Group 2", dep_var="Score", alpha=0.05):
    R = {}
    n1, n2 = len(g1), len(g2)
    m1, m2 = np.mean(g1), np.mean(g2)
    sd1, sd2 = np.std(g1,ddof=1), np.std(g2,ddof=1)
    se1, se2 = sd1/np.sqrt(n1), sd2/np.sqrt(n2)
    ci1_l,ci1_u = ci_mean(g1,alpha); ci2_l,ci2_u = ci_mean(g2,alpha)

    R["desc"] = pd.DataFrame([
        {"Group": label1, "N": n1, "Mean": m1, "Std. Deviation": sd1,
         "Std. Error Mean": se1,
         f"{int((1-alpha)*100)}% CI Lower": ci1_l, f"{int((1-alpha)*100)}% CI Upper": ci1_u,
         "Minimum": np.min(g1), "Maximum": np.max(g1),
         "Skewness": float(stats.skew(g1)), "Kurtosis": float(stats.kurtosis(g1))},
        {"Group": label2, "N": n2, "Mean": m2, "Std. Deviation": sd2,
         "Std. Error Mean": se2,
         f"{int((1-alpha)*100)}% CI Lower": ci2_l, f"{int((1-alpha)*100)}% CI Upper": ci2_u,
         "Minimum": np.min(g2), "Maximum": np.max(g2),
         "Skewness": float(stats.skew(g2)), "Kurtosis": float(stats.kurtosis(g2))},
    ])

    # Normality per group
    n1_res = test_normality(g1, label1)
    n2_res = test_normality(g2, label2)
    R["normality"] = [n1_res, n2_res]
    R["use_param"] = n1_res["pass"] and n2_res["pass"]

    # Levene's test
    lev_f, lev_p = stats.levene(g1, g2)
    R["levene"] = {"F": float(lev_f), "df1": 1, "df2": n1+n2-2,
                   "Sig.": float(lev_p), "equal_var": float(lev_p) > alpha}

    # ── Parametric: Independent t-test (equal + Welch) ────────────────────────
    t_eq, p_eq   = stats.ttest_ind(g1, g2, equal_var=True)
    t_welch, p_welch = stats.ttest_ind(g1, g2, equal_var=False)
    # Welch df (Satterthwaite)
    df_welch = (sd1**2/n1 + sd2**2/n2)**2 / ((sd1**2/n1)**2/(n1-1) + (sd2**2/n2)**2/(n2-1))
    mean_diff = m1 - m2
    se_eq    = np.sqrt(((n1-1)*sd1**2+(n2-1)*sd2**2)/(n1+n2-2) * (1/n1+1/n2))
    se_welch = np.sqrt(sd1**2/n1 + sd2**2/n2)
    t_crit_eq    = stats.t.ppf(1-alpha/2, n1+n2-2)
    t_crit_welch = stats.t.ppf(1-alpha/2, df_welch)
    d = cohens_d_independent(g1, g2)
    R["parametric"] = {
        "test": "Independent Samples T-Test",
        "label1": label1, "label2": label2, "dep_var": dep_var,
        "mean_diff": mean_diff,
        # Equal variances assumed
        "t_eq": float(t_eq), "df_eq": n1+n2-2, "p_eq": float(p_eq),
        "se_eq": se_eq,
        "ci_eq_l": mean_diff - t_crit_eq*se_eq,
        "ci_eq_u": mean_diff + t_crit_eq*se_eq,
        # Welch (equal variances not assumed)
        "t_welch": float(t_welch), "df_welch": df_welch, "p_welch": float(p_welch),
        "se_welch": se_welch,
        "ci_welch_l": mean_diff - t_crit_welch*se_welch,
        "ci_welch_u": mean_diff + t_crit_welch*se_welch,
        "cohens_d": d
    }

    # ── Non-parametric: Mann-Whitney U ────────────────────────────────────────
    U_stat, p_mw = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    r_eff = r_from_mannwhitney(U_stat, n1, n2)
    # U2 (complement)
    U2 = n1*n2 - U_stat
    R["nonparametric"] = {
        "test": "Mann-Whitney U Test",
        "U": float(U_stat), "U2": float(U2),
        "n1": n1, "n2": n2,
        "p": float(p_mw), "r_effect": r_eff,
        "label1": label1, "label2": label2
    }
    return R

# ══════════════════════════════════════════════════════════════════════════════
# INTERPRETATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def interpret_one_sample(R, var_name, alpha):
    lines = []
    use_p = R["use_param"]
    norm  = R["normality"][0]
    mu0   = R["parametric"]["mu0"]

    lines.append(
        f"<b>Normality Assessment:</b> The Shapiro-Wilk test on <i>{var_name}</i> "
        f"(W = {norm['W']:.3f}, p = {_p(norm['p'])}) indicated that the data "
        f"{'were approximately normally distributed (p > .05), supporting use of parametric analysis' if norm['pass'] else 'deviated significantly from normality (p ≤ .05), warranting non-parametric analysis'}."
    )

    if use_p:
        pr = R["parametric"]
        sig = pr["p_two"] < alpha
        lines.append(
            f"<b>One-Sample T-Test:</b> A one-sample t-test was conducted to determine whether "
            f"<i>{var_name}</i> differed significantly from the test value of μ₀ = {mu0}. "
            f"Results {'indicated a statistically significant difference' if sig else 'did not indicate a statistically significant difference'}, "
            f"t({pr['df']}) = {pr['t']:.3f}, p {'< .001' if pr['p_two'] < .001 else '= ' + _p(pr['p_two'])} (two-tailed). "
            f"The mean difference was {pr['mean_diff']:.3f} "
            f"(95% CI [{pr['ci_lower']:.3f}, {pr['ci_upper']:.3f}]), "
            f"with a {'significant' if sig else 'non-significant'} {effect_label(pr['cohens_d'])} effect size "
            f"(Cohen's d = {pr['cohens_d']:.3f})."
        )
    else:
        np_r = R["nonparametric"]
        sig = np_r["p"] < alpha
        lines.append(
            f"<b>Wilcoxon Signed-Rank Test:</b> Due to violation of normality, a Wilcoxon signed-rank test "
            f"was performed to test whether the median of <i>{var_name}</i> differed from {mu0}. "
            f"{'The test was statistically significant' if sig else 'The test was not statistically significant'}, "
            f"W = {np_r['W']:.0f}, p {'< .001' if np_r['p'] < .001 else '= ' + _p(np_r['p'])} (two-tailed), "
            f"with a {r_label(np_r['r_effect'])} effect size (r = {np_r['r_effect']:.3f})."
        )
    return lines

def interpret_paired(R, alpha):
    lines = []
    use_p = R["use_param"]
    norm  = R["normality"][0]
    pr    = R["parametric"]

    lines.append(
        f"<b>Normality of Differences:</b> The Shapiro-Wilk test on the paired differences "
        f"(W = {norm['W']:.3f}, p = {_p(norm['p'])}) indicated that differences "
        f"{'were approximately normally distributed (p > .05)' if norm['pass'] else 'were not normally distributed (p ≤ .05)'}. "
        f"{'Parametric analysis is appropriate.' if norm['pass'] else 'Non-parametric analysis is recommended.'}"
    )

    corr_r = R["correlation"]["Pearson r"].iloc[0]
    corr_p = R["correlation"]["Sig. (2-tailed)"].iloc[0]
    lines.append(
        f"<b>Paired Correlation:</b> <i>{pr['label1']}</i> and <i>{pr['label2']}</i> were "
        f"{'significantly' if corr_p < alpha else 'not significantly'} correlated, "
        f"r({pr['df']}) = {corr_r:.3f}, p {'< .001' if corr_p < .001 else '= ' + _p(corr_p)}. "
        f"This {'justifies the paired design' if abs(corr_r) > .3 else 'suggests a weak relationship between measurements'}."
    )

    if use_p:
        sig = pr["p_two"] < alpha
        lines.append(
            f"<b>Paired T-Test:</b> A paired-samples t-test examined whether <i>{pr['label1']}</i> "
            f"and <i>{pr['label2']}</i> differed significantly. "
            f"{'A statistically significant difference was found' if sig else 'No statistically significant difference was found'}, "
            f"t({pr['df']}) = {pr['t']:.3f}, p {'< .001' if pr['p_two'] < .001 else '= ' + _p(pr['p_two'])} (two-tailed). "
            f"The mean difference was {pr['mean_diff']:.3f} (SD = {pr['sd_diff']:.3f}), "
            f"95% CI [{pr['ci_lower']:.3f}, {pr['ci_upper']:.3f}], "
            f"Cohen's d = {pr['cohens_d']:.3f} ({effect_label(pr['cohens_d'])} effect)."
        )
    else:
        np_r = R["nonparametric"]
        sig  = np_r["p"] < alpha
        lines.append(
            f"<b>Wilcoxon Signed-Rank Test:</b> Due to non-normal differences, a Wilcoxon signed-rank test "
            f"was used. {'A significant difference was found' if sig else 'No significant difference was found'}, "
            f"W = {np_r['W']:.0f}, p {'< .001' if np_r['p'] < .001 else '= ' + _p(np_r['p'])} (two-tailed), "
            f"r = {np_r['r_effect']:.3f} ({r_label(np_r['r_effect'])} effect)."
        )
    return lines

def interpret_independent(R, dep_var, alpha):
    lines = []
    use_p = R["use_param"]
    norms = R["normality"]
    lev   = R["levene"]
    pr    = R["parametric"]

    norm_txt = "; ".join(
        [f"{n['label']}: W = {n['W']:.3f}, p = {_p(n['p'])}" for n in norms])
    lines.append(
        f"<b>Normality Assessment:</b> Shapiro-Wilk tests were conducted for each group ({norm_txt}). "
        f"{'Both groups showed approximately normal distributions, supporting parametric analysis.' if use_p else 'At least one group deviated from normality; non-parametric analysis is recommended.'}"
    )

    lines.append(
        f"<b>Homogeneity of Variance (Levene's Test):</b> Levene's test "
        f"{'indicated equal variances' if lev['equal_var'] else 'indicated unequal variances'} "
        f"across groups, F({lev['df1']}, {lev['df2']}) = {lev['F']:.3f}, p = {_p(lev['Sig.'])}. "
        f"{'Equal variances assumed.' if lev['equal_var'] else 'Welch correction applied (equal variances not assumed).'}"
    )

    if use_p:
        t_v = pr['t_eq'] if lev['equal_var'] else pr['t_welch']
        p_v = pr['p_eq'] if lev['equal_var'] else pr['p_welch']
        df_v= pr['df_eq'] if lev['equal_var'] else pr['df_welch']
        ci_l= pr['ci_eq_l'] if lev['equal_var'] else pr['ci_welch_l']
        ci_u= pr['ci_eq_u'] if lev['equal_var'] else pr['ci_welch_u']
        sig  = p_v < alpha
        assumption = "equal variances assumed" if lev['equal_var'] else "equal variances not assumed (Welch)"
        lines.append(
            f"<b>Independent T-Test ({assumption}):</b> "
            f"{'A statistically significant difference' if sig else 'No statistically significant difference'} "
            f"was found in <i>{dep_var}</i> between {pr['label1']} and {pr['label2']}, "
            f"t({_f(df_v,2)}) = {t_v:.3f}, p {'< .001' if p_v < .001 else '= ' + _p(p_v)} (two-tailed). "
            f"The mean difference was {pr['mean_diff']:.3f} (95% CI [{ci_l:.3f}, {ci_u:.3f}]), "
            f"Cohen's d = {pr['cohens_d']:.3f} ({effect_label(pr['cohens_d'])} effect)."
        )
    else:
        np_r = R["nonparametric"]
        sig  = np_r["p"] < alpha
        lines.append(
            f"<b>Mann-Whitney U Test:</b> A Mann-Whitney U test compared <i>{dep_var}</i> "
            f"between {np_r['label1']} and {np_r['label2']}. "
            f"{'A statistically significant difference was found' if sig else 'No significant difference was found'}, "
            f"U = {np_r['U']:.0f}, p {'< .001' if np_r['p'] < .001 else '= ' + _p(np_r['p'])} (two-tailed), "
            f"r = {np_r['r_effect']:.3f} ({r_label(np_r['r_effect'])} effect)."
        )
    return lines

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
PAL = ["#1a1a2e","#e94560","#0f3460","#16213e","#533483","#0284c7"]

def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def plot_one_sample(data, mu0, var_name):
    fig = plt.figure(figsize=(14, 4), facecolor="#f8fafc")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=.35)

    # Histogram + normal curve
    ax1 = fig.add_subplot(gs[0]); ax1.set_facecolor("#f8fafc")
    ax1.hist(data, bins="auto", color=PAL[1], alpha=0.75, edgecolor="white", linewidth=0.8)
    x = np.linspace(min(data)-.5, max(data)+.5, 200)
    scale = len(data)*(max(data)-min(data))/max(len(data)//3,1)
    ax1.plot(x, stats.norm.pdf(x,np.mean(data),np.std(data,ddof=1))*scale,
             color=PAL[0], linewidth=2)
    ax1.axvline(mu0, color="#e94560", linestyle="--", linewidth=1.8, label=f"μ₀={mu0}")
    ax1.axvline(np.mean(data), color=PAL[2], linestyle="-", linewidth=1.8, label=f"x̄={np.mean(data):.2f}")
    ax1.set_xlabel(var_name, fontsize=8); ax1.set_ylabel("Frequency", fontsize=8)
    ax1.set_title("Distribution", fontsize=9, fontweight="bold", color=PAL[0])
    ax1.legend(fontsize=7); ax1.spines[["top","right"]].set_visible(False)

    # Q-Q Plot
    ax2 = fig.add_subplot(gs[1]); ax2.set_facecolor("#f8fafc")
    (osm, osr), (slope, intercept, _) = stats.probplot(data)
    ax2.plot(osm, osr, "o", color=PAL[1], markersize=5, markeredgecolor="white", alpha=.8)
    ax2.plot(osm, slope*np.array(osm)+intercept, "--", color=PAL[0], linewidth=1.5)
    ax2.set_xlabel("Theoretical Quantiles", fontsize=8)
    ax2.set_ylabel("Sample Quantiles", fontsize=8)
    ax2.set_title("Normal Q-Q Plot", fontsize=9, fontweight="bold", color=PAL[0])
    ax2.spines[["top","right"]].set_visible(False)

    # Box plot
    ax3 = fig.add_subplot(gs[2]); ax3.set_facecolor("#f8fafc")
    bp = ax3.boxplot(data, patch_artist=True, widths=0.5,
                     medianprops={"color":"white","linewidth":2},
                     boxprops={"facecolor":PAL[1],"alpha":0.8})
    ax3.axhline(mu0, color="#e94560", linestyle="--", linewidth=1.8, label=f"μ₀={mu0}")
    ax3.set_xticklabels([var_name], fontsize=8)
    ax3.set_ylabel("Value", fontsize=8)
    ax3.set_title("Box Plot", fontsize=9, fontweight="bold", color=PAL[0])
    ax3.legend(fontsize=7); ax3.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    return fig

def plot_paired(data1, data2, label1, label2, diff):
    fig = plt.figure(figsize=(14, 4), facecolor="#f8fafc")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=.35)

    # Before-After lines
    ax1 = fig.add_subplot(gs[0]); ax1.set_facecolor("#f8fafc")
    for v1, v2 in zip(data1, data2):
        color = PAL[0] if v2 >= v1 else PAL[1]
        ax1.plot([0,1], [v1,v2], "-o", color=color, alpha=0.5, markersize=4)
    ax1.plot([0,1],[np.mean(data1),np.mean(data2)],"-o",color=PAL[1],
             linewidth=3,markersize=8,label="Group Mean")
    ax1.set_xticks([0,1]); ax1.set_xticklabels([label1,label2], fontsize=9)
    ax1.set_ylabel("Value", fontsize=8)
    ax1.set_title("Individual Changes", fontsize=9, fontweight="bold", color=PAL[0])
    ax1.legend(fontsize=7); ax1.spines[["top","right"]].set_visible(False)

    # Distribution of differences + Q-Q
    ax2 = fig.add_subplot(gs[1]); ax2.set_facecolor("#f8fafc")
    ax2.hist(diff, bins="auto", color=PAL[1], alpha=0.75, edgecolor="white")
    ax2.axvline(0, color=PAL[0], linestyle="--", linewidth=1.8)
    ax2.axvline(np.mean(diff), color=PAL[2], linestyle="-", linewidth=1.8,
                label=f"Mean diff={np.mean(diff):.2f}")
    ax2.set_xlabel("Difference", fontsize=8); ax2.set_ylabel("Frequency", fontsize=8)
    ax2.set_title("Distribution of Differences", fontsize=9, fontweight="bold", color=PAL[0])
    ax2.legend(fontsize=7); ax2.spines[["top","right"]].set_visible(False)

    # Box plots side by side
    ax3 = fig.add_subplot(gs[2]); ax3.set_facecolor("#f8fafc")
    ax3.boxplot([data1, data2], patch_artist=True,
                boxprops={"facecolor":PAL[1],"alpha":0.75},
                medianprops={"color":"white","linewidth":2},
                widths=0.5)
    ax3.set_xticklabels([label1, label2], fontsize=8)
    ax3.set_ylabel("Value", fontsize=8)
    ax3.set_title("Box Plots", fontsize=9, fontweight="bold", color=PAL[0])
    ax3.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    return fig

def plot_independent(g1, g2, label1, label2, dep_var):
    fig = plt.figure(figsize=(14, 4), facecolor="#f8fafc")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=.35)

    # Box plots
    ax1 = fig.add_subplot(gs[0]); ax1.set_facecolor("#f8fafc")
    bp = ax1.boxplot([g1,g2], patch_artist=True, widths=0.5,
                     medianprops={"color":"white","linewidth":2})
    for patch, c in zip(bp["boxes"], [PAL[0], PAL[1]]):
        patch.set_facecolor(c); patch.set_alpha(0.8)
    ax1.set_xticklabels([label1, label2], fontsize=8)
    ax1.set_ylabel(dep_var, fontsize=8)
    ax1.set_title("Box Plots by Group", fontsize=9, fontweight="bold", color=PAL[0])
    ax1.spines[["top","right"]].set_visible(False)

    # Histogram overlay
    ax2 = fig.add_subplot(gs[1]); ax2.set_facecolor("#f8fafc")
    bins = np.linspace(min(min(g1),min(g2))-.5, max(max(g1),max(g2))+.5, 15)
    ax2.hist(g1, bins=bins, color=PAL[0], alpha=0.65, label=label1, edgecolor="white")
    ax2.hist(g2, bins=bins, color=PAL[1], alpha=0.65, label=label2, edgecolor="white")
    ax2.set_xlabel(dep_var, fontsize=8); ax2.set_ylabel("Frequency", fontsize=8)
    ax2.set_title("Distribution by Group", fontsize=9, fontweight="bold", color=PAL[0])
    ax2.legend(fontsize=7); ax2.spines[["top","right"]].set_visible(False)

    # Q-Q for each group
    ax3 = fig.add_subplot(gs[2]); ax3.set_facecolor("#f8fafc")
    for grp_data, c, lbl in [(g1,PAL[0],label1),(g2,PAL[1],label2)]:
        (osm, osr),(sl,ic,_) = stats.probplot(grp_data)
        ax3.plot(osm, osr, "o", color=c, markersize=4, alpha=.75, label=lbl)
        ax3.plot(osm, sl*np.array(osm)+ic, "--", color=c, linewidth=1.2)
    ax3.set_xlabel("Theoretical Quantiles", fontsize=8)
    ax3.set_ylabel("Sample Quantiles", fontsize=8)
    ax3.set_title("Normal Q-Q by Group", fontsize=9, fontweight="bold", color=PAL[0])
    ax3.legend(fontsize=7); ax3.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# PDF GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def build_pdf(test_type, R, meta, interps, fig_bytes_list):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             rightMargin=1.8*cm, leftMargin=1.8*cm,
                             topMargin=2*cm, bottomMargin=2*cm)

    H1  = ParagraphStyle("H1", fontSize=12, fontName="Helvetica-Bold",
                          textColor=colors.white, backColor=colors.HexColor("#1a1a2e"),
                          spaceAfter=5, spaceBefore=12, borderPadding=(5,8,5,8))
    H2  = ParagraphStyle("H2", fontSize=10, fontName="Helvetica-Bold",
                          textColor=colors.HexColor("#1a1a2e"), spaceAfter=3, spaceBefore=8)
    BD  = ParagraphStyle("BD", fontSize=8.5, fontName="Helvetica", leading=13, spaceAfter=4)
    IT  = ParagraphStyle("IT", fontSize=8.5, fontName="Helvetica", leading=13,
                          backColor=colors.HexColor("#eff6ff"),
                          borderPadding=(5,8,5,8), spaceAfter=5)
    NT  = ParagraphStyle("NT", fontSize=7.5, fontName="Helvetica-Oblique",
                          textColor=colors.HexColor("#64748b"), spaceAfter=4)
    TIT = ParagraphStyle("TIT", fontSize=17, fontName="Helvetica-Bold",
                          textColor=colors.HexColor("#1a1a2e"), alignment=TA_CENTER)
    SUB = ParagraphStyle("SUB", fontSize=10, fontName="Helvetica",
                          textColor=colors.HexColor("#64748b"),
                          alignment=TA_CENTER, spaceAfter=16)

    TS = TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTNAME",(0,1),(-1,-1),"Helvetica"),
        ("FONTSIZE",(0,0),(-1,-1),7.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f8fafc")]),
        ("GRID",(0,0),(-1,-1),.5,colors.HexColor("#e2e8f0")),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),
        ("ALIGN",(0,0),(0,-1),"LEFT"),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
        ("LEFTPADDING",(0,0),(-1,-1),6),
        ("RIGHTPADDING",(0,0),(-1,-1),6),
    ])

    def mktbl(df_or_rows):
        if isinstance(df_or_rows, pd.DataFrame):
            rows = [list(df_or_rows.columns)]
            for _, r in df_or_rows.iterrows():
                rows.append([str(v) for v in r.values])
        else:
            rows = df_or_rows
        t = Table(rows, repeatRows=1)
        t.setStyle(TS)
        return t

    story = []
    story.append(Spacer(1,.4*cm))
    story.append(Paragraph("INFERENTIAL STATISTICS REPORT", TIT))
    story.append(Paragraph("SPSS-Equivalent Output · Parametric & Non-Parametric Analysis", SUB))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#e94560")))
    story.append(Spacer(1,6))

    meta_rows = [[k, str(v)] for k,v in meta.items()]
    mt = Table(meta_rows, colWidths=[5*cm,11*cm])
    mt.setStyle(TableStyle([("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
                             ("FONTNAME",(1,0),(1,-1),"Helvetica"),
                             ("FONTSIZE",(0,0),(-1,-1),8.5),
                             ("TEXTCOLOR",(0,0),(0,-1),colors.HexColor("#1a1a2e")),
                             ("TOPPADDING",(0,0),(-1,-1),3),
                             ("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story.append(mt)
    story.append(Spacer(1,6))
    story.append(HRFlowable(width="100%",thickness=.5,color=colors.HexColor("#e2e8f0")))

    # Normality
    story.append(Paragraph("  1. NORMALITY ASSESSMENT — SHAPIRO-WILK TEST", H1))
    n_rows = [["Variable","N","Statistic (W)","Sig.","Result"]]
    for n in R["normality"]:
        n_rows.append([n["label"], str(n["n"]), _f(n["W"]), _p(n["p"]),
                       "Normal (p > .05)" if n["pass"] else "Non-Normal (p ≤ .05)"])
    story.append(mktbl(n_rows))
    method = "Parametric" if R["use_param"] else "Non-Parametric"
    story.append(Paragraph(f"Note. Based on normality, {method} analysis was selected.", NT))

    # Descriptives
    story.append(Paragraph("  2. DESCRIPTIVE STATISTICS", H1))
    desc_fmt = R["desc"].copy()
    for c in desc_fmt.select_dtypes(include=float).columns:
        desc_fmt[c] = desc_fmt[c].apply(_f)
    story.append(mktbl(desc_fmt))

    # Correlation (paired only)
    if "correlation" in R:
        story.append(Paragraph("  3. PAIRED SAMPLES CORRELATIONS", H1))
        corr = R["correlation"].copy()
        corr["Pearson r"]       = corr["Pearson r"].apply(_f)
        corr["Sig. (2-tailed)"] = corr["Sig. (2-tailed)"].apply(_p)
        story.append(mktbl(corr))

    # Levene (independent only)
    sec_n = 3
    if "levene" in R:
        sec_n += 1
        story.append(Paragraph(f"  {sec_n}. LEVENE'S TEST FOR EQUALITY OF VARIANCES", H1))
        lev = R["levene"]
        story.append(mktbl([["F","df1","df2","Sig.","Result"],
                              [_f(lev["F"]),str(lev["df1"]),str(lev["df2"]),
                               _p(lev["Sig."]),"Equal" if lev["equal_var"] else "Not Equal"]]))

    # Main test results
    sec_n += 1
    if R["use_param"]:
        story.append(Paragraph(f"  {sec_n}. PARAMETRIC TEST RESULTS", H1))
        pr = R["parametric"]
        if test_type == "One-Sample T-Test":
            rows = [["","t","df","Sig. (2-tailed)","Mean Diff","95% CI Lower","95% CI Upper","Cohen's d"],
                    ["Test Value = " + str(pr["mu0"]),
                     _f(pr["t"]),str(pr["df"]),_p(pr["p_two"]),
                     _f(pr["mean_diff"]),_f(pr["ci_lower"]),_f(pr["ci_upper"]),_f(pr["cohens_d"])]]
            story.append(mktbl(rows))
        elif test_type == "Paired-Sample T-Test":
            rows = [["Pair","Mean Diff","Std. Dev.","Std. Error","95% CI Lower","95% CI Upper","t","df","Sig. (2-tailed)","Cohen's d"],
                    [f"{pr['label1']} – {pr['label2']}",
                     _f(pr["mean_diff"]),_f(pr["sd_diff"]),_f(pr["se_diff"]),
                     _f(pr["ci_lower"]),_f(pr["ci_upper"]),
                     _f(pr["t"]),str(pr["df"]),_p(pr["p_two"]),_f(pr["cohens_d"])]]
            story.append(mktbl(rows))
        else:  # Independent
            rows = [["","Levene","","t-test","","","","","","",""],
                    ["","F","Sig.","t","df","Sig. (2-tail)","Mean Diff","SE Diff","95% CI L","95% CI U","Cohen's d"],
                    ["Equal var. assumed",
                     _f(R["levene"]["F"]),_p(R["levene"]["Sig."]),
                     _f(pr["t_eq"]),str(pr["df_eq"]),_p(pr["p_eq"]),
                     _f(pr["mean_diff"]),_f(pr["se_eq"]),_f(pr["ci_eq_l"]),_f(pr["ci_eq_u"]),_f(pr["cohens_d"])],
                    ["Equal var. not assumed","","",
                     _f(pr["t_welch"]),_f(pr["df_welch"],1),_p(pr["p_welch"]),
                     _f(pr["mean_diff"]),_f(pr["se_welch"]),_f(pr["ci_welch_l"]),_f(pr["ci_welch_u"]),"—"]]
            story.append(mktbl(rows))
    else:
        story.append(Paragraph(f"  {sec_n}. NON-PARAMETRIC TEST RESULTS", H1))
        np_r = R["nonparametric"]
        if test_type == "Independent-Sample T-Test":
            rows = [["","N","Mann-Whitney U","Wilcoxon W","Sig. (2-tailed)","Effect Size r"],
                    [f"{np_r['label1']}",str(np_r["n1"]),"","","",""],
                    [f"{np_r['label2']}",str(np_r["n2"]),"","","",""],
                    ["Total","",_f(np_r["U"]),"—",_p(np_r["p"]),_f(np_r["r_effect"])]]
            story.append(mktbl(rows))
        else:
            rows = [["","N","Wilcoxon W","Sig. (2-tailed)","Effect Size r"],
                    ["Test",str(np_r["n_nonzero"]),_f(np_r["W"]),_p(np_r["p"]),_f(np_r["r_effect"])]]
            story.append(mktbl(rows))

    # Interpretation
    sec_n += 1
    story.append(Paragraph(f"  {sec_n}. STATISTICAL INTERPRETATION", H1))
    for line in interps:
        clean = (line.replace("<b>","").replace("</b>","")
                     .replace("<i>","").replace("</i>","")
                     .replace("✓","").replace("✗",""))
        story.append(Paragraph(clean, IT))

    # Figures
    story.append(PageBreak())
    sec_n += 1
    story.append(Paragraph(f"  {sec_n}. FIGURES", H1))
    for i, fb in enumerate(fig_bytes_list, 1):
        story.append(Image(io.BytesIO(fb), width=17*cm, height=5*cm))
        story.append(Paragraph(f"Figure {i}. Diagnostic plots.", NT))
        story.append(Spacer(1,8))

    story.append(HRFlowable(width="100%",thickness=.5,color=colors.HexColor("#e2e8f0")))
    story.append(Paragraph(
        "Generated by Inferential Statistics App · SPSS-equivalent output · "
        "Automatic parametric/non-parametric selection via Shapiro-Wilk test", NT))

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ══════════════════════════════════════════════════════════════════════════════
# HTML TABLE RENDERER
# ══════════════════════════════════════════════════════════════════════════════
def html_tbl(df_or_rows, first_cols_left=2):
    if isinstance(df_or_rows, pd.DataFrame):
        cols = list(df_or_rows.columns)
        rows_data = [list(r.values) for _, r in df_or_rows.iterrows()]
    else:
        cols = df_or_rows[0]
        rows_data = df_or_rows[1:]

    html = '<div class="spss-wrap"><table class="spss-tbl"><thead><tr>'
    for c in cols:
        html += f"<th>{c}</th>"
    html += "</tr></thead><tbody>"
    for row in rows_data:
        html += "<tr>"
        for i, v in enumerate(row):
            style = ' style="text-align:left"' if i < first_cols_left else ""
            html += f"<td{style}>{v}</td>"
        html += "</tr>"
    html += "</tbody></table></div>"
    return html

def fmt_df(df, num_cols=None, p_cols=None, int_cols=None):
    d = df.copy()
    num_cols  = num_cols  or []
    p_cols    = p_cols    or []
    int_cols  = int_cols  or []
    for c in d.columns:
        if c in p_cols:
            d[c] = d[c].apply(lambda v: _p(float(v)) if not (isinstance(v,float) and np.isnan(v)) else ".")
        elif c in int_cols:
            d[c] = d[c].apply(lambda v: str(int(v)) if not (isinstance(v,float) and np.isnan(v)) else ".")
        elif c in num_cols or d[c].dtype in [float, np.float64]:
            d[c] = d[c].apply(lambda v: _f(float(v)) if isinstance(v,(float,np.floating)) and not np.isnan(v) else (str(v) if not isinstance(v,(float,np.floating)) else "."))
    return d

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.markdown("""
    <div class="main-hdr">
        <h1>📐 Inferential Statistics Suite <span class="badge">SPSS-Equivalent</span></h1>
        <p>Parametric & Non-Parametric Tests · Automatic Selection via Normality Testing · One-Sample · Paired-Sample · Independent-Sample</p>
    </div>""", unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Test Configuration")
        st.markdown("---")
        test_type = st.selectbox("📊 Select Test", [
            "One-Sample T-Test",
            "Paired-Sample T-Test",
            "Independent-Sample T-Test"
        ])
        alpha = st.selectbox("α Level", [0.05, 0.01, 0.001], index=0)
        st.markdown("---")

        # Show appropriate sample
        samp = SAMPLES[test_type]
        st.markdown(f"**📄 Template — {test_type}**")
        st.markdown(samp["note"])
        st.download_button(
            f"⬇️ Download Sample CSV",
            samp["csv"].encode(),
            f"sample_{test_type.lower().replace(' ','_').replace('-','_')}.csv",
            "text/csv", use_container_width=True
        )
        st.markdown("---")

        uploaded = st.file_uploader("📂 Upload CSV", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.success(f"✅ {len(df)} rows × {len(df.columns)} cols")
            except Exception as e:
                st.error(f"Read error: {e}"); df = None
        else:
            df = pd.read_csv(io.StringIO(samp["csv"]))
            st.info("ℹ️ Using built-in sample data")

        if df is not None:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
            st.markdown("---")

            # Variable selection per test
            if test_type == "One-Sample T-Test":
                test_var = st.selectbox("🎯 Test Variable", num_cols,
                                         index=num_cols.index("score") if "score" in num_cols else 0)
                mu0 = st.number_input("📏 Test Value (μ₀)", value=75.0, step=0.5)
                cfg = {"test_var": test_var, "mu0": mu0}

            elif test_type == "Paired-Sample T-Test":
                v1 = st.selectbox("Variable 1 (Pre/Before)", num_cols,
                                   index=num_cols.index("pre_score") if "pre_score" in num_cols else 0)
                v2 = st.selectbox("Variable 2 (Post/After)", num_cols,
                                   index=num_cols.index("post_score") if "post_score" in num_cols else
                                   min(1, len(num_cols)-1))
                cfg = {"v1": v1, "v2": v2}

            else:  # Independent
                grp_col = st.selectbox("👥 Grouping Variable",
                                        cat_cols if cat_cols else num_cols,
                                        index=(cat_cols.index("group") if "group" in cat_cols else 0))
                dep_col = st.selectbox("🎯 Dependent Variable", num_cols,
                                        index=num_cols.index("score") if "score" in num_cols else 0)
                groups  = sorted(df[grp_col].dropna().unique())
                if len(groups) >= 2:
                    g1_label = st.selectbox("Group 1", groups, index=0)
                    g2_label = st.selectbox("Group 2", groups,
                                             index=min(1, len(groups)-1))
                    cfg = {"grp_col": grp_col, "dep_col": dep_col,
                           "g1": g1_label, "g2": g2_label}
                else:
                    st.error("Need ≥ 2 groups"); cfg = None

            st.markdown("---")
            run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        else:
            run_btn = False; cfg = None

    if df is None:
        return

    # ── Test description card ──────────────────────────────────────────────────
    samp = SAMPLES[test_type]
    st.markdown(f"""
    <div class="test-card">
        <h4>📋 {test_type}</h4>
        <p>{samp['desc']}<br><small><b>Parametric test</b> → used when normality assumption is met (Shapiro-Wilk p > .05)<br>
        <b>Non-parametric equivalent</b> → used when normality is violated (Shapiro-Wilk p ≤ .05)</small></p>
    </div>""", unsafe_allow_html=True)

    # Data preview
    with st.expander("🔍 Data Preview", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    if not run_btn and "stats_R" not in st.session_state:
        st.info("👈 Configure variables in the sidebar, then click **Run Analysis**.")
        return

    if run_btn:
        if cfg is None:
            st.error("⚠️ Configuration incomplete."); return
        with st.spinner("Running analysis…"):
            try:
                if test_type == "One-Sample T-Test":
                    data = df[cfg["test_var"]].dropna().values.tolist()
                    R    = run_one_sample(data, cfg["mu0"], alpha)
                    meta = {"Test": test_type, "Variable": cfg["test_var"],
                            "Test Value (μ₀)": cfg["mu0"], "N": len(data),
                            "α": alpha, "Date": datetime.now().strftime("%B %d, %Y %H:%M")}
                    interps = interpret_one_sample(R, cfg["test_var"], alpha)
                    fig_main = plot_one_sample(data, cfg["mu0"], cfg["test_var"])
                    figs = [fig_main]

                elif test_type == "Paired-Sample T-Test":
                    paired_df = df[[cfg["v1"], cfg["v2"]]].dropna()
                    d1 = paired_df[cfg["v1"]].values.tolist()
                    d2 = paired_df[cfg["v2"]].values.tolist()
                    R  = run_paired(d1, d2, cfg["v1"], cfg["v2"], alpha)
                    meta = {"Test": test_type,
                            "Variable 1": cfg["v1"], "Variable 2": cfg["v2"],
                            "N pairs": len(d1), "α": alpha,
                            "Date": datetime.now().strftime("%B %d, %Y %H:%M")}
                    interps = interpret_paired(R, alpha)
                    diff = np.array(d1)-np.array(d2)
                    fig_main = plot_paired(d1, d2, cfg["v1"], cfg["v2"], diff)
                    figs = [fig_main]

                else:  # Independent
                    g1_data = df[df[cfg["grp_col"]]==cfg["g1"]][cfg["dep_col"]].dropna().values.tolist()
                    g2_data = df[df[cfg["grp_col"]]==cfg["g2"]][cfg["dep_col"]].dropna().values.tolist()
                    R = run_independent(g1_data, g2_data, cfg["g1"], cfg["g2"], cfg["dep_col"], alpha)
                    meta = {"Test": test_type,
                            "Grouping Variable": cfg["grp_col"],
                            "Dependent Variable": cfg["dep_col"],
                            f"Group 1": f"{cfg['g1']} (n={len(g1_data)})",
                            f"Group 2": f"{cfg['g2']} (n={len(g2_data)})",
                            "α": alpha,
                            "Date": datetime.now().strftime("%B %d, %Y %H:%M")}
                    interps = interpret_independent(R, cfg["dep_col"], alpha)
                    fig_main = plot_independent(g1_data, g2_data, cfg["g1"], cfg["g2"], cfg["dep_col"])
                    figs = [fig_main]

            except Exception as e:
                st.error(f"Analysis error: {e}"); return

        st.session_state["stats_R"]    = R
        st.session_state["stats_meta"] = meta
        st.session_state["stats_interps"] = interps
        st.session_state["stats_test"] = test_type
        st.session_state["stats_cfg"]  = cfg
        st.session_state["stats_figs"] = [fig_to_bytes(f) for f in figs]
        for f in figs: plt.close(f)
        st.session_state["stats_df"]   = df

    R       = st.session_state.get("stats_R")
    meta    = st.session_state.get("stats_meta")
    interps = st.session_state.get("stats_interps")
    test_type = st.session_state.get("stats_test", test_type)
    cfg     = st.session_state.get("stats_cfg", cfg)
    fig_bytes_list = st.session_state.get("stats_figs", [])
    if R is None: return

    st.success("✅ Analysis complete!")

    # ── Normality decision banner ──────────────────────────────────────────────
    method_used = "Parametric" if R["use_param"] else "Non-Parametric"
    banner_cls  = "use-param" if R["use_param"] else "use-nonparam"
    banner_icon = "✅" if R["use_param"] else "⚠️"
    norm_desc   = "Shapiro-Wilk p > .05 → Normal distribution → " if R["use_param"] else "Shapiro-Wilk p ≤ .05 → Non-normal → "
    test_name_used = (
        ("One-Sample T-Test" if test_type=="One-Sample T-Test"
         else "Paired T-Test" if test_type=="Paired-Sample T-Test"
         else "Independent T-Test")
        if R["use_param"] else
        ("Wilcoxon Signed-Rank" if test_type in ("One-Sample T-Test","Paired-Sample T-Test")
         else "Mann-Whitney U")
    )
    st.markdown(
        f'<div class="decision-banner {banner_cls}">'
        f'{banner_icon} {norm_desc}<b>{test_name_used}</b> selected automatically'
        f'</div>', unsafe_allow_html=True)

    # ── Quick metrics ──────────────────────────────────────────────────────────
    if R["use_param"]:
        pr = R["parametric"]
        if test_type == "One-Sample T-Test":
            metrics = [
                (_f(pr["t"]),   f"t({pr['df']})"),
                (_p(pr["p_two"]), "Sig. (2-tailed)"),
                (_f(pr["mean_diff"]), "Mean Diff"),
                (_f(pr["cohens_d"]),  "Cohen's d"),
                (effect_label(pr["cohens_d"]).title(), "Effect Size"),
            ]
        elif test_type == "Paired-Sample T-Test":
            metrics = [
                (_f(pr["t"]),   f"t({pr['df']})"),
                (_p(pr["p_two"]), "Sig. (2-tailed)"),
                (_f(pr["mean_diff"]), "Mean Diff"),
                (_f(pr["sd_diff"]),   "SD of Diff"),
                (_f(pr["cohens_d"]),  "Cohen's d"),
            ]
        else:
            metrics = [
                (_f(pr["t_eq"]),   f"t({pr['df_eq']}) Equal"),
                (_f(pr["t_welch"]),f"t({_f(pr['df_welch'],1)}) Welch"),
                (_p(pr["p_eq"]),   "Sig. Equal"),
                (_p(pr["p_welch"]),"Sig. Welch"),
                (_f(pr["cohens_d"]),"Cohen's d"),
            ]
    else:
        np_r = R["nonparametric"]
        if test_type == "Independent-Sample T-Test":
            metrics = [
                (_f(np_r["U"]),      "Mann-Whitney U"),
                (_p(np_r["p"]),      "Sig. (2-tailed)"),
                (_f(np_r["r_effect"]),"Effect Size r"),
                (r_label(np_r["r_effect"]).title(), "Effect Label"),
                (str(np_r["n1"]+np_r["n2"]), "Total N"),
            ]
        else:
            metrics = [
                (_f(np_r["W"]),       "Wilcoxon W"),
                (_p(np_r["p"]),       "Sig. (2-tailed)"),
                (_f(np_r["r_effect"]),"Effect Size r"),
                (r_label(np_r["r_effect"]).title(), "Effect Label"),
                (str(np_r["n_nonzero"]), "N (non-zero)"),
            ]

    cols = st.columns(5)
    for col, (val, lbl) in zip(cols, metrics):
        col.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div>'
                     f'<div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab_labels = ["📋 Normality", "📊 Descriptives"]
    if "correlation" in R: tab_labels.append("🔗 Correlation")
    if "levene" in R:      tab_labels.append("⚖️ Levene's")
    tab_labels += ["📈 Parametric Results", "📉 Non-Parametric Results",
                   "📈 Plots", "💬 Interpretation"]
    tabs = st.tabs(tab_labels)
    ti = 0

    # Tab: Normality
    tab_idx_0 = ti; ti += 1
    with tabs[tab_idx_0]:
        st.markdown('<div class="sec-title">Shapiro-Wilk Test of Normality</div>', unsafe_allow_html=True)
        norm_rows = [["Variable","N","Statistic (W)","Sig.","Result"]]
        for n in R["normality"]:
            res = '<span class="pass">✓ Normal (p > .05)</span>' if n["pass"] else '<span class="fail">✗ Non-Normal (p ≤ .05)</span>'
            norm_rows.append([n["label"], str(n["n"]), _f(n["W"]), _p(n["p"]), res])
        st.markdown(html_tbl(norm_rows, first_cols_left=1), unsafe_allow_html=True)
        st.markdown('<p class="note-txt">If p > .05: data is approximately normal → parametric test used.<br>'
                    'If p ≤ .05: normality violated → non-parametric test used.</p>', unsafe_allow_html=True)
        if R["use_param"]:
            st.markdown('<div class="info-box">✅ <b>Normality assumption met.</b> Parametric analysis applied.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="warn-box">⚠️ <b>Normality assumption violated.</b> Non-parametric analysis applied.</div>',
                        unsafe_allow_html=True)

    # Tab: Descriptives
    tab_idx_1 = ti; ti += 1
    with tabs[tab_idx_1]:
        st.markdown('<div class="sec-title">Descriptive Statistics</div>', unsafe_allow_html=True)
        desc_disp = R["desc"].copy()
        for c in desc_disp.select_dtypes(include=[float,np.float64]).columns:
            desc_disp[c] = desc_disp[c].apply(_f)
        st.markdown(html_tbl(desc_disp), unsafe_allow_html=True)

    # Tab: Correlation (paired only)
    if "correlation" in R:
        tab_idx_2 = ti; ti += 1
        with tabs[tab_idx_2]:
            st.markdown('<div class="sec-title">Paired Samples Correlations</div>', unsafe_allow_html=True)
            corr = R["correlation"].copy()
            corr["Pearson r"]       = corr["Pearson r"].apply(_f)
            corr["Sig. (2-tailed)"] = corr["Sig. (2-tailed)"].apply(_p)
            st.markdown(html_tbl(corr), unsafe_allow_html=True)

    # Tab: Levene (independent only)
    if "levene" in R:
        tab_idx_3 = ti; ti += 1
        with tabs[tab_idx_3]:
            lev = R["levene"]
            res = '<span class="pass">✓ Equal variances assumed</span>' if lev["equal_var"] else '<span class="fail">✗ Equal variances not assumed</span>'
            st.markdown('<div class="sec-title">Levene\'s Test for Equality of Variances</div>', unsafe_allow_html=True)
            st.markdown(f'<table class="spss-tbl"><thead><tr>'
                        f'<th>F</th><th>df1</th><th>df2</th><th>Sig.</th><th>Result</th>'
                        f'</tr></thead><tbody><tr>'
                        f'<td>{_f(lev["F"])}</td><td>{lev["df1"]}</td><td>{lev["df2"]}</td>'
                        f'<td>{_p(lev["Sig."])}</td><td style="text-align:left">{res}</td>'
                        f'</tr></tbody></table>', unsafe_allow_html=True)
            st.markdown('<p class="note-txt">If Levene p > .05: equal variances assumed → t-test row 1.<br>'
                        'If Levene p ≤ .05: equal variances not assumed → Welch t-test (row 2).</p>',
                        unsafe_allow_html=True)

    # Tab: Parametric results
    tab_idx_4 = ti; ti += 1
    with tabs[tab_idx_4]:
        pr = R["parametric"]
        if test_type == "One-Sample T-Test":
            st.markdown(f'<div class="sec-title">One-Sample Test · Test Value (μ₀) = {pr["mu0"]}</div>',
                        unsafe_allow_html=True)
            rows = [["","t","df","Sig. (2-tailed)","Sig. (1-tailed Lower)","Sig. (1-tailed Upper)",
                     "Mean Diff","95% CI Lower","95% CI Upper","Cohen's d","Effect Size"],
                    ["Test value = " + str(pr["mu0"]),
                     _f(pr["t"]), str(pr["df"]), _p(pr["p_two"]),
                     _p(pr["p_one_l"]), _p(pr["p_one_u"]),
                     _f(pr["mean_diff"]), _f(pr["ci_lower"]), _f(pr["ci_upper"]),
                     _f(pr["cohens_d"]), effect_label(pr["cohens_d"])]]
            st.markdown(html_tbl(rows, first_cols_left=1), unsafe_allow_html=True)

        elif test_type == "Paired-Sample T-Test":
            st.markdown('<div class="sec-title">Paired Samples Test</div>', unsafe_allow_html=True)
            rows = [["Pair","Mean Diff","Std. Dev.","Std. Error Mean",
                     "95% CI Lower","95% CI Upper","t","df",
                     "Sig. (2-tailed)","Sig. (1-tailed L)","Sig. (1-tailed U)","Cohen's d"],
                    [f"{pr['label1']} – {pr['label2']}",
                     _f(pr["mean_diff"]), _f(pr["sd_diff"]), _f(pr["se_diff"]),
                     _f(pr["ci_lower"]), _f(pr["ci_upper"]),
                     _f(pr["t"]), str(pr["df"]),
                     _p(pr["p_two"]), _p(pr["p_one_l"]), _p(pr["p_one_u"]),
                     _f(pr["cohens_d"])]]
            st.markdown(html_tbl(rows, first_cols_left=1), unsafe_allow_html=True)

        else:  # Independent
            st.markdown('<div class="sec-title">Independent Samples Test</div>', unsafe_allow_html=True)
            rows = [["","F (Levene)","Sig.","t","df","Sig. (2-tail)","Sig. (1-tail L)","Sig. (1-tail U)",
                     "Mean Diff","SE Diff","95% CI Lower","95% CI Upper","Cohen's d"],
                    ["Equal var. assumed",
                     _f(R["levene"]["F"]), _p(R["levene"]["Sig."]),
                     _f(pr["t_eq"]), str(pr["df_eq"]), _p(pr["p_eq"]),
                     _p(stats.t.cdf(pr["t_eq"], pr["df_eq"])),
                     _p(1-stats.t.cdf(pr["t_eq"], pr["df_eq"])),
                     _f(pr["mean_diff"]), _f(pr["se_eq"]),
                     _f(pr["ci_eq_l"]), _f(pr["ci_eq_u"]), _f(pr["cohens_d"])],
                    ["Equal var. NOT assumed", "", "",
                     _f(pr["t_welch"]), _f(pr["df_welch"],2), _p(pr["p_welch"]),
                     _p(stats.t.cdf(pr["t_welch"], pr["df_welch"])),
                     _p(1-stats.t.cdf(pr["t_welch"], pr["df_welch"])),
                     _f(pr["mean_diff"]), _f(pr["se_welch"]),
                     _f(pr["ci_welch_l"]), _f(pr["ci_welch_u"]), "—"]]
            st.markdown(html_tbl(rows, first_cols_left=1), unsafe_allow_html=True)

        if not R["use_param"]:
            st.markdown('<div class="warn-box">⚠️ Normality assumption was violated. '
                        'Use the <b>Non-Parametric Results</b> tab for the recommended analysis.</div>',
                        unsafe_allow_html=True)

    # Tab: Non-parametric results
    tab_idx_5 = ti; ti += 1
    with tabs[tab_idx_5]:
        np_r = R["nonparametric"]
        if test_type == "One-Sample T-Test":
            st.markdown('<div class="sec-title">Wilcoxon Signed-Rank Test (Non-Parametric One-Sample)</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="info-box">Test value (hypothesized median): <b>μ₀ = {np_r["mu0"]}</b></div>',
                        unsafe_allow_html=True)
            rows = [["N (non-zero diff)","Test Statistic (W)","Sig. (2-tailed)","Effect Size r","Effect Label"],
                    [str(np_r["n_nonzero"]), _f(np_r["W"],0), _p(np_r["p"]),
                     _f(np_r["r_effect"]), r_label(np_r["r_effect"])]]
            st.markdown(html_tbl(rows, first_cols_left=0), unsafe_allow_html=True)

        elif test_type == "Paired-Sample T-Test":
            st.markdown('<div class="sec-title">Wilcoxon Signed-Rank Test (Non-Parametric Paired)</div>',
                        unsafe_allow_html=True)
            pos = np_r["pos_ranks"]; neg = np_r["neg_ranks"]
            rows = [["","N"],
                    ["Negative Ranks", str(int(neg)) if not (isinstance(neg,float) and np.isnan(neg)) else "."],
                    ["Positive Ranks", str(int(pos)) if not (isinstance(pos,float) and np.isnan(pos)) else "."],
                    ["Ties", str(int(np_r["n"]-np_r["n_nonzero"]))],
                    ["Total", str(np_r["n"])]]
            st.markdown('<div class="sec-title">Ranks</div>', unsafe_allow_html=True)
            st.markdown(html_tbl(rows, first_cols_left=1), unsafe_allow_html=True)
            st.markdown('<div class="sec-title">Test Statistics</div>', unsafe_allow_html=True)
            rows2 = [["","Value"],
                     ["Test Statistic (W)", _f(np_r["W"],0)],
                     ["Sig. (2-tailed)", _p(np_r["p"])],
                     ["Effect Size r", _f(np_r["r_effect"])],
                     ["Effect Label", r_label(np_r["r_effect"])]]
            st.markdown(html_tbl(rows2, first_cols_left=1), unsafe_allow_html=True)
            st.markdown(f'<p class="note-txt">a. Based on positive ranks. b. Wilcoxon Signed Rank Test.<br>'
                        f'Direction: {np_r["label1"]} {">" if np_r.get("pos_ranks",0) > np_r.get("neg_ranks",0) else "<"} {np_r["label2"]}</p>',
                        unsafe_allow_html=True)

        else:  # Independent → Mann-Whitney U
            st.markdown('<div class="sec-title">Mann-Whitney U Test (Non-Parametric Independent)</div>',
                        unsafe_allow_html=True)
            rows = [["","N","Mean Rank","Sum of Ranks"],
                    [np_r["label1"], str(np_r["n1"]), "—", "—"],
                    [np_r["label2"], str(np_r["n2"]), "—", "—"],
                    ["Total", str(np_r["n1"]+np_r["n2"]), "", ""]]
            st.markdown('<div class="sec-title">Ranks</div>', unsafe_allow_html=True)
            st.markdown(html_tbl(rows, first_cols_left=1), unsafe_allow_html=True)
            st.markdown('<div class="sec-title">Test Statistics</div>', unsafe_allow_html=True)
            rows2 = [["","Value"],
                     ["Mann-Whitney U", _f(np_r["U"],0)],
                     ["Wilcoxon W", "—"],
                     ["Sig. (2-tailed)", _p(np_r["p"])],
                     ["Effect Size r", _f(np_r["r_effect"])],
                     ["Effect Label", r_label(np_r["r_effect"])]]
            st.markdown(html_tbl(rows2, first_cols_left=1), unsafe_allow_html=True)

        if R["use_param"]:
            st.markdown('<div class="info-box">ℹ️ Normality assumption was met. '
                        'The <b>Parametric Results</b> tab contains the recommended analysis.</div>',
                        unsafe_allow_html=True)

    # Tab: Plots
    tab_idx_6 = ti; ti += 1
    with tabs[tab_idx_6]:
        for fb in fig_bytes_list:
            st.image(fb, use_container_width=True)

    # Tab: Interpretation
    tab_idx_7 = ti; ti += 1
    with tabs[tab_idx_7]:
        st.markdown("### 📝 Statistical Interpretation")
        st.markdown("*Automated interpretation following APA 7th edition guidelines:*")
        for line in interps:
            sig_class = ""
            if "significant difference was found" in line.lower() and "no statistically" not in line.lower():
                sig_class = "sig"
            elif "not statistically significant" in line.lower() or "no significant" in line.lower():
                sig_class = "nonsig"
            st.markdown(f'<div class="interp-box {sig_class}">{line}</div>', unsafe_allow_html=True)

        # APA write-up
        st.markdown("---")
        st.markdown("**APA 7th Edition Write-Up:**")
        if R["use_param"]:
            pr = R["parametric"]
            if test_type == "One-Sample T-Test":
                apa = (f"A one-sample t-test was conducted to determine whether {list(meta.values())[1]} "
                       f"(M = {_f(float(R['desc']['Mean'].iloc[0]))}, SD = {_f(float(R['desc']['Std. Deviation'].iloc[0]))}) "
                       f"differed significantly from the test value of {pr['mu0']}. "
                       f"The result was {'statistically significant' if pr['p_two']<alpha else 'not statistically significant'}, "
                       f"t({pr['df']}) = {pr['t']:.2f}, p {'< .001' if pr['p_two']<.001 else '= '+_p(pr['p_two'])}, "
                       f"d = {pr['cohens_d']:.2f}.")
            elif test_type == "Paired-Sample T-Test":
                apa = (f"A paired-samples t-test indicated that scores on {pr['label2']} "
                       f"(M = {_f(float(R['desc'].iloc[1]['Mean']))}, SD = {_f(float(R['desc'].iloc[1]['Std. Deviation']))}) "
                       f"were {'significantly' if pr['p_two']<alpha else 'not significantly'} different from {pr['label1']} "
                       f"(M = {_f(float(R['desc'].iloc[0]['Mean']))}, SD = {_f(float(R['desc'].iloc[0]['Std. Deviation']))}), "
                       f"t({pr['df']}) = {pr['t']:.2f}, p {'< .001' if pr['p_two']<.001 else '= '+_p(pr['p_two'])}, "
                       f"d = {pr['cohens_d']:.2f}.")
            else:
                t_v=pr['t_eq'] if R['levene']['equal_var'] else pr['t_welch']
                p_v=pr['p_eq'] if R['levene']['equal_var'] else pr['p_welch']
                df_v=pr['df_eq'] if R['levene']['equal_var'] else pr['df_welch']
                apa = (f"An independent-samples t-test revealed {'a statistically significant' if p_v<alpha else 'no statistically significant'} "
                       f"difference in {pr['dep_var']} between {pr['label1']} "
                       f"(M = {_f(float(R['desc'].iloc[0]['Mean']))}, SD = {_f(float(R['desc'].iloc[0]['Std. Deviation']))}) "
                       f"and {pr['label2']} "
                       f"(M = {_f(float(R['desc'].iloc[1]['Mean']))}, SD = {_f(float(R['desc'].iloc[1]['Std. Deviation']))}), "
                       f"t({_f(df_v,2)}) = {t_v:.2f}, p {'< .001' if p_v<.001 else '= '+_p(p_v)}, "
                       f"d = {pr['cohens_d']:.2f}.")
        else:
            np_r = R["nonparametric"]
            if test_type == "Independent-Sample T-Test":
                apa = (f"A Mann-Whitney U test indicated {'a statistically significant' if np_r['p']<alpha else 'no significant'} "
                       f"difference between {np_r['label1']} and {np_r['label2']}, "
                       f"U = {np_r['U']:.0f}, p {'< .001' if np_r['p']<.001 else '= '+_p(np_r['p'])}, "
                       f"r = {np_r['r_effect']:.2f}.")
            else:
                apa = (f"A Wilcoxon signed-rank test indicated {'a statistically significant' if np_r['p']<alpha else 'no significant'} "
                       f"result, W = {np_r['W']:.0f}, p {'< .001' if np_r['p']<.001 else '= '+_p(np_r['p'])}, "
                       f"r = {np_r['r_effect']:.2f}.")
        st.code(apa, language=None)

    # ── Downloads ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Download Results")
    dc1, dc2, dc3 = st.columns(3)

    with dc1:
        with st.spinner("Building PDF…"):
            pdf_data = build_pdf(test_type, R, meta, interps, fig_bytes_list)
        st.download_button("📄 PDF Report", pdf_data,
                           f"Stats_{test_type.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                           "application/pdf", use_container_width=True)

    with dc2:
        xbuf = io.BytesIO()
        with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
            R["desc"].to_excel(writer, sheet_name="Descriptive Statistics", index=False)
            norm_df = pd.DataFrame(R["normality"])
            norm_df.to_excel(writer, sheet_name="Shapiro-Wilk", index=False)
            if "correlation" in R:
                R["correlation"].to_excel(writer, sheet_name="Paired Correlation", index=False)
            if "levene" in R:
                pd.DataFrame([R["levene"]]).to_excel(writer, sheet_name="Levene Test", index=False)
            pd.DataFrame([R["parametric"]]).to_excel(writer, sheet_name="Parametric Results", index=False)
            pd.DataFrame([R["nonparametric"]]).to_excel(writer, sheet_name="Non-Parametric Results", index=False)
        xbuf.seek(0)
        st.download_button("📊 Excel Workbook", xbuf.getvalue(),
                           f"Stats_{test_type.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

    with dc3:
        csv_parts = [f"=== {test_type.upper()} ===\n"]
        csv_parts.append("=== DESCRIPTIVE STATISTICS ===\n" + R["desc"].to_csv(index=False))
        csv_parts.append("=== SHAPIRO-WILK NORMALITY ===\n" + pd.DataFrame(R["normality"]).to_csv(index=False))
        if "levene" in R:
            csv_parts.append("=== LEVENE TEST ===\n" + pd.DataFrame([R["levene"]]).to_csv(index=False))
        csv_parts.append("=== PARAMETRIC RESULTS ===\n" + pd.DataFrame([R["parametric"]]).to_csv(index=False))
        csv_parts.append("=== NON-PARAMETRIC RESULTS ===\n" + pd.DataFrame([R["nonparametric"]]).to_csv(index=False))
        st.download_button("📝 CSV Tables", "\n\n".join(csv_parts).encode(),
                           f"Stats_{test_type.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           "text/csv", use_container_width=True)


# ── Fix walrus operator for Python < 3.8 compatibility ─────────────────────
# Use a simple counter instead
if __name__ == "__main__":
    main()
