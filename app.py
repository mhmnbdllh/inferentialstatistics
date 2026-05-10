"""
Inferential Statistics App — SPSS-Equivalent Output
====================================================
Parametric & Non-parametric tests with automatic selection via normality testing.
Normality: Shapiro-Wilk + Kolmogorov-Smirnov (Lilliefors correction) — identical to SPSS.
Tests: One-Sample T, Paired-Sample T, Independent-Sample T
       + Wilcoxon Signed-Rank (Z with ties correction), Mann-Whitney U (SPSS-exact)

SPSS Formula Notes:
  1. Levene's test uses center='mean' (SPSS default)
  2. KS uses Lilliefors correction (statsmodels) — same as SPSS Explore
  3. Wilcoxon Z = (W − E[W]) / sqrt(Var[W] − ties_correction)
  4. Mann-Whitney: U=min(U1,U2), W=rank_sum(group1), Mean Rank, Sum of Ranks, Z
  5. Paired Samples Correlations shown on parametric tab and all downloads
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors
from collections import Counter
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
from reportlab.lib.enums import TA_CENTER

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
.main-hdr h1{color:#fff;font-size:2rem;font-weight:700;margin:0 0 .4rem 0;}
.main-hdr p{color:#94a3b8;margin:0;font-size:.92rem;}
.badge{display:inline-block;background:#e94560;color:#fff;font-size:.7rem;
  padding:2px 8px;border-radius:20px;margin-left:8px;font-weight:600;vertical-align:middle;}
.sec-title{background:linear-gradient(90deg,#1a1a2e,#16213e);color:#e2e8f0;
  padding:9px 16px;border-radius:6px 6px 0 0;font-weight:600;font-size:.84rem;
  letter-spacing:.6px;margin-top:1.4rem;font-family:'DM Mono',monospace;
  border-bottom:2px solid #e94560;}
.spss-wrap{overflow-x:auto;margin-bottom:.4rem;}
.spss-tbl{font-family:'DM Mono',monospace;font-size:.77rem;border-collapse:collapse;
  width:100%;min-width:400px;}
.spss-tbl th{background:#1a1a2e;color:#e2e8f0;padding:7px 12px;text-align:center;
  font-weight:600;border:1px solid #334155;font-size:.74rem;white-space:nowrap;}
.spss-tbl td{padding:5px 12px;border:1px solid #e2e8f0;text-align:right;
  color:#1e293b;white-space:nowrap;background:#fff;}
.spss-tbl tr:nth-child(even) td{background:#f8fafc;}
.spss-tbl td.left{text-align:left;font-weight:500;background:#f1f5f9!important;}
.interp-box{background:linear-gradient(135deg,#f8fafc,#f1f5f9);
  border-left:4px solid #0284c7;padding:1rem 1.2rem;border-radius:0 10px 10px 0;
  margin:.7rem 0;font-size:.87rem;line-height:1.8;color:#1e293b;}
.interp-box b{color:#0284c7;}
.interp-box.sig{border-left-color:#16a34a;background:linear-gradient(135deg,#f0fdf4,#dcfce7);}
.interp-box.sig b{color:#16a34a;}
.interp-box.nonsig{border-left-color:#dc2626;
  background:linear-gradient(135deg,#fef2f2,#fee2e2);}
.interp-box.nonsig b{color:#dc2626;}
.metric-card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;
  padding:.9rem;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.05);}
.metric-val{font-size:1.45rem;font-weight:700;color:#1a1a2e;
  font-family:'DM Mono',monospace;}
.metric-lbl{font-size:.7rem;color:#64748b;margin-top:.25rem;
  text-transform:uppercase;letter-spacing:.6px;}
.pass{color:#16a34a;font-weight:700;}
.fail{color:#dc2626;font-weight:700;}
.warn-box{background:#fffbeb;border-left:4px solid #f59e0b;padding:.7rem 1rem;
  border-radius:0 6px 6px 0;font-size:.82rem;color:#92400e;margin:.4rem 0;}
.info-box{background:#eff6ff;border-left:4px solid #3b82f6;padding:.7rem 1rem;
  border-radius:0 6px 6px 0;font-size:.82rem;color:#1e40af;margin:.4rem 0;}
.note-txt{font-size:.74rem;color:#64748b;font-style:italic;margin-top:.3rem;}
.decision-banner{padding:1rem 1.4rem;border-radius:10px;margin:1rem 0;
  font-size:.9rem;font-weight:600;}
.decision-banner.use-param{background:#dcfce7;color:#14532d;
  border:1px solid #86efac;}
.decision-banner.use-nonparam{background:#ffedd5;color:#7c2d12;
  border:1px solid #fdba74;}
</style>
""", unsafe_allow_html=True)

# ── Sample CSV templates ───────────────────────────────────────────────────────
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
        "desc": "Test whether the mean **score** differs from a known population value (μ₀).",
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
        "desc": "Test whether the mean difference between **pre_score** and **post_score** is zero.",
        "note": "Two numeric columns required: Variable 1 (pre) and Variable 2 (post)."
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
        "note": "One grouping column (categorical) and one numeric outcome column required."
    }
}

# ── Utilities ──────────────────────────────────────────────────────────────────
def _f(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "."
    return f"{v:.{d}f}"

def _p(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "."
    return ".000" if v < .001 else f"{v:.3f}"

def effect_label_d(d):
    a = abs(d)
    if a < .2:  return "negligible"
    if a < .5:  return "small"
    if a < .8:  return "medium"
    return "large"

def effect_label_r(r):
    a = abs(r)
    if a < .1:  return "negligible"
    if a < .3:  return "small"
    if a < .5:  return "medium"
    return "large"

def ci_mean(data, alpha=0.05):
    n = len(data); m = np.mean(data); se = stats.sem(data)
    tc = stats.t.ppf(1 - alpha / 2, n - 1)
    return m - tc * se, m + tc * se

def cohens_d_1s(data, mu0):
    return (np.mean(data) - mu0) / np.std(data, ddof=1)

def cohens_d_paired(diff):
    return np.mean(diff) / np.std(diff, ddof=1)

def cohens_d_ind(g1, g2):
    n1, n2 = len(g1), len(g2)
    sp = np.sqrt(((n1-1)*np.var(g1,ddof=1)+(n2-1)*np.var(g2,ddof=1))/(n1+n2-2))
    return (np.mean(g1) - np.mean(g2)) / sp if sp else np.nan

# ── Normality tests (Shapiro-Wilk + KS/Lilliefors) ───────────────────────────
def test_normality(data, label=""):
    """
    Run both Shapiro-Wilk and Kolmogorov-Smirnov (Lilliefors correction).
    Decision (use_param) based on BOTH tests: pass only if BOTH p > alpha.
    KS with Lilliefors correction = SPSS 'Kolmogorov-Smirnov with Lilliefors correction'.
    """
    data = np.array(data, dtype=float)
    n = len(data)
    result = {"label": label, "n": n}

    if n < 3:
        result.update({"sw_W": np.nan, "sw_p": np.nan, "sw_pass": False,
                       "ks_D": np.nan, "ks_p": np.nan, "ks_pass": False,
                       "pass": False})
        return result

    # Shapiro-Wilk
    sw_W, sw_p = stats.shapiro(data)
    result["sw_W"] = float(sw_W)
    result["sw_p"] = float(sw_p)
    result["sw_pass"] = float(sw_p) > 0.05

    # Kolmogorov-Smirnov with Lilliefors correction (SPSS method)
    try:
        ks_D, ks_p = lilliefors(data, dist='norm', pvalmethod='approx')
        # Lilliefors p is capped at 0.200 on the upper end (table limit)
        result["ks_D"] = float(ks_D)
        result["ks_p"] = float(ks_p)
        result["ks_pass"] = float(ks_p) >= 0.05
    except Exception:
        result["ks_D"] = np.nan
        result["ks_p"] = np.nan
        result["ks_pass"] = True  # if KS can't run, don't penalize

    # Overall: pass if BOTH tests pass (conservative, matches SPSS logic)
    result["pass"] = result["sw_pass"] and result["ks_pass"]
    return result

# ── SPSS-exact Wilcoxon Z (with ties correction) ──────────────────────────────
def wilcoxon_spss(diff_arr):
    """
    Wilcoxon signed-rank test exactly as SPSS.
    W = min(positive rank sum, negative rank sum)
    Z uses ties-corrected variance.
    """
    diff_arr = np.array(diff_arr, dtype=float)
    diff_nz  = diff_arr[diff_arr != 0]
    n        = len(diff_nz)
    n_ties   = len(diff_arr) - n

    if n < 1:
        return dict(W=np.nan, Z=np.nan, p=np.nan,
                    pos_rank_sum=0.0, neg_rank_sum=0.0,
                    n_pos=0, n_neg=0, n_ties=n_ties,
                    n_total=len(diff_arr))

    abs_d    = np.abs(diff_nz)
    ranks    = stats.rankdata(abs_d)
    pos_rs   = float(np.sum(ranks[diff_nz > 0]))
    neg_rs   = float(np.sum(ranks[diff_nz < 0]))
    n_pos    = int(np.sum(diff_nz > 0))
    n_neg    = int(np.sum(diff_nz < 0))
    W        = min(pos_rs, neg_rs)

    # Ties-corrected variance (SPSS formula)
    tie_grps  = Counter(abs_d)
    ties_corr = sum(t**3 - t for t in tie_grps.values()) / 48.0
    Var_W     = n*(n+1)*(2*n+1)/24 - ties_corr
    E_W       = n*(n+1)/4
    Z  = (W - E_W) / np.sqrt(Var_W) if Var_W > 0 else np.nan
    p  = float(2 * stats.norm.sf(abs(Z))) if not np.isnan(Z) else np.nan

    return dict(W=W, Z=Z, p=p,
                pos_rank_sum=pos_rs, neg_rank_sum=neg_rs,
                n_pos=n_pos, n_neg=n_neg,
                n_ties=n_ties, n_total=len(diff_arr))

# ── SPSS-exact Mann-Whitney U ──────────────────────────────────────────────────
def mannwhitney_spss(g1, g2, label1, label2):
    """
    Mann-Whitney U exactly as SPSS:
      U  = min(U1, U2)
      W  = rank sum of first group (Wilcoxon W in SPSS output)
      Z  = normal approximation with ties correction
    """
    g1, g2   = np.array(g1, dtype=float), np.array(g2, dtype=float)
    n1, n2   = len(g1), len(g2)
    N        = n1 + n2
    all_data = np.concatenate([g1, g2])
    all_rnks = stats.rankdata(all_data)
    R1 = float(np.sum(all_rnks[:n1]))
    R2 = float(np.sum(all_rnks[n1:]))

    U1 = n1*n2 + n1*(n1+1)/2 - R1
    U2 = n1*n2 + n2*(n2+1)/2 - R2
    U  = min(U1, U2)
    W_wilcoxon = R1          # SPSS Wilcoxon W = rank sum of group 1

    # Ties-corrected Z
    tie_grps  = Counter(all_rnks)
    ties_corr = sum(t**3 - t for t in tie_grps.values()) / (N*(N-1)) if N > 1 else 0
    Var_U     = n1*n2/12 * (N + 1 - ties_corr)
    E_U       = n1*n2 / 2
    Z  = (U - E_U) / np.sqrt(Var_U) if Var_U > 0 else np.nan
    p  = float(2 * stats.norm.sf(abs(Z))) if not np.isnan(Z) else np.nan

    return dict(U=U, U1=U1, U2=U2,
                W_wilcoxon=W_wilcoxon,
                R1=R1, R2=R2,
                mean_rank1=R1/n1, mean_rank2=R2/n2,
                n1=n1, n2=n2, Z=Z, p=p,
                label1=label1, label2=label2)

# ══════════════════════════════════════════════════════════════════════════════
# ONE-SAMPLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def run_one_sample(data, mu0, alpha=0.05):
    R    = {}
    data = np.array(data, dtype=float)
    n    = len(data)
    m    = np.mean(data); sd = np.std(data, ddof=1); se = sd/np.sqrt(n)
    ci_l, ci_u = ci_mean(data, alpha)

    R["desc"] = pd.DataFrame([{
        "N": n, "Mean": m, "Std. Deviation": sd, "Std. Error Mean": se,
        f"{int((1-alpha)*100)}% CI Lower": ci_l,
        f"{int((1-alpha)*100)}% CI Upper": ci_u,
        "Minimum": float(np.min(data)), "Maximum": float(np.max(data)),
        "Skewness": float(stats.skew(data)),
        "Kurtosis": float(stats.kurtosis(data))
    }])

    norm = test_normality(data, "Variable")
    R["normality"]  = [norm]
    R["use_param"]  = norm["pass"]

    # Parametric
    t_stat, p_two = stats.ttest_1samp(data, mu0)
    df = n - 1; tc = stats.t.ppf(1 - alpha/2, df)
    diff_m = m - mu0
    d = cohens_d_1s(data, mu0)
    R["parametric"] = {
        "test": "One-Sample T-Test", "mu0": mu0,
        "t": float(t_stat), "df": df,
        "p_two": float(p_two),
        "p_one_lower": float(stats.t.cdf(t_stat, df)),
        "p_one_upper": float(1 - stats.t.cdf(t_stat, df)),
        "mean_diff": diff_m,
        "ci_lower":  diff_m - tc*(sd/np.sqrt(n)),
        "ci_upper":  diff_m + tc*(sd/np.sqrt(n)),
        "cohens_d": d
    }

    # Non-parametric (Wilcoxon, SPSS-exact Z)
    w_res = wilcoxon_spss(data - mu0)
    R["nonparametric"] = {**w_res,
                          "test": "Wilcoxon Signed-Rank Test",
                          "mu0": mu0}
    return R

# ══════════════════════════════════════════════════════════════════════════════
# PAIRED-SAMPLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def run_paired(data1, data2, label1="Var1", label2="Var2", alpha=0.05):
    R = {}
    data1, data2 = np.array(data1, dtype=float), np.array(data2, dtype=float)
    n    = len(data1)
    diff = data1 - data2
    m1, m2   = np.mean(data1), np.mean(data2)
    sd1, sd2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    se1, se2 = sd1/np.sqrt(n), sd2/np.sqrt(n)
    ci1_l, ci1_u = ci_mean(data1, alpha)
    ci2_l, ci2_u = ci_mean(data2, alpha)

    R["desc"] = pd.DataFrame([
        {"Variable": label1, "N": n, "Mean": m1,
         "Std. Deviation": sd1, "Std. Error Mean": se1,
         f"{int((1-alpha)*100)}% CI Lower": ci1_l,
         f"{int((1-alpha)*100)}% CI Upper": ci1_u},
        {"Variable": label2, "N": n, "Mean": m2,
         "Std. Deviation": sd2, "Std. Error Mean": se2,
         f"{int((1-alpha)*100)}% CI Lower": ci2_l,
         f"{int((1-alpha)*100)}% CI Upper": ci2_u},
    ])

    # Pearson correlation (SPSS Paired Samples Correlations table)
    r_corr, p_corr = stats.pearsonr(data1, data2)
    R["correlation"] = pd.DataFrame([{
        "Pair":  f"{label1} & {label2}",
        "N":     n,
        "Pearson Correlation": float(r_corr),
        "Sig. (2-tailed)":     float(p_corr)
    }])

    # Normality on differences (both SW + KS)
    norm = test_normality(diff, f"{label1} − {label2}")
    R["normality"]  = [norm]
    R["use_param"]  = norm["pass"]

    # Parametric
    t_stat, p_two = stats.ttest_rel(data1, data2)
    df = n - 1
    m_diff  = np.mean(diff); sd_diff = np.std(diff, ddof=1)
    se_diff = sd_diff / np.sqrt(n)
    tc      = stats.t.ppf(1 - alpha/2, df)
    d = cohens_d_paired(diff)
    R["parametric"] = {
        "test": "Paired Samples T-Test",
        "label1": label1, "label2": label2,
        "mean_diff": m_diff, "sd_diff": sd_diff, "se_diff": se_diff,
        "t": float(t_stat), "df": df,
        "p_two":      float(p_two),
        "p_one_lower": float(stats.t.cdf(t_stat, df)),
        "p_one_upper": float(1 - stats.t.cdf(t_stat, df)),
        "ci_lower": m_diff - tc*se_diff,
        "ci_upper": m_diff + tc*se_diff,
        "cohens_d": d
    }

    # Non-parametric: Wilcoxon SPSS-exact
    w_res = wilcoxon_spss(diff)
    R["nonparametric"] = {**w_res,
                          "test": "Wilcoxon Signed-Rank Test",
                          "label1": label1, "label2": label2}
    return R

# ══════════════════════════════════════════════════════════════════════════════
# INDEPENDENT-SAMPLE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def run_independent(g1, g2, label1="Group 1", label2="Group 2",
                    dep_var="Score", alpha=0.05):
    R = {}
    g1, g2 = np.array(g1, dtype=float), np.array(g2, dtype=float)
    n1, n2 = len(g1), len(g2)
    m1, m2 = np.mean(g1), np.mean(g2)
    sd1, sd2 = np.std(g1, ddof=1), np.std(g2, ddof=1)
    se1, se2 = sd1/np.sqrt(n1), sd2/np.sqrt(n2)
    ci1_l, ci1_u = ci_mean(g1, alpha)
    ci2_l, ci2_u = ci_mean(g2, alpha)

    R["desc"] = pd.DataFrame([
        {"Group": label1, "N": n1, "Mean": m1,
         "Std. Deviation": sd1, "Std. Error Mean": se1,
         f"{int((1-alpha)*100)}% CI Lower": ci1_l,
         f"{int((1-alpha)*100)}% CI Upper": ci1_u,
         "Minimum": float(np.min(g1)), "Maximum": float(np.max(g1)),
         "Skewness": float(stats.skew(g1)),
         "Kurtosis": float(stats.kurtosis(g1))},
        {"Group": label2, "N": n2, "Mean": m2,
         "Std. Deviation": sd2, "Std. Error Mean": se2,
         f"{int((1-alpha)*100)}% CI Lower": ci2_l,
         f"{int((1-alpha)*100)}% CI Upper": ci2_u,
         "Minimum": float(np.min(g2)), "Maximum": float(np.max(g2)),
         "Skewness": float(stats.skew(g2)),
         "Kurtosis": float(stats.kurtosis(g2))},
    ])

    # Normality per group (SW + KS)
    n1_res = test_normality(g1, label1)
    n2_res = test_normality(g2, label2)
    R["normality"]  = [n1_res, n2_res]
    R["use_param"]  = n1_res["pass"] and n2_res["pass"]

    # Levene center='mean' (SPSS default)
    lev_f, lev_p = stats.levene(g1, g2, center='mean')
    R["levene"] = {
        "F": float(lev_f), "df1": 1, "df2": n1+n2-2,
        "Sig.": float(lev_p), "equal_var": float(lev_p) > alpha
    }

    # Parametric: equal-var + Welch
    t_eq,    p_eq    = stats.ttest_ind(g1, g2, equal_var=True)
    t_welch, p_welch = stats.ttest_ind(g1, g2, equal_var=False)
    df_eq    = n1 + n2 - 2
    df_welch = (sd1**2/n1 + sd2**2/n2)**2 / \
               ((sd1**2/n1)**2/(n1-1) + (sd2**2/n2)**2/(n2-1))
    mean_diff = m1 - m2
    se_eq     = np.sqrt(((n1-1)*sd1**2+(n2-1)*sd2**2)/df_eq*(1/n1+1/n2))
    se_welch  = np.sqrt(sd1**2/n1 + sd2**2/n2)
    tc_eq     = stats.t.ppf(1 - alpha/2, df_eq)
    tc_welch  = stats.t.ppf(1 - alpha/2, df_welch)
    d = cohens_d_ind(g1, g2)
    R["parametric"] = {
        "test": "Independent Samples T-Test",
        "label1": label1, "label2": label2, "dep_var": dep_var,
        "mean_diff": mean_diff,
        "t_eq": float(t_eq), "df_eq": df_eq, "p_eq": float(p_eq),
        "se_eq": se_eq,
        "ci_eq_l": mean_diff - tc_eq*se_eq,
        "ci_eq_u": mean_diff + tc_eq*se_eq,
        "t_welch": float(t_welch), "df_welch": df_welch,
        "p_welch": float(p_welch),
        "se_welch": se_welch,
        "ci_welch_l": mean_diff - tc_welch*se_welch,
        "ci_welch_u": mean_diff + tc_welch*se_welch,
        "cohens_d": d
    }

    # Non-parametric: Mann-Whitney SPSS-exact
    mw = mannwhitney_spss(g1, g2, label1, label2)
    R["nonparametric"] = {**mw, "test": "Mann-Whitney U Test", "dep_var": dep_var}
    return R

# ══════════════════════════════════════════════════════════════════════════════
# INTERPRETATION
# ══════════════════════════════════════════════════════════════════════════════
def interpret_one_sample(R, var_name, alpha):
    lines = []
    norm = R["normality"][0]; pr = R["parametric"]
    sw_ok = norm["sw_pass"]; ks_ok = norm["ks_pass"]
    lines.append(
        f"<b>Normality:</b> Shapiro-Wilk W = {_f(norm['sw_W'])}, "
        f"p = {_p(norm['sw_p'])} ({'normal' if sw_ok else 'non-normal'}); "
        f"Kolmogorov-Smirnov D = {_f(norm['ks_D'])}, "
        f"p = {_p(norm['ks_p'])} ({'normal' if ks_ok else 'non-normal'}). "
        f"{'Both tests indicate normality → parametric test applied.' if R['use_param'] else 'Normality violated → non-parametric test applied.'}"
    )
    if R["use_param"]:
        sig = pr["p_two"] < alpha
        lines.append(
            f"<b>One-Sample T-Test:</b> t({pr['df']}) = {_f(pr['t'])}, "
            f"p = {_p(pr['p_two'])} (2-tailed). "
            f"Mean difference = {_f(pr['mean_diff'])}, "
            f"95% CI [{_f(pr['ci_lower'])}, {_f(pr['ci_upper'])}], "
            f"Cohen's d = {_f(pr['cohens_d'])} ({effect_label_d(pr['cohens_d'])}). "
            f"The mean {'significantly differs from' if sig else 'does not significantly differ from'} "
            f"μ₀ = {pr['mu0']}."
        )
    else:
        np_r = R["nonparametric"]
        sig  = np_r["p"] < alpha if not np.isnan(np_r["p"]) else False
        lines.append(
            f"<b>Wilcoxon Signed-Rank Test:</b> W = {_f(np_r['W'],0)}, "
            f"Z = {_f(np_r['Z'])}, p = {_p(np_r['p'])} (2-tailed). "
            f"{'Significant' if sig else 'No significant'} difference from μ₀ = {np_r['mu0']}."
        )
    return lines

def interpret_paired(R, alpha):
    lines = []
    norm = R["normality"][0]; pr = R["parametric"]
    corr = R["correlation"]
    r_v  = float(corr["Pearson Correlation"].iloc[0])
    p_v  = float(corr["Sig. (2-tailed)"].iloc[0])
    sw_ok = norm["sw_pass"]; ks_ok = norm["ks_pass"]
    lines.append(
        f"<b>Normality of Differences:</b> "
        f"Shapiro-Wilk W = {_f(norm['sw_W'])}, p = {_p(norm['sw_p'])} "
        f"({'normal' if sw_ok else 'non-normal'}); "
        f"Kolmogorov-Smirnov D = {_f(norm['ks_D'])}, p = {_p(norm['ks_p'])} "
        f"({'normal' if ks_ok else 'non-normal'}). "
        f"{'Parametric analysis applied.' if R['use_param'] else 'Non-parametric analysis applied.'}"
    )
    lines.append(
        f"<b>Paired Correlation:</b> {pr['label1']} and {pr['label2']} were "
        f"{'significantly' if p_v < alpha else 'not significantly'} correlated, "
        f"r({pr['df']}) = {_f(r_v)}, p = {_p(p_v)}."
    )
    if R["use_param"]:
        sig = pr["p_two"] < alpha
        lines.append(
            f"<b>Paired T-Test:</b> t({pr['df']}) = {_f(pr['t'])}, "
            f"p = {_p(pr['p_two'])} (2-tailed). "
            f"Mean difference = {_f(pr['mean_diff'])} (SD = {_f(pr['sd_diff'])}), "
            f"95% CI [{_f(pr['ci_lower'])}, {_f(pr['ci_upper'])}], "
            f"Cohen's d = {_f(pr['cohens_d'])} ({effect_label_d(pr['cohens_d'])}). "
            f"{'Significant difference found.' if sig else 'No significant difference found.'}"
        )
    else:
        np_r = R["nonparametric"]
        sig  = np_r["p"] < alpha if not np.isnan(np_r["p"]) else False
        lines.append(
            f"<b>Wilcoxon Signed-Rank Test:</b> W = {_f(np_r['W'],0)}, "
            f"Z = {_f(np_r['Z'])}, p = {_p(np_r['p'])} (2-tailed). "
            f"{'Significant' if sig else 'No significant'} difference between "
            f"{pr['label1']} and {pr['label2']}."
        )
    return lines

def interpret_independent(R, dep_var, alpha):
    lines = []
    norms = R["normality"]; lev = R["levene"]; pr = R["parametric"]
    for n in norms:
        sw_ok = n["sw_pass"]; ks_ok = n["ks_pass"]
        lines.append(
            f"<b>Normality — {n['label']}:</b> "
            f"Shapiro-Wilk W = {_f(n['sw_W'])}, p = {_p(n['sw_p'])} "
            f"({'normal' if sw_ok else 'non-normal'}); "
            f"KS D = {_f(n['ks_D'])}, p = {_p(n['ks_p'])} "
            f"({'normal' if ks_ok else 'non-normal'})."
        )
    lines.append(
        f"<b>Overall:</b> "
        f"{'Both groups are normally distributed → parametric test applied.' if R['use_param'] else 'Non-normality detected → non-parametric test applied.'}"
    )
    lines.append(
        f"<b>Levene's Test:</b> F({lev['df1']}, {lev['df2']}) = {_f(lev['F'])}, "
        f"p = {_p(lev['Sig.'])}. "
        f"{'Equal variances assumed.' if lev['equal_var'] else 'Equal variances NOT assumed → Welch correction.'}"
    )
    if R["use_param"]:
        use_eq = lev["equal_var"]
        tv  = pr["t_eq"]    if use_eq else pr["t_welch"]
        pv  = pr["p_eq"]    if use_eq else pr["p_welch"]
        dfv = pr["df_eq"]   if use_eq else pr["df_welch"]
        cil = pr["ci_eq_l"] if use_eq else pr["ci_welch_l"]
        ciu = pr["ci_eq_u"] if use_eq else pr["ci_welch_u"]
        sig = pv < alpha
        lines.append(
            f"<b>Independent T-Test ({'equal var.' if use_eq else 'Welch'}):</b> "
            f"t({_f(dfv,2)}) = {_f(tv)}, p = {_p(pv)} (2-tailed). "
            f"Mean diff = {_f(pr['mean_diff'])}, "
            f"95% CI [{_f(cil)}, {_f(ciu)}], "
            f"Cohen's d = {_f(pr['cohens_d'])} ({effect_label_d(pr['cohens_d'])}). "
            f"{'Significant difference.' if sig else 'No significant difference.'}"
        )
    else:
        np_r = R["nonparametric"]
        sig  = np_r["p"] < alpha if not np.isnan(np_r["p"]) else False
        lines.append(
            f"<b>Mann-Whitney U:</b> U = {_f(np_r['U'],0)}, "
            f"W = {_f(np_r['W_wilcoxon'],1)}, Z = {_f(np_r['Z'])}, "
            f"p = {_p(np_r['p'])} (2-tailed). "
            f"{'Significant difference.' if sig else 'No significant difference.'}"
        )
    return lines

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
PAL = ["#1a1a2e","#e94560","#0f3460","#16213e"]

def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def plot_one_sample(data, mu0, var_name):
    data = np.array(data, dtype=float)
    fig  = plt.figure(figsize=(13, 4), facecolor="#f8fafc")
    gs   = gridspec.GridSpec(1, 3, figure=fig, wspace=.35)

    ax1 = fig.add_subplot(gs[0]); ax1.set_facecolor("#f8fafc")
    ax1.hist(data, bins="auto", color=PAL[1], alpha=0.75, edgecolor="white")
    xr  = np.linspace(data.min()-1, data.max()+1, 200)
    sc  = len(data)*(data.max()-data.min())/max(len(data)//3, 1)
    ax1.plot(xr, stats.norm.pdf(xr, data.mean(), data.std(ddof=1))*sc,
             color=PAL[0], lw=2)
    ax1.axvline(mu0, color="#e94560", ls="--", lw=1.8, label=f"μ₀={mu0}")
    ax1.axvline(data.mean(), color=PAL[2], lw=1.8, label=f"x̄={data.mean():.2f}")
    ax1.set_xlabel(var_name, fontsize=8); ax1.set_ylabel("Frequency", fontsize=8)
    ax1.set_title("Distribution", fontsize=9, fontweight="bold", color=PAL[0])
    ax1.legend(fontsize=7); ax1.spines[["top","right"]].set_visible(False)

    ax2 = fig.add_subplot(gs[1]); ax2.set_facecolor("#f8fafc")
    (osm, osr),(sl, ic, _) = stats.probplot(data)
    ax2.plot(osm, osr, "o", color=PAL[1], markersize=5, markeredgecolor="white", alpha=.8)
    ax2.plot(osm, sl*np.array(osm)+ic, "--", color=PAL[0], lw=1.5)
    ax2.set_xlabel("Theoretical Quantiles", fontsize=8)
    ax2.set_ylabel("Sample Quantiles", fontsize=8)
    ax2.set_title("Normal Q-Q", fontsize=9, fontweight="bold", color=PAL[0])
    ax2.spines[["top","right"]].set_visible(False)

    ax3 = fig.add_subplot(gs[2]); ax3.set_facecolor("#f8fafc")
    ax3.boxplot(data, patch_artist=True, widths=0.5,
                medianprops={"color":"white","linewidth":2},
                boxprops={"facecolor":PAL[1],"alpha":0.8})
    ax3.axhline(mu0, color="#e94560", ls="--", lw=1.8, label=f"μ₀={mu0}")
    ax3.set_xticklabels([var_name], fontsize=8)
    ax3.set_ylabel("Value", fontsize=8)
    ax3.set_title("Box Plot", fontsize=9, fontweight="bold", color=PAL[0])
    ax3.legend(fontsize=7); ax3.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); return fig

def plot_paired(data1, data2, label1, label2):
    data1, data2 = np.array(data1, dtype=float), np.array(data2, dtype=float)
    diff = data1 - data2
    fig  = plt.figure(figsize=(13, 4), facecolor="#f8fafc")
    gs   = gridspec.GridSpec(1, 3, figure=fig, wspace=.35)

    ax1 = fig.add_subplot(gs[0]); ax1.set_facecolor("#f8fafc")
    for v1, v2 in zip(data1, data2):
        ax1.plot([0,1],[v1,v2],"-o",
                 color=PAL[0] if v2>=v1 else PAL[1], alpha=0.45, markersize=4)
    ax1.plot([0,1],[data1.mean(),data2.mean()],"-o",
             color=PAL[1], lw=3, markersize=8)
    ax1.set_xticks([0,1]); ax1.set_xticklabels([label1,label2], fontsize=9)
    ax1.set_ylabel("Value", fontsize=8)
    ax1.set_title("Individual Changes", fontsize=9, fontweight="bold", color=PAL[0])
    ax1.spines[["top","right"]].set_visible(False)

    ax2 = fig.add_subplot(gs[1]); ax2.set_facecolor("#f8fafc")
    ax2.hist(diff, bins="auto", color=PAL[1], alpha=0.75, edgecolor="white")
    ax2.axvline(0, color=PAL[0], ls="--", lw=1.8)
    ax2.axvline(diff.mean(), color=PAL[2], lw=1.8,
                label=f"Mean={diff.mean():.2f}")
    ax2.set_xlabel("Difference", fontsize=8); ax2.set_ylabel("Frequency", fontsize=8)
    ax2.set_title("Distribution of Differences", fontsize=9,
                  fontweight="bold", color=PAL[0])
    ax2.legend(fontsize=7); ax2.spines[["top","right"]].set_visible(False)

    ax3 = fig.add_subplot(gs[2]); ax3.set_facecolor("#f8fafc")
    ax3.boxplot([data1, data2], patch_artist=True,
                boxprops={"facecolor":PAL[1],"alpha":0.75},
                medianprops={"color":"white","linewidth":2}, widths=0.5)
    ax3.set_xticklabels([label1,label2], fontsize=8)
    ax3.set_ylabel("Value", fontsize=8)
    ax3.set_title("Box Plots", fontsize=9, fontweight="bold", color=PAL[0])
    ax3.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); return fig

def plot_independent(g1, g2, label1, label2, dep_var):
    g1, g2 = np.array(g1, dtype=float), np.array(g2, dtype=float)
    fig    = plt.figure(figsize=(13, 4), facecolor="#f8fafc")
    gs     = gridspec.GridSpec(1, 3, figure=fig, wspace=.35)

    ax1 = fig.add_subplot(gs[0]); ax1.set_facecolor("#f8fafc")
    bp  = ax1.boxplot([g1, g2], patch_artist=True, widths=0.5,
                      medianprops={"color":"white","linewidth":2})
    for patch, c in zip(bp["boxes"], [PAL[0], PAL[1]]):
        patch.set_facecolor(c); patch.set_alpha(0.8)
    ax1.set_xticklabels([label1, label2], fontsize=8)
    ax1.set_ylabel(dep_var, fontsize=8)
    ax1.set_title("Box Plots by Group", fontsize=9, fontweight="bold", color=PAL[0])
    ax1.spines[["top","right"]].set_visible(False)

    ax2 = fig.add_subplot(gs[1]); ax2.set_facecolor("#f8fafc")
    bins = np.linspace(min(g1.min(),g2.min())-.5,
                       max(g1.max(),g2.max())+.5, 15)
    ax2.hist(g1, bins=bins, color=PAL[0], alpha=0.65,
             label=label1, edgecolor="white")
    ax2.hist(g2, bins=bins, color=PAL[1], alpha=0.65,
             label=label2, edgecolor="white")
    ax2.set_xlabel(dep_var, fontsize=8); ax2.set_ylabel("Frequency", fontsize=8)
    ax2.set_title("Distribution by Group", fontsize=9, fontweight="bold", color=PAL[0])
    ax2.legend(fontsize=7); ax2.spines[["top","right"]].set_visible(False)

    ax3 = fig.add_subplot(gs[2]); ax3.set_facecolor("#f8fafc")
    for gd, c, lb in [(g1,PAL[0],label1),(g2,PAL[1],label2)]:
        (osm, osr),(sl, ic, _) = stats.probplot(gd)
        ax3.plot(osm, osr, "o", color=c, markersize=4, alpha=.75, label=lb)
        ax3.plot(osm, sl*np.array(osm)+ic, "--", color=c, lw=1.2)
    ax3.set_xlabel("Theoretical Quantiles", fontsize=8)
    ax3.set_ylabel("Sample Quantiles", fontsize=8)
    ax3.set_title("Normal Q-Q by Group", fontsize=9, fontweight="bold", color=PAL[0])
    ax3.legend(fontsize=7); ax3.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); return fig

# ══════════════════════════════════════════════════════════════════════════════
# HTML TABLE HELPER
# ══════════════════════════════════════════════════════════════════════════════
def html_tbl(rows, left_cols=None):
    if left_cols is None:
        left_cols = {0}
    html = '<div class="spss-wrap"><table class="spss-tbl"><thead><tr>'
    for h in rows[0]:
        html += f"<th>{h}</th>"
    html += "</tr></thead><tbody>"
    for row in rows[1:]:
        html += "<tr>"
        for i, v in enumerate(row):
            cls = ' class="left"' if i in left_cols else ""
            html += f"<td{cls}>{v}</td>"
        html += "</tr>"
    html += "</tbody></table></div>"
    return html

def df_to_rows(df):
    rows = [list(df.columns)]
    for _, r in df.iterrows():
        rows.append([str(v) for v in r.values])
    return rows

# ══════════════════════════════════════════════════════════════════════════════
# PDF GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def build_pdf(test_type, R, meta, interps, fig_bytes_list):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                             rightMargin=1.8*cm, leftMargin=1.8*cm,
                             topMargin=2*cm, bottomMargin=2*cm)

    H1  = ParagraphStyle("H1", fontSize=12, fontName="Helvetica-Bold",
                          textColor=colors.white,
                          backColor=colors.HexColor("#1a1a2e"),
                          spaceAfter=5, spaceBefore=12,
                          borderPadding=(5,8,5,8))
    H2  = ParagraphStyle("H2", fontSize=10, fontName="Helvetica-Bold",
                          textColor=colors.HexColor("#1a1a2e"),
                          spaceAfter=3, spaceBefore=8)
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
    BD  = ParagraphStyle("BD", fontSize=8.5, fontName="Helvetica",
                          leading=13, spaceAfter=4)

    TS = TableStyle([
        ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",(0,0),(-1,0),  colors.white),
        ("FONTNAME",(0,0),(-1,0),   "Helvetica-Bold"),
        ("FONTNAME",(0,1),(-1,-1),  "Helvetica"),
        ("FONTSIZE",(0,0),(-1,-1),  7.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f8fafc")]),
        ("GRID",(0,0),(-1,-1), .5, colors.HexColor("#e2e8f0")),
        ("ALIGN",(1,0),(-1,-1),    "CENTER"),
        ("ALIGN",(0,0),(0,-1),     "LEFT"),
        ("VALIGN",(0,0),(-1,-1),   "MIDDLE"),
        ("TOPPADDING",(0,0),(-1,-1),    4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING",(0,0),(-1,-1),   6),
        ("RIGHTPADDING",(0,0),(-1,-1),  6),
    ])

    def mktbl(rows_data):
        t = Table(rows_data, repeatRows=1)
        t.setStyle(TS)
        return t

    story = []
    story.append(Spacer(1, .4*cm))
    story.append(Paragraph("INFERENTIAL STATISTICS REPORT", TIT))
    story.append(Paragraph(
        "SPSS-Equivalent · Shapiro-Wilk + Kolmogorov-Smirnov · "
        "Parametric & Non-Parametric", SUB))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor("#e94560")))
    story.append(Spacer(1, 6))

    mt = Table([[k, str(v)] for k,v in meta.items()],
               colWidths=[5*cm, 11*cm])
    mt.setStyle(TableStyle([
        ("FONTNAME",(0,0),(0,-1),"Helvetica-Bold"),
        ("FONTNAME",(1,0),(1,-1),"Helvetica"),
        ("FONTSIZE",(0,0),(-1,-1),8.5),
        ("TEXTCOLOR",(0,0),(0,-1),colors.HexColor("#1a1a2e")),
        ("TOPPADDING",(0,0),(-1,-1),3),
        ("BOTTOMPADDING",(0,0),(-1,-1),3)]))
    story.append(mt)
    story.append(Spacer(1,6))
    story.append(HRFlowable(width="100%",thickness=.5,
                             color=colors.HexColor("#e2e8f0")))

    sec = 1

    # 1. Normality (both SW + KS)
    story.append(Paragraph(f"  {sec}. TESTS OF NORMALITY", H1)); sec += 1
    story.append(Paragraph(
        "a) Shapiro-Wilk Test", H2))
    sw_rows = [["Variable","N","Statistic (W)","Sig.","Result"]]
    for n in R["normality"]:
        sw_rows.append([n["label"], str(n["n"]), _f(n["sw_W"]), _p(n["sw_p"]),
                        "Normal" if n["sw_pass"] else "Non-Normal"])
    story.append(mktbl(sw_rows))

    story.append(Paragraph(
        "b) Kolmogorov-Smirnov Test (Lilliefors Significance Correction)", H2))
    ks_rows = [["Variable","N","Statistic (D)","Sig.ᵃ","Result"]]
    for n in R["normality"]:
        ks_rows.append([n["label"], str(n["n"]), _f(n["ks_D"]), _p(n["ks_p"]),
                        "Normal" if n["ks_pass"] else "Non-Normal"])
    story.append(mktbl(ks_rows))
    story.append(Paragraph(
        "ᵃ Lilliefors Significance Correction applied. "
        f"{'Parametric' if R['use_param'] else 'Non-parametric'} analysis selected "
        "(pass = BOTH tests p > .05).", NT))

    # 2. Descriptives
    story.append(Paragraph(f"  {sec}. DESCRIPTIVE STATISTICS", H1)); sec += 1
    desc = R["desc"].copy()
    for c in desc.select_dtypes(include=float).columns:
        desc[c] = desc[c].apply(_f)
    story.append(mktbl(df_to_rows(desc)))

    # 3. Paired correlation (paired only)
    if "correlation" in R:
        story.append(Paragraph(f"  {sec}. PAIRED SAMPLES CORRELATIONS", H1)); sec += 1
        corr = R["correlation"].copy()
        corr["Pearson Correlation"] = corr["Pearson Correlation"].apply(_f)
        corr["Sig. (2-tailed)"]     = corr["Sig. (2-tailed)"].apply(_p)
        story.append(mktbl(df_to_rows(corr)))

    # 4. Levene (independent only)
    if "levene" in R:
        story.append(Paragraph(
            f"  {sec}. LEVENE'S TEST FOR EQUALITY OF VARIANCES", H1)); sec += 1
        lev = R["levene"]
        story.append(mktbl([
            ["F","df1","df2","Sig.","Result"],
            [_f(lev["F"]), str(lev["df1"]), str(lev["df2"]), _p(lev["Sig."]),
             "Equal var. assumed" if lev["equal_var"] else "Equal var. NOT assumed"]
        ]))
        story.append(Paragraph("Note. Based on mean (SPSS default).", NT))

    # 5. Parametric results
    story.append(Paragraph(f"  {sec}. PARAMETRIC TEST RESULTS", H1)); sec += 1
    pr = R["parametric"]
    if test_type == "One-Sample T-Test":
        story.append(mktbl([
            ["","t","df","Sig.(2-tail)","Sig.(1-tail L)","Sig.(1-tail U)",
             "Mean Diff","95% CI Lower","95% CI Upper","Cohen's d"],
            [f"Test value = {pr['mu0']}",
             _f(pr["t"]), str(pr["df"]), _p(pr["p_two"]),
             _p(pr["p_one_lower"]), _p(pr["p_one_upper"]),
             _f(pr["mean_diff"]), _f(pr["ci_lower"]), _f(pr["ci_upper"]),
             _f(pr["cohens_d"])]
        ]))
    elif test_type == "Paired-Sample T-Test":
        # correlation sub-table repeated for context
        story.append(Paragraph("Paired Samples Correlations", H2))
        corr2 = R["correlation"].copy()
        corr2["Pearson Correlation"] = corr2["Pearson Correlation"].apply(_f)
        corr2["Sig. (2-tailed)"]     = corr2["Sig. (2-tailed)"].apply(_p)
        story.append(mktbl(df_to_rows(corr2)))
        story.append(Paragraph("Paired Samples Test", H2))
        story.append(mktbl([
            ["Pair","Mean Diff","SD","SE","95% CI L","95% CI U",
             "t","df","Sig.(2-tail)","Sig.(1-tail L)","Sig.(1-tail U)","Cohen's d"],
            [f"{pr['label1']} – {pr['label2']}",
             _f(pr["mean_diff"]), _f(pr["sd_diff"]), _f(pr["se_diff"]),
             _f(pr["ci_lower"]), _f(pr["ci_upper"]),
             _f(pr["t"]), str(pr["df"]),
             _p(pr["p_two"]), _p(pr["p_one_lower"]), _p(pr["p_one_upper"]),
             _f(pr["cohens_d"])]
        ]))
    else:
        lev = R["levene"]
        story.append(mktbl([
            ["","F(Levene)","Sig.","t","df","Sig.(2-tail)",
             "Mean Diff","SE Diff","95% CI L","95% CI U","Cohen's d"],
            ["Equal var. assumed",
             _f(lev["F"]), _p(lev["Sig."]),
             _f(pr["t_eq"]), str(pr["df_eq"]), _p(pr["p_eq"]),
             _f(pr["mean_diff"]), _f(pr["se_eq"]),
             _f(pr["ci_eq_l"]), _f(pr["ci_eq_u"]), _f(pr["cohens_d"])],
            ["Equal var. NOT assumed","","",
             _f(pr["t_welch"]), _f(pr["df_welch"],2), _p(pr["p_welch"]),
             _f(pr["mean_diff"]), _f(pr["se_welch"]),
             _f(pr["ci_welch_l"]), _f(pr["ci_welch_u"]), "—"]
        ]))

    # 6. Non-parametric results
    story.append(Paragraph(f"  {sec}. NON-PARAMETRIC TEST RESULTS", H1)); sec += 1
    np_r = R["nonparametric"]
    if test_type == "Independent-Sample T-Test":
        story.append(Paragraph("Ranks", H2))
        story.append(mktbl([
            ["Group","N","Mean Rank","Sum of Ranks"],
            [np_r["label1"], str(np_r["n1"]),
             _f(np_r["mean_rank1"]), _f(np_r["R1"],1)],
            [np_r["label2"], str(np_r["n2"]),
             _f(np_r["mean_rank2"]), _f(np_r["R2"],1)],
            ["Total", str(np_r["n1"]+np_r["n2"]), "", ""]
        ]))
        story.append(Paragraph("Test Statistics", H2))
        story.append(mktbl([
            ["Statistic","Value"],
            ["Mann-Whitney U",          _f(np_r["U"],0)],
            ["Wilcoxon W",              _f(np_r["W_wilcoxon"],1)],
            ["Z",                       _f(np_r["Z"])],
            ["Asymp. Sig. (2-tailed)",  _p(np_r["p"])],
        ]))
    else:
        n_neg  = np_r.get("n_neg",0)
        n_pos  = np_r.get("n_pos",0)
        n_ties = np_r.get("n_ties",0)
        n_tot  = np_r.get("n_total", n_neg+n_pos+n_ties)
        neg_rs = np_r.get("neg_rank_sum", np.nan)
        pos_rs = np_r.get("pos_rank_sum", np.nan)
        neg_mr = neg_rs/n_neg if n_neg > 0 else np.nan
        pos_mr = pos_rs/n_pos if n_pos > 0 else np.nan
        story.append(Paragraph("Ranks", H2))
        story.append(mktbl([
            ["","N","Mean Rank","Sum of Ranks"],
            ["Negative Ranks", str(n_neg), _f(neg_mr), _f(neg_rs,1)],
            ["Positive Ranks", str(n_pos), _f(pos_mr), _f(pos_rs,1)],
            ["Ties",           str(n_ties),"",""],
            ["Total",          str(n_tot), "",""]
        ]))
        pair_lbl = (f"{pr['label1']} − {pr['label2']}"
                    if test_type == "Paired-Sample T-Test"
                    else "Variable − μ₀")
        story.append(Paragraph("Test Statistics", H2))
        story.append(mktbl([
            ["Statistic", pair_lbl],
            ["Test Statistic (W)", _f(np_r["W"],0)],
            ["Z",                  _f(np_r["Z"])],
            ["Asymp. Sig. (2-tailed)", _p(np_r["p"])],
        ]))

    # 7. Interpretation
    story.append(Paragraph(f"  {sec}. INTERPRETATION", H1)); sec += 1
    for line in interps:
        clean = (line.replace("<b>","").replace("</b>","")
                     .replace("<i>","").replace("</i>",""))
        story.append(Paragraph(clean, IT))

    # 8. Figures
    story.append(PageBreak())
    story.append(Paragraph(f"  {sec}. FIGURES", H1))
    for i, fb in enumerate(fig_bytes_list, 1):
        story.append(Image(io.BytesIO(fb), width=17*cm, height=5*cm))
        story.append(Paragraph(f"Figure {i}. Diagnostic plots.", NT))
        story.append(Spacer(1, 8))

    story.append(HRFlowable(width="100%", thickness=.5,
                             color=colors.HexColor("#e2e8f0")))
    story.append(Paragraph(
        "Generated by Inferential Statistics App · SPSS-equivalent · "
        "KS with Lilliefors correction · Levene center=mean · "
        "Wilcoxon Z ties-corrected · Mann-Whitney SPSS exact", NT))

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.markdown("""
    <div class="main-hdr">
      <h1>📐 Inferential Statistics Suite
        <span class="badge">SPSS-Equivalent</span></h1>
      <p>Parametric & Non-Parametric · Shapiro-Wilk + Kolmogorov-Smirnov ·
         Auto-selection · One-Sample · Paired · Independent</p>
    </div>""", unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        test_type = st.selectbox("📊 Select Test", [
            "One-Sample T-Test",
            "Paired-Sample T-Test",
            "Independent-Sample T-Test"
        ])
        alpha = st.selectbox("α Level", [0.05, 0.01, 0.001], index=0)
        st.markdown("---")

        samp = SAMPLES[test_type]
        st.markdown(f"**📄 Template — {test_type}**")
        st.markdown(samp["note"])
        st.download_button(
            "⬇️ Download Sample CSV",
            samp["csv"].encode(),
            f"sample_{test_type.replace(' ','_').replace('-','_').lower()}.csv",
            "text/csv", use_container_width=True)
        st.markdown("---")

        uploaded = st.file_uploader("📂 Upload CSV", type=["csv"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.success(f"✅ {len(df)} rows × {len(df.columns)} cols")
            except Exception as e:
                st.error(f"Error: {e}"); df = None
        else:
            df = pd.read_csv(io.StringIO(samp["csv"]))
            st.info("ℹ️ Using built-in sample data")

        cfg = None
        if df is not None:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
            st.markdown("---")

            if test_type == "One-Sample T-Test":
                tv  = st.selectbox("🎯 Test Variable", num_cols,
                                    index=num_cols.index("score")
                                    if "score" in num_cols else 0)
                mu0 = st.number_input("📏 Test Value (μ₀)", value=75.0, step=0.5)
                cfg = {"test_var": tv, "mu0": mu0}

            elif test_type == "Paired-Sample T-Test":
                v1 = st.selectbox("Variable 1 (Pre)", num_cols,
                                   index=num_cols.index("pre_score")
                                   if "pre_score" in num_cols else 0)
                v2 = st.selectbox("Variable 2 (Post)", num_cols,
                                   index=num_cols.index("post_score")
                                   if "post_score" in num_cols
                                   else min(1, len(num_cols)-1))
                cfg = {"v1": v1, "v2": v2}

            else:
                gc = st.selectbox("👥 Grouping Variable",
                                   cat_cols if cat_cols else num_cols,
                                   index=cat_cols.index("group")
                                   if "group" in cat_cols else 0)
                dc = st.selectbox("🎯 Dependent Variable", num_cols,
                                   index=num_cols.index("score")
                                   if "score" in num_cols else 0)
                groups = sorted(df[gc].dropna().unique())
                if len(groups) >= 2:
                    g1l = st.selectbox("Group 1", groups, index=0)
                    g2l = st.selectbox("Group 2", groups,
                                        index=min(1, len(groups)-1))
                    cfg = {"grp_col": gc, "dep_col": dc, "g1": g1l, "g2": g2l}
                else:
                    st.error("Need ≥ 2 groups")

            st.markdown("---")
            run_btn = st.button("🚀 Run Analysis", type="primary",
                                 use_container_width=True)
        else:
            run_btn = False

    if df is None:
        return

    with st.expander("🔍 Data Preview", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    if not run_btn and "stats_R" not in st.session_state:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#f0f9ff,#e0f2fe);
             border:1px solid #bae6fd;border-radius:12px;padding:1.2rem 1.4rem;
             margin:.8rem 0;border-left:4px solid #0284c7;">
          <h4 style="margin:0 0 .4rem 0;font-size:.95rem;font-weight:700;">
            📋 {test_type}</h4>
          <p style="margin:0;font-size:.84rem;color:#475569;">{samp['desc']}</p>
        </div>""", unsafe_allow_html=True)
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
                            "α": alpha,
                            "Date": datetime.now().strftime("%B %d, %Y %H:%M")}
                    interps  = interpret_one_sample(R, cfg["test_var"], alpha)
                    fig_main = plot_one_sample(data, cfg["mu0"], cfg["test_var"])

                elif test_type == "Paired-Sample T-Test":
                    pdf = df[[cfg["v1"], cfg["v2"]]].dropna()
                    d1  = pdf[cfg["v1"]].values.tolist()
                    d2  = pdf[cfg["v2"]].values.tolist()
                    R   = run_paired(d1, d2, cfg["v1"], cfg["v2"], alpha)
                    meta = {"Test": test_type,
                            "Variable 1": cfg["v1"], "Variable 2": cfg["v2"],
                            "N pairs": len(d1), "α": alpha,
                            "Date": datetime.now().strftime("%B %d, %Y %H:%M")}
                    interps  = interpret_paired(R, alpha)
                    fig_main = plot_paired(d1, d2, cfg["v1"], cfg["v2"])

                else:
                    g1d = (df[df[cfg["grp_col"]]==cfg["g1"]]
                           [cfg["dep_col"]].dropna().values.tolist())
                    g2d = (df[df[cfg["grp_col"]]==cfg["g2"]]
                           [cfg["dep_col"]].dropna().values.tolist())
                    R   = run_independent(g1d, g2d, cfg["g1"], cfg["g2"],
                                          cfg["dep_col"], alpha)
                    meta = {"Test": test_type,
                            "Grouping Variable": cfg["grp_col"],
                            "Dependent Variable": cfg["dep_col"],
                            "Group 1": f"{cfg['g1']} (n={len(g1d)})",
                            "Group 2": f"{cfg['g2']} (n={len(g2d)})",
                            "α": alpha,
                            "Date": datetime.now().strftime("%B %d, %Y %H:%M")}
                    interps  = interpret_independent(R, cfg["dep_col"], alpha)
                    fig_main = plot_independent(g1d, g2d, cfg["g1"], cfg["g2"],
                                               cfg["dep_col"])

            except Exception as e:
                st.error(f"Analysis error: {e}")
                import traceback; st.code(traceback.format_exc())
                return

        figs_b = [fig_to_bytes(fig_main)]; plt.close(fig_main)
        st.session_state.update({
            "stats_R": R, "stats_meta": meta, "stats_interps": interps,
            "stats_test": test_type, "stats_cfg": cfg,
            "stats_figs": figs_b, "stats_df": df, "stats_alpha": alpha
        })

    R         = st.session_state["stats_R"]
    meta      = st.session_state["stats_meta"]
    interps   = st.session_state["stats_interps"]
    test_type = st.session_state["stats_test"]
    figs_b    = st.session_state["stats_figs"]
    alpha     = st.session_state.get("stats_alpha", 0.05)
    if R is None: return

    st.success("✅ Analysis complete!")

    # Decision banner
    use_p = R["use_param"]
    test_name_used = (
        {"One-Sample T-Test":       "One-Sample T-Test",
         "Paired-Sample T-Test":    "Paired T-Test",
         "Independent-Sample T-Test":"Independent T-Test"}[test_type]
        if use_p else
        {"One-Sample T-Test":       "Wilcoxon Signed-Rank",
         "Paired-Sample T-Test":    "Wilcoxon Signed-Rank",
         "Independent-Sample T-Test":"Mann-Whitney U"}[test_type]
    )
    cls = "use-param" if use_p else "use-nonparam"
    st.markdown(
        f'<div class="decision-banner {cls}">'
        f'{"✅" if use_p else "⚠️"} '
        f'SW & KS: both p {">" if use_p else "≤"} .05 → '
        f'<b>{test_name_used}</b> selected automatically'
        f'</div>', unsafe_allow_html=True)

    # Quick metrics
    pr   = R["parametric"]
    np_r = R["nonparametric"]
    if use_p:
        if test_type == "One-Sample T-Test":
            metrics = [(_f(pr["t"]),f"t({pr['df']})"),
                       (_p(pr["p_two"]),"Sig. (2-tailed)"),
                       (_f(pr["mean_diff"]),"Mean Diff"),
                       (_f(pr["cohens_d"]),"Cohen's d"),
                       (effect_label_d(pr["cohens_d"]).title(),"Effect Size")]
        elif test_type == "Paired-Sample T-Test":
            metrics = [(_f(pr["t"]),f"t({pr['df']})"),
                       (_p(pr["p_two"]),"Sig. (2-tailed)"),
                       (_f(pr["mean_diff"]),"Mean Diff"),
                       (_f(pr["sd_diff"]),"SD of Diff"),
                       (_f(pr["cohens_d"]),"Cohen's d")]
        else:
            metrics = [(_f(pr["t_eq"]),f"t({pr['df_eq']}) Equal"),
                       (_f(pr["t_welch"]),"t Welch"),
                       (_p(pr["p_eq"]),"Sig. Equal"),
                       (_p(pr["p_welch"]),"Sig. Welch"),
                       (_f(pr["cohens_d"]),"Cohen's d")]
    else:
        if test_type == "Independent-Sample T-Test":
            metrics = [(_f(np_r["U"],0),"Mann-Whitney U"),
                       (_f(np_r["W_wilcoxon"],1),"Wilcoxon W"),
                       (_f(np_r["Z"]),"Z"),
                       (_p(np_r["p"]),"Sig. (2-tailed)"),
                       (str(np_r["n1"]+np_r["n2"]),"Total N")]
        else:
            metrics = [(_f(np_r["W"],0),"Wilcoxon W"),
                       (_f(np_r["Z"]),"Z"),
                       (_p(np_r["p"]),"Sig. (2-tailed)"),
                       (str(np_r.get("n_total","—")),"Total N"),
                       (str(np_r.get("n_ties",0)),"Ties")]

    cols = st.columns(5)
    for col, (val, lbl) in zip(cols, metrics):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-val">{val}</div>'
            f'<div class="metric-lbl">{lbl}</div>'
            f'</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab_labels = ["📋 Normality", "📊 Descriptives"]
    if "correlation" in R: tab_labels.append("🔗 Paired Correlation")
    if "levene"      in R: tab_labels.append("⚖️ Levene's Test")
    tab_labels += ["📈 Parametric Results", "📉 Non-Parametric Results",
                   "📈 Plots", "💬 Interpretation"]
    tabs = st.tabs(tab_labels)
    ti   = 0

    # ── Tab: Normality (SW + KS) ──────────────────────────────────────────────
    _t0 = ti; ti += 1
    with tabs[_t0]:
        # Shapiro-Wilk
        st.markdown(
            '<div class="sec-title">Shapiro-Wilk Test</div>',
            unsafe_allow_html=True)
        sw_rows = [["Variable","N","Statistic (W)","Sig.","Result"]]
        for n in R["normality"]:
            res = ('<span class="pass">✓ Normal</span>'
                   if n["sw_pass"] else '<span class="fail">✗ Non-Normal</span>')
            sw_rows.append([n["label"], str(n["n"]),
                            _f(n["sw_W"]), _p(n["sw_p"]), res])
        st.markdown(html_tbl(sw_rows, left_cols={0,4}), unsafe_allow_html=True)

        # Kolmogorov-Smirnov (Lilliefors)
        st.markdown(
            '<div class="sec-title">'
            'Kolmogorov-Smirnov Test (Lilliefors Significance Correction)'
            '</div>', unsafe_allow_html=True)
        ks_rows = [["Variable","N","Statistic (D)","Sig.ᵃ","Result"]]
        for n in R["normality"]:
            res = ('<span class="pass">✓ Normal</span>'
                   if n["ks_pass"] else '<span class="fail">✗ Non-Normal</span>')
            ks_rows.append([n["label"], str(n["n"]),
                            _f(n["ks_D"]), _p(n["ks_p"]), res])
        st.markdown(html_tbl(ks_rows, left_cols={0,4}), unsafe_allow_html=True)
        st.markdown(
            '<p class="note-txt">'
            'ᵃ Lilliefors Significance Correction. '
            'Parametric test applied only when BOTH Shapiro-Wilk AND '
            'Kolmogorov-Smirnov p > .05.</p>',
            unsafe_allow_html=True)

        if use_p:
            st.markdown(
                '<div class="info-box">✅ <b>Both normality tests passed.</b> '
                'Parametric analysis applied.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="warn-box">⚠️ <b>Normality violated</b> '
                '(at least one test p ≤ .05). '
                'Non-parametric analysis applied.</div>',
                unsafe_allow_html=True)

    # ── Tab: Descriptives ─────────────────────────────────────────────────────
    _t1 = ti; ti += 1
    with tabs[_t1]:
        st.markdown('<div class="sec-title">Descriptive Statistics</div>',
                    unsafe_allow_html=True)
        dd = R["desc"].copy()
        for c in dd.select_dtypes(include=[float,np.float64]).columns:
            dd[c] = dd[c].apply(_f)
        st.markdown(html_tbl(df_to_rows(dd)), unsafe_allow_html=True)

    # ── Tab: Paired Correlation ───────────────────────────────────────────────
    if "correlation" in R:
        _tc = ti; ti += 1
        with tabs[_tc]:
            st.markdown(
                '<div class="sec-title">Paired Samples Correlations</div>',
                unsafe_allow_html=True)
            corr = R["correlation"].copy()
            corr["Pearson Correlation"] = corr["Pearson Correlation"].apply(_f)
            corr["Sig. (2-tailed)"]     = corr["Sig. (2-tailed)"].apply(_p)
            st.markdown(html_tbl(df_to_rows(corr), left_cols={0}),
                        unsafe_allow_html=True)
            st.markdown(
                '<p class="note-txt">Pearson r between the two paired variables.</p>',
                unsafe_allow_html=True)

    # ── Tab: Levene's test ────────────────────────────────────────────────────
    if "levene" in R:
        _tl = ti; ti += 1
        with tabs[_tl]:
            lev = R["levene"]
            res = ('<span class="pass">✓ Equal variances assumed</span>'
                   if lev["equal_var"]
                   else '<span class="fail">✗ Equal variances NOT assumed (Welch)</span>')
            st.markdown(
                '<div class="sec-title">'
                "Levene's Test for Equality of Variances (center = mean)"
                '</div>', unsafe_allow_html=True)
            st.markdown(html_tbl(
                [["F","df1","df2","Sig.","Result"],
                 [_f(lev["F"]),str(lev["df1"]),str(lev["df2"]),
                  _p(lev["Sig."]),res]],
                left_cols={4}), unsafe_allow_html=True)
            st.markdown(
                '<p class="note-txt">'
                'p > .05 → equal variances assumed → use Row 1 of t-test table.<br>'
                'p ≤ .05 → equal variances not assumed → use Welch row (Row 2).'
                '</p>', unsafe_allow_html=True)

    # ── Tab: Parametric Results ───────────────────────────────────────────────
    _tp = ti; ti += 1
    with tabs[_tp]:
        if test_type == "One-Sample T-Test":
            st.markdown(
                f'<div class="sec-title">'
                f'One-Sample Test · Test Value (μ₀) = {pr["mu0"]}'
                f'</div>', unsafe_allow_html=True)
            st.markdown(html_tbl([
                ["","t","df","Sig. (2-tailed)","Sig. (1-tail Lower)",
                 "Sig. (1-tail Upper)","Mean Diff",
                 "95% CI Lower","95% CI Upper","Cohen's d","Effect Size"],
                [f"Test value = {pr['mu0']}",
                 _f(pr["t"]), str(pr["df"]), _p(pr["p_two"]),
                 _p(pr["p_one_lower"]), _p(pr["p_one_upper"]),
                 _f(pr["mean_diff"]),
                 _f(pr["ci_lower"]), _f(pr["ci_upper"]),
                 _f(pr["cohens_d"]), effect_label_d(pr["cohens_d"])]
            ], left_cols={0,10}), unsafe_allow_html=True)

        elif test_type == "Paired-Sample T-Test":
            # Correlation sub-table
            st.markdown(
                '<div class="sec-title">Paired Samples Correlations</div>',
                unsafe_allow_html=True)
            corr2 = R["correlation"].copy()
            corr2["Pearson Correlation"] = corr2["Pearson Correlation"].apply(_f)
            corr2["Sig. (2-tailed)"]     = corr2["Sig. (2-tailed)"].apply(_p)
            st.markdown(html_tbl(df_to_rows(corr2), left_cols={0}),
                        unsafe_allow_html=True)
            st.markdown(
                '<div class="sec-title">Paired Samples Test</div>',
                unsafe_allow_html=True)
            st.markdown(html_tbl([
                ["Pair","Mean Diff","Std. Dev.","Std. Error Mean",
                 "95% CI Lower","95% CI Upper","t","df",
                 "Sig. (2-tailed)","Sig. (1-tail L)","Sig. (1-tail U)","Cohen's d"],
                [f"{pr['label1']} – {pr['label2']}",
                 _f(pr["mean_diff"]), _f(pr["sd_diff"]), _f(pr["se_diff"]),
                 _f(pr["ci_lower"]), _f(pr["ci_upper"]),
                 _f(pr["t"]), str(pr["df"]),
                 _p(pr["p_two"]),
                 _p(pr["p_one_lower"]), _p(pr["p_one_upper"]),
                 _f(pr["cohens_d"])]
            ], left_cols={0}), unsafe_allow_html=True)

        else:  # Independent
            lev = R["levene"]
            st.markdown(
                '<div class="sec-title">Independent Samples Test</div>',
                unsafe_allow_html=True)
            st.markdown(html_tbl([
                ["","F (Levene)","Sig.","t","df","Sig. (2-tailed)",
                 "Sig. (1-tail L)","Sig. (1-tail U)",
                 "Mean Diff","SE Diff","95% CI Lower","95% CI Upper","Cohen's d"],
                ["Equal var. assumed",
                 _f(lev["F"]), _p(lev["Sig."]),
                 _f(pr["t_eq"]), str(pr["df_eq"]), _p(pr["p_eq"]),
                 _p(stats.t.cdf(pr["t_eq"], pr["df_eq"])),
                 _p(1-stats.t.cdf(pr["t_eq"], pr["df_eq"])),
                 _f(pr["mean_diff"]), _f(pr["se_eq"]),
                 _f(pr["ci_eq_l"]), _f(pr["ci_eq_u"]), _f(pr["cohens_d"])],
                ["Equal var. NOT assumed","","",
                 _f(pr["t_welch"]), _f(pr["df_welch"],2), _p(pr["p_welch"]),
                 _p(stats.t.cdf(pr["t_welch"], pr["df_welch"])),
                 _p(1-stats.t.cdf(pr["t_welch"], pr["df_welch"])),
                 _f(pr["mean_diff"]), _f(pr["se_welch"]),
                 _f(pr["ci_welch_l"]), _f(pr["ci_welch_u"]), "—"]
            ], left_cols={0}), unsafe_allow_html=True)
            active = ("Row 1 (equal var.)" if lev["equal_var"]
                      else "Row 2 (Welch)")
            st.markdown(
                f'<p class="note-txt">Based on Levene p = {_p(lev["Sig."])}: '
                f'use <b>{active}</b>.</p>', unsafe_allow_html=True)

        if not use_p:
            st.markdown(
                '<div class="warn-box">⚠️ Normality violated — '
                'see <b>Non-Parametric Results</b> tab for recommended analysis.'
                '</div>', unsafe_allow_html=True)

    # ── Tab: Non-Parametric Results ───────────────────────────────────────────
    _tnp = ti; ti += 1
    with tabs[_tnp]:
        if test_type == "Independent-Sample T-Test":
            st.markdown(
                '<div class="sec-title">Mann-Whitney U Test — Ranks</div>',
                unsafe_allow_html=True)
            st.markdown(html_tbl([
                ["Group","N","Mean Rank","Sum of Ranks"],
                [np_r["label1"], str(np_r["n1"]),
                 _f(np_r["mean_rank1"]), _f(np_r["R1"],1)],
                [np_r["label2"], str(np_r["n2"]),
                 _f(np_r["mean_rank2"]), _f(np_r["R2"],1)],
                ["Total", str(np_r["n1"]+np_r["n2"]), "", ""]
            ], left_cols={0}), unsafe_allow_html=True)

            st.markdown(
                '<div class="sec-title">'
                'Mann-Whitney U Test — Test Statistics'
                '</div>', unsafe_allow_html=True)
            st.markdown(html_tbl([
                ["Statistic","Value"],
                ["Mann-Whitney U",         _f(np_r["U"],0)],
                ["Wilcoxon W",             _f(np_r["W_wilcoxon"],1)],
                ["Z",                      _f(np_r["Z"])],
                ["Asymp. Sig. (2-tailed)", _p(np_r["p"])],
            ], left_cols={0}), unsafe_allow_html=True)
            st.markdown(
                f'<p class="note-txt">'
                f'Grouping: {np_r["label1"]} vs. {np_r["label2"]}. '
                f'Z based on normal approximation with ties correction.</p>',
                unsafe_allow_html=True)

        else:
            n_neg  = np_r.get("n_neg",  0)
            n_pos  = np_r.get("n_pos",  0)
            n_ties = np_r.get("n_ties", 0)
            n_tot  = np_r.get("n_total", n_neg+n_pos+n_ties)
            neg_rs = np_r.get("neg_rank_sum", np.nan)
            pos_rs = np_r.get("pos_rank_sum", np.nan)
            neg_mr = neg_rs/n_neg if n_neg > 0 else np.nan
            pos_mr = pos_rs/n_pos if n_pos > 0 else np.nan

            footnotes = []
            if test_type == "Paired-Sample T-Test":
                footnotes = [
                    f"ᵃ {pr['label2']} < {pr['label1']}",
                    f"ᵇ {pr['label2']} > {pr['label1']}",
                    f"ᶜ {pr['label2']} = {pr['label1']}"
                ]

            st.markdown(
                '<div class="sec-title">Wilcoxon Signed-Rank Test — Ranks</div>',
                unsafe_allow_html=True)
            st.markdown(html_tbl([
                ["","N","Mean Rank","Sum of Ranks"],
                ["Negative Ranks" + (" ᵃ" if footnotes else ""),
                 str(n_neg), _f(neg_mr), _f(neg_rs,1)],
                ["Positive Ranks" + (" ᵇ" if footnotes else ""),
                 str(n_pos), _f(pos_mr), _f(pos_rs,1)],
                ["Ties" + (" ᶜ" if footnotes else ""),
                 str(n_ties),"",""],
                ["Total", str(n_tot),"",""]
            ], left_cols={0}), unsafe_allow_html=True)
            for fn in footnotes:
                st.markdown(f'<p class="note-txt">{fn}</p>',
                            unsafe_allow_html=True)

            pair_lbl = (f"{pr['label1']} − {pr['label2']}"
                        if test_type == "Paired-Sample T-Test"
                        else "Variable − μ₀")
            st.markdown(
                '<div class="sec-title">'
                'Wilcoxon Signed-Rank Test — Test Statistics'
                '</div>', unsafe_allow_html=True)
            st.markdown(html_tbl([
                ["Statistic", pair_lbl],
                ["Test Statistic (W)",     _f(np_r["W"],0)],
                ["Z",                      _f(np_r["Z"])],
                ["Asymp. Sig. (2-tailed)", _p(np_r["p"])],
            ], left_cols={0}), unsafe_allow_html=True)
            st.markdown(
                f'<p class="note-txt">'
                f'Based on {"negative" if n_neg < n_pos else "positive"} ranks. '
                f'Z uses ties-corrected variance (SPSS method).</p>',
                unsafe_allow_html=True)

        if use_p:
            st.markdown(
                '<div class="info-box">ℹ️ Normality was met — '
                'Parametric Results tab contains the recommended analysis.</div>',
                unsafe_allow_html=True)

    # ── Tab: Plots ────────────────────────────────────────────────────────────
    _tpl = ti; ti += 1
    with tabs[_tpl]:
        for fb in figs_b:
            st.image(fb, use_container_width=True)

    # ── Tab: Interpretation ───────────────────────────────────────────────────
    _ti2 = ti; ti += 1
    with tabs[_ti2]:
        st.markdown("### 📝 Statistical Interpretation")
        for line in interps:
            cls = ""
            lw  = line.lower()
            if ("significant difference" in lw and
                    "no statistically" not in lw and
                    "not significant" not in lw):
                cls = "sig"
            elif ("no significant" in lw or "not significant" in lw or
                  "does not significantly" in lw):
                cls = "nonsig"
            st.markdown(f'<div class="interp-box {cls}">{line}</div>',
                        unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**APA 7th Edition Write-Up:**")
        if use_p:
            if test_type == "One-Sample T-Test":
                m_v  = float(R["desc"]["Mean"].iloc[0])
                sd_v = float(R["desc"]["Std. Deviation"].iloc[0])
                apa  = (
                    f"A one-sample t-test was conducted to examine whether "
                    f"{meta.get('Variable','the variable')} "
                    f"(M = {_f(m_v)}, SD = {_f(sd_v)}) "
                    f"differed from μ₀ = {pr['mu0']}. "
                    f"The test was "
                    f"{'statistically significant' if pr['p_two']<alpha else 'not significant'}, "
                    f"t({pr['df']}) = {pr['t']:.2f}, "
                    f"p {'< .001' if pr['p_two']<.001 else '= '+_p(pr['p_two'])}, "
                    f"d = {pr['cohens_d']:.2f}."
                )
            elif test_type == "Paired-Sample T-Test":
                apa = (
                    f"A paired-samples t-test compared "
                    f"{pr['label1']} "
                    f"(M = {_f(float(R['desc'].iloc[0]['Mean']))}, "
                    f"SD = {_f(float(R['desc'].iloc[0]['Std. Deviation']))}) "
                    f"with {pr['label2']} "
                    f"(M = {_f(float(R['desc'].iloc[1]['Mean']))}, "
                    f"SD = {_f(float(R['desc'].iloc[1]['Std. Deviation']))}). "
                    f"The difference was "
                    f"{'significant' if pr['p_two']<alpha else 'not significant'}, "
                    f"t({pr['df']}) = {pr['t']:.2f}, "
                    f"p {'< .001' if pr['p_two']<.001 else '= '+_p(pr['p_two'])}, "
                    f"d = {pr['cohens_d']:.2f}."
                )
            else:
                use_eq = R["levene"]["equal_var"]
                tv = pr["t_eq"]    if use_eq else pr["t_welch"]
                pv = pr["p_eq"]    if use_eq else pr["p_welch"]
                dv = pr["df_eq"]   if use_eq else pr["df_welch"]
                apa = (
                    f"An independent-samples t-test compared {pr['dep_var']} "
                    f"between {pr['label1']} "
                    f"(M = {_f(float(R['desc'].iloc[0]['Mean']))}, "
                    f"SD = {_f(float(R['desc'].iloc[0]['Std. Deviation']))}) "
                    f"and {pr['label2']} "
                    f"(M = {_f(float(R['desc'].iloc[1]['Mean']))}, "
                    f"SD = {_f(float(R['desc'].iloc[1]['Std. Deviation']))}). "
                    f"The difference was "
                    f"{'significant' if pv<alpha else 'not significant'}, "
                    f"t({_f(dv,2)}) = {tv:.2f}, "
                    f"p {'< .001' if pv<.001 else '= '+_p(pv)}, "
                    f"d = {pr['cohens_d']:.2f}."
                )
        else:
            if test_type == "Independent-Sample T-Test":
                apa = (
                    f"A Mann-Whitney U test compared "
                    f"{np_r['dep_var']} between "
                    f"{np_r['label1']} and {np_r['label2']}, "
                    f"U = {np_r['U']:.0f}, W = {np_r['W_wilcoxon']:.1f}, "
                    f"Z = {np_r['Z']:.3f}, "
                    f"p {'< .001' if np_r['p']<.001 else '= '+_p(np_r['p'])} "
                    f"(asymptotic, 2-tailed)."
                )
            else:
                apa = (
                    f"A Wilcoxon signed-rank test was conducted: "
                    f"W = {np_r['W']:.0f}, Z = {np_r['Z']:.3f}, "
                    f"p {'< .001' if np_r['p']<.001 else '= '+_p(np_r['p'])} "
                    f"(2-tailed)."
                )
        st.code(apa, language=None)

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📥 Download Results")
    dc1, dc2, dc3 = st.columns(3)

    with dc1:
        with st.spinner("Building PDF…"):
            pdf_data = build_pdf(test_type, R, meta, interps, figs_b)
        st.download_button(
            "📄 PDF Report",
            pdf_data,
            f"Stats_{test_type.replace(' ','_')}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            "application/pdf", use_container_width=True)

    with dc2:
        xbuf = io.BytesIO()
        with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
            R["desc"].to_excel(
                writer, sheet_name="Descriptive Statistics", index=False)
            # Normality table (both tests)
            norm_rows = []
            for n in R["normality"]:
                norm_rows.append({
                    "Variable":  n["label"], "N": n["n"],
                    "SW W":      n["sw_W"],  "SW Sig.":  n["sw_p"],
                    "SW Normal": n["sw_pass"],
                    "KS D":      n["ks_D"],  "KS Sig.":  n["ks_p"],
                    "KS Normal": n["ks_pass"],
                    "Pass (both)": n["pass"]
                })
            pd.DataFrame(norm_rows).to_excel(
                writer, sheet_name="Normality Tests", index=False)
            if "correlation" in R:
                R["correlation"].to_excel(
                    writer, sheet_name="Paired Correlation", index=False)
            if "levene" in R:
                pd.DataFrame([R["levene"]]).to_excel(
                    writer, sheet_name="Levene Test", index=False)
            pd.DataFrame([R["parametric"]]).to_excel(
                writer, sheet_name="Parametric Results", index=False)
            pd.DataFrame([R["nonparametric"]]).to_excel(
                writer, sheet_name="Non-Parametric Results", index=False)
        xbuf.seek(0)
        st.download_button(
            "📊 Excel Workbook",
            xbuf.getvalue(),
            f"Stats_{test_type.replace(' ','_')}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)

    with dc3:
        norm_df_rows = []
        for n in R["normality"]:
            norm_df_rows.append({
                "Variable": n["label"], "N": n["n"],
                "SW_W": n["sw_W"], "SW_p": n["sw_p"], "SW_pass": n["sw_pass"],
                "KS_D": n["ks_D"], "KS_p": n["ks_p"], "KS_pass": n["ks_pass"],
                "pass_both": n["pass"]
            })
        parts = [f"=== {test_type.upper()} ===\n"]
        parts.append("=== DESCRIPTIVE STATISTICS ===\n" +
                     R["desc"].to_csv(index=False))
        parts.append("=== NORMALITY TESTS (SW + KS) ===\n" +
                     pd.DataFrame(norm_df_rows).to_csv(index=False))
        if "correlation" in R:
            parts.append("=== PAIRED CORRELATION ===\n" +
                         R["correlation"].to_csv(index=False))
        if "levene" in R:
            parts.append("=== LEVENE TEST ===\n" +
                         pd.DataFrame([R["levene"]]).to_csv(index=False))
        parts.append("=== PARAMETRIC RESULTS ===\n" +
                     pd.DataFrame([R["parametric"]]).to_csv(index=False))
        parts.append("=== NON-PARAMETRIC RESULTS ===\n" +
                     pd.DataFrame([R["nonparametric"]]).to_csv(index=False))
        st.download_button(
            "📝 CSV Tables",
            "\n\n".join(parts).encode(),
            f"Stats_{test_type.replace(' ','_')}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv", use_container_width=True)


if __name__ == "__main__":
    main()
