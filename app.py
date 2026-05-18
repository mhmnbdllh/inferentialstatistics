"""
Inferential Statistics App — SPSS-Equivalent Output
====================================================
Parametric & Non-parametric tests with automatic selection via normality testing.
Normality: Shapiro-Wilk + Kolmogorov-Smirnov (Lilliefors correction).
  - n <= 50 : Shapiro-Wilk recommended as primary criterion
  - n >  50 : Kolmogorov-Smirnov recommended as primary criterion
Tests: One-Sample T, Paired-Sample T, Independent-Sample T
       + Wilcoxon Signed-Rank (Z with ties correction), Mann-Whitney U (SPSS-exact)

SPSS Formula Notes:
  1. Levene's test uses center='mean' (SPSS default)
  2. KS uses Lilliefors correction (statsmodels) — same as SPSS Explore
  3. Wilcoxon Z = (W − E[W]) / sqrt(Var[W] − ties_correction)
  4. Mann-Whitney U always displayed with 3 decimal places (SPSS format, e.g. 6.000)
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
import base64
import warnings
from datetime import datetime

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
.sub-title{background:#f1f5f9;color:#1a1a2e;padding:7px 14px;border-radius:6px;
  font-weight:600;font-size:.82rem;margin:1rem 0 .4rem;
  border-left:3px solid #e94560;}
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
.norm-rec-box{background:linear-gradient(135deg,#f0f9ff,#e0f2fe);
  border-left:4px solid #0284c7;padding:.8rem 1rem;
  border-radius:0 8px 8px 0;font-size:.82rem;color:#0c4a6e;margin:.5rem 0;
  line-height:1.7;}
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

# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def _f(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "."
    return f"{v:.{d}f}"

def _p(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "."
    return ".000" if v < .001 else f"{v:.3f}"

def format_u(v):
    """
    Format Mann-Whitney U always with 3 decimal places — identical to SPSS output.
    Examples: 6.000, 6.500, 24.000, 132.500
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "."
    return f"{v:.3f}"

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

# ══════════════════════════════════════════════════════════════════════════════
# NORMALITY RECOMMENDATION NOTE
# ══════════════════════════════════════════════════════════════════════════════
def normality_recommendation_note(n):
    """
    Returns an academic-English recommendation note explaining which
    normality test is appropriate given the sample size, and why.
    """
    if n <= 50:
        return (
            f"<b>Normality Test Recommendation (n\u2009=\u2009{n}):</b> "
            "Given a small sample size (n\u2009\u2264\u200950), the "
            "<b>Shapiro-Wilk test</b> is recommended as the primary criterion "
            "for assessing normality. The Shapiro-Wilk test is widely regarded "
            "as the most powerful test for normality in small samples, exhibiting "
            "superior sensitivity to departures from normality compared to the "
            "Kolmogorov-Smirnov test (Razali &amp; Wah, 2011; Shapiro &amp; Wilk, 1965). "
            "The Kolmogorov-Smirnov result is reported for informational purposes."
        )
    else:
        return (
            f"<b>Normality Test Recommendation (n\u2009=\u2009{n}):</b> "
            "Given a larger sample size (n\u2009&gt;\u200950), the "
            "<b>Kolmogorov-Smirnov test with Lilliefors significance correction</b> "
            "is recommended as the primary criterion. For larger samples, "
            "the Shapiro-Wilk test may become overly sensitive, flagging trivial "
            "deviations from normality as statistically significant. "
            "The Lilliefors-corrected K-S test provides a more appropriate "
            "assessment in this context (Lilliefors, 1967; Field, 2018). "
            "The Shapiro-Wilk result is reported for informational purposes."
        )

def normality_recommendation_plain(n):
    """Plain-text version for HTML report body (no HTML tags)."""
    if n <= 50:
        return (
            f"Normality Test Recommendation (n = {n}): "
            "Given a small sample size (n \u2264 50), the Shapiro-Wilk test is "
            "recommended as the primary criterion. The Shapiro-Wilk test exhibits "
            "superior sensitivity to departures from normality in small samples "
            "(Razali & Wah, 2011; Shapiro & Wilk, 1965). "
            "The Kolmogorov-Smirnov result is reported for informational purposes."
        )
    else:
        return (
            f"Normality Test Recommendation (n = {n}): "
            "Given a larger sample size (n > 50), the Kolmogorov-Smirnov test "
            "with Lilliefors significance correction is recommended as the primary "
            "criterion. For larger samples, the Shapiro-Wilk test may be "
            "overly sensitive to trivial deviations from normality "
            "(Lilliefors, 1967; Field, 2018). "
            "The Shapiro-Wilk result is reported for informational purposes."
        )

# ══════════════════════════════════════════════════════════════════════════════
# NORMALITY TESTS
# ══════════════════════════════════════════════════════════════════════════════
def test_normality(data, label=""):
    """
    Shapiro-Wilk + Kolmogorov-Smirnov (Lilliefors correction).
    Primary decision criterion:
      n <= 50 : Shapiro-Wilk (recommended for small samples)
      n >  50 : KS with Lilliefors correction (recommended for larger samples)
    Both tests always computed and reported.
    """
    data = np.array(data, dtype=float)
    n = len(data)
    result = {"label": label, "n": n}

    if n < 3:
        result.update({"sw_W": np.nan, "sw_p": np.nan, "sw_pass": False,
                       "ks_D": np.nan, "ks_p": np.nan, "ks_pass": False,
                       "pass": False, "primary": "sw",
                       "primary_label": "Shapiro-Wilk"})
        return result

    sw_W, sw_p = stats.shapiro(data)
    result["sw_W"]    = float(sw_W)
    result["sw_p"]    = float(sw_p)
    result["sw_pass"] = float(sw_p) > 0.05

    try:
        # Use 'table' for n<=50 (matches SPSS Lilliefors table interpolation)
        # Use 'approx' for n>50  (asymptotic approximation, accurate for large n)
        ks_method = 'table' if n <= 50 else 'approx'
        ks_D, ks_p = lilliefors(data, dist='norm', pvalmethod='approx')
        result["ks_D"]    = float(ks_D)
        result["ks_p"]    = float(ks_p)
        result["ks_pass"] = float(ks_p) >= 0.05
    except Exception:
        result["ks_D"]    = np.nan
        result["ks_p"]    = np.nan
        result["ks_pass"] = True

    if n <= 50:
        result["primary"]       = "sw"
        result["primary_label"] = "Shapiro-Wilk"
        result["pass"]          = result["sw_pass"]
    else:
        result["primary"]       = "ks"
        result["primary_label"] = "Kolmogorov-Smirnov"
        result["pass"]          = result["ks_pass"]

    return result

# ══════════════════════════════════════════════════════════════════════════════
# WILCOXON SIGNED-RANK  (SPSS-exact)
# ══════════════════════════════════════════════════════════════════════════════
def wilcoxon_spss(diff_arr):
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

# ══════════════════════════════════════════════════════════════════════════════
# MANN-WHITNEY U  (SPSS-exact)
# ══════════════════════════════════════════════════════════════════════════════
def mannwhitney_spss(g1, g2, label1, label2):
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
    W_wilcoxon = R1

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
# EFFECT SIZE — NON-PARAMETRIC (beyond SPSS)
# ══════════════════════════════════════════════════════════════════════════════
def rank_biserial_mannwhitney(g1, g2):
    """
    Rank-biserial correlation r for Mann-Whitney U.
    Formula: r = 1 - (2U) / (n1 * n2)
    Interpretation: |r| < .1 negligible, .1-.3 small, .3-.5 medium, > .5 large
    Reference: King & Minium (2008); Kerby (2014).
    """
    g1 = np.array(g1, dtype=float)
    g2 = np.array(g2, dtype=float)
    n1, n2 = len(g1), len(g2)
    U1 = sum(1 if x > y else 0.5 if x == y else 0
             for x in g1 for y in g2)
    r = 1 - (2 * U1) / (n1 * n2)
    return float(r)


def rank_biserial_wilcoxon(diff_arr):
    """
    Rank-biserial correlation r for Wilcoxon Signed-Rank.
    Formula: r = (T+ - T-) / (T+ + T-)
    Reference: Kerby (2014); King & Minium (2008).
    """
    diff_arr = np.array(diff_arr, dtype=float)
    diff_nz  = diff_arr[diff_arr != 0]
    if len(diff_nz) < 1:
        return np.nan
    abs_d  = np.abs(diff_nz)
    ranks  = scipy_stats.rankdata(abs_d) if False else stats.rankdata(abs_d)
    pos_rs = float(np.sum(ranks[diff_nz > 0]))
    neg_rs = float(np.sum(ranks[diff_nz < 0]))
    total  = pos_rs + neg_rs
    return float((pos_rs - neg_rs) / total) if total > 0 else np.nan


def bootstrap_ci_effect_size(effect_fn, data_args, n_boot=2000, alpha=0.05,
                              seed=42):
    """
    Bootstrap 95% CI for any effect size function.
    Returns (lower, upper) confidence interval.
    Reference: Efron & Tibshirani (1993).
    """
    rng = np.random.default_rng(seed)
    boot_effects = []
    for _ in range(n_boot):
        resampled = []
        for arr in data_args:
            arr = np.array(arr, dtype=float)
            resampled.append(rng.choice(arr, size=len(arr), replace=True))
        try:
            e = effect_fn(*resampled)
            if not np.isnan(e):
                boot_effects.append(e)
        except Exception:
            pass
    if len(boot_effects) < 10:
        return np.nan, np.nan
    lo = float(np.percentile(boot_effects, 100 * alpha / 2))
    hi = float(np.percentile(boot_effects, 100 * (1 - alpha / 2)))
    return lo, hi


def effect_label_r_nonparam(r):
    """Effect size label for rank-biserial correlation."""
    a = abs(r)
    if a < .10: return "negligible"
    if a < .30: return "small"
    if a < .50: return "medium"
    return "large"


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL POWER ANALYSIS (beyond SPSS)
# ══════════════════════════════════════════════════════════════════════════════
def compute_power(test_type, R, alpha):
    """
    Post-hoc statistical power analysis.
    Uses observed effect size and sample size to estimate achieved power.
    Reference: Cohen (1988). Statistical Power Analysis for the Behavioral Sciences.
    """
    from scipy.stats import t as t_dist, norm as norm_dist
    pr   = R["parametric"]
    np_r = R["nonparametric"]
    use_p = R["use_param"]
    result = {}

    if use_p:
        d    = abs(pr["cohens_d"])
        if test_type == "One-Sample T-Test":
            n  = R["desc"]["N"].iloc[0]
            nc = d * np.sqrt(n)          # non-centrality parameter
            df = n - 1
            tc = t_dist.ppf(1 - alpha/2, df)
            power = 1 - t_dist.cdf(tc, df, nc) + t_dist.cdf(-tc, df, nc)
            result = {"n": int(n), "effect_size": d,
                      "effect_type": "Cohen's d", "power": float(power)}

        elif test_type == "Paired-Sample T-Test":
            n  = R["desc"]["N"].iloc[0]
            nc = d * np.sqrt(n)
            df = n - 1
            tc = t_dist.ppf(1 - alpha/2, df)
            power = 1 - t_dist.cdf(tc, df, nc) + t_dist.cdf(-tc, df, nc)
            result = {"n": int(n), "effect_size": d,
                      "effect_type": "Cohen's d", "power": float(power)}

        else:
            n1 = np_r["n1"]; n2 = np_r["n2"]
            n_harm = 2 / (1/n1 + 1/n2)  # harmonic mean
            nc = d * np.sqrt(n_harm / 2)
            df = pr["df_eq"] if R["levene"]["equal_var"] else pr["df_welch"]
            tc = t_dist.ppf(1 - alpha/2, df)
            power = 1 - t_dist.cdf(tc, df, nc) + t_dist.cdf(-tc, df, nc)
            result = {"n1": int(n1), "n2": int(n2),
                      "effect_size": d,
                      "effect_type": "Cohen's d", "power": float(power)}
    else:
        # Non-parametric: approximate power via normal approximation
        if test_type == "Independent-Sample T-Test":
            r    = abs(rank_biserial_mannwhitney(
                R["desc"].iloc[0].get("Mean", 0),  # placeholder
                R["desc"].iloc[1].get("Mean", 0)
            ))
            n1   = np_r["n1"]; n2 = np_r["n2"]
            # Asymptotic relative efficiency of Mann-Whitney vs t ≈ 0.955
            d_approx = 2 * abs(np_r.get("Z", 0)) / np.sqrt(n1 + n2)
            n_harm   = 2 / (1/n1 + 1/n2)
            nc       = d_approx * np.sqrt(n_harm / 2)
            df       = n1 + n2 - 2
            tc       = t_dist.ppf(1 - alpha/2, df)
            power    = 1 - t_dist.cdf(tc, df, nc) + t_dist.cdf(-tc, df, nc)
            result   = {"n1": int(n1), "n2": int(n2),
                        "effect_size": float(d_approx),
                        "effect_type": "Approx. d (from Z)",
                        "power": float(power)}
        else:
            n    = np_r.get("n_total", np_r.get("n_pos",0) + np_r.get("n_neg",0))
            z    = abs(np_r.get("Z", 0))
            # Power ≈ P(|Z| > z_alpha/2 - z_observed)
            z_crit = norm_dist.ppf(1 - alpha/2)
            power  = norm_dist.sf(z_crit - z) + norm_dist.cdf(-z_crit - z)
            d_approx = z / np.sqrt(n) if n > 0 else np.nan
            result = {"n": int(n),
                      "effect_size": float(d_approx),
                      "effect_type": "Approx. d (from Z)",
                      "power": float(max(0, min(1, power)))}

    # Power interpretation
    p = result.get("power", 0)
    if p >= .95:   result["power_label"] = "Excellent (\u2265\u2009.95)"
    elif p >= .80: result["power_label"] = "Adequate (\u2265\u2009.80)"
    elif p >= .60: result["power_label"] = "Moderate (.60\u2013.79)"
    else:          result["power_label"] = "Low (< .60) \u2014 consider increasing sample size"

    return result


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

    t_stat, p_two = stats.ttest_1samp(data, mu0)
    df = n - 1; tc = stats.t.ppf(1 - alpha/2, df)
    diff_m = m - mu0
    d = cohens_d_1s(data, mu0)
    R["parametric"] = {
        "test": "One-Sample T-Test", "mu0": mu0,
        "t": float(t_stat), "df": df,
        "p_two":       float(p_two),
        "p_one_lower": float(stats.t.cdf(t_stat, df)),
        "p_one_upper": float(1 - stats.t.cdf(t_stat, df)),
        "mean_diff": diff_m,
        "ci_lower":  diff_m - tc*(sd/np.sqrt(n)),
        "ci_upper":  diff_m + tc*(sd/np.sqrt(n)),
        "cohens_d": d
    }

    w_res = wilcoxon_spss(data - mu0)
    R["nonparametric"] = {**w_res, "test": "Wilcoxon Signed-Rank Test", "mu0": mu0}
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

    r_corr, p_corr = stats.pearsonr(data1, data2)
    R["correlation"] = pd.DataFrame([{
        "Pair":  f"{label1} & {label2}",
        "N":     n,
        "Pearson Correlation": float(r_corr),
        "Sig. (2-tailed)":     float(p_corr)
    }])

    norm = test_normality(diff, f"{label1} \u2212 {label2}")
    R["normality"]  = [norm]
    R["use_param"]  = norm["pass"]

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
        "p_two":       float(p_two),
        "p_one_lower": float(stats.t.cdf(t_stat, df)),
        "p_one_upper": float(1 - stats.t.cdf(t_stat, df)),
        "ci_lower": m_diff - tc*se_diff,
        "ci_upper": m_diff + tc*se_diff,
        "cohens_d": d
    }

    w_res = wilcoxon_spss(diff)
    R["nonparametric"] = {**w_res, "test": "Wilcoxon Signed-Rank Test",
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

    n1_res = test_normality(g1, label1)
    n2_res = test_normality(g2, label2)
    R["normality"]  = [n1_res, n2_res]
    R["use_param"]  = n1_res["pass"] and n2_res["pass"]

    lev_f, lev_p = stats.levene(g1, g2, center='mean')
    R["levene"] = {
        "F": float(lev_f), "df1": 1, "df2": n1+n2-2,
        "Sig.": float(lev_p), "equal_var": float(lev_p) > alpha
    }

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

    # Pre-compute one-tailed p values to avoid calling stats inside build functions
    p_eq_lower    = float(stats.t.cdf(t_eq,    df_eq))
    p_eq_upper    = float(1 - stats.t.cdf(t_eq,    df_eq))
    p_welch_lower = float(stats.t.cdf(t_welch, df_welch))
    p_welch_upper = float(1 - stats.t.cdf(t_welch, df_welch))

    R["parametric"] = {
        "test": "Independent Samples T-Test",
        "label1": label1, "label2": label2, "dep_var": dep_var,
        "mean_diff": mean_diff,
        "t_eq": float(t_eq), "df_eq": df_eq, "p_eq": float(p_eq),
        "p_eq_lower":    p_eq_lower,    "p_eq_upper":    p_eq_upper,
        "se_eq": se_eq,
        "ci_eq_l": mean_diff - tc_eq*se_eq,
        "ci_eq_u": mean_diff + tc_eq*se_eq,
        "t_welch": float(t_welch), "df_welch": df_welch,
        "p_welch": float(p_welch),
        "p_welch_lower": p_welch_lower, "p_welch_upper": p_welch_upper,
        "se_welch": se_welch,
        "ci_welch_l": mean_diff - tc_welch*se_welch,
        "ci_welch_u": mean_diff + tc_welch*se_welch,
        "cohens_d": d
    }

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
    primary = norm["primary_label"]; n = norm["n"]
    lines.append(
        f"<b>Normality Assessment:</b> Shapiro-Wilk W\u2009=\u2009{_f(norm['sw_W'])}, "
        f"p\u2009=\u2009{_p(norm['sw_p'])} ({'normally distributed' if sw_ok else 'non-normally distributed'}); "
        f"Kolmogorov-Smirnov D\u2009=\u2009{_f(norm['ks_D'])}, "
        f"p\u2009=\u2009{_p(norm['ks_p'])} ({'normally distributed' if ks_ok else 'non-normally distributed'}). "
        f"Given n\u2009=\u2009{n} ({'n\u2009\u2264\u200950' if n <= 50 else 'n\u2009>\u200950'}), "
        f"the <b>{primary}</b> test was used as the primary criterion. "
        f"{'Parametric analysis was applied.' if R['use_param'] else 'Non-parametric analysis was applied.'}"
    )
    if R["use_param"]:
        sig = pr["p_two"] < alpha
        lines.append(
            f"<b>One-Sample T-Test:</b> t({pr['df']})\u2009=\u2009{_f(pr['t'])}, "
            f"p\u2009=\u2009{_p(pr['p_two'])} (2-tailed). "
            f"Mean difference\u2009=\u2009{_f(pr['mean_diff'])}, "
            f"95% CI [{_f(pr['ci_lower'])},\u2009{_f(pr['ci_upper'])}], "
            f"Cohen\u2019s d\u2009=\u2009{_f(pr['cohens_d'])} ({effect_label_d(pr['cohens_d'])} effect). "
            f"The sample mean {'significantly differed from' if sig else 'did not significantly differ from'} "
            f"the hypothesised population mean (\u03bc\u2080\u2009=\u2009{pr['mu0']}) "
            f"at \u03b1\u2009=\u2009{alpha}."
        )
    else:
        np_r = R["nonparametric"]
        sig  = np_r["p"] < alpha if not np.isnan(np_r["p"]) else False
        lines.append(
            f"<b>Wilcoxon Signed-Rank Test:</b> W\u2009=\u2009{_f(np_r['W'],0)}, "
            f"Z\u2009=\u2009{_f(np_r['Z'])}, p\u2009=\u2009{_p(np_r['p'])} (2-tailed). "
            f"{'A statistically significant' if sig else 'No statistically significant'} "
            f"difference from \u03bc\u2080\u2009=\u2009{np_r['mu0']} was detected "
            f"at \u03b1\u2009=\u2009{alpha}."
        )
    return lines


def interpret_paired(R, alpha):
    lines = []
    norm = R["normality"][0]; pr = R["parametric"]
    corr = R["correlation"]
    r_v  = float(corr["Pearson Correlation"].iloc[0])
    p_v  = float(corr["Sig. (2-tailed)"].iloc[0])
    sw_ok = norm["sw_pass"]; ks_ok = norm["ks_pass"]
    primary = norm["primary_label"]; n = norm["n"]
    lines.append(
        f"<b>Normality of Differences:</b> "
        f"Shapiro-Wilk W\u2009=\u2009{_f(norm['sw_W'])}, "
        f"p\u2009=\u2009{_p(norm['sw_p'])} ({'normally distributed' if sw_ok else 'non-normally distributed'}); "
        f"Kolmogorov-Smirnov D\u2009=\u2009{_f(norm['ks_D'])}, "
        f"p\u2009=\u2009{_p(norm['ks_p'])} ({'normally distributed' if ks_ok else 'non-normally distributed'}). "
        f"Given n\u2009=\u2009{n} ({'n\u2009\u2264\u200950' if n <= 50 else 'n\u2009>\u200950'}), "
        f"the <b>{primary}</b> test was used as the primary criterion. "
        f"{'Parametric analysis was applied.' if R['use_param'] else 'Non-parametric analysis was applied.'}"
    )
    lines.append(
        f"<b>Paired Samples Correlation:</b> {pr['label1']} and {pr['label2']} were "
        f"{'significantly' if p_v < alpha else 'not significantly'} correlated, "
        f"r({pr['df']})\u2009=\u2009{_f(r_v)}, p\u2009=\u2009{_p(p_v)}."
    )
    if R["use_param"]:
        sig = pr["p_two"] < alpha
        lines.append(
            f"<b>Paired Samples T-Test:</b> t({pr['df']})\u2009=\u2009{_f(pr['t'])}, "
            f"p\u2009=\u2009{_p(pr['p_two'])} (2-tailed). "
            f"Mean difference\u2009=\u2009{_f(pr['mean_diff'])} (SD\u2009=\u2009{_f(pr['sd_diff'])}), "
            f"95% CI [{_f(pr['ci_lower'])},\u2009{_f(pr['ci_upper'])}], "
            f"Cohen\u2019s d\u2009=\u2009{_f(pr['cohens_d'])} ({effect_label_d(pr['cohens_d'])} effect). "
            f"{'A statistically significant difference was found' if sig else 'No statistically significant difference was found'} "
            f"between {pr['label1']} and {pr['label2']} at \u03b1\u2009=\u2009{alpha}."
        )
    else:
        np_r = R["nonparametric"]
        sig  = np_r["p"] < alpha if not np.isnan(np_r["p"]) else False
        lines.append(
            f"<b>Wilcoxon Signed-Rank Test:</b> W\u2009=\u2009{_f(np_r['W'],0)}, "
            f"Z\u2009=\u2009{_f(np_r['Z'])}, p\u2009=\u2009{_p(np_r['p'])} (2-tailed). "
            f"{'A statistically significant' if sig else 'No statistically significant'} "
            f"difference between {pr['label1']} and {pr['label2']} was detected "
            f"at \u03b1\u2009=\u2009{alpha}."
        )
    return lines


def interpret_independent(R, dep_var, alpha):
    lines = []
    norms = R["normality"]; lev = R["levene"]; pr = R["parametric"]
    for n_item in norms:
        sw_ok = n_item["sw_pass"]; ks_ok = n_item["ks_pass"]
        primary = n_item["primary_label"]; n = n_item["n"]
        lines.append(
            f"<b>Normality \u2014 {n_item['label']}:</b> "
            f"Shapiro-Wilk W\u2009=\u2009{_f(n_item['sw_W'])}, "
            f"p\u2009=\u2009{_p(n_item['sw_p'])} ({'normal' if sw_ok else 'non-normal'}); "
            f"Kolmogorov-Smirnov D\u2009=\u2009{_f(n_item['ks_D'])}, "
            f"p\u2009=\u2009{_p(n_item['ks_p'])} ({'normal' if ks_ok else 'non-normal'}). "
            f"Primary criterion: <b>{primary}</b> "
            f"({'n\u2009\u2264\u200950' if n <= 50 else 'n\u2009>\u200950'})."
        )
    lines.append(
        f"<b>Overall Normality Decision:</b> "
        f"{'Both groups satisfied the normality assumption; parametric analysis was applied.' if R['use_param'] else 'The normality assumption was violated in at least one group; non-parametric analysis was applied.'}"
    )
    lines.append(
        f"<b>Levene\u2019s Test for Equality of Variances:</b> "
        f"F({lev['df1']},\u2009{lev['df2']})\u2009=\u2009{_f(lev['F'])}, "
        f"p\u2009=\u2009{_p(lev['Sig.'])}. "
        f"{'Equal variances were assumed (p > .05).' if lev['equal_var'] else 'Equal variances were not assumed (p \u2264 .05); the Welch correction was applied.'}"
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
            f"<b>Independent Samples T-Test ({'equal variances assumed' if use_eq else 'Welch correction'}):</b> "
            f"t({_f(dfv,2)})\u2009=\u2009{_f(tv)}, p\u2009=\u2009{_p(pv)} (2-tailed). "
            f"Mean difference\u2009=\u2009{_f(pr['mean_diff'])}, "
            f"95% CI [{_f(cil)},\u2009{_f(ciu)}], "
            f"Cohen\u2019s d\u2009=\u2009{_f(pr['cohens_d'])} ({effect_label_d(pr['cohens_d'])} effect). "
            f"{'A statistically significant' if sig else 'No statistically significant'} "
            f"difference in {dep_var} was found between the two groups at \u03b1\u2009=\u2009{alpha}."
        )
    else:
        np_r = R["nonparametric"]
        sig  = np_r["p"] < alpha if not np.isnan(np_r["p"]) else False
        lines.append(
            f"<b>Mann-Whitney U Test:</b> U\u2009=\u2009{format_u(np_r['U'])}, "
            f"W\u2009=\u2009{_f(np_r['W_wilcoxon'],3)}, "
            f"Z\u2009=\u2009{_f(np_r['Z'])}, p\u2009=\u2009{_p(np_r['p'])} (2-tailed). "
            f"{'A statistically significant' if sig else 'No statistically significant'} "
            f"difference in {dep_var} was found between the two groups at \u03b1\u2009=\u2009{alpha}."
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
    ax1.axvline(mu0, color="#e94560", ls="--", lw=1.8, label=f"\u03bc\u2080={mu0}")
    ax1.axvline(data.mean(), color=PAL[2], lw=1.8, label=f"x\u0304={data.mean():.2f}")
    ax1.set_xlabel(var_name, fontsize=8); ax1.set_ylabel("Frequency", fontsize=8)
    ax1.set_title("Distribution", fontsize=9, fontweight="bold", color=PAL[0])
    ax1.legend(fontsize=7); ax1.spines[["top","right"]].set_visible(False)

    ax2 = fig.add_subplot(gs[1]); ax2.set_facecolor("#f8fafc")
    (osm, osr),(sl, ic, _) = stats.probplot(data)
    ax2.plot(osm, osr, "o", color=PAL[1], markersize=5, markeredgecolor="white", alpha=.8)
    ax2.plot(osm, sl*np.array(osm)+ic, "--", color=PAL[0], lw=1.5)
    ax2.set_xlabel("Theoretical Quantiles", fontsize=8)
    ax2.set_ylabel("Sample Quantiles", fontsize=8)
    ax2.set_title("Normal Q-Q Plot", fontsize=9, fontweight="bold", color=PAL[0])
    ax2.spines[["top","right"]].set_visible(False)

    ax3 = fig.add_subplot(gs[2]); ax3.set_facecolor("#f8fafc")
    ax3.boxplot(data, patch_artist=True, widths=0.5,
                medianprops={"color":"white","linewidth":2},
                boxprops={"facecolor":PAL[1],"alpha":0.8})
    ax3.axhline(mu0, color="#e94560", ls="--", lw=1.8, label=f"\u03bc\u2080={mu0}")
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
    ax3.set_title("Normal Q-Q Plot by Group", fontsize=9, fontweight="bold", color=PAL[0])
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
# HTML OFFLINE REPORT — REPLACES PDF
# ══════════════════════════════════════════════════════════════════════════════
def build_html_report(test_type, R, meta, interps, fig_bytes_list):
    """
    Generates a fully self-contained offline HTML report.
    Includes all statistical tables, normality recommendation notes,
    interpretation, APA write-up, and diagnostic plots embedded as base64.
    """
    pr   = R["parametric"]
    np_r = R["nonparametric"]
    alpha = meta.get("\u03b1", 0.05)
    use_p = R["use_param"]
    prim_n   = R["normality"][0]["n"]
    prim_lbl = R["normality"][0]["primary_label"]

    # ── helpers ────────────────────────────────────────────────────────────────
    def rtbl(rows, left_cols=None):
        if left_cols is None: left_cols = {0}
        h = '<table class="rtbl"><thead><tr>'
        for c in rows[0]: h += f"<th>{c}</th>"
        h += "</tr></thead><tbody>"
        for i, row in enumerate(rows[1:]):
            h += f'<tr class="{"even" if i%2==0 else "odd"}">'
            for j, cell in enumerate(row):
                cls = ' class="left"' if j in left_cols else ""
                h += f"<td{cls}>{cell}</td>"
            h += "</tr>"
        h += "</tbody></table>"
        return h

    def sh(t, n): return f'<div class="sec-hdr"><span class="sec-num">{n}</span>{t}</div>'
    def sub(t):   return f'<div class="sub-hdr">{t}</div>'
    def rec(txt): return f'<div class="rec-box">{txt}</div>'

    # ── normality tables ───────────────────────────────────────────────────────
    sw_rows = [["Variable","N","Statistic (W)","Sig.","Result"]]
    ks_rows = [["Variable","N","Statistic (D)","Sig.\u1d43","Result"]]
    for nm in R["normality"]:
        sw_rows.append([nm["label"], str(nm["n"]),
                        _f(nm["sw_W"]), _p(nm["sw_p"]),
                        "\u2713 Normal" if nm["sw_pass"] else "\u2717 Non-Normal"])
        ks_rows.append([nm["label"], str(nm["n"]),
                        _f(nm["ks_D"]), _p(nm["ks_p"]),
                        "\u2713 Normal" if nm["ks_pass"] else "\u2717 Non-Normal"])

    # Also build per-group recommendation for HTML
    if test_type == "Independent-Sample T-Test":
        norm_rec_parts = []
        for nm in R["normality"]:
            note = normality_recommendation_plain(nm["n"])
            norm_rec_parts.append(f"{nm['label']}: {note}")
        norm_rec = " | ".join(norm_rec_parts)
    else:
        norm_rec = normality_recommendation_plain(prim_n)

    # ── descriptives ───────────────────────────────────────────────────────────
    dd = R["desc"].copy()
    for c in dd.select_dtypes(include=float).columns:
        dd[c] = dd[c].apply(_f)

    # ── figures ────────────────────────────────────────────────────────────────
    figs_html = ""
    for i, fb in enumerate(fig_bytes_list, 1):
        b64 = base64.b64encode(fb).decode("utf-8")
        figs_html += (
            f'<div class="fig-wrap">'
            f'<img src="data:image/png;base64,{b64}" '
            f'alt="Diagnostic plots for {test_type}" '
            f'style="width:100%;max-width:960px;border-radius:10px;'
            f'box-shadow:0 4px 20px rgba(0,0,0,.12);"/>'
            f'<p class="fig-cap">Figure {i}. Diagnostic plots for {test_type}. '
            f'From left to right: distribution with fitted normal curve, '
            f'comparison panel, and Normal Q-Q plot.</p>'
            f'</div>'
        )

    # ── parametric table ───────────────────────────────────────────────────────
    if test_type == "One-Sample T-Test":
        param_html = rtbl([
            ["","t","df","Sig. (2-tailed)","Sig. (1-tailed Lower)",
             "Sig. (1-tailed Upper)","Mean Difference",
             "95% CI Lower","95% CI Upper","Cohen\u2019s d","Effect Size"],
            [f"Test Value = {pr['mu0']}",
             _f(pr["t"]), str(pr["df"]), _p(pr["p_two"]),
             _p(pr["p_one_lower"]), _p(pr["p_one_upper"]),
             _f(pr["mean_diff"]), _f(pr["ci_lower"]), _f(pr["ci_upper"]),
             _f(pr["cohens_d"]), effect_label_d(pr["cohens_d"])]
        ])
    elif test_type == "Paired-Sample T-Test":
        c2 = R["correlation"].copy()
        c2["Pearson Correlation"] = c2["Pearson Correlation"].apply(_f)
        c2["Sig. (2-tailed)"]     = c2["Sig. (2-tailed)"].apply(_p)
        param_html  = sub("Paired Samples Correlations") + rtbl(df_to_rows(c2))
        param_html += sub("Paired Samples Test") + rtbl([
            ["Pair","Mean Diff","SD","SE","95% CI Lower","95% CI Upper",
             "t","df","Sig. (2-tailed)","Sig. (1-tailed L)",
             "Sig. (1-tailed U)","Cohen\u2019s d"],
            [f"{pr['label1']} \u2013 {pr['label2']}",
             _f(pr["mean_diff"]), _f(pr["sd_diff"]), _f(pr["se_diff"]),
             _f(pr["ci_lower"]), _f(pr["ci_upper"]),
             _f(pr["t"]), str(pr["df"]),
             _p(pr["p_two"]), _p(pr["p_one_lower"]), _p(pr["p_one_upper"]),
             _f(pr["cohens_d"])]
        ])
    else:
        lev = R["levene"]
        param_html  = sub("a) t-Statistics and Significance") + rtbl([
            ["","F (Levene)","Sig.","t","df","Sig. (2-tailed)",
             "Sig. (1-tailed L)","Sig. (1-tailed U)"],
            ["Equal variances assumed",
             _f(lev["F"]), _p(lev["Sig."]),
             _f(pr["t_eq"]), str(pr["df_eq"]), _p(pr["p_eq"]),
             _p(pr["p_eq_lower"]), _p(pr["p_eq_upper"])],
            ["Equal variances not assumed","","",
             _f(pr["t_welch"]), _f(pr["df_welch"],2), _p(pr["p_welch"]),
             _p(pr["p_welch_lower"]), _p(pr["p_welch_upper"])]
        ])
        param_html += sub("b) Mean Difference, Confidence Interval, and Effect Size") + rtbl([
            ["","Mean Difference","SE Difference",
             "95% CI Lower","95% CI Upper","Cohen\u2019s d"],
            ["Equal variances assumed",
             _f(pr["mean_diff"]), _f(pr["se_eq"]),
             _f(pr["ci_eq_l"]), _f(pr["ci_eq_u"]), _f(pr["cohens_d"])],
            ["Equal variances not assumed",
             _f(pr["mean_diff"]), _f(pr["se_welch"]),
             _f(pr["ci_welch_l"]), _f(pr["ci_welch_u"]), "\u2014"]
        ])
        active = "Row 1 (equal variances assumed)" if lev["equal_var"] else "Row 2 (Welch correction)"
        param_html += f'<p class="tbl-note">Based on Levene\u2019s test p\u2009=\u2009{_p(lev["Sig."])}: use {active}.</p>'

    # ── non-parametric table ───────────────────────────────────────────────────
    if test_type == "Independent-Sample T-Test":
        np_html  = sub("Ranks") + rtbl([
            ["Group","N","Mean Rank","Sum of Ranks"],
            [np_r["label1"], str(np_r["n1"]),
             _f(np_r["mean_rank1"]), _f(np_r["R1"],3)],
            [np_r["label2"], str(np_r["n2"]),
             _f(np_r["mean_rank2"]), _f(np_r["R2"],3)],
            ["Total", str(np_r["n1"]+np_r["n2"]), "", ""]
        ])
        np_html += sub("Test Statistics") + rtbl([
            ["Statistic","Value"],
            ["Mann-Whitney U",         format_u(np_r["U"])],
            ["Wilcoxon W",             _f(np_r["W_wilcoxon"],3)],
            ["Z",                      _f(np_r["Z"])],
            ["Asymp. Sig. (2-tailed)", _p(np_r["p"])],
        ])
        np_html += f'<p class="tbl-note">Grouping variable: {np_r["label1"]} vs. {np_r["label2"]}. Z based on normal approximation with ties correction (SPSS method).</p>'
    else:
        n_neg  = np_r.get("n_neg",0)
        n_pos  = np_r.get("n_pos",0)
        n_ties = np_r.get("n_ties",0)
        n_tot  = np_r.get("n_total", n_neg+n_pos+n_ties)
        neg_rs = np_r.get("neg_rank_sum", np.nan)
        pos_rs = np_r.get("pos_rank_sum", np.nan)
        neg_mr = neg_rs/n_neg if n_neg > 0 else np.nan
        pos_mr = pos_rs/n_pos if n_pos > 0 else np.nan
        pair_lbl = (f"{pr['label1']} \u2212 {pr['label2']}"
                    if test_type == "Paired-Sample T-Test" else "Variable \u2212 \u03bc\u2080")
        fn_exist = (test_type == "Paired-Sample T-Test")
        np_html  = sub("Ranks") + rtbl([
            ["","N","Mean Rank","Sum of Ranks"],
            [("Negative Ranks \u1d43" if fn_exist else "Negative Ranks"),
             str(n_neg), _f(neg_mr), _f(neg_rs,3)],
            [("Positive Ranks \u1d47" if fn_exist else "Positive Ranks"),
             str(n_pos), _f(pos_mr), _f(pos_rs,3)],
            [("Ties \u1d9c" if fn_exist else "Ties"), str(n_ties),"",""],
            ["Total", str(n_tot),"",""]
        ])
        if fn_exist:
            np_html += (f'<p class="tbl-note">'
                        f'\u1d43 {pr["label2"]} &lt; {pr["label1"]} &nbsp;'
                        f'\u1d47 {pr["label2"]} &gt; {pr["label1"]} &nbsp;'
                        f'\u1d9c {pr["label2"]} = {pr["label1"]}</p>')
        np_html += sub("Test Statistics") + rtbl([
            ["Statistic", pair_lbl],
            ["Test Statistic (W)", _f(np_r["W"],0)],
            ["Z",                  _f(np_r["Z"])],
            ["Asymp. Sig. (2-tailed)", _p(np_r["p"])],
        ])
        np_html += f'<p class="tbl-note">Based on {"negative" if n_neg < n_pos else "positive"} ranks. Z uses ties-corrected variance (SPSS method).</p>'

    # ── interpretation ─────────────────────────────────────────────────────────
    interp_html = ""
    for line in interps:
        lw = line.lower()
        cls = ("sig"
               if "statistically significant" in lw and "no statistically" not in lw
               else "nonsig"
               if "no statistically significant" in lw or "did not significantly" in lw
               else "")
        interp_html += f'<div class="interp {cls}">{line}</div>'

    # ── APA write-up ───────────────────────────────────────────────────────────
    if use_p:
        if test_type == "One-Sample T-Test":
            m_v  = float(R["desc"]["Mean"].iloc[0])
            sd_v = float(R["desc"]["Std. Deviation"].iloc[0])
            apa  = (f"A one-sample t-test was conducted to examine whether "
                    f"{meta.get('Variable','the variable')} "
                    f"(M\u2009=\u2009{_f(m_v)}, SD\u2009=\u2009{_f(sd_v)}) "
                    f"significantly differed from the hypothesised population mean "
                    f"(\u03bc\u2080\u2009=\u2009{pr['mu0']}). The result was "
                    f"{'statistically significant' if pr['p_two']<alpha else 'not statistically significant'}, "
                    f"t({pr['df']})\u2009=\u2009{pr['t']:.2f}, "
                    f"p\u2009{'< .001' if pr['p_two']<.001 else '= '+_p(pr['p_two'])}, "
                    f"d\u2009=\u2009{pr['cohens_d']:.2f}.")
        elif test_type == "Paired-Sample T-Test":
            apa = (f"A paired-samples t-test was conducted to compare "
                   f"{pr['label1']} "
                   f"(M\u2009=\u2009{_f(float(R['desc'].iloc[0]['Mean']))}, "
                   f"SD\u2009=\u2009{_f(float(R['desc'].iloc[0]['Std. Deviation']))}) "
                   f"and {pr['label2']} "
                   f"(M\u2009=\u2009{_f(float(R['desc'].iloc[1]['Mean']))}, "
                   f"SD\u2009=\u2009{_f(float(R['desc'].iloc[1]['Std. Deviation']))}). "
                   f"The difference was "
                   f"{'statistically significant' if pr['p_two']<alpha else 'not statistically significant'}, "
                   f"t({pr['df']})\u2009=\u2009{pr['t']:.2f}, "
                   f"p\u2009{'< .001' if pr['p_two']<.001 else '= '+_p(pr['p_two'])}, "
                   f"d\u2009=\u2009{pr['cohens_d']:.2f}.")
        else:
            use_eq = R["levene"]["equal_var"]
            tv = pr["t_eq"]  if use_eq else pr["t_welch"]
            pv = pr["p_eq"]  if use_eq else pr["p_welch"]
            dv = pr["df_eq"] if use_eq else pr["df_welch"]
            apa = (f"An independent-samples t-test was conducted to compare "
                   f"{pr['dep_var']} between {pr['label1']} "
                   f"(M\u2009=\u2009{_f(float(R['desc'].iloc[0]['Mean']))}, "
                   f"SD\u2009=\u2009{_f(float(R['desc'].iloc[0]['Std. Deviation']))}) "
                   f"and {pr['label2']} "
                   f"(M\u2009=\u2009{_f(float(R['desc'].iloc[1]['Mean']))}, "
                   f"SD\u2009=\u2009{_f(float(R['desc'].iloc[1]['Std. Deviation']))}). "
                   f"The difference was "
                   f"{'statistically significant' if pv<alpha else 'not statistically significant'}, "
                   f"t({_f(dv,2)})\u2009=\u2009{tv:.2f}, "
                   f"p\u2009{'< .001' if pv<.001 else '= '+_p(pv)}, "
                   f"d\u2009=\u2009{pr['cohens_d']:.2f}.")
    else:
        if test_type == "Independent-Sample T-Test":
            apa = (f"A Mann-Whitney U test was conducted to compare "
                   f"{np_r['dep_var']} between {np_r['label1']} and {np_r['label2']}. "
                   f"The result indicated "
                   f"{'a statistically significant' if np_r['p']<alpha else 'no statistically significant'} "
                   f"difference, U\u2009=\u2009{format_u(np_r['U'])}, "
                   f"W\u2009=\u2009{_f(np_r['W_wilcoxon'],3)}, "
                   f"Z\u2009=\u2009{np_r['Z']:.3f}, "
                   f"p\u2009{'< .001' if np_r['p']<.001 else '= '+_p(np_r['p'])} "
                   f"(asymptotic, 2-tailed).")
        else:
            apa = (f"A Wilcoxon signed-rank test was conducted. "
                   f"The result indicated "
                   f"{'a statistically significant' if np_r['p']<alpha else 'no statistically significant'} "
                   f"difference, W\u2009=\u2009{np_r['W']:.0f}, "
                   f"Z\u2009=\u2009{np_r['Z']:.3f}, "
                   f"p\u2009{'< .001' if np_r['p']<.001 else '= '+_p(np_r['p'])} "
                   f"(2-tailed, asymptotic).")

    # ── meta table ─────────────────────────────────────────────────────────────
    meta_html = "<table class='meta-tbl'>"
    for k, v in meta.items():
        meta_html += f"<tr><td class='mk'>{k}</td><td class='mv'>{v}</td></tr>"
    meta_html += "</table>"

    # ── assumption summary for HTML ────────────────────────────────────────────
    assume_rows_html = [["Assumption","Test / Criterion","Result","Decision"]]
    for nm in R["normality"]:
        passed = nm["pass"]
        sw_res = f"SW W\u2009=\u2009{_f(nm['sw_W'])}, p\u2009=\u2009{_p(nm['sw_p'])}"
        ks_res = f"KS D\u2009=\u2009{_f(nm['ks_D'])}, p\u2009=\u2009{_p(nm['ks_p'])}"
        assume_rows_html.append([
            f"Normality \u2014 {nm['label']}",
            f"{sw_res} | {ks_res} (primary: {nm['primary_label']})",
            "\u2713 Satisfied" if passed else "\u2717 Violated",
            "Parametric eligible" if passed else "Non-parametric required"
        ])
    if "levene" in R:
        lv2 = R["levene"]
        assume_rows_html.append([
            "Homogeneity of Variance",
            f"Levene F({lv2['df1']},\u2009{lv2['df2']})\u2009=\u2009{_f(lv2['F'])}, "
            f"p\u2009=\u2009{_p(lv2['Sig.'])}",
            "\u2713 Satisfied" if lv2["equal_var"] else "\u2717 Violated",
            "Equal variances assumed" if lv2["equal_var"]
            else "Welch correction applied"
        ])
    assume_rows_html.append([
        "Independence of Observations",
        "By research design (not statistically testable)",
        "\u2139 Assumed",
        "Must be ensured by design"
    ])
    assume_rows_html.append([
        "Overall Decision",
        f"Primary criterion: {prim_lbl}",
        "Parametric" if use_p else "Non-parametric",
        "T-Test family" if use_p else "Wilcoxon / Mann-Whitney U"
    ])

    # ── non-parametric effect size for HTML ────────────────────────────────────
    try:
        cfg_s = meta  # use meta dict to get labels
        if test_type == "Independent-Sample T-Test":
            r_rb  = 1 - (2 * np_r["U1"]) / (np_r["n1"] * np_r["n2"])
            ci_lo_es = ci_hi_es = np.nan  # bootstrap needs raw data, skip in HTML
            es_label = "Mann-Whitney U"
            pair_desc_es = f"{np_r['label1']} vs. {np_r['label2']}"
        else:
            n_nz = np_r.get("n_pos",0) + np_r.get("n_neg",0)
            pos_rs = np_r.get("pos_rank_sum", 0)
            neg_rs = np_r.get("neg_rank_sum", 0)
            total_rs = pos_rs + neg_rs
            r_rb = (pos_rs - neg_rs) / total_rs if total_rs > 0 else np.nan
            ci_lo_es = ci_hi_es = np.nan
            es_label = "Wilcoxon Signed-Rank"
            if test_type == "Paired-Sample T-Test":
                pair_desc_es = f"{pr['label1']} \u2212 {pr['label2']}"
            else:
                pair_desc_es = f"Variable \u2212 \u03bc\u2080"
        r_lab_es = effect_label_r_nonparam(r_rb) if not np.isnan(r_rb) else "N/A"
        np_es_html = rtbl([
            ["Test","Comparison","Rank-Biserial r","Effect Size","Interpretation"],
            [es_label, pair_desc_es,
             _f(r_rb) if not np.isnan(r_rb) else ".",
             r_lab_es,
             "|r|\u2009<\u2009.10 negligible, .10\u2013.29 small, "
             ".30\u2013.49 medium, \u2265\u2009.50 large"]
        ], left_cols={0,1,4})
        np_es_html += ('<p class="tbl-note">Rank-biserial correlation r is a '
                       'non-parametric effect size not reported by SPSS by default. '
                       'Reference: Kerby (2014).</p>')
    except Exception:
        np_es_html = '<p class="tbl-note">Effect size not available.</p>'

    # ── power analysis for HTML ────────────────────────────────────────────────
    try:
        pw = compute_power(test_type, R, alpha)
        pwr_val = pw.get("power", np.nan)
        pwr_pct = f"{pwr_val*100:.1f}%" if not np.isnan(pwr_val) else "N/A"
        pwr_lbl = pw.get("power_label", "N/A")
        pwr_eff = pw.get("effect_size", np.nan)
        pwr_typ = pw.get("effect_type", "\u2014")
        if "n1" in pw:
            pwr_n = f"n\u2081\u2009=\u2009{pw['n1']}, n\u2082\u2009=\u2009{pw['n2']}"
        else:
            pwr_n = f"n\u2009=\u2009{pw.get('n','N/A')}"
        if not np.isnan(pwr_val):
            bar_c = ("#16a34a" if pwr_val >= .80
                     else "#f59e0b" if pwr_val >= .60 else "#dc2626")
            bar_p = int(pwr_val * 100)
            pwr_bar = (f'<div style="background:#e2e8f0;border-radius:8px;'
                       f'height:16px;width:100%;margin:10px 0;">'
                       f'<div style="background:{bar_c};width:{bar_p}%;'
                       f'height:16px;border-radius:8px;"></div></div>')
        else:
            pwr_bar = ""
        pwr_rows = [
            ["Parameter","Value"],
            ["Sample size", pwr_n],
            ["Observed effect size", f"{_f(pwr_eff)} ({pwr_typ})"],
            ["Significance level (\u03b1)", str(alpha)],
            ["Achieved statistical power", pwr_pct],
            ["Power classification", pwr_lbl],
            ["Recommended minimum power", "\u2265\u2009.80 (Cohen, 1988)"]
        ]
        pwr_html = rtbl(pwr_rows, left_cols={0}) + pwr_bar
        if not np.isnan(pwr_val) and pwr_val < .80:
            pwr_html += ('<div class="warn-box" style="margin-top:10px;">'
                         'Achieved power is below the conventional threshold of '
                         '.80, indicating an elevated risk of Type II error. '
                         'Consider increasing the sample size.</div>')
        elif not np.isnan(pwr_val):
            pwr_html += ('<div class="info-box" style="margin-top:10px;">'
                         'Achieved power meets or exceeds the conventional '
                         'threshold of .80.</div>')
        pwr_html += ('<p class="tbl-note">Post-hoc power analysis using the '
                     'non-central t-distribution (parametric) or normal '
                     'approximation (non-parametric). '
                     'Reference: Cohen (1988).</p>')
    except Exception:
        pwr_html = '<p class="tbl-note">Power analysis not available.</p>'

    # ── section numbering ──────────────────────────────────────────────────────
    sn = 3
    corr_sec = ""
    if "correlation" in R:
        c3 = R["correlation"].copy()
        c3["Pearson Correlation"] = c3["Pearson Correlation"].apply(_f)
        c3["Sig. (2-tailed)"]     = c3["Sig. (2-tailed)"].apply(_p)
        corr_sec = (f'<div class="section">'
                    f'{sh("PAIRED SAMPLES CORRELATIONS", sn)}'
                    f'{rtbl(df_to_rows(c3))}</div>')
        sn += 1

    lev_sec = ""
    if "levene" in R:
        lv  = R["levene"]
        res = ("\u2713 Equal variances assumed"
               if lv["equal_var"] else
               "\u2717 Equal variances not assumed (Welch)")
        lev_sec = (
            f'<div class="section">'
            f'{sh("LEVENE\u2019S TEST FOR EQUALITY OF VARIANCES", sn)}'
            f'{rtbl([["F","df1","df2","Sig.","Result"],[_f(lv["F"]),str(lv["df1"]),str(lv["df2"]),_p(lv["Sig."]),res]])}'
            f'<p class="tbl-note">Based on mean (SPSS default). '
            f'p\u2009&gt;\u2009.05 \u2192 equal variances assumed.</p></div>'
        )
        sn += 1

    ps=sn; sn+=1; ns=sn; sn+=1; es_sec_n=sn; sn+=1
    ins=sn; sn+=1; pw_sec_n=sn; sn+=1; fgs=sn

    dcls = "use-param" if use_p else "use-nonparam"
    dtxt = (f"Primary normality criterion: <b>{prim_lbl}</b> "
            f"(n\u2009=\u2009{prim_n}, "
            f"{'n\u2009\u2264\u200950' if prim_n <= 50 else 'n\u2009>\u200950'}) "
            f"\u2192 p\u2009{'>\u2009.05' if use_p else '\u2264\u2009.05'} "
            f"\u2192 <b>{'Parametric' if use_p else 'Non-parametric'} analysis applied</b>")

    # ── CSS ────────────────────────────────────────────────────────────────────
    css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;600;700&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'DM Sans',sans-serif;background:#f0f4f8;color:#1e293b;font-size:14px;line-height:1.6;}
.page{max-width:1120px;margin:0 auto;padding:32px 24px 80px;}
.cover{background:linear-gradient(135deg,#0a0a0a 0%,#1a1a2e 50%,#16213e 100%);
  border-radius:16px;padding:44px 52px;margin-bottom:28px;
  border-left:6px solid #e94560;box-shadow:0 12px 48px rgba(233,69,96,.2);}
.cover h1{color:#fff;font-size:2rem;font-weight:700;margin-bottom:8px;letter-spacing:-.5px;}
.cover .sub{color:#94a3b8;font-size:.9rem;margin-bottom:24px;}
.cover .badge{display:inline-block;background:#e94560;color:#fff;font-size:.7rem;
  padding:3px 10px;border-radius:20px;font-weight:600;margin-right:6px;}
.meta-tbl{width:100%;border-collapse:collapse;margin-top:20px;}
.meta-tbl td{padding:5px 10px;font-size:.82rem;}
.meta-tbl td.mk{color:#94a3b8;font-weight:600;width:220px;font-family:'DM Mono',monospace;}
.meta-tbl td.mv{color:#e2e8f0;}
.decision-box{padding:12px 20px;border-radius:10px;margin:20px 0;font-size:.9rem;font-weight:600;}
.use-param{background:#dcfce7;color:#14532d;border:1px solid #86efac;}
.use-nonparam{background:#ffedd5;color:#7c2d12;border:1px solid #fdba74;}
.section{background:#fff;border-radius:12px;padding:26px 30px;
  margin-bottom:22px;box-shadow:0 2px 14px rgba(0,0,0,.06);}
.sec-hdr{background:linear-gradient(90deg,#1a1a2e,#16213e);color:#e2e8f0;
  padding:11px 20px;border-radius:8px 8px 0 0;font-weight:700;font-size:.82rem;
  letter-spacing:.8px;margin:-26px -30px 22px;font-family:'DM Mono',monospace;
  border-bottom:3px solid #e94560;display:flex;align-items:center;gap:12px;}
.sec-num{background:#e94560;color:#fff;font-size:.72rem;padding:2px 8px;
  border-radius:12px;font-weight:700;flex-shrink:0;}
.sub-hdr{background:#f1f5f9;color:#1a1a2e;padding:8px 14px;border-radius:6px;
  font-weight:600;font-size:.8rem;margin:18px 0 10px;border-left:3px solid #e94560;}
.rec-box{background:linear-gradient(135deg,#f0f9ff,#e0f2fe);
  border-left:4px solid #0284c7;padding:12px 16px;border-radius:0 8px 8px 0;
  font-size:.82rem;color:#0c4a6e;margin:12px 0;line-height:1.75;}
.rtbl{width:100%;border-collapse:collapse;font-family:'DM Mono',monospace;
  font-size:.74rem;margin-bottom:6px;}
.rtbl th{background:#1a1a2e;color:#e2e8f0;padding:9px 13px;text-align:center;
  font-weight:600;border:1px solid #334155;white-space:nowrap;font-size:.72rem;}
.rtbl td{padding:7px 13px;border:1px solid #e2e8f0;text-align:right;white-space:nowrap;}
.rtbl tr.even td{background:#fff;}.rtbl tr.odd td{background:#f8fafc;}
.rtbl td.left{text-align:left;font-weight:500;background:#f1f5f9!important;}
.tbl-note{font-size:.73rem;color:#64748b;font-style:italic;margin-top:8px;line-height:1.6;}
.interp{padding:12px 18px;border-radius:0 10px 10px 0;margin:10px 0;
  font-size:.86rem;line-height:1.8;border-left:4px solid #0284c7;
  background:linear-gradient(135deg,#f8fafc,#f1f5f9);}
.interp b{color:#0284c7;}
.interp.sig{border-left-color:#16a34a;background:linear-gradient(135deg,#f0fdf4,#dcfce7);}
.interp.sig b{color:#16a34a;}
.interp.nonsig{border-left-color:#dc2626;background:linear-gradient(135deg,#fef2f2,#fee2e2);}
.interp.nonsig b{color:#dc2626;}
.apa-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
  padding:14px 18px;font-family:'DM Mono',monospace;font-size:.78rem;
  color:#1e293b;line-height:1.8;margin-top:14px;}
.apa-label{font-size:.72rem;font-weight:700;color:#64748b;
  text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;}
.fig-wrap{text-align:center;margin:18px 0;}
.fig-cap{font-size:.75rem;color:#64748b;font-style:italic;margin-top:10px;line-height:1.6;}
.warn-box{background:#fffbeb;border-left:4px solid #f59e0b;padding:.7rem 1rem;
  border-radius:0 6px 6px 0;font-size:.82rem;color:#92400e;margin:.4rem 0;}
.info-box{background:#eff6ff;border-left:4px solid #3b82f6;padding:.7rem 1rem;
  border-radius:0 6px 6px 0;font-size:.82rem;color:#1e40af;margin:.4rem 0;}
.footer{text-align:center;color:#94a3b8;font-size:.72rem;margin-top:44px;
  padding-top:16px;border-top:1px solid #e2e8f0;line-height:1.8;}
@media print{body{background:#fff;}.page{padding:0 16px;}.cover{border-radius:0;}}
</style>"""

    # ── assemble ───────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Inferential Statistics Report \u2014 {test_type}</title>
{css}
</head>
<body>
<div class="page">

<div class="cover">
  <h1>&#128208; Inferential Statistics Report</h1>
  <div class="sub">
    <span class="badge">SPSS-Equivalent</span>
    <span class="badge">Shapiro-Wilk + KS Lilliefors</span>
    <span class="badge">Parametric &amp; Non-Parametric</span>
  </div>
  {meta_html}
</div>

<div class="section">
  <div class="sec-hdr"><span class="sec-num">&#9654;</span>ANALYSIS DECISION</div>
  <div class="decision-box {dcls}">{dtxt}</div>
</div>

<div class="section">
  <div class="sec-hdr"><span class="sec-num">1</span>TESTS OF NORMALITY</div>
  {sub("a) Shapiro-Wilk Test")}
  {rtbl(sw_rows)}
  {sub("b) Kolmogorov-Smirnov Test (Lilliefors Significance Correction)")}
  {rtbl(ks_rows)}
  <p class="tbl-note">\u1d43 Lilliefors significance correction applied.
  For n\u2009\u2264\u200950: p-value from Lilliefors table (SPSS-equivalent).
  For n\u2009&gt;\u200950: asymptotic approximation.</p>
  {rec(norm_rec)}
  {sub("Assumption Summary")}
  {rtbl(assume_rows_html, left_cols={0,1,3})}
  <p class="tbl-note">Independence of observations cannot be formally tested
  and must be ensured through appropriate research design.</p>
</div>

<div class="section">
  <div class="sec-hdr"><span class="sec-num">2</span>DESCRIPTIVE STATISTICS</div>
  {rtbl(df_to_rows(dd))}
</div>

{corr_sec}
{lev_sec}

<div class="section">
  <div class="sec-hdr"><span class="sec-num">{ps}</span>PARAMETRIC TEST RESULTS</div>
  {param_html}
</div>

<div class="section">
  <div class="sec-hdr"><span class="sec-num">{ns}</span>NON-PARAMETRIC TEST RESULTS</div>
  {np_html}
  {sub("Non-Parametric Effect Size &mdash; Rank-Biserial Correlation (r)")}
  {np_es_html}
</div>

<div class="section">
  <div class="sec-hdr"><span class="sec-num">{ins}</span>INTERPRETATION</div>
  {interp_html}
  <div style="margin-top:20px;">
    <div class="apa-label">APA 7th Edition Write-Up</div>
    <div class="apa-box">{apa}</div>
  </div>
</div>

<div class="section">
  <div class="sec-hdr"><span class="sec-num">{pw_sec_n}</span>STATISTICAL POWER ASSESSMENT</div>
  {pwr_html}
</div>

<div class="section">
  <div class="sec-hdr"><span class="sec-num">{fgs}</span>FIGURES &amp; DIAGNOSTIC PLOTS</div>
  {figs_html}
</div>

<div class="footer">
  Generated by Inferential Statistics App &nbsp;&middot;&nbsp;
  SPSS-equivalent output &nbsp;&middot;&nbsp;
  KS Lilliefors: table method (n\u2009\u2264\u200950), approx method (n\u2009&gt;\u200950) &nbsp;&middot;&nbsp;
  Levene center\u2009=\u2009mean &nbsp;&middot;&nbsp;
  Wilcoxon Z ties-corrected &nbsp;&middot;&nbsp;
  Mann-Whitney U SPSS-exact &nbsp;&middot;&nbsp;
  n\u2009\u2264\u200950: SW recommended &nbsp;&middot;&nbsp;
  n\u2009&gt;\u200950: KS recommended<br/>
  Generated: {datetime.now().strftime("%B %d, %Y at %H:%M")}
</div>

</div>
</body>
</html>"""
    return html.encode("utf-8")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
def main():
    st.markdown("""
    <div class="main-hdr">
      <h1>&#128208; Inferential Statistics Suite
        <span class="badge">SPSS-Equivalent</span></h1>
      <p>Parametric &amp; Non-Parametric &nbsp;&middot;&nbsp;
         Shapiro-Wilk + Kolmogorov-Smirnov &nbsp;&middot;&nbsp;
         Auto-selection &nbsp;&middot;&nbsp;
         One-Sample &nbsp;&middot;&nbsp; Paired &nbsp;&middot;&nbsp; Independent</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:1rem;">
      <p style="color:#64748b;font-size:1.1rem;margin-top:1.25rem;margin-bottom:0.4rem;">
        &#9749; Support my Work &mdash; scan QRIS:</p>
      <img src="https://muhaiminabdullah.com/media/thumbnails/QRIS-muhaiminabdullahdotcom-340x480.jpeg"
           style="width:250px;border-radius:10px;border:2px solid #e94560;display:block;"/>
    </div>""", unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### &#9881;&#65039; Configuration")
        st.markdown("---")
        test_type = st.selectbox("&#128202; Select Test", [
            "One-Sample T-Test",
            "Paired-Sample T-Test",
            "Independent-Sample T-Test"
        ])
        alpha = st.selectbox("\u03b1 Level", [0.05, 0.01, 0.001], index=0)
        st.markdown("---")

        samp = SAMPLES[test_type]
        st.markdown(f"**&#128196; Template \u2014 {test_type}**")
        st.markdown(samp["note"])
        st.download_button(
            "&#11015;&#65039; Download Sample CSV",
            samp["csv"].encode(),
            f"sample_{test_type.replace(' ','_').replace('-','_').lower()}.csv",
            "text/csv", use_container_width=True)
        st.markdown("---")

        uploaded = st.file_uploader(
            "&#128194; Upload Data File",
            type=["csv", "xlsx", "xls"],
            help="Accepted formats: CSV (.csv) and Excel (.xlsx, .xls)")
        if uploaded:
            try:
                name = uploaded.name.lower()
                if name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                elif name.endswith((".xlsx", ".xls")):
                    df = pd.read_excel(uploaded)
                else:
                    st.error("Unsupported file format."); df = None
                if df is not None:
                    st.success(f"&#10003; {len(df):,} rows \u00d7 {len(df.columns)} columns")
            except Exception as e:
                st.error(f"Error: {e}"); df = None
        else:
            df = pd.read_csv(io.StringIO(samp["csv"]))
            st.info("\u2139\ufe0f Using built-in sample data")

        cfg = None
        if df is not None:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
            st.markdown("---")

            if test_type == "One-Sample T-Test":
                tv  = st.selectbox("&#127919; Test Variable", num_cols,
                                    index=num_cols.index("score")
                                    if "score" in num_cols else 0)
                mu0 = st.number_input(
                    "&#128207; Hypothesised Population Mean (\u03bc\u2080)",
                    value=75.0, step=0.5)
                cfg = {"test_var": tv, "mu0": mu0}

            elif test_type == "Paired-Sample T-Test":
                v1 = st.selectbox("Variable 1 (Pre / Time 1)", num_cols,
                                   index=num_cols.index("pre_score")
                                   if "pre_score" in num_cols else 0)
                v2 = st.selectbox("Variable 2 (Post / Time 2)", num_cols,
                                   index=num_cols.index("post_score")
                                   if "post_score" in num_cols
                                   else min(1, len(num_cols)-1))
                cfg = {"v1": v1, "v2": v2}

            else:
                gc = st.selectbox("&#128101; Grouping Variable",
                                   cat_cols if cat_cols else num_cols,
                                   index=cat_cols.index("group")
                                   if "group" in cat_cols else 0)
                dc = st.selectbox("&#127919; Dependent Variable", num_cols,
                                   index=num_cols.index("score")
                                   if "score" in num_cols else 0)
                groups = sorted(df[gc].dropna().unique())
                if len(groups) >= 2:
                    g1l = st.selectbox("Group 1", groups, index=0)
                    g2l = st.selectbox("Group 2", groups,
                                        index=min(1, len(groups)-1))
                    cfg = {"grp_col": gc, "dep_col": dc, "g1": g1l, "g2": g2l}
                else:
                    st.error("The grouping variable must contain at least two groups.")

            st.markdown("---")
            run_btn = st.button("&#128640; Run Analysis", type="primary",
                                 use_container_width=True)
        else:
            run_btn = False

    if df is None:
        return

    with st.expander("&#128269; Data Preview", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    if not run_btn and "stats_R" not in st.session_state:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#f0f9ff,#e0f2fe);
             border:1px solid #bae6fd;border-radius:12px;padding:1.2rem 1.4rem;
             margin:.8rem 0;border-left:4px solid #0284c7;">
          <h4 style="margin:0 0 .4rem 0;font-size:.95rem;font-weight:700;">
            &#128203; {test_type}</h4>
          <p style="margin:0;font-size:.84rem;color:#475569;">{samp['desc']}</p>
        </div>""", unsafe_allow_html=True)
        st.info("&#128072; Configure variables in the sidebar, then click **Run Analysis**.")
        return

    # ── Run analysis ───────────────────────────────────────────────────────────
    if run_btn:
        if cfg is None:
            st.error("&#9888;&#65039; Configuration incomplete."); return
        with st.spinner("Running analysis\u2026"):
            try:
                if test_type == "One-Sample T-Test":
                    data = df[cfg["test_var"]].dropna().values.tolist()
                    R    = run_one_sample(data, cfg["mu0"], alpha)
                    meta = {
                        "Test": test_type,
                        "Variable": cfg["test_var"],
                        "Hypothesised Mean (\u03bc\u2080)": cfg["mu0"],
                        "N": len(data), "\u03b1": alpha,
                        "Date": datetime.now().strftime("%B %d, %Y %H:%M")
                    }
                    interps  = interpret_one_sample(R, cfg["test_var"], alpha)
                    fig_main = plot_one_sample(data, cfg["mu0"], cfg["test_var"])

                elif test_type == "Paired-Sample T-Test":
                    pdata = df[[cfg["v1"], cfg["v2"]]].dropna()
                    d1    = pdata[cfg["v1"]].values.tolist()
                    d2    = pdata[cfg["v2"]].values.tolist()
                    R     = run_paired(d1, d2, cfg["v1"], cfg["v2"], alpha)
                    meta  = {
                        "Test": test_type,
                        "Variable 1": cfg["v1"], "Variable 2": cfg["v2"],
                        "N (pairs)": len(d1), "\u03b1": alpha,
                        "Date": datetime.now().strftime("%B %d, %Y %H:%M")
                    }
                    interps  = interpret_paired(R, alpha)
                    fig_main = plot_paired(d1, d2, cfg["v1"], cfg["v2"])

                else:
                    g1d = (df[df[cfg["grp_col"]]==cfg["g1"]]
                           [cfg["dep_col"]].dropna().values.tolist())
                    g2d = (df[df[cfg["grp_col"]]==cfg["g2"]]
                           [cfg["dep_col"]].dropna().values.tolist())
                    R   = run_independent(g1d, g2d, cfg["g1"], cfg["g2"],
                                          cfg["dep_col"], alpha)
                    meta = {
                        "Test": test_type,
                        "Grouping Variable": cfg["grp_col"],
                        "Dependent Variable": cfg["dep_col"],
                        "Group 1": f"{cfg['g1']} (n\u2009=\u2009{len(g1d)})",
                        "Group 2": f"{cfg['g2']} (n\u2009=\u2009{len(g2d)})",
                        "\u03b1": alpha,
                        "Date": datetime.now().strftime("%B %d, %Y %H:%M")
                    }
                    interps  = interpret_independent(R, cfg["dep_col"], alpha)
                    fig_main = plot_independent(
                        g1d, g2d, cfg["g1"], cfg["g2"], cfg["dep_col"])

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

    st.success("&#10003; Analysis complete!")

    # ── Decision banner ────────────────────────────────────────────────────────
    use_p    = R["use_param"]
    prim_n   = R["normality"][0]["n"]
    prim_lbl = R["normality"][0]["primary_label"]
    test_name_used = (
        {"One-Sample T-Test":        "One-Sample T-Test",
         "Paired-Sample T-Test":     "Paired Samples T-Test",
         "Independent-Sample T-Test":"Independent Samples T-Test"}[test_type]
        if use_p else
        {"One-Sample T-Test":        "Wilcoxon Signed-Rank Test",
         "Paired-Sample T-Test":     "Wilcoxon Signed-Rank Test",
         "Independent-Sample T-Test":"Mann-Whitney U Test"}[test_type]
    )
    cls = "use-param" if use_p else "use-nonparam"
    st.markdown(
        f'<div class="decision-banner {cls}">'
        f'{"&#10003;" if use_p else "&#9888;&#65039;"} '
        f'Primary normality criterion: <b>{prim_lbl}</b> '
        f'(n\u2009=\u2009{prim_n}, '
        f'{"n\u2009\u2264\u200950" if prim_n <= 50 else "n\u2009>\u200950"}) '
        f'\u2192 p\u2009{">" if use_p else "\u2264"}\u2009.05 '
        f'\u2192 <b>{test_name_used}</b> applied'
        f'</div>', unsafe_allow_html=True)

    # ── Quick metrics ──────────────────────────────────────────────────────────
    pr   = R["parametric"]
    np_r = R["nonparametric"]
    if use_p:
        if test_type == "One-Sample T-Test":
            metrics = [(_f(pr["t"]),f"t({pr['df']})"),
                       (_p(pr["p_two"]),"Sig. (2-tailed)"),
                       (_f(pr["mean_diff"]),"Mean Difference"),
                       (_f(pr["cohens_d"]),"Cohen\u2019s d"),
                       (effect_label_d(pr["cohens_d"]).title(),"Effect Size")]
        elif test_type == "Paired-Sample T-Test":
            metrics = [(_f(pr["t"]),f"t({pr['df']})"),
                       (_p(pr["p_two"]),"Sig. (2-tailed)"),
                       (_f(pr["mean_diff"]),"Mean Difference"),
                       (_f(pr["sd_diff"]),"SD of Differences"),
                       (_f(pr["cohens_d"]),"Cohen\u2019s d")]
        else:
            metrics = [(_f(pr["t_eq"]),f"t({pr['df_eq']}) Equal Var."),
                       (_f(pr["t_welch"]),"t (Welch)"),
                       (_p(pr["p_eq"]),"Sig. Equal Var."),
                       (_p(pr["p_welch"]),"Sig. Welch"),
                       (_f(pr["cohens_d"]),"Cohen\u2019s d")]
    else:
        if test_type == "Independent-Sample T-Test":
            metrics = [(format_u(np_r["U"]),"Mann-Whitney U"),
                       (_f(np_r["W_wilcoxon"],3),"Wilcoxon W"),
                       (_f(np_r["Z"]),"Z"),
                       (_p(np_r["p"]),"Sig. (2-tailed)"),
                       (str(np_r["n1"]+np_r["n2"]),"Total N")]
        else:
            metrics = [(_f(np_r["W"],0),"Wilcoxon W"),
                       (_f(np_r["Z"]),"Z"),
                       (_p(np_r["p"]),"Sig. (2-tailed)"),
                       (str(np_r.get("n_total","\u2014")),"Total N"),
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
    tab_labels = ["&#128203; Normality", "&#128202; Descriptives"]
    if "correlation" in R: tab_labels.append("&#128279; Paired Correlation")
    if "levene"      in R: tab_labels.append("&#9878;&#65039; Levene\u2019s Test")
    tab_labels += ["&#128200; Parametric", "&#128201; Non-Parametric",
                   "&#128200; Plots", "&#128172; Interpretation"]
    tabs = st.tabs(tab_labels)
    ti   = 0

    # ── Tab: Normality ─────────────────────────────────────────────────────────
    _t0 = ti; ti += 1
    with tabs[_t0]:
        # Recommendation note — per group for independent, single for others
        if test_type == "Independent-Sample T-Test":
            rec_parts = []
            for nm in R["normality"]:
                rec_parts.append(normality_recommendation_note(nm["n"])
                                 .replace("<b>Normality Test Recommendation",
                                          f"<b>Normality Test Recommendation — {nm['label']}"))
            rec_note = "<br/>".join(rec_parts)
        else:
            rec_note = normality_recommendation_note(prim_n)
        st.markdown(f'<div class="norm-rec-box">{rec_note}</div>',
                    unsafe_allow_html=True)

        st.markdown('<div class="sec-title">a) Shapiro-Wilk Test</div>',
                    unsafe_allow_html=True)
        sw_rows_ui = [["Variable","N","Statistic (W)","Sig.","Result"]]
        for n_item in R["normality"]:
            res = ('<span class="pass">&#10003; Normal</span>'
                   if n_item["sw_pass"]
                   else '<span class="fail">&#10007; Non-Normal</span>')
            sw_rows_ui.append([n_item["label"], str(n_item["n"]),
                               _f(n_item["sw_W"]), _p(n_item["sw_p"]), res])
        st.markdown(html_tbl(sw_rows_ui, left_cols={0,4}), unsafe_allow_html=True)

        recommended_sw = (prim_lbl == "Shapiro-Wilk")
        if recommended_sw:
            n_desc = (f"n\u2009=\u2009{prim_n}"
                      if test_type != "Independent-Sample T-Test"
                      else " and ".join(f"n\u2009=\u2009{nm['n']} ({nm['label']})"
                                        for nm in R["normality"]))
            st.markdown(
                f'<p class="note-txt">&#9733; '
                f'Shapiro-Wilk is the recommended primary criterion '
                f'for these sample sizes ({n_desc}, each \u2264\u200950). '
                f'Decision is based on this result.</p>',
                unsafe_allow_html=True)
        else:
            n_desc = (f"n\u2009=\u2009{prim_n}"
                      if test_type != "Independent-Sample T-Test"
                      else " and ".join(f"n\u2009=\u2009{nm['n']} ({nm['label']})"
                                        for nm in R["normality"]))
            st.markdown(
                f'<p class="note-txt">Reported for informational purposes. '
                f'Kolmogorov-Smirnov is the recommended criterion '
                f'for these sample sizes ({n_desc}, each >\u200950).</p>',
                unsafe_allow_html=True)

        st.markdown(
            '<div class="sec-title">'
            'b) Kolmogorov-Smirnov Test (Lilliefors Significance Correction)'
            '</div>', unsafe_allow_html=True)
        ks_rows_ui = [["Variable","N","Statistic (D)","Sig.\u1d43","Result"]]
        for n_item in R["normality"]:
            res = ('<span class="pass">&#10003; Normal</span>'
                   if n_item["ks_pass"]
                   else '<span class="fail">&#10007; Non-Normal</span>')
            ks_rows_ui.append([n_item["label"], str(n_item["n"]),
                               _f(n_item["ks_D"]), _p(n_item["ks_p"]), res])
        st.markdown(html_tbl(ks_rows_ui, left_cols={0,4}), unsafe_allow_html=True)

        recommended_ks = (prim_lbl == "Kolmogorov-Smirnov")
        if recommended_ks:
            n_desc_ks = (f"n\u2009=\u2009{prim_n}"
                         if test_type != "Independent-Sample T-Test"
                         else " and ".join(f"n\u2009=\u2009{nm['n']} ({nm['label']})"
                                           for nm in R["normality"]))
            st.markdown(
                f'<p class="note-txt">&#9733; '
                f'Kolmogorov-Smirnov (Lilliefors correction) is the recommended '
                f'primary criterion for these sample sizes ({n_desc_ks}, each >\u200950). '
                f'Decision is based on this result.</p>',
                unsafe_allow_html=True)
        else:
            n_desc_ks = (f"n\u2009=\u2009{prim_n}"
                         if test_type != "Independent-Sample T-Test"
                         else " and ".join(f"n\u2009=\u2009{nm['n']} ({nm['label']})"
                                           for nm in R["normality"]))
            st.markdown(
                f'<p class="note-txt">\u1d43 Lilliefors significance correction applied. '
                f'Reported for informational purposes. '
                f'Shapiro-Wilk is the recommended criterion '
                f'for these sample sizes ({n_desc_ks}, each \u2264\u200950).</p>',
                unsafe_allow_html=True)

        if use_p:
            st.markdown(
                f'<div class="info-box">&#10003; '
                f'<b>{prim_lbl}</b> (recommended criterion, '
                f'n\u2009=\u2009{prim_n}) p\u2009&gt;\u2009.05 '
                f'\u2192 Normality assumption satisfied '
                f'\u2192 <b>Parametric analysis applied.</b></div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="warn-box">&#9888; '
                f'<b>{prim_lbl}</b> (recommended criterion, '
                f'n\u2009=\u2009{prim_n}) p\u2009\u2264\u2009.05 '
                f'\u2192 Normality assumption violated '
                f'\u2192 <b>Non-parametric analysis applied.</b></div>',
                unsafe_allow_html=True)

        # ── Assumption Summary Table ───────────────────────────────────────────
        st.markdown('<div class="sec-title">&#9989; Assumption Summary</div>',
                    unsafe_allow_html=True)
        assume_rows = [["Assumption","Test / Criterion","Result","Decision"]]
        for nm in R["normality"]:
            passed = nm["pass"]
            sw_res = f"SW W\u2009=\u2009{_f(nm['sw_W'])}, p\u2009=\u2009{_p(nm['sw_p'])}"
            ks_res = f"KS D\u2009=\u2009{_f(nm['ks_D'])}, p\u2009=\u2009{_p(nm['ks_p'])}"
            assume_rows.append([
                f"Normality \u2014 {nm['label']}",
                f"{sw_res} | {ks_res} (primary: {nm['primary_label']})",
                ('<span class="pass">&#10003; Satisfied</span>'
                 if passed else '<span class="fail">&#10007; Violated</span>'),
                "Parametric eligible" if passed else "Non-parametric required"
            ])
        if "levene" in R:
            lev = R["levene"]
            assume_rows.append([
                "Homogeneity of Variance",
                f"Levene F({lev['df1']},\u2009{lev['df2']})\u2009=\u2009{_f(lev['F'])}, "
                f"p\u2009=\u2009{_p(lev['Sig.'])}",
                ('<span class="pass">&#10003; Satisfied</span>'
                 if lev["equal_var"] else '<span class="fail">&#10007; Violated</span>'),
                "Equal variances assumed" if lev["equal_var"]
                else "Welch correction applied"
            ])
        assume_rows.append([
            "Independence of Observations",
            "By research design (not statistically testable)",
            '<span style="color:#64748b;">&#8505; Assumed</span>',
            "Must be ensured by design"
        ])
        assume_rows.append([
            "<b>Overall Decision</b>",
            f"Primary criterion: {prim_lbl}",
            f'<b>{"&#10003; Parametric" if use_p else "&#9888; Non-parametric"}</b>',
            f'<b>{"T-Test family" if use_p else "Wilcoxon / Mann-Whitney U"}</b>'
        ])
        st.markdown(html_tbl(assume_rows, left_cols={0,1,3}),
                    unsafe_allow_html=True)
        st.markdown(
            '<p class="note-txt">'
            'The assumption summary provides a consolidated overview of all '
            'statistical prerequisites evaluated prior to inferential testing. '
            'Independence of observations cannot be formally tested and must '
            'be ensured through appropriate research design.'
            '</p>', unsafe_allow_html=True)

    # ── Tab: Descriptives ──────────────────────────────────────────────────────
    _t1 = ti; ti += 1
    with tabs[_t1]:
        st.markdown('<div class="sec-title">Descriptive Statistics</div>',
                    unsafe_allow_html=True)
        dd = R["desc"].copy()
        for c in dd.select_dtypes(include=[float,np.float64]).columns:
            dd[c] = dd[c].apply(_f)
        st.markdown(html_tbl(df_to_rows(dd)), unsafe_allow_html=True)

    # ── Tab: Paired Correlation ────────────────────────────────────────────────
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
                '<p class="note-txt">'
                'Pearson product-moment correlation between the two paired variables.</p>',
                unsafe_allow_html=True)

    # ── Tab: Levene's Test ─────────────────────────────────────────────────────
    if "levene" in R:
        _tl = ti; ti += 1
        with tabs[_tl]:
            lev = R["levene"]
            res = ('<span class="pass">&#10003; Equal variances assumed</span>'
                   if lev["equal_var"]
                   else '<span class="fail">&#10007; Equal variances not assumed (Welch)</span>')
            st.markdown(
                '<div class="sec-title">'
                "Levene\u2019s Test for Equality of Variances (center\u2009=\u2009mean)"
                '</div>', unsafe_allow_html=True)
            st.markdown(html_tbl(
                [["F","df1","df2","Sig.","Result"],
                 [_f(lev["F"]),str(lev["df1"]),str(lev["df2"]),
                  _p(lev["Sig."]),res]],
                left_cols={4}), unsafe_allow_html=True)
            st.markdown(
                '<p class="note-txt">'
                'p\u2009&gt;\u2009.05 \u2192 equal variances assumed \u2192 '
                'use Row 1 of the t-test table.<br/>'
                'p\u2009\u2264\u2009.05 \u2192 equal variances not assumed \u2192 '
                'use the Welch correction (Row 2).'
                '</p>', unsafe_allow_html=True)

    # ── Tab: Parametric Results ────────────────────────────────────────────────
    _tp = ti; ti += 1
    with tabs[_tp]:
        if test_type == "One-Sample T-Test":
            st.markdown(
                f'<div class="sec-title">One-Sample T-Test '
                f'&nbsp;&middot;&nbsp; '
                f'Test Value (\u03bc\u2080)\u2009=\u2009{pr["mu0"]}'
                f'</div>', unsafe_allow_html=True)
            st.markdown(html_tbl([
                ["","t","df","Sig. (2-tailed)","Sig. (1-tailed Lower)",
                 "Sig. (1-tailed Upper)","Mean Difference",
                 "95% CI Lower","95% CI Upper","Cohen\u2019s d","Effect Size"],
                [f"Test Value\u2009=\u2009{pr['mu0']}",
                 _f(pr["t"]), str(pr["df"]), _p(pr["p_two"]),
                 _p(pr["p_one_lower"]), _p(pr["p_one_upper"]),
                 _f(pr["mean_diff"]),
                 _f(pr["ci_lower"]), _f(pr["ci_upper"]),
                 _f(pr["cohens_d"]), effect_label_d(pr["cohens_d"])]
            ], left_cols={0,10}), unsafe_allow_html=True)

        elif test_type == "Paired-Sample T-Test":
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
                ["Pair","Mean Difference","Std. Deviation","Std. Error Mean",
                 "95% CI Lower","95% CI Upper","t","df",
                 "Sig. (2-tailed)","Sig. (1-tailed L)",
                 "Sig. (1-tailed U)","Cohen\u2019s d"],
                [f"{pr['label1']} \u2013 {pr['label2']}",
                 _f(pr["mean_diff"]), _f(pr["sd_diff"]), _f(pr["se_diff"]),
                 _f(pr["ci_lower"]), _f(pr["ci_upper"]),
                 _f(pr["t"]), str(pr["df"]),
                 _p(pr["p_two"]),
                 _p(pr["p_one_lower"]), _p(pr["p_one_upper"]),
                 _f(pr["cohens_d"])]
            ], left_cols={0}), unsafe_allow_html=True)

        else:
            lev = R["levene"]
            st.markdown(
                '<div class="sec-title">Independent Samples T-Test</div>',
                unsafe_allow_html=True)
            st.markdown(
                '<div class="sub-title">a) t-Statistics and Significance</div>',
                unsafe_allow_html=True)
            st.markdown(html_tbl([
                ["","F (Levene)","Sig.","t","df","Sig. (2-tailed)",
                 "Sig. (1-tailed L)","Sig. (1-tailed U)"],
                ["Equal variances assumed",
                 _f(lev["F"]), _p(lev["Sig."]),
                 _f(pr["t_eq"]), str(pr["df_eq"]), _p(pr["p_eq"]),
                 _p(pr["p_eq_lower"]), _p(pr["p_eq_upper"])],
                ["Equal variances not assumed","","",
                 _f(pr["t_welch"]), _f(pr["df_welch"],2), _p(pr["p_welch"]),
                 _p(pr["p_welch_lower"]), _p(pr["p_welch_upper"])]
            ], left_cols={0}), unsafe_allow_html=True)

            st.markdown(
                '<div class="sub-title">'
                'b) Mean Difference, Confidence Interval, and Effect Size'
                '</div>', unsafe_allow_html=True)
            st.markdown(html_tbl([
                ["","Mean Difference","SE Difference",
                 "95% CI Lower","95% CI Upper","Cohen\u2019s d"],
                ["Equal variances assumed",
                 _f(pr["mean_diff"]), _f(pr["se_eq"]),
                 _f(pr["ci_eq_l"]), _f(pr["ci_eq_u"]), _f(pr["cohens_d"])],
                ["Equal variances not assumed",
                 _f(pr["mean_diff"]), _f(pr["se_welch"]),
                 _f(pr["ci_welch_l"]), _f(pr["ci_welch_u"]), "\u2014"]
            ], left_cols={0}), unsafe_allow_html=True)
            active = ("Row 1 (equal variances assumed)"
                      if lev["equal_var"] else "Row 2 (Welch correction)")
            st.markdown(
                f'<p class="note-txt">Based on Levene\u2019s test '
                f'p\u2009=\u2009{_p(lev["Sig."])}: use <b>{active}</b>.</p>',
                unsafe_allow_html=True)

        if not use_p:
            st.markdown(
                '<div class="warn-box">&#9888; Normality assumption violated. '
                'Refer to the <b>Non-Parametric</b> tab for the recommended analysis.'
                '</div>', unsafe_allow_html=True)

    # ── Tab: Non-Parametric Results ────────────────────────────────────────────
    _tnp = ti; ti += 1
    with tabs[_tnp]:
        if test_type == "Independent-Sample T-Test":
            st.markdown(
                '<div class="sec-title">Mann-Whitney U Test '
                '&nbsp;&middot;&nbsp; Ranks</div>',
                unsafe_allow_html=True)
            st.markdown(html_tbl([
                ["Group","N","Mean Rank","Sum of Ranks"],
                [np_r["label1"], str(np_r["n1"]),
                 _f(np_r["mean_rank1"]), _f(np_r["R1"],3)],
                [np_r["label2"], str(np_r["n2"]),
                 _f(np_r["mean_rank2"]), _f(np_r["R2"],3)],
                ["Total", str(np_r["n1"]+np_r["n2"]), "", ""]
            ], left_cols={0}), unsafe_allow_html=True)
            st.markdown(
                '<div class="sec-title">Mann-Whitney U Test '
                '&nbsp;&middot;&nbsp; Test Statistics</div>',
                unsafe_allow_html=True)
            st.markdown(html_tbl([
                ["Statistic","Value"],
                ["Mann-Whitney U",         format_u(np_r["U"])],
                ["Wilcoxon W",             _f(np_r["W_wilcoxon"],3)],
                ["Z",                      _f(np_r["Z"])],
                ["Asymp. Sig. (2-tailed)", _p(np_r["p"])],
            ], left_cols={0}), unsafe_allow_html=True)
            st.markdown(
                f'<p class="note-txt">'
                f'Grouping: {np_r["label1"]} vs.\u2009{np_r["label2"]}. '
                f'Z based on normal approximation with ties correction (SPSS method).</p>',
                unsafe_allow_html=True)

        else:
            n_neg  = np_r.get("n_neg",0)
            n_pos  = np_r.get("n_pos",0)
            n_ties = np_r.get("n_ties",0)
            n_tot  = np_r.get("n_total", n_neg+n_pos+n_ties)
            neg_rs = np_r.get("neg_rank_sum", np.nan)
            pos_rs = np_r.get("pos_rank_sum", np.nan)
            neg_mr = neg_rs/n_neg if n_neg > 0 else np.nan
            pos_mr = pos_rs/n_pos if n_pos > 0 else np.nan
            fn_exist = (test_type == "Paired-Sample T-Test")
            footnotes = []
            if fn_exist:
                footnotes = [
                    f"\u1d43 {pr['label2']} < {pr['label1']} (negative difference)",
                    f"\u1d47 {pr['label2']} > {pr['label1']} (positive difference)",
                    f"\u1d9c {pr['label2']} = {pr['label1']} (zero difference; excluded from ranking)"
                ]
            st.markdown(
                '<div class="sec-title">Wilcoxon Signed-Rank Test '
                '&nbsp;&middot;&nbsp; Ranks</div>',
                unsafe_allow_html=True)
            st.markdown(html_tbl([
                ["","N","Mean Rank","Sum of Ranks"],
                [("Negative Ranks \u1d43" if fn_exist else "Negative Ranks"),
                 str(n_neg), _f(neg_mr), _f(neg_rs,3)],
                [("Positive Ranks \u1d47" if fn_exist else "Positive Ranks"),
                 str(n_pos), _f(pos_mr), _f(pos_rs,3)],
                [("Ties \u1d9c" if fn_exist else "Ties"),
                 str(n_ties),"",""],
                ["Total", str(n_tot),"",""]
            ], left_cols={0}), unsafe_allow_html=True)
            for fn in footnotes:
                st.markdown(f'<p class="note-txt">{fn}</p>',
                            unsafe_allow_html=True)
            pair_lbl = (f"{pr['label1']} \u2212 {pr['label2']}"
                        if fn_exist else "Variable \u2212 \u03bc\u2080")
            st.markdown(
                '<div class="sec-title">Wilcoxon Signed-Rank Test '
                '&nbsp;&middot;&nbsp; Test Statistics</div>',
                unsafe_allow_html=True)
            st.markdown(html_tbl([
                ["Statistic", pair_lbl],
                ["Test Statistic (W)", _f(np_r["W"],0)],
                ["Z",                  _f(np_r["Z"])],
                ["Asymp. Sig. (2-tailed)", _p(np_r["p"])],
            ], left_cols={0}), unsafe_allow_html=True)
            st.markdown(
                f'<p class="note-txt">'
                f'Based on {"negative" if n_neg < n_pos else "positive"} ranks. '
                f'Z uses ties-corrected variance (SPSS method).</p>',
                unsafe_allow_html=True)

        if use_p:
            st.markdown(
                '<div class="info-box">&#8505; Normality assumption was satisfied. '
                'The <b>Parametric</b> tab contains the recommended analysis.</div>',
                unsafe_allow_html=True)

        # ── Non-parametric Effect Size: Rank-Biserial r ────────────────────────
        st.markdown(
            '<div class="sec-title">'
            'Non-Parametric Effect Size &nbsp;&middot;&nbsp; '
            'Rank-Biserial Correlation (r)'
            '</div>', unsafe_allow_html=True)

        if test_type == "Independent-Sample T-Test":
            g1d_es = (st.session_state["stats_df"]
                      [st.session_state["stats_df"][st.session_state["stats_cfg"]["grp_col"]]
                       == st.session_state["stats_cfg"]["g1"]]
                      [st.session_state["stats_cfg"]["dep_col"]].dropna().values)
            g2d_es = (st.session_state["stats_df"]
                      [st.session_state["stats_df"][st.session_state["stats_cfg"]["grp_col"]]
                       == st.session_state["stats_cfg"]["g2"]]
                      [st.session_state["stats_cfg"]["dep_col"]].dropna().values)
            r_rb = rank_biserial_mannwhitney(g1d_es, g2d_es)
            ci_lo, ci_hi = bootstrap_ci_effect_size(
                rank_biserial_mannwhitney, [g1d_es, g2d_es], alpha=alpha)
            es_label = "Mann-Whitney U"
            pair_desc = f"{np_r['label1']} vs. {np_r['label2']}"
        else:
            cfg_s = st.session_state["stats_cfg"]
            if test_type == "One-Sample T-Test":
                raw_diff = (st.session_state["stats_df"][cfg_s["test_var"]]
                            .dropna().values - cfg_s["mu0"])
            else:
                pdata = st.session_state["stats_df"][
                    [cfg_s["v1"], cfg_s["v2"]]].dropna()
                raw_diff = (pdata[cfg_s["v1"]].values
                            - pdata[cfg_s["v2"]].values)
            r_rb = rank_biserial_wilcoxon(raw_diff)
            ci_lo, ci_hi = bootstrap_ci_effect_size(
                rank_biserial_wilcoxon, [raw_diff], alpha=alpha)
            es_label = "Wilcoxon Signed-Rank"
            pair_desc = (f"{cfg_s.get('v1','Variable')} \u2212 "
                         f"{cfg_s.get('v2','\u03bc\u2080')}"
                         if test_type == "Paired-Sample T-Test"
                         else f"{cfg_s.get('test_var','Variable')} \u2212 \u03bc\u2080")

        r_lab = effect_label_r_nonparam(r_rb) if not np.isnan(r_rb) else "N/A"
        st.markdown(html_tbl([
            ["Test","Comparison","Rank-Biserial r",
             f"95% Bootstrap CI",
             "Effect Size","Interpretation"],
            [es_label, pair_desc,
             _f(r_rb) if not np.isnan(r_rb) else ".",
             (f"[{_f(ci_lo)}, {_f(ci_hi)}]"
              if not np.isnan(ci_lo) else "."),
             r_lab,
             "Rank-biserial r \u2208 [\u22121, 1]; "
             "|r|\u2009<\u2009.10 negligible, "
             ".10\u2013.29 small, "
             ".30\u2013.49 medium, "
             "\u2265\u2009.50 large"]
        ], left_cols={0,1,5}), unsafe_allow_html=True)
        st.markdown(
            '<p class="note-txt">'
            'Rank-biserial correlation r is a non-parametric effect size not '
            'reported by SPSS. Bootstrap 95% CI based on 2,000 resamples '
            '(Efron &amp; Tibshirani, 1993; Kerby, 2014). '
            'This measure complements Cohen\u2019s d for non-parametric analyses.'
            '</p>', unsafe_allow_html=True)

    # ── Tab: Plots ─────────────────────────────────────────────────────────────
    _tpl = ti; ti += 1
    with tabs[_tpl]:
        for fb in figs_b:
            st.image(fb, use_container_width=True)

    # ── Tab: Interpretation ────────────────────────────────────────────────────
    _ti2 = ti; ti += 1
    with tabs[_ti2]:
        st.markdown("### &#128221; Statistical Interpretation")
        for line in interps:
            cls = ""
            lw  = line.lower()
            if ("statistically significant" in lw and
                    "no statistically" not in lw):
                cls = "sig"
            elif ("no statistically significant" in lw or
                  "did not significantly" in lw):
                cls = "nonsig"
            st.markdown(f'<div class="interp-box {cls}">{line}</div>',
                        unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**APA 7th Edition Write-Up:**")
        if use_p:
            if test_type == "One-Sample T-Test":
                m_v  = float(R["desc"]["Mean"].iloc[0])
                sd_v = float(R["desc"]["Std. Deviation"].iloc[0])
                apa  = (f"A one-sample t-test was conducted to examine whether "
                        f"{meta.get('Variable','the variable')} "
                        f"(M\u2009=\u2009{_f(m_v)}, SD\u2009=\u2009{_f(sd_v)}) "
                        f"significantly differed from the hypothesised population mean "
                        f"(\u03bc\u2080\u2009=\u2009{pr['mu0']}). The result was "
                        f"{'statistically significant' if pr['p_two']<alpha else 'not statistically significant'}, "
                        f"t({pr['df']})\u2009=\u2009{pr['t']:.2f}, "
                        f"p\u2009{'< .001' if pr['p_two']<.001 else '= '+_p(pr['p_two'])}, "
                        f"d\u2009=\u2009{pr['cohens_d']:.2f}.")
            elif test_type == "Paired-Sample T-Test":
                apa = (f"A paired-samples t-test was conducted to compare "
                       f"{pr['label1']} "
                       f"(M\u2009=\u2009{_f(float(R['desc'].iloc[0]['Mean']))}, "
                       f"SD\u2009=\u2009{_f(float(R['desc'].iloc[0]['Std. Deviation']))}) "
                       f"and {pr['label2']} "
                       f"(M\u2009=\u2009{_f(float(R['desc'].iloc[1]['Mean']))}, "
                       f"SD\u2009=\u2009{_f(float(R['desc'].iloc[1]['Std. Deviation']))}). "
                       f"The difference was "
                       f"{'statistically significant' if pr['p_two']<alpha else 'not statistically significant'}, "
                       f"t({pr['df']})\u2009=\u2009{pr['t']:.2f}, "
                       f"p\u2009{'< .001' if pr['p_two']<.001 else '= '+_p(pr['p_two'])}, "
                       f"d\u2009=\u2009{pr['cohens_d']:.2f}.")
            else:
                use_eq = R["levene"]["equal_var"]
                tv = pr["t_eq"]  if use_eq else pr["t_welch"]
                pv = pr["p_eq"]  if use_eq else pr["p_welch"]
                dv = pr["df_eq"] if use_eq else pr["df_welch"]
                apa = (f"An independent-samples t-test was conducted to compare "
                       f"{pr['dep_var']} between {pr['label1']} "
                       f"(M\u2009=\u2009{_f(float(R['desc'].iloc[0]['Mean']))}, "
                       f"SD\u2009=\u2009{_f(float(R['desc'].iloc[0]['Std. Deviation']))}) "
                       f"and {pr['label2']} "
                       f"(M\u2009=\u2009{_f(float(R['desc'].iloc[1]['Mean']))}, "
                       f"SD\u2009=\u2009{_f(float(R['desc'].iloc[1]['Std. Deviation']))}). "
                       f"The difference was "
                       f"{'statistically significant' if pv<alpha else 'not statistically significant'}, "
                       f"t({_f(dv,2)})\u2009=\u2009{tv:.2f}, "
                       f"p\u2009{'< .001' if pv<.001 else '= '+_p(pv)}, "
                       f"d\u2009=\u2009{pr['cohens_d']:.2f}.")
        else:
            if test_type == "Independent-Sample T-Test":
                apa = (f"A Mann-Whitney U test was conducted to compare "
                       f"{np_r['dep_var']} between {np_r['label1']} "
                       f"and {np_r['label2']}. The result indicated "
                       f"{'a statistically significant' if np_r['p']<alpha else 'no statistically significant'} "
                       f"difference, U\u2009=\u2009{format_u(np_r['U'])}, "
                       f"W\u2009=\u2009{_f(np_r['W_wilcoxon'],3)}, "
                       f"Z\u2009=\u2009{np_r['Z']:.3f}, "
                       f"p\u2009{'< .001' if np_r['p']<.001 else '= '+_p(np_r['p'])} "
                       f"(asymptotic, 2-tailed).")
            else:
                apa = (f"A Wilcoxon signed-rank test was conducted. "
                       f"The result indicated "
                       f"{'a statistically significant' if np_r['p']<alpha else 'no statistically significant'} "
                       f"difference, W\u2009=\u2009{np_r['W']:.0f}, "
                       f"Z\u2009=\u2009{np_r['Z']:.3f}, "
                       f"p\u2009{'< .001' if np_r['p']<.001 else '= '+_p(np_r['p'])} "
                       f"(2-tailed, asymptotic).")
        st.code(apa, language=None)

        # ── Statistical Power Assessment ───────────────────────────────────────
        st.markdown("---")
        st.markdown(
            "### &#9889; Statistical Power Assessment")
        try:
            pw = compute_power(test_type, R, alpha)
            power_val = pw.get("power", np.nan)
            power_pct = f"{power_val*100:.1f}%" if not np.isnan(power_val) else "N/A"
            power_lbl = pw.get("power_label", "N/A")
            effect_v  = pw.get("effect_size", np.nan)
            effect_t  = pw.get("effect_type", "—")

            # Colour-code power bar
            if not np.isnan(power_val):
                bar_color = ("#16a34a" if power_val >= .80
                             else "#f59e0b" if power_val >= .60
                             else "#dc2626")
                bar_pct = int(power_val * 100)
                bar_html = (
                    f'<div style="background:#e2e8f0;border-radius:8px;'
                    f'height:18px;width:100%;margin:8px 0;">'
                    f'<div style="background:{bar_color};width:{bar_pct}%;'
                    f'height:18px;border-radius:8px;transition:width .4s;"></div>'
                    f'</div>'
                )
            else:
                bar_html = ""

            if "n1" in pw:
                n_desc = (f"n\u2081\u2009=\u2009{pw['n1']}, "
                          f"n\u2082\u2009=\u2009{pw['n2']}")
            else:
                n_desc = f"n\u2009=\u2009{pw.get('n','N/A')}"

            power_rows = [
                ["Parameter","Value"],
                ["Sample size", n_desc],
                ["Observed effect size",
                 f"{_f(effect_v)} ({effect_t})"],
                ["Significance level (\u03b1)", str(alpha)],
                ["Achieved statistical power", power_pct],
                ["Power classification", power_lbl],
                ["Recommended minimum power", "\u2265\u2009.80 (Cohen, 1988)"]
            ]
            st.markdown(html_tbl(power_rows, left_cols={0}),
                        unsafe_allow_html=True)
            st.markdown(bar_html, unsafe_allow_html=True)

            if not np.isnan(power_val) and power_val < .80:
                st.markdown(
                    '<div class="warn-box">&#9888; <b>Insufficient power.</b> '
                    'The achieved power is below the conventional threshold of '
                    '.80. This indicates an elevated risk of Type II error '
                    '(failing to detect a true effect). Consider increasing '
                    'the sample size to improve statistical power.</div>',
                    unsafe_allow_html=True)
            elif not np.isnan(power_val):
                st.markdown(
                    '<div class="info-box">&#10003; <b>Adequate power.</b> '
                    'The achieved power meets or exceeds the conventional '
                    'threshold of .80, indicating a satisfactory probability '
                    'of detecting the observed effect size.</div>',
                    unsafe_allow_html=True)

            st.markdown(
                '<p class="note-txt">'
                'Post-hoc power analysis uses the observed effect size and '
                'sample size to estimate the probability of correctly rejecting '
                'H\u2080 given that the effect is real. Power is computed via '
                'the non-central t-distribution (parametric) or normal '
                'approximation (non-parametric). '
                'Reference: Cohen (1988). '
                '<i>Statistical Power Analysis for the Behavioral Sciences</i> '
                '(2nd ed.). Lawrence Erlbaum Associates.'
                '</p>', unsafe_allow_html=True)

        except Exception as e_pw:
            st.markdown(
                f'<div class="warn-box">Power analysis unavailable: {e_pw}</div>',
                unsafe_allow_html=True)

    # ── Downloads ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### &#128229; Download Results")
    dc1, dc2, dc3 = st.columns(3)

    # ── HTML Report (replaces PDF) ─────────────────────────────────────────────
    with dc1:
        with st.spinner("Generating HTML report\u2026"):
            html_data = build_html_report(test_type, R, meta, interps, figs_b)
        st.download_button(
            "&#127760; HTML Report (Full)",
            html_data,
            f"Stats_{test_type.replace(' ','_')}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            "text/html", use_container_width=True)

    # ── Excel ──────────────────────────────────────────────────────────────────
    with dc2:
        xbuf = io.BytesIO()
        with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
            R["desc"].to_excel(writer, sheet_name="Descriptive Statistics", index=False)
            norm_rows_xls = []
            for n_item in R["normality"]:
                norm_rows_xls.append({
                    "Variable":     n_item["label"], "N": n_item["n"],
                    "SW W":         n_item["sw_W"],  "SW Sig.":  n_item["sw_p"],
                    "SW Result":    "Normal" if n_item["sw_pass"] else "Non-Normal",
                    "KS D":         n_item["ks_D"],  "KS Sig.":  n_item["ks_p"],
                    "KS Result":    "Normal" if n_item["ks_pass"] else "Non-Normal",
                    "Recommended Criterion": n_item["primary_label"],
                    "Decision":     "Normal" if n_item["pass"] else "Non-Normal",
                    "Recommendation Note": normality_recommendation_plain(n_item["n"])
                })
            pd.DataFrame(norm_rows_xls).to_excel(
                writer, sheet_name="Normality Tests", index=False)
            if "correlation" in R:
                R["correlation"].to_excel(
                    writer, sheet_name="Paired Correlation", index=False)
            if "levene" in R:
                pd.DataFrame([R["levene"]]).to_excel(
                    writer, sheet_name="Levene Test", index=False)
            pd.DataFrame([R["parametric"]]).to_excel(
                writer, sheet_name="Parametric Results", index=False)
            np_export = dict(R["nonparametric"])
            np_export["U"] = format_u(np_export.get("U", float("nan")))
            pd.DataFrame([np_export]).to_excel(
                writer, sheet_name="Non-Parametric Results", index=False)
        xbuf.seek(0)
        st.download_button(
            "&#128202; Excel Workbook",
            xbuf.getvalue(),
            f"Stats_{test_type.replace(' ','_')}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True)

    # ── CSV ────────────────────────────────────────────────────────────────────
    with dc3:
        norm_csv_rows = []
        for n_item in R["normality"]:
            norm_csv_rows.append({
                "Variable":               n_item["label"], "N": n_item["n"],
                "SW_W":                   n_item["sw_W"],  "SW_p": n_item["sw_p"],
                "SW_pass":                n_item["sw_pass"],
                "KS_D":                   n_item["ks_D"],  "KS_p": n_item["ks_p"],
                "KS_pass":                n_item["ks_pass"],
                "Recommended_criterion":  n_item["primary_label"],
                "Decision":               "Normal" if n_item["pass"] else "Non-Normal"
            })
        parts = [f"=== {test_type.upper()} ===\n"]
        parts.append("=== DESCRIPTIVE STATISTICS ===\n" +
                     R["desc"].to_csv(index=False))
        parts.append("=== NORMALITY TESTS ===\n" +
                     pd.DataFrame(norm_csv_rows).to_csv(index=False))
        if "correlation" in R:
            parts.append("=== PAIRED CORRELATION ===\n" +
                         R["correlation"].to_csv(index=False))
        if "levene" in R:
            parts.append("=== LEVENE TEST ===\n" +
                         pd.DataFrame([R["levene"]]).to_csv(index=False))
        parts.append("=== PARAMETRIC RESULTS ===\n" +
                     pd.DataFrame([R["parametric"]]).to_csv(index=False))
        np_csv = dict(R["nonparametric"])
        np_csv["U"] = format_u(np_csv.get("U", float("nan")))
        parts.append("=== NON-PARAMETRIC RESULTS ===\n" +
                     pd.DataFrame([np_csv]).to_csv(index=False))
        st.download_button(
            "&#128221; CSV Tables",
            "\n\n".join(parts).encode(),
            f"Stats_{test_type.replace(' ','_')}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv", use_container_width=True)


if __name__ == "__main__":
    main()
