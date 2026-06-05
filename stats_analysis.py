"""
Statistical Analysis Module — NeuroQA Copilot
==============================================
Comprehensive statistical toolkit for radiotherapy QA data analysis.
Showcases: descriptive statistics, hypothesis testing, effect sizes,
bootstrap confidence intervals, distribution fitting, outlier detection,
correlation analysis, and group comparisons.

All functions are designed for a data analyst portfolio — clinically
relevant, well-documented, and statistically rigorous.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.stats import (
    shapiro, kruskal, mannwhitneyu, spearmanr, pearsonr,
    f_oneway, norm, lognorm, gamma, kstest, iqr,
)
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════════════
# 1. DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════

def compute_descriptive_stats(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    group_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute comprehensive descriptive statistics for numeric columns.

    Returns: mean, median, std, IQR, skewness, kurtosis, min, max,
             range, coefficient of variation, SEM (standard error of mean).

    If group_by is provided (e.g., 'Risk_Level'), computes per-group stats.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    def _stats_for_series(s: pd.Series, label: str = "") -> dict:
        clean = s.dropna()
        n = len(clean)
        if n < 3:
            return {}
        mean_val = clean.mean()
        std_val = clean.std(ddof=1)
        return {
            f"{label}N": n,
            f"{label}Mean": round(mean_val, 3),
            f"{label}Median": round(clean.median(), 3),
            f"{label}Std Dev": round(std_val, 3),
            f"{label}IQR": round(iqr(clean), 3),
            f"{label}Skewness": round(clean.skew(), 3),
            f"{label}Kurtosis": round(clean.kurtosis(), 3),
            f"{label}Min": round(clean.min(), 3),
            f"{label}Max": round(clean.max(), 3),
            f"{label}Range": round(clean.max() - clean.min(), 3),
            f"{label}CV (%)": round((std_val / mean_val * 100) if mean_val != 0 else np.nan, 2),
            f"{label}SEM": round(std_val / np.sqrt(n), 4),
        }

    if group_by and group_by in df.columns:
        rows = []
        for grp_name, grp_df in df.groupby(group_by):
            row = {"Group": grp_name}
            for col in columns:
                if col in grp_df.columns and col != group_by:
                    stats = _stats_for_series(grp_df[col])
                    row.update(stats)
            rows.append(row)
        return pd.DataFrame(rows)

    rows = []
    for col in columns:
        row = {"Variable": col}
        row.update(_stats_for_series(df[col]))
        rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 2. NORMALITY TESTING
# ══════════════════════════════════════════════════════════════════════

def test_normality(
    data: pd.Series, alpha: float = 0.05
) -> Dict[str, object]:
    """
    Shapiro-Wilk test for normality.

    Returns: W statistic, p-value, and interpretation.
    In radiotherapy, tumor volumes are typically log-normal — this test
    confirms or rejects that assumption.
    """
    clean = data.dropna()
    if len(clean) < 3:
        return {"error": "Insufficient data (n < 3)"}
    if len(clean) > 5000:
        # Shapiro-Wilk maxes out at 5000; fall back to D'Agostino-Pearson
        stat, p = sp_stats.normaltest(clean)
        test_name = "D'Agostino-Pearson"
    else:
        stat, p = shapiro(clean)
        test_name = "Shapiro-Wilk"

    return {
        "test": test_name,
        "statistic": round(stat, 4),
        "p_value": round(p, 6),
        "is_normal": p > alpha,
        "interpretation": (
            f"Data appears {'normally' if p > alpha else 'non-normally'} "
            f"distributed (p={p:.4f}). "
            f"{'Parametric tests appropriate.' if p > alpha else 'Consider non-parametric tests or log-transform.'}"
        ),
    }


def test_lognormality(
    data: pd.Series, alpha: float = 0.05
) -> Dict[str, object]:
    """
    Test if log-transformed data follows a normal distribution.

    Tumor volumes in oncology frequently follow a log-normal distribution.
    This is clinically relevant because it affects how we set action limits.
    """
    clean = data.dropna()
    positive = clean[clean > 0]
    if len(positive) < 3:
        return {"error": "Insufficient positive values (n < 3)"}
    log_data = np.log(positive)
    return test_normality(pd.Series(log_data), alpha)


# ══════════════════════════════════════════════════════════════════════
# 3. CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "spearman",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute pairwise correlation matrix with p-values.

    Uses Spearman (rank) correlation by default — appropriate for
    clinical data that may be non-normal or have outliers.

    Returns:
        r_matrix: correlation coefficients
        p_matrix: associated p-values (Bonferroni-corrected within matrix)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    corr_func = spearmanr if method == "spearman" else pearsonr
    n = len(columns)
    r_mat = np.eye(n)
    p_mat = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            clean = df[[columns[i], columns[j]]].dropna()
            if len(clean) < 3:
                r_mat[i, j] = r_mat[j, i] = np.nan
                p_mat[i, j] = p_mat[j, i] = np.nan
                continue
            r, p = corr_func(clean[columns[i]], clean[columns[j]])
            r_mat[i, j] = r_mat[j, i] = round(r, 4)
            p_mat[i, j] = p_mat[j, i] = round(p, 6)

    # Bonferroni correction for multiple comparisons
    n_comparisons = n * (n - 1) / 2
    p_corrected = np.minimum(p_mat * n_comparisons, 1.0)

    r_df = pd.DataFrame(r_mat, index=columns, columns=columns)
    p_df = pd.DataFrame(p_corrected, index=columns, columns=columns)

    return r_df, p_df


def correlation_significance_labels(
    p_matrix: pd.DataFrame, alpha: float = 0.05
) -> pd.DataFrame:
    """Convert p-value matrix to significance labels for heatmap annotation."""
    labels = p_matrix.copy()
    for col in labels.columns:
        labels[col] = labels[col].apply(
            lambda p: "***" if p < 0.001 else "**" if p < 0.01
            else "*" if p < 0.05 else "n.s."
        )
    return labels


# ══════════════════════════════════════════════════════════════════════
# 4. GROUP COMPARISONS (Non-parametric, clinically appropriate)
# ══════════════════════════════════════════════════════════════════════

def compare_groups_kruskal(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str = "Risk_Level",
) -> Dict[str, object]:
    """
    Kruskal-Wallis H-test: non-parametric one-way ANOVA.

    Compares a continuous metric across 3+ independent groups (HIGH,
    MODERATE, LOW risk). Appropriate when normality assumptions are
    violated — common in clinical tumor data.
    """
    groups = [
        grp[metric_col].dropna().values
        for _, grp in df.groupby(group_col)
        if len(grp) >= 3
    ]
    if len(groups) < 2:
        return {"error": "Need ≥ 2 groups with ≥ 3 observations each"}

    h_stat, p_val = kruskal(*groups)

    # Effect size: eta-squared (η²) from H-statistic
    n_total = sum(len(g) for g in groups)
    eta_sq = (h_stat - len(groups) + 1) / (n_total - len(groups))
    eta_sq = max(0, min(eta_sq, 1))

    # Pairwise Mann-Whitney U with Bonferroni correction
    pairwise = []
    group_names = sorted(df[group_col].dropna().unique())
    n_comparisons = len(group_names) * (len(group_names) - 1) / 2

    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            g1 = df[df[group_col] == group_names[i]][metric_col].dropna()
            g2 = df[df[group_col] == group_names[j]][metric_col].dropna()
            if len(g1) >= 3 and len(g2) >= 3:
                u_stat, u_p = mannwhitneyu(g1, g2, alternative="two-sided")
                # Cliff's delta effect size
                cliff_d = _cliffs_delta(g1.values, g2.values)
                pairwise.append({
                    "comparison": f"{group_names[i]} vs {group_names[j]}",
                    "U": round(u_stat, 2),
                    "p_raw": round(u_p, 6),
                    "p_corrected": round(min(u_p * n_comparisons, 1.0), 6),
                    "cliffs_delta": round(cliff_d, 3),
                    "effect_magnitude": _cliff_interpretation(cliff_d),
                })

    return {
        "metric": metric_col,
        "test": "Kruskal-Wallis H",
        "H_statistic": round(h_stat, 4),
        "p_value": round(p_val, 6),
        "significant": p_val < 0.05,
        "eta_squared": round(eta_sq, 4),
        "eta_interpretation": (
            "Large effect" if eta_sq > 0.14
            else "Medium effect" if eta_sq > 0.06
            else "Small effect" if eta_sq > 0.01
            else "Negligible effect"
        ),
        "pairwise_comparisons": pairwise,
    }


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta: robust, non-parametric effect size for two groups.

    Range: [-1, 1]. 0 = complete overlap, ±1 = complete separation.
    More robust than Cohen's d for non-normal clinical data.
    """
    nx, ny = len(x), len(y)
    dominance = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                dominance += 1
            elif xi < yj:
                dominance -= 1
    return dominance / (nx * ny)


def _cliff_interpretation(d: float) -> str:
    ad = abs(d)
    if ad < 0.147:
        return "Negligible"
    if ad < 0.33:
        return "Small"
    if ad < 0.474:
        return "Medium"
    return "Large"


# ══════════════════════════════════════════════════════════════════════
# 5. BOOTSTRAP CONFIDENCE INTERVALS
# ══════════════════════════════════════════════════════════════════════

def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic: str = "mean",
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> Dict[str, object]:
    """
    Bootstrap confidence intervals for arbitrary statistics.

    Non-parametric resampling — no distributional assumptions.
    Clinically useful for: "We are 95% confident the mean tumor volume
    for HIGH-risk patients is between X and Y cc."

    Supported statistics: mean, median, std, iqr, cv (coefficient of variation)
    """
    rng = np.random.RandomState(random_seed)
    clean = data[~np.isnan(data)]
    n = len(clean)

    if n < 10:
        return {"error": "Need ≥ 10 observations for bootstrap"}

    stat_funcs = {
        "mean": np.mean,
        "median": np.median,
        "std": lambda x: np.std(x, ddof=1),
        "iqr": lambda x: iqr(x),
        "cv": lambda x: np.std(x, ddof=1) / np.mean(x) * 100 if np.mean(x) != 0 else np.nan,
    }

    func = stat_funcs.get(statistic, np.mean)
    observed = func(clean)

    boot_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(clean, size=n, replace=True)
        boot_stats[i] = func(sample)

    alpha = 1 - ci_level
    lower_p = alpha / 2 * 100
    upper_p = (1 - alpha / 2) * 100
    lower = np.percentile(boot_stats, lower_p)
    upper = np.percentile(boot_stats, upper_p)

    return {
        "statistic": statistic,
        "observed": round(observed, 4),
        "ci_level": ci_level,
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "bootstrap_std": round(np.std(boot_stats, ddof=1), 4),
        "n_bootstrap": n_bootstrap,
        "n_observations": n,
        "interpretation": (
            f"We are {ci_level*100:.0f}% confident that the true {statistic} "
            f"lies between {lower:.3f} and {upper:.3f}."
        ),
    }


def bootstrap_ci_by_group(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str = "Risk_Level",
    statistic: str = "mean",
    n_bootstrap: int = 10_000,
) -> pd.DataFrame:
    """Bootstrap CIs for a metric, stratified by group."""
    rows = []
    for grp_name in sorted(df[group_col].dropna().unique()):
        data = df[df[group_col] == grp_name][metric_col].dropna().values
        if len(data) < 10:
            continue
        result = bootstrap_confidence_interval(
            data, statistic=statistic, n_bootstrap=n_bootstrap
        )
        result["group"] = grp_name
        rows.append(result)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 6. OUTLIER DETECTION
# ══════════════════════════════════════════════════════════════════════

def detect_outliers(
    data: pd.Series,
    method: str = "iqr",
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Detect outliers using either IQR or Z-score method.

    IQR method: x < Q1 - 1.5*IQR  or  x > Q3 + 1.5*IQR  (Tukey's fences)
    Z-score method: |z| > multiplier (default 3)

    Returns DataFrame of outliers with deviation metrics.
    """
    clean = data.dropna()
    name = data.name or "value"

    if method == "iqr":
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr_val = q3 - q1
        lower = q1 - multiplier * iqr_val
        upper = q3 + multiplier * iqr_val
        mask = (clean < lower) | (clean > upper)
        deviation = np.where(
            clean < lower, clean - lower, np.where(clean > upper, clean - upper, 0)
        )
    else:  # z-score
        z_scores = np.abs((clean - clean.mean()) / clean.std(ddof=1))
        mask = z_scores > multiplier
        lower, upper = np.nan, np.nan
        deviation = z_scores

    if not mask.any():
        return pd.DataFrame(columns=["index", name, "deviation", "direction"])

    outliers = clean[mask]
    dev = deviation[mask]
    direction = ["low" if clean.loc[idx] < (lower if method == "iqr" else clean.mean())
                 else "high" for idx in outliers.index]

    result = pd.DataFrame({
        "index": outliers.index,
        name: outliers.values,
        "deviation": list(dev) if isinstance(dev, np.ndarray) else dev.values,
        "direction": direction,
    })
    return result.sort_values("deviation", ascending=False, key=abs)


def outlier_summary(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Summarize outlier counts across multiple columns."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    rows = []
    for col in columns:
        outliers = detect_outliers(df[col], method="iqr")
        rows.append({
            "Variable": col,
            "N": len(df[col].dropna()),
            "Outliers (IQR)": len(outliers),
            "% Outliers": round(len(outliers) / len(df[col].dropna()) * 100, 1),
            "Low": (outliers["direction"] == "low").sum() if len(outliers) > 0 else 0,
            "High": (outliers["direction"] == "high").sum() if len(outliers) > 0 else 0,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# 7. EFFECT SIZE — Cohen's d
# ══════════════════════════════════════════════════════════════════════

def cohens_d(x: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """
    Cohen's d: standardized mean difference between two groups.

    Uses pooled standard deviation. Interpretation:
    |d| < 0.2: negligible, 0.2–0.5: small, 0.5–0.8: medium, > 0.8: large.
    """
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return {"error": "Need ≥ 2 observations per group"}

    mean_diff = np.mean(x) - np.mean(y)
    # Pooled SD
    pooled_sd = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
        / (nx + ny - 2)
    )
    d = mean_diff / pooled_sd if pooled_sd > 0 else 0.0

    # Hedges' g correction for small samples
    correction = 1 - 3 / (4 * (nx + ny) - 9)
    g = d * correction

    ad = abs(d)
    magnitude = (
        "Large" if ad > 0.8 else "Medium" if ad > 0.5
        else "Small" if ad > 0.2 else "Negligible"
    )

    return {
        "cohens_d": round(d, 4),
        "hedges_g": round(g, 4),
        "magnitude": magnitude,
        "mean_diff": round(mean_diff, 4),
        "pooled_sd": round(pooled_sd, 4),
    }


# ══════════════════════════════════════════════════════════════════════
# 8. DISTRIBUTION FITTING
# ══════════════════════════════════════════════════════════════════════

def fit_distributions(
    data: pd.Series,
) -> pd.DataFrame:
    """
    Fit normal, log-normal, and gamma distributions to data.

    Uses MLE parameter estimation and Kolmogorov-Smirnov goodness-of-fit.
    In oncology, tumor volumes are classically log-normal — this provides
    evidence for or against that assumption.
    """
    clean = data.dropna().values
    clean = clean[clean > 0]  # positive values only for log-normal & gamma
    if len(clean) < 10:
        return pd.DataFrame({"error": ["Need ≥ 10 positive observations"]})

    results = []

    # Normal
    mu, sigma = norm.fit(clean)
    ks_stat, ks_p = kstest(clean, "norm", args=(mu, sigma))
    results.append({
        "Distribution": "Normal",
        "Params": f"μ={mu:.2f}, σ={sigma:.2f}",
        "KS Statistic": round(ks_stat, 4),
        "KS p-value": round(ks_p, 6),
        "Good Fit (p>0.05)": ks_p > 0.05,
        "AIC": _aic_normal(clean, mu, sigma),
    })

    # Log-normal
    shape, loc, scale = lognorm.fit(clean, floc=0)
    ks_stat, ks_p = kstest(clean, "lognorm", args=(shape, loc, scale))
    results.append({
        "Distribution": "Log-Normal",
        "Params": f"σ={shape:.3f}, μ={np.log(scale):.3f}",
        "KS Statistic": round(ks_stat, 4),
        "KS p-value": round(ks_p, 6),
        "Good Fit (p>0.05)": ks_p > 0.05,
        "AIC": _aic_lognormal(clean, shape, loc, scale),
    })

    # Gamma
    shape_g, loc_g, scale_g = gamma.fit(clean, floc=0)
    ks_stat, ks_p = kstest(clean, "gamma", args=(shape_g, loc_g, scale_g))
    results.append({
        "Distribution": "Gamma",
        "Params": f"k={shape_g:.3f}, θ={scale_g:.3f}",
        "KS Statistic": round(ks_stat, 4),
        "KS p-value": round(ks_p, 6),
        "Good Fit (p>0.05)": ks_p > 0.05,
        "AIC": _aic_gamma(clean, shape_g, loc_g, scale_g),
    })

    result_df = pd.DataFrame(results)
    # Highlight best model by AIC
    if "AIC" in result_df.columns:
        best_idx = result_df["AIC"].idxmin()
        result_df["Best Model"] = False
        result_df.loc[best_idx, "Best Model"] = True

    return result_df


def _aic_normal(data, mu, sigma):
    """AIC for normal distribution."""
    n = len(data)
    log_lik = np.sum(norm.logpdf(data, mu, sigma))
    return 2 * 2 - 2 * log_lik  # 2 params


def _aic_lognormal(data, shape, loc, scale):
    """AIC for log-normal distribution."""
    n = len(data)
    log_lik = np.sum(lognorm.logpdf(data, shape, loc, scale))
    return 2 * 2 - 2 * log_lik


def _aic_gamma(data, shape, loc, scale):
    """AIC for gamma distribution."""
    n = len(data)
    log_lik = np.sum(gamma.logpdf(data, shape, loc, scale))
    return 2 * 2 - 2 * log_lik


# ══════════════════════════════════════════════════════════════════════
# 9. RISK FACTOR ANALYSIS (Logistic-style)
# ══════════════════════════════════════════════════════════════════════

def univariate_risk_association(
    df: pd.DataFrame,
    outcome_col: str = "Risk_Level",
    binary_split: bool = True,  # HIGH vs rest
) -> pd.DataFrame:
    """
    Univariate analysis: which continuous variables best discriminate
    HIGH-risk from non-HIGH-risk patients?

    Uses Mann-Whitney U + Cliff's delta + ROC AUC for ranking.
    This is clinically framed as "risk factor identification."
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if binary_split:
        high_mask = df[outcome_col] == "HIGH"
        low_mask = df[outcome_col] != "HIGH"

    rows = []
    for col in numeric_cols:
        if col == outcome_col:
            continue
        high_vals = df.loc[high_mask, col].dropna()
        low_vals = df.loc[low_mask, col].dropna()

        if len(high_vals) < 3 or len(low_vals) < 3:
            continue

        # Mann-Whitney U
        u_stat, u_p = mannwhitneyu(high_vals, low_vals, alternative="two-sided")
        # Cliff's delta
        cd = _cliffs_delta(high_vals.values, low_vals.values)
        # Simple AUC (ranking-based)
        combined = np.concatenate([high_vals, low_vals])
        labels = np.concatenate([np.ones(len(high_vals)), np.zeros(len(low_vals))])
        # Rank AUC approximation
        try:
            from scipy.stats import rankdata
            ranks = rankdata(combined)
            r1 = np.mean(ranks[:len(high_vals)])
            auc = (r1 - (len(high_vals) + 1) / 2) / len(low_vals)
        except Exception:
            auc = np.nan

        rows.append({
            "Variable": col,
            "HIGH_Median": round(high_vals.median(), 3),
            "Non-HIGH_Median": round(low_vals.median(), 3),
            "MannWhitney_U": round(u_stat, 2),
            "p_value": round(u_p, 6),
            "Significant": u_p < 0.05,
            "Cliffs_Delta": round(cd, 3),
            "Effect_Size": _cliff_interpretation(cd),
            "AUC": round(auc, 4) if not np.isnan(auc) else None,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("Cliffs_Delta", ascending=False, key=abs)
    return result


# ══════════════════════════════════════════════════════════════════════
# 10. SAMPLE SIZE / POWER ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def minimum_detectable_effect(
    data: pd.Series,
    alpha: float = 0.05,
    power: float = 0.80,
) -> Dict[str, object]:
    """
    What's the smallest effect we can detect with this sample size?

    For a given sample, computes the minimum detectable standardized
    effect size (Cohen's d) at 80% power and α=0.05 (two-sided t-test).

    Clinically relevant: "With 50 patients, can we detect a clinically
    meaningful difference?"
    """
    clean = data.dropna()
    n = len(clean)
    if n < 3:
        return {"error": "Need ≥ 3 observations"}

    # Non-central t distribution approach
    from scipy.stats import nct, t as t_dist
    df = n - 1
    t_crit = t_dist.ppf(1 - alpha / 2, df)

    # Solve for non-centrality parameter
    def power_at_ncp(ncp_val):
        return 1 - nct.cdf(t_crit, df, ncp_val) + nct.cdf(-t_crit, df, ncp_val)

    # Binary search for MDE
    lo, hi = 0.0, 10.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if power_at_ncp(mid * np.sqrt(n)) < power:
            lo = mid
        else:
            hi = mid

    mde = (lo + hi) / 2

    return {
        "sample_size": n,
        "alpha": alpha,
        "target_power": power,
        "min_detectable_d": round(mde, 3),
        "interpretation": (
            f"With n={n}, we can detect effects of d ≥ {mde:.3f} "
            f"({_cohens_label(mde)}) with {power*100:.0f}% power."
        ),
    }


def _cohens_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"
