"""Pure, domain-agnostic math for conformal recovery-deadline certificates.

Object: a runtime-assurance engagement deadline ``d_alpha`` such that, for an
exchangeable future fault episode, the supervised controller returns to the
safe (warn) set within ``d_alpha`` steps with probability >= 1 - alpha. The
deadline is the split-conformal upper quantile of calibration recovery times,
with the never-recovered episodes entering as a +inf atom (censoring at the
horizon) — validity needs no continuity, so the atom is handled by the same
rank argument, and an unattainable coverage level returns ``inf`` (the
certificate's honest refusal).

Everything here consumes plain float arrays (recovery times, weights, group
labels) — no simulator, no controller. Domain adapters produce the arrays.

Extensions beyond the WS2 spacecraft module:
- ``weighted_conformal_deadline``: coverage under a KNOWN fault-distribution
  shift via likelihood-ratio weights (Tibshirani et al. 2019, specialised to
  parametric fault samplers where the ratio is exact).
- ``mondrian_deadlines``: per-fault-class (group-conditional) deadlines.
- ``split_validate``: the repeated random calibration/test split engine used
  to validate expected coverage empirically.
- ``cumulative_breach_count``: an alternative conformal score that also
  catches in-and-out chattering episodes a run-length deadline cannot.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = [
    "recovery_steps",
    "cumulative_breach_count",
    "conformal_deadline",
    "weighted_conformal_deadline",
    "mondrian_deadlines",
    "coverage",
    "heuristic_deadline",
    "effective_sample_size",
    "split_validate",
]


def recovery_steps(trace: Sequence[float] | np.ndarray, warn: float) -> float:
    """Recovery time = first step after which the trace stays within ``warn``.

    = (last breach step) + 1; 0 if it never breaches; +inf if still breaching at
    the end (censored: did not recover within the horizon)."""
    a = np.asarray(trace, dtype=float)
    breaches = np.where(a > warn)[0]
    if breaches.size == 0:
        return 0.0
    last = int(breaches[-1])
    return float("inf") if last >= a.size - 1 else float(last + 1)


def cumulative_breach_count(trace: Sequence[float] | np.ndarray, warn: float) -> float:
    """Total steps spent above ``warn`` — a conformal score whose budget rule also
    catches chattering episodes (every run short, never settles) that a
    run-length deadline misses."""
    a = np.asarray(trace, dtype=float)
    return float(np.sum(a > warn))


def conformal_deadline(cal: np.ndarray, alpha: float) -> float:
    """Split-conformal upper quantile of the recovery time.

    Finite-sample, distribution-free under exchangeability: ``P(R_test <= d) >=
    1 - alpha``. ``d`` is the ``ceil((1-alpha)(n+1))``-th smallest calibration
    value; if that rank exceeds ``n`` (recovery rate below the requested
    coverage), ``d = inf`` — the certificate refuses rather than fakes."""
    cal = np.asarray(cal, dtype=float)
    n = cal.size
    rank = math.ceil((1.0 - alpha) * (n + 1))  # 1..n+1
    if rank > n:
        return float("inf")
    return float(np.sort(cal)[rank - 1])


def weighted_conformal_deadline(
    cal: np.ndarray, w_cal: np.ndarray, w_test: float, alpha: float
) -> float:
    """Weighted split-conformal deadline under covariate shift (Tibshirani et al.
    2019): the (1-alpha)-quantile of ``sum_i p_i delta_{V_i} + p_inf delta_inf``
    with ``p_i = w(X_i) / (sum_j w(X_j) + w(x_test))`` and the test point's mass
    at +inf. ``w`` is the deployment/calibration likelihood ratio dQ/dP — exact
    for parametric fault samplers. Reduces to ``conformal_deadline`` when all
    weights are equal."""
    cal = np.asarray(cal, dtype=float)
    w = np.asarray(w_cal, dtype=float)
    if cal.size != w.size:
        raise ValueError("cal and w_cal must align")
    if np.any(w < 0) or w_test < 0:
        raise ValueError("weights must be nonnegative")
    order = np.argsort(cal)  # +inf values sort to the end, as required
    v, wo = cal[order], w[order]
    total = float(wo.sum()) + float(w_test)
    if total <= 0:
        return float("inf")
    cum = np.cumsum(wo) / total
    idx = int(np.searchsorted(cum, 1.0 - alpha, side="left"))
    if idx >= v.size or not math.isfinite(float(v[idx])):
        return float("inf")
    return float(v[idx])


def mondrian_deadlines(
    cal: np.ndarray, groups: Sequence[object], alpha: float
) -> dict[object, float]:
    """Group-conditional (Mondrian) deadlines: the plain split-conformal deadline
    within each a-priori group (e.g. fault class). Coverage then holds PER GROUP;
    small groups may honestly return inf (need n >= (1-alpha)/alpha)."""
    cal = np.asarray(cal, dtype=float)
    garr = np.asarray(groups, dtype=object)
    if cal.size != garr.size:
        raise ValueError("cal and groups must align")
    return {g: conformal_deadline(cal[garr == g], alpha) for g in dict.fromkeys(garr.tolist())}


def coverage(test: np.ndarray, d: float) -> float:
    """Empirical fraction of test episodes that recover within deadline ``d``."""
    test = np.asarray(test, dtype=float)
    return float(np.mean(test <= d)) if test.size else float("nan")


def heuristic_deadline(cal: np.ndarray, margin: float = 0.30) -> float:
    """The pre-certificate practice: p95 of the FINITE calibration recoveries
    x (1 + margin). No coverage guarantee — kept as the comparison baseline."""
    cal = np.asarray(cal, dtype=float)
    finite = cal[np.isfinite(cal)]
    if finite.size == 0:
        return float("inf")
    return float(math.ceil(np.percentile(finite, 95) * (1.0 + margin)))


def effective_sample_size(w: np.ndarray) -> float:
    """Kish effective sample size of importance weights: (sum w)^2 / sum w^2.
    Reported alongside weighted certificates — a small ESS means the shift is
    severe and the weighted deadline rests on few effective calibration points."""
    w = np.asarray(w, dtype=float)
    s = float(w.sum())
    q = float((w * w).sum())
    return s * s / q if q > 0 else 0.0


def split_validate(
    rec: np.ndarray,
    alphas: Sequence[float],
    *,
    n_splits: int,
    cal_frac: float,
    rng: np.random.Generator,
    weights: np.ndarray | None = None,
) -> list[dict]:
    """Validate expected coverage by repeated random calibration/test splits.

    A single split's test coverage is a noisy estimate; the conformal guarantee
    is on the EXPECTED coverage, so we report the mean (and std) over splits.
    With ``weights`` (per-episode likelihood ratios dQ/dP), the deadline is the
    weighted one and the test coverage is importance-weighted toward Q."""
    rec = np.asarray(rec, dtype=float)
    n = rec.size
    n_cal = int(round(n * cal_frac))
    out: list[dict] = []
    for alpha in alphas:
        covs, ds = [], []
        for _ in range(n_splits):
            perm = rng.permutation(n)
            cal_idx, test_idx = perm[:n_cal], perm[n_cal:]
            if weights is None:
                d = conformal_deadline(rec[cal_idx], float(alpha))
            else:
                d = weighted_conformal_deadline(
                    rec[cal_idx], weights[cal_idx], float(np.mean(weights[test_idx])), float(alpha)
                )
            if math.isfinite(d):
                ds.append(d)
                t, wt = rec[test_idx], None if weights is None else weights[test_idx]
                if wt is None:
                    covs.append(coverage(t, d))
                else:  # importance-weighted test coverage (estimates P_Q)
                    covs.append(float(np.sum(wt * (t <= d)) / np.sum(wt)))
        out.append(
            {
                "alpha": float(alpha),
                "target_coverage": round(1.0 - float(alpha), 4),
                "splits_with_finite_deadline": round(len(ds) / n_splits, 4),
                "mean_coverage": round(float(np.mean(covs)), 4) if covs else None,
                "std_coverage": round(float(np.std(covs)), 4) if covs else None,
                "median_deadline": float(np.median(ds)) if ds else None,
            }
        )
    return out
