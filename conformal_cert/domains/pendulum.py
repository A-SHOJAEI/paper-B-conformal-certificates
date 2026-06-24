"""Inverted-pendulum domain for the conformal recovery-deadline certificate (Paper B, B3).

A second, non-spacecraft Simplex testbed that reproduces the WS2 story end-to-end on an
unrelated plant, turning the spacecraft certificate into a domain-general *mechanism*:

  * Plant: a torque-controlled inverted pendulum, unstable at upright. The fault is an
    actuator *effectiveness gain* ``b`` on the control (applied torque ``= b * u``) — the
    1-DOF analog of the spacecraft per-axis gain fault ``a_i g_i``. ``b < 0`` is a
    control-direction reversal; ``b > 0`` with ``|b| != 1`` is a benign magnitude error.
  * Capable adapting controller: an analytic PD law fed an ONLINE estimate of ``sign(b)``
    from the command-vs-acceleration correlation ``sign(u)·sign(Δθ̇)`` (the same
    system-identification signal the spacecraft RMA / classical-adaptive law uses) under
    NOISY rate sensing, latched once detected. On a reversal it drives the wrong way,
    crosses the safe-set boundary, detects the sign, then recovers — a *bounded recovery
    transient*. The noise is why a naive latch is tempting and why detection takes time.
  * Incapable baseline: the fault-unaware fixed-PD law (``b̂ = +1`` always); on a reversal
    it diverges and never recovers.

Three results, each mirroring WS2 on a different plant:
  1. The pathology + cure. Every reversal forces a transient excursion past the safe-set
     boundary, so a latching Simplex shield trips and never returns control — SUPPRESSING a
     capable controller. A split-conformal recovery-deadline (calibrated on held-out faults)
     plus recovery-aware engagement keeps it autonomous while still catching the diverging
     baseline at the deadline.
  2. Coverage under a known fault-distribution shift (Paper B, B2(i)). Calibration sees a
     fault mix with few reversals; deployment sees more. Because the sign sampler is ours,
     the deployment/calibration likelihood ratio is EXACT, so ``weighted_conformal_deadline``
     restores coverage where the unweighted certificate provably undercovers — the clean
     answer to the exchangeability caveat the spacecraft module could only flag.
  3. Per-class (Mondrian) deadlines (Paper B, B2(ii)): group-conditional coverage, a
     separate deadline per fault class.

Pure NumPy, deterministic. Run: ``python -m conformal_cert.domains.pendulum`` ->
``evidence/conformal_cert/pendulum_domain.json``.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from conformal_cert import core

logger = logging.getLogger(__name__)
OUT = Path("evidence/conformal_cert/pendulum_domain.json")

# --- plant + control constants ---
OMEGA2 = 9.81  # g/L (rad/s^2): inverted-pendulum upright instability; m L^2 = 1
DT = 0.02  # s
HORIZON = 400  # steps (8.0 s)
KP, KD = 40.0, 12.0  # PD gains for the nominal plant (b=1): stable, ~1 s settle
WARN_RAD = math.radians(5.0)  # safe-set boundary; recovery = return within and stay
CRITICAL_RAD = math.radians(45.0)  # the verified backstop limit (Phi_c analog): hard safety floor
SIGMA_RATE = 0.15  # rad/s rate-sensing noise (corrupts the sysID sign signal)
THETA_CAP, RATE_CAP = 6.0, 50.0  # numerical guards for the diverging (fallen) case

# --- online sign-ID (the adapting controller's learned component) ---
CORR_LATCH = 8.0  # latch the sign estimate once |running correlation| reaches this
MOVE_EPS = 1e-4  # ignore near-zero-command steps when accumulating the sysID correlation

# --- fault distribution ---
BMAG = (0.5, 1.5)  # |b| ~ U(0.5, 1.5)
THETA0 = (0.05, 0.08)  # initial perturbation (rad), within the 5deg safe set
P_REV_CAL = 0.30  # calibration reversal fraction (mostly benign faults)
P_REV_DEPLOY = 0.70  # deployment reversal fraction (harder; the shift Q)

REVERSED, NORMAL = "reversed", "normal"


@dataclass(frozen=True)
class Fault:
    """An actuator fault: signed effectiveness gain ``b`` and initial angle ``theta0``."""

    b: float
    theta0: float

    @property
    def fault_class(self) -> str:
        return REVERSED if self.b < 0 else NORMAL


def sample_faults(rng: np.random.Generator, n: int, p_rev: float) -> list[Fault]:
    """Sample ``n`` faults: each is a reversal with probability ``p_rev`` (else a benign
    magnitude fault); ``|b|`` and the initial perturbation are randomized."""
    out: list[Fault] = []
    for _ in range(n):
        rev = rng.random() < p_rev
        mag = float(rng.uniform(*BMAG))
        th0 = float(rng.uniform(*THETA0))
        out.append(Fault(b=-mag if rev else mag, theta0=th0))
    return out


def class_shift_weights(faults: list[Fault], p_cal: float, p_deploy: float) -> np.ndarray:
    """Exact per-fault likelihood ratio ``w = q(class)/p(class)`` for a shift in the
    reversal fraction from ``p_cal`` to ``p_deploy`` (the sign sampler is ours, so the
    ratio is exact): ``q/p`` for reversals, ``(1-q)/(1-p)`` for benign faults."""
    return np.array(
        [
            (p_deploy / p_cal) if f.fault_class == REVERSED else ((1 - p_deploy) / (1 - p_cal))
            for f in faults
        ],
        dtype=float,
    )


def rollout(fault: Fault, *, adaptive: bool, seed: int) -> np.ndarray:
    """Simulate one episode; return the per-step |theta| trace (rad).

    ``adaptive`` controllers infer ``sign(b)`` online from noisy rate sensing and latch it;
    the fixed baseline always assumes ``b̂ = +1``. Control law: the analytic PD
    ``u = -(KP·theta + KD·thetadot_meas) / b̂`` with ``b̂`` in {-1, +1}."""
    rng = np.random.default_rng(seed)
    theta, thetadot = fault.theta0, 0.0
    b = fault.b
    bhat = 1.0  # nominal sign assumption
    corr = 0.0  # running command-vs-acceleration correlation (sysID)
    latched = False
    trace = np.empty(HORIZON, dtype=float)
    for t in range(HORIZON):
        thetadot_meas = thetadot + rng.normal(0.0, SIGMA_RATE)
        u = -(KP * theta + KD * thetadot_meas) / bhat
        thetaddot = OMEGA2 * math.sin(theta) + b * u
        new_thetadot = thetadot + thetaddot * DT
        if adaptive and not latched:
            dwdot_meas = (new_thetadot - thetadot) + rng.normal(0.0, SIGMA_RATE * DT)
            if abs(u) > MOVE_EPS and abs(dwdot_meas) > 0.0:
                corr += math.copysign(1.0, u) * math.copysign(1.0, dwdot_meas)
            if abs(corr) >= CORR_LATCH:
                bhat = math.copysign(1.0, corr)  # sign(u)·sign(Δθ̇) = sign(b)
                latched = True
        thetadot = float(np.clip(new_thetadot, -RATE_CAP, RATE_CAP))
        theta = float(np.clip(theta + thetadot * DT, -THETA_CAP, THETA_CAP))
        trace[t] = abs(theta)
    return trace


def collect(faults: list[Fault], *, adaptive: bool, seed0: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-fault recovery time (steps; +inf if never recovers) and a boolean: whether the
    episode ever left the safe set (i.e., a latching Simplex shield would trip)."""
    rec = np.empty(len(faults), dtype=float)
    breached = np.empty(len(faults), dtype=bool)
    for i, f in enumerate(faults):
        tr = rollout(f, adaptive=adaptive, seed=seed0 + i)
        rec[i] = core.recovery_steps(tr, WARN_RAD)
        breached[i] = bool(np.any(tr > WARN_RAD))
    return rec, breached


def _stats(rec: np.ndarray) -> dict:
    finite = rec[np.isfinite(rec)]
    return {
        "n": int(rec.size),
        "recovered_rate": round(float(np.mean(np.isfinite(rec))), 4),
        "median_recovery_steps": round(float(np.median(finite)), 1) if finite.size else None,
        "p95_recovery_steps": round(float(np.percentile(finite, 95)), 1) if finite.size else None,
    }


# --- B4: RTA engagement-rule baselines -------------------------------------------------
# Each rule decides, per step, whether to keep the controller engaged or escalate (trip to
# safe-hold). A verified Phi_c backstop trips ANY trajectory that crosses CRITICAL_RAD,
# regardless of the rule (the formal safety floor). A rule returns the trip step, or None if
# it never escalates (the controller is kept fully autonomous).


def _first_above(trace: np.ndarray, thr: float) -> int | None:
    idx = np.where(trace > thr)[0]
    return int(idx[0]) if idx.size else None


def _trip_latching(trace: np.ndarray) -> int | None:
    """Latching Simplex: escalate on the first breach of the safe set, coast thereafter."""
    return _first_above(trace, WARN_RAD)


def _trip_deadline(trace: np.ndarray, deadline: float) -> int | None:
    """Recovery-deadline rule (heuristic or conformal): keep engaged until the recovery
    deadline; escalate if not recovered by then, OR earlier if the verified backstop fires."""
    crit = _first_above(trace, CRITICAL_RAD)
    rec = core.recovery_steps(trace, WARN_RAD)
    missed = rec > deadline  # did not recover within the certified/heuristic deadline
    cands = [
        s
        for s in (crit, (int(deadline) if (missed and math.isfinite(deadline)) else None))
        if s is not None
    ]
    return min(cands) if cands else None


def _trip_value(trace: np.ndarray, tau: float) -> int | None:
    """CP-on-safety-value gating (Tabbara et al. 2025 style): escalate when the safety value
    |theta| exceeds a conformally-calibrated bound tau (with the Phi_c backstop)."""
    cands = [
        s for s in (_first_above(trace, tau), _first_above(trace, CRITICAL_RAD)) if s is not None
    ]
    return min(cands) if cands else None


def _peaks_and_traces(
    faults: list[Fault], *, adaptive: bool, seed0: int
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Recovery times, peak |theta| (rad), and the full traces for a fault set."""
    rec = np.empty(len(faults), dtype=float)
    peak = np.empty(len(faults), dtype=float)
    traces: list[np.ndarray] = []
    for i, f in enumerate(faults):
        tr = rollout(f, adaptive=adaptive, seed=seed0 + i)
        rec[i] = core.recovery_steps(tr, WARN_RAD)
        peak[i] = float(tr.max())
        traces.append(tr)
    return rec, peak, traces


def engagement_baselines(seed: int, alpha: float = 0.10) -> dict:
    """Compare RTA engagement rules on the reversal regime (the regime where the decision
    is non-trivial): autonomy retention on the capable adapting controller, and catch-rate /
    time-to-catch on the diverging fault-unaware controller. Thresholds are calibrated on a
    held-out reversal calibration split; metrics are reported on a disjoint reversal test
    split. Latching and heuristic rules carry no coverage guarantee; the value-gating and
    recovery-deadline rules carry a 1-alpha conformal guarantee on their respective scores."""
    m = 200
    rng = np.random.default_rng(seed + 5)
    cal_faults = [
        Fault(b=-float(rng.uniform(*BMAG)), theta0=float(rng.uniform(*THETA0))) for _ in range(m)
    ]
    test_faults = [
        Fault(b=-float(rng.uniform(*BMAG)), theta0=float(rng.uniform(*THETA0))) for _ in range(m)
    ]

    cal_rec, cal_peak, _ = _peaks_and_traces(cal_faults, adaptive=True, seed0=40_000)
    d_alpha = core.conformal_deadline(cal_rec, alpha)  # recovery-TIME deadline (ours)
    d_heur = core.heuristic_deadline(cal_rec)  # p95 x 1.3 (no guarantee)
    tau = core.conformal_deadline(cal_peak, alpha)  # safety-VALUE bound (peak |theta|)

    _, _, ad_traces = _peaks_and_traces(test_faults, adaptive=True, seed0=50_000)  # recoverable
    _, _, fx_traces = _peaks_and_traces(test_faults, adaptive=False, seed0=60_000)  # diverging

    # each rule pre-bound to its calibrated threshold -> uniform Callable[[trace], step|None]
    rules: dict[str, tuple[Callable[[np.ndarray], int | None], str]] = {
        "latching_simplex": (_trip_latching, "none"),
        "heuristic_p95x1.3": (lambda t: _trip_deadline(t, d_heur), "none (arbitrary margin)"),
        "cp_safety_value": (
            lambda t: _trip_value(t, tau),
            f"1-alpha on peak |theta| (tau={math.degrees(tau):.1f}deg)",
        ),
        "conformal_recovery_deadline_ours": (
            lambda t: _trip_deadline(t, d_alpha),
            "1-alpha on recovery time",
        ),
    }
    out: dict[str, dict] = {}
    for name, (fn, guarantee) in rules.items():
        ad_trips = [fn(t) for t in ad_traces]
        fx_trips = [fn(t) for t in fx_traces]
        autonomy = float(np.mean([s is None for s in ad_trips]))  # kept engaged through recovery
        crit_fx = [_first_above(t, CRITICAL_RAD) for t in fx_traces]
        caught = float(
            np.mean(
                [
                    (s is not None and (c is None or s <= c))
                    for s, c in zip(fx_trips, crit_fx, strict=True)
                ]
            )
        )
        ttc = [s for s in fx_trips if s is not None]
        out[name] = {
            "autonomy_retention": round(autonomy, 4),
            "divergence_caught_before_critical": round(caught, 4),
            "mean_time_to_catch_steps": round(float(np.mean(ttc)), 1) if ttc else None,
            "coverage_guarantee": guarantee,
        }
    out["_thresholds"] = {
        "alpha": alpha,
        "conformal_recovery_deadline_steps": int(d_alpha) if math.isfinite(d_alpha) else None,
        "heuristic_deadline_steps": int(d_heur) if math.isfinite(d_heur) else None,
        "cp_value_peak_threshold_deg": round(math.degrees(tau), 2) if math.isfinite(tau) else None,
        "critical_backstop_deg": round(math.degrees(CRITICAL_RAD), 1),
        "note": (
            "Neural-Simplex reverse switching is omitted: on an open-loop-unstable plant a "
            "control-direction reversal disables the safe controller too (it shares the "
            "reversed actuator), so reverse switching degenerates to latching here."
        ),
    }
    return out


def build(seed: int = 0) -> dict:
    n = 400
    rng = np.random.default_rng(seed)
    faults = sample_faults(rng, n, P_REV_CAL)  # calibration mix
    rev = np.array([f.fault_class == REVERSED for f in faults], dtype=bool)
    groups = np.array([f.fault_class for f in faults], dtype=object)

    rec_ad, breach_ad = collect(faults, adaptive=True, seed0=10_000)
    rec_fx, _ = collect(faults, adaptive=False, seed0=20_000)

    alphas = [0.20, 0.10, 0.05]
    cov = core.split_validate(
        rec_ad, alphas, n_splits=300, cal_frac=0.5, rng=np.random.default_rng(seed + 1)
    )

    # one calibration/test split shared by the pathology + shift experiments
    perm = np.random.default_rng(seed + 2).permutation(n)
    cal, test = perm[: n // 2], perm[n // 2 :]
    d10 = core.conformal_deadline(rec_ad[cal], 0.10)

    # --- pathology + cure + discrimination, on the held-out reversal class ---
    rev_test = test[rev[test]]
    latched_auto = float(np.mean(~breach_ad[rev_test]))  # latch trips on any breach
    recovery_aware_auto = float(np.mean(rec_ad[rev_test] <= d10))
    fixed_auto = float(np.mean(rec_fx[rev_test] <= d10))  # diverges -> caught

    # --- B2(i): coverage under a known fault-mix shift Q (more reversals) ---
    rng_q = np.random.default_rng(seed + 3)
    faults_q = sample_faults(rng_q, n, P_REV_DEPLOY)
    rec_q, _ = collect(faults_q, adaptive=True, seed0=30_000)
    w = class_shift_weights(faults, P_REV_CAL, P_REV_DEPLOY)
    d_unw = core.conformal_deadline(rec_ad[cal], 0.10)
    d_w = core.weighted_conformal_deadline(rec_ad[cal], w[cal], float(np.mean(w[test])), 0.10)
    cov_unw_q = float(np.mean(rec_q <= d_unw))
    cov_w_q = float(np.mean(rec_q <= d_w))
    ess = core.effective_sample_size(w[cal])

    # --- B2(ii): per-class (Mondrian) deadlines ---
    mondrian = core.mondrian_deadlines(rec_ad[cal], groups[cal], 0.10)

    # --- B4: RTA engagement-rule baselines (reversal regime) ---
    baselines = engagement_baselines(seed, 0.10)

    return {
        "domain": "inverted_pendulum",
        "description": (
            "Torque-controlled inverted pendulum; actuator effectiveness/sign fault b "
            "(applied torque = b*u). DT=0.02s, horizon 8s, WARN=5deg, noisy rate sensing "
            "(sigma=0.15 rad/s). Mirrors the spacecraft WS2 conformal certificate."
        ),
        "n_faults": n,
        "calibration_reversal_fraction": P_REV_CAL,
        "controllers": {
            "adapting_online_signID": _stats(rec_ad),
            "fixed_pd_fault_unaware": _stats(rec_fx),
        },
        "certificate_coverage": cov,
        "pathology_and_cure": {
            "alpha": 0.10,
            "evaluated_on": "held-out reversal class",
            "conformal_deadline_steps": (None if not math.isfinite(d10) else int(d10)),
            "conformal_deadline_s": (None if not math.isfinite(d10) else round(d10 * DT, 2)),
            "latching_simplex_autonomy_adapting": round(latched_auto, 4),
            "recovery_aware_autonomy_adapting": round(recovery_aware_auto, 4),
            "recovery_aware_autonomy_fixed": round(fixed_auto, 4),
            "note": (
                "Every reversal forces a transient breach of the safe set, so a latching "
                "Simplex shield suppresses the adapting controller; recovery-aware "
                "engagement keeps it autonomous while catching the fault-unaware baseline."
            ),
        },
        "shift_coverage_B2": {
            "alpha": 0.10,
            "target_coverage": 0.90,
            "calibration_dist": f"P: {int(P_REV_CAL * 100)}% reversals",
            "deployment_dist": f"Q: {int(P_REV_DEPLOY * 100)}% reversals (harder)",
            "unweighted_coverage_on_Q": round(cov_unw_q, 4),
            "weighted_coverage_on_Q": round(cov_w_q, 4),
            "unweighted_deadline_steps": (None if not math.isfinite(d_unw) else int(d_unw)),
            "weighted_deadline_steps": (None if not math.isfinite(d_w) else int(d_w)),
            "effective_sample_size": round(ess, 1),
            "note": (
                "The P-calibrated deadline undercovers under the deployment shift Q; the "
                "weighted (exact likelihood-ratio) deadline restores coverage to >= 1-alpha."
            ),
        },
        "mondrian_deadlines_B2": {
            "alpha": 0.10,
            "per_class_steps": {
                str(k): (None if not math.isfinite(v) else int(v)) for k, v in mondrian.items()
            },
            "note": "Group-conditional deadlines: coverage holds per fault class.",
        },
        "baselines_B4": baselines,
        "guarantee": (
            "split-conformal: E[P(recovery_steps <= d_alpha)] >= 1 - alpha over exchangeable "
            "faults; weighted variant extends this to a known fault-distribution shift."
        ),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    c, s = payload["pathology_and_cure"], payload["shift_coverage_B2"]
    logger.info("Inverted-pendulum conformal certificate (Paper B, B3):")
    for row in payload["certificate_coverage"]:
        logger.info(
            "  alpha=%.2f target=%.2f mean_coverage=%s deadline=%s steps",
            row["alpha"],
            row["target_coverage"],
            row["mean_coverage"],
            row["median_deadline"],
        )
    logger.info(
        "  PATHOLOGY: latching-Simplex autonomy (adapting) = %.1f%%  ->"
        " CURE: recovery-aware = %.1f%%",
        100 * c["latching_simplex_autonomy_adapting"],
        100 * c["recovery_aware_autonomy_adapting"],
    )
    logger.info(
        "  DISCRIMINATION: fault-unaware fixed-PD autonomy = %.1f%% (deadline %s steps)",
        100 * c["recovery_aware_autonomy_fixed"],
        c["conformal_deadline_steps"],
    )
    logger.info(
        "  B2 SHIFT: unweighted coverage on Q = %.1f%% (undercovers) ->"
        " weighted = %.1f%% (ESS %.0f)",
        100 * s["unweighted_coverage_on_Q"],
        100 * s["weighted_coverage_on_Q"],
        s["effective_sample_size"],
    )
    logger.info(
        "  B2 Mondrian per-class deadlines: %s", payload["mondrian_deadlines_B2"]["per_class_steps"]
    )
    logger.info(
        "  B4 engagement-rule baselines (reversal regime, autonomy / caught / time-to-catch):"
    )
    for name, r in payload["baselines_B4"].items():
        if name.startswith("_"):
            continue
        logger.info(
            "    %-34s auto=%5.1f%% caught=%5.1f%% ttc=%s steps  [%s]",
            name,
            100 * r["autonomy_retention"],
            100 * r["divergence_caught_before_critical"],
            r["mean_time_to_catch_steps"],
            r["coverage_guarantee"],
        )
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
