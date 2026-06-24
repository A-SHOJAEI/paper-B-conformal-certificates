"""Split-conformal recovery-time certificate for the recovery-aware RTA (WS2 core).

The adaptation-aware RTA (``program/wp2b_adaptive_rta.py``) lets a capable *adapting*
controller keep control inside the recovery zone until a recovery DEADLINE; WP7 set that
deadline with a ``p95 x 1.3`` heuristic (``wp7_reachability_rta.py:129``) — no coverage
guarantee, an arbitrary margin. This module replaces it with a distribution-free,
finite-sample **split-conformal** upper bound: a deadline ``d_alpha`` such that, for an
exchangeable future fault, the deployed controller returns to the safe (warn) set within
``d_alpha`` steps with probability ``>= 1 - alpha``. The deadline is calibrated on a
held-out calibration split and its coverage is *validated* on a disjoint test split.

This is the conformal recovery-time certificate that **licenses delaying fallback**: the
RTA may keep the controller engaged up to ``d_alpha`` (autonomy), and — with guaranteed
coverage ``1 - alpha`` — catches any controller that misses its certified recovery
deadline. If the deployed controller's recovery rate is below ``1 - alpha`` there is *no*
finite deadline (``d_alpha = inf``), which the certificate reports honestly.

Uses the WS1-validated rollout harness + the deployed (latched) RMA student. Run:
``python -m program.conformal_rta``  ->  ``evidence/program/conformal_rta.json``.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from conformal_cert import core as _core
from program import rollout
from program.fault_taxonomy import FaultClass, heldout_test_faults

logger = logging.getLogger(__name__)
OUT = Path("evidence/program/conformal_rta.json")

WARN_DEG = 5.0  # safe-set boundary (I4 breach threshold); recovery = return within this
LATCH_DEG = 3.0  # the deployed WS1 latched RMA student
HEURISTIC_MARGIN = 0.30  # WP7's p95 x (1 + margin), reproduced here for comparison

# The pure conformal math lives in conformal_cert.core (domain-agnostic; Paper B).
# These wrappers keep this module's audited signatures/defaults byte-compatible.
conformal_deadline = _core.conformal_deadline
coverage = _core.coverage


def recovery_steps(trace: Sequence[float], warn: float = WARN_DEG) -> float:
    """Recovery time (see ``conformal_cert.core.recovery_steps``) with the I4 warn
    threshold as the default safe set."""
    return _core.recovery_steps(trace, warn)


def collect_recovery(
    maker: rollout.MakePolicy, cfg: object, faults: list, seed0: int
) -> np.ndarray:
    """Per-fault recovery time of a controller (one seed per fault, so each fault is one
    exchangeable calibration/test unit)."""
    fn = rollout.make_rollout_fn(maker, cfg=cfg)
    return np.array([recovery_steps(fn(f, seed0 + i)) for i, f in enumerate(faults)], dtype=float)


def _heuristic_deadline(cal: np.ndarray, margin: float = HEURISTIC_MARGIN) -> float:
    """WP7's deadline: p95 of the FINITE calibration recoveries x (1 + margin)."""
    return _core.heuristic_deadline(cal, margin)


# collect(n_faults) -> (recovery times of the deployed controller, of the PD baseline).
CollectFn = Callable[[int], tuple[np.ndarray, np.ndarray]]


def basilisk_collect(
    n_faults: int, *, student_ckpt: str | Path | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The spacecraft collection path (current default): deployed latched RMA student +
    fault-unaware PD over the canonical held-out GAIN battery on real Basilisk."""
    if not rollout.basilisk_available():
        raise ModuleNotFoundError("Basilisk required (HANDOFF.md §6).", name="Basilisk")
    from program import determinism

    determinism.set_global_determinism(11)
    cfg = rollout.default_cfg()
    student = (
        rollout.load_rma_student()
        if student_ckpt is None
        else rollout.load_rma_student(student_ckpt)
    )
    faults = heldout_test_faults(FaultClass.GAIN, n=n_faults, seed=7_000)
    rma_maker = rollout.rma_student_maker(student, cfg, latch_below_deg=LATCH_DEG)
    rec = collect_recovery(rma_maker, cfg, faults, seed0=0)  # fault i -> seed i (consistent)
    rec_pd = collect_recovery(rollout.pd_maker(cfg), cfg, faults, seed0=0)
    return rec, rec_pd


def _summary(rec: np.ndarray) -> dict:
    finite = rec[np.isfinite(rec)]
    return {
        "n": int(rec.size),
        "recovered_frac": round(float(np.mean(np.isfinite(rec))), 4),
        "median_recovery_steps": (round(float(np.median(finite)), 1) if finite.size else None),
        "p95_recovery_steps": (round(float(np.percentile(finite, 95)), 1) if finite.size else None),
        "max_finite_recovery_steps": (int(finite.max()) if finite.size else None),
    }


def run(
    n_faults: int = 400,
    n_splits: int = 300,
    cal_frac: float = 0.5,
    alphas: Sequence[float] = (0.2, 0.1, 0.05),
    *,
    collect: CollectFn | None = None,
) -> dict:
    """Collect recovery times once, then validate the conformal guarantee by averaging
    test coverage over many random calibration/test splits (a single split is a noisy
    coverage estimate; the guarantee is on the EXPECTED coverage).

    ``collect`` is injectable (Paper B domain adapters); ``None`` = the spacecraft
    Basilisk path, byte-compatible with the committed evidence."""
    rec, rec_pd = (collect or basilisk_collect)(n_faults)
    dt_s = 0.5
    n_cal = int(round(n_faults * cal_frac))
    rng = np.random.default_rng(2024)

    certs = []
    for alpha in alphas:
        cov_c, cov_h, ds = [], [], []
        for _ in range(n_splits):
            perm = rng.permutation(n_faults)
            cal, test = rec[perm[:n_cal]], rec[perm[n_cal:]]
            d = conformal_deadline(cal, alpha)
            if math.isfinite(d):
                cov_c.append(coverage(test, d))
                ds.append(d)
            dh = _heuristic_deadline(cal)
            cov_h.append(coverage(test, dh) if math.isfinite(dh) else 1.0)
        certs.append(
            {
                "alpha": alpha,
                "target_coverage": round(1.0 - alpha, 4),
                "splits_with_finite_deadline": round(len(cov_c) / n_splits, 3),
                "mean_conformal_coverage": (round(float(np.mean(cov_c)), 4) if cov_c else None),
                "std_conformal_coverage": (round(float(np.std(cov_c)), 4) if cov_c else None),
                "median_conformal_deadline_steps": (int(np.median(ds)) if ds else None),
                "median_conformal_deadline_s": (
                    round(float(np.median(ds)) * dt_s, 1) if ds else None
                ),
                "mean_heuristic_coverage": round(float(np.mean(cov_h)), 4),
            }
        )

    # Safety(exposure)-vs-autonomy(coverage) Pareto frontier over a fine alpha grid:
    # higher recovery coverage (autonomy) is only attainable with a longer deadline (more
    # time the controller is trusted in the breach zone before fallback).
    pareto = []
    for a in np.round(np.linspace(0.02, 0.5, 13), 3):
        covs, ds = [], []
        for _ in range(n_splits):
            perm = rng.permutation(n_faults)
            d = conformal_deadline(rec[perm[:n_cal]], float(a))
            if math.isfinite(d):
                covs.append(coverage(rec[perm[n_cal:]], d))
                ds.append(d)
        if covs:
            pareto.append(
                {
                    "alpha": float(a),
                    "median_deadline_steps": int(np.median(ds)),
                    "median_deadline_s": round(float(np.median(ds)) * dt_s, 1),
                    "mean_autonomy_coverage": round(float(np.mean(covs)), 4),
                }
            )

    # Recovery-aware RTA discrimination at the operating point (alpha=0.1): the conformal
    # deadline keeps the RECOVERING controller autonomous and CATCHES the non-recovering one.
    perm = rng.permutation(n_faults)
    d_op = conformal_deadline(rec[perm[:n_cal]], 0.1)
    rta = {
        "operating_alpha": 0.1,
        "deadline_steps": (int(d_op) if math.isfinite(d_op) else None),
        "rma_latched_autonomy": round(coverage(rec[perm[n_cal:]], d_op), 4),
        "pd_fault_unaware_autonomy": round(coverage(rec_pd[perm[n_cal:]], d_op), 4),
        "interpretation": "the recovering controller (RMA) is kept autonomous near the "
        "1-alpha rate; the non-recovering one (PD) misses the deadline and is safe-held.",
    }

    payload = {
        "experiment": "conformal_recovery_deadline_rta",
        "controller": "rma_student_latched (deployed)",
        "fault_class": "GAIN (held-out, reversal-heavy)",
        "warn_deg": WARN_DEG,
        "dt_s": dt_s,
        "validation": {
            "n_faults": n_faults,
            "n_splits": n_splits,
            "cal_frac": cal_frac,
            "method": "repeated random calibration/test splits; test coverage averaged over splits",
        },
        "recovery": _summary(rec),
        "recovery_pd_baseline": _summary(rec_pd),
        "certificates": certs,
        "pareto_safety_vs_autonomy": pareto,
        "rta_discrimination": rta,
        "guarantee": "split-conformal: E[P(recovery_steps <= d_alpha)] >= 1 - alpha over "
        "exchangeable faults (finite-sample, distribution-free), validated by mean test "
        "coverage ~ 1 - alpha across random splits. Replaces WP7's p95 x 1.3 heuristic, "
        "which has NO coverage guarantee.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s", OUT)
    return payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = run()
    r = p["recovery"]
    logger.info(
        "recovery to warn(5deg): %.1f%% of %d held-out GAIN faults (median %s, p95 %s steps)",
        100 * r["recovered_frac"],
        r["n"],
        r["median_recovery_steps"],
        r["p95_recovery_steps"],
    )
    logger.info("conformal coverage validated over %d random splits:", p["validation"]["n_splits"])
    for c in p["certificates"]:
        mc = c["mean_conformal_coverage"]
        logger.info(
            "  alpha=%.2f target %.0f%%: conformal mean-coverage %s (std %s), median d=%s steps"
            " (%ss) | heuristic mean-coverage %.1f%%  [finite-deadline splits %.0f%%]",
            c["alpha"],
            100 * c["target_coverage"],
            f"{100 * mc:.1f}%" if mc is not None else "n/a",
            f"{100 * c['std_conformal_coverage']:.1f}%" if mc is not None else "n/a",
            c["median_conformal_deadline_steps"],
            c["median_conformal_deadline_s"],
            100 * c["mean_heuristic_coverage"],
            100 * c["splits_with_finite_deadline"],
        )
    rta = p["rta_discrimination"]
    logger.info(
        "recovery-aware RTA @ alpha=0.1 (deadline %s steps): RMA-latched autonomy %.1f%% "
        "vs PD-fault-unaware autonomy %.1f%% (caught) -> the deadline discriminates capable "
        "from incapable controllers with the conformal coverage guarantee.",
        rta["deadline_steps"],
        100 * rta["rma_latched_autonomy"],
        100 * rta["pd_fault_unaware_autonomy"],
    )
    logger.info("safety(exposure)-vs-autonomy(coverage) Pareto frontier:")
    for pt in p["pareto_safety_vs_autonomy"]:
        logger.info(
            "  alpha=%.3f: deadline %s steps (%ss) -> autonomy %.1f%%",
            pt["alpha"],
            pt["median_deadline_steps"],
            pt["median_deadline_s"],
            100 * pt["mean_autonomy_coverage"],
        )


if __name__ == "__main__":
    main()
