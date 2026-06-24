"""Generate Paper B figures from committed evidence (vector PDFs).

  fig_calibration.pdf  coverage calibration on BOTH domains (target vs achieved)
  fig_pareto.pdf       spacecraft safety-autonomy Pareto frontier (tunable alpha)
  fig_engagement.pdf   pendulum engagement-rule autonomy retention (the pathology + cure)

Numbers come straight from the reproduced evidence, namely
evidence/program/conformal_rta.json (spacecraft, Paper A 19/19 manifest) and
evidence/conformal_cert/pendulum_domain.json (pendulum, deterministic).

Run from paper/conformal/ with `python figures.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPACE = json.loads((ROOT / "evidence/program/conformal_rta.json").read_text())
PEND = json.loads((ROOT / "evidence/conformal_cert/pendulum_domain.json").read_text())
FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 9})


def fig_calibration() -> None:
    """Achieved vs target coverage, both domains, with the y=x ideal."""
    sp = [(c["target_coverage"], c["mean_conformal_coverage"]) for c in SPACE["certificates"]]
    pe = [(1 - r["alpha"], r["mean_coverage"]) for r in PEND["certificate_coverage"]]
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.plot([0.75, 1.0], [0.75, 1.0], "--", color="gray", lw=1, label="ideal ($1-\\alpha$)")
    ax.plot(*zip(*sp, strict=True), "o-", color="#1f77b4", label="spacecraft (6-DOF)")
    ax.plot(*zip(*pe, strict=True), "s-", color="#d62728", label="inverted pendulum")
    ax.set_xlabel("target coverage $1-\\alpha$")
    ax.set_ylabel("achieved mean coverage")
    ax.set_title("Certificate calibration (two domains)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_calibration.pdf")
    plt.close(fig)


def fig_pareto() -> None:
    """Spacecraft safety-autonomy Pareto, where a longer deadline buys more autonomy."""
    pts = sorted(SPACE["pareto_safety_vs_autonomy"], key=lambda p: p["median_deadline_s"])
    x = [p["median_deadline_s"] for p in pts]
    y = [100 * p["mean_autonomy_coverage"] for p in pts]
    fig, ax = plt.subplots(figsize=(3.3, 3.0))
    ax.plot(x, y, "o-", color="#1f77b4", ms=3)
    disc = SPACE["rta_discrimination"]
    ax.set_xlabel("certified deadline (s)")
    ax.set_ylabel("autonomy retention (%)")
    ax.set_title("Safety-autonomy frontier (spacecraft)")
    auto_pct = 100 * disc["rma_latched_autonomy"]
    pd_pct = 100 * disc["pd_fault_unaware_autonomy"]
    ax.annotate(
        f"$\\alpha$={disc['operating_alpha']}, {auto_pct:.1f}% autonomous\n"
        f"(PD diverger {pd_pct:.0f}%)",
        xy=(0.5, 0.06),
        xycoords="axes fraction",
        fontsize=7.5,
        ha="center",
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_pareto.pdf")
    plt.close(fig)


def fig_engagement() -> None:
    """Pendulum autonomy retention across engagement rules (the pathology and the cure)."""
    order = [
        ("latching_simplex", "Latching\nSimplex"),
        ("heuristic_p95x1.3", "Heuristic\np95x1.3"),
        ("cp_safety_value", "CP-on-\nvalue"),
        ("conformal_recovery_deadline_ours", "Recovery\ndeadline\n(ours)"),
    ]
    b = PEND["baselines_B4"]
    labels = [lab for _, lab in order]
    auto = [100 * b[k]["autonomy_retention"] for k, _ in order]
    colors = ["#7f7f7f", "#ff7f0e", "#9467bd", "#2ca02c"]
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    bars = ax.bar(labels, auto, color=colors)
    ax.set_ylabel("autonomy retention (%)")
    ax.set_title("Engagement rules (pendulum, reversals)")
    ax.set_ylim(0, 126)
    for rect, v in zip(bars, auto, strict=True):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 2.5, f"{v:.0f}", ha="center", fontsize=8)
    ax.text(
        0.5,
        0.97,
        "all catch 100% of divergence (verified backstop)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        style="italic",
    )
    fig.tight_layout()
    fig.savefig(FIGS / "fig_engagement.pdf")
    plt.close(fig)


def main() -> None:
    fig_calibration()
    fig_pareto()
    fig_engagement()
    print(f"wrote 3 figures to {FIGS}")


if __name__ == "__main__":
    main()
