"""Build every figure for the blog and README from outputs/2024_spanish/.

    python scripts/build_figures.py

Writes PNGs to blog/figures/. Style matches the dashboard: timing-screen
dark theme, monospace annotations, one accent per meaning (purple traces,
green positive, yellow flags, red anomalies).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = Path("outputs/2024_spanish")
OUT = Path("blog/figures")
OUT.mkdir(parents=True, exist_ok=True)

BG, PANEL, GRID = "#101318", "#171B22", "#262C36"
FG, DIM = "#C9D1D9", "#7A8494"
PURPLE, GREEN, YELLOW, RED, BLUE = "#B57BFF", "#3DDC84", "#F5C542", "#F0524D", "#58A6FF"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL,
    "axes.edgecolor": GRID, "axes.labelcolor": FG,
    "text.color": FG, "xtick.color": DIM, "ytick.color": DIM,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "font.family": "monospace", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "figure.dpi": 160, "savefig.dpi": 160,
    "savefig.facecolor": BG, "savefig.bbox": "tight",
})

results = json.loads((SRC / "results.json").read_text())
backtests = sorted((json.loads(f.read_text()) for f in SRC.glob("backtest_*.json")),
                   key=lambda b: b["freeze_lap"])


def fig_architecture():
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

    def box(x, y, w, h, label, ec, body="", fc=PANEL, lw=1.4, fs=9):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw))
        ax.text(x + 1.6, y + h - 2.6, label, fontsize=fs, weight="bold", color=ec)
        if body:
            ax.text(x + 1.6, y + h - 5.4, body, fontsize=7.2, color=FG, va="top")

    def arrow(x1, y1, x2, y2, color=DIM, ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.3, ls=ls))

    box(2, 50, 26, 10, "DATA", BLUE,
        "live timing / FastF1 -> Kinesis\nOpenF1 REST (3s behind track)\nErgast-schema history (offline)")
    box(34, 50, 30, 10, "S3 TELEMETRY LAKE", BLUE,
        "Parquet per session\nGlue catalog + Athena")
    box(2, 32, 62, 13, "DETERMINISTIC LAYER  (every number computed here)", YELLOW,
        "ECS Fargate per session: fuel-corrected Theil-Sen stint fits, pit loss,\n"
        "undercut swings, anomaly flags (robust z), counterfactual pit curves,\n"
        "frozen-clock backtest  ->  payload JSON with data hash + fit CIs")
    box(70, 32, 28, 28, "GOVERNANCE", RED,
        "rules engine verdicts\n(compound, pit window,\nstint life, advisory-only)\n\n"
        "S3 Object Lock audit:\npayload hash, model ID,\nruleset version\n\n"
        "agents explain verdicts,\nnever override them")

    box(2, 12, 30, 14, "SLOW LOOP  30-120s", GREEN,
        "Strands Agents on\nAgentCore Runtime + Memory\n\nStint Analyst, Rival Watcher,\nDeg Explainer, Compliance\nGuardian")
    box(36, 12, 28, 14, "POST-SESSION  minutes", PURPLE,
        "Step Functions ->\nBedrock Batch\n\nDriver Coach,\nRace Reporter")
    box(68, 12, 30, 14, "CROSS-SEASON  weekly", YELLOW,
        "Bedrock Batch\n(24 races, one job)\n\nTrack Historian")

    box(2, 1, 96, 7, "TIMING-SCREEN DASHBOARD", DIM,
        "gap trace, counterfactual curves, undercut ledger, anomaly desk, agent briefs, prediction audit, audit log", fs=8.5)

    arrow(28, 55, 34, 55)
    arrow(49, 50, 40, 45)
    arrow(15, 50, 15, 45)
    arrow(33, 32, 33, 28); ax.text(34, 29.4, "payload contract: fitted parameters only,\nno raw telemetry crosses this line",
                                   fontsize=7, color=YELLOW, style="italic")
    arrow(17, 32, 17, 26, GREEN)
    arrow(50, 32, 50, 26, PURPLE)
    arrow(64, 38, 70, 38, RED, ls="--")
    arrow(83, 32, 83, 26, YELLOW)
    arrow(17, 12, 17, 8); arrow(50, 12, 50, 8); arrow(83, 12, 83, 8)
    ax.set_title("Latency-tiered agentic race analysis on Amazon Bedrock", loc="left", pad=12)
    fig.savefig(OUT / "fig1_architecture.png"); plt.close(fig)


def fig_gap_trace():
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    traces = results["lap_traces"]
    colors = {"VER": BLUE, "NOR": YELLOW, "LEC": RED, "HAM": "#9BE1FF",
              "RUS": GREEN, "SAI": PURPLE}
    pit_laps = {c: [] for c in traces}
    for c, t in traces.items():
        ls = np.array(t["lap_s"])
        med = np.median(ls)
        for i in range(1, len(ls) - 1):
            if ls[i] > med + 8 and ls[i + 1] < med + 4:
                pit_laps[c].append(t["laps"][i])
    for code, t in traces.items():
        ax.plot(t["laps"], t["gap_s"], color=colors[code], lw=1.5, label=code)
        for pl in pit_laps[code]:
            idx = t["laps"].index(pl)
            ax.plot(pl, t["gap_s"][idx], "o", ms=5, mfc=BG, mec=colors[code], mew=1.4)
    ax.invert_yaxis()
    ax.set_xlabel("lap"); ax.set_ylabel("gap to leader (s)")
    ax.grid(True, alpha=0.6)
    ax.legend(loc="lower left", ncol=6, frameon=False, fontsize=8)
    ax.set_title("2024 Spanish Grand Prix: gap to leader, top six finishers (markers: pit stops)", loc="left")
    fig.savefig(OUT / "fig2_gap_trace.png"); plt.close(fig)


def fig_counterfactual():
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    styles = {"VER": (BLUE, "o"), "NOR": (YELLOW, "s")}
    for cf in results["counterfactuals"]:
        code = cf["driver"]; color, marker = styles[code]
        curve = cf["curve"]
        ax.plot([c["pit_lap"] for c in curve], [c["delta_vs_actual_s"] for c in curve],
                color=color, marker=marker, ms=4, lw=1.5, label=f"{code} (actual stop L{cf['actual_pit_lap']})")
        best = min(curve, key=lambda c: c["delta_vs_actual_s"])
        ax.annotate(f"{code} optimum L{best['pit_lap']}: {best['delta_vs_actual_s']:+.2f}s",
                    xy=(best["pit_lap"], best["delta_vs_actual_s"]),
                    xytext=(best["pit_lap"] + 0.4, best["delta_vs_actual_s"] - 0.55),
                    fontsize=8, color=color)
    ax.axhline(0, color=DIM, lw=0.9, ls="--")
    ax.text(23.1, 0.12, "actual strategy baseline", fontsize=7.5, color=DIM)
    ax.set_xlabel("first pit lap"); ax.set_ylabel("modeled race time delta (s)")
    ax.grid(True, alpha=0.6); ax.legend(frameon=False, fontsize=8, loc="upper center")
    ax.set_title("Counterfactual pit lap: VER stopped 0.06s from optimum, NOR left 4.38s (finish gap: 2.22s)", loc="left")
    fig.savefig(OUT / "fig3_counterfactual.png"); plt.close(fig)


def fig_deg_anomalies():
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    fits = results["stint_fits"]
    slopes = [f["deg_slope_s_per_lap"] for f in fits]
    anoms = {(a["code"], a["stint"]): a for a in results["deg_anomalies"]}
    ax.hist(slopes, bins=28, color=GRID, edgecolor=DIM, zorder=1)
    med = float(np.median(slopes))
    ax.axvline(med, color=GREEN, lw=1.4)
    ax.text(med + 0.004, ax.get_ylim()[1] * 0.92, f"field median {med:.3f} s/lap",
            color=GREEN, fontsize=8)
    bot = anoms.get(("BOT", 2))
    for (code, stint), a in anoms.items():
        x = a["deg_slope_s_per_lap"]
        ax.axvline(x, color=RED if a is bot else YELLOW, lw=1.2, alpha=0.85, zorder=2)
    if bot:
        ax.annotate(f"BOT stint 2: {bot['deg_slope_s_per_lap']:.3f} s/lap, z={bot['robust_z']}\n"
                    f"CI [{bot['slope_ci_low']:.3f}, {bot['slope_ci_high']:.3f}] "
                    f"(teammate + adjacent stints normal)",
                    xy=(bot["deg_slope_s_per_lap"], 2.0),
                    xytext=(0.155, 7.5), fontsize=8, color=RED,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax.set_xlabel("fitted degradation slope (s/lap, fuel-corrected Theil-Sen)")
    ax.set_ylabel("stints")
    ax.set_title("62 stint fits: slope distribution with anomaly flags (|robust z| >= 1.5)", loc="left")
    ax.grid(True, axis="y", alpha=0.6)
    fig.savefig(OUT / "fig4_deg_anomalies.png"); plt.close(fig)


def fig_undercut():
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ev = sorted(results["undercut_events"], key=lambda e: e["swing_s"], reverse=True)[:12]
    labels = [f"{e['attacker'].split()[-1]} vs {e['defender'].split()[-1]} L{e['attacker_pit_lap']}/L{e['defender_pit_lap']}" for e in ev]
    swings = [e["swing_s"] for e in ev]
    colors = [GREEN if s > 0 else RED for s in swings]
    y = np.arange(len(ev))
    ax.barh(y, swings, color=colors, height=0.62, zorder=2)
    ax.set_yticks(y, labels, fontsize=7.6)
    ax.invert_yaxis()
    ax.axvline(0, color=DIM, lw=0.8)
    pl = results["pit_loss"]
    ax.axvline(pl["pit_loss_s"], color=YELLOW, lw=1.2, ls="--")
    ax.text(pl["pit_loss_s"] + 0.25, len(ev) - 1.2,
            f"pit loss {pl['pit_loss_s']}s\n(IQR {pl['pit_loss_iqr_s']}s, n={pl['n_stops_used']})",
            fontsize=7.5, color=YELLOW)
    ax.set_xlabel("realized gap swing across the pit cycle (s)")
    ax.set_title("Undercut ledger: largest realized swings of 18 measured pit battles", loc="left")
    ax.grid(True, axis="x", alpha=0.6)
    fig.savefig(OUT / "fig5_undercut_ledger.png"); plt.close(fig)


def fig_backtest():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    freezes = [b["freeze_lap"] for b in backtests]
    mae = [b["deg_summary"]["mae_s"] for b in backtests]
    bias = [b["deg_summary"]["bias_s"] for b in backtests]
    n = [b["deg_summary"]["n_stints"] for b in backtests]
    x = np.arange(len(freezes))
    ax1.bar(x - 0.18, mae, 0.36, color=PURPLE, label="MAE", zorder=2)
    ax1.bar(x + 0.18, bias, 0.36, color=YELLOW, label="bias", zorder=2)
    for i, (m, b_, cnt) in enumerate(zip(mae, bias, n)):
        ax1.text(i - 0.18, m + 0.03, f"{m:.2f}", ha="center", fontsize=8, color=PURPLE)
        ax1.text(i + 0.18, b_ + 0.03, f"{b_:+.2f}", ha="center", fontsize=8, color=YELLOW)
        ax1.text(i, -0.14, f"n={cnt}", ha="center", fontsize=7.4, color=DIM)
    ax1.set_xticks(x, [f"freeze L{f}" for f in freezes])
    ax1.set_ylabel("seconds per lap"); ax1.set_ylim(bottom=-0.2)
    ax1.legend(frameon=False, fontsize=8)
    ax1.set_title("Degradation continuation error by freeze point", loc="left")
    ax1.grid(True, axis="y", alpha=0.6)

    calls = [dict(c, freeze=b["freeze_lap"]) for b in backtests for c in b["undercut_calls"]]
    labels = [f"{c['attacker'].split()[-1]} vs {c['defender'].split()[-1]}\n(open at L{c['freeze']})" for c in calls]
    xx = np.arange(len(calls))
    ax2.bar(xx - 0.18, [c["predicted_swing_s"] for c in calls], 0.36,
            color=PURPLE, label="predicted at freeze", zorder=2)
    ax2.bar(xx + 0.18, [c["realized_swing_s"] for c in calls], 0.36,
            color=GREEN, label="realized full cycle", zorder=2)
    for i, c in enumerate(calls):
        ax2.text(i, max(c["predicted_swing_s"], c["realized_swing_s"]) + 0.35,
                 f"err {c['abs_error_s']}s", ha="center", fontsize=8, color=FG)
    ax2.set_xticks(xx, labels, fontsize=7.8)
    ax2.set_ylabel("pit-cycle swing (s)")
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_title("Open undercut calls: forecast vs outcome", loc="left")
    ax2.grid(True, axis="y", alpha=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_backtest.png"); plt.close(fig)


if __name__ == "__main__":
    fig_architecture(); fig_gap_trace(); fig_counterfactual()
    fig_deg_anomalies(); fig_undercut(); fig_backtest()
    print("figures ->", OUT, ":", sorted(p.name for p in OUT.glob("*.png")))
