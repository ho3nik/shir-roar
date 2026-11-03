"""Figures for the paper, from the CSV/JSON files in results/."""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIGS, RESULTS, TRACE_NAME, log, tagged  # noqa: E402

# No creation date in PDF output, so regenerating an unchanged figure gives an identical file.
STABLE = {"pdf": {"CreationDate": None}}

# Categorical colours in a fixed order (distinguishable in grey scale and for colour-blind readers); ink colours for text.
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a4fd6"
AK_CHOSEN = "adaptive-k dither step0.01"   # the adaptive-k setting chosen on the arXiv tuning hour
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# Liberation Sans has Arial metrics; the IEEE graphics guidelines list Arial and Helvetica among accepted figure fonts.
plt.rcParams.update({
    "font.family": "Liberation Sans", "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "lines.linewidth": 1.4, "lines.markersize": 4, "savefig.dpi": 300, "pdf.fonttype": 42,
})


def rows(name):
    with open(os.path.join(RESULTS, name)) as f:
        return list(csv.DictReader(f))


def fnum(x):
    return float(x) if x not in (None, "", "None") else None


def static_point(trace_tag_csv, R):
    """(written %, value %) of the best constant price in hindsight, refined if e3_static.py has run."""
    ref = os.path.join(RESULTS, trace_tag_csv)
    src = rows(trace_tag_csv) if os.path.exists(ref) else [r for r in R if r["policy"] == "static price (hindsight)"]
    return (fnum(src[0]["written_share"]) * 100, fnum(src[0]["value_share"]) * 100) if src else None


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_value_vs_writes():
    R = rows(tagged("e3_days_0-1.csv"))

    def series(name):
        pts = sorted((fnum(r["written_share"]) * 100, fnum(r["value_share"]) * 100) for r in R if r["policy"] == name)
        return [p[0] for p in pts], [p[1] for p in pts]

    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    x, y = series("oracle (sweep)")
    ax.plot(x, y, color=INK2, linestyle=(0, (4, 2)), linewidth=1.0, label="offline oracle (knows the future)")
    x, y = series("ROAR (sweep)")
    ax.plot(x, y, color=BLUE, marker="o", label="ROAR (wear price)")
    x, y = series("period-quota k=2 (sweep)")
    ax.plot(x, y, color=ORANGE, marker="s", linestyle="--", label="period quota, 2nd-reference rule")
    x, y = series("stepwise-quota (sweep)")
    ax.plot(x, y, color=AQUA, marker="^", linestyle="-.", label="stepwise quota")
    ak = [(r["policy"], fnum(r["written_share"]) * 100, fnum(r["value_share"]) * 100) for r in R
          if r["policy"].startswith("admit-on-")]
    ax.scatter([p[1] for p in ak], [p[2] for p in ak], color=INK, marker="D", s=14, zorder=3,
               label="admit on k-th reference (no budget)")
    for name, xx, yy in ak:
        ax.annotate("k=" + name.split("-")[-1], (xx, yy), textcoords="offset points", xytext=(4, -9), color=INK, fontsize=6.5)
    sp = static_point(tagged("e3_static_0-1.csv"), R)
    if sp:
        ax.scatter([sp[0]], [sp[1]], color=INK2, marker="*", s=36, zorder=4, label="best constant price (hindsight)")
    ax.set_xlabel("Bytes written to flash (% of writing every evicted block)")
    ax.set_ylabel("GPU time saved by flash hits\n(% of writing every block)")
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 100)
    style(ax)
    ax.legend(loc="lower right", frameon=False, handlelength=2.2)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig_value_vs_writes_{TRACE_NAME}.{ext}"), metadata=STABLE.get(ext))
    plt.close(fig)
    log("fig_value_vs_writes")


def fig_value_vs_writes_both(column=False):
    """Two panels, both traces: side by side for a figure spanning both columns, or stacked for a
    one-column figure (column=True)."""
    if column:
        fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.45), sharex=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.75), sharey=True)
    for ax, trace, title in zip(axes, ("toolagent", "conversation"), ("Tool and agent", "Conversation")):
        R = rows(f"e3_days_0-1_{trace}.csv")

        def series(name):
            pts = sorted((fnum(r["written_share"]) * 100, fnum(r["value_share"]) * 100) for r in R if r["policy"] == name)
            return [p[0] for p in pts], [p[1] for p in pts]

        x, y = series("oracle (sweep)")
        ax.plot(x, y, color=INK2, linestyle=(0, (4, 2)), linewidth=1.0, label="offline oracle (knows the future)")
        x, y = series("ROAR (sweep)")
        ax.plot(x, y, color=BLUE, marker="o", label="ROAR (wear price)")
        x, y = series("period-quota k=2 (sweep)")
        ax.plot(x, y, color=ORANGE, marker="s", linestyle="--", label="period quota, 2nd-reference rule")
        x, y = series("stepwise-quota (sweep)")
        ax.plot(x, y, color=AQUA, marker="^", linestyle="-.", label="stepwise quota")
        ak = [(r["policy"], fnum(r["written_share"]) * 100, fnum(r["value_share"]) * 100) for r in R
              if r["policy"].startswith("admit-on-")]
        ax.scatter([p[1] for p in ak], [p[2] for p in ak], color=INK, marker="D", s=14, zorder=3,
                   label="admit on k-th reference (no budget)")
        for name, xx, yy in ak:
            ax.annotate("k=" + name.split("-")[-1], (xx, yy), textcoords="offset points", xytext=(-4, 5), ha="right",
                        color=INK, fontsize=6.5)
        sp = static_point(f"e3_static_0-1_{trace}.csv", R)
        if sp:
            ax.scatter([sp[0]], [sp[1]], color=INK2, marker="*", s=36, zorder=4, label="best constant price (hindsight)")
        e6 = os.path.join(RESULTS, f"e6_adaptive_k_0-1_{trace}.csv")
        if os.path.exists(e6):
            pts = sorted((fnum(r["written_share"]) * 100, fnum(r["value_share"]) * 100) for r in rows(os.path.basename(e6))
                         if r["policy"] == AK_CHOSEN)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=PURPLE, marker="v", linestyle=(0, (1, 1)),
                        label="adaptive $k$ (budget-tracking)")
        e10 = os.path.join(RESULTS, f"e10_load_aware_known_0-1_{trace}.csv")
        if os.path.exists(e10):
            pts = sorted((fnum(r["written_share"]) * 100, fnum(r["value_share"]) * 100) for r in rows(os.path.basename(e10))
                         if r["policy"] == "load-prop adaptive-k")
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=PURPLE, marker="v", linestyle="-", linewidth=1.0,
                        markerfacecolor="white", label="adaptive $k$, load-proportional")
        ax.set_title(title, color=INK)
        if not column or ax is axes[-1]:
            ax.set_xlabel("Bytes written to flash (% of write-all)")
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 100)
        style(ax)
        if column:
            ax.set_ylabel("GPU time saved\n(% of write-all)")
    if not column:
        axes[0].set_ylabel("GPU time saved by flash hits\n(% of write-all)")
    h, lab = axes[0].get_legend_handles_labels()
    if column:
        fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, handlelength=2.0, fontsize=5.6,
                   columnspacing=0.8, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(pad=0.2, rect=(0, 0.16, 1, 1))
    else:
        fig.legend(h, lab, loc="center right", ncol=1, frameon=False, handlelength=2.2, bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout(pad=0.3, rect=(0, 0, 0.70, 1))
    name = "fig_value_vs_writes_column" if column else "fig_value_vs_writes_both"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), metadata=STABLE.get(ext))
    plt.close(fig)
    log(name)


def fig_dynamics():
    with open(os.path.join(RESULTS, tagged("e3_days_series_0-1.json"))) as f:
        S = json.load(f)
    ref = os.path.join(RESULTS, tagged("e3_static_series_0-1.json"))
    if os.path.exists(ref):
        with open(ref) as f:
            S.update(json.load(f))
    meta = rows(tagged("e3_days_0-1.csv"))
    wa_tb = float(meta[0]["write_all_TB"])
    budget_tb = 0.10 * wa_tb
    fig, (a, b) = plt.subplots(2, 1, figsize=(3.45, 1.95), sharex=True)
    show = [("admit-on-3", INK, "-", "admit on 3rd ref. (no budget)"),
            ("static price (hindsight)", INK2, ":", "constant price (hindsight)"),
            ("period-quota k=2", ORANGE, "--", "period quota (2nd ref.)"),
            ("ROAR", BLUE, "-", "ROAR (wear price)")]
    show = [x for x in show if x[0] in S]
    for key, col, ls, lab in show:
        s = S[key]   # day 2 only, one sample per minute; cumulative values count from the start of day 1
        t = [x / 3600.0 - 24.0 for x in s["t"]]
        cum = [x / 1e12 for x in s["r0_bytes"]]
        sched = [x / 1e12 for x in s["r0_sched"]]
        base_c, base_s = cum[0], sched[0]
        per_hour = []
        for h in range(24):
            idx = [i for i, tt in enumerate(t) if h <= tt < h + 1]
            if len(idx) < 2:
                per_hour.append(0.0)
                continue
            per_hour.append((cum[idx[-1]] - cum[idx[0]]) * 60.0 / (len(idx) - 1))
        a.step([h + 1 for h in range(24)], per_hour, where="pre", color=col, linestyle=ls, label=lab)
        b.plot(t, [((c - base_c) - (sc - base_s)) / budget_tb * 100 for c, sc in zip(cum, sched)], color=col,
               linestyle=ls, label=lab)
    a.axhline(budget_tb / 24, color=INK2, linewidth=0.8, linestyle=":")
    a.annotate("budget rate", (0.6, budget_tb / 24), textcoords="offset points", xytext=(0, 3), color=INK2, fontsize=6.5)
    a.set_ylabel("Writes (TB/h)")
    b.axhline(0, color=INK2, linewidth=0.8, linestyle=":")
    b.set_ylabel("Lead (%)")
    b.set_xlabel("Hour of day 2 (load peaks at 14:00)")
    b.set_xlim(0, 24)
    b.set_xticks(range(0, 25, 4))
    for ax in (a, b):
        style(ax)
    h, lab = a.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.54, 1.0), columnspacing=1.0,
               handlelength=2.0, fontsize=6.3)
    fig.tight_layout(pad=0.5, h_pad=0.4, rect=(0.02, 0, 1, 0.851))
    fig.align_ylabels((a, b))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig_dynamics_{TRACE_NAME}.{ext}"), metadata=STABLE.get(ext))
    plt.close(fig)
    log("fig_dynamics")


def fig_hour_sweep():
    R = [r for r in rows(tagged("e2_hour_sweep.csv"))]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, cap in zip(axes, ("0", "6000")):
        sub = [r for r in R if r["hbm_blocks"] == cap]
        for name, col, mk, ls in (("oracle", INK2, None, (0, (4, 2))), ("ROAR (trained)", BLUE, "o", "-"),
                                  ("ROAR (cold)", BLUE, "o", ":"), ("period-quota k=2", ORANGE, "s", "--"),
                                  ("stepwise-quota", AQUA, "^", "-.")):
            pts = sorted((fnum(r["written_share"]) * 100, fnum(r["value_share"]) * 100) for r in sub if r["policy"] == name)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, linestyle=ls, label=name,
                    markersize=3.5, linewidth=1.1)
        ak = [(r["policy"], fnum(r["written_share"]) * 100, fnum(r["value_share"]) * 100) for r in sub
              if r["policy"].startswith("admit-on-")]
        ax.scatter([p[1] for p in ak], [p[2] for p in ak], color=INK, marker="D", s=12, zorder=3, label="admit on k-th ref.")
        for name, xx, yy in ak:
            ax.annotate("k=" + name.split("-")[-1], (xx, yy), textcoords="offset points", xytext=(3, -8), fontsize=6)
        ax.set_title("no HBM tier" if cap == "0" else "1 TB HBM prefix cache (6,000 blocks)", color=INK)
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Bytes written (% of write-all)")
        style(ax)
    axes[0].set_ylabel("GPU time saved by flash\n(% of write-all)")
    axes[1].legend(loc="lower right", frameon=False)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"fig_hour_sweep_{TRACE_NAME}.{ext}"), metadata=STABLE.get(ext))
    plt.close(fig)
    log("fig_hour_sweep")


if __name__ == "__main__":
    for fn in (fig_value_vs_writes, fig_value_vs_writes_both, lambda: fig_value_vs_writes_both(column=True),
               fig_dynamics, fig_hour_sweep):
        try:
            fn()
        except FileNotFoundError as e:
            log("skipped", fn.__name__, e)
