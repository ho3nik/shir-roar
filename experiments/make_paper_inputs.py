"""Generate the number macros (numbers.tex), the tables and numbers.json that the paper reads, from the
CSV files in results/. The prose in results_text.tex is written against these macros, so every number
in the paper traces back to a file here.
Usage: python make_paper_inputs.py PAPER_DIR"""
import csv
import glob
import json
import os
import re
import statistics as st
import sys

from common import HBM_BLOCKS, RESULTS, TRACES, log
from shirroar import hbm_pass, load_mooncake
from shirroar.hbm import EVICT

OUT = sys.argv[1]
POLICIES = ["ROAR", "ROAR-refs", "ROAR-refs-depth", "adaptive-k", "load-prop adaptive-k", "yesterday adaptive-k", "period-quota k=2",
            "load-prop quota", "stepwise-quota", "period-quota k=1", "day-1 price", "static price (hindsight)", "oracle"]
TABLE_POLICIES = ["ROAR", "ROAR-refs", "adaptive-k", "load-prop adaptive-k", "period-quota k=2", "load-prop quota",
                  "stepwise-quota", "day-1 price", "oracle"]   # best constant price in hindsight: Fig. 1 (star)
# e6_adaptive_k.py rows that join the main results (renamed); everything else in those files is a duplicate of
# e3_days.py or part of the adaptive-k grid, which is summarized separately
AK_CHOSEN = "adaptive-k dither step0.01"     # chosen on the arXiv tuning hour (e6 --grid, SHIR_TRACE=arxiv)
E6_KEEP = {AK_CHOSEN: "adaptive-k", "ROAR-refs": "ROAR-refs", "ROAR-refs-depth": "ROAR-refs-depth", "day-1 price": "day-1 price"}
BUDGETS = (0.05, 0.10, 0.20)


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def F(x):
    return float(x) if x not in (None, "", "None") else None


def gap_stats(trace):
    s = hbm_pass(load_mooncake(TRACES[trace]), HBM_BLOCKS)
    last, gaps = {}, []
    for k in range(len(s.kind)):
        b = s.blk[k]
        if s.kind[k] == EVICT:
            last[b] = s.t[k]
        elif not s.hbm_hit[k] and b in last:
            gaps.append(s.t[k] - last.pop(b))
    gaps.sort()
    n = len(gaps)
    return {"n": n, "median_s": gaps[n // 2], "p90_s": gaps[int(0.9 * (n - 1))], "p99_s": gaps[int(0.99 * (n - 1))],
            "share_under_26min": sum(g < 26.3 * 60 for g in gaps) / n}


def leaf_reuse(trace):
    """Mean later HBM-missing references after evictions of once-referenced blocks, split by whether
    any request had continued past the block at eviction time (whole remaining trace)."""
    import bisect
    from collections import defaultdict
    s = hbm_pass(load_mooncake(TRACES[trace]), HBM_BLOCKS)
    miss = defaultdict(list)
    for k in range(len(s.kind)):
        if s.kind[k] != EVICT and not s.hbm_hit[k]:
            miss[s.blk[k]].append(k)
    refs = defaultdict(int)
    acc = {0: [0, 0], 1: [0, 0]}
    for k in range(len(s.kind)):
        b = s.blk[k]
        if s.kind[k] != EVICT:
            refs[b] += 1
            continue
        if refs[b] != 1:
            continue
        lst = miss[b]
        fut = len(lst) - bisect.bisect_right(lst, k)
        a = acc[s.has_child[k]]
        a[0] += fut
        a[1] += 1
    return {"no_continuation": acc[0][0] / max(1, acc[0][1]), "continued": acc[1][0] / max(1, acc[1][1]),
            "n_no_continuation": acc[0][1], "n_continued": acc[1][1]}


def budget_table(trace="toolagent"):
    R = rows(os.path.join(RESULTS, f"e1_endurance_{trace}.csv"))
    pick = {(r["cell"], int(r["stacks_per_node"]), r["model"]): r for r in R}
    names = [("SLC 100K, 512 GB", "SLC (100K)"), ("pSLC 20K, 512 GB", "pSLC (20K)"), ("TLC 3K, 1.5 TB", "TLC (3K)"),
             ("QLC 1K, 2 TB", "QLC (1K)")]
    m = "Llama-3.1-70B"
    lines = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
             r"Cell (cycles) & \multicolumn{2}{c}{Budget share} & \multicolumn{2}{c}{Write-all life (y)} & Residence \\",
             r" & 8 st. & 32 st. & 8 st. & 32 st. & at budget \\", r"\midrule"]
    for key, lab in names:
        a, b = pick[(key, 8, m)], pick[(key, 32, m)]
        res_h = F(a["fifo_residence_h_at_budget"])
        res = f"{res_h * 60:.0f} min" if res_h < 1 else f"{res_h:.1f} h"
        lines.append(f"{lab} & {100 * F(a['budget_share_of_write_all']):.1f}\\% & {100 * F(b['budget_share_of_write_all']):.1f}\\% & "
                     f"{F(a['write_all_lifetime_years']):.2f} & {F(b['write_all_lifetime_years']):.2f} & {res} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT, "table_budget.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    tlcqlc = [F(pick[(k, n, m)]["budget_share_of_write_all"]) for k in ("TLC 3K, 1.5 TB", "QLC 1K, 2 TB") for n in (8, 32)]
    out = {"tlcqlc_min": min(tlcqlc), "tlcqlc_max": max(tlcqlc),
           "wa_day_tb": F(pick[("TLC 3K, 1.5 TB", 8, m)]["write_all_TB_per_day"]),
           "slc8_life": F(pick[("SLC 100K, 512 GB", 8, m)]["write_all_lifetime_years"]),
           "rows": [[lab] + [F(pick[(key, n, mm)][c]) for n in (8, 32) for mm in (m, "DeepSeek-V3")
                             for c in ("budget_share_of_write_all",)] + [F(pick[(key, 8, m)]["fifo_residence_h_at_budget"])]
                    for key, lab in names],
           "full": R}
    return out


def day_summary(trace):
    files = sorted(glob.glob(os.path.join(RESULTS, f"e3_days_*-*_{trace}.csv")))
    R = [r for p in files for r in rows(p)]
    # the refined constant price (e3_static.py) replaces the coarse one from e3_days.py
    ref = os.path.join(RESULTS, f"e3_static_0-1_{trace}.csv")
    if os.path.exists(ref):
        R = [r for r in R if not (r["policy"] == "static price (hindsight)" and r["seeds"] == "0-1")] + rows(ref)
    for p in sorted(glob.glob(os.path.join(RESULTS, f"e6_adaptive_k_*-*_{trace}.csv"))):
        for r in rows(p):
            if r["policy"] in E6_KEEP:
                R.append(dict(r, policy=E6_KEEP[r["policy"]]))
    for p in sorted(glob.glob(os.path.join(RESULTS, f"e10_load_aware_known_*-*_{trace}.csv"))):
        R += rows(p)
    pairs = sorted({r["seeds"] for r in R})
    S = {}
    for b in BUDGETS:
        S[b] = {}
        for p in POLICIES:
            sel = [r for r in R if r["policy"] == p and r["budget_share"] and abs(F(r["budget_share"]) - b) < 1e-9]
            if sel:
                S[b][p] = {"value": [F(r["value_share"]) for r in sel], "written": [F(r["written_over_budget"]) for r in sel],
                           "ahead": [F(r["max_ahead_share_of_daily_budget"]) for r in sel], "pairs": [r["seeds"] for r in sel]}
    U = {}
    for k in (2, 3, 4):
        sel = [r for r in R if r["policy"] == f"admit-on-{k}"]
        U[k] = {"value": [F(r["value_share"]) for r in sel], "written": [F(r["written_share"]) for r in sel],
                "by_pair": {r["seeds"]: (F(r["written_share"]), F(r["value_share"])) for r in sel}}
    meta = {"pairs": pairs,
            "wa_flash_gpu_h": st.mean(F(r["write_all_flash_saved_s"]) for r in R if r["policy"] == "ROAR") / 3600.0,
            "hbm_gpu_h": st.mean(F(r["hbm_saved_s"]) for r in R if r["policy"] == "ROAR") / 3600.0}
    sweep = {}
    for p in ("ROAR (sweep)", "period-quota k=2 (sweep)", "stepwise-quota (sweep)", "oracle (sweep)"):
        sweep[p] = sorted((F(r["budget_share"]), F(r["written_share"]), F(r["value_share"])) for r in R if r["policy"] == p)
    # does ROAR keep more than both quotas on every day pair and budget?
    wins, cells, margins = 0, 0, {b: [] for b in BUDGETS}
    for b in BUDGETS:
        for pair in pairs:
            def val(p):
                x = [F(r["value_share"]) for r in R if r["policy"] == p and r["seeds"] == pair and r["budget_share"]
                     and abs(F(r["budget_share"]) - b) < 1e-9]
                return x[0] if x else None
            ro, q2, sq = val("ROAR"), val("period-quota k=2"), val("stepwise-quota")
            if None in (ro, q2, sq):
                continue
            cells += 1
            wins += ro > max(q2, sq)
            margins[b].append(ro - max(q2, sq))
    meta["roar_wins"] = wins
    meta["cells"] = cells
    meta["margins"] = margins
    bounds = [F(r["bound_intervals"]) for r in R if r["policy"] == "ROAR" and r.get("bound_intervals")]
    meta["bound_days"] = max(bounds) * 10.0 / 86400.0 if bounds else None
    return {"S": S, "U": U, "meta": meta, "sweep": sweep}


QUIET, BUSY = (0, 1, 2, 3, 4), (12, 13, 14, 15, 16)   # hours of the day around the load minimum (02:00) and peak (14:00)


def hourly(trace):
    """Per-hour flash bytes and GPU time saved on day 2 of day pair 0-1 at the 10% budget, from the
    recorded series (cumulative counters sampled once a minute). With per-hour block names, every hit
    on a block falls in the hour it was written, so value per byte by hour is meaningful."""
    path = os.path.join(RESULTS, f"e3_days_series_0-1_{trace}.json")
    if not os.path.exists(path):
        return None
    D = json.load(open(path))
    ref = os.path.join(RESULTS, f"e3_static_series_0-1_{trace}.json")
    if os.path.exists(ref):
        D.update(json.load(open(ref)))
    out = {}
    for name, ser in D.items():
        t, v, w = ser["t"], ser["saved_flash_s"], ser["r0_bytes"]
        day0 = 86400.0
        last_v, last_w = v[0], w[0]
        hv, hw = [0.0] * 24, [0.0] * 24
        for i in range(1, len(t)):
            h = min(23, int((t[i] - day0 - 1e-6) // 3600))
            hv[h] += v[i] - v[i - 1]
            hw[h] += w[i] - w[i - 1]
        tv, tw = sum(hv), sum(hw)
        qv, qw = sum(hv[h] for h in QUIET), sum(hw[h] for h in QUIET)
        bv, bw = sum(hv[h] for h in BUSY), sum(hw[h] for h in BUSY)
        out[name] = {"value_by_hour_s": hv, "bytes_by_hour": hw,
                     "quiet_bytes_share": qw / tw if tw else None, "busy_bytes_share": bw / tw if tw else None,
                     "quiet_value_per_byte_vs_busy": ((qv / qw) / (bv / bw)) if qw and bw and bv else None}
    return out


def sensitivity(trace="toolagent"):
    path = os.path.join(RESULTS, f"e5_sensitivity_{trace}.csv")
    return rows(path) if os.path.exists(path) else None


def cell_modes(trace="toolagent"):
    path = os.path.join(RESULTS, f"e4_cell_modes_{trace}.csv")
    return rows(path) if os.path.exists(path) else None


def mean(x):
    return st.mean(x) if x else float("nan")


def interp(points, x):
    """Value at x bytes on a policy's budget sweep, piecewise linear through (0, 0). Beyond the largest
    amount the policy wrote it keeps its last value: a rule that runs out of eligible blocks cannot use
    more budget."""
    pts = [(0.0, 0.0)] + sorted((w, v) for _, w, v in points)
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y1
    return float("nan")


MACRO = {"ROAR": "Roar", "period-quota k=2": "QuotaTwo", "stepwise-quota": "Stepwise", "period-quota k=1": "QuotaOne",
         "static price (hindsight)": "Static", "oracle": "Oracle", "ROAR-refs": "RoarRefs", "ROAR-refs-depth": "RoarRefsDepth",
         "adaptive-k": "AdaptK", "day-1 price": "DayOne", "load-prop adaptive-k": "AdaptKLoad",
         "load-prop quota": "QuotaLoad", "yesterday adaptive-k": "AdaptKYest", "yesterday quota": "QuotaYest"}


def macro_name(s):
    return MACRO[s]


def main():
    os.makedirs(OUT, exist_ok=True)
    M = {}
    g_ta, g_cv = gap_stats("toolagent"), gap_stats("conversation")
    bt = budget_table("toolagent")
    M["gapsInside"] = f"{100 * g_ta['share_under_26min']:.1f}\\%"
    M["gapsInsideConv"] = f"{100 * g_cv['share_under_26min']:.1f}\\%"
    M["gapMedian"] = f"{g_ta['median_s']:.0f}\\,s"
    M["gapMedianConv"] = f"{g_cv['median_s']:.0f}\\,s"
    M["budgetRangeTLCQLC"] = f"{100 * bt['tlcqlc_min']:.1f}--{100 * bt['tlcqlc_max']:.0f}\\%"
    M["waDayTB"] = f"{bt['wa_day_tb']:.0f}"
    M["slcLife"] = f"{bt['slc8_life']:.1f}"
    from common import ROAR_DEFAULT
    M["chosenEta"] = f"$\\eta={ROAR_DEFAULT['eta']:g}$"
    lr = leaf_reuse("toolagent")
    M["leafReuse"] = f"{lr['no_continuation']:.3f}"
    M["contReuse"] = f"{lr['continued']:.2f}"
    D = {t: day_summary(t) for t in ("toolagent", "conversation")}
    FR = {}
    WINS = {}
    for t, tag in (("toolagent", "TA"), ("conversation", "CV")):
        S, U, meta = D[t]["S"], D[t]["U"], D[t]["meta"]
        M[f"nPairs{tag}"] = str(len(meta["pairs"]))
        M[f"waFlashGPUh{tag}"] = f"{meta['wa_flash_gpu_h']:.1f}"
        M[f"hbmGPUh{tag}"] = f"{meta['hbm_gpu_h']:.1f}"
        for b in BUDGETS:
            bn = ['five', 'ten', 'twenty'][BUDGETS.index(b)]
            for p in POLICIES:
                if p in S[b]:
                    M[f"v{tag}{macro_name(p)}B{bn}"] = f"{100 * mean(S[b][p]['value']):.1f}"
                    if "0-1" in S[b][p]["pairs"]:
                        i = S[b][p]["pairs"].index("0-1")
                        M[f"first{tag}{macro_name(p)}B{bn}"] = f"{100 * S[b][p]['value'][i]:.1f}"
        for k in (2, 3, 4):
            if U[k]["value"]:
                M[f"admit{['two', 'three', 'four'][k - 2]}W{tag}"] = f"{100 * mean(U[k]['written']):.1f}"
                M[f"admit{['two', 'three', 'four'][k - 2]}V{tag}"] = f"{100 * mean(U[k]['value']):.1f}"
        sw = D[t]["sweep"]
        FR[t] = {}
        for k in (2, 3, 4):
            kw_ = ['two', 'three', 'four'][k - 2]
            if "0-1" not in U[k].get("by_pair", {}):
                continue
            w, v = U[k]["by_pair"]["0-1"]
            M[f"admit{kw_}WFirst{tag}"] = f"{100 * w:.1f}"
            M[f"admit{kw_}VFirst{tag}"] = f"{100 * v:.1f}"
            FR[t][k] = {"written": w, "value": v}
            for p, mac in (("ROAR (sweep)", "roarAt"), ("period-quota k=2 (sweep)", "quotaAt"),
                           ("stepwise-quota (sweep)", "stepAt"), ("oracle (sweep)", "oracleAt")):
                hit = [e for e in sw.get(p, []) if abs(e[0] - round(w, 4)) < 1e-9]
                if hit:
                    M[f"{mac}{kw_}{tag}"] = f"{100 * hit[0][2]:.1f}"
                    M[f"{mac}W{kw_}{tag}"] = f"{100 * hit[0][1]:.1f}"
                    FR[t][k][p] = {"written": hit[0][1], "value": hit[0][2]}
        M[f"roarWins{tag}"] = f"{meta['roar_wins']} of {meta['cells']}"
        WINS[t] = (meta["roar_wins"], meta["cells"])
        mt = meta["margins"][0.10]
        if mt:
            M[f"marginTenMin{tag}"] = f"{100 * min(mt):.1f}"
            M[f"marginTenMax{tag}"] = f"{100 * max(mt):.1f}"
        if meta.get("bound_days"):
            M[f"boundDays{tag}"] = f"{meta['bound_days']:.1f}"
        if U[3]["written"]:
            w3 = mean(U[3]["written"])
            M[f"overFive{tag}"] = f"{100 * (w3 / 0.05 - 1):.0f}\\%" if w3 > 0.05 else "0\\%"
            M[f"lifeFive{tag}"] = f"{5 * 0.05 / w3:.1f}"
            M[f"unusedTen{tag}"] = f"{100 * (1 - w3 / 0.10):.0f}\\%" if w3 < 0.10 else "none"
        if "ROAR" in S[0.10]:
            ahead = max(max(S[b]["ROAR"]["ahead"]) for b in BUDGETS if "ROAR" in S[b])
            written = [x for b in BUDGETS if "ROAR" in S[b] for x in S[b]["ROAR"]["written"]]
            M[f"roarAhead{tag}"] = f"{100 * ahead:.1f}\\%"
            M[f"roarWrittenMin{tag}"] = f"{min(written):.3f}"
            M[f"roarWrittenMax{tag}"] = f"{max(written):.3f}"
    # --- value at equal day-2 bytes (day pair 0-1 sweeps) ----------------------------------------
    EQ = {}
    for t, tag in (("toolagent", "TA"), ("conversation", "CV")):
        sw = D[t]["sweep"]
        EQ[t] = {}
        for b in BUDGETS:
            bn = ['five', 'ten', 'twenty'][BUDGETS.index(b)]
            EQ[t][b] = {}
            for p, mac in (("ROAR (sweep)", "Roar"), ("period-quota k=2 (sweep)", "QuotaTwo"), ("stepwise-quota (sweep)", "Stepwise"),
                           ("oracle (sweep)", "Oracle")):
                if sw.get(p):
                    val = interp(sw[p], b)
                    EQ[t][b][p] = val
                    M[f"eq{tag}{mac}B{bn}"] = f"{100 * val:.1f}"
            e = EQ[t][b]
            if "ROAR (sweep)" in e and "period-quota k=2 (sweep)" in e:
                M[f"eqMargin{tag}B{bn}"] = f"{100 * (e['ROAR (sweep)'] - max(e['period-quota k=2 (sweep)'], e.get('stepwise-quota (sweep)', 0))):.1f}"
    # --- constant price in hindsight against ROAR, same days (pair 0-1, 10%) -----------------------
    def first(t, p, b=0.10):
        e = D[t]["S"][b].get(p)
        if not e or "0-1" not in e["pairs"]:
            return None
        return e["value"][e["pairs"].index("0-1")]

    # compare at equal bytes: the constant price cannot hit its budget exactly (whole classes of blocks
    # cross its threshold together), so set its value against ROAR's sweep at the bytes it wrote
    STATIC = {}
    for t, tag in (("toolagent", "TA"), ("conversation", "CV")):
        e = D[t]["S"][0.10].get("static price (hindsight)")
        sw = D[t]["sweep"].get("ROAR (sweep)")
        if not e or "0-1" not in e["pairs"] or not sw:
            continue
        i = e["pairs"].index("0-1")
        val, spend = e["value"][i], e["written"][i]
        ws = spend * 0.10
        roar_eq = interp(sw, ws)
        STATIC[t] = {"value": val, "written_over_budget": spend, "roar_at_same_bytes": roar_eq, "diff": val - roar_eq}
        M[f"staticSpend{tag}"] = f"{spend:.2f}"
        M[f"staticEqDiff{tag}"] = f"{100 * abs(val - roar_eq):.1f}"
    if len(STATIC) == 2:
        diffs = [STATIC[t]["diff"] for t in ("toolagent", "conversation")]
        if min(diffs) >= 0.005:
            M["staticVerdict"] = (f"at the same bytes it keeps {100 * diffs[0]:.1f} and {100 * diffs[1]:.1f} points more than ROAR, "
                                  f"which is what knowing the right price in advance is worth")
        elif max(diffs) <= -0.005:
            M["staticVerdict"] = (f"at the same bytes ROAR keeps {100 * -diffs[0]:.1f} and {100 * -diffs[1]:.1f} points more, "
                                  f"because its price follows the load")
        else:
            M["staticVerdict"] = "at the same bytes the two are within half a point"
    # --- wording for the 20% comparison at equal bytes ------------------------------------------
    m20 = [EQ[t][0.20].get("ROAR (sweep)", float("nan")) - max(EQ[t][0.20].get("period-quota k=2 (sweep)", float("nan")),
                                                                EQ[t][0.20].get("stepwise-quota (sweep)", float("-inf")))
           for t in ("toolagent", "conversation")]
    if all(x == x for x in m20):
        def lead(x):
            x100 = 100 * x
            return f"leads by {x100:.1f}" if x100 >= 0.05 else (f"trails by {-x100:.1f}" if x100 <= -0.05 else "ties")
        M["eqTwentyPhrase"] = f"it {lead(m20[0])} points on tool and agent and {lead(m20[1])} on conversation"
        M["eqWithinTwenty"] = f"{100 * max(abs(m20[0]), abs(m20[1])):.1f}"
    # --- where ROAR's lead comes from (check_handover.py, conversation) ----------------------------
    hpath = os.path.join(RESULTS, "check_handover_conversation.json")
    HAND = None
    if os.path.exists(hpath):
        HAND = json.load(open(hpath))
        d2 = [x["price"] for x in HAND["samples"] if x["hour"] > 24.0]
        if d2:
            import math as _m
            gm = _m.exp(sum(_m.log(x) for x in d2) / len(d2))
            M["handoverRatio"] = f"{HAND['price_at_handover'] / gm:.2f}"
            day1_end = [x["lead_days_of_budget"] for x in HAND["samples"] if abs(x["hour"] - 24.0) < 1e-6]
            if day1_end:
                M["leadDayOneEnd"] = f"{100 * day1_end[0]:.0f}\\%"
    # --- day-2 spend relative to budget, by policy ----------------------------------------------
    for p, mac in (("ROAR", "Roar"), ("period-quota k=2", "QuotaTwo"), ("stepwise-quota", "Stepwise"), ("adaptive-k", "AdaptK"),
                   ("ROAR-refs", "RoarRefs")):
        xs = [x for t in D for b in BUDGETS if p in D[t]["S"][b] for x in D[t]["S"][b][p]["written"] if x is not None]
        if xs:
            M[f"spendLow{mac}"] = f"{min(xs):.2f}"
            M[f"spendHigh{mac}"] = f"{max(xs):.2f}"
            lo_s, hi_s = f"{min(xs):.2f}", f"{max(xs):.2f}"
            M[f"spend{mac}"] = lo_s if lo_s == hi_s else f"{lo_s}--{hi_s}"
    # --- how closely each budgeted policy spent its day-2 budget ----------------------------------
    wr = [x for t in D for b in BUDGETS for p, e in D[t]["S"][b].items() if p != "oracle" for x in e["written"] if x is not None]
    if wr:
        M["writtenLow"] = f"{min(wr):.2f}"
        M["writtenHigh"] = f"{max(wr):.2f}"
    # --- tuning trace: gains below the chosen one ----------------------------------------------
    tune = []
    for f_ in ("e5d_tune_arxiv_high.csv", "e5e_slower_gain_arxiv.csv"):
        pth = os.path.join(RESULTS, f_)
        if os.path.exists(pth):
            tune += [r for r in rows(pth) if r.get("policy") == "ROAR" and abs(F(r["budget_share"]) - 0.10) < 1e-9
                     and not F(r.get("soft") or 0)]
    TUNE = sorted(((F(r["eta"]), F(r["value_share"]), F(r["max_ahead_share_of_daily_budget"]), F(r["bound_intervals"]),
                    F(r["written_over_budget"])) for r in tune), reverse=True)
    for eta_, val_, ah_, bd_, wob_ in TUNE:
        key = {0.01: "TuneA", 0.003: "TuneB", 0.001: "TuneC", 0.0003: "TuneD", 0.0001: "TuneE", 0.00003: "TuneF"}.get(eta_)
        if key:
            M[f"val{key}"] = f"{100 * val_:.1f}"
            M[f"spend{key}"] = f"{wob_:.2f}"
            bdd = bd_ * 10.0 / 86400.0
            M[f"bound{key}"] = f"{bdd:.2f}" if bdd < 1 else f"{bdd:.1f}"
    if len(WINS) == 2:
        if all(w == c and c > 0 for w, c in WINS.values()):
            M["roarWinsSentence"] = "ROAR kept more than both quotas on every seed and budget."
        else:
            M["roarWinsSentence"] = (f"ROAR kept more than both quotas in {WINS['toolagent'][0]} of {WINS['toolagent'][1]} (tool and agent) "
                                     f"and {WINS['conversation'][0]} of {WINS['conversation'][1]} (conversation) combinations of seed "
                                     f"and budget.")
    # --- frequency rules at equal bytes (day pair 0-1) -------------------------------------------
    def fr(t, k, p=None):
        e = FR.get(t, {}).get(k)
        if e is None:
            return None
        return e["value"] if p is None else (e.get(p) or {}).get("value")

    if all(fr(t, 3) is not None and fr(t, 3, "ROAR (sweep)") is not None for t in ("toolagent", "conversation")):
        d = [fr(t, 3, "ROAR (sweep)") - fr(t, 3) for t in ("toolagent", "conversation")]
        if min(d) >= 0.01:
            verdict = "so ROAR keeps more for about the same bytes; beyond that, the rule's write rate"
        elif max(d) <= -0.01:
            verdict = "so at its own write rate the rule keeps more than ROAR; its weakness is that this rate"
        else:
            verdict = "so the two are within a point for about the same bytes; the rule's weakness is that its write rate"
        M["freqVerdict"] = verdict
    # --- hourly analysis -----------------------------------------------------------------------
    H = {t: hourly(t) for t in ("toolagent", "conversation")}
    for t, tag in (("toolagent", "TA"), ("conversation", "CV")):
        h = H[t]
        if not h:
            continue
        for name, mac in (("period-quota k=2", "Quota"), ("stepwise-quota", "Step"), ("ROAR", "Roar"),
                          ("admit-on-3", "AdmitThree"), ("static price (hindsight)", "Static")):
            e = h.get(name)
            if not e or e["quiet_bytes_share"] is None:
                continue
            M[f"quiet{mac}Bytes{tag}"] = f"{100 * e['quiet_bytes_share']:.0f}\\%"
            M[f"busy{mac}Bytes{tag}"] = f"{100 * e['busy_bytes_share']:.0f}\\%"
            if e["quiet_value_per_byte_vs_busy"] is not None:
                M[f"quiet{mac}ValueRatio{tag}"] = f"{e['quiet_value_per_byte_vs_busy']:.2f}"
    # --- cell modes (same dies) ----------------------------------------------------------------
    C = cell_modes("toolagent")
    if C:
        def gh(n, cfg, pol="ROAR", col="flash_saved_gpu_h_per_day"):
            r = [x for x in C if int(x["stacks"]) == n and x["config"] == cfg and x["policy"] == pol]
            return F(r[0][col]) if r and r[0].get(col) not in (None, "") else float("nan")

        losses = [abs(F(x["capacity_loss_share"])) for x in C if x["policy"] == "ROAR" and x.get("capacity_loss_share")]
        loss_max = max(losses) if losses else float("nan")
        cap_txt = (f"Unlimited capacity changed these by at most {max(0.1, 100 * loss_max):.1f}\\%"
                   + (", as Section~\\ref{sec:wear} predicts." if loss_max < 0.02 else "."))
        M["cellModeSentence"] = (
            f"On the same dies, pSLC instead of TLC raised ROAR's savings from {gh(8, 'all TLC'):.1f} to {gh(8, 'all pSLC'):.1f} "
            f"GPU-hours per day with 8 stacks ({gh(32, 'all TLC'):.1f} to {gh(32, 'all pSLC'):.1f} with 32; an even split: "
            f"{gh(8, 'half pSLC, half TLC'):.1f} and {gh(32, 'half pSLC, half TLC'):.1f}), assuming 20K pSLC cycles. {cap_txt}")
        numbers_cell = C
    else:
        M["cellModeSentence"] = ""
        numbers_cell = None
    # --- sensitivity (tool and agent, day pair 0-1, 10% budget) ---------------------------------
    E5 = sensitivity("toolagent")
    if E5:
        var = {r["variant"]: F(r["value_share"]) for r in E5 if r["study"] == "ROAR variants"}
        # "tick 60 s" keeps the gain per interval, so its price moves six times more slowly per second and
        # it overspends; "tick 60 s, gains x6" is the like-for-like interval change. The parent check is
        # reported on its own because dropping it is not a small change.
        small_keys = ("tick 60 s, gains x6", "horizon 300 s", "horizon 1800 s", "no depth weighting", "no warm-up gain",
                      "soft band 0.3")
        others = {k: var[k] for k in small_keys if k in var}
        if "default" in var:
            M["sensDefault"] = f"{100 * var['default']:.1f}"
            for k, mac in (("soft band 0.3", "sensSoft"), ("tick 60 s", "sensTick"), ("tick 60 s, gains x6", "sensTickScaled"),
                           ("horizon 300 s", "sensHshort"), ("horizon 1800 s", "sensHlong"), ("no depth weighting", "sensNoDepth"),
                           ("no parent check", "sensNoParent"), ("no warm-up gain", "sensNoWarm")):
                if k in var:
                    M[mac] = f"{100 * var[k]:.1f}"
            if others:
                M["sensMin"] = f"{100 * min(others.values()):.1f}"
                M["sensMax"] = f"{100 * max(others.values()):.1f}"
        robust = []
        for study in ("HBM 1500 blocks", "HBM 24000 blocks", "DeepSeek-V3 (same HBM bytes)"):
            st_ = {r["variant"]: F(r["value_share"]) for r in E5 if r["study"] == study}
            if {"ROAR", "period quota k=2", "stepwise quota"} <= set(st_):
                robust.append(st_["ROAR"] > max(st_["period quota k=2"], st_["stepwise quota"]))
        if "default" in var and others and robust and "no parent check" in var:
            lo_, hi_ = min(others.values()), max(others.values())
            M["sensSentence"] = (
                f"Six other controller settings kept {100 * lo_:.1f}--{100 * hi_:.1f}\\% (default {100 * var['default']:.1f}\\%, tool and "
                f"agent, 10\\%); without the parent check ROAR kept {100 * var['no parent check']:.1f}\\%, as a block whose parent is in "
                f"neither tier is useless until the parent is cached again. "
                + ("With a 4$\\times$ smaller or larger HBM cache, or DeepSeek-V3's smaller KV, ROAR still beat both quotas."
                   if all(robust) else
                   f"With a 4$\\times$ smaller or larger HBM cache, or DeepSeek-V3's smaller KV, ROAR beat both quotas in {sum(robust)} of "
                   f"{len(robust)} cases."))
        for study, mac in (("HBM 1500 blocks", "Small"), ("HBM 24000 blocks", "Large"), ("DeepSeek-V3 (same HBM bytes)", "DS")):
            st_ = {r["variant"]: F(r["value_share"]) for r in E5 if r["study"] == study}
            for k, m2 in (("ROAR", "Roar"), ("period quota k=2", "Quota"), ("stepwise quota", "Step"), ("admit-on-3", "AdmitThree"),
                          ("oracle", "Oracle")):
                if k in st_:
                    M[f"sens{mac}{m2}"] = f"{100 * st_[k]:.1f}"
    import paper_macros
    paper_macros.add(M, D)
    paper_macros.unknown(M, D, OUT)
    # Any number macro the text uses but the results do not (yet) provide prints as a bold ?? instead
    # of stopping the build; the list of missing names is printed so they can be found and fixed.
    pat = re.compile(r"^(v(TA|CV)|first|eq|admit|roarAt|roarWins|roarAhead|roarWritten|quiet|busy|sens|spend|bound|valTune|gap|"
                     r"budgetRange|waDay|waFlash|hbmGPU|slcLife|leafReuse|contReuse|chosenEta|nPairs|overFive|lifeFive|unusedTen|"
                     r"written(Low|High)|freqVerdict|staticVerdict|cellModeSentence|marginTen|quotaAt|stepAt|oracleAt|eqTwenty|eqWithin|staticSpend|staticEq|handover|leadDay|roarWinsSentence|"
                     r"prefill|waShare|hbmShare|tlcEight|pslcEight|quotaTlc|pslcBreak|pslcRes|peakWrite|akBest|akGrid|ab[A-Z]|marginAk|marginRefs|spreadRoar|maxPairSpread|xfer|lg[A-Z]|peakWriteShare|spendDayOne|dayOneSpend|dayOneEq|eqLead|roarAkGPUh|nUnk|unk[A-Z]|tuneEta|etaSens|lead[A-Z]|readsPerBlock|quotaTwenty|wc[A-Z]|ci[A-Z]|week[A-Z]|qw[A-Z]|stepTune|peakRead|maxReads|mixed|refsGap|quotaWins|bound[A-Z]|lamInit|lamMin|lamSwitch|rhoMax|wMax|cost[A-Z]|part[A-Z]|pcWa|rule[A-Z]|ds[A-Z]|sess[A-Z]|waf[A-Z]|baek[A-Z]|ours[A-Z]|life[A-Z]|ll[A-Z]|qwLeadQuota|quotaTwentyAbs|pslcEightW)")
    used, optional = set(), set()
    for name in ("main.tex", "results_text.tex"):
        pth = os.path.join(OUT, name)
        if os.path.exists(pth):
            txt = open(pth).read()
            used |= {m for m in re.findall(r"\\([A-Za-z]+)", txt) if pat.match(m)}
            optional |= set(re.findall(r"\\ifdefined\\([A-Za-z]+)", txt))
    missing = sorted(used - set(M) - optional)
    with open(os.path.join(OUT, "numbers.tex"), "w") as f:
        for k, val in M.items():
            f.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
        for k in missing:
            f.write(f"\\providecommand{{\\{k}}}{{\\textbf{{??}}}}\n")
    if missing:
        log("MISSING macros (printed as ??):", " ".join(missing))
    # results table: both traces, means over day pairs
    lab = {"ROAR": "ROAR (this work)", "ROAR-refs": "\\quad value = ref.\\ count", "adaptive-k": "Adaptive $k$",
           "load-prop adaptive-k": "\\quad load-prop.", "load-prop quota": "\\quad load-prop.",
           "period-quota k=2": "Period quota", "stepwise-quota": "Stepwise quota",
           "period-quota k=1": "Period quota, 1st ref.", "day-1 price": "Constant price, day 1",
           "static price (hindsight)": "Constant price, day 2$^\\dagger$", "oracle": "Offline oracle*"}
    with open(os.path.join(OUT, "table_results.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n & \\multicolumn{3}{c}{Tool and agent} & \\multicolumn{3}{c}{Conversation} \\\\\n"
                "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\nBudget & 5\\% & 10\\% & 20\\% & 5\\% & 10\\% & 20\\% \\\\\n\\midrule\n")
        for p in TABLE_POLICIES:
            cells = []
            for t in ("toolagent", "conversation"):
                for b in BUDGETS:
                    S = D[t]["S"][b]
                    cells.append(f"{100 * mean(S[p]['value']):.1f}" if p in S else "--")
            f.write(f"{lab[p]} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    def vv(t, b, p):
        s = D[t]["S"][b].get(p)
        return 100 * mean(s["value"]) if s else float("nan")

    ex = (f"At a write budget of 10% of the write-all bytes, the disclosed method saved {vv('toolagent', 0.10, 'ROAR'):.1f}% "
          f"(tool-and-agent) and {vv('conversation', 0.10, 'ROAR'):.1f}% (conversation) of the accelerator time that writing every evicted "
          f"block saves, compared with {vv('toolagent', 0.10, 'period-quota k=2'):.1f}% and {vv('conversation', 0.10, 'period-quota k=2'):.1f}% "
          f"for a per-period write quota with second-reference admission and {vv('toolagent', 0.10, 'stepwise-quota'):.1f}% and "
          f"{vv('conversation', 0.10, 'stepwise-quota'):.1f}% for a stepwise erasure quota. Its cumulative wear never led the linear schedule "
          f"by more than {M.get('roarAheadTA', '?').replace(chr(92), '')} of one day's budget on the tool-and-agent days. An offline "
          f"reference with knowledge of future requests saved {vv('toolagent', 0.10, 'oracle'):.1f}% and {vv('conversation', 0.10, 'oracle'):.1f}%.")
    if all(x == x for x in m20):
        def more(x):
            return f"{100 * x:.1f} percentage points more" if x >= 0.0005 else (f"{-100 * x:.1f} percentage points less" if x <= -0.0005 else "the same")
        ex += (f" At a budget of 20%, compared at equal bytes written, the disclosed method saved {more(m20[0])} than the better quota on the "
               f"tool-and-agent days and {more(m20[1])} on the conversation days, because second-reference admission alone then writes about "
               f"that many bytes.")
    numbers = {"example_text": ex, "cell_modes": numbers_cell, "macros": M, "gaps": {"toolagent": g_ta, "conversation": g_cv}, "budget_table": bt, "leaf_reuse": lr,
               "days": {t: {"S": {str(b): D[t]["S"][b] for b in BUDGETS}, "U": D[t]["U"], "meta": D[t]["meta"],
                            "sweep": D[t]["sweep"]} for t in D},
               "frequency_rules_pair01": FR, "hourly_pair01_10pct": H, "sensitivity": E5, "tuning_10pct": TUNE,
               "equal_bytes_pair01": {t: {str(b): v for b, v in EQ[t].items()} for t in EQ}, "static_vs_roar_pair01": STATIC, "handover_check": HAND}
    with open(os.path.join(OUT, "numbers.json"), "w") as f:
        json.dump(numbers, f, indent=1, default=str)
    log("wrote numbers.tex, table_budget.tex, table_results.tex, numbers.json into", OUT)


if __name__ == "__main__":
    main()
