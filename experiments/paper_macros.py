"""Number macros for the paper beyond the core tables: adaptive k and the ablations (E6), estimator
transfer (E7), long-gap reuse (E8), unknown-load days and confidence intervals (E10), the Qwen-Bailian
hours (E17), the admission rules and DeepSeek-V3 volume (E18), the session feature (E19), the write-
amplification loop (E20), the payoff as a share of prefill compute, and the pSLC break-even. Called from
make_paper_inputs.py; every value comes from a file in results/ or from model.py."""
import csv
import glob
import json
import os
import statistics as st
import glob

from common import RESULTS, TRACES
from shirroar import B200, LLAMA31_70B, load_mooncake
from shirroar.model import PSLC_512G, TLC_1536G, YEAR_DAYS
from shirroar.trace import diurnal_multiplier


def _rows(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def _f(x):
    return float(x) if x not in (None, "", "None") else None


def prefill_gpu_h_per_day(trace):
    """GPU-hours to prefill every prompt of the synthetic day with no cache at all."""
    tot = 0.0
    for r in load_mooncake(TRACES[trace]):
        for i in range(len(r.blocks)):
            tot += LLAMA31_70B.recompute_seconds(i * 512, B200.flops)
    return tot / 3600.0 * sum(diurnal_multiplier(h) for h in range(24))


def add(M, D):
    tags = (("toolagent", "TA"), ("conversation", "CV"))
    # ---- payoff: flash savings as a share of the day's prefill compute ----------------------------
    for t, tag in tags:
        pf = prefill_gpu_h_per_day(t)
        M[f"prefillGPUh{tag}"] = f"{pf:.0f}"
        wa = D[t]["meta"]["wa_flash_gpu_h"]
        hb = D[t]["meta"]["hbm_gpu_h"]
        M[f"waShare{tag}"] = f"{100 * wa / pf:.0f}\\%"
        M[f"hbmShare{tag}"] = f"{100 * hb / pf:.0f}\\%"
    e4 = os.path.join(RESULTS, "e4_cell_modes_toolagent.csv")
    if os.path.exists(e4):
        C = _rows(e4)

        def gh(n, cfg, pol="ROAR"):
            r = [x for x in C if int(x["stacks"]) == n and x["config"] == cfg and x["policy"] == pol]
            return _f(r[0]["flash_saved_gpu_h_per_day"]) if r else None
        pf = prefill_gpu_h_per_day("toolagent")
        tlc8, pslc8, q8 = gh(8, "all TLC"), gh(8, "all pSLC"), gh(8, "all TLC", "period-quota k=2")
        M["tlcEightGPUh"] = f"{tlc8:.1f}"
        M["tlcEightShare"] = f"{100 * tlc8 / pf:.1f}\\%"
        M["pslcEightGPUh"] = f"{pslc8:.1f}"
        r8 = [x for x in C if int(x["stacks"]) == 8 and x["config"] == "all pSLC" and x["policy"] == "ROAR"]
        if r8:
            M["pslcEightW"] = f"{_f(r8[0]['written_over_budget']):.2f}"
        M["pslcEightShare"] = f"{100 * pslc8 / pf:.1f}\\%"
        M["quotaTlcEightGPUh"] = f"{q8:.1f}"
    # ---- pSLC break-even (writable bytes equal on the same dies) ----------------------------------
    ratio = TLC_1536G.capacity_bytes / PSLC_512G.capacity_bytes          # 3: one bit per cell instead of three
    M["pslcBreakEven"] = f"{ratio * TLC_1536G.pe_cycles / 1000:.0f}K"
    M["pslcBreakEvenX"] = f"{ratio:.0f}"
    M["pslcResBreakEven"] = f"{5 * YEAR_DAYS * 24 / (ratio * TLC_1536G.pe_cycles):.1f}"
    # ---- peak write rate of ROAR (day 2, pair 0-1, 10% budget) ------------------------------------
    peaks = []
    for t, tag in tags:
        p = os.path.join(RESULTS, f"e3_days_series_0-1_{t}.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p))["ROAR"]
        hours = {}
        for i in range(1, len(s["t"])):
            h = int((s["t"][i] - 86400) // 3600)
            hours[h] = hours.get(h, 0.0) + s["r0_bytes"][i] - s["r0_bytes"][i - 1]
        peaks.append(max(hours.values()))
    if peaks:
        pk = max(peaks)
        M["peakWriteTBh"] = f"{pk / 1e12:.1f}"
        M["peakWriteGBs"] = f"{pk / 3600 / 1e9:.1f}"
        M["peakWriteStackMBs"] = f"{pk / 3600 / 1e6 / 8:.0f}"
        # Kyung et al. (IEEE CAL 2026) assume 48 GB/s write bandwidth per HBF stack
        M["peakWriteShareKyung"] = f"{100 * pk / 3600 / 8 / 48e9:.1f}\\%"
    # ---- E12: peak flash read rate and reads per block (ROAR, 10%, pair 0-1) ----------------------
    e12 = [json.load(open(os.path.join(RESULTS, f"e12_read_rate_{t}.json"))) for t, tag in tags
           if os.path.exists(os.path.join(RESULTS, f"e12_read_rate_{t}.json"))]
    if e12:
        M["peakReadHourStackGBs"] = f"{max(e['peak_hour_stack_GBs'] for e in e12):.1f}"
        M["peakReadTickStackGBs"] = f"{max(e['peak_tick_stack_GBs'] for e in e12):.1f}"
        M["peakReadTickShare"] = f"{100 * max(e['peak_tick_share_of_stack_bw'] for e in e12):.2f}\\%"
        M["maxReadsBlock"] = f"{max(e['max_reads_per_block'] for e in e12):,}"
    # ---- E6: adaptive-k grid and ablations ---------------------------------------------------------
    for t, tag in tags + (("arxiv", "AX"),):
        p = os.path.join(RESULTS, f"e6_adaptive_k_0-1_{t}.csv")
        if not os.path.exists(p):
            continue
        R = _rows(p)
        for b, bn in ((0.05, "five"), (0.10, "ten"), (0.20, "twenty")):
            g = [r for r in R if r["policy"].startswith("adaptive-k") and abs(_f(r["budget_share"]) - b) < 1e-9
                 and _f(r["written_over_budget"]) <= 1.05]
            if g:
                best = max(g, key=lambda r: _f(r["value_share"]))
                M[f"akBest{tag}B{bn}"] = f"{100 * _f(best['value_share']):.1f}"
            n_all = [r for r in R if r["policy"].startswith("adaptive-k") and abs(_f(r["budget_share"]) - b) < 1e-9]
            if n_all:
                M[f"akGridN{tag}"] = str(len({r['policy'] for r in n_all}))
            wide = [r for r in R if r["policy"] == "adaptive-k band21600 dwell300" and abs(_f(r["budget_share"]) - b) < 1e-9]
            if wide:
                M[f"akWide{tag}B{bn}"] = f"{100 * _f(wide[0]['value_share']):.1f}"
                M[f"akWideSpend{tag}B{bn}"] = f"{_f(wide[0]['written_over_budget']):.2f}"
                M[f"akWideOver{tag}B{bn}"] = f"{100 * (_f(wide[0]['written_over_budget']) - 1):.0f}\\%"
            for pol, mac in (("ROAR", "Roar"), ("ROAR-refs", "Refs"), ("ROAR-refs-depth", "RefsDepth"),
                             ("ROAR-refs no parent check", "RefsNoParent"), ("adaptive-k dither step0.01", "AkChosen")):
                x = [r for r in R if r["policy"] == pol and abs(_f(r["budget_share"]) - b) < 1e-9]
                if x:
                    M[f"ab{mac}{tag}B{bn}"] = f"{100 * _f(x[0]['value_share']):.1f}"
    # margins of ROAR over the chosen adaptive-k and over ROAR-refs, per pair (all pairs that have both)
    for t, tag in tags:
        S = D[t]["S"]
        for b, bn in ((0.05, "five"), (0.10, "ten"), (0.20, "twenty")):
            def by_pair(p):
                e = S[b].get(p)
                return dict(zip(e["pairs"], e["value"])) if e else {}
            ro, ak, rf = by_pair("ROAR"), by_pair("adaptive-k"), by_pair("ROAR-refs")
            m_ak = [ro[k] - ak[k] for k in ro if k in ak]
            m_rf = [ro[k] - rf[k] for k in ro if k in rf]
            if m_ak:
                lo_, hi_ = f"{100 * min(m_ak):.1f}", f"{100 * max(m_ak):.1f}"
                M[f"marginAk{tag}B{bn}"] = lo_ if lo_ == hi_ else f"{lo_}--{hi_}"
                M[f"marginAkMin{tag}B{bn}"] = f"{100 * min(m_ak):.1f}"
            if m_rf:
                M[f"marginRefs{tag}B{bn}"] = f"{100 * st.mean(m_rf):.1f}"
            # spread of ROAR over day pairs
            if ro:
                M[f"spreadRoar{tag}B{bn}"] = f"{100 * (max(ro.values()) - min(ro.values())):.1f}"
    for t, tag in tags:
        e = D[t]["S"][0.10].get("day-1 price")
        if e:
            M[f"spendDayOne{tag}"] = f"{st.mean(e['written']):.2f}"
    # day-1 constant price against ROAR's budget sweep at the bytes it actually wrote (pair 0-1)
    diffs = []
    for t in ("toolagent", "conversation"):
        sw = D[t]["sweep"].get("ROAR (sweep)")
        for b in (0.05, 0.10, 0.20):
            e = D[t]["S"][b].get("day-1 price")
            if not sw or not e or "0-1" not in e["pairs"]:
                continue
            i = e["pairs"].index("0-1")
            pts = [(0.0, 0.0)] + sorted((w, v) for _, w, v in sw)
            x = b * e["written"][i]
            y = next((y0 + (y1 - y0) * (x - x0) / (x1 - x0) for (x0, y0), (x1, y1) in zip(pts, pts[1:]) if x0 <= x <= x1),
                     pts[-1][1])
            diffs.append(e["value"][i] - y)
    # ROAR's lead over adaptive k and over the quotas at the bytes each of those wrote (pair 0-1, ROAR's sweep)
    for t, tag in tags:
        sw = D[t]["sweep"].get("ROAR (sweep)")
        if not sw:
            continue
        pts = [(0.0, 0.0)] + sorted((w, v) for _, w, v in sw)

        def roar_at(x):
            return next((y0 + (y1 - y0) * (x - x0) / (x1 - x0) for (x0, y0), (x1, y1) in zip(pts, pts[1:]) if x0 <= x <= x1),
                        pts[-1][1])
        for b, bn in ((0.05, "five"), (0.10, "ten"), (0.20, "twenty")):
            for p_, mac in (("adaptive-k", "Ak"), ("ROAR-refs", "Refs")):
                e = D[t]["S"][b].get(p_)
                if not e or "0-1" not in e["pairs"]:
                    continue
                i = e["pairs"].index("0-1")
                M[f"eqLead{mac}{tag}B{bn}"] = f"{100 * (roar_at(b * e['written'][i]) - e['value'][i]):.1f}"
    g = [float(M[f"eqLeadAk{tag}Bten"]) / 100 * D[t]["meta"]["wa_flash_gpu_h"] for t, tag in tags if f"eqLeadAk{tag}Bten" in M]
    if g:
        M["roarAkGPUh"] = f"{st.mean(g):.1f}"
    if diffs:
        M["dayOneEqMin"] = f"{100 * min(diffs):.1f}"
        M["dayOneEqMax"] = f"{100 * max(diffs):.1f}"
    sp = [x for t in ("toolagent", "conversation") for b in (0.05, 0.10, 0.20)
          for x in (D[t]["S"][b].get("day-1 price") or {}).get("written", [])]
    if sp:
        M["dayOneSpendMin"] = f"{min(sp):.2f}"
        M["dayOneSpendMax"] = f"{max(sp):.2f}"
    allspread = []
    for t in ("toolagent", "conversation"):
        for b in (0.05, 0.10, 0.20):
            for p, e in D[t]["S"][b].items():
                if len(e["value"]) > 1 and p not in ("static price (hindsight)", "day-1 price"):
                    allspread.append(max(e["value"]) - min(e["value"]))
    if allspread:
        M["maxPairSpread"] = f"{100 * max(allspread):.1f}"
    # ---- E7: estimator transfer --------------------------------------------------------------------
    for t, tag in tags:
        p = os.path.join(RESULTS, f"e7_transfer_0-1_{t}.csv")
        if not os.path.exists(p):
            continue
        R = _rows(p)
        for b, bn in ((0.05, "five"), (0.10, "ten"), (0.20, "twenty")):
            sel = {r["estimator"]: _f(r["value_share"]) for r in R if abs(_f(r["budget_share"]) - b) < 1e-9}
            on = sel.get("online (default)")
            fz = [v for k, v in sel.items() if k.startswith("frozen") and t not in k]
            wm = [v for k, v in sel.items() if k.startswith("warm") and t not in k]
            if on is not None and fz:
                M[f"xferFrozen{tag}B{bn}"] = f"{100 * fz[0]:.1f}"
                M[f"xferOnline{tag}B{bn}"] = f"{100 * on:.1f}"
                M[f"xferLoss{tag}B{bn}"] = f"{100 * (on - fz[0]):.1f}"
            if on is not None and wm:
                M[f"xferWarm{tag}B{bn}"] = f"{100 * wm[0]:.1f}"
    # ---- E8: long-gap reuse ------------------------------------------------------------------------
    wcp = os.path.join(RESULTS, "wildchat_gaps.json")
    if os.path.exists(wcp):
        W = json.load(open(wcp))
        M["wcConvs"] = f"{W['conversations']:,}"
        M["wcGaps"] = f"{W['follow_up_gaps']:,}"
        M["wcMedian"] = f"{W['median_gap_s']:.0f}"
        for h in (1, 2, 15, 24):
            M[f"wcOver{ {1: 'One', 2: 'Two', 15: 'Fifteen', 24: 'Day'}[h] }"] = f"{100 * W[f'share_of_follow_ups_over_{h}h']:.1f}\\%"
        M["wcReturn"] = f"{100 * W['return_after_1h_per_turn']:.1f}\\%"
    for frac, ftag in ((0.2, "Re"), (0.0, "No"), ("wildchat", "Wc")):
        p = os.path.join(RESULTS, f"e8_long_gaps_{frac if isinstance(frac, str) else format(frac, 'g')}_0-1_conversation.csv")
        if not os.path.exists(p):
            continue
        R = _rows(p)

        def g(cfg, pol="ROAR", cap="real", col="flash_saved_gpu_h_per_day"):
            x = [r for r in R if r["config"] == cfg and r["policy"] == pol and r["capacity"] == cap]
            return _f(x[0][col]) if x else None
        for cfg, mac in (("all TLC", "Tlc"), ("all pSLC 20K", "Pslc"), ("all pSLC 9K", "PslcBE"),
                         ("half pSLC 20K, half TLC", "Half")):
            for pol, pm in (("ROAR", ""), ("ROAR, 6 h window", "Long")):
                v, u = g(cfg, pol), g(cfg, pol, "unlimited")
                if v is not None:
                    M[f"lg{ftag}{mac}{pm}"] = f"{v:.1f}"
                if v is not None and u:
                    M[f"lg{ftag}{mac}{pm}CapLoss"] = f"{100 * (1 - v / u):.0f}\\%"
        if ftag == "Re":
            d = [abs(_f(a["flash_saved_gpu_h_per_day"]) - _f(b["flash_saved_gpu_h_per_day"]))
                 for a in R for b in R if a["config"] == b["config"] and a["capacity"] == b["capacity"]
                 and a["policy"] == "ROAR" and b["policy"] == "ROAR, 6 h window"]
            if d:
                M["lgWindowDelta"] = f"{max(d) + 0.05:.1f}"
        wa = [r for r in R]
        if wa:
            M[f"lg{ftag}WaGPUh"] = f"{_f(wa[0]['write_all_flash_gpu_h']):.1f}"
            M[f"lg{ftag}HbmGPUh"] = f"{_f(wa[0]['hbm_saved_gpu_h']):.1f}"
    # ---- E15: stepwise quota tuned on the synthetic trace ------------------------------------------
    p = os.path.join(RESULTS, "e15_stepwise_choice_synthetic.json")
    if os.path.exists(p):
        c = json.load(open(p))
        M["stepTune"] = (f"{c['quantum_s']:g}-second quanta, steps of {'/'.join(map(str, c['steps']))} references"
                         + (f", and {c['base_k']} references even within the allowance" if c["base_k"] > 1 else ""))
    # ---- E16: a week of known-load days, measured on days 3-7 --------------------------------------
    for p in sorted(glob.glob(os.path.join(RESULTS, "e16_week_*d_conversation.csv"))):
        R = {r["policy"]: r for r in _rows(p)}
        for pol, mac in (("ROAR", "Roar"), ("period-quota k=2", "Quota"), ("load-prop adaptive-k", "AkLoad")):
            if pol in R:
                M[f"week{mac}"] = f"{100 * _f(R[pol]['value_share']):.1f}"
                M[f"week{mac}W"] = f"{_f(R[pol]['written_over_budget']):.2f}"
        if "ROAR" in R:
            M["weekRoarLead"] = f"{_f(R['ROAR']['max_ahead_days_of_budget']):.2f}"
            M["weekDays"] = R["ROAR"]["days"]
    # ---- E17: Qwen-Bailian hours, 10% budget, equal-bytes leads --------------------------------------
    Q = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, "e17_qwen_0-1_qwen*.csv"))):
        name = os.path.basename(p)[len("e17_qwen_0-1_"):-4]
        R = _rows(p)
        sw = [(_f(r["written_share"]), _f(r["value_share"])) for r in R if r["policy"] in ("ROAR", "ROAR (sweep)")]
        for r in R:
            if r["policy"] in ("ROAR", "ROAR (sweep)"):
                continue
            y = _interp(sw, _f(r["written_share"]))
            if y is not None:
                Q.setdefault(r["policy"], {})[name] = 100 * (y - _f(r["value_share"]))
        Q.setdefault("_hours", []).append(name)
    if Q:
        M["qwHours"] = str(len(Q["_hours"]))
        for pol, mac in (("load-prop adaptive-k", "AkLoad"), ("adaptive-k", "Ak"), ("period-quota k=2", "Quota"),
                         ("load-prop quota", "QuotaLoad"), ("stepwise-quota", "Step")):
            xs = list(Q.get(pol, {}).values())
            if xs:
                lo, hi = round(min(xs), 1) + 0.0, round(max(xs), 1) + 0.0
                sl = f"$-${-lo:.1f}" if lo < 0 else f"{lo:.1f}"
                M[f"qwLead{mac}"] = (f"{sl} to {hi:.1f}" if lo < 0 else f"{sl}--{hi:.1f}") if lo != hi else sl
                M[f"qwPos{mac}"] = f"{sum(x > 0 for x in xs)} of {len(xs)}"
                M[f"qwMean{mac}"] = f"{st.mean(xs):.1f}"
        xs = [x for pol in ("period-quota k=2", "load-prop quota", "stepwise-quota") for x in Q.get(pol, {}).values()]
        if xs:
            M["qwLeadQuotaRange"] = f"{round(min(xs), 1) + 0.0:.1f}--{max(xs):.1f}"
            M["qwNQuota"] = str(len(xs))
    new_items(M)
    return M


def new_items(M):
    """E18 (partial-block and parent-check rules, DeepSeek write volume), E19 (session-aware estimate),
    E20 (wear loop closed on the device counter), the comparison with Baek et al., and the life trade-off."""
    tags = (("toolagent", "TA"), ("conversation", "CV"))
    from shirroar import DEEPSEEK_V3
    from shirroar.model import CELLS
    # ---- E18 rules ---------------------------------------------------------------------------------
    for t, tag in tags:
        p = os.path.join(RESULTS, f"e18_rules_{t}.csv")
        if not os.path.exists(p):
            continue
        R = {r["policy"]: r for r in _rows(p)}
        g = lambda k, c="value_share": _f(R[k][c]) if k in R else None
        if g("write-all [skip partial]") is not None:
            M[f"partWrite{tag}"] = f"{100 * (1 - g('write-all [skip partial]', 'written_share')):.1f}"
            M[f"partValue{tag}"] = f"{100 * (1 - g('write-all [skip partial]')):.1f}"
        if g("write-all [parent check]") is not None:
            M[f"pcWaWrite{tag}"] = f"{100 * (1 - g('write-all [parent check]', 'written_share')):.1f}\\%"
        for k, mac in (("ROAR", "Roar"), ("ROAR [skip partial]", "RoarSkip"), ("ROAR [no parent check]", "RoarNoPc"),
                       ("adaptive-k", "Ak"), ("adaptive-k [skip partial]", "AkSkip"), ("adaptive-k [no parent check]", "AkNoPc"),
                       ("period-quota k=2", "Quota"), ("period-quota k=2 [skip partial]", "QuotaSkip"),
                       ("period-quota k=2 [parent check]", "QuotaPc")):
            if g(k) is not None:
                M[f"rule{mac}{tag}"] = f"{100 * g(k):.1f}"
    # ---- E18 DeepSeek: write-all bytes at equal HBM bytes ---------------------------------------------
    for t, tag in tags:
        p = os.path.join(RESULTS, f"e18_deepseek_{t}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        M[f"dsRatio{tag}"] = f"{d['ratio_llama_over_deepseek']:.2f}"
        M["dsKvRatio"] = f"{d['kv_ratio']:.2f}"
        ds_day = d["deepseek"]["write_all_bytes_day2"]
        M[f"dsWaTB{tag}"] = f"{ds_day / 1e12:.0f}"
        for cell, mac in (("tlc", "Tlc"), ("pslc", "Pslc"), ("qlc", "Qlc")):
            c = CELLS.get(cell)
            if c is not None:
                for n, nm in ((8, "Eight"), (32, "ThirtyTwo")):
                    M[f"ds{mac}{nm}{tag}"] = f"{100 * n * c.daily_budget(5.0) / ds_day:.0f}"
    # Llama-3.1-70B budgets on 8 stacks, for the same comparison
    meta = os.path.join(RESULTS, "e1_endurance_meta_toolagent.json")
    if os.path.exists(meta):
        wa_day = json.load(open(meta))["write_all_day_TB"] * 1e12
        for cell, mac in (("tlc", "Tlc"), ("qlc", "Qlc")):
            M[f"ll{mac}EightTA"] = f"{100 * 8 * CELLS[cell].daily_budget(5.0) / wa_day:.1f}"
    # ---- E19 session-aware estimate on the Qwen hours ---------------------------------------------------
    rows = []
    for p in sorted(glob.glob(os.path.join(RESULTS, "e19_session_qwen*.csv"))):
        R = {r["policy"]: r for r in _rows(p)}
        if all(k in R for k in ("ROAR", "ROAR session-aware", "oracle")):
            a, b_, o = (_f(R[k]["value_share"]) for k in ("ROAR", "ROAR session-aware", "oracle"))
            rows.append((100 * (b_ - a), (b_ - a) / (o - a) if o > a else 0.0, _f(R["ROAR session-aware"]["written_over_budget"])))
    if rows:
        xs = [r[0] for r in rows]
        M["sessHours"] = str(len(rows))
        M["sessGain"] = f"{min(xs):.1f}--{max(xs):.1f}"
        M["sessGainMean"] = f"{st.mean(xs):.1f}"
        M["sessGapClosed"] = f"{100 * min(r[1] for r in rows):.0f}--{100 * max(r[1] for r in rows):.0f}\\%"
        M["sessWmax"] = f"{max(r[2] for r in rows):.2f}"
    # ---- E20 wear loop on the device counter -------------------------------------------------------------
    for t, tag in tags:
        p = os.path.join(RESULTS, f"e20_waf_{t}.csv")
        if not os.path.exists(p):
            continue
        R = _rows(p)
        sel = lambda mode: [r for r in R if r["controller"] == mode]
        lp, kn, nv = sel("loop"), sel("known"), sel("naive")
        if lp and kn and nv:
            w = [_f(r["device_written_over_budget"]) for r in lp]
            M[f"wafLoopW{tag}"] = f"{min(w):.2f}--{max(w):.2f}"
            w = [_f(r["device_written_over_budget"]) for r in nv]
            M[f"wafNaiveW{tag}"] = f"{min(w):.2f}--{max(w):.2f}"
            dv = [100 * (_f(a["value_share"]) - _f(b["value_share"])) for a, b in zip(lp, kn)]
            M[f"wafLoopDv{tag}"] = f"{max(abs(x) for x in dv):.1f}"
            M[f"wafLoopLead{tag}"] = f"{max(_f(r['max_device_lead_days']) for r in lp):.2f}"
    # ---- Baek et al.: 3 TiB SLC (100K cycles, WAF 1.02) lasting about 20 years under write-on-evict -----
    baek_bytes_day = 3 * 2 ** 40 * 100_000 / (20 * 365.25 * 1.02)
    baek_tok_s = baek_bytes_day / (96 * 1024) / 86400.0
    meta = os.path.join(RESULTS, "e1_endurance_meta_toolagent.json")
    if os.path.exists(meta):
        wa_tb = json.load(open(meta))["write_all_day_TB"]
        ours_tok_s = wa_tb * 1e12 / LLAMA31_70B.kv_bytes_per_token / 86400.0
        M["baekTBday"] = f"{baek_bytes_day / 1e12:.0f}"
        M["baekTokS"] = f"{baek_tok_s / 1000:.1f}K"
        M["oursTokS"] = f"{ours_tok_s / 1000:.0f}K"
        M["baekShare"] = f"{100 * baek_bytes_day / (wa_tb * 1e12):.1f}\\%"
        M["baekOurLifeSLC"] = f"{3 * 2 ** 40 * 100_000 / (1.02 * wa_tb * 1e12) / 365.25:.1f}"
        M["baekOurLifeTLCdays"] = f"{3 * 2 ** 40 * 3000 / (1.02 * wa_tb * 1e12):.0f}"
    # ---- life trade-off: GPU-hours per day ROAR saves on a fixed node at 3, 5 and 10 years ----------------
    for t, tag in tags:
        curves = _known_roar_curves(t)
        wa = [_f(r["write_all_flash_saved_s"]) for p in glob.glob(os.path.join(RESULTS, f"e3_days_*-*_{t}.csv"))
              for r in _rows(p) if r["policy"] == "ROAR" and r.get("write_all_flash_saved_s")]
        meta = os.path.join(RESULTS, f"e1_endurance_meta_{t}.json")
        if not curves or not wa or not os.path.exists(meta):
            continue
        wa_h = st.mean(wa) / 3600.0
        wa_day = json.load(open(meta))["write_all_day_TB"] * 1e12
        for cell, n, mac in (("pslc", 8, "PslcEight"), ("tlc", 32, "TlcThirtyTwo")):
            c = CELLS[cell]
            for yrs, ym in ((3, "Three"), (5, "Five"), (10, "Ten")):
                b = n * c.daily_budget(float(yrs)) / wa_day
                ys = [_interp(pts, b) for pts in curves.values()]
                ys = [y for y in ys if y is not None]
                if ys:
                    M[f"life{mac}{ym}{tag}"] = f"{wa_h * st.mean(ys):.1f}"
                    M[f"lifeB{mac}{ym}{tag}"] = f"{100 * b:.1f}"


# ---- unknown-load days, held-out tuning, gain sensitivity, Table 3 ------------------------------------
UNK_POLICIES = [("ROAR", "ROAR (this work)", "Roar"), ("ROAR-refs", "\\quad value = ref.\\ count", "Refs"),
                ("adaptive-k", "Adaptive $k$", "Ak"), ("load-prop adaptive-k", "\\quad load-prop.", "AkLoad"),
                ("yesterday adaptive-k", "\\quad yesterday's load", "AkYest"),
                ("period-quota k=2", "Period quota", "Quota"), ("load-prop quota", "\\quad load-prop.", "QuotaLoad"),
                ("stepwise-quota", "Stepwise quota", "Step"), ("day-1 price", "Constant price, day 1", "DayOne"),
                ("oracle", "Offline oracle*", "Oracle")]


def _roar_at(D, t, x):
    sw = D[t]["sweep"].get("ROAR (sweep)")
    if not sw:
        return None
    pts = [(0.0, 0.0)] + sorted((w, v) for _, w, v in sw)
    return next((y0 + (y1 - y0) * (x - x0) / (x1 - x0) for (x0, y0), (x1, y1) in zip(pts, pts[1:]) if x0 <= x <= x1),
                pts[-1][1])


def _interp(pts, x):
    pts = [(0.0, 0.0)] + sorted(pts)
    if x < pts[1][0] * 0.98 or x > pts[-1][0] * 1.03:
        return None
    if x > pts[-1][0]:       # up to 3% past ROAR's largest point: extend its last segment
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        return y1 + (y1 - y0) * (x - x1) / (x1 - x0) if x1 > x0 else y1
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0) if x1 > x0 else y1
    return None


def _known_roar_curves(t):
    """Per seed pair: ROAR's (bytes share, savings share) points on known-load days, from e3 (main budgets
    and the pair 0-1 sweep) and e10 known --sweep."""
    cur = {}
    for p in glob.glob(os.path.join(RESULTS, f"e3_days_*-*_{t}.csv")):
        for r in _rows(p):
            if r["policy"].startswith("ROAR") and not r["policy"].startswith("ROAR-"):
                cur.setdefault(r["seeds"], set()).add((_f(r["written_share"]), _f(r["value_share"])))
    for p in glob.glob(os.path.join(RESULTS, f"e10_load_aware_known_*-*_{t}.csv")):
        for r in _rows(p):
            if r["policy"] == "ROAR (sweep)":
                cur.setdefault(r["seeds"], set()).add((_f(r["written_share"]), _f(r["value_share"])))
    return {k: list(v) for k, v in cur.items()}


def unknown(M, D, out_dir):
    """Unknown-load days with a fixed byte budget (e10 'unknownfix'), equal-bytes leads on both load
    families from three seeds, per-seed signs, and Table 3."""
    tags = (("toolagent", "TA"), ("conversation", "CV"))
    BUD = ((0.05, "five"), (0.10, "ten"), (0.20, "twenty"))
    U, ALL = {}, {}
    for t, tag in tags:
        R = [r for p in sorted(glob.glob(os.path.join(RESULTS, f"e10_load_aware_unknownfix_*-*_{t}.csv"))) for r in _rows(p)]
        ALL[t] = R
        R = [r for r in R if int(r["seeds"].split("-")[0]) < 106]     # the three full seed pairs; 106+ are lean
        U[t] = R
        M[f"nUnk{tag}"] = str(len({r["seeds"] for r in R}))
        for pol, _, mac in UNK_POLICIES:
            for b, bn in BUD:
                sel = [r for r in R if r["policy"] == pol and abs(_f(r["budget_share"]) - b) < 1e-9]
                if sel:
                    M[f"unk{mac}V{tag}B{bn}"] = f"{100 * st.mean(_f(r['value_share']) for r in sel):.1f}"
                    M[f"unk{mac}W{tag}B{bn}"] = f"{st.mean(_f(r['written_over_budget']) for r in sel):.2f}"
    for pol, _, mac in UNK_POLICIES:
        ws = [_f(r["written_over_budget"]) for t in U for r in U[t] if r["policy"] == pol]
        if ws:
            M[f"unk{mac}Wmin"] = f"{min(ws):.2f}"
            M[f"unk{mac}Wmax"] = f"{max(ws):.2f}"
    la = [_f(r["max_ahead_share_of_daily_budget"]) for t in U for r in U[t] if r["policy"] == "ROAR"]
    if la:
        M["unkRoarLeadMax"] = f"{100 * max(la):.0f}\\%"
        M["unkRoarLeadLife"] = f"{100 * max(la) / (5 * 365.25):.2f}\\%"
    rules = ("adaptive-k", "load-prop adaptive-k", "period-quota k=2", "load-prop quota", "stepwise-quota")
    ws = [_f(r["written_over_budget"]) for t in U for r in U[t] if r["policy"] in rules]
    if ws:
        M["unkRulesWmin"] = f"{min(ws):.2f}"
        M["unkRulesWmax"] = f"{max(ws):.2f}"
    # reads of flash per written block (ROAR, 10%, unknown-load days): compare with read-reclaim thresholds
    rr = [_f(r["flash_hits"]) / _f(r["blocks_written"]) for t in U for r in U[t]
          if r["policy"] == "ROAR" and abs(_f(r["budget_share"]) - 0.10) < 1e-9 and _f(r.get("blocks_written") or 0)]
    if rr:
        M["readsPerBlock"] = f"{max(rr):.1f}"
    # equal-bytes leads, per seed pair, both families
    LEAD = {}     # (family, trace, policy, budget) -> list of per-pair leads
    for t, tag in tags:
        curves = _known_roar_curves(t)
        for pol, _, mac in UNK_POLICIES:
            if pol in ("ROAR", "oracle"):
                continue
            for b, bn in BUD:
                e = D[t]["S"][b].get(pol)
                if e:
                    for pair, v, w in zip(e["pairs"], e["value"], e["written"]):
                        y = _interp(curves.get(pair, []), b * w) if w is not None else None
                        if y is not None:
                            LEAD.setdefault(("known", t, pol, b), []).append(y - v)
                # the load-aware adaptive-k rules also ran on the extra (lean) seed pairs: use them all, so
                # Table 3, the text and the confidence interval agree
                R = ALL[t] if pol in ("load-prop adaptive-k", "yesterday adaptive-k") else U[t]
                for pair in sorted({r["seeds"] for r in R}):
                    sw = [(_f(r["written_share"]), _f(r["value_share"])) for r in R if r["seeds"] == pair
                          and r["policy"] in ("ROAR", "ROAR (sweep)")]
                    e2 = [r for r in R if r["seeds"] == pair and r["policy"] == pol and abs(_f(r["budget_share"]) - b) < 1e-9]
                    if e2:
                        y = _interp(sw, _f(e2[0]["written_share"]))
                        if y is not None:
                            LEAD.setdefault(("unknown", t, pol, b), []).append(y - _f(e2[0]["value_share"]))
    for (fam, t, pol, b), xs in LEAD.items():
        mac = next(m for p_, _, m in UNK_POLICIES if p_ == pol)
        tag = dict(tags)[t]
        bn = dict(BUD)[b]
        F = "Known" if fam == "known" else "Unk"
        M[f"lead{F}{mac}{tag}B{bn}"] = f"{100 * st.mean(xs):.1f}"
        M[f"leadPos{F}{mac}{tag}B{bn}"] = f"{sum(x > 0 for x in xs)}/{len(xs)}"
    # ranges used in the text
    for pol, mac in (("load-prop adaptive-k", "AkLoad"), ("yesterday adaptive-k", "AkYest"), ("ROAR-refs", "Refs"), ("load-prop quota", "QuotaLoad"),
                     ("period-quota k=2", "Quota"), ("adaptive-k", "Ak"), ("day-1 price", "DayOne")):
        for b, bn in BUD:
            xs = [st.mean(LEAD[k]) for k in LEAD if k[2] == pol and k[3] == b]
            if xs:
                M[f"leadRange{mac}B{bn}"] = (f"{100 * min(xs):.1f}--{100 * max(xs):.1f}" if round(100 * min(xs), 1) != round(100 * max(xs), 1)
                                              else f"{100 * min(xs):.1f}")
        pos = [sum(x > 0 for x in LEAD[k]) for k in LEAD if k[2] == pol and k[3] == 0.10]
        tot = [len(LEAD[k]) for k in LEAD if k[2] == pol and k[3] == 0.10]
        if tot:
            M[f"leadPosAll{mac}"] = f"{sum(pos)} of {sum(tot)}"
    dx = [-st.mean(LEAD[k]) for k in LEAD if k[0] == "known" and k[2] == "day-1 price" and k[3] == 0.10]
    if dx:
        M["dayOneEqGain"] = (f"{100 * min(dx):.1f}--{100 * max(dx):.1f}" if round(100 * min(dx), 1) != round(100 * max(dx), 1)
                             else f"{100 * min(dx):.1f}")
    zx = [-st.mean(LEAD[k]) for k in LEAD if k[2] == "period-quota k=2" and k[3] == 0.20]
    if zx:
        lo, hi = round(100 * min(zx), 1) + 0.0, round(100 * max(zx), 1) + 0.0   # + 0.0 turns -0.0 into 0.0
        M["quotaTwentyGain"] = f"{lo:.1f}--{hi:.1f}"
        M["quotaTwentyAbs"] = f"{max(abs(lo), abs(hi)):.1f}"
    qx = [st.mean(LEAD[k]) for k in LEAD if k[2] in ("period-quota k=2", "load-prop quota", "stepwise-quota") and k[3] == 0.10]
    if qx:
        M["leadQuotaMin"] = f"{100 * min(qx):.1f}"
        M["leadQuotaMax"] = f"{100 * max(qx):.1f}"
    # 95% confidence interval of ROAR's equal-bytes lead at 10% over every unknown-load seed pair (full + lean)
    def tcrit(n):
        tab = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26, 11: 2.23, 12: 2.20}
        return tab.get(n, 1.96 if n > 30 else 2.1)
    for pol, mac in (("load-prop adaptive-k", "AkLoad"), ("yesterday adaptive-k", "AkYest")):
        for t, tag in tags:
            R = ALL[t]
            leads = []
            for pair in sorted({r["seeds"] for r in R}):
                sw = [(_f(r["written_share"]), _f(r["value_share"])) for r in R if r["seeds"] == pair
                      and r["policy"] in ("ROAR", "ROAR (sweep)")]
                e2 = [r for r in R if r["seeds"] == pair and r["policy"] == pol and abs(_f(r["budget_share"]) - 0.10) < 1e-9]
                if e2:
                    y = _interp(sw, _f(e2[0]["written_share"]))
                    if y is not None:
                        leads.append(100 * (y - _f(e2[0]["value_share"])))
            if len(leads) >= 2:
                m, sd = st.mean(leads), st.stdev(leads)
                h = tcrit(len(leads)) * sd / len(leads) ** 0.5
                M[f"ci{mac}{tag}"] = f"{m:.1f}\\,$\\pm$\\,{h:.1f}"
                M[f"ciN{mac}{tag}"] = str(len(leads))
                M[f"ciPos{mac}{tag}"] = f"{sum(x > 0 for x in leads)} of {len(leads)}"
    # Table 3 (10% budget): equal-bytes lead (mean of three seeds) and day-2 writes per budget, both families
    def wmean(fam, pol, b=0.10):
        xs = []
        for t, tag in tags:
            if fam == "known":
                e = D[t]["S"][b].get(pol)
                xs += [w for w in (e["written"] if e else []) if w is not None]
            else:
                xs += [_f(r["written_over_budget"]) for r in U[t] if r["policy"] == pol and abs(_f(r["budget_share"]) - b) < 1e-9]
        return f"{st.mean(xs):.2f}" if xs else "--"

    def fmt(x):
        return f"{x:+.1f}".replace("-", "$-$")
    lines = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
             " & \\multicolumn{3}{c}{Known load} & \\multicolumn{3}{c}{Unknown load} \\\\",
             " & \\multicolumn{2}{c}{ROAR lead} & & \\multicolumn{2}{c}{ROAR lead} & \\\\",
             "Policy & TA & CV & W/B & TA & CV & W/B \\\\", "\\midrule"]
    for pol, lab, mac in UNK_POLICIES:
        if pol == "oracle":
            continue
        cells = []
        for fam in ("known", "unknown"):
            for t, tag in tags:
                xs = LEAD.get((fam, t, pol, 0.10))
                cells.append(fmt(100 * st.mean(xs)) if xs and pol != "ROAR" else "--")
            cells.append(wmean(fam, pol))
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(out_dir, "table_unknown.tex"), "w").write("\n".join(lines) + "\n")
    # held-out tuning (synthetic trace) and gain sensitivity (conversation)
    p = os.path.join(RESULTS, "e9_tune_synthetic.csv")
    if os.path.exists(p):
        R = [r for r in _rows(p) if r["policy"] == "ROAR"]
        etas = sorted({_f(r["eta"]) for r in R})
        M["tuneEtaN"] = str(len(etas))
        M["tuneEtaMin"] = f"{min(etas):g}"
        M["tuneEtaMax"] = f"{max(etas):g}"
    p = os.path.join(RESULTS, "e9b_eta_sensitivity_conversation.csv")
    if os.path.exists(p):
        R = _rows(p)
        v = [_f(r["value_share"]) for r in R]
        base = [_f(r["value_share"]) for r in R if abs(_f(r["eta"]) - 0.0003) < 1e-12]
        if base:
            M["etaSensMax"] = f"{100 * max(abs(x - base[0]) for x in v):.1f}"
            M["etaSensLo"] = f"{100 * min(v):.1f}"
            M["etaSensHi"] = f"{100 * max(v):.1f}"
    # gaps computed from the same rounded means the tables print (so text and tables agree)
    for t, tag in tags:
        for b, bn in BUD:
            a, r = M.get(f"v{tag}RoarB{bn}"), M.get(f"v{tag}RoarRefsB{bn}")
            if a and r:
                M[f"refsGap{tag}B{bn}"] = f"{float(a) - float(r):.1f}"
    # does ROAR keep more than every quota (period, load-proportional period, stepwise) on every seed and budget?
    fails = []
    n = 0
    for t, tag in tags:
        for b, bn in BUD:
            ro = D[t]["S"][b].get("ROAR")
            if not ro:
                continue
            rv = dict(zip(ro["pairs"], ro["value"]))
            for q in ("period-quota k=2", "load-prop quota", "stepwise-quota"):
                e = D[t]["S"][b].get(q)
                if not e:
                    continue
                for pair, v in zip(e["pairs"], e["value"]):
                    if pair in rv:
                        n += 1
                        if rv[pair] <= v:
                            fails.append((tag, b, q, pair, 100 * (rv[pair] - v)))
    if n:
        if not fails:
            M["quotaWinsSentence"] = (f"ROAR kept more than each of the three quotas on every seed pair and budget "
                                      f"({n} comparisons).")
        else:
            worst = min(fails, key=lambda x: x[4])
            M["quotaWinsSentence"] = (f"ROAR kept more than each of the three quotas in {n - len(fails)} of {n} seed-pair and "
                                      f"budget comparisons; the exceptions are at 20\\%, by at most "
                                      f"{-worst[4]:.1f} points." if all(f[1] == 0.20 for f in fails) else
                                      f"ROAR kept more than each of the three quotas in {n - len(fails)} of {n} comparisons.")
    # mixed regions on the same dies (E4: half pSLC, half TLC), 8 stacks, tool and agent
    e4 = os.path.join(RESULTS, "e4_cell_modes_toolagent.csv")
    if os.path.exists(e4):
        C = _rows(e4)
        x = [r for r in C if int(r["stacks"]) == 8 and r["config"] == "half pSLC, half TLC" and r["policy"] == "ROAR"]
        if x:
            M["mixedEightGPUh"] = f"{_f(x[0]['flash_saved_gpu_h_per_day']):.1f}"
    # cost-model constants and the inputs of the bound (e11_params.py)
    p = os.path.join(RESULTS, "e11_params_conversation.json")
    if os.path.exists(p):
        e = json.load(open(p))
        M["costPhi"] = f"{e['Phi_flops'] / 1e15:.3g}"
        M["costFlin"] = f"{e['F_lin'] / 1e9:.1f}"
        M["costFatt"] = f"{e['F_att'] / 1e6:.2f}"
        M["costTrec"] = f"{1e3 * e['t_rec_depth0_s']:.0f}"
        M["costTread"] = f"{1e6 * e['t_read_s']:.0f}"
        M["costBlockMiB"] = f"{e['block_bytes'] / 2**20:.0f}"
        M["lamInit"] = f"{e['lambda1']:.3g}"
        M["lamMin"] = f"{e['lambda_min']:.3g}"
        M["rhoMax"] = f"{e['rho_max']:.3g}"
        M["wMax"] = f"{e['w_max']:.2f}"
        M["lamSwitch"] = f"{e['lambda_switch']:.3g}"
        M["boundDaysParams"] = f"{e['bound_days']:.1f}"
    return M
