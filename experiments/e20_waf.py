"""E20. Closing the wear loop on the device counter when the true write amplification is unknown.
Known-load days 0-1, a device budget of 10% of day 2's write-all bytes times the true WAF (so the host-byte
budget is the usual 10%), day 2 measured. The true WAF is 1.5, 2.0, a step from 1.5 to 2.0 at hour 30, or an
hourly value drawn uniformly from [1.3, 2.2]; the controller never sees it. Three controllers run ROAR:
'naive' believes WAF = 1; 'known' is told the true WAF (reference); 'loop' reads the device's NAND-bytes
counter (OCP SMART-1) and the host counter every hour and uses their ratio over the last six hours.
Reports device writes over device budget on day 2, savings, and the largest lead of device wear over its
schedule. Usage: SHIR_TRACE=toolagent python e20_waf.py"""
import random

from common import DAY_S, HBM_BLOCKS, WriteAll, base_trace, hbm_pass, log, new_estimator, roar, tagged, write_csv
from e3_days import BB, T, sim
from shirroar import LLAMA31_70B, Simulator, WAFLoop, WAFRegion, build_days

B = 0.10


def waf_cases():
    rng = random.Random(5)
    hourly = [rng.uniform(1.3, 2.2) for _ in range(48)]
    return {"1.5": lambda t: 1.5, "2.0": lambda t: 2.0,
            "1.5 -> 2.0 at hour 30": lambda t: 1.5 if t < 30 * 3600.0 else 2.0,
            "hourly U[1.3,2.2]": lambda t: hourly[min(47, int(t // 3600.0))],
            "_mean_hourly": sum(hourly[24:]) / 24.0}


def main():
    s = hbm_pass(build_days(base_trace(), seeds=(0, 1)), HBM_BLOCKS)
    wa = sim(s, WriteAll(), 1e30)
    WA2, V2 = wa.bytes_written[0], wa.saved_flash_s
    cases = waf_cases()
    mean_h = cases.pop("_mean_hourly")
    rows = []
    for cname, fn in cases.items():
        # device budget per day: 10% of write-all host bytes times the day-2 mean true WAF
        waf2 = {"1.5": 1.5, "2.0": 2.0, "1.5 -> 2.0 at hour 30": 2.0}.get(cname, mean_h)
        dev_day = B * WA2 * waf2
        for mode in ("naive", "known", "loop"):
            reg = WAFRegion("flash", BB, 2 * dev_day, T, fn)
            pol = WAFLoop(roar(), mode, measure_from=DAY_S)
            r = Simulator(s, LLAMA31_70B, [reg], pol, tick_s=10.0, t_start=0.0, t_end=T, warmup_s=DAY_S,
                          estimator=new_estimator()).run()
            dev2 = reg.device_bytes - pol.dev_at_measure
            rows.append({"waf_case": cname, "controller": mode, "budget_share": B,
                         "device_written_over_budget": dev2 / dev_day, "value_share": r.saved_flash_s / V2,
                         "max_device_lead_days": pol.max_lead / dev_day,
                         "final_waf_estimate": reg.waf_hat, "true_waf_end": fn(T - 1.0)})
            write_csv(rows, tagged("e20_waf.csv"))
            log(f"  {cname:22s} {mode:5s} dev w/b {rows[-1]['device_written_over_budget']:.3f} "
                f"value {rows[-1]['value_share']:.4f} lead {rows[-1]['max_device_lead_days']:.2f} d "
                f"waf_hat {reg.waf_hat:.2f}")


if __name__ == "__main__":
    main()
