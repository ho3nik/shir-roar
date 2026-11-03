"""E11. The constants a reader needs to recompute the wear bound (3) and the cost model (2), from one ROAR
run (conversation, seed pair 0-1, 10% budget): lambda(1), lambda_min, the largest value-to-cost ratio and
interval wear seen, the price at the hand-over from the fast to the slow gain, and the bound in days."""
import json
import math
import os
os.environ.setdefault("SHIR_TRACE", "conversation")
from common import DAY_S, HBM_BLOCKS, ROAR_DEFAULT, WriteAll, base_trace, hbm_pass, new_estimator, roar, tagged, write_json  # noqa: E402
from e3_days import sim  # noqa: E402
from shirroar import B200, LLAMA31_70B, build_days  # noqa: E402
from shirroar.trace import BLOCK_TOKENS  # noqa: E402


def main():
    s = hbm_pass(build_days(base_trace(), seeds=(0, 1)), HBM_BLOCKS)
    wa = sim(s, WriteAll(), 1e30)
    pol = roar()
    r = sim(s, pol, 0.10 * wa.bytes_written[0], est=new_estimator())
    c = pol.ctrl[0]
    out = {"Phi_flops": B200.flops, "F_lin": LLAMA31_70B.flops_linear, "F_att": LLAMA31_70B.flops_attn,
           "block_bytes": LLAMA31_70B.block_bytes(), "t_rec_depth0_s": LLAMA31_70B.recompute_seconds(0, B200.flops),
           "t_read_s": LLAMA31_70B.block_bytes() / 6.4e12, "block_tokens": BLOCK_TOKENS,
           "lambda1": c.lam0, "lambda_min": c.lam_min, "rho_max": pol.max_rho, "w_max": pol.max_w,
           "lambda_switch": pol.lam_switch[0] if pol.lam_switch else None, "eta0": ROAR_DEFAULT["eta0"],
           "eta": ROAR_DEFAULT["eta"], "bound_intervals": pol.bound_intervals(),
           "bound_days": pol.bound_intervals() * 10.0 / DAY_S,
           "phase1_intervals": math.log(max(pol.max_rho, c.lam0) / c.lam0) / ROAR_DEFAULT["eta0"] + pol.max_w - 1.0,
           "flash_hits": r.flash_hits, "blocks_written": r.blocks_written[0]}
    write_json(out, tagged("e11_params.json"))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
