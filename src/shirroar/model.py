"""Cost models: KV size, recomputation time, and flash endurance arithmetic.

Every constant here is an input assumption with its source noted, not a measurement.
"""
from __future__ import annotations

from dataclasses import dataclass

from .trace import BLOCK_TOKENS

SECONDS_PER_DAY = 86_400.0
YEAR_DAYS = 365.25


@dataclass(frozen=True)
class ModelSpec:
    """Per-token KV bytes and the FLOPs to recompute one token of prefill.

    flops_linear: dense-layer FLOPs per token (2 x active parameters).
    flops_attn: attention FLOPs per query token per earlier context token.
    """
    name: str
    kv_bytes_per_token: int
    flops_linear: float
    flops_attn: float

    def block_bytes(self, block_tokens: int = BLOCK_TOKENS) -> int:
        return self.kv_bytes_per_token * block_tokens

    def recompute_seconds(self, depth_tokens: float, gpu_flops: float, block_tokens: int = BLOCK_TOKENS) -> float:
        """GPU time to recompute one block whose first token sits at depth_tokens.
        t_rec(d) = block_tokens * (F_lin + F_att * d) / Phi  (Eq. 2 of the paper)."""
        return block_tokens * (self.flops_linear + self.flops_attn * depth_tokens) / gpu_flops


# Llama-3.1-70B: 80 layers, 8 KV heads of dim 128, hidden 8,192, 70.6e9 parameters (Llama 3 report).
# KV per token: 2 (K and V) x 80 x 8 x 128 x 2 bytes (BF16) = 327,680 B.
# Attention FLOPs per (query, key) pair: QK^T and AV over 64 heads x 128 = 8,192 dims, 2 FLOPs each,
# so 4 x 8,192 per layer, x 80 layers.
LLAMA31_70B = ModelSpec("Llama-3.1-70B", 2 * 80 * 8 * 128 * 2, 2 * 70.6e9, 4 * 80 * 8192)

# DeepSeek-V3: 61 layers, MLA with a 512-dim latent and a 64-dim decoupled key (KV cache 576 values
# per token per layer, BF16) = 70,272 B per token. 37e9 activated parameters per token.
# Prefill attention (non-absorbed form): 128 heads, QK dim 192, V dim 128 -> 2*(192+128)*128 FLOPs
# per pair per layer, x 61 layers. These are estimates for value ranking, not kernel timings.
DEEPSEEK_V3 = ModelSpec("DeepSeek-V3", 61 * (512 + 64) * 2, 2 * 37e9, 2 * (192 + 128) * 128 * 61)

MODELS = {"llama70b": LLAMA31_70B, "dsv3": DEEPSEEK_V3}


@dataclass(frozen=True)
class GPUSpec:
    name: str
    peak_flops: float
    mfu: float

    @property
    def flops(self) -> float:
        return self.peak_flops * self.mfu


# NVIDIA B200: 2.25 PFLOPS dense BF16 (Lenovo Press LP2226); 50% utilization assumed for prefill.
B200 = GPUSpec("B200", 2.25e15, 0.5)


@dataclass(frozen=True)
class CellSpec:
    """One HBF stack run in one cell mode. capacity in bytes (decimal), rated P/E cycles."""
    name: str
    capacity_bytes: float
    pe_cycles: float
    read_bw: float = 1.6e12   # bytes/s per stack, near the HBF specification's 1.536 TB/s speed grade

    def endurance_bytes(self, waf: float = 1.0) -> float:
        """Host bytes the stack can absorb over its rated life: C * N / WAF."""
        return self.capacity_bytes * self.pe_cycles / waf

    def daily_budget(self, life_years: float = 5.0, waf: float = 1.0) -> float:
        """B = C * N / (D * WAF), bytes per day (Eq. 1)."""
        return self.endurance_bytes(waf) / (life_years * YEAR_DAYS)

    def fifo_residence_seconds(self, life_years: float = 5.0, waf: float = 1.0) -> float:
        """Time a block stays in a FIFO log that is written at exactly its wear budget:
        C / (C*N/(D*WAF)) = D*WAF/N. Independent of capacity."""
        return life_years * YEAR_DAYS * SECONDS_PER_DAY * waf / self.pe_cycles


# Stack presets. SLC 100,000 cycles at 512 GB is what the 2026 HBF papers assume. TLC is rated for
# 1,000-3,000 cycles (3,000 used here) and QLC for about 1,000 under sequential writes. pSLC is a TLC die
# run at one bit per cell: a third of the capacity, rated for 20,000 cycles by Virtium (2021). The
# specification's cube is 512 GiB and names no cell type; the capacities below are assumptions.
SLC_512G = CellSpec("SLC 100K, 512 GB", 512e9, 100_000)
PSLC_512G = CellSpec("pSLC 20K, 512 GB", 512e9, 20_000)   # vendor rating for pSLC on 3D TLC (Virtium, 2021)
TLC_512G = CellSpec("TLC 3K, 512 GB", 512e9, 3_000)
TLC_1536G = CellSpec("TLC 3K, 1.5 TB", 1536e9, 3_000)
QLC_2048G = CellSpec("QLC 1K, 2 TB", 2048e9, 1_000)
CELLS = {"slc": SLC_512G, "pslc": PSLC_512G, "tlc512": TLC_512G, "tlc": TLC_1536G, "qlc": QLC_2048G}


def tokens_per_second_absorbable(cell: CellSpec, model: ModelSpec, life_years: float = 5.0, waf: float = 1.0) -> float:
    """KV tokens per second whose KV one stack can absorb for life_years if each is written once."""
    return cell.daily_budget(life_years, waf) / SECONDS_PER_DAY / model.kv_bytes_per_token
