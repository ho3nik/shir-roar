"""SHIR-ROAR simulator: lifetime-budgeted KV-cache tiering across HBM and High Bandwidth Flash."""
from .controller import PriceController
from .estimator import ReuseEstimator, SessionEstimator
from .flash import Region
from .hbm import EVICT, REF, EventStream, hbm_pass
from .model import B200, CELLS, DEEPSEEK_V3, LLAMA31_70B, MODELS, CellSpec, GPUSpec, ModelSpec
from .oracle import oracle_select
from .policies import (ROAR, AdaptiveK, AdmitOnK, LoadProfile, ProfileQuota, FixedPrice, OracleSet, PeriodQuota, Policy, RandomFraction,
                       StepwiseQuota, WriteAll)
from .sim import Result, Simulator, ideal_savings
from .trace import BLOCK_TOKENS, Request, build_day, build_days, build_days_reinject, build_days_random, check_prefix_hashing, load_mooncake, trace_stats

from .waf import ParentCheck, SkipBlocks, WAFLoop, WAFRegion

__version__ = "0.1.0"
