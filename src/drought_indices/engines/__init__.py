from __future__ import annotations

from ..config import Config
from .base import Engine, Inputs, Job
from .climate_indices_engine import ClimateIndicesEngine
from .xclim_engine import XclimEngine

__all__ = ["Engine", "Inputs", "Job", "make_engine"]


def make_engine(name: str, config: Config) -> Engine:
    cfg = config["engines"][name]
    compute_cfg = config["compute"]
    if name == "climate_indices":
        return ClimateIndicesEngine(cfg, compute_cfg)
    if name.startswith("xclim"):
        return XclimEngine(name, cfg, compute_cfg)
    raise ValueError(f"Unknown engine {name!r}")
