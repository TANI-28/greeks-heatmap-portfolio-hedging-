from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    risk_free_rate: float = 0.06
    dividend_yield: float = 0.0
    min_iv: float = 0.01
    spot_shock_min: float = -0.10
    spot_shock_max: float = 0.10
    vol_shock_min: float = -0.05
    vol_shock_max: float = 0.05
    grid_size: int = 41
    output_dir: Path = Path("outputs")

