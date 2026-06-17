from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .greeks import bs_delta, bs_vega, compute_time_to_expiry


@dataclass
class ScenarioResult:
    spot_shocks: np.ndarray
    vol_shocks: np.ndarray
    delta_matrix: np.ndarray
    vega_matrix: np.ndarray


def run_scenario_grid(
    portfolio: pd.DataFrame,
    asof: pd.Timestamp,
    risk_free_rate: float,
    dividend_yield: float,
    min_iv: float,
    spot_shocks: np.ndarray,
    vol_shocks: np.ndarray,
) -> ScenarioResult:
    S0 = portfolio["spot"].to_numpy(dtype=float)
    K = portfolio["strike"].to_numpy(dtype=float)
    T = compute_time_to_expiry(portfolio["expiry"], asof)
    sigma0 = portfolio["implied_vol"].to_numpy(dtype=float)
    is_call = portfolio["option_type"].to_numpy(dtype=str) == "call"
    weight = (portfolio["qty"] * portfolio["side"] * portfolio["multiplier"]).to_numpy(dtype=float)

    delta_matrix = np.zeros((len(vol_shocks), len(spot_shocks)))
    vega_matrix = np.zeros((len(vol_shocks), len(spot_shocks)))

    for i, v_shock in enumerate(vol_shocks):
        sigma = np.maximum(sigma0 + v_shock, min_iv)
        for j, s_shock in enumerate(spot_shocks):
            S = S0 * (1.0 + s_shock)
            delta = bs_delta(S, K, T, risk_free_rate, dividend_yield, sigma, is_call)
            vega = bs_vega(S, K, T, risk_free_rate, dividend_yield, sigma)
            delta_matrix[i, j] = np.sum(delta * weight)
            vega_matrix[i, j] = np.sum(vega * weight)

    return ScenarioResult(
        spot_shocks=spot_shocks,
        vol_shocks=vol_shocks,
        delta_matrix=delta_matrix,
        vega_matrix=vega_matrix,
    )

