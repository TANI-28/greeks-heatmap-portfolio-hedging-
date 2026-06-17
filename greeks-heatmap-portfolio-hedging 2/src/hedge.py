from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .greeks import bs_delta, bs_vega, compute_time_to_expiry


@dataclass
class HedgeRecommendation:
    stock_units: float
    option_contracts: float
    residual_delta: float
    residual_vega: float


def compute_net_exposure(portfolio: pd.DataFrame) -> Dict[str, float]:
    return {
        "delta": float(portfolio["delta_exposure"].sum()),
        "vega": float(portfolio["vega_exposure"].sum()),
    }


def recommend_two_instrument_hedge(
    portfolio_with_greeks: pd.DataFrame,
    hedge_option_symbol: str,
    hedge_option_type: str,
    hedge_option_strike: float,
    hedge_option_expiry: pd.Timestamp,
    hedge_option_iv: float,
    hedge_option_spot: float,
    multiplier: float,
    asof: pd.Timestamp,
    risk_free_rate: float,
    dividend_yield: float,
) -> HedgeRecommendation:
    net = compute_net_exposure(portfolio_with_greeks)
    net_delta = net["delta"]
    net_vega = net["vega"]

    T = np.array([(hedge_option_expiry - asof).total_seconds() / (365.0 * 24 * 3600)], dtype=float)
    T = np.maximum(T, 1e-8)
    S = np.array([hedge_option_spot], dtype=float)
    K = np.array([hedge_option_strike], dtype=float)
    sigma = np.array([hedge_option_iv], dtype=float)
    is_call = np.array([hedge_option_type == "call"], dtype=bool)

    hedge_delta = float(bs_delta(S, K, T, risk_free_rate, dividend_yield, sigma, is_call)[0]) * multiplier
    hedge_vega = float(bs_vega(S, K, T, risk_free_rate, dividend_yield, sigma)[0]) * multiplier

    A = np.array(
        [
            [1.0, hedge_delta],
            [0.0, hedge_vega],
        ],
        dtype=float,
    )
    b = -np.array([net_delta, net_vega], dtype=float)

    stock_units, option_contracts = np.linalg.solve(A, b)
    residual = A @ np.array([stock_units, option_contracts]) + np.array([net_delta, net_vega])

    return HedgeRecommendation(
        stock_units=float(stock_units),
        option_contracts=float(option_contracts),
        residual_delta=float(residual[0]),
        residual_vega=float(residual[1]),
    )

