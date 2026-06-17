from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def compute_time_to_expiry(expiry: pd.Series, asof: pd.Timestamp) -> np.ndarray:
    t = (expiry - asof).dt.total_seconds() / (365.0 * 24 * 3600)
    return np.maximum(t.to_numpy(dtype=float), 1e-8)


def d1(S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float, q: float, sigma: np.ndarray) -> np.ndarray:
    numerator = np.log(S / K) + (r - q + 0.5 * sigma**2) * T
    denominator = sigma * np.sqrt(T)
    return numerator / denominator


def bs_delta(S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float, q: float, sigma: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    _d1 = d1(S, K, T, r, q, sigma)
    call_delta = np.exp(-q * T) * norm.cdf(_d1)
    put_delta = np.exp(-q * T) * (norm.cdf(_d1) - 1.0)
    return np.where(is_call, call_delta, put_delta)


def bs_vega(S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float, q: float, sigma: np.ndarray) -> np.ndarray:
    _d1 = d1(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.pdf(_d1) * np.sqrt(T)


def add_greeks(df: pd.DataFrame, r: float, q: float, asof: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    T = compute_time_to_expiry(out["expiry"], asof)
    S = out["spot"].to_numpy(dtype=float)
    K = out["strike"].to_numpy(dtype=float)
    sigma = out["implied_vol"].to_numpy(dtype=float)
    is_call = out["option_type"].to_numpy(dtype=str) == "call"

    delta = bs_delta(S, K, T, r, q, sigma, is_call)
    vega = bs_vega(S, K, T, r, q, sigma)

    out["T"] = T
    out["delta"] = delta
    out["vega"] = vega
    out["delta_exposure"] = out["delta"] * out["qty"] * out["side"] * out["multiplier"]
    out["vega_exposure"] = out["vega"] * out["qty"] * out["side"] * out["multiplier"]
    return out


def finite_difference_delta(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    q: float,
    sigma: np.ndarray,
    is_call: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    bump_up = bs_price(S * (1.0 + eps), K, T, r, q, sigma, is_call)
    bump_down = bs_price(S * (1.0 - eps), K, T, r, q, sigma, is_call)
    return (bump_up - bump_down) / (2.0 * S * eps)


def finite_difference_vega(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    q: float,
    sigma: np.ndarray,
    is_call: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    bump_up = bs_price(S, K, T, r, q, sigma + eps, is_call)
    bump_down = bs_price(S, K, T, r, q, np.maximum(sigma - eps, 1e-8), is_call)
    return (bump_up - bump_down) / (2.0 * eps)


def bs_price(S: np.ndarray, K: np.ndarray, T: np.ndarray, r: float, q: float, sigma: np.ndarray, is_call: np.ndarray) -> np.ndarray:
    _d1 = d1(S, K, T, r, q, sigma)
    d2 = _d1 - sigma * np.sqrt(T)
    call = S * np.exp(-q * T) * norm.cdf(_d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-_d1)
    return np.where(is_call, call, put)

