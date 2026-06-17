from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .data_loader import build_market_table, clean_and_merge, load_positions
from .greeks import add_greeks, bs_delta, bs_vega, compute_time_to_expiry, finite_difference_delta, finite_difference_vega
from .hedge import compute_net_exposure, recommend_two_instrument_hedge
from .scenarios import run_scenario_grid
from .visualize import save_heatmap


def validate_greeks(df: pd.DataFrame, r: float, q: float, asof: pd.Timestamp) -> pd.DataFrame:
    S = df["spot"].to_numpy(dtype=float)
    K = df["strike"].to_numpy(dtype=float)
    T = compute_time_to_expiry(df["expiry"], asof)
    sigma = df["implied_vol"].to_numpy(dtype=float)
    is_call = df["option_type"].to_numpy(dtype=str) == "call"

    analytic_delta = bs_delta(S, K, T, r, q, sigma, is_call)
    analytic_vega = bs_vega(S, K, T, r, q, sigma)
    fd_delta = finite_difference_delta(S, K, T, r, q, sigma, is_call)
    fd_vega = finite_difference_vega(S, K, T, r, q, sigma, is_call)

    checks = pd.DataFrame(
        {
            "symbol": df["symbol"],
            "option_type": df["option_type"],
            "strike": df["strike"],
            "delta_abs_error": np.abs(analytic_delta - fd_delta),
            "vega_abs_error": np.abs(analytic_vega - fd_vega),
        }
    )
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Greeks sensitivity heatmap for portfolio hedging")
    parser.add_argument("--positions", default="data/positions_sample.csv", help="Path to positions CSV")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig(output_dir=Path(args.output_dir))
    positions = load_positions(args.positions)
    spot_table, chain_table = build_market_table(positions)
    merged = clean_and_merge(positions, spot_table, chain_table)
    if merged.empty:
        raise RuntimeError("No rows remained after merge/cleaning. Adjust positions or market data inputs.")

    asof = pd.Timestamp.utcnow().tz_localize(None)
    portfolio = add_greeks(merged, config.risk_free_rate, config.dividend_yield, asof)
    exposure = compute_net_exposure(portfolio)

    spot_shocks = np.linspace(config.spot_shock_min, config.spot_shock_max, config.grid_size)
    vol_shocks = np.linspace(config.vol_shock_min, config.vol_shock_max, config.grid_size)
    scenario_result = run_scenario_grid(
        portfolio=portfolio,
        asof=asof,
        risk_free_rate=config.risk_free_rate,
        dividend_yield=config.dividend_yield,
        min_iv=config.min_iv,
        spot_shocks=spot_shocks,
        vol_shocks=vol_shocks,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_heatmap(
        matrix=scenario_result.delta_matrix,
        spot_shocks=spot_shocks,
        vol_shocks=vol_shocks,
        title="Portfolio Delta Exposure Heatmap",
        output_path=config.output_dir / "delta_heatmap.png",
        cbar_label="Net Delta Exposure",
    )
    save_heatmap(
        matrix=scenario_result.vega_matrix,
        spot_shocks=spot_shocks,
        vol_shocks=vol_shocks,
        title="Portfolio Vega Exposure Heatmap",
        output_path=config.output_dir / "vega_heatmap.png",
        cbar_label="Net Vega Exposure",
    )

    greek_checks = validate_greeks(portfolio, config.risk_free_rate, config.dividend_yield, asof)
    greek_checks.to_csv(config.output_dir / "greek_validation.csv", index=False)
    portfolio.to_csv(config.output_dir / "portfolio_with_greeks.csv", index=False)

    hedge_row = portfolio.sort_values("vega", ascending=False).iloc[0]
    hedge = recommend_two_instrument_hedge(
        portfolio_with_greeks=portfolio,
        hedge_option_symbol=str(hedge_row["symbol"]),
        hedge_option_type=str(hedge_row["option_type"]),
        hedge_option_strike=float(hedge_row["strike"]),
        hedge_option_expiry=pd.Timestamp(hedge_row["expiry"]),
        hedge_option_iv=float(hedge_row["implied_vol"]),
        hedge_option_spot=float(hedge_row["spot"]),
        multiplier=float(hedge_row["multiplier"]),
        asof=asof,
        risk_free_rate=config.risk_free_rate,
        dividend_yield=config.dividend_yield,
    )

    summary = pd.DataFrame(
        [
            {"metric": "net_delta_before", "value": exposure["delta"]},
            {"metric": "net_vega_before", "value": exposure["vega"]},
            {"metric": "recommended_stock_units", "value": hedge.stock_units},
            {"metric": "recommended_option_contracts", "value": hedge.option_contracts},
            {"metric": "residual_delta_after", "value": hedge.residual_delta},
            {"metric": "residual_vega_after", "value": hedge.residual_vega},
            {"metric": "max_delta_fd_error", "value": float(greek_checks["delta_abs_error"].max())},
            {"metric": "max_vega_fd_error", "value": float(greek_checks["vega_abs_error"].max())},
        ]
    )
    summary.to_csv(config.output_dir / "summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()

