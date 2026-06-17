# Greeks Sensitivity Heatmap for Portfolio Hedging

This project builds an end-to-end quant risk tool that:

- Ingests an options portfolio
- Pulls live market data from Yahoo Finance
- Computes Black-Scholes Delta and Vega exposures
- Runs a spot-volatility scenario grid
- Generates heatmaps for portfolio risk concentration
- Produces a two-instrument hedge recommendation
- Validates analytic Greeks against finite differences

It is designed to be resume-defendable for quant interviews (Optiver/IMC/Jane Street/HRT/Citadel style).

## Tech Stack

- Python
- NumPy
- pandas
- matplotlib
- yfinance
- scipy
- pytest

## Repository Structure

```text
greeks-heatmap-portfolio-hedging/
  data/
    positions_sample.csv
  outputs/                         # generated at runtime
  src/
    config.py
    data_loader.py
    greeks.py
    scenarios.py
    hedge.py
    visualize.py
    main.py
  tests/
    test_greeks.py
  requirements.txt
  README.md
```

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m src.main --positions data/positions_sample.csv --output-dir outputs
```

## Outputs

The run generates:

- `outputs/delta_heatmap.png`
- `outputs/vega_heatmap.png`
- `outputs/portfolio_with_greeks.csv`
- `outputs/greek_validation.csv`
- `outputs/summary.csv`

`summary.csv` includes key metrics:

- Net Delta/Vega before hedge
- Recommended stock and option hedge quantities
- Residual exposures after hedge
- Max analytic-vs-finite-difference Greek errors

## Data and Assumptions

- Uses Yahoo Finance options chains for prototyping.
- Black-Scholes assumptions: lognormal dynamics, constant volatility, continuous trading.
- Rates/dividend defaults are configurable in `src/config.py`.

## Validation

Run tests:

```bash
pytest -q
```

The tests compare analytic Delta/Vega to central finite-difference approximations.



## Extending to Production

For hedge-fund-grade implementation, add:

- Smile-aware surface construction
- Real-time feed synchronization and staleness controls
- Liquidity and transaction-cost aware constrained optimizer
- OMS/EMS integration for live hedge execution
- Monitoring, alerts, and audit trail

## License

Educational use for portfolio/interview preparation.
