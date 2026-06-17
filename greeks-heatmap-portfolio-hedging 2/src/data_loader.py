from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
import yfinance as yf


REQUIRED_POSITION_COLUMNS = [
    "symbol",
    "option_type",
    "strike",
    "expiry",
    "qty",
    "side",
    "multiplier",
]


@dataclass
class MarketSnapshot:
    symbol: str
    spot: float
    timestamp: pd.Timestamp


def load_positions(path: str) -> pd.DataFrame:
    positions = pd.read_csv(path)
    missing = [col for col in REQUIRED_POSITION_COLUMNS if col not in positions.columns]
    if missing:
        raise ValueError(f"Missing required columns in positions file: {missing}")

    positions["option_type"] = positions["option_type"].str.lower().str.strip()
    positions["expiry"] = pd.to_datetime(positions["expiry"], errors="coerce")
    positions["strike"] = pd.to_numeric(positions["strike"], errors="coerce")
    positions["qty"] = pd.to_numeric(positions["qty"], errors="coerce")
    positions["side"] = pd.to_numeric(positions["side"], errors="coerce")
    positions["multiplier"] = pd.to_numeric(positions["multiplier"], errors="coerce")

    positions = positions.dropna(subset=["expiry", "strike", "qty", "side", "multiplier"])
    positions = positions[positions["option_type"].isin(["call", "put"])]
    positions = positions[(positions["side"] == 1) | (positions["side"] == -1)]
    positions = positions[positions["qty"] > 0]
    positions = positions.reset_index(drop=True)
    return positions


def fetch_spot(symbol: str) -> MarketSnapshot:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="5d", interval="1d")
    if hist.empty:
        raise ValueError(f"Unable to fetch spot for symbol {symbol}")

    spot = float(hist["Close"].iloc[-1])
    ts = pd.Timestamp(hist.index[-1]).tz_localize(None)
    return MarketSnapshot(symbol=symbol, spot=spot, timestamp=ts)


def fetch_option_chain(symbol: str, expiry: pd.Timestamp) -> pd.DataFrame:
    expiry_str = expiry.strftime("%Y-%m-%d")
    ticker = yf.Ticker(symbol)
    chain = ticker.option_chain(expiry_str)

    calls = chain.calls.copy()
    calls["option_type"] = "call"
    puts = chain.puts.copy()
    puts["option_type"] = "put"
    raw = pd.concat([calls, puts], ignore_index=True)

    normalized = raw.rename(
        columns={
            "strike": "strike",
            "impliedVolatility": "implied_vol",
            "bid": "bid",
            "ask": "ask",
            "lastTradeDate": "last_trade_date",
            "openInterest": "open_interest",
            "volume": "volume",
        }
    )
    keep_cols = ["strike", "option_type", "implied_vol", "bid", "ask", "last_trade_date", "open_interest", "volume"]
    normalized = normalized[keep_cols]
    normalized["expiry"] = expiry
    return normalized


def build_market_table(positions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    snapshots = []
    option_frames = []

    for symbol in positions["symbol"].unique():
        snapshot = fetch_spot(symbol)
        snapshots.append(snapshot.__dict__)

        symbol_positions = positions[positions["symbol"] == symbol]
        for expiry in symbol_positions["expiry"].drop_duplicates().sort_values():
            chain = fetch_option_chain(symbol, expiry)
            chain["symbol"] = symbol
            option_frames.append(chain)

    spot_table = pd.DataFrame(snapshots)
    chain_table = pd.concat(option_frames, ignore_index=True)
    return spot_table, chain_table


def clean_and_merge(positions: pd.DataFrame, spot_table: pd.DataFrame, chain_table: pd.DataFrame) -> pd.DataFrame:
    chain = chain_table.copy()
    chain["implied_vol"] = pd.to_numeric(chain["implied_vol"], errors="coerce")
    chain["bid"] = pd.to_numeric(chain["bid"], errors="coerce")
    chain["ask"] = pd.to_numeric(chain["ask"], errors="coerce")
    chain = chain.dropna(subset=["implied_vol"])
    chain = chain[chain["implied_vol"] > 0]

    merged = positions.merge(
        chain[["symbol", "expiry", "strike", "option_type", "implied_vol", "bid", "ask", "open_interest", "volume"]],
        on=["symbol", "expiry", "strike", "option_type"],
        how="left",
    )
    merged = merged.merge(spot_table[["symbol", "spot", "timestamp"]], on="symbol", how="left")
    merged = merged.dropna(subset=["spot", "implied_vol"])
    return merged.reset_index(drop=True)

