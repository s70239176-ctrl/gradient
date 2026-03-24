"""
Data fetching from Reservoir API + preprocessing pipeline.
Supports both live API calls and synthetic demo data for testing.
"""
import numpy as np
import pandas as pd
import requests
import time
import logging
from typing import Optional, Dict, List, Tuple
from sklearn.preprocessing import RobustScaler
import os

logger = logging.getLogger(__name__)

RESERVOIR_BASE = "https://api.reservoir.tools"
RESERVOIR_API_KEY = os.getenv("RESERVOIR_API_KEY", "demo")

FEATURE_COLS = [
    "floor_price_eth", "volume_eth", "sales_count",
    "floor_pct_change", "volume_ma7", "volume_ma30",
    "floor_ma7", "floor_volatility_7d", "sales_ma7",
    "floor_mom_7d", "floor_mom_30d",
]

# Well-known collections for quick demo
KNOWN_COLLECTIONS = {
    "boredapeyachtclub": "0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d",
    "cryptopunks": "0xb47e3cd837ddf8e4c57f05d70ab865de6e193bbb",
    "mutant-ape-yacht-club": "0x60e4d786628fea6478f785a6d7e704777c86a7c6",
    "clonex": "0x49cf6f5d44e70224e2e23fdcdd2c053f30ada28b",
    "moonbirds": "0x23581767a106ae21c074b2276d25e5c3e136a68b",
    "azuki": "0xed5af388653567af2f388e6224dc7c4b3241c544",
    "pudgypenguins": "0xbd3531da5cf5857e7cfaa92426877b022e612cf8",
    "doodles-official": "0x8a90cab2b38dba80c64b7734e58ee1db38b8992e",
}


def fetch_floor_history(collection_address: str, days: int = 90) -> Optional[pd.DataFrame]:
    """Fetch daily floor + volume from Reservoir API."""
    headers = {"x-api-key": RESERVOIR_API_KEY}
    url = f"{RESERVOIR_BASE}/collections/{collection_address}/daily-volumes/v1"
    params = {"limit": min(days, 365), "startTimestamp": int(time.time()) - days * 86400}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json().get("collections", [])
        if not raw:
            return None

        rows = []
        for day in raw:
            floor = day.get("floorSale", {})
            rows.append({
                "date": pd.to_datetime(day.get("timestamp", 0), unit="s"),
                "floor_price_eth": float(floor.get("1day") or floor.get("7day") or 0),
                "volume_eth": float(day.get("volume") or 0),
                "sales_count": int(day.get("salesCount") or 0),
            })

        df = pd.DataFrame(rows).sort_values("date").set_index("date")
        df = df.replace(0, np.nan).ffill().bfill().fillna(0)
        return df

    except Exception as e:
        logger.warning(f"Reservoir API error for {collection_address}: {e}")
        return None


def fetch_collection_meta(collection_address: str) -> Dict:
    """Fetch collection name, image, supply, owner count."""
    headers = {"x-api-key": RESERVOIR_API_KEY}
    url = f"{RESERVOIR_BASE}/collections/v7"
    try:
        resp = requests.get(url, headers=headers, params={"contract": collection_address}, timeout=8)
        data = resp.json().get("collections", [{}])[0]
        return {
            "name": data.get("name", collection_address[:10] + "..."),
            "image": data.get("image", ""),
            "supply": data.get("tokenCount", 0),
            "owner_count": data.get("ownerCount", 0),
            "floor_ask": data.get("floorAsk", {}).get("price", {}).get("amount", {}).get("decimal", 0),
            "volume_24h": data.get("volume", {}).get("1day", 0),
        }
    except Exception as e:
        logger.warning(f"Meta fetch error: {e}")
        return {"name": "Unknown", "image": "", "supply": 0, "owner_count": 0, "floor_ask": 0, "volume_24h": 0}


def generate_synthetic_data(collection_address: str, days: int = 120) -> pd.DataFrame:
    """
    Generate realistic synthetic NFT floor price data for demo/testing.
    Uses GBM + mean reversion + volume correlation.
    """
    np.random.seed(abs(hash(collection_address)) % (2**31))

    # Base params per "tier" of collection
    addr_hash = abs(hash(collection_address)) % 100
    if addr_hash < 20:      # Blue chip
        base_floor, vol, drift = 50.0, 0.04, 0.0005
    elif addr_hash < 50:    # Mid-tier
        base_floor, vol, drift = 5.0, 0.07, -0.001
    else:                   # Long-tail
        base_floor, vol, drift = 0.5, 0.12, -0.003

    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq="D")
    prices = [base_floor]
    for _ in range(days - 1):
        # GBM with mean reversion
        shock = np.random.normal(drift, vol)
        reversion = -0.05 * (prices[-1] / base_floor - 1)
        prices.append(max(prices[-1] * (1 + shock + reversion), base_floor * 0.05))

    volumes = np.abs(np.random.lognormal(mean=np.log(max(base_floor * 2, 1)), sigma=0.8, size=days))
    # Correlate volume with price moves
    price_changes = np.diff(prices, prepend=prices[0])
    volumes = volumes * (1 + 0.5 * np.abs(price_changes) / (np.array(prices) + 1e-6))
    sales = np.random.poisson(lam=max(int(volumes.mean() / max(base_floor, 0.1)), 5), size=days)

    df = pd.DataFrame({
        "floor_price_eth": prices,
        "volume_eth": volumes,
        "sales_count": sales.astype(float),
    }, index=dates)
    df.index.name = "date"
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator features."""
    df = df.copy()
    df["floor_pct_change"] = df["floor_price_eth"].pct_change().clip(-0.5, 0.5)
    df["volume_ma7"] = df["volume_eth"].rolling(7, min_periods=1).mean()
    df["volume_ma30"] = df["volume_eth"].rolling(30, min_periods=1).mean()
    df["floor_ma7"] = df["floor_price_eth"].rolling(7, min_periods=1).mean()
    df["floor_volatility_7d"] = df["floor_price_eth"].rolling(7, min_periods=2).std().fillna(0)
    df["sales_ma7"] = df["sales_count"].rolling(7, min_periods=1).mean()
    df["floor_mom_7d"] = (df["floor_price_eth"] / df["floor_price_eth"].shift(7).replace(0, np.nan) - 1).fillna(0)
    df["floor_mom_30d"] = (df["floor_price_eth"] / df["floor_price_eth"].shift(30).replace(0, np.nan) - 1).fillna(0)
    return df.fillna(0)


def get_timeseries_window(collection_address: str, lookback: int = 90) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Returns (feature_window, raw_df).
    feature_window shape: (1, lookback, 11) — ready for model input.
    Tries live API first, falls back to synthetic.
    """
    df = fetch_floor_history(collection_address, days=lookback + 40)
    if df is None or len(df) < 30:
        logger.info(f"Using synthetic data for {collection_address}")
        df = generate_synthetic_data(collection_address, days=lookback + 40)

    df = engineer_features(df)

    # Ensure we have enough rows
    df = df.tail(lookback + 10)
    if len(df) < lookback:
        # Pad with repeats if needed
        pad = lookback - len(df)
        df = pd.concat([df.iloc[:1].loc[df.index[:1]].reindex(
            pd.date_range(end=df.index[0], periods=pad+1, freq="D")[:-1]
        ).ffill(), df])

    values = df[FEATURE_COLS].values[-lookback:]
    scaler = RobustScaler()
    values_scaled = scaler.fit_transform(values)

    return values_scaled.reshape(1, lookback, len(FEATURE_COLS)), df


def get_collection_address(slug_or_address: str) -> str:
    """Resolve slug or raw address to checksummed address."""
    slug = slug_or_address.lower().strip()
    if slug in KNOWN_COLLECTIONS:
        return KNOWN_COLLECTIONS[slug]
    if slug.startswith("0x") and len(slug) == 42:
        return slug
    # Try Reservoir slug lookup
    try:
        headers = {"x-api-key": RESERVOIR_API_KEY}
        resp = requests.get(
            f"{RESERVOIR_BASE}/collections/v7",
            headers=headers,
            params={"slug": slug},
            timeout=8,
        )
        colls = resp.json().get("collections", [])
        if colls:
            return colls[0].get("primaryContract", slug)
    except Exception:
        pass
    return slug


def generate_visual_embedding(collection_address: str) -> np.ndarray:
    """
    Generate a deterministic pseudo-CLIP embedding for a collection.
    In production: fetch real images + run CLIP ViT-B/32.
    Here: seeded noise that's consistent per collection.
    """
    np.random.seed(abs(hash(collection_address)) % (2**31))
    emb = np.random.randn(512).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    return emb.reshape(1, 512)
