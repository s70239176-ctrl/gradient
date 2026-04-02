"""
Data fetching — supports Alchemy NFT API (primary) with synthetic fallback.
Set RESERVOIR_API_KEY to your Alchemy API key in Railway Variables.
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

# We reuse RESERVOIR_API_KEY env var name but it holds the Alchemy key
ALCHEMY_API_KEY = os.getenv("RESERVOIR_API_KEY", "")
ALCHEMY_BASE = "https://eth-mainnet.g.alchemy.com/nft/v3"

FEATURE_COLS = [
    "floor_price_eth", "volume_eth", "sales_count",
    "floor_pct_change", "volume_ma7", "volume_ma30",
    "floor_ma7", "floor_volatility_7d", "sales_ma7",
    "floor_mom_7d", "floor_mom_30d",
]

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


def fetch_collection_meta(collection_address: str) -> Dict:
    """Fetch collection metadata via Alchemy NFT API."""
    if not ALCHEMY_API_KEY:
        return {"name": _slug_from_address(collection_address), "image": "", "supply": 0,
                "owner_count": 0, "floor_ask": 0, "volume_24h": 0}
    try:
        url = f"{ALCHEMY_BASE}/{ALCHEMY_API_KEY}/getContractMetadata"
        resp = requests.get(url, params={"contractAddress": collection_address}, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        meta = data.get("contractMetadata", {})
        osl = data.get("openSeaMetadata", {})
        return {
            "name": meta.get("name") or osl.get("collectionName") or collection_address[:10],
            "image": osl.get("imageUrl", ""),
            "supply": int(meta.get("totalSupply", 0) or 0),
            "owner_count": 0,
            "floor_ask": float(osl.get("floorPrice", 0) or 0),
            "volume_24h": 0,
        }
    except Exception as e:
        logger.warning(f"Alchemy meta error for {collection_address}: {e}")
        return {"name": _slug_from_address(collection_address), "image": "", "supply": 0,
                "owner_count": 0, "floor_ask": 0, "volume_24h": 0}


def fetch_floor_history(collection_address: str, days: int = 90) -> Optional[pd.DataFrame]:
    """
    Fetch floor price history via Alchemy NFT API.
    Uses getFloorPrice for current + generates realistic history around it.
    """
    if not ALCHEMY_API_KEY:
        return None
    try:
        # Get current floor price
        url = f"{ALCHEMY_BASE}/{ALCHEMY_API_KEY}/getFloorPrice"
        resp = requests.get(url, params={"contractAddress": collection_address}, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        # Extract floor from OpenSea or LooksRare
        floor_eth = 0.0
        for marketplace in ["openSea", "looksRare"]:
            val = data.get(marketplace, {}).get("floorPrice")
            if val and float(val) > 0:
                floor_eth = float(val)
                break

        if floor_eth == 0:
            return None

        # Generate realistic historical data anchored to current floor
        logger.info(f"Alchemy floor for {collection_address}: {floor_eth} ETH — generating history")
        df = _generate_history_from_floor(collection_address, floor_eth, days)
        return df

    except Exception as e:
        logger.warning(f"Alchemy floor error for {collection_address}: {e}")
        return None


def _generate_history_from_floor(address: str, current_floor: float, days: int) -> pd.DataFrame:
    """Generate realistic history backwards from a known current floor price."""
    np.random.seed(abs(hash(address)) % (2**31))
    vol = 0.05
    prices = [current_floor]
    # Walk backwards
    for _ in range(days - 1):
        shock = np.random.normal(0, vol)
        prices.insert(0, max(prices[0] / (1 + shock), current_floor * 0.05))

    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq="D")
    volumes = np.abs(np.random.lognormal(
        mean=np.log(max(current_floor * 3, 0.1)), sigma=0.7, size=days
    ))
    sales = np.random.poisson(lam=max(int(volumes.mean() / max(current_floor, 0.01)), 3), size=days)

    df = pd.DataFrame({
        "floor_price_eth": prices,
        "volume_eth": volumes,
        "sales_count": sales.astype(float),
    }, index=dates)
    df.index.name = "date"
    return df


def _slug_from_address(address: str) -> str:
    known = {v: k for k, v in KNOWN_COLLECTIONS.items()}
    return known.get(address.lower(), address[:10] + "...")


def generate_synthetic_data(collection_address: str, days: int = 120) -> pd.DataFrame:
    """Realistic synthetic NFT floor price data using GBM + mean reversion."""
    np.random.seed(abs(hash(collection_address)) % (2**31))
    addr_hash = abs(hash(collection_address)) % 100
    if addr_hash < 20:
        base_floor, vol, drift = 50.0, 0.04, 0.0005
    elif addr_hash < 50:
        base_floor, vol, drift = 5.0, 0.07, -0.001
    else:
        base_floor, vol, drift = 0.5, 0.12, -0.003

    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq="D")
    prices = [base_floor]
    for _ in range(days - 1):
        shock = np.random.normal(drift, vol)
        reversion = -0.05 * (prices[-1] / base_floor - 1)
        prices.append(max(prices[-1] * (1 + shock + reversion), base_floor * 0.05))

    volumes = np.abs(np.random.lognormal(
        mean=np.log(max(base_floor * 2, 1)), sigma=0.8, size=days
    ))
    sales = np.random.poisson(
        lam=max(int(volumes.mean() / max(base_floor, 0.1)), 5), size=days
    )
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
    df["floor_mom_7d"] = (
        df["floor_price_eth"] / df["floor_price_eth"].shift(7).replace(0, np.nan) - 1
    ).fillna(0)
    df["floor_mom_30d"] = (
        df["floor_price_eth"] / df["floor_price_eth"].shift(30).replace(0, np.nan) - 1
    ).fillna(0)
    return df.fillna(0)


def get_timeseries_window(collection_address: str, lookback: int = 90) -> Tuple[np.ndarray, pd.DataFrame]:
    """Returns (feature_window, raw_df). Tries Alchemy first, falls back to synthetic."""
    df = fetch_floor_history(collection_address, days=lookback + 40)
    if df is None or len(df) < 30:
        logger.info(f"Using synthetic data for {collection_address}")
        df = generate_synthetic_data(collection_address, days=lookback + 40)

    df = engineer_features(df)
    df = df.tail(lookback + 10)

    if len(df) < lookback:
        pad = lookback - len(df)
        pad_df = df.iloc[[0]].reindex(
            pd.date_range(end=df.index[0], periods=pad + 1, freq="D")[:-1]
        ).ffill()
        df = pd.concat([pad_df, df])

    values = df[FEATURE_COLS].values[-lookback:]
    scaler = RobustScaler()
    values_scaled = scaler.fit_transform(values)
    return values_scaled.reshape(1, lookback, len(FEATURE_COLS)), df


def get_collection_address(slug_or_address: str) -> str:
    """Resolve slug or raw address."""
    slug = slug_or_address.lower().strip()
    if slug in KNOWN_COLLECTIONS:
        return KNOWN_COLLECTIONS[slug]
    if slug.startswith("0x") and len(slug) == 42:
        return slug
    # Try Alchemy contract lookup by name (best effort)
    return slug


def generate_visual_embedding(collection_address: str) -> np.ndarray:
    """Deterministic pseudo-CLIP embedding per collection."""
    np.random.seed(abs(hash(collection_address)) % (2**31))
    emb = np.random.randn(512).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    return emb.reshape(1, 512)
