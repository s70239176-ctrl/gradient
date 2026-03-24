"""
NFT Floor Forecaster — FastAPI Backend
Endpoints:
  GET  /                     → API info
  POST /api/forecast         → Run forecast for a collection
  GET  /api/collections      → List known collections
  GET  /api/health           → Health check
  GET  /api/chart/{address}  → Historical chart data
"""
import os
import time
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np

from data.pipeline import (
    get_timeseries_window, fetch_collection_meta, get_collection_address,
    generate_visual_embedding, KNOWN_COLLECTIONS, engineer_features,
    generate_synthetic_data, fetch_floor_history,
)
from model.nft_forecaster import NFTFloorForecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NFT Floor Forecaster",
    description="Multimodal NFT floor price prediction API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
_model: Optional[NFTFloorForecaster] = None

def get_model() -> NFTFloorForecaster:
    global _model
    if _model is None:
        _model = NFTFloorForecaster()
        weights_path = os.getenv("MODEL_WEIGHTS_PATH", "model/weights.pt")
        if os.path.exists(weights_path):
            import torch
            _model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            logger.info(f"Loaded weights from {weights_path}")
        else:
            logger.info("No weights found — using random init (demo mode)")
        _model.eval()
    return _model


class ForecastRequest(BaseModel):
    collection: str          # slug or 0x address
    horizon: int = 7         # 7 or 30 days
    include_chart: bool = True


class ForecastResponse(BaseModel):
    collection_address: str
    collection_name: str
    collection_image: str
    direction: str
    confidence: float
    prob_up: float
    predicted_pct_change: float
    horizon_days: int
    current_floor_eth: float
    predicted_floor_eth: float
    chart_data: Optional[list] = None
    data_source: str
    generated_at: float


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/api/collections")
async def list_collections():
    return {
        "collections": [
            {"slug": slug, "address": addr}
            for slug, addr in KNOWN_COLLECTIONS.items()
        ]
    }


@app.post("/api/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest):
    t0 = time.time()

    if req.horizon not in (7, 14, 30):
        raise HTTPException(400, "horizon must be 7, 14, or 30")

    # Resolve collection
    address = get_collection_address(req.collection)

    # Fetch time-series
    try:
        ts_window, raw_df = get_timeseries_window(address, lookback=90)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(500, f"Data pipeline error: {str(e)}")

    # Visual embedding
    vis_emb = generate_visual_embedding(address)

    # Run model
    model = get_model()
    result = model.predict(ts_window, vis_emb)

    # Scale up confidence by horizon (longer = less certain)
    horizon_penalty = {7: 1.0, 14: 0.9, 30: 0.78}
    result["confidence"] = round(result["confidence"] * horizon_penalty[req.horizon], 4)
    result["predicted_pct_change"] = round(
        result["predicted_pct_change"] * (req.horizon / 7) ** 0.6, 2
    )

    # Fetch metadata
    meta = fetch_collection_meta(address)
    current_floor = float(raw_df["floor_price_eth"].iloc[-1])
    if current_floor == 0:
        current_floor = meta.get("floor_ask", 1.0) or 1.0

    predicted_floor = current_floor * (1 + result["predicted_pct_change"] / 100)

    # Build chart data
    chart_data = None
    if req.include_chart:
        chart_df = raw_df.tail(90)
        chart_data = [
            {
                "date": str(idx.date()),
                "floor": round(float(row["floor_price_eth"]), 4),
                "volume": round(float(row["volume_eth"]), 2),
                "sales": int(row["sales_count"]),
            }
            for idx, row in chart_df.iterrows()
        ]

    # Determine data source
    data_source = "reservoir_api" if os.getenv("RESERVOIR_API_KEY", "demo") != "demo" else "synthetic_demo"

    logger.info(f"Forecast for {address} in {time.time()-t0:.2f}s — {result['direction']} {result['predicted_pct_change']}%")

    return ForecastResponse(
        collection_address=address,
        collection_name=meta.get("name", req.collection),
        collection_image=meta.get("image", ""),
        direction=result["direction"],
        confidence=result["confidence"],
        prob_up=result["prob_up"],
        predicted_pct_change=result["predicted_pct_change"],
        horizon_days=req.horizon,
        current_floor_eth=round(current_floor, 4),
        predicted_floor_eth=round(predicted_floor, 4),
        chart_data=chart_data,
        data_source=data_source,
        generated_at=time.time(),
    )


@app.get("/api/chart/{address}")
async def chart_data(address: str, days: int = 90):
    resolved = get_collection_address(address)
    _, raw_df = get_timeseries_window(resolved, lookback=days)
    return {
        "address": resolved,
        "data": [
            {
                "date": str(idx.date()),
                "floor": round(float(row["floor_price_eth"]), 4),
                "volume": round(float(row["volume_eth"]), 2),
            }
            for idx, row in raw_df.tail(days).iterrows()
        ]
    }


# Serve React frontend from /frontend/dist
FRONTEND_DIR = "/app/frontend/dist"
if os.path.exists(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=f"{FRONTEND_DIR}/assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = f"{FRONTEND_DIR}/{full_path}"
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(f"{FRONTEND_DIR}/index.html")
