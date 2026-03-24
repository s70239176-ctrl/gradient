"""
inference_example.py — Run NFT floor forecasts three ways:
  1. Local PyTorch model
  2. Local ONNX model
  3. On-chain via OpenGradient TEE (verifiable)

Usage:
    python inference_example.py --collection boredapeyachtclub
    python inference_example.py --collection 0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d --mode onnx
    python inference_example.py --collection azuki --mode onchain
"""
import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from data.pipeline import (
    get_collection_address, get_timeseries_window,
    generate_visual_embedding, fetch_collection_meta
)


def run_pytorch(collection: str, weights: str = "backend/model/weights.pt") -> dict:
    """Run inference with PyTorch model (no ONNX needed)."""
    from model.nft_forecaster import NFTFloorForecaster
    import torch

    model = NFTFloorForecaster()
    if os.path.exists(weights):
        model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

    address = get_collection_address(collection)
    ts_window, raw_df = get_timeseries_window(address, lookback=90)
    vis_emb = generate_visual_embedding(address)

    result = model.predict(ts_window, vis_emb)
    current_floor = float(raw_df["floor_price_eth"].iloc[-1])
    result["current_floor_eth"] = current_floor
    result["predicted_floor_eth"] = round(current_floor * (1 + result["predicted_pct_change"] / 100), 4)
    result["collection"] = collection
    result["mode"] = "pytorch"
    return result


def run_onnx(collection: str, onnx_path: str = "artifacts/nft_forecaster_int8.onnx") -> dict:
    """Run inference with exported ONNX model."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError("onnxruntime not installed. Run: pip install onnxruntime")

    address = get_collection_address(collection)
    ts_window, raw_df = get_timeseries_window(address, lookback=90)
    vis_emb = generate_visual_embedding(address)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dir_logit, magnitude = sess.run(None, {
        "ts_features": ts_window.astype(np.float32),
        "clip_embedding": vis_emb.astype(np.float32),
    })

    prob_up = float(1 / (1 + np.exp(-dir_logit[0, 0])))
    pct_change = float(magnitude[0, 0]) * 100
    current_floor = float(raw_df["floor_price_eth"].iloc[-1])

    return {
        "collection": collection,
        "direction": "UP" if prob_up > 0.5 else "DOWN",
        "confidence": round(max(prob_up, 1 - prob_up), 4),
        "prob_up": round(prob_up, 4),
        "predicted_pct_change": round(pct_change, 2),
        "current_floor_eth": current_floor,
        "predicted_floor_eth": round(current_floor * (1 + pct_change / 100), 4),
        "mode": "onnx",
    }


def run_onchain(collection: str, model_cid: str) -> dict:
    """
    Run verifiable inference on OpenGradient network.
    Requires: OG_EMAIL, OG_PASSWORD, OG_PRIVATE_KEY env vars
    """
    try:
        import opengradient as og
    except ImportError:
        raise RuntimeError("opengradient not installed. Run: pip install opengradient")

    og.init(
        email=os.environ["OG_EMAIL"],
        password=os.environ["OG_PASSWORD"],
        private_key=os.environ["OG_PRIVATE_KEY"],
    )

    address = get_collection_address(collection)
    ts_window, raw_df = get_timeseries_window(address, lookback=90)
    vis_emb = generate_visual_embedding(address)

    print(f"  Submitting to OpenGradient TEE...")
    tx_hash, result = og.infer(
        model_cid=model_cid,
        model_input={
            "ts_features": ts_window.flatten().tolist(),
            "clip_embedding": vis_emb.flatten().tolist(),
        },
        inference_mode=og.InferenceMode.TEE,
    )

    dir_logit = result.get("direction_logit", [0])[0]
    magnitude = result.get("magnitude", [0])[0]
    prob_up = float(1 / (1 + np.exp(-dir_logit)))
    pct_change = float(magnitude) * 100
    current_floor = float(raw_df["floor_price_eth"].iloc[-1])

    return {
        "collection": collection,
        "direction": "UP" if prob_up > 0.5 else "DOWN",
        "confidence": round(max(prob_up, 1 - prob_up), 4),
        "prob_up": round(prob_up, 4),
        "predicted_pct_change": round(pct_change, 2),
        "current_floor_eth": current_floor,
        "predicted_floor_eth": round(current_floor * (1 + pct_change / 100), 4),
        "tx_hash": tx_hash,
        "mode": "onchain_tee",
        "verifiable": True,
    }


def batch_forecast(collections: list, mode: str = "pytorch") -> list:
    """Run forecasts for multiple collections."""
    results = []
    for col in collections:
        print(f"Forecasting {col}...")
        try:
            if mode == "pytorch":
                r = run_pytorch(col)
            elif mode == "onnx":
                r = run_onnx(col)
            else:
                raise ValueError(f"Use run_onchain() for on-chain mode")
            results.append(r)
            arrow = "▲" if r["direction"] == "UP" else "▼"
            print(f"  {arrow} {r['direction']} {r['predicted_pct_change']:+.1f}% (conf: {r['confidence']:.2f})")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"collection": col, "error": str(e)})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="boredapeyachtclub")
    parser.add_argument("--mode", default="pytorch", choices=["pytorch", "onnx", "onchain"])
    parser.add_argument("--weights", default="backend/model/weights.pt")
    parser.add_argument("--onnx", default="artifacts/nft_forecaster_int8.onnx")
    parser.add_argument("--model-cid", default="", help="Required for --mode onchain")
    parser.add_argument("--batch", action="store_true", help="Run all known collections")
    args = parser.parse_args()

    if args.batch:
        from data.pipeline import KNOWN_COLLECTIONS
        results = batch_forecast(list(KNOWN_COLLECTIONS.keys()), args.mode)
        print("\n=== BATCH RESULTS ===")
        for r in results:
            if "error" not in r:
                arrow = "▲" if r["direction"] == "UP" else "▼"
                print(f"{r['collection']:35s} {arrow} {r['predicted_pct_change']:+6.1f}%  conf={r['confidence']:.2f}  floor={r['current_floor_eth']}Ξ")
    else:
        if args.mode == "pytorch":
            result = run_pytorch(args.collection, args.weights)
        elif args.mode == "onnx":
            result = run_onnx(args.collection, args.onnx)
        else:
            if not args.model_cid:
                print("--model-cid required for onchain mode")
                sys.exit(1)
            result = run_onchain(args.collection, args.model_cid)

        print(json.dumps(result, indent=2))
