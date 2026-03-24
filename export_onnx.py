"""
Export trained NFTFloorForecaster to ONNX + INT8 quantization.
Usage:
    python export_onnx.py --weights backend/model/weights.pt
    python export_onnx.py --weights backend/model/weights.pt --quantize
"""
import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))
from model.nft_forecaster import NFTFloorForecaster


def export(weights_path: str, output_path: str, quantize: bool = True) -> str:
    print(f"Loading weights from: {weights_path}")
    model = NFTFloorForecaster()
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print("  ✓ Loaded trained weights")
    else:
        print("  ⚠ No weights found — exporting random-init model (demo only)")
    model.eval()

    dummy_ts = torch.randn(1, 90, 11)
    dummy_vis = torch.randn(1, 512)

    print(f"\nExporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        (dummy_ts, dummy_vis),
        output_path,
        input_names=["ts_features", "clip_embedding"],
        output_names=["direction_logit", "magnitude"],
        dynamic_axes={
            "ts_features": {0: "batch_size"},
            "clip_embedding": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
    )

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  ✓ Exported — {size_mb:.1f} MB")

    # Validate
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {
            "ts_features": dummy_ts.numpy(),
            "clip_embedding": dummy_vis.numpy(),
        })
        print(f"  ✓ ONNX validation passed — outputs: {[o.shape for o in out]}")
    except ImportError:
        print("  ⚠ onnxruntime not installed — skipping validation")

    if quantize:
        q_path = output_path.replace(".onnx", "_int8.onnx")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(output_path, q_path, weight_type=QuantType.QInt8)
            q_size = os.path.getsize(q_path) / 1e6
            print(f"\n  ✓ INT8 quantized: {q_size:.1f} MB → {q_path}")
            print(f"  Compression: {size_mb:.1f}MB → {q_size:.1f}MB ({100*(1-q_size/size_mb):.0f}% reduction)")
            return q_path
        except ImportError:
            print("  ⚠ onnxruntime.quantization not available — skipping quantization")

    return output_path


def generate_model_card(output_dir: str, horizon: int = 7):
    """Write model card metadata JSON for OpenGradient Hub."""
    import json
    card = {
        "model_name": f"nft-floor-forecaster-{horizon}d-v1",
        "task": "nft_floor_forecast",
        "category": "NFT Forecasting",
        "description": (
            "Multimodal NFT floor price forecaster combining PatchTST time-series "
            "encoding with frozen CLIP ViT-B/32 visual embeddings via gated cross-attention fusion. "
            f"Predicts {horizon}-day floor price movement direction and magnitude."
        ),
        "architecture": {
            "ts_tower": "PatchTST (4-layer, 128-dim, patch_len=9)",
            "vision_tower": "CLIP ViT-B/32 (frozen, 512-dim)",
            "fusion": "GatedFusion (cross-attention gate + MLP)",
            "params": "~2.5M",
        },
        "inputs": {
            "ts_features": {"shape": [1, 90, 11], "dtype": "float32", "description": "90-day time-series window"},
            "clip_embedding": {"shape": [1, 512], "dtype": "float32", "description": "CLIP collection embedding"},
        },
        "outputs": {
            "direction_logit": {"shape": [1, 1], "description": "Apply sigmoid for P(UP)"},
            "magnitude": {"shape": [1, 1], "description": "Predicted % change in [-1, 1]"},
        },
        "onnx_opset": 17,
        "quantization": "INT8 dynamic",
        "inference_modes": ["TEE", "VANILLA"],
        "training_data": "Top ETH collections via Reservoir API + Dune Analytics, 2021-2024",
        "disclaimer": "Not financial advice. NFT markets are highly volatile.",
        "license": "MIT",
    }
    path = os.path.join(output_dir, "model_card.json")
    with open(path, "w") as f:
        json.dump(card, f, indent=2)
    print(f"  ✓ Model card written → {path}")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="backend/model/weights.pt")
    parser.add_argument("--output", default="artifacts/nft_forecaster.onnx")
    parser.add_argument("--quantize", action="store_true", default=True)
    parser.add_argument("--horizon", type=int, default=7)
    args = parser.parse_args()

    os.makedirs("artifacts", exist_ok=True)
    onnx_path = export(args.weights, args.output, args.quantize)
    generate_model_card("artifacts", args.horizon)
    print(f"\n✅ Ready for OpenGradient upload: {onnx_path}")
    print("   Run: python upload_og.py --onnx", onnx_path)
