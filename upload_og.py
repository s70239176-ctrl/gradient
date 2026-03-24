"""
Upload NFTFloorForecaster to OpenGradient Model Hub.
Usage:
    export OG_EMAIL=you@email.com
    export OG_PASSWORD=yourpassword
    export OG_PRIVATE_KEY=0x...
    python upload_og.py --onnx artifacts/nft_forecaster_int8.onnx
"""
import argparse
import json
import os
import sys


MODEL_DESCRIPTION = """
## NFT Floor Price Forecaster (Multimodal) 🔮

**A novel multimodal forecasting model** that combines on-chain time-series analysis
with visual trait embeddings to predict NFT collection floor price movements.

---

### 🏗 Architecture

| Tower | Component | Dim |
|---|---|---|
| Time-Series | PatchTST Encoder (4-layer) | 128-d |
| Vision | CLIP ViT-B/32 (frozen) | 512-d |
| Fusion | Gated Cross-Attention MLP | 128-d |
| Direction Head | Linear + Sigmoid | 1 |
| Magnitude Head | Linear + Tanh | 1 |

**Total params: ~2.5M** · Trains in <1hr on consumer GPU · CPU inference <100ms

---

### 📥 Inputs

| Name | Shape | Type | Description |
|---|---|---|---|
| `ts_features` | `(1, 90, 11)` | float32 | 90-day window of 11 features |
| `clip_embedding` | `(1, 512)` | float32 | CLIP ViT-B/32 collection embedding |

**Time-series features (11):**
floor_price_eth, volume_eth, sales_count, floor_pct_change,
volume_ma7, volume_ma30, floor_ma7, floor_volatility_7d,
sales_ma7, floor_mom_7d, floor_mom_30d

---

### 📤 Outputs

| Name | Shape | Usage |
|---|---|---|
| `direction_logit` | `(1, 1)` | `sigmoid(x)` → P(floor UP >2%) |
| `magnitude` | `(1, 1)` | Multiply × 100 → predicted % change |

---

### 🔐 Verifiable Inference

- ONNX opset 17, INT8 dynamic quantization (~12MB)
- TEE mode: fast trusted execution environment
- ZKML mode: zero-knowledge proof of inference (set batch=1, remove dynamic axes)

---

### 📊 Training Data

- Top 100 Ethereum NFT collections · Reservoir API + Dune Analytics
- Date range: 2021–2024 (~180k training windows)
- Normalization: RobustScaler per-collection

---

### 🚀 Use Cases

- NFT lending protocol floor oracle (Gondi, NFTfi, Arcade)
- Portfolio risk management signals
- Collection alpha screening
- Scheduled on-chain inference via OpenGradient workflows

---

> ⚠️ **Disclaimer**: Research model only. Not financial advice.
> NFT markets are highly volatile. Always verify with additional signals.
"""


def upload(onnx_path: str, model_name: str, dry_run: bool = False):
    try:
        import opengradient as og
    except ImportError:
        print("ERROR: opengradient SDK not installed.")
        print("Run: pip install opengradient")
        sys.exit(1)

    email = os.environ.get("OG_EMAIL")
    password = os.environ.get("OG_PASSWORD")
    private_key = os.environ.get("OG_PRIVATE_KEY")

    if not all([email, password, private_key]):
        print("ERROR: Missing environment variables.")
        print("Required: OG_EMAIL, OG_PASSWORD, OG_PRIVATE_KEY")
        sys.exit(1)

    if dry_run:
        print(f"[DRY RUN] Would upload {onnx_path} as {model_name}")
        return

    print(f"Initializing OpenGradient SDK...")
    og.init(email=email, password=password, private_key=private_key)

    # Load model card
    card_path = "artifacts/model_card.json"
    card = {}
    if os.path.exists(card_path):
        with open(card_path) as f:
            card = json.load(f)

    print(f"Creating model entry: {model_name}")
    og.create_model(
        model_name=model_name,
        model_desc=MODEL_DESCRIPTION,
        model_path=onnx_path,
    )

    print(f"\n✅ Model uploaded!")
    print(f"   Hub URL: https://hub.opengradient.ai/models/{model_name}")
    print(f"\nNext steps:")
    print(f"  1. View your model: https://hub.opengradient.ai/models/{model_name}")
    print(f"  2. Get the model CID from the hub")
    print(f"  3. Run inference: og.infer(model_cid='...', model_input={{...}}, inference_mode=og.InferenceMode.TEE)")


def get_model_cid(model_name: str):
    """Retrieve CID after upload for use in inference calls."""
    try:
        import opengradient as og
        og.init(
            email=os.environ["OG_EMAIL"],
            password=os.environ["OG_PASSWORD"],
            private_key=os.environ["OG_PRIVATE_KEY"],
        )
        # List models to find CID
        # (exact API call depends on SDK version)
        print(f"Retrieving CID for {model_name}...")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="artifacts/nft_forecaster_int8.onnx")
    parser.add_argument("--name", default="nft-floor-forecaster-7d-v1")
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without uploading")
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        print(f"ONNX file not found: {args.onnx}")
        print("Run: python export_onnx.py first")
        sys.exit(1)

    upload(args.onnx, args.name, args.dry_run)
