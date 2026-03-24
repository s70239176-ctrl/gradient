# ◈ NFT Floor Forecaster

**Multimodal AI forecasting for NFT collection floor prices.**  
Combines on-chain time-series (PatchTST) with visual trait embeddings (CLIP) via gated fusion.  
Deployable on Railway · Uploadable to OpenGradient Hub for verifiable on-chain inference.

![Architecture](https://img.shields.io/badge/model-PatchTST%20%2B%20CLIP-7c5cfc?style=flat-square)
![Params](https://img.shields.io/badge/params-~2.5M-5cf0a0?style=flat-square)
![ONNX](https://img.shields.io/badge/export-ONNX%20opset%2017-fc5c7d?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)

---

## Architecture

```
[Image Tower]                    [Time-Series Tower]
CLIP ViT-B/32 (frozen)          PatchTST (4-layer, 128-dim)
512-d collection embedding       90-day window, 11 features
         │                               │
         └──────── GatedFusion ──────────┘
                        │
              Direction Head + Magnitude Head
                  P(UP) + % change
```

**Inputs:**
- `ts_features` — 90-day window: floor price, volume, sales, 7 derived features
- `clip_embedding` — CLIP ViT-B/32 mean-pooled over 50 collection images

**Outputs:**
- `direction_logit` → apply sigmoid → P(floor UP >2% in N days)
- `magnitude` → × 100 → predicted % change

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOU/nft-floor-forecaster
cd nft-floor-forecaster

# Backend
cd backend && pip install -r requirements.txt && cd ..

# Frontend
cd frontend && npm install && cd ..

# Copy env
cp .env.example .env
```

### 2. Run Locally (Demo Mode)

Works out of the box with synthetic data — no API keys needed:

```bash
# Terminal 1: Backend
cd backend && uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd frontend && npm run dev
```

Visit `http://localhost:5173` — forecasts load automatically for top collections.

---

## Training Your Own Model

### Step 1: Get a Reservoir API key (free)

Sign up at [reservoir.tools/api-keys](https://reservoir.tools/api-keys) — free tier gives 50 req/min.

```bash
# Add to .env
RESERVOIR_API_KEY=your_key_here
```

### Step 2: Train

```bash
# Quick test (synthetic data, CPU, 5 min)
python train.py --collections 8 --epochs 20 --horizon 7

# Full training (real data, GPU recommended)
python train.py --collections 100 --epochs 60 --horizon 7 --device cuda

# 30-day horizon model
python train.py --collections 100 --epochs 60 --horizon 30 --output backend/model/weights_30d.pt
```

Weights are saved to `backend/model/weights.pt` — the API server loads them automatically.

**Expected results on 8 synthetic collections:**
```
Epoch  20/20 | loss=0.6821 | val_loss=0.6714 | val_acc=0.557 | 3.2s
Training complete! Best val_acc=0.571
```

Real collection data significantly improves accuracy — synthetic data is for pipeline validation only.

### Step 3: Export to ONNX

```bash
# Export + INT8 quantize
python export_onnx.py --weights backend/model/weights.pt --quantize

# Output: artifacts/nft_forecaster_int8.onnx (~12MB)
```

### Step 4: Upload to OpenGradient

```bash
export OG_EMAIL=you@email.com
export OG_PASSWORD=yourpassword
export OG_PRIVATE_KEY=0x...

python upload_og.py --onnx artifacts/nft_forecaster_int8.onnx --name nft-floor-forecaster-v1
```

---

## Deploy to Railway

### Option A: One-Click (Recommended)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/YOU/nft-floor-forecaster)

### Option B: Manual Deploy

1. **Push to GitHub**
```bash
git init && git add . && git commit -m "Initial commit"
git remote add origin https://github.com/YOU/nft-floor-forecaster
git push -u origin main
```

2. **Create Railway project**
   - Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
   - Select your repo
   - Railway auto-detects the Dockerfile

3. **Set environment variables** in Railway dashboard:
   ```
   RESERVOIR_API_KEY    = your_reservoir_key     (optional, for live data)
   MODEL_WEIGHTS_PATH   = backend/model/weights.pt
   PORT                 = 8000                   (auto-set by Railway)
   ```

4. **Deploy** — Railway builds and deploys automatically on every push to `main`

5. **Get your URL** — Railway assigns `https://nft-forecaster-xxx.up.railway.app`

### Option C: Railway CLI

```bash
npm install -g @railway/cli
railway login
railway init
railway up
```

---

## API Reference

### `POST /api/forecast`

```bash
curl -X POST https://your-app.railway.app/api/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "boredapeyachtclub",
    "horizon": 7,
    "include_chart": true
  }'
```

**Parameters:**
| Field | Type | Default | Description |
|---|---|---|---|
| `collection` | string | required | Slug (e.g. `azuki`) or `0x` address |
| `horizon` | int | 7 | Forecast horizon: 7, 14, or 30 days |
| `include_chart` | bool | true | Include 90-day chart data in response |

**Response:**
```json
{
  "collection_address": "0xed5af3...",
  "collection_name": "Azuki",
  "direction": "UP",
  "confidence": 0.712,
  "prob_up": 0.712,
  "predicted_pct_change": 8.43,
  "current_floor_eth": 9.25,
  "predicted_floor_eth": 10.03,
  "horizon_days": 7,
  "data_source": "reservoir_api",
  "chart_data": [{"date": "2024-01-01", "floor": 9.1, "volume": 45.2, "sales": 12}],
  "generated_at": 1720000000.0
}
```

### `GET /api/collections`

List all pre-configured collection slugs and addresses.

### `GET /api/chart/{address}?days=90`

Historical floor + volume data for charting.

### `GET /api/health`

Health check — returns `{"status": "ok"}`.

### Interactive Docs

Available at `/api/docs` (Swagger UI) and `/api/redoc` on your deployed app.

---

## Run Inference Directly

```bash
# PyTorch (no ONNX needed)
python inference_example.py --collection azuki

# ONNX (after export)
python inference_example.py --collection azuki --mode onnx

# Batch all collections
python inference_example.py --batch

# On-chain verifiable (after OpenGradient upload)
python inference_example.py --collection azuki --mode onchain --model-cid QmXxx...
```

---

## Project Structure

```
nft-floor-forecaster/
├── backend/
│   ├── main.py                  # FastAPI app
│   ├── requirements.txt
│   ├── model/
│   │   ├── nft_forecaster.py    # PatchTST + CLIP + GatedFusion
│   │   └── weights.pt           # Trained weights (gitignored)
│   └── data/
│       └── pipeline.py          # Reservoir API + preprocessing
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main UI
│   │   └── index.css            # Design system
│   ├── package.json
│   └── vite.config.js
├── train.py                     # Training loop
├── export_onnx.py               # ONNX + INT8 export
├── upload_og.py                 # OpenGradient Hub upload
├── inference_example.py         # 3-mode inference demo
├── Dockerfile                   # Multi-stage build
├── railway.toml                 # Railway config
├── docker-compose.yml           # Local dev
└── .github/workflows/deploy.yml # CI/CD
```

---

## OpenGradient Verifiable Inference

After uploading to the hub, use on-chain TEE inference for trustless oracle use cases:

```python
import opengradient as og
import numpy as np

og.init(email="...", password="...", private_key="0x...")

tx_hash, result = og.infer(
    model_cid="YOUR_MODEL_CID",
    model_input={
        "ts_features": ts_window.flatten().tolist(),     # (90*11,) flat
        "clip_embedding": clip_emb.flatten().tolist(),   # (512,) flat
    },
    inference_mode=og.InferenceMode.TEE,
)

prob_up = 1 / (1 + np.exp(-result["direction_logit"][0]))
print(f"Verified on-chain: tx={tx_hash}, P(UP)={prob_up:.2%}")
```

**Use as an NFT lending protocol oracle:**
- Deploy as a scheduled OpenGradient workflow (every 24h)
- Push results to a Chainlink-compatible oracle contract
- NFTfi, Gondi, Arcade can consume floor price signals for collateral valuations

---

## Data Sources

| Source | What | How |
|---|---|---|
| [Reservoir API](https://reservoir.tools) | Floor price history, volume, sales | `fetch_floor_history()` |
| [Dune Analytics](https://dune.com) | On-chain nft.trades | SQL query (see architecture doc) |
| Synthetic generator | Demo/testing | `generate_synthetic_data()` — GBM + mean reversion |

---

## Limitations & Disclaimer

> ⚠️ **This model is for research and educational purposes only.**
> It does not constitute financial advice. NFT markets are highly illiquid and
> volatile. The model uses synthetic data in demo mode and real data
> requires a Reservoir API key + training run.

- Demo mode uses seeded synthetic data — forecasts are illustrative only
- Real accuracy depends on training data quality and market regime
- The vision tower uses deterministic pseudo-embeddings in demo mode (not real CLIP)
- Never use model output as sole basis for financial decisions

---

## License

MIT — see LICENSE

---

*Built for [OpenGradient](https://opengradient.ai) Hub · Multimodal AI · On-chain verifiable inference*
