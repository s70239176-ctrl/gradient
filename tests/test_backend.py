"""
Test suite for NFT Floor Forecaster backend.
Run: pytest tests/ -v
"""
import sys
import os
import pytest
import numpy as np
import torch
import pandas as pd

# Make backend importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


# ── Model Tests ──────────────────────────────────────────────────────────

class TestNFTFloorForecaster:
    def setup_method(self):
        from model.nft_forecaster import NFTFloorForecaster
        self.model = NFTFloorForecaster()
        self.model.eval()

    def test_param_count(self):
        n = sum(p.numel() for p in self.model.parameters())
        assert 500_000 < n < 10_000_000, f"Unexpected param count: {n:,}"

    def test_forward_shapes(self):
        ts = torch.randn(1, 90, 11)
        vis = torch.randn(1, 512)
        dir_logit, magnitude = self.model(ts, vis)
        assert dir_logit.shape == (1, 1)
        assert magnitude.shape == (1, 1)

    def test_batch_forward(self):
        ts = torch.randn(4, 90, 11)
        vis = torch.randn(4, 512)
        dir_logit, magnitude = self.model(ts, vis)
        assert dir_logit.shape == (4, 1)
        assert magnitude.shape == (4, 1)

    def test_magnitude_bounded(self):
        """Tanh output must be in (-1, 1)."""
        ts = torch.randn(32, 90, 11)
        vis = torch.randn(32, 512)
        _, magnitude = self.model(ts, vis)
        assert magnitude.abs().max().item() < 1.0

    def test_predict_dict_keys(self):
        result = self.model.predict(
            np.random.randn(1, 90, 11).astype(np.float32),
            np.random.randn(1, 512).astype(np.float32)
        )
        assert "direction" in result
        assert "confidence" in result
        assert "prob_up" in result
        assert "predicted_pct_change" in result

    def test_predict_direction_valid(self):
        result = self.model.predict(
            np.random.randn(1, 90, 11).astype(np.float32),
            np.random.randn(1, 512).astype(np.float32)
        )
        assert result["direction"] in ("UP", "DOWN")

    def test_predict_confidence_range(self):
        result = self.model.predict(
            np.random.randn(1, 90, 11).astype(np.float32),
            np.random.randn(1, 512).astype(np.float32)
        )
        assert 0.5 <= result["confidence"] <= 1.0

    def test_predict_2d_input(self):
        """Model should handle 2D input (no batch dim) gracefully."""
        result = self.model.predict(
            np.random.randn(90, 11).astype(np.float32),
            np.random.randn(512).astype(np.float32)
        )
        assert result["direction"] in ("UP", "DOWN")

    def test_deterministic_with_same_seed(self):
        torch.manual_seed(42)
        ts = torch.randn(1, 90, 11)
        vis = torch.randn(1, 512)
        r1 = self.model.predict(ts.numpy(), vis.numpy())
        r2 = self.model.predict(ts.numpy(), vis.numpy())
        assert r1["predicted_pct_change"] == r2["predicted_pct_change"]


# ── Architecture Component Tests ─────────────────────────────────────────

class TestPatchTST:
    def test_patch_embedding_shape(self):
        from model.nft_forecaster import PatchEmbedding
        emb = PatchEmbedding(seq_len=90, patch_len=9, d_model=128, n_features=11)
        x = torch.randn(2, 90, 11)
        out = emb(x)
        # 90 // 9 = 10 patches
        assert out.shape == (2, 10, 128)

    def test_encoder_output(self):
        from model.nft_forecaster import SmallPatchTST
        enc = SmallPatchTST()
        x = torch.randn(3, 90, 11)
        out = enc(x)
        assert out.shape == (3, 128)


class TestGatedFusion:
    def test_fusion_output_shape(self):
        from model.nft_forecaster import GatedFusion
        fusion = GatedFusion(ts_dim=128, vis_dim=512, hidden=256)
        ts_emb = torch.randn(2, 128)
        vis_emb = torch.randn(2, 512)
        out = fusion(ts_emb, vis_emb)
        assert out.shape == (2, 128)  # hidden // 2

    def test_gate_bounds(self):
        """Gates must be in [0, 1] — sigmoid output."""
        from model.nft_forecaster import GatedFusion
        fusion = GatedFusion()
        ts_emb = torch.randn(8, 128)
        vis_emb = torch.randn(8, 512)
        with torch.no_grad():
            vis = fusion.vis_proj(vis_emb)
            combined = torch.cat([ts_emb, vis], dim=-1)
            gate = fusion.gate(combined)
        assert gate.min().item() >= 0.0
        assert gate.max().item() <= 1.0


# ── Data Pipeline Tests ──────────────────────────────────────────────────

class TestSyntheticData:
    def test_shape(self):
        from data.pipeline import generate_synthetic_data
        df = generate_synthetic_data("0xtest", days=120)
        assert len(df) == 120
        assert "floor_price_eth" in df.columns
        assert "volume_eth" in df.columns
        assert "sales_count" in df.columns

    def test_no_negatives(self):
        from data.pipeline import generate_synthetic_data
        df = generate_synthetic_data("0xtest", days=100)
        assert (df["floor_price_eth"] > 0).all()
        assert (df["volume_eth"] >= 0).all()

    def test_deterministic(self):
        from data.pipeline import generate_synthetic_data
        df1 = generate_synthetic_data("0xsameaddr", days=50)
        df2 = generate_synthetic_data("0xsameaddr", days=50)
        # Values are deterministic; index uses pd.Timestamp.now() so microseconds differ
        pd.testing.assert_frame_equal(
            df1.reset_index(drop=True),
            df2.reset_index(drop=True)
        )

    def test_different_collections_differ(self):
        from data.pipeline import generate_synthetic_data
        df1 = generate_synthetic_data("0xaddr1", days=50)
        df2 = generate_synthetic_data("0xaddr2", days=50)
        assert not df1["floor_price_eth"].equals(df2["floor_price_eth"])


class TestFeatureEngineering:
    def test_all_feature_cols_present(self):
        from data.pipeline import generate_synthetic_data, engineer_features, FEATURE_COLS
        df = generate_synthetic_data("0xtest", days=120)
        df = engineer_features(df)
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature: {col}"

    def test_no_nans_after_engineering(self):
        from data.pipeline import generate_synthetic_data, engineer_features, FEATURE_COLS
        df = generate_synthetic_data("0xtest", days=120)
        df = engineer_features(df)
        assert not df[FEATURE_COLS].isnull().any().any()

    def test_window_shape(self):
        from data.pipeline import get_timeseries_window
        window, df = get_timeseries_window("0xtest123", lookback=90)
        assert window.shape == (1, 90, 11)
        assert isinstance(df, pd.DataFrame)


class TestVisualEmbedding:
    def test_shape(self):
        from data.pipeline import generate_visual_embedding
        emb = generate_visual_embedding("0xtest")
        assert emb.shape == (1, 512)

    def test_normalized(self):
        from data.pipeline import generate_visual_embedding
        emb = generate_visual_embedding("0xtest")
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-5

    def test_deterministic(self):
        from data.pipeline import generate_visual_embedding
        e1 = generate_visual_embedding("0xsamecollection")
        e2 = generate_visual_embedding("0xsamecollection")
        np.testing.assert_array_equal(e1, e2)

    def test_different_collections_differ(self):
        from data.pipeline import generate_visual_embedding
        e1 = generate_visual_embedding("0xaddr1111")
        e2 = generate_visual_embedding("0xaddr2222")
        assert not np.allclose(e1, e2)


class TestCollectionResolver:
    def test_known_slug(self):
        from data.pipeline import get_collection_address
        addr = get_collection_address("boredapeyachtclub")
        assert addr.startswith("0x")
        assert len(addr) == 42

    def test_raw_address_passthrough(self):
        from data.pipeline import get_collection_address
        addr = "0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d"
        assert get_collection_address(addr) == addr

    def test_unknown_slug_returns_something(self):
        from data.pipeline import get_collection_address
        result = get_collection_address("some-unknown-collection-xyz")
        assert isinstance(result, str)
        assert len(result) > 0


# ── API Endpoint Tests ───────────────────────────────────────────────────

class TestAPI:
    def setup_method(self):
        import os, shutil
        # Ensure frontend dist exists for SPA route
        os.makedirs("/app/frontend/dist", exist_ok=True)
        with open("/app/frontend/dist/index.html", "w") as f:
            f.write("<html><body>NFT Forecaster</body></html>")
        from fastapi.testclient import TestClient
        from main import app
        self.client = TestClient(app)

    def test_health(self):
        r = self.client.get("/api/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data

    def test_collections(self):
        r = self.client.get("/api/collections")
        assert r.status_code == 200
        data = r.json()
        assert "collections" in data
        assert len(data["collections"]) > 0
        assert all("slug" in c and "address" in c for c in data["collections"])

    def test_forecast_basic(self):
        r = self.client.post("/api/forecast", json={
            "collection": "boredapeyachtclub",
            "horizon": 7,
            "include_chart": False
        })
        assert r.status_code == 200
        data = r.json()
        assert data["direction"] in ("UP", "DOWN")
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["predicted_pct_change"], float)
        assert data["current_floor_eth"] > 0

    def test_forecast_with_chart(self):
        r = self.client.post("/api/forecast", json={
            "collection": "azuki",
            "horizon": 7,
            "include_chart": True
        })
        assert r.status_code == 200
        data = r.json()
        assert data["chart_data"] is not None
        assert len(data["chart_data"]) > 0
        assert all("date" in pt and "floor" in pt for pt in data["chart_data"])

    def test_forecast_by_address(self):
        r = self.client.post("/api/forecast", json={
            "collection": "0xed5af388653567af2f388e6224dc7c4b3241c544",
            "horizon": 14,
            "include_chart": False
        })
        assert r.status_code == 200

    def test_forecast_30d_horizon(self):
        r = self.client.post("/api/forecast", json={
            "collection": "cryptopunks",
            "horizon": 30,
            "include_chart": False
        })
        assert r.status_code == 200
        data = r.json()
        assert data["horizon_days"] == 30

    def test_forecast_invalid_horizon(self):
        r = self.client.post("/api/forecast", json={
            "collection": "azuki",
            "horizon": 99
        })
        assert r.status_code == 400

    def test_chart_endpoint(self):
        r = self.client.get("/api/chart/boredapeyachtclub?days=30")
        assert r.status_code == 200
        data = r.json()
        assert "data" in data
        assert len(data["data"]) > 0

    def test_predicted_floor_computed_correctly(self):
        r = self.client.post("/api/forecast", json={
            "collection": "pudgypenguins",
            "horizon": 7,
            "include_chart": False
        })
        data = r.json()
        expected = data["current_floor_eth"] * (1 + data["predicted_pct_change"] / 100)
        assert abs(data["predicted_floor_eth"] - expected) < 0.01

    def test_response_has_generated_at(self):
        r = self.client.post("/api/forecast", json={"collection": "moonbirds", "horizon": 7})
        assert r.status_code == 200
        assert "generated_at" in r.json()

    def test_data_source_field(self):
        r = self.client.post("/api/forecast", json={"collection": "azuki", "horizon": 7})
        data = r.json()
        assert data["data_source"] in ("reservoir_api", "synthetic_demo")


# ── Integration Test ─────────────────────────────────────────────────────

class TestEndToEnd:
    def test_full_pipeline_all_collections(self):
        """Run forecast for every known collection — none should crash."""
        import os, shutil
        os.makedirs("/app/frontend/dist", exist_ok=True)
        with open("/app/frontend/dist/index.html", "w") as f:
            f.write("<html><body>ok</body></html>")
        from fastapi.testclient import TestClient
        from main import app
        from data.pipeline import KNOWN_COLLECTIONS
        client = TestClient(app)

        errors = []
        for slug in KNOWN_COLLECTIONS:
            r = client.post("/api/forecast", json={"collection": slug, "horizon": 7, "include_chart": False})
            if r.status_code != 200:
                errors.append(f"{slug}: HTTP {r.status_code}")

        assert not errors, f"Failed collections: {errors}"
