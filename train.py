"""
Training script for NFTFloorForecaster.
Usage:
    python train.py --collections 10 --epochs 40 --horizon 7
    python train.py --collections 100 --epochs 60 --horizon 30 --device cuda
"""
import argparse
import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path

# Add backend to path when running from project root
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from data.pipeline import (
    fetch_floor_history, generate_synthetic_data, engineer_features,
    generate_visual_embedding, KNOWN_COLLECTIONS, FEATURE_COLS
)
from model.nft_forecaster import NFTFloorForecaster
from sklearn.preprocessing import RobustScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LOOKBACK = 90


def build_windows(df: pd.DataFrame, horizon: int):
    df = engineer_features(df)
    values = df[FEATURE_COLS].values
    prices = df["floor_price_eth"].values

    if len(values) < LOOKBACK + horizon + 5:
        return None, None, None

    scaler = RobustScaler()
    values_scaled = scaler.fit_transform(values)

    X, y_dir, y_mag = [], [], []
    for i in range(LOOKBACK, len(values_scaled) - horizon):
        X.append(values_scaled[i - LOOKBACK:i])
        fut = prices[i + horizon]
        cur = prices[i]
        pct = (fut - cur) / (cur + 1e-8)
        pct = np.clip(pct, -1.0, 1.0)
        y_mag.append(pct)
        y_dir.append(1 if pct > 0.02 else 0)

    return np.array(X, dtype=np.float32), np.array(y_dir, dtype=np.int64), np.array(y_mag, dtype=np.float32)


def load_dataset(n_collections: int, horizon: int, data_dir: str = "data/raw/"):
    all_X, all_y_dir, all_y_mag, all_vis = [], [], [], []
    collections = list(KNOWN_COLLECTIONS.items())

    # Pad with synthetic collections if needed
    synthetic_addrs = [f"0xsynthetic{i:04d}000000000000000000000000000000000000" for i in range(max(0, n_collections - len(collections)))]

    for i, (slug, addr) in enumerate(collections[:n_collections]):
        df = fetch_floor_history(addr, days=LOOKBACK + horizon + 100)
        if df is None or len(df) < 60:
            logger.info(f"  Synthetic fallback for {slug}")
            df = generate_synthetic_data(addr, days=LOOKBACK + horizon + 100)

        X, y_dir, y_mag = build_windows(df, horizon)
        if X is None:
            continue

        vis = generate_visual_embedding(addr)
        vis_tiled = np.tile(vis, (len(X), 1))

        all_X.append(X)
        all_y_dir.append(y_dir)
        all_y_mag.append(y_mag)
        all_vis.append(vis_tiled)
        logger.info(f"  [{i+1}/{n_collections}] {slug}: {len(X)} windows")

    for addr in synthetic_addrs:
        df = generate_synthetic_data(addr, days=LOOKBACK + horizon + 100)
        X, y_dir, y_mag = build_windows(df, horizon)
        if X is None:
            continue
        vis = generate_visual_embedding(addr)
        vis_tiled = np.tile(vis, (len(X), 1))
        all_X.append(X)
        all_y_dir.append(y_dir)
        all_y_mag.append(y_mag)
        all_vis.append(vis_tiled)

    if not all_X:
        raise RuntimeError("No data loaded!")

    return (
        torch.FloatTensor(np.concatenate(all_X)),
        torch.LongTensor(np.concatenate(all_y_dir)),
        torch.FloatTensor(np.concatenate(all_y_mag)).unsqueeze(-1),
        torch.FloatTensor(np.concatenate(all_vis)),
    )


def train(args):
    device = args.device
    logger.info(f"Training on {device} | horizon={args.horizon}d | collections={args.collections}")

    logger.info("Loading dataset...")
    ts, y_dir, y_mag, vis = load_dataset(args.collections, args.horizon)
    logger.info(f"Total windows: {len(ts)} | UP rate: {y_dir.float().mean():.2%}")

    dataset = TensorDataset(ts, vis, y_dir, y_mag)
    n_val = max(1, int(0.15 * len(dataset)))
    n_test = max(1, int(0.10 * len(dataset)))
    n_train = len(dataset) - n_val - n_test
    train_ds, val_ds, _ = random_split(dataset, [n_train, n_val, n_test])

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    model = NFTFloorForecaster().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")

    bce = nn.BCEWithLogitsLoss()
    huber = nn.HuberLoss(delta=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_val_acc = 0.0
    best_path = args.output

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = 0

        for ts_b, vis_b, dir_b, mag_b in train_dl:
            ts_b, vis_b = ts_b.to(device), vis_b.to(device)
            dir_b, mag_b = dir_b.float().to(device), mag_b.to(device)

            optimizer.zero_grad()
            dir_logit, magnitude = model(ts_b, vis_b)
            loss = bce(dir_logit.squeeze(), dir_b) + 0.5 * huber(magnitude, mag_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for ts_b, vis_b, dir_b, mag_b in val_dl:
                ts_b, vis_b = ts_b.to(device), vis_b.to(device)
                dir_b, mag_b = dir_b.float().to(device), mag_b.to(device)
                dir_logit, magnitude = model(ts_b, vis_b)
                val_loss += bce(dir_logit.squeeze(), dir_b).item()
                preds = (torch.sigmoid(dir_logit.squeeze()) > 0.5).long()
                correct += (preds == dir_b.long()).sum().item()
                total += len(dir_b)

        acc = correct / total
        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={total_loss/len(train_dl):.4f} | "
            f"val_loss={val_loss/len(val_dl):.4f} | "
            f"val_acc={acc:.3f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.1f}s"
        )

        if acc > best_val_acc:
            best_val_acc = acc
            os.makedirs(os.path.dirname(best_path) if os.path.dirname(best_path) else ".", exist_ok=True)
            torch.save(model.state_dict(), best_path)
            logger.info(f"  ✓ Saved best model → {best_path} (val_acc={best_val_acc:.3f})")

    logger.info(f"\nTraining complete! Best val_acc={best_val_acc:.3f}")
    logger.info(f"Weights saved to: {best_path}")
    return best_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NFTFloorForecaster")
    parser.add_argument("--collections", type=int, default=10, help="Number of collections to train on")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=7, choices=[7, 14, 30])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="backend/model/weights.pt")
    args = parser.parse_args()
    train(args)
