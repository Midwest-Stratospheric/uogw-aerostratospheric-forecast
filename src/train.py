#!/usr/bin/env python3
"""Train a compact aerostratospheric profile forecast model for Casey, IL."""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)
LEVELS = [250, 200, 150, 100, 70, 50]
LAGS_H = [0, 6, 24]
HORIZONS_H = [6, 24, 48]
LOOKBACK_OK = 24
SPLIT_FRAC = 0.85
SEED = 42

def merge_open_meteo(paths):
    merged = None
    for p in paths:
        d = json.loads(Path(p).read_text())
        h = d["hourly"]
        if merged is None:
            merged = {k: list(v) for k, v in h.items()}
            meta = {k: d[k] for k in ("latitude", "longitude", "elevation", "hourly_units") if k in d}
        else:
            for k, v in h.items():
                merged[k].extend(v)
    return merged, meta

def wind_uv(speed, direction_deg):
    rad = np.deg2rad(np.asarray(direction_deg, dtype=np.float64))
    spd = np.asarray(speed, dtype=np.float64)
    return -spd * np.sin(rad), -spd * np.cos(rad)

def build_arrays(hourly):
    n = len(hourly["time"]); times = np.array(hourly["time"]); cols = {}
    for lev in LEVELS:
        t = np.array(hourly[f"temperature_{lev}hPa"], dtype=np.float64)
        ws = np.array(hourly[f"wind_speed_{lev}hPa"], dtype=np.float64)
        wd = np.array(hourly[f"wind_direction_{lev}hPa"], dtype=np.float64)
        z = np.array(hourly[f"geopotential_height_{lev}hPa"], dtype=np.float64)
        u, v = wind_uv(ws, wd)
        cols[f"t_{lev}"] = t; cols[f"ws_{lev}"] = ws; cols[f"u_{lev}"] = u; cols[f"v_{lev}"] = v; cols[f"z_{lev}"] = z
    cols["t2m"] = np.array(hourly["temperature_2m"], dtype=np.float64)
    cols["pmsl"] = np.array(hourly["pressure_msl"], dtype=np.float64)
    cols["ws10"] = np.array(hourly["wind_speed_10m"], dtype=np.float64)
    hour = np.array([int(ts[11:13]) for ts in times], dtype=np.float64)
    doy = np.array([(np.datetime64(ts[:10]) - np.datetime64(ts[:4] + "-01-01")).astype(int) + 1 for ts in times], dtype=np.float64)
    cols["sin_hour"] = np.sin(2 * math.pi * hour / 24)
    cols["cos_hour"] = np.cos(2 * math.pi * hour / 24)
    cols["sin_doy"] = np.sin(2 * math.pi * doy / 365.25)
    cols["cos_doy"] = np.cos(2 * math.pi * doy / 365.25)
    profile_keys = [f"{p}_{lev}" for lev in LEVELS for p in ("t", "ws", "u", "v", "z")]
    feature_now_keys = profile_keys + ["t2m", "pmsl", "ws10", "sin_hour", "cos_hour", "sin_doy", "cos_doy"]
    stacked = np.column_stack([cols[k] for k in feature_now_keys])
    target_stack = np.column_stack([cols[k] for k in profile_keys])
    valid = np.isfinite(stacked).all(axis=1) & np.isfinite(target_stack).all(axis=1)
    X_list, y_list, t_list = [], [], []
    max_h = max(HORIZONS_H)
    for i in range(LOOKBACK_OK, n - max_h):
        if not valid[i] or not all(valid[i - lag] for lag in LAGS_H if lag > 0):
            continue
        if not all(valid[i + h] for h in HORIZONS_H):
            continue
        x = np.concatenate([stacked[i - lag] for lag in LAGS_H])
        y = np.concatenate([target_stack[i + h] for h in HORIZONS_H])
        X_list.append(x); y_list.append(y); t_list.append(times[i])
    return np.asarray(X_list, np.float32), np.asarray(y_list, np.float32), t_list, feature_now_keys, profile_keys

class ProfileMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
    def forward(self, x):
        return self.net(x)

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    hourly, meta = merge_open_meteo(["/tmp/om1.json", "/tmp/om2.json", "/tmp/om3.json"])
    X, y, times, feat_keys, tgt_keys = build_arrays(hourly)
    cut = int(len(X) * SPLIT_FRAC)
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
    x_mean, x_std = Xtr.mean(0), Xtr.std(0).clip(min=1e-6)
    y_mean, y_std = ytr.mean(0), ytr.std(0).clip(min=1e-6)
    model = ProfileMLP(X.shape[1], y.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    loader = DataLoader(TensorDataset(torch.from_numpy((Xtr-x_mean)/x_std), torch.from_numpy((ytr-y_mean)/y_std)), batch_size=256, shuffle=True)
    best, best_val, wait = None, 1e9, 0
    for epoch in range(1, 81):
        model.train(); running = 0.0
        for xb, yb in loader:
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step(); running += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(torch.from_numpy((Xte-x_mean)/x_std)), torch.from_numpy((yte-y_mean)/y_std)).item()
        print(f"epoch {epoch:02d} train={running/len(Xtr):.4f} val={val:.4f}")
        if val < best_val - 1e-4:
            best_val = val; best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= 12:
                print("early stop"); break
    model.load_state_dict(best)
    print("best_val", best_val)
    torch.save(model.state_dict(), ART_DIR / "model.pt")

if __name__ == "__main__":
    main()
