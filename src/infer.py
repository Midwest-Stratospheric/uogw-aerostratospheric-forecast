#!/usr/bin/env python3
"""NumPy inference for the UOGW aerostratospheric profile forecast model."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "artifacts" / "model.json"

WEIGHT_FILES = {
    "net.0.weight": "w_net_0_weight.json",
    "net.0.bias": "w_net_0_bias.json",
    "net.3.weight": "w_net_3_weight.json",
    "net.3.bias": "w_net_3_bias.json",
    "net.5.weight": "w_net_5_weight.json",
    "net.5.bias": "w_net_5_bias.json",
}


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def load_bundle(path: Path = DEFAULT_MODEL) -> dict:
    path = Path(path)
    if path.exists():
        bundle = json.loads(path.read_text())
        if "weights" in bundle:
            return bundle
    meta_path = path.parent / "model_meta.json"
    bundle = json.loads(meta_path.read_text())
    weights = {}
    for key, fn in WEIGHT_FILES.items():
        weights[key] = json.loads((path.parent / fn).read_text())
    bundle["weights"] = weights
    return bundle


def forward(bundle: dict, x_raw: np.ndarray) -> np.ndarray:
    x = (np.asarray(x_raw, dtype=np.float32) - np.asarray(bundle["x_mean"])) / np.asarray(bundle["x_std"])
    w = bundle["weights"]
    h = x @ np.asarray(w["net.0.weight"]).T + np.asarray(w["net.0.bias"])
    h = gelu(h)
    h = h @ np.asarray(w["net.3.weight"]).T + np.asarray(w["net.3.bias"])
    h = gelu(h)
    y = h @ np.asarray(w["net.5.weight"]).T + np.asarray(w["net.5.bias"])
    return y * np.asarray(bundle["y_std"]) + np.asarray(bundle["y_mean"])


def pack_current_features(obs: dict, bundle: dict) -> np.ndarray:
    keys = bundle["feature_now_keys"]
    parts = []
    for lag in bundle["lags_h"]:
        block = obs[f"lag_{lag}"]
        parts.append(np.array([block[k] for k in keys], dtype=np.float32))
    return np.concatenate(parts)


def decode_prediction(bundle: dict, y: np.ndarray) -> dict:
    tgt = bundle["target_keys"]
    n = len(tgt)
    out = {}
    for i, h in enumerate(bundle["horizons_h"]):
        sl = y[i * n : (i + 1) * n]
        out[f"+{h}h"] = {k: float(sl[j]) for j, k in enumerate(tgt)}
    return out


def demo_from_open_meteo(bundle: dict) -> dict:
    import json as _json
    import urllib.request

    levels = bundle["source"]["levels_hpa"]
    vars_ = []
    for lev in levels:
        vars_ += [
            f"temperature_{lev}hPa",
            f"wind_speed_{lev}hPa",
            f"wind_direction_{lev}hPa",
            f"geopotential_height_{lev}hPa",
        ]
    vars_ += ["temperature_2m", "pressure_msl", "wind_speed_10m"]
    site = bundle["site"]
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={site['lat']}&longitude={site['lon']}"
        f"&hourly={','.join(vars_)}&past_days=2&forecast_days=1"
        "&timezone=UTC&models=gfs_seamless"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = _json.loads(resp.read().decode())
    h = data["hourly"]
    times = h["time"]

    def row_at(idx: int) -> dict:
        ts = times[idx]
        hour = int(ts[11:13])
        doy = (np.datetime64(ts[:10]) - np.datetime64(ts[:4] + "-01-01")).astype(int) + 1
        block = {
            "t2m": h["temperature_2m"][idx],
            "pmsl": h["pressure_msl"][idx],
            "ws10": h["wind_speed_10m"][idx],
            "sin_hour": math.sin(2 * math.pi * hour / 24),
            "cos_hour": math.cos(2 * math.pi * hour / 24),
            "sin_doy": math.sin(2 * math.pi * doy / 365.25),
            "cos_doy": math.cos(2 * math.pi * doy / 365.25),
        }
        for lev in levels:
            t = h[f"temperature_{lev}hPa"][idx]
            ws = h[f"wind_speed_{lev}hPa"][idx]
            wd = h[f"wind_direction_{lev}hPa"][idx]
            z = h[f"geopotential_height_{lev}hPa"][idx]
            rad = math.radians(wd)
            block[f"t_{lev}"] = t
            block[f"ws_{lev}"] = ws
            block[f"u_{lev}"] = -ws * math.sin(rad)
            block[f"v_{lev}"] = -ws * math.cos(rad)
            block[f"z_{lev}"] = z
        return block

    i0 = len(times) - 1
    while i0 >= 24:
        ok = all(h[f"temperature_{lev}hPa"][i0] is not None for lev in levels)
        if ok:
            break
        i0 -= 1
    obs = {
        "init_time": times[i0],
        "lag_0": row_at(i0),
        "lag_6": row_at(i0 - 6),
        "lag_24": row_at(i0 - 24),
    }
    x = pack_current_features(obs, bundle)
    y = forward(bundle, x)
    pred = decode_prediction(bundle, y)
    return {"init_time_utc": obs["init_time"], "forecast": pred, "site": bundle["site"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    bundle = load_bundle(Path(args.model))
    if args.live:
        print(json.dumps(demo_from_open_meteo(bundle), indent=2))
    else:
        print(json.dumps({"name": bundle["name"], "metrics": bundle["metrics"], "site": bundle["site"]}, indent=2))


if __name__ == "__main__":
    main()
