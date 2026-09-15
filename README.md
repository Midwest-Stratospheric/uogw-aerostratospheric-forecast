---
license: cc-by-4.0
library_name: pytorch
tags:
  - weather
  - forecasting
  - stratosphere
  - upper-air
  - high-altitude-balloon
  - uogw
  - aerostratospheric
  - gfs
language:
  - en
pipeline_tag: time-series-forecasting
---

# UOGW Aerostratospheric Forecast (v0.1)

Site-specific **upper-air / lower-stratosphere profile forecast** trained for Aerostratospheric flight planning at **Casey, Illinois** (X2Griffon pad).

This is **not** a global foundation weather model (GraphCast / AIFS / Aurora). Those need multi-year global ERA5 and large GPU clusters. This package is the first public model that sits on top of the [Unified Open Global Weather (UOGW)](https://github.com/Midwest-Stratospheric/Unified-Open-Global-Weather) layer selection: **upper-air + stratospheric** fields at the MSDS launch site.

## What was selected from UOGW

From the UOGW six-layer commons (`ground`, `marine`, `upper_air`, `stratospheric`, `satellite`, `flight`):

| UOGW layer | Used here | Why |
|---|---|---|
| upper_air | Yes | Radiosonde / pressure-level analog for HAB climb |
| stratospheric | Yes | 70 / 50 hPa float band near X2Griffon target (~115 kft) |
| ground | Context only | Surface T, MSLP, 10 m wind as predictors |
| flight | Reserved | X2Griffon maiden science flight is 19 Sep 2026 — no recovered profiles yet |
| marine / satellite | Indexed, not trained | Available in the UOGW catalog for later versions |

Pressure levels (hPa): **250, 200, 150, 100, 70, 50**  
(~10–20 km, tropopause through lower stratosphere).

Training grid: NOAA **GFS seamless** via the Open-Meteo historical forecast API (the same family UOGW already samples in `space-gfs-snapshot.json`).

Site: Casey, IL `39.2992, -87.9925` — MSDS / NASA GLOBE GO-4VW9B.

## Task

Given the current profile plus 6 h and 24 h lags, predict the profile at **+6 h, +24 h, +48 h**:

- temperature (°C)
- wind speed (km/h, Open-Meteo native)
- wind components u, v (km/h, meteorological)
- geopotential height (m)

## Held-out test (2026-06-19 → 2026-09-11 inits)

| Horizon | T MAE | Wind-speed MAE | Z MAE |
|---|---:|---:|---:|
| +6 h | 1.02 °C | 12.1 km/h | 32 m |
| +24 h | 1.23 °C | 15.9 km/h | 36 m |
| +48 h | 1.50 °C | 19.4 km/h | 43 m |

11,403 train samples / 2,013 test samples. Chronological split (no shuffle).

## Architecture

3-layer MLP (111 → 96 GELU → 96 GELU → 90), dropout 0.1, Smooth-L1, AdamW. ~30k parameters. Weights live in `artifacts/model.json` (portable) and `artifacts/model.pt` (PyTorch).

## Run

```bash
python src/infer.py --live
```

Pulls the latest GFS seamless profile for Casey and prints +6/+24/+48 h forecasts.

Train again:

```bash
# after refreshing /tmp/om1.json … om3.json from Open-Meteo
python src/train.py
```

## Push to Hugging Face

This repo is GitHub-canonical. Hugging Face Hub is not connected to this workspace, so publish with a write token:

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create Midwest-Stratospheric/uogw-aerostratospheric-forecast --type model
huggingface-cli upload Midwest-Stratospheric/uogw-aerostratospheric-forecast . --repo-type model
```

Or from Python:

```python
from huggingface_hub import HfApi
HfApi().create_repo("Midwest-Stratospheric/uogw-aerostratospheric-forecast", repo_type="model", exist_ok=True)
HfApi().upload_folder(folder_path=".", repo_id="Midwest-Stratospheric/uogw-aerostratospheric-forecast", repo_type="model")
```

## Attribution

- Curator: Aerostratospheric / Midwest Stratospheric Data Systems
- UOGW: https://github.com/Midwest-Stratospheric/Unified-Open-Global-Weather
- Data paper: Higgs, J. D. (2026). *UOGW: An Open Multi-Layer Atmospheric Data Commons*. Zenodo. https://doi.org/10.5281/zenodo.21880000
- Grid: NOAA GFS via Open-Meteo (CC BY 4.0)
- License of this model package: CC BY 4.0

**Disclaimer.** Research and flight-planning aid only. Not an operational NWP product and not a substitute for FAA weather briefing.
