# Symplectic Geometry – Poincaré Section Entropy for ETFs

Applies symplectic geometry to ETF returns by embedding in phase space (return, momentum). The Poincaré section at zero momentum captures the dynamics, and the entropy of the first return map quantifies chaos. Low entropy = predictable/trending; high entropy = chaotic.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Phase space: return, momentum (return difference over 5 days)
- Poincaré section at momentum = 0
- First return map → histogram → Shannon entropy
- Score = entropy (high = chaotic, low = predictable)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-symplectic-geometry-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High entropy → chaotic, unpredictable dynamics.
- Low entropy → regular, predictable, trending.
- Use low entropy ETFs for trend‑following strategies.

## Requirements

See `requirements.txt`.
