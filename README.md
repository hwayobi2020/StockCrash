# StockCrash

ML/DL models for stock crash prediction using implied volatility index and wavelet-transformed features.

## Overview

This repository contains the implementation for the paper:
> **"Stock Crash Risk Prediction with Implied Volatility Index: A Comparison of Tree-Based and Transformer-Based Models"**
> (Submitted to IEEE Access)

## Dataset

| Item | Description |
|------|-------------|
| Source | KOSPI/KOSDAQ listed companies (FnGuide, KRX) |
| Period | 2015Q1 ~ 2024Q2 (10 years) |
| Observations | ~112,000 stock-quarter pairs |
| Target | 60-day MDD ≥ 40% (binary classification) |
| Crash Rate | 7.01% |

## Features

| Category | Count | Description |
|----------|-------|-------------|
| Base (Financial + Market) | 19 | ROE, DR, PER, VKOSPI, PRICE_TO_52W_HIGH/LOW, etc. |
| Wavelet | 16 | Daubechies-4 (db4), Level 3 decomposition (cA3, cD3, cD2, cD1 × 4 statistics) |
| **Total** | **35** | |

## Models

- **Tree-based Ensemble**: CatBoost, XGBoost, LightGBM, Random Forest
- **Deep Learning**: CNN, LSTM, GRU, Transformer
- **Baseline**: Logistic Regression

## Validation

- **Method**: 10-fold Walk-forward Cross-Validation
- **Test Periods**: 2019Q4 ~ 2024Q2
- **Buffer**: 3-month gap between validation and test sets to prevent data leakage
- **Hyperparameter Tuning**: Optuna TPE sampler (50 trials per fold)

## Results (Best Model: CatBoost + Wavelet)

| Metric | Mean | Std |
|--------|------|-----|
| AUC | 0.841 | 0.053 |
| KS | 0.559 | 0.078 |
| PR-AUC | 0.320 | 0.144 |
| Lift@10% | 4.74 | 1.32 |
| Recall@FPR10% | 0.540 | 0.142 |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python combine.py
```

## File Structure

```
├── combine.py               # Main experiment code
├── requirements.txt         # Dependencies
└── stock_fin_sample.parquet # Sample training data (1/10 of full dataset)
```

## Citation

If you use this code, please cite:

```bibtex
@article{chung2025stockcrash,
  title={Stock Crash Risk Prediction with Implied Volatility Index: A Comparison of Tree-Based and Transformer-Based Models},
  author={Chung, Heeseung},
  journal={IEEE Access},
  year={2025}
}
```

## License

MIT License
