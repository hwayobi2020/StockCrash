====================================================
Stock Crash Risk Prediction
====================================================

Author: Heeseung Chung

----------------------------------------------------
1. Description
----------------------------------------------------
This repository provides the source code for the paper submitted to IEEE Access:

"Stock Crash Risk Prediction with Implied Volatility Index Using
Wavelet-Enhanced Gradient Boosting and Transformer-Based Deep Learning Models."

The code implements:
 - Wavelet feature extraction (Daubechies-4, level 3)
 - Gradient boosting models (CatBoost, XGBoost, LightGBM)
 - Deep learning models (CNN, LSTM, GRU, Transformer)
 - Walk-forward cross-validation (10 folds)
 - Evaluation metrics (AUC, KS, Brier Score)

----------------------------------------------------
2. File
----------------------------------------------------
models.py         : Main script for data preprocessing, feature generation, and model training

----------------------------------------------------
3. Environment
----------------------------------------------------
Python >= 3.10
Required packages (see requirements.txt):
  pandas, numpy, scikit-learn, scipy, matplotlib, seaborn,
  pywavelets, tensorflow, catboost, xgboost, lightgbm

----------------------------------------------------
4. How to Run
----------------------------------------------------
Execute:
  python models.py

This will:
  1. Load stock and financial datasets (Parquet format)
  2. Generate 60-day time-series sequences
  3. Apply wavelet decomposition ('db4', level = 3)
  4. Train CatBoost, XGBoost, LightGBM, CNN, LSTM, GRU, Transformer
  5. Output AUC, KS, LogLoss, and Brier Score results

----------------------------------------------------
5. Data
----------------------------------------------------
Data files (sample data):
 - stock_all_vkospi_sample.parquet
 - stock_fin_result_with_vkospi_all_sample.parquet

----------------------------------------------------
6. Repository
----------------------------------------------------
[Online]. Available: https://github.com/hwayobi2020/StockCrash
Accessed: Oct 11, 2025
