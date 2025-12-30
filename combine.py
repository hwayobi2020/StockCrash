def run_fold_experiment(fold_index, feature_set='base', verbose=True, use_tuned_params=False):
  """
  Run experiment for a specific fold with selected feature set.

  Args:
      fold_index: Fold number (0-9)
      feature_set: Feature group to use
          - 'base': fin_features only (시장+재무 혼합)
          - 'base_control': fin_features + control_features (ret_60d, rvol_60d, mdd_60d)
          - 'base_wavelet': fin_features + wavelet_features (16개)
      verbose: Print detailed output
      use_tuned_params: If True, load tuned hyperparameters from best_params_{feature_set}.json
  """
  import pandas as pd
  import numpy as np
  # Tensorflow / Keras Imports - Modify for Transformer (Keep TF for seed)
  import tensorflow as tf
  # from tensorflow.keras.models import Model # No longer needed for AE
  # >>> MODIFICATION START: Remove Transformer/AE specific layers <<<
  # from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, TimeDistributed, Reshape, Embedding
  # >>> MODIFICATION END <<<
  # from tensorflow.keras.optimizers import Adam # No longer needed for AE
  # from tensorflow.keras.callbacks import EarlyStopping # No longer needed for AE
  # >>> MODIFICATION START: Import MSE <<<
  # 라인 20 수정: log_loss, brier_score_loss 추가
  from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error, log_loss, brier_score_loss, average_precision_score, precision_recall_curve, recall_score # MSE, LogLoss, Brier, PR-AUC, Recall 추가
  # >>> MODIFICATION END <<<
  from sklearn.preprocessing import StandardScaler, LabelEncoder
  import matplotlib.pyplot as plt
  from scipy import stats # For KS statistic calculation

  # CatBoost
  # !pip install catboost imblearn pywavelets # imblearn은 이제 필요 없음, pywavelets 설치 필요
  from catboost import CatBoostClassifier
  # from imblearn.over_sampling import RandomOverSampler # 삭제

  # >>> MODIFICATION START: Import other models <<<
  from sklearn.linear_model import LogisticRegression
  from sklearn.ensemble import RandomForestClassifier
  import xgboost as xgb
  import lightgbm as lgb
  # >>> MODIFICATION END <<<


  # --- KS Statistic Calculation Function ---
  def calculate_ks(y_true, y_prob):
      """
      Calculates the Kolmogorov-Smirnov statistic for binary classification predictions.

      Args:
          y_true: Array-like of true binary labels (0 or 1).
          y_prob: Array-like of predicted probabilities for the positive class.

      Returns:
          The KS statistic (float). Returns np.nan if calculation is not possible.
      """
      try:
          # Combine actual values (0/1) and predicted probabilities
          data = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})

          # Separate probabilities for actual positive (1) and negative (0) cases
          prob_positive = data[data['y_true'] == 1]['y_prob']
          prob_negative = data[data['y_true'] == 0]['y_prob']

          # Check if either class is empty
          if len(prob_positive) == 0 or len(prob_negative) == 0:
              # print("Warning: One class is empty, KS statistic cannot be calculated reliably.")
              return np.nan # Or 0, depending on desired behavior

          # Use scipy's ks_2samp which directly compares the distributions
          # It returns KS statistic and p-value. We only need the statistic.
          ks_stat, p_value = stats.ks_2samp(prob_positive, prob_negative)

          return ks_stat
      except Exception as e:
          print(f"Error calculating KS statistic: {e}")
          return np.nan
  # --- End KS Function ---

  # --- Top Decile Lift Calculation Function ---
  def calculate_top_decile_lift(y_true, y_prob):
      """
      Calculates the lift in the top decile (top 10% of predicted probabilities).
      Lift = (% of positives in top decile) / (% of positives in overall population)
      """
      try:
          n_samples = len(y_true)
          top_decile_size = max(1, int(n_samples * 0.1))

          # Sort by predicted probability (descending)
          sorted_indices = np.argsort(y_prob)[::-1]
          top_decile_indices = sorted_indices[:top_decile_size]

          # Calculate positive rate in top decile
          top_decile_positive_rate = np.mean(y_true[top_decile_indices])

          # Calculate overall positive rate
          overall_positive_rate = np.mean(y_true)

          if overall_positive_rate == 0:
              return np.nan

          lift = top_decile_positive_rate / overall_positive_rate
          return lift
      except Exception as e:
          print(f"Error calculating top decile lift: {e}")
          return np.nan
  # --- End Lift Function ---

  # --- Recall at Fixed FPR Function ---
  def calculate_recall_at_fpr(y_true, y_prob, target_fpr=0.1):
      """
      Calculates recall (TPR) at a fixed false positive rate.
      """
      try:
          fpr, tpr, thresholds = roc_curve(y_true, y_prob)
          # Find TPR at target FPR
          idx = np.searchsorted(fpr, target_fpr)
          if idx >= len(tpr):
              return tpr[-1]
          return tpr[idx]
      except Exception as e:
          print(f"Error calculating recall at FPR: {e}")
          return np.nan
  # --- End Recall at FPR Function ---

  # --------------------------------------------


  ############################################################
  # 1) 파일 로드 - config.py에서 경로 설정 가져오기
  ############################################################
  # 환경별 경로 설정 (로컬 PC / Google Colab 자동 감지)
  from config import DATA_FILE, IS_COLAB

  try:
      fin_df = pd.read_parquet(DATA_FILE)
      env_name = "Google Colab" if IS_COLAB else "Local PC"
      if verbose: print(f"Successfully loaded data from {env_name}: {DATA_FILE}")
  except FileNotFoundError:
      print(f"Error: Data file not found: {DATA_FILE}")
      if IS_COLAB:
          print("Colab에서 실행 중입니다. Google Drive가 마운트되었는지 확인하세요.")
          print("  1. from google.colab import drive")
          print("  2. drive.mount('/content/drive')")
          print(f"  3. 데이터 파일 경로: {DATA_FILE}")
      exit()
  except Exception as e:
      print(f"An error occurred during file loading: {e}")
      exit()


  ############################################################
  # 2) 날짜 전처리
  ############################################################
  fin_df['BSOP_DATE'] = pd.to_datetime(fin_df['BSOP_DATE'], errors='coerce')

  # 2024-07-01 이전 데이터
  fin_df = fin_df[fin_df['BSOP_DATE'] < '2024-07-01'].copy()

  ############################################################
  # 2-1) 장기 거래정지 종목 제외
  # 현재 분기 rvol_60d=0 AND 다음 분기 rvol_60d=0 → 최소 2분기 연속 거래정지
  ############################################################
  fin_df = fin_df.sort_values(['SHRN_ISCD', 'BSOP_DATE'])
  fin_df['_next_rvol_60d'] = fin_df.groupby('SHRN_ISCD')['rvol_60d'].shift(-1)

  # 연속 거래정지 케이스 제외 (현재=0 AND 다음=0)
  consecutive_halt_mask = (fin_df['rvol_60d'] == 0) & (fin_df['_next_rvol_60d'] == 0)
  n_excluded = consecutive_halt_mask.sum()
  fin_df = fin_df[~consecutive_halt_mask].copy()

  # 임시 컬럼 제거
  fin_df = fin_df.drop(columns=['_next_rvol_60d'])

  if verbose:
      print(f"Excluded {n_excluded} observations with consecutive trading halts (2+ quarters)")

  ############################################################
  # 3) ROE 결측치 제거 및 피처 생성
  ############################################################
  mask_fin = fin_df.groupby('SHRN_ISCD')['ROE'].transform('count') > 0
  fin_df = fin_df[mask_fin].copy()
  if verbose: print("Fin DF Columns:", fin_df.columns.tolist())
  fin_df = fin_df.dropna(subset=['windowdrop_60d_40']).copy() # Ensure target is not NaN (MDD-based)
  fin_df['MONTH'] = fin_df['BSOP_DATE'].dt.month
  # Calculate features safely, avoiding division by zero or near-zero
  fin_df.loc[:, 'PRICE_TO_52W_HIGH'] = np.where(fin_df['W52_HGPR'] != 0, fin_df['STCK_PRPR'] / fin_df['W52_HGPR'], 0)
  fin_df.loc[:, 'PRICE_TO_52W_LOW'] = np.where(fin_df['W52_LWPR'] != 0, fin_df['STCK_PRPR'] / fin_df['W52_LWPR'], 0)
  fin_df.loc[:, 'PER'] = np.where(fin_df['MARKET_CAP'] != 0, fin_df['NET_INCOME'] / fin_df['MARKET_CAP'], 0)
  fin_df.loc[:, 'PSR'] = np.where(fin_df['STCK_PRPR'] != 0, fin_df['PS'] / fin_df['STCK_PRPR'], 0)
  fin_df.loc[:, 'FTRUTH'] = np.where(fin_df['NET_INCOME'] != 0, fin_df['PS'] / fin_df['NET_INCOME'], 0)
  fin_df.loc[:, 'NET_DEBT_RATIO'] = np.where(fin_df['MARKET_CAP'] != 0, fin_df['NET_DEBT'] / fin_df['MARKET_CAP'], 0)
  fin_df.loc[:, 'DEBT_COST_RATIO'] = np.where(fin_df['MARKET_CAP'] != 0, fin_df['DEBT_COST'] / fin_df['MARKET_CAP'], 0)

  le_scrt = LabelEncoder()
  le_mrkt = LabelEncoder()
  fin_df['SCRT_GRP_CLS_CODE'] = fin_df['SCRT_GRP_CLS_CODE'].astype(str)
  fin_df['MRKT_DIV_CLS_CODE'] = fin_df['MRKT_DIV_CLS_CODE'].astype(str)
  fin_df['SCRT_GRP_CLS_CODE'] = le_scrt.fit_transform(fin_df['SCRT_GRP_CLS_CODE'])
  fin_df['MRKT_DIV_CLS_CODE'] = le_mrkt.fit_transform(fin_df['MRKT_DIV_CLS_CODE'])

  ############################################################
  # 4) 주요 설정값
  ############################################################
  # Wavelet 피처 (전처리 파일에서 이미 계산됨)
  wavelet_features = [
      'Wav_STCK_PRPR_cA3_energy', 'Wav_STCK_PRPR_cA3_std', 'Wav_STCK_PRPR_cA3_skew', 'Wav_STCK_PRPR_cA3_kurt',
      'Wav_STCK_PRPR_cD3_energy', 'Wav_STCK_PRPR_cD3_std', 'Wav_STCK_PRPR_cD3_skew', 'Wav_STCK_PRPR_cD3_kurt',
      'Wav_STCK_PRPR_cD2_energy', 'Wav_STCK_PRPR_cD2_std', 'Wav_STCK_PRPR_cD2_skew', 'Wav_STCK_PRPR_cD2_kurt',
      'Wav_STCK_PRPR_cD1_energy', 'Wav_STCK_PRPR_cD1_std', 'Wav_STCK_PRPR_cD1_skew', 'Wav_STCK_PRPR_cD1_kurt',
  ]
  # 제어 피처 (전처리 파일에서 이미 계산됨)
  control_features = ['ret_60d', 'rvol_60d', 'mdd_60d']

  # 재무 피처 (VKOSPI 포함)
  fin_features = [
      'PRICE_TO_52W_HIGH', 'PRICE_TO_52W_LOW', 'SCRT_GRP_CLS_CODE',
      'MONTH', 'VKOSPI',
      'FN_ACML_TR_PBMN', 'ORGN_NTBY_QTY', 'DR', 'PSR', 'ROE', 'ROE_INC',
      'NET_DEBT_RATIO', 'DEBT_COST_RATIO', 'DEBT_DEPENDENCY', 'PER', 'ICR', 'FTRUTH', 'ASSET_GROWTH', 'TOTAL_ASSET_GROWTH',
  ]
  # 재무 피처 (VKOSPI 제외)
  fin_features_no_vkospi = [f for f in fin_features if f != 'VKOSPI']

  # Feature set 선택에 따라 사용할 피처 결정
  if feature_set == 'base':
      all_features = fin_features
      feature_set_name = "Base (fin_features only)"
  elif feature_set == 'base_no_vkospi':
      all_features = fin_features_no_vkospi
      feature_set_name = "Base without VKOSPI"
  elif feature_set == 'base_control':
      all_features = fin_features + control_features
      feature_set_name = "Base + Control (fin + ret_60d, rvol_60d, mdd_60d)"
  elif feature_set == 'base_control_no_vkospi':
      all_features = fin_features_no_vkospi + control_features
      feature_set_name = "Base + Control without VKOSPI"
  elif feature_set == 'base_wavelet':
      all_features = fin_features + wavelet_features
      feature_set_name = "Base + Wavelet (fin + 16 wavelet features)"
  elif feature_set == 'base_wavelet_no_vkospi':
      all_features = fin_features_no_vkospi + wavelet_features
      feature_set_name = "Base + Wavelet without VKOSPI"
  else:
      raise ValueError(f"Unknown feature_set: {feature_set}. Use 'base', 'base_no_vkospi', 'base_control', 'base_control_no_vkospi', 'base_wavelet', or 'base_wavelet_no_vkospi'")

  all_features = [f for f in all_features if f in fin_df.columns]  # Ensure features exist
  if verbose:
      print(f"\n=== Feature Set: {feature_set_name} ===")
      print(f"Number of features: {len(all_features)}")
      print("Using Features:", all_features)

  ############################################################
  # 5) 피처 데이터 준비 (전처리된 파일에서 직접 추출)
  ############################################################
  # Handle potential division by zero or inf/NaN before processing
  numeric_cols_fin = fin_df.select_dtypes(include=np.number).columns
  fin_df[numeric_cols_fin] = fin_df[numeric_cols_fin].replace([np.inf, -np.inf], np.nan)

  # Fill NaNs
  fin_df.fillna(0, inplace=True)

  if verbose: print("Preparing feature data from preprocessed file...")

  # 피처 데이터 추출
  feature_data = fin_df[all_features].values
  matched_targets = fin_df['windowdrop_60d_40'].values  # MDD-based target: 60-day MDD >= 40%
  dates = fin_df['BSOP_DATE'].values

  if verbose:
    print("feature_data.shape   :", feature_data.shape)
    print("matched_targets.shape:", matched_targets.shape)
    print("dates.shape          :", dates.shape)

  ############################################################
  # 6) Train/Test 분할 (변경 없음)
  ############################################################
  # (2015, 2019, 2020, [2021]),  # Train: 2016~2018, Val: 2019, Test: 2021
  # (2016, 2020, 2021, [2022]),  # Train: 2016~2018, Val: 2019, Test: 2021
  # (2017, 2021, 2022, [2023]),  # Train: 2016~2018, Val: 2019, Test: 2021
  # (2018, 2022, 2023, [2024]),  # Train: 2016~2018, Val: 2019, Test: 2021
  # train_mask = (dates <= pd.Timestamp('2022-12-31'))
  # val_mask = (dates >= pd.Timestamp('2023-01-01')) & (dates <= pd.Timestamp('2023-12-31'))
  # test_mask  = dates >= pd.Timestamp(pd.Timestamp('2024-01-01'))

  #feature importance
  # train_mask = (dates >= pd.Timestamp('2015-01-01')) & (dates <= pd.Timestamp('2021-12-31'))
  # val_mask = (dates >= pd.Timestamp('2022-01-01')) & (dates <= pd.Timestamp('2022-12-31'))
  # test_mask = (dates >= pd.Timestamp('2023-01-01')) & (dates <= pd.Timestamp('2024-06-30'))

  # --- 10개 기간(Fold)을 하드코딩하여 리스트에 저장 (Train: 4년, Val: 6개월, Buffer: 3개월, Test: 3개월, Fold간격: 6개월) ---
  folds = [
        # Fold 0: Train 15.1~18.12, Val 19.1~19.6, Buffer 19.7~19.9, Test 19.10~19.12
        {'train_start': pd.to_datetime('2015-01-01'), 'train_end': pd.to_datetime('2018-12-31'),
        'val_start'  : pd.to_datetime('2019-01-01'), 'val_end'  : pd.to_datetime('2019-06-30'),
        'test_start' : pd.to_datetime('2019-10-01'), 'test_end' : pd.to_datetime('2019-12-31')},
        # Fold 1: Train 15.7~19.6, Val 19.7~19.12, Buffer 20.1~20.3, Test 20.4~20.6
        {'train_start': pd.to_datetime('2015-07-01'), 'train_end': pd.to_datetime('2019-06-30'),
        'val_start'  : pd.to_datetime('2019-07-01'), 'val_end'  : pd.to_datetime('2019-12-31'),
        'test_start' : pd.to_datetime('2020-04-01'), 'test_end' : pd.to_datetime('2020-06-30')},
        # Fold 2: Train 16.1~19.12, Val 20.1~20.6, Buffer 20.7~20.9, Test 20.10~20.12
        {'train_start': pd.to_datetime('2016-01-01'), 'train_end': pd.to_datetime('2019-12-31'),
        'val_start'  : pd.to_datetime('2020-01-01'), 'val_end'  : pd.to_datetime('2020-06-30'),
        'test_start' : pd.to_datetime('2020-10-01'), 'test_end' : pd.to_datetime('2020-12-31')},
        # Fold 3: Train 16.7~20.6, Val 20.7~20.12, Buffer 21.1~21.3, Test 21.4~21.6
        {'train_start': pd.to_datetime('2016-07-01'), 'train_end': pd.to_datetime('2020-06-30'),
        'val_start'  : pd.to_datetime('2020-07-01'), 'val_end'  : pd.to_datetime('2020-12-31'),
        'test_start' : pd.to_datetime('2021-04-01'), 'test_end' : pd.to_datetime('2021-06-30')},
        # Fold 4: Train 17.1~20.12, Val 21.1~21.6, Buffer 21.7~21.9, Test 21.10~21.12
        {'train_start': pd.to_datetime('2017-01-01'), 'train_end': pd.to_datetime('2020-12-31'),
        'val_start'  : pd.to_datetime('2021-01-01'), 'val_end'  : pd.to_datetime('2021-06-30'),
        'test_start' : pd.to_datetime('2021-10-01'), 'test_end' : pd.to_datetime('2021-12-31')},
        # Fold 5: Train 17.7~21.6, Val 21.7~21.12, Buffer 22.1~22.3, Test 22.4~22.6
        {'train_start': pd.to_datetime('2017-07-01'), 'train_end': pd.to_datetime('2021-06-30'),
        'val_start'  : pd.to_datetime('2021-07-01'), 'val_end'  : pd.to_datetime('2021-12-31'),
        'test_start' : pd.to_datetime('2022-04-01'), 'test_end' : pd.to_datetime('2022-06-30')},
        # Fold 6: Train 18.1~21.12, Val 22.1~22.6, Buffer 22.7~22.9, Test 22.10~22.12
        {'train_start': pd.to_datetime('2018-01-01'), 'train_end': pd.to_datetime('2021-12-31'),
        'val_start'  : pd.to_datetime('2022-01-01'), 'val_end'  : pd.to_datetime('2022-06-30'),
        'test_start' : pd.to_datetime('2022-10-01'), 'test_end' : pd.to_datetime('2022-12-31')},
        # Fold 7: Train 18.7~22.6, Val 22.7~22.12, Buffer 23.1~23.3, Test 23.4~23.6
        {'train_start': pd.to_datetime('2018-07-01'), 'train_end': pd.to_datetime('2022-06-30'),
        'val_start'  : pd.to_datetime('2022-07-01'), 'val_end'  : pd.to_datetime('2022-12-31'),
        'test_start' : pd.to_datetime('2023-04-01'), 'test_end' : pd.to_datetime('2023-06-30')},
        # Fold 8: Train 19.1~22.12, Val 23.1~23.6, Buffer 23.7~23.9, Test 23.10~23.12
        {'train_start': pd.to_datetime('2019-01-01'), 'train_end': pd.to_datetime('2022-12-31'),
        'val_start'  : pd.to_datetime('2023-01-01'), 'val_end'  : pd.to_datetime('2023-06-30'),
        'test_start' : pd.to_datetime('2023-10-01'), 'test_end' : pd.to_datetime('2023-12-31')},
        # Fold 9: Train 19.7~23.6, Val 23.7~23.12, Buffer 24.1~24.3, Test 24.4~24.6
        {'train_start': pd.to_datetime('2019-07-01'), 'train_end': pd.to_datetime('2023-06-30'),
        'val_start'  : pd.to_datetime('2023-07-01'), 'val_end'  : pd.to_datetime('2023-12-31'),
        'test_start' : pd.to_datetime('2024-04-01'), 'test_end' : pd.to_datetime('2024-06-30')},
  ]

  

  # --- 설정 (이 숫자만 수정하여 폴드 선택) ---
  #fold_index = 9  # <<< 0부터 9까지의 숫자로 원하는 폴드를 선택

  # --- 선택된 폴드의 날짜로 마스크 자동 생성 ---
  selected_fold = folds[fold_index]

  train_mask = (dates >= selected_fold['train_start']) & (dates <= selected_fold['train_end'])
  val_mask   = (dates >= selected_fold['val_start'])   & (dates <= selected_fold['val_end'])
  test_mask  = (dates >= selected_fold['test_start'])  & (dates <= selected_fold['test_end'])

  if not np.any(train_mask): raise ValueError("No training data after split.")
  if not np.any(val_mask): raise ValueError("No val data after split.")
  if not np.any(test_mask): raise ValueError("No test data after split.")

  X_train = feature_data[train_mask]
  X_val = feature_data[val_mask]
  X_test = feature_data[test_mask]

  y_train = matched_targets[train_mask]
  y_val = matched_targets[val_mask]
  y_test = matched_targets[test_mask]

  if verbose:
    print("Train features shape:", X_train.shape)
    print("Val features shape:", X_val.shape)
    print("Test features shape:", X_test.shape)
    print("Train target size:", len(y_train))
    print("Val target size:", len(y_val))
    print("Test target size:", len(y_test))
    print("Train target distribution (Class 1):", np.mean(y_train))
    print("Val target distribution (Class 1):", np.mean(y_val))
    print("Test target distribution (Class 1):", np.mean(y_test))

  ############################################################
  # 7) 데이터 정규화
  ############################################################
  if verbose: print("\n--- Section 7: Scaling Features ---")
  feature_scaler = StandardScaler()
  train_features_scaled = feature_scaler.fit_transform(X_train)
  val_features_scaled = feature_scaler.transform(X_val)
  test_features_scaled = feature_scaler.transform(X_test)

  # Handle potential NaNs/Infs after scaling
  train_features_scaled = np.nan_to_num(train_features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
  val_features_scaled = np.nan_to_num(val_features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
  test_features_scaled = np.nan_to_num(test_features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

  if verbose:
    print("Scaled train features shape:", train_features_scaled.shape)
    print("Scaled val features shape:", val_features_scaled.shape)
    print("Scaled test features shape:", test_features_scaled.shape)

  #시드 고정 (변경 없음)
  import random
  import os
  seed_value = 42
  os.environ['PYTHONHASHSEED'] = str(seed_value)
  random.seed(seed_value)
  np.random.seed(seed_value)
  tf.random.set_seed(seed_value)
  if verbose: print(f"Set random seeds to {seed_value}")

  ############################################################
  # 8) 피처 이름 생성 (Feature Importance용)
  ############################################################
  feature_names = all_features

  if verbose:
    print("Scaled train features shape:", train_features_scaled.shape)
    print("Scaled val features shape:", val_features_scaled.shape)
    print("Scaled test features shape:", test_features_scaled.shape)
    print("Feature names:", feature_names)

  ############################################################
  # 9) 클래스 가중치 계산
  ############################################################
  n_negatives = np.sum(y_train == 0)
  n_positives = np.sum(y_train == 1)
  if n_positives == 0 or n_negatives == 0:
      if verbose: print("Warning: One class is missing in the training data. Setting scale_pos_weight=1.0")
      scale_pos_weight_value = 1.0
  else:
      # Ensure n_positives is not zero before division
      scale_pos_weight_value = n_negatives / n_positives if n_positives > 0 else 1.0
  if verbose: print(f"Calculated scale_pos_weight: {scale_pos_weight_value:.4f}")

  ############################################################
  # 9-1) 튜닝된 파라미터 로드 (선택사항)
  ############################################################
  tuned_params = None
  if use_tuned_params:
      import json
      from config import RESULTS_PATH
      params_file = os.path.join(RESULTS_PATH, f'best_params_{feature_set}.json')
      if os.path.exists(params_file):
          with open(params_file, 'r', encoding='utf-8') as f:
              tuned_params = json.load(f)
          if verbose: print(f"\n[INFO] Loaded tuned parameters from: {params_file}")
      else:
          if verbose: print(f"\n[WARNING] Tuned params file not found: {params_file}")
          if verbose: print("[WARNING] Using default parameters instead.")

  def get_tuned_param(model_name, param_name, default_value):
      """Helper function to get tuned parameter or default"""
      if tuned_params and model_name in tuned_params:
          best_params = tuned_params[model_name].get('best_params', {})
          if best_params and param_name in best_params:
              return best_params[param_name]
      return default_value

  ############################################################
  # 10) CatBoost 모델 학습 및 평가
  ############################################################
  if verbose: print("\n--- Section 10: Training and Evaluating CatBoost ---")

  # CatBoost 모델 선언 (튜닝된 파라미터 또는 기본값 사용)
  cat_model = CatBoostClassifier(
          iterations=5000,
          depth=get_tuned_param('CatBoost', 'depth', 7),
          learning_rate=get_tuned_param('CatBoost', 'learning_rate', 0.01),
          random_seed=seed_value,
          verbose=100 if verbose else False,
          loss_function='Logloss',
          early_stopping_rounds=1000,
          l2_leaf_reg=get_tuned_param('CatBoost', 'l2_leaf_reg', 100),
          random_strength=get_tuned_param('CatBoost', 'random_strength', 1),
          bootstrap_type='Bayesian',
          bagging_temperature=get_tuned_param('CatBoost', 'bagging_temperature', 10),
          eval_metric='AUC',
          use_best_model=True,
          scale_pos_weight=scale_pos_weight_value
  )

  # CatBoost 학습 (Scaled Features 사용)
  if verbose: print("--- Starting CatBoost Training ---")
  # >>> MODIFICATION START: Use scaled features for eval_set <<<
  eval_set = (val_features_scaled, y_val)
  cat_model.fit(train_features_scaled, y_train, eval_set=eval_set, verbose=100 if verbose else False)
  # >>> MODIFICATION END <<<

  # 최종 평가
  # >>> MODIFICATION START: Use scaled features for prediction <<<
  pred_train_proba_cat = cat_model.predict_proba(train_features_scaled)[:, 1]
  pred_test_proba_cat  = cat_model.predict_proba(test_features_scaled)[:, 1]
  # >>> MODIFICATION END <<<

  roc_auc_train_cat = roc_auc_score(y_train, pred_train_proba_cat)
  roc_auc_test_cat  = roc_auc_score(y_test, pred_test_proba_cat)
  ks_train_cat = calculate_ks(y_train, pred_train_proba_cat) # Changed variable name for clarity
  ks_test_cat = calculate_ks(y_test, pred_test_proba_cat)   # Changed variable name for clarity

  # Log Loss 및 Brier Score 계산
  logloss_train_cat = log_loss(y_train, pred_train_proba_cat)
  logloss_test_cat = log_loss(y_test, pred_test_proba_cat)
  brier_train_cat = brier_score_loss(y_train, pred_train_proba_cat)
  brier_test_cat = brier_score_loss(y_test, pred_test_proba_cat)

  # PR-AUC, Top Decile Lift, Recall@FPR 계산
  pr_auc_train_cat = average_precision_score(y_train, pred_train_proba_cat)
  pr_auc_test_cat = average_precision_score(y_test, pred_test_proba_cat)
  lift_train_cat = calculate_top_decile_lift(y_train, pred_train_proba_cat)
  lift_test_cat = calculate_top_decile_lift(y_test, pred_test_proba_cat)
  recall_at_fpr_train_cat = calculate_recall_at_fpr(y_train, pred_train_proba_cat, target_fpr=0.1)
  recall_at_fpr_test_cat = calculate_recall_at_fpr(y_test, pred_test_proba_cat, target_fpr=0.1)

  # 결과 딕셔너리
  results = {} # Initialize results dictionary here
  results['CatBoost'] = {'train_auc': roc_auc_train_cat, 'test_auc': roc_auc_test_cat,
                        'train_ks': ks_train_cat, 'test_ks': ks_test_cat,
                        'train_logloss': logloss_train_cat, 'test_logloss': logloss_test_cat,
                        'train_brier': brier_train_cat, 'test_brier': brier_test_cat,
                        'train_pr_auc': pr_auc_train_cat, 'test_pr_auc': pr_auc_test_cat,
                        'train_lift': lift_train_cat, 'test_lift': lift_test_cat,
                        'train_recall_at_fpr10': recall_at_fpr_train_cat, 'test_recall_at_fpr10': recall_at_fpr_test_cat}


  # 출력문
  if verbose:
    print(f'\n[CatBoost | {feature_set_name}] Train AUC: {roc_auc_train_cat:.5f} | Train KS: {ks_train_cat:.5f} | Train LogLoss: {logloss_train_cat:.5f} | Train Brier: {brier_train_cat:.5f}')
    print(f'[CatBoost | {feature_set_name}] Test  AUC: {roc_auc_test_cat:.5f}  | Test KS: {ks_test_cat:.5f}  | Test LogLoss: {logloss_test_cat:.5f}  | Test Brier: {brier_test_cat:.5f}')
    print(f'[CatBoost | {feature_set_name}] Test PR-AUC: {pr_auc_test_cat:.5f} | Test Lift@10%: {lift_test_cat:.3f} | Test Recall@FPR10%: {recall_at_fpr_test_cat:.5f}')


  # ROC Curve 시각화 (CatBoost)
  if verbose:
    fpr_train_cat, tpr_train_cat, _ = roc_curve(y_train, pred_train_proba_cat)
    fpr_test_cat,  tpr_test_cat,  _ = roc_curve(y_test, pred_test_proba_cat)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr_train_cat, tpr_train_cat, label=f'Train AUC: {roc_auc_train_cat:.5f}')
    plt.plot([0,1],[0,1],'k--')
    plt.title(f'ROC Curve - Train (CatBoost | {feature_set})')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(fpr_test_cat, tpr_test_cat, label=f'Test AUC: {roc_auc_test_cat:.5f}')
    plt.plot([0,1],[0,1],'k--')
    plt.title(f'ROC Curve - Test (CatBoost | {feature_set})')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()

    # 예측 확률 히스토그램 (CatBoost)
    plt.figure(figsize=(10, 4))
    plt.hist(pred_train_proba_cat, bins=50, alpha=0.7)
    plt.title(f'Train Prediction Probabilities (CatBoost | {feature_set})')
    plt.xlabel('Predicted Probability (Class 1)'); plt.ylabel('Frequency'); plt.grid(True, axis='y'); plt.show()
    plt.figure(figsize=(10, 4))
    plt.hist(pred_test_proba_cat, bins=50, alpha=0.7, color='orange')
    plt.title(f'Test Prediction Probabilities (CatBoost | {feature_set})')
    plt.xlabel('Predicted Probability (Class 1)'); plt.ylabel('Frequency'); plt.grid(True, axis='y'); plt.show()


  ############################################################
  # 14) 다른 모델 학습 및 평가 (LR, RF, XGB, LGBM)
  ############################################################
  if verbose: print("\n--- Section 14: Training and Evaluating Other Models ---")

  # 결과 저장 딕셔너리는 Section 13에서 초기화됨

  # --- Logistic Regression (LR) ---
  if verbose: print("\n--- Training and Evaluating Logistic Regression ---")
  try:
      lr_model = LogisticRegression(
          random_state=seed_value,
          class_weight='balanced',
          max_iter=1000,
          solver='liblinear',
          C=get_tuned_param('LogisticRegression', 'C', 1.0),
          penalty=get_tuned_param('LogisticRegression', 'penalty', 'l2')
      )
      lr_model.fit(train_features_scaled, y_train) # Use scaled features

      pred_train_proba_lr = lr_model.predict_proba(train_features_scaled)[:, 1]
      pred_test_proba_lr = lr_model.predict_proba(test_features_scaled)[:, 1]

      roc_auc_train_lr = roc_auc_score(y_train, pred_train_proba_lr)
      roc_auc_test_lr = roc_auc_score(y_test, pred_test_proba_lr)
      ks_train_lr = calculate_ks(y_train, pred_train_proba_lr) # Changed variable name
      ks_test_lr = calculate_ks(y_test, pred_test_proba_lr)   # Changed variable name

      logloss_train_lr = log_loss(y_train, pred_train_proba_lr)
      logloss_test_lr = log_loss(y_test, pred_test_proba_lr)
      brier_train_lr = brier_score_loss(y_train, pred_train_proba_lr)
      brier_test_lr = brier_score_loss(y_test, pred_test_proba_lr)
      pr_auc_train_lr = average_precision_score(y_train, pred_train_proba_lr)
      pr_auc_test_lr = average_precision_score(y_test, pred_test_proba_lr)
      lift_train_lr = calculate_top_decile_lift(y_train, pred_train_proba_lr)
      lift_test_lr = calculate_top_decile_lift(y_test, pred_test_proba_lr)
      recall_at_fpr_train_lr = calculate_recall_at_fpr(y_train, pred_train_proba_lr, target_fpr=0.1)
      recall_at_fpr_test_lr = calculate_recall_at_fpr(y_test, pred_test_proba_lr, target_fpr=0.1)

      if verbose:
        print(f'[LR] Train AUC: {roc_auc_train_lr:.5f} | Test AUC: {roc_auc_test_lr:.5f} | Test PR-AUC: {pr_auc_test_lr:.5f} | Test Lift@10%: {lift_test_lr:.3f}')

      results['LR'] = {'train_auc': roc_auc_train_lr, 'test_auc': roc_auc_test_lr,
                      'train_ks': ks_train_lr, 'test_ks': ks_test_lr,
                      'train_logloss': logloss_train_lr, 'test_logloss': logloss_test_lr,
                      'train_brier': brier_train_lr, 'test_brier': brier_test_lr,
                      'train_pr_auc': pr_auc_train_lr, 'test_pr_auc': pr_auc_test_lr,
                      'train_lift': lift_train_lr, 'test_lift': lift_test_lr,
                      'train_recall_at_fpr10': recall_at_fpr_train_lr, 'test_recall_at_fpr10': recall_at_fpr_test_lr}
  except Exception as e:
      if verbose: print(f"Error training/evaluating LR: {e}")
      results['LR'] = {'train_auc': np.nan, 'test_auc': np.nan, 'train_ks': np.nan, 'test_ks': np.nan,
                      'train_logloss': np.nan, 'test_logloss': np.nan, 'train_brier': np.nan, 'test_brier': np.nan,
                      'train_pr_auc': np.nan, 'test_pr_auc': np.nan, 'train_lift': np.nan, 'test_lift': np.nan,
                      'train_recall_at_fpr10': np.nan, 'test_recall_at_fpr10': np.nan}


  # --- Random Forest (RF) ---
  if verbose: print("\n--- Training and Evaluating Random Forest ---")
  try:
      custom_class_weight_for_rf = {0: 1.0, 1: scale_pos_weight_value}
      rf_model = RandomForestClassifier(
          random_state=seed_value,
          class_weight=custom_class_weight_for_rf,
          n_estimators=get_tuned_param('RandomForest', 'n_estimators', 200),
          n_jobs=-1,
          max_depth=get_tuned_param('RandomForest', 'max_depth', 7),
          min_samples_split=get_tuned_param('RandomForest', 'min_samples_split', 2),
          min_samples_leaf=get_tuned_param('RandomForest', 'min_samples_leaf', 5),
          max_features=get_tuned_param('RandomForest', 'max_features', 'sqrt')
      )
      rf_model.fit(train_features_scaled, y_train) # Use scaled features (less critical but consistent)

      pred_train_proba_rf = rf_model.predict_proba(train_features_scaled)[:, 1]
      pred_test_proba_rf = rf_model.predict_proba(test_features_scaled)[:, 1]

      roc_auc_train_rf = roc_auc_score(y_train, pred_train_proba_rf)
      roc_auc_test_rf = roc_auc_score(y_test, pred_test_proba_rf)
      ks_train_rf = calculate_ks(y_train, pred_train_proba_rf) # Changed variable name
      ks_test_rf = calculate_ks(y_test, pred_test_proba_rf)   # Changed variable name

      logloss_train_rf = log_loss(y_train, pred_train_proba_rf)
      logloss_test_rf = log_loss(y_test, pred_test_proba_rf)
      brier_train_rf = brier_score_loss(y_train, pred_train_proba_rf)
      brier_test_rf = brier_score_loss(y_test, pred_test_proba_rf)
      pr_auc_train_rf = average_precision_score(y_train, pred_train_proba_rf)
      pr_auc_test_rf = average_precision_score(y_test, pred_test_proba_rf)
      lift_train_rf = calculate_top_decile_lift(y_train, pred_train_proba_rf)
      lift_test_rf = calculate_top_decile_lift(y_test, pred_test_proba_rf)
      recall_at_fpr_train_rf = calculate_recall_at_fpr(y_train, pred_train_proba_rf, target_fpr=0.1)
      recall_at_fpr_test_rf = calculate_recall_at_fpr(y_test, pred_test_proba_rf, target_fpr=0.1)

      if verbose:
        print(f'[RF] Train AUC: {roc_auc_train_rf:.5f} | Test AUC: {roc_auc_test_rf:.5f} | Test PR-AUC: {pr_auc_test_rf:.5f} | Test Lift@10%: {lift_test_rf:.3f}')

      results['RF'] = {'train_auc': roc_auc_train_rf, 'test_auc': roc_auc_test_rf,
                      'train_ks': ks_train_rf, 'test_ks': ks_test_rf,
                      'train_logloss': logloss_train_rf, 'test_logloss': logloss_test_rf,
                      'train_brier': brier_train_rf, 'test_brier': brier_test_rf,
                      'train_pr_auc': pr_auc_train_rf, 'test_pr_auc': pr_auc_test_rf,
                      'train_lift': lift_train_rf, 'test_lift': lift_test_rf,
                      'train_recall_at_fpr10': recall_at_fpr_train_rf, 'test_recall_at_fpr10': recall_at_fpr_test_rf}
  except Exception as e:
      if verbose: print(f"Error training/evaluating RF: {e}")
      results['RF'] = {'train_auc': np.nan, 'test_auc': np.nan, 'train_ks': np.nan, 'test_ks': np.nan,
                      'train_logloss': np.nan, 'test_logloss': np.nan, 'train_brier': np.nan, 'test_brier': np.nan,
                      'train_pr_auc': np.nan, 'test_pr_auc': np.nan, 'train_lift': np.nan, 'test_lift': np.nan,
                      'train_recall_at_fpr10': np.nan, 'test_recall_at_fpr10': np.nan}


  # --- XGBoost (XGB) ---
  if verbose: print("\n--- Training and Evaluating XGBoost ---")
  try:
      xgb_model = xgb.XGBClassifier(
          objective='binary:logistic',
          eval_metric='auc',
          use_label_encoder=False,
          random_state=seed_value,
          n_estimators=1000,
          learning_rate=get_tuned_param('XGBoost', 'learning_rate', 0.05),
          max_depth=get_tuned_param('XGBoost', 'max_depth', 5),
          subsample=get_tuned_param('XGBoost', 'subsample', 0.8),
          colsample_bytree=get_tuned_param('XGBoost', 'colsample_bytree', 0.8),
          reg_alpha=get_tuned_param('XGBoost', 'reg_alpha', 0),
          reg_lambda=get_tuned_param('XGBoost', 'reg_lambda', 1),
          scale_pos_weight=scale_pos_weight_value,
          early_stopping_rounds=50
      )
      eval_set_xgb = [(val_features_scaled, y_val)] # Use scaled features
      xgb_model.fit(train_features_scaled, y_train, eval_set=eval_set_xgb, verbose=100 if verbose else False) # Pass eval_set

      pred_train_proba_xgb = xgb_model.predict_proba(train_features_scaled)[:, 1]
      # Use best iteration for prediction if early stopping occurred
      best_iteration_xgb = xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else None
      pred_test_proba_xgb = xgb_model.predict_proba(test_features_scaled, iteration_range=(0, best_iteration_xgb) if best_iteration_xgb is not None else None)[:, 1]


      roc_auc_train_xgb = roc_auc_score(y_train, pred_train_proba_xgb)
      roc_auc_test_xgb = roc_auc_score(y_test, pred_test_proba_xgb)
      ks_train_xgb = calculate_ks(y_train, pred_train_proba_xgb) # Changed variable name
      ks_test_xgb = calculate_ks(y_test, pred_test_proba_xgb)   # Changed variable name

      logloss_train_xgb = log_loss(y_train, pred_train_proba_xgb)
      logloss_test_xgb = log_loss(y_test, pred_test_proba_xgb)
      brier_train_xgb = brier_score_loss(y_train, pred_train_proba_xgb)
      brier_test_xgb = brier_score_loss(y_test, pred_test_proba_xgb)
      pr_auc_train_xgb = average_precision_score(y_train, pred_train_proba_xgb)
      pr_auc_test_xgb = average_precision_score(y_test, pred_test_proba_xgb)
      lift_train_xgb = calculate_top_decile_lift(y_train, pred_train_proba_xgb)
      lift_test_xgb = calculate_top_decile_lift(y_test, pred_test_proba_xgb)
      recall_at_fpr_train_xgb = calculate_recall_at_fpr(y_train, pred_train_proba_xgb, target_fpr=0.1)
      recall_at_fpr_test_xgb = calculate_recall_at_fpr(y_test, pred_test_proba_xgb, target_fpr=0.1)

      if verbose:
        print(f'[XGB] Train AUC: {roc_auc_train_xgb:.5f} | Test AUC: {roc_auc_test_xgb:.5f} | Test PR-AUC: {pr_auc_test_xgb:.5f} | Test Lift@10%: {lift_test_xgb:.3f}')

      results['XGB'] = {'train_auc': roc_auc_train_xgb, 'test_auc': roc_auc_test_xgb,
                        'train_ks': ks_train_xgb, 'test_ks': ks_test_xgb,
                        'train_logloss': logloss_train_xgb, 'test_logloss': logloss_test_xgb,
                        'train_brier': brier_train_xgb, 'test_brier': brier_test_xgb,
                        'train_pr_auc': pr_auc_train_xgb, 'test_pr_auc': pr_auc_test_xgb,
                        'train_lift': lift_train_xgb, 'test_lift': lift_test_xgb,
                        'train_recall_at_fpr10': recall_at_fpr_train_xgb, 'test_recall_at_fpr10': recall_at_fpr_test_xgb}
  except Exception as e:
      if verbose: print(f"Error training/evaluating XGB: {e}")
      results['XGB'] = {'train_auc': np.nan, 'test_auc': np.nan, 'train_ks': np.nan, 'test_ks': np.nan,
                        'train_logloss': np.nan, 'test_logloss': np.nan, 'train_brier': np.nan, 'test_brier': np.nan,
                        'train_pr_auc': np.nan, 'test_pr_auc': np.nan, 'train_lift': np.nan, 'test_lift': np.nan,
                        'train_recall_at_fpr10': np.nan, 'test_recall_at_fpr10': np.nan}

  # --- LightGBM (LGBM) ---
  if verbose: print("\n--- Training and Evaluating LightGBM ---")
  try:
      lgbm_model = lgb.LGBMClassifier(
          objective='binary',
          metric='auc',
          random_state=seed_value,
          n_estimators=1000,
          learning_rate=get_tuned_param('LightGBM', 'learning_rate', 0.05),
          num_leaves=get_tuned_param('LightGBM', 'num_leaves', 31),
          max_depth=get_tuned_param('LightGBM', 'max_depth', -1),
          subsample=get_tuned_param('LightGBM', 'subsample', 0.8),
          colsample_bytree=get_tuned_param('LightGBM', 'colsample_bytree', 0.8),
          reg_alpha=get_tuned_param('LightGBM', 'reg_alpha', 0),
          reg_lambda=get_tuned_param('LightGBM', 'reg_lambda', 0),
          min_child_samples=get_tuned_param('LightGBM', 'min_child_samples', 20),
          scale_pos_weight=scale_pos_weight_value,
          n_jobs=-1,
          verbose=-1
      )
      eval_set_lgbm = [(val_features_scaled, y_val)] # Use scaled features
      # Need callbacks for early stopping in LGBM
      callbacks_lgbm = [lgb.early_stopping(stopping_rounds=50, verbose=100 if verbose else False)]
      lgbm_model.fit(train_features_scaled, y_train, eval_set=eval_set_lgbm, callbacks=callbacks_lgbm) # Pass eval_set and callbacks

      pred_train_proba_lgbm = lgbm_model.predict_proba(train_features_scaled)[:, 1]
      # Use best iteration for prediction if early stopping occurred
      best_iteration_lgbm = lgbm_model.best_iteration_ if hasattr(lgbm_model, 'best_iteration_') else None
      pred_test_proba_lgbm = lgbm_model.predict_proba(test_features_scaled, num_iteration=best_iteration_lgbm if best_iteration_lgbm is not None else None)[:, 1]


      roc_auc_train_lgbm = roc_auc_score(y_train, pred_train_proba_lgbm)
      roc_auc_test_lgbm = roc_auc_score(y_test, pred_test_proba_lgbm)
      ks_train_lgbm = calculate_ks(y_train, pred_train_proba_lgbm) # Changed variable name
      ks_test_lgbm = calculate_ks(y_test, pred_test_proba_lgbm)   # Changed variable name

      logloss_train_lgbm = log_loss(y_train, pred_train_proba_lgbm)
      logloss_test_lgbm = log_loss(y_test, pred_test_proba_lgbm)
      brier_train_lgbm = brier_score_loss(y_train, pred_train_proba_lgbm)
      brier_test_lgbm = brier_score_loss(y_test, pred_test_proba_lgbm)
      pr_auc_train_lgbm = average_precision_score(y_train, pred_train_proba_lgbm)
      pr_auc_test_lgbm = average_precision_score(y_test, pred_test_proba_lgbm)
      lift_train_lgbm = calculate_top_decile_lift(y_train, pred_train_proba_lgbm)
      lift_test_lgbm = calculate_top_decile_lift(y_test, pred_test_proba_lgbm)
      recall_at_fpr_train_lgbm = calculate_recall_at_fpr(y_train, pred_train_proba_lgbm, target_fpr=0.1)
      recall_at_fpr_test_lgbm = calculate_recall_at_fpr(y_test, pred_test_proba_lgbm, target_fpr=0.1)

      if verbose:
        print(f'[LGBM] Train AUC: {roc_auc_train_lgbm:.5f} | Test AUC: {roc_auc_test_lgbm:.5f} | Test PR-AUC: {pr_auc_test_lgbm:.5f} | Test Lift@10%: {lift_test_lgbm:.3f}')

      results['LGBM'] = {'train_auc': roc_auc_train_lgbm, 'test_auc': roc_auc_test_lgbm,
                        'train_ks': ks_train_lgbm, 'test_ks': ks_test_lgbm,
                        'train_logloss': logloss_train_lgbm, 'test_logloss': logloss_test_lgbm,
                        'train_brier': brier_train_lgbm, 'test_brier': brier_test_lgbm,
                        'train_pr_auc': pr_auc_train_lgbm, 'test_pr_auc': pr_auc_test_lgbm,
                        'train_lift': lift_train_lgbm, 'test_lift': lift_test_lgbm,
                        'train_recall_at_fpr10': recall_at_fpr_train_lgbm, 'test_recall_at_fpr10': recall_at_fpr_test_lgbm}
  except Exception as e:
      if verbose: print(f"Error training/evaluating LGBM: {e}")
      results['LGBM'] = {'train_auc': np.nan, 'test_auc': np.nan, 'train_ks': np.nan, 'test_ks': np.nan,
                        'train_logloss': np.nan, 'test_logloss': np.nan, 'train_brier': np.nan, 'test_brier': np.nan,
                        'train_pr_auc': np.nan, 'test_pr_auc': np.nan, 'train_lift': np.nan, 'test_lift': np.nan,
                        'train_recall_at_fpr10': np.nan, 'test_recall_at_fpr10': np.nan}

  # --- Feature Importance Visualization for LightGBM ---
  import seaborn as sns # <--- Add this import!
  if verbose: print("\n--- Visualizing LightGBM Feature Importance ---")

  if verbose:
    try:
        # Get feature importances
        importances = lgbm_model.feature_importances_

        # Get feature names from your training data
        if hasattr(train_features_scaled, 'columns'):
            feature_names = train_features_scaled.columns
        else:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
            print("Warning: `train_features_scaled` does not have column names. Using generic feature names.")

        # Create a DataFrame for easy sorting and plotting
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })

        # Sort by importance in descending order
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Plotting
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, color='orange')
        plt.title('Feature Importance', fontsize=16, color='dimgray')
        plt.xlabel('Importance', fontsize=12, color='dimgray')
        plt.ylabel('Feature', fontsize=12, color='dimgray')
        plt.grid(axis='x', linestyle='--', alpha=0.6, color='lightgray')
        plt.grid(axis='y', linestyle='--', alpha=0.6, color='lightgray')
        plt.tick_params(axis='x', colors='dimgray')
        plt.tick_params(axis='y', colors='dimgray')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error visualizing feature importance: {e}")


  ############################################################
  # 15) 결과 요약
  ############################################################
  if verbose: print("\n--- Section 15: Model Performance Summary ---")
  # 라인 546: 결과 DataFrame 생성 시 LogLoss, Brier Score 자동 포함됨
  results_df = pd.DataFrame.from_dict(results, orient='index')
  # Display all columns for better readability
  pd.set_option('display.max_columns', None)
  pd.set_option('display.width', 1000)
  if verbose: print(results_df)



  ############################################################
  # 16) LightGBM Feature Importance (추가)
  ############################################################
  if verbose: print("\n--- Section 16: LightGBM Feature Importance ---")

  if verbose:
    try:
        # all_features를 그대로 사용 (이미 정의됨)
        importances = lgbm_model.feature_importances_
        importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

        print(f"\nTop 20 LightGBM Feature Importances ({feature_set}):")
        print(importance_df.head(20))

        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20], color='skyblue')
        plt.xlabel("LightGBM Feature Importance")
        plt.ylabel("Feature")
        plt.title(f"Top 20 LightGBM Feature Importances ({feature_set})")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error displaying feature importances: {e}")

  if verbose: print("\n Tree Script finished.")

  # ############################################################
  # # --- PREVIOUS CODE (Sections 1-14) ---
  # # This part includes all the code you provided, ending
  # # with the evaluation of LR, RF, XGB, LGBM in Section 14.
  # # Assume results dictionary 'results' is initialized and
  # # populated by previous models (CatBoost, LR, RF, XGB, LGBM)
  # # Assume train_features_scaled, test_features_scaled,
  # # y_train, y_test, and scale_pos_weight_value are available.
  # ############################################################

  # >>> ADDITION START: Imports needed for DL models (Add near top of script) <<<
  from tensorflow.keras.models import Model
  from tensorflow.keras.layers import Input, Dense, Dropout, Reshape, Conv1D, MaxPooling1D, LSTM, GRU, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
  from tensorflow.keras.optimizers import Adam # Keep Adam import
  from tensorflow.keras.callbacks import EarlyStopping # Keep EarlyStopping import
  # >>> ADDITION END <<<


  # >>> ADDITION START: Calculate Class Weights for Keras (Add after Section 12 / before Section 13) <<<
  # Calculate class weights for Keras models
  if scale_pos_weight_value == 1.0:
      keras_class_weight = None # No weighting needed
      if verbose: print("Keras class_weight: None (balanced or single class)")
  else:
      keras_class_weight = {0: 1.0, 1: scale_pos_weight_value}
      if verbose: print(f"Keras class_weight: {keras_class_weight}")
  # >>> ADDITION END <<<


  ############################################################
  # 15) Deep Learning Model Definitions (Using Combined Features)
  ############################################################
  if verbose: print("\n--- Section 15: Defining Deep Learning Models (on Combined Features) ---")

  # --- Define Input Shape ---
  # Input shape is the number of combined features
  input_shape_flat = (train_features_scaled.shape[1],)
  # Reshape target for CNN/RNN layers: (batch_size, steps, features)
  # We'll treat the flat features as 'steps' with 1 feature per step.
  reshape_target = (train_features_scaled.shape[1], 1)
  if verbose:
    print(f"DL Input Shape (Flat): {input_shape_flat}")
    print(f"DL Reshape Target: {reshape_target}")


  # --- CNN Model ---
  def build_cnn_model(input_shape, reshape_target_cnn, learning_rate=0.001, dropout_rate=0.3,
                      filters_1=32, filters_2=64, dense_units=64, verbose_model=True):
      input_layer = Input(shape=input_shape, name='Input_Combined')
      reshaped = Reshape(target_shape=reshape_target_cnn, name='Reshape')(input_layer)
      conv1 = Conv1D(filters=filters_1, kernel_size=3, activation='relu', padding='same', name='Conv1D_1')(reshaped)
      pool1 = MaxPooling1D(pool_size=2, name='MaxPool_1')(conv1)
      conv2 = Conv1D(filters=filters_2, kernel_size=3, activation='relu', padding='same', name='Conv1D_2')(pool1)
      gap = GlobalAveragePooling1D(name='GlobalAvgPool')(conv2)
      dense1 = Dense(dense_units, activation='relu', name='Dense_1')(gap)
      dropout1 = Dropout(dropout_rate, name='Dropout_1')(dense1)
      output_layer = Dense(1, activation='sigmoid', name='Output_Sigmoid')(dropout1)

      model = Model(inputs=input_layer, outputs=output_layer)
      optimizer = Adam(learning_rate=learning_rate)
      model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
      if verbose_model:
        print("--- CNN Model Summary ---")
        model.summary()
      return model

  # --- LSTM Model ---
  def build_lstm_model(input_shape, reshape_target_lstm, learning_rate=0.001, dropout_rate=0.3,
                       lstm_units=50, dense_units=32, verbose_model=True):
      input_layer = Input(shape=input_shape, name='Input_Combined')
      reshaped = Reshape(target_shape=reshape_target_lstm, name='Reshape')(input_layer)
      lstm1 = LSTM(units=lstm_units, return_sequences=False, name='LSTM_1')(reshaped)
      dense1 = Dense(dense_units, activation='relu', name='Dense_1')(lstm1)
      dropout1 = Dropout(dropout_rate, name='Dropout_1')(dense1)
      output_layer = Dense(1, activation='sigmoid', name='Output_Sigmoid')(dropout1)

      model = Model(inputs=input_layer, outputs=output_layer)
      optimizer = Adam(learning_rate=learning_rate)
      model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
      if verbose_model:
        print("--- LSTM Model Summary ---")
        model.summary()
      return model

  # --- GRU Model ---
  def build_gru_model(input_shape, reshape_target_gru, learning_rate=0.001, dropout_rate=0.3,
                      gru_units=50, dense_units=32, verbose_model=True):
      input_layer = Input(shape=input_shape, name='Input_Combined')
      reshaped = Reshape(target_shape=reshape_target_gru, name='Reshape')(input_layer)
      gru1 = GRU(units=gru_units, return_sequences=False, name='GRU_1')(reshaped)
      dense1 = Dense(dense_units, activation='relu', name='Dense_1')(gru1)
      dropout1 = Dropout(dropout_rate, name='Dropout_1')(dense1)
      output_layer = Dense(1, activation='sigmoid', name='Output_Sigmoid')(dropout1)

      model = Model(inputs=input_layer, outputs=output_layer)
      optimizer = Adam(learning_rate=learning_rate)
      model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
      if verbose_model:
        print("--- GRU Model Summary ---")
        model.summary()
      return model

  # --- Transformer Encoder Block Model ---
  def build_transformer_model(input_shape, reshape_target_tf, head_size=256, num_heads=4, ff_dim=4, dropout=0.2, learning_rate=0.001, verbose_model=True):
      input_layer = Input(shape=input_shape, name='Input_Combined')
      reshaped = Reshape(target_shape=reshape_target_tf, name='Reshape')(input_layer)

      norm1 = LayerNormalization(epsilon=1e-6, name='LayerNorm_1')(reshaped)
      attn_output = MultiHeadAttention(
          num_heads=num_heads, key_dim=head_size // num_heads, dropout=dropout, name='MultiHeadAttention'
      )(norm1, norm1)
      add1 = tf.keras.layers.Add(name='Add_1')([reshaped, attn_output])

      norm2 = LayerNormalization(epsilon=1e-6, name='LayerNorm_2')(add1)
      ffn1 = Dense(ff_dim, activation="relu", name='FFN_1')(norm2)
      ffn_dropout = Dropout(dropout, name='FFN_Dropout')(ffn1)
      ffn2 = Dense(reshape_target_tf[-1], name='FFN_2')(ffn_dropout)
      add2 = tf.keras.layers.Add(name='Add_2')([add1, ffn2])

      pool = GlobalAveragePooling1D(name='GlobalAvgPool')(add2)
      dense_out1 = Dense(64, activation="relu", name='Dense_Out_1')(pool)
      dropout_out = Dropout(0.3, name='Dropout_Out')(dense_out1)
      output_layer = Dense(1, activation="sigmoid", name='Output_Sigmoid')(dropout_out)

      model = Model(inputs=input_layer, outputs=output_layer)
      optimizer = Adam(learning_rate=learning_rate)
      model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
      if verbose_model:
        print("--- Transformer Model Summary ---")
        model.summary()
      return model

  from tensorflow.keras.layers import Layer

  class PositionalEmbedding(Layer):
      def __init__(self, sequence_length, output_dim, **kwargs):
          super().__init__(**kwargs)
          self.position_embeddings = self.add_weight(
              shape=(sequence_length, output_dim),
              initializer="uniform",
              trainable=True,
              name="position_embeddings"
          )
          self.sequence_length = sequence_length
          self.output_dim = output_dim

      def call(self, inputs):
          # 입력 텐서에 위치 임베딩을 더해줍니다.
          return inputs + self.position_embeddings

      def compute_mask(self, inputs, mask=None):
          return mask
  # >>>>> 추가 끝 <<<<<

  # --- Transformer Encoder Block Model ---
  def build_transformer_model_pos_encoding(input_shape, reshape_target_tf, head_size=64, num_heads=4, ff_dim=4, dropout=0.2, learning_rate=0.01, verbose_model=True):
      input_layer = Input(shape=input_shape, name='Input_Combined')
      reshaped = Reshape(target_shape=reshape_target_tf, name='Reshape')(input_layer)

      sequence_length = reshape_target_tf[0]
      output_dim = reshape_target_tf[1]
      x = Dense(64)(reshaped)
      x = PositionalEmbedding(sequence_length=sequence_length, output_dim=output_dim)(reshaped)

      norm1 = LayerNormalization(epsilon=1e-6, name='LayerNorm_1')(x)
      attn_output = MultiHeadAttention(
          num_heads=num_heads, key_dim=head_size // num_heads, dropout=dropout, name='MultiHeadAttention'
      )(norm1, norm1)
      add1 = tf.keras.layers.Add(name='Add_1')([x, attn_output])

      norm2 = LayerNormalization(epsilon=1e-6, name='LayerNorm_2')(add1)
      ffn1 = Dense(ff_dim, activation="relu", name='FFN_1')(norm2)
      ffn_dropout = Dropout(dropout, name='FFN_Dropout')(ffn1)
      ffn2 = Dense(reshape_target_tf[-1], name='FFN_2')(ffn_dropout)
      add2 = tf.keras.layers.Add(name='Add_2')([add1, ffn2])

      pool = GlobalAveragePooling1D(name='GlobalAvgPool')(add2)
      dense_out1 = Dense(64, activation="relu", name='Dense_Out_1')(pool)
      dropout_out = Dropout(0.3, name='Dropout_Out')(dense_out1)
      output_layer = Dense(1, activation="sigmoid", name='Output_Sigmoid')(dropout_out)

      model = Model(inputs=input_layer, outputs=output_layer)
      optimizer = Adam(learning_rate=learning_rate)
      model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
      if verbose_model:
        print("--- Transformer Model Summary (with Positional Embedding) ---")
        model.summary()
      return model

  # 동일한 입력을 받되, 내부 처리 방식을 개선한 트랜스포머
  def build_tabular_transformer_model(input_shape, head_size=64, num_heads=4, ff_dim=32, embedding_dim=16, dropout=0.2, learning_rate=0.001, verbose_model=True):
      input_layer = Input(shape=input_shape, name='Input_Flat_Features')
      reshaped = Reshape((input_shape[0], 1), name='Reshape_for_Embedding')(input_layer)
      feature_embedding = Dense(embedding_dim, activation='relu', name='Feature_Embedding')(reshaped)

      norm1 = LayerNormalization(epsilon=1e-6)(feature_embedding)
      attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size // num_heads, dropout=dropout)(norm1, norm1)
      add1 = tf.keras.layers.Add()([feature_embedding, attn_output])

      norm2 = LayerNormalization(epsilon=1e-6)(add1)
      ffn = Dense(ff_dim, activation="relu")(norm2)
      ffn = Dense(embedding_dim)(ffn)
      add2 = tf.keras.layers.Add()([add1, ffn])

      pool = GlobalAveragePooling1D(name='GlobalAvgPool')(add2)
      dense_out = Dense(64, activation="relu")(pool)
      dropout_out = Dropout(0.3)(dense_out)
      output_layer = Dense(1, activation="sigmoid", name='Output_Sigmoid')(dropout_out)

      model = Model(inputs=input_layer, outputs=output_layer)
      optimizer = Adam(learning_rate=learning_rate)
      model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])

      if verbose_model:
        print("--- Tabular Transformer Model Summary ---")
        model.summary()
      return model

  ############################################################
  # 16) Deep Learning Model Training and Evaluation
  ############################################################
  if verbose: print("\n--- Section 16: Training and Evaluating Deep Learning Models ---")

  # --- Training Parameters ---
  epochs = 50
  batch_size = 128
  patience = 10

  # Common Early Stopping Callback
  early_stopping = EarlyStopping(monitor='val_loss',
                                patience=patience,
                                restore_best_weights=True,
                                verbose=1 if verbose else 0)

  # --- Train and Evaluate CNN ---
  if verbose: print("\n--- Training and Evaluating CNN ---")
  try:
      cnn_batch_size = get_tuned_param('CNN', 'batch_size', batch_size)
      model_cnn = build_cnn_model(
          input_shape_flat, reshape_target,
          learning_rate=get_tuned_param('CNN', 'learning_rate', 0.001),
          dropout_rate=get_tuned_param('CNN', 'dropout_rate', 0.3),
          filters_1=get_tuned_param('CNN', 'filters_1', 32),
          filters_2=get_tuned_param('CNN', 'filters_2', 64),
          dense_units=get_tuned_param('CNN', 'dense_units', 64),
          verbose_model=verbose
      )
      history_cnn = model_cnn.fit(train_features_scaled, y_train,
                                  validation_data=(val_features_scaled, y_val),
                                  epochs=epochs,
                                  batch_size=cnn_batch_size,
                                  callbacks=[early_stopping],
                                  class_weight=keras_class_weight,
                                  verbose=1 if verbose else 0)

      pred_train_proba_cnn = model_cnn.predict(train_features_scaled).ravel()
      pred_test_proba_cnn = model_cnn.predict(test_features_scaled).ravel()

      roc_auc_train_cnn = roc_auc_score(y_train, pred_train_proba_cnn)
      roc_auc_test_cnn = roc_auc_score(y_test, pred_test_proba_cnn)
      ks_train_cnn = calculate_ks(y_train, pred_train_proba_cnn)
      ks_test_cnn = calculate_ks(y_test, pred_test_proba_cnn)
      logloss_train_cnn = log_loss(y_train, pred_train_proba_cnn)
      logloss_test_cnn = log_loss(y_test, pred_test_proba_cnn)
      brier_train_cnn = brier_score_loss(y_train, pred_train_proba_cnn)
      brier_test_cnn = brier_score_loss(y_test, pred_test_proba_cnn)
      pr_auc_train_cnn = average_precision_score(y_train, pred_train_proba_cnn)
      pr_auc_test_cnn = average_precision_score(y_test, pred_test_proba_cnn)
      lift_train_cnn = calculate_top_decile_lift(y_train, pred_train_proba_cnn)
      lift_test_cnn = calculate_top_decile_lift(y_test, pred_test_proba_cnn)
      recall_at_fpr_train_cnn = calculate_recall_at_fpr(y_train, pred_train_proba_cnn, target_fpr=0.1)
      recall_at_fpr_test_cnn = calculate_recall_at_fpr(y_test, pred_test_proba_cnn, target_fpr=0.1)

      if verbose:
        print(f'[CNN] Train AUC: {roc_auc_train_cnn:.5f} | Test AUC: {roc_auc_test_cnn:.5f} | Test PR-AUC: {pr_auc_test_cnn:.5f} | Test Lift@10%: {lift_test_cnn:.3f}')

      results['CNN'] = {'train_auc': roc_auc_train_cnn, 'test_auc': roc_auc_test_cnn,
                        'train_ks': ks_train_cnn, 'test_ks': ks_test_cnn,
                        'train_logloss': logloss_train_cnn, 'test_logloss': logloss_test_cnn,
                        'train_brier': brier_train_cnn, 'test_brier': brier_test_cnn,
                        'train_pr_auc': pr_auc_train_cnn, 'test_pr_auc': pr_auc_test_cnn,
                        'train_lift': lift_train_cnn, 'test_lift': lift_test_cnn,
                        'train_recall_at_fpr10': recall_at_fpr_train_cnn, 'test_recall_at_fpr10': recall_at_fpr_test_cnn}
  except Exception as e:
      if verbose: print(f"Error training/evaluating CNN: {e}")
      results['CNN'] = {'train_auc': np.nan, 'test_auc': np.nan, 'train_ks': np.nan, 'test_ks': np.nan,
                        'train_logloss': np.nan, 'test_logloss': np.nan, 'train_brier': np.nan, 'test_brier': np.nan,
                        'train_pr_auc': np.nan, 'test_pr_auc': np.nan, 'train_lift': np.nan, 'test_lift': np.nan,
                        'train_recall_at_fpr10': np.nan, 'test_recall_at_fpr10': np.nan}


  # --- Train and Evaluate LSTM ---
  if verbose: print("\n--- Training and Evaluating LSTM ---")
  try:
      lstm_batch_size = get_tuned_param('LSTM', 'batch_size', batch_size)
      model_lstm = build_lstm_model(
          input_shape_flat, reshape_target,
          learning_rate=get_tuned_param('LSTM', 'learning_rate', 0.001),
          dropout_rate=get_tuned_param('LSTM', 'dropout_rate', 0.3),
          lstm_units=get_tuned_param('LSTM', 'lstm_units', 50),
          dense_units=get_tuned_param('LSTM', 'dense_units', 32),
          verbose_model=verbose
      )
      history_lstm = model_lstm.fit(train_features_scaled, y_train,
                                    validation_data=(val_features_scaled, y_val),
                                    epochs=epochs,
                                    batch_size=lstm_batch_size,
                                    callbacks=[early_stopping],
                                    class_weight=keras_class_weight,
                                    verbose=1 if verbose else 0)

      pred_train_proba_lstm = model_lstm.predict(train_features_scaled).ravel()
      pred_test_proba_lstm = model_lstm.predict(test_features_scaled).ravel()

      roc_auc_train_lstm = roc_auc_score(y_train, pred_train_proba_lstm)
      roc_auc_test_lstm = roc_auc_score(y_test, pred_test_proba_lstm)
      ks_train_lstm = calculate_ks(y_train, pred_train_proba_lstm)
      ks_test_lstm = calculate_ks(y_test, pred_test_proba_lstm)
      logloss_train_lstm = log_loss(y_train, pred_train_proba_lstm)
      logloss_test_lstm = log_loss(y_test, pred_test_proba_lstm)
      brier_train_lstm = brier_score_loss(y_train, pred_train_proba_lstm)
      brier_test_lstm = brier_score_loss(y_test, pred_test_proba_lstm)
      pr_auc_train_lstm = average_precision_score(y_train, pred_train_proba_lstm)
      pr_auc_test_lstm = average_precision_score(y_test, pred_test_proba_lstm)
      lift_train_lstm = calculate_top_decile_lift(y_train, pred_train_proba_lstm)
      lift_test_lstm = calculate_top_decile_lift(y_test, pred_test_proba_lstm)
      recall_at_fpr_train_lstm = calculate_recall_at_fpr(y_train, pred_train_proba_lstm, target_fpr=0.1)
      recall_at_fpr_test_lstm = calculate_recall_at_fpr(y_test, pred_test_proba_lstm, target_fpr=0.1)

      if verbose:
        print(f'[LSTM] Train AUC: {roc_auc_train_lstm:.5f} | Test AUC: {roc_auc_test_lstm:.5f} | Test PR-AUC: {pr_auc_test_lstm:.5f} | Test Lift@10%: {lift_test_lstm:.3f}')

      results['LSTM'] = {'train_auc': roc_auc_train_lstm, 'test_auc': roc_auc_test_lstm,
                        'train_ks': ks_train_lstm, 'test_ks': ks_test_lstm,
                        'train_logloss': logloss_train_lstm, 'test_logloss': logloss_test_lstm,
                        'train_brier': brier_train_lstm, 'test_brier': brier_test_lstm,
                        'train_pr_auc': pr_auc_train_lstm, 'test_pr_auc': pr_auc_test_lstm,
                        'train_lift': lift_train_lstm, 'test_lift': lift_test_lstm,
                        'train_recall_at_fpr10': recall_at_fpr_train_lstm, 'test_recall_at_fpr10': recall_at_fpr_test_lstm}
  except Exception as e:
      if verbose: print(f"Error training/evaluating LSTM: {e}")
      results['LSTM'] = {'train_auc': np.nan, 'test_auc': np.nan, 'train_ks': np.nan, 'test_ks': np.nan,
                        'train_logloss': np.nan, 'test_logloss': np.nan, 'train_brier': np.nan, 'test_brier': np.nan,
                        'train_pr_auc': np.nan, 'test_pr_auc': np.nan, 'train_lift': np.nan, 'test_lift': np.nan,
                        'train_recall_at_fpr10': np.nan, 'test_recall_at_fpr10': np.nan}


  # --- Train and Evaluate GRU ---
  if verbose: print("\n--- Training and Evaluating GRU ---")
  try:
      gru_batch_size = get_tuned_param('GRU', 'batch_size', batch_size)
      model_gru = build_gru_model(
          input_shape_flat, reshape_target,
          learning_rate=get_tuned_param('GRU', 'learning_rate', 0.001),
          dropout_rate=get_tuned_param('GRU', 'dropout_rate', 0.3),
          gru_units=get_tuned_param('GRU', 'gru_units', 50),
          dense_units=get_tuned_param('GRU', 'dense_units', 32),
          verbose_model=verbose
      )
      history_gru = model_gru.fit(train_features_scaled, y_train,
                                  validation_data=(val_features_scaled, y_val),
                                  epochs=epochs,
                                  batch_size=gru_batch_size,
                                  callbacks=[early_stopping],
                                  class_weight=keras_class_weight,
                                  verbose=1 if verbose else 0)

      pred_train_proba_gru = model_gru.predict(train_features_scaled).ravel()
      pred_test_proba_gru = model_gru.predict(test_features_scaled).ravel()

      roc_auc_train_gru = roc_auc_score(y_train, pred_train_proba_gru)
      roc_auc_test_gru = roc_auc_score(y_test, pred_test_proba_gru)
      ks_train_gru = calculate_ks(y_train, pred_train_proba_gru)
      ks_test_gru = calculate_ks(y_test, pred_test_proba_gru)
      logloss_train_gru = log_loss(y_train, pred_train_proba_gru)
      logloss_test_gru = log_loss(y_test, pred_test_proba_gru)
      brier_train_gru = brier_score_loss(y_train, pred_train_proba_gru)
      brier_test_gru = brier_score_loss(y_test, pred_test_proba_gru)
      pr_auc_train_gru = average_precision_score(y_train, pred_train_proba_gru)
      pr_auc_test_gru = average_precision_score(y_test, pred_test_proba_gru)
      lift_train_gru = calculate_top_decile_lift(y_train, pred_train_proba_gru)
      lift_test_gru = calculate_top_decile_lift(y_test, pred_test_proba_gru)
      recall_at_fpr_train_gru = calculate_recall_at_fpr(y_train, pred_train_proba_gru, target_fpr=0.1)
      recall_at_fpr_test_gru = calculate_recall_at_fpr(y_test, pred_test_proba_gru, target_fpr=0.1)

      if verbose:
        print(f'[GRU] Train AUC: {roc_auc_train_gru:.5f} | Test AUC: {roc_auc_test_gru:.5f} | Test PR-AUC: {pr_auc_test_gru:.5f} | Test Lift@10%: {lift_test_gru:.3f}')

      results['GRU'] = {'train_auc': roc_auc_train_gru, 'test_auc': roc_auc_test_gru,
                        'train_ks': ks_train_gru, 'test_ks': ks_test_gru,
                        'train_logloss': logloss_train_gru, 'test_logloss': logloss_test_gru,
                        'train_brier': brier_train_gru, 'test_brier': brier_test_gru,
                        'train_pr_auc': pr_auc_train_gru, 'test_pr_auc': pr_auc_test_gru,
                        'train_lift': lift_train_gru, 'test_lift': lift_test_gru,
                        'train_recall_at_fpr10': recall_at_fpr_train_gru, 'test_recall_at_fpr10': recall_at_fpr_test_gru}
  except Exception as e:
      if verbose: print(f"Error training/evaluating GRU: {e}")
      results['GRU'] = {'train_auc': np.nan, 'test_auc': np.nan, 'train_ks': np.nan, 'test_ks': np.nan,
                        'train_logloss': np.nan, 'test_logloss': np.nan, 'train_brier': np.nan, 'test_brier': np.nan,
                        'train_pr_auc': np.nan, 'test_pr_auc': np.nan, 'train_lift': np.nan, 'test_lift': np.nan,
                        'train_recall_at_fpr10': np.nan, 'test_recall_at_fpr10': np.nan}


  # --- Train and Evaluate Transformer ---
  if verbose: print("\n--- Training and Evaluating Transformer ---")
  try:
      # 1. 수정된 트랜스포머 모델 생성
      input_shape_flat = (train_features_scaled.shape[1],)
      tf_batch_size = get_tuned_param('Transformer', 'batch_size', batch_size)

      model_tf = build_tabular_transformer_model(
          input_shape_flat,
          head_size=get_tuned_param('Transformer', 'head_size', 128),
          num_heads=get_tuned_param('Transformer', 'num_heads', 4),
          ff_dim=get_tuned_param('Transformer', 'ff_dim', 64),
          embedding_dim=get_tuned_param('Transformer', 'embedding_dim', 16),
          dropout=get_tuned_param('Transformer', 'dropout_rate', 0.25),
          learning_rate=get_tuned_param('Transformer', 'learning_rate', 0.001),
          verbose_model=verbose
      )
      history_tf = model_tf.fit(train_features_scaled, y_train,
                                validation_data=(val_features_scaled, y_val),
                                epochs=epochs,
                                batch_size=tf_batch_size,
                                callbacks=[early_stopping],
                                class_weight=keras_class_weight,
                                verbose=1 if verbose else 0)

      pred_train_proba_tf = model_tf.predict(train_features_scaled).ravel()
      pred_test_proba_tf = model_tf.predict(test_features_scaled).ravel()

      roc_auc_train_tf = roc_auc_score(y_train, pred_train_proba_tf)
      roc_auc_test_tf = roc_auc_score(y_test, pred_test_proba_tf)
      ks_train_tf = calculate_ks(y_train, pred_train_proba_tf)
      ks_test_tf = calculate_ks(y_test, pred_test_proba_tf)
      logloss_train_tf = log_loss(y_train, pred_train_proba_tf)
      logloss_test_tf = log_loss(y_test, pred_test_proba_tf)
      brier_train_tf = brier_score_loss(y_train, pred_train_proba_tf)
      brier_test_tf = brier_score_loss(y_test, pred_test_proba_tf)
      pr_auc_train_tf = average_precision_score(y_train, pred_train_proba_tf)
      pr_auc_test_tf = average_precision_score(y_test, pred_test_proba_tf)
      lift_train_tf = calculate_top_decile_lift(y_train, pred_train_proba_tf)
      lift_test_tf = calculate_top_decile_lift(y_test, pred_test_proba_tf)
      recall_at_fpr_train_tf = calculate_recall_at_fpr(y_train, pred_train_proba_tf, target_fpr=0.1)
      recall_at_fpr_test_tf = calculate_recall_at_fpr(y_test, pred_test_proba_tf, target_fpr=0.1)

      if verbose:
        print(f'[Transformer] Train AUC: {roc_auc_train_tf:.5f} | Test AUC: {roc_auc_test_tf:.5f} | Test PR-AUC: {pr_auc_test_tf:.5f} | Test Lift@10%: {lift_test_tf:.3f}')

      results['Transformer'] = {'train_auc': roc_auc_train_tf, 'test_auc': roc_auc_test_tf,
                                'train_ks': ks_train_tf, 'test_ks': ks_test_tf,
                                'train_logloss': logloss_train_tf, 'test_logloss': logloss_test_tf,
                                'train_brier': brier_train_tf, 'test_brier': brier_test_tf,
                                'train_pr_auc': pr_auc_train_tf, 'test_pr_auc': pr_auc_test_tf,
                                'train_lift': lift_train_tf, 'test_lift': lift_test_tf,
                                'train_recall_at_fpr10': recall_at_fpr_train_tf, 'test_recall_at_fpr10': recall_at_fpr_test_tf}
  except Exception as e:
      if verbose: print(f"Error training/evaluating Transformer: {e}")
      results['Transformer'] = {'train_auc': np.nan, 'test_auc': np.nan, 'train_ks': np.nan, 'test_ks': np.nan,
                                'train_logloss': np.nan, 'test_logloss': np.nan, 'train_brier': np.nan, 'test_brier': np.nan,
                                'train_pr_auc': np.nan, 'test_pr_auc': np.nan, 'train_lift': np.nan, 'test_lift': np.nan,
                                'train_recall_at_fpr10': np.nan, 'test_recall_at_fpr10': np.nan}


  ############################################################
  # 17) 결과 요약 (Renumbered)
  ############################################################
  if verbose: print("\n--- Section 17: Model Performance Summary ---")
  results_df = pd.DataFrame.from_dict(results, orient='index')
  pd.set_option('display.max_columns', None)
  pd.set_option('display.width', 1000)
  if verbose: print(results_df.sort_values(by='test_auc', ascending=False))


  ############################################################
  # 18) CatBoost Feature Importance (Renumbered)
  ############################################################
  if verbose:
    print("\n--- Section 18: CatBoost Feature Importance ---")
    try:
        # all_features를 그대로 사용 (이미 정의됨)
        importances = cat_model.feature_importances_
        importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

        print(f"\nTop 20 CatBoost Feature Importances ({feature_set}):")
        print(importance_df.head(20))

        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20], color='skyblue')
        plt.xlabel("CatBoost Feature Importance")
        plt.ylabel("Feature")
        plt.title(f"Top 20 CatBoost Feature Importances ({feature_set})")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error displaying CatBoost feature importances: {e}")

  if verbose: print("\nScript finished.")
  return results


############################################################
# 10-Fold 전체 실행 및 결과 저장
############################################################
def run_all_folds(feature_set='base', use_tuned_params=False, verbose=True, save_results=True):
    """
    Run experiments for all 10 folds and aggregate results.

    Args:
        feature_set: Feature set to use
        use_tuned_params: Whether to use tuned hyperparameters
        verbose: Print progress
        save_results: Save results to JSON file

    Returns:
        Dictionary with fold-level results and aggregated statistics
    """
    import json
    import os
    import numpy as np
    from config import RESULTS_PATH

    # Fold별 테스트 기간 및 시장 상황 매핑
    fold_info = {
        0: {'test_period': '2019.10~12', 'market': 'Pre-COVID'},
        1: {'test_period': '2020.4~6', 'market': 'COVID Crash'},
        2: {'test_period': '2020.10~12', 'market': 'COVID Recovery'},
        3: {'test_period': '2021.4~6', 'market': 'Bull Market'},
        4: {'test_period': '2021.10~12', 'market': 'Peak'},
        5: {'test_period': '2022.4~6', 'market': 'Rate Hike Drop'},
        6: {'test_period': '2022.10~12', 'market': 'Bear Market'},
        7: {'test_period': '2023.4~6', 'market': 'Recovery'},
        8: {'test_period': '2023.10~12', 'market': 'Stabilization'},
        9: {'test_period': '2024.4~6', 'market': 'Recent'},
    }

    all_results = {
        'feature_set': feature_set,
        'use_tuned_params': use_tuned_params,
        'fold_results': {},
        'summary': {}
    }

    # 모델별 지표 수집용
    model_metrics = {}

    print(f"\n{'='*70}")
    print(f"  Running 10-Fold Cross Validation")
    print(f"  Feature Set: {feature_set}, Tuned Params: {use_tuned_params}")
    print(f"{'='*70}\n")

    for fold_idx in range(10):
        print(f"\n{'='*50}")
        print(f"  Fold {fold_idx}: {fold_info[fold_idx]['test_period']} ({fold_info[fold_idx]['market']})")
        print(f"{'='*50}")

        try:
            results = run_fold_experiment(
                fold_index=fold_idx,
                feature_set=feature_set,
                verbose=verbose,
                use_tuned_params=use_tuned_params
            )

            # Fold 결과 저장
            fold_data = {
                'test_period': fold_info[fold_idx]['test_period'],
                'market_condition': fold_info[fold_idx]['market'],
                'models': {}
            }

            for model_name, model_result in results.items():
                # test_auc 추출
                test_auc = model_result.get('test_auc')
                if test_auc is not None and not (isinstance(test_auc, float) and test_auc != test_auc):
                    fold_data['models'][model_name] = {
                        'test_auc': float(test_auc),
                        'val_auc': float(model_result.get('val_auc', 0)),
                    }

                    # 추가 지표가 있으면 포함
                    for key in ['pr_auc', 'top_decile_lift', 'recall_at_fpr10', 'ks_stat', 'brier_score']:
                        if key in model_result and model_result[key] is not None:
                            fold_data['models'][model_name][key] = float(model_result[key])

                    # 모델별 지표 수집
                    if model_name not in model_metrics:
                        model_metrics[model_name] = {'test_auc': [], 'val_auc': []}
                    model_metrics[model_name]['test_auc'].append(float(test_auc))
                    if model_result.get('val_auc'):
                        model_metrics[model_name]['val_auc'].append(float(model_result['val_auc']))

            all_results['fold_results'][f'fold_{fold_idx}'] = fold_data

        except Exception as e:
            print(f"  [ERROR] Fold {fold_idx} failed: {e}")
            all_results['fold_results'][f'fold_{fold_idx}'] = {'error': str(e)}

    # 통계 계산 (평균, 표준편차)
    print(f"\n{'='*70}")
    print(f"  Summary Statistics (10-Fold CV)")
    print(f"{'='*70}\n")

    print(f"{'Model':<20} | {'Mean AUC':>10} | {'Std':>8} | {'Min':>8} | {'Max':>8}")
    print("-" * 60)

    for model_name, metrics in model_metrics.items():
        if len(metrics['test_auc']) > 0:
            aucs = np.array(metrics['test_auc'])
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            min_auc = np.min(aucs)
            max_auc = np.max(aucs)

            all_results['summary'][model_name] = {
                'mean_auc': float(mean_auc),
                'std_auc': float(std_auc),
                'min_auc': float(min_auc),
                'max_auc': float(max_auc),
                'n_folds': len(aucs),
                'fold_aucs': [float(x) for x in aucs]
            }

            print(f"{model_name:<20} | {mean_auc:>10.5f} | {std_auc:>8.5f} | {min_auc:>8.5f} | {max_auc:>8.5f}")

    # 결과 저장
    if save_results:
        tuned_suffix = '_tuned' if use_tuned_params else '_default'
        filename = f'cv_results_{feature_set}{tuned_suffix}.json'
        filepath = os.path.join(RESULTS_PATH, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n  Results saved to: {filepath}")

    return all_results


def print_fold_comparison(results_dict):
    """
    Print fold-by-fold comparison table for a specific model.

    Args:
        results_dict: Output from run_all_folds()
    """
    fold_results = results_dict.get('fold_results', {})

    # 모든 모델 이름 수집
    all_models = set()
    for fold_data in fold_results.values():
        if 'models' in fold_data:
            all_models.update(fold_data['models'].keys())

    for model_name in sorted(all_models):
        print(f"\n{'='*60}")
        print(f"  {model_name} - Fold-by-Fold Results")
        print(f"{'='*60}")
        print(f"{'Fold':<6} | {'Period':<12} | {'Market':<18} | {'Test AUC':>10}")
        print("-" * 55)

        for fold_idx in range(10):
            fold_key = f'fold_{fold_idx}'
            if fold_key in fold_results and 'models' in fold_results[fold_key]:
                fold_data = fold_results[fold_key]
                if model_name in fold_data['models']:
                    auc = fold_data['models'][model_name]['test_auc']
                    print(f"{fold_idx:<6} | {fold_data['test_period']:<12} | {fold_data['market_condition']:<18} | {auc:>10.5f}")


############################################################
# Main 실행
############################################################
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Stock Crash Prediction - 10-Fold CV')
    parser.add_argument('--feature_set', type=str, default='base',
                        choices=['base', 'base_no_vkospi', 'base_control',
                                 'base_control_no_vkospi', 'base_wavelet', 'base_wavelet_no_vkospi'],
                        help='Feature set to use')
    parser.add_argument('--tuned', action='store_true',
                        help='Use tuned hyperparameters')
    parser.add_argument('--fold', type=int, default=-1,
                        help='Specific fold to run (-1 for all folds)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    if args.fold >= 0:
        # 단일 fold 실행
        results = run_fold_experiment(
            fold_index=args.fold,
            feature_set=args.feature_set,
            verbose=not args.quiet,
            use_tuned_params=args.tuned
        )
    else:
        # 전체 10-fold 실행
        results = run_all_folds(
            feature_set=args.feature_set,
            use_tuned_params=args.tuned,
            verbose=not args.quiet,
            save_results=True
        )

        # Fold별 상세 출력
        print_fold_comparison(results)

