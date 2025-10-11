
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error, log_loss, brier_score_loss # MSE, LogLoss, Brier 추가
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from scipy import stats # For skew, kurtosis if added later
import pywt

# CatBoost
# !pip install catboost imblearn pywavelets # imblearn은 이제 필요 없음, pywavelets 설치 필요
from catboost import CatBoostClassifier
# from imblearn.over_sampling import RandomOverSampler # 삭제

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb


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

def calculate_wavelet_features(sequences, wavelet='db4', level=3, feature_stats=['energy', 'std', 'skew', 'kurt']):
    """Calculates statistical features from DWT coefficients for each sequence."""
    n_samples, _, n_features = sequences.shape
    feature_list = []

    # Define stats functions safely
    def safe_skew(a):
        try: return stats.skew(a)
        except ValueError: return 0
    def safe_kurtosis(a):
        try: return stats.kurtosis(a)
        except ValueError: return 0
    def safe_std(a):
        try: return np.std(a)
        except ValueError: return 0
    def safe_energy(a):
        return np.sum(a**2)
    def safe_entropy(a):
      try:
        sq_coeffs = a**2
        coeff_sum = np.sum(sq_coeffs)
        if coeff_sum == 0: return 0
        probs = sq_coeffs / coeff_sum
        return stats.entropy(probs)
      except ValueError: return 0

    stat_funcs = {'energy': safe_energy, 'std': safe_std, 'skew': safe_skew, 'kurt': safe_kurtosis, 'entropy': safe_entropy}
    valid_feature_stats = [s for s in feature_stats if s in stat_funcs] # Ensure only valid stats are used

    print(f"Calculating Wavelet features using: wavelet='{wavelet}', level={level}, stats={valid_feature_stats}")

    for i in range(n_samples):
        sample_features = []
        for j in range(n_features):
            sequence = sequences[i, :, j]
            sequence = np.nan_to_num(sequence) # Ensure no NaNs/Infs before DWT

            try:
                # Perform multilevel DWT
                coeffs = pywt.wavedec(sequence, wavelet, level=level) # List: [cA_n, cD_n, ..., cD_1]

                # Calculate stats for each coefficient array
                for coeff_array in coeffs:
                    if len(coeff_array) == 0: # Handle potential empty arrays
                        for stat_name in valid_feature_stats:
                            sample_features.append(0)
                        continue

                    for stat_name in valid_feature_stats:
                        func = stat_funcs[stat_name]
                        stat_value = func(coeff_array)
                        sample_features.append(stat_value)

            except Exception as e:
                print(f"Warning: Wavelet decomposition failed for sample {i}, feature {j}. Error: {e}. Appending zeros.")
                # Calculate expected number of features per stock feature
                num_coeff_arrays = level + 1
                num_expected_features = num_coeff_arrays * len(valid_feature_stats)
                sample_features.extend([0] * num_expected_features) # Append zeros if DWT fails

        feature_list.append(sample_features)

    return np.array(feature_list)


############################################################
# file load
############################################################
try:
    stock_df = pd.read_parquet('D://assist//00.SCIE paper 1//codes//stock_all_vkospi_sample.parquet')
    fin_df   = pd.read_parquet('D://assist//00.SCIE paper 1//codes//stock_fin_result_with_vkospi_all_sample.parquet')
    print("Successfully loaded data from Parquet files.")
except FileNotFoundError:
    print("Warning: Parquet files not found. Please ensure the path is correct or provide dummy data.")
    # Try loading from a local path as a fallback (adjust as needed)
    try:
        stock_df = pd.read_parquet('stock_all_vkospi.parquet')
        fin_df   = pd.read_parquet('stock_fin_result_with_vkospi_all.parquet')
        print("Loaded data from local Parquet files.")
    except FileNotFoundError:
        print("Error: Could not load data from Drive or local path. Exiting.")
        exit()
except Exception as e:
    print(f"An error occurred during file loading: {e}")
    exit()


############################################################
# date preprocessing
############################################################
stock_df['BSOP_DATE'] = pd.to_datetime(stock_df['BSOP_DATE'], errors='coerce')
fin_df['BSOP_DATE']   = pd.to_datetime(fin_df['BSOP_DATE'],   errors='coerce')

# 2024-07-01 before 
stock_df = stock_df[stock_df['BSOP_DATE'] < '2024-07-01'].copy()
fin_df   = fin_df[fin_df['BSOP_DATE']   < '2024-07-01'].copy()

############################################################
# feature engineering
############################################################
mask_fin = fin_df.groupby('SHRN_ISCD')['ROE'].transform('count') > 0
fin_df = fin_df[mask_fin].copy()
valid_codes = fin_df['SHRN_ISCD'].unique()
stock_df = stock_df[stock_df['SHRN_ISCD'].isin(valid_codes)].copy()
print("Stock DF Columns:", stock_df.columns.tolist())
fin_df = fin_df.dropna(subset=['windowdrop']).copy() # Ensure target is not NaN
fin_df['MONTH'] = fin_df['BSOP_DATE'].dt.month
# Calculate features safely, avoiding division by zero or near-zero
fin_df.loc[:, 'PRICE_TO_52W_HIGH'] = np.where(fin_df['W52_HGPR'] != 0, fin_df['STCK_PRPR'] / fin_df['W52_HGPR'], 0)
fin_df.loc[:, 'PRICE_TO_52W_LOW'] = np.where(fin_df['W52_LWPR'] != 0, fin_df['STCK_PRPR'] / fin_df['W52_LWPR'], 0)
fin_df.loc[:, 'PER'] = np.where(fin_df['MARKET_CAP'] != 0, fin_df['NET_INCOME'] / fin_df['MARKET_CAP'], 0)
fin_df.loc[:, 'PSR'] = np.where(fin_df['STCK_PRPR'] != 0, fin_df['PS'] / fin_df['STCK_PRPR'], 0)
fin_df.loc[:, 'FTRUTH'] = np.where(fin_df['NET_INCOME'] != 0, fin_df['PS'] / fin_df['NET_INCOME'], 0)
fin_df.loc[:, 'NET_DEBT_RATIO'] = np.where(fin_df['MARKET_CAP'] != 0, fin_df['NET_DEBT'] / fin_df['MARKET_CAP'], 0) # Renamed to avoid conflict
fin_df.loc[:, 'DEBT_COST_RATIO'] = np.where(fin_df['MARKET_CAP'] != 0, fin_df['DEBT_COST'] / fin_df['MARKET_CAP'], 0) # Renamed to avoid conflict

le_scrt = LabelEncoder()
le_mrkt = LabelEncoder()
fin_df['SCRT_GRP_CLS_CODE'] = fin_df['SCRT_GRP_CLS_CODE'].astype(str)
fin_df['MRKT_DIV_CLS_CODE'] = fin_df['MRKT_DIV_CLS_CODE'].astype(str)
fin_df['SCRT_GRP_CLS_CODE'] = le_scrt.fit_transform(fin_df['SCRT_GRP_CLS_CODE'])
fin_df['MRKT_DIV_CLS_CODE'] = le_mrkt.fit_transform(fin_df['MRKT_DIV_CLS_CODE'])

sequence_length = 60
stock_features = [
    'STCK_PRPR',
    #'vkospi', # If VKOSPI is only daily, consider adding it to fin_features instead
]
n_stock_features = len(stock_features)
fin_features = [
    'PRICE_TO_52W_HIGH', 'PRICE_TO_52W_LOW', 'SCRT_GRP_CLS_CODE',
    'MONTH', 'VKOSPI',
    'FN_ACML_TR_PBMN', 'ORGN_NTBY_QTY', 'DR', 'PSR', 'ROE', 'ROE_INC',
    'NET_DEBT_RATIO', 'DEBT_COST_RATIO', 'DEBT_DEPENDENCY', 'PER', 'ICR', 'FTRUTH', 'ASSET_GROWTH', 'TOTAL_ASSET_GROWTH', # Updated names
]
fin_features = [f for f in fin_features if f in fin_df.columns] # Ensure features exist
print("Using Fin Features:", fin_features)

############################################################
# create sequences
############################################################
stock_series_data   = []
stock_current_data = []
fin_series_data     = []
matched_targets     = []
dates               = []
stock_groups = stock_df.groupby('SHRN_ISCD')

# Handle potential division by zero or inf/NaN before processing
numeric_cols_fin = fin_df.select_dtypes(include=np.number).columns
fin_df[numeric_cols_fin] = fin_df[numeric_cols_fin].replace([np.inf, -np.inf], np.nan)
numeric_cols_stock = stock_df.select_dtypes(include=np.number).columns
stock_df[numeric_cols_stock] = stock_df[numeric_cols_stock].replace([np.inf, -np.inf], np.nan)

# Fill NaNs - consider more sophisticated methods if appropriate (e.g., ffill, mean/median)
fin_df.fillna(0, inplace=True)
stock_df.fillna(0, inplace=True)

print("Starting sequence generation...")
processed_count = 0
skipped_short_seq = 0
skipped_no_today = 0
for stock_code, fin_group in fin_df.groupby('SHRN_ISCD'):
    if stock_code not in stock_groups.groups: continue
    stock_history_group = stock_groups.get_group(stock_code).sort_values('BSOP_DATE')
    if len(stock_history_group) < sequence_length: continue

    for row in fin_group.itertuples():
        date = row.BSOP_DATE
        # Find the index *before* or *at* the current fin_date in the stock history
        end_index = stock_history_group['BSOP_DATE'].searchsorted(date, side='right') # Use 'right' to include the day itself if present
        start_index = max(0, end_index - sequence_length)
        stock_history = stock_history_group.iloc[start_index:end_index]

        if len(stock_history) != sequence_length:
            # Pad if necessary and feasible, or skip
            # For simplicity, we skip here
            skipped_short_seq += 1
            continue

        # Get 'today's' stock data (corresponding to the fin_date)
        # It should be the last row of the extracted sequence
        today_data = stock_history[stock_features].iloc[-1].values

        # Check if today_data contains NaNs (might happen if last day had missing stock data)
        if np.isnan(today_data).any():
            skipped_no_today +=1
            continue

        stock_series_data.append(stock_history[stock_features].values)
        # stock_current_data.append(today_data) # This might not be needed if not used later
        fin_series_data.append([getattr(row, f) for f in fin_features])
        matched_targets.append(row.windowdrop)
        dates.append(date)
        processed_count += 1

print(f"Finished sequence generation. Processed: {processed_count}, Skipped (short seq): {skipped_short_seq}, Skipped (no today data): {skipped_no_today}")
if not stock_series_data: raise ValueError("No sequences generated.")

stock_series_data   = np.array(stock_series_data)
# stock_current_data = np.array(stock_current_data)
fin_series_data     = np.array(fin_series_data)
matched_targets     = np.array(matched_targets)
dates               = np.array(dates)

print("stock_series_data.shape :", stock_series_data.shape)
# print("stock_current_data.shape:", stock_current_data.shape)
print("fin_series_data.shape   :", fin_series_data.shape)
print("matched_targets.shape :", matched_targets.shape)
print("dates.shape           :", dates.shape)

############################################################
# 6) Train/Test split 
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

folds = [
    # Fold 0
    {'train_start': pd.to_datetime('2015-01-01'), 'train_end': pd.to_datetime('2020-12-31'),
     'val_start'  : pd.to_datetime('2021-01-01'), 'val_end'  : pd.to_datetime('2021-12-31'),
     'test_start' : pd.to_datetime('2022-01-01'), 'test_end' : pd.to_datetime('2022-03-31')},
    # Fold 1
    {'train_start': pd.to_datetime('2015-04-01'), 'train_end': pd.to_datetime('2021-03-31'),
     'val_start'  : pd.to_datetime('2021-04-01'), 'val_end'  : pd.to_datetime('2022-03-31'),
     'test_start' : pd.to_datetime('2022-04-01'), 'test_end' : pd.to_datetime('2022-06-30')},
    # Fold 2
    {'train_start': pd.to_datetime('2015-07-01'), 'train_end': pd.to_datetime('2021-06-30'),
     'val_start'  : pd.to_datetime('2021-07-01'), 'val_end'  : pd.to_datetime('2022-06-30'),
     'test_start' : pd.to_datetime('2022-07-01'), 'test_end' : pd.to_datetime('2022-09-30')},
    # Fold 3
    {'train_start': pd.to_datetime('2015-10-01'), 'train_end': pd.to_datetime('2021-09-30'),
     'val_start'  : pd.to_datetime('2021-10-01'), 'val_end'  : pd.to_datetime('2022-09-30'),
     'test_start' : pd.to_datetime('2022-10-01'), 'test_end' : pd.to_datetime('2022-12-31')},
    # Fold 4
    {'train_start': pd.to_datetime('2016-01-01'), 'train_end': pd.to_datetime('2021-12-31'),
     'val_start'  : pd.to_datetime('2022-01-01'), 'val_end'  : pd.to_datetime('2022-12-31'),
     'test_start' : pd.to_datetime('2023-01-01'), 'test_end' : pd.to_datetime('2023-03-31')},
    # Fold 5
    {'train_start': pd.to_datetime('2016-04-01'), 'train_end': pd.to_datetime('2022-03-31'),
     'val_start'  : pd.to_datetime('2022-04-01'), 'val_end'  : pd.to_datetime('2023-03-31'),
     'test_start' : pd.to_datetime('2023-04-01'), 'test_end' : pd.to_datetime('2023-06-30')},
    # Fold 6
    {'train_start': pd.to_datetime('2016-07-01'), 'train_end': pd.to_datetime('2022-06-30'),
     'val_start'  : pd.to_datetime('2022-07-01'), 'val_end'  : pd.to_datetime('2023-06-30'),
     'test_start' : pd.to_datetime('2023-07-01'), 'test_end' : pd.to_datetime('2023-09-30')},
    # Fold 7
    {'train_start': pd.to_datetime('2016-10-01'), 'train_end': pd.to_datetime('2022-09-30'),
     'val_start'  : pd.to_datetime('2022-10-01'), 'val_end'  : pd.to_datetime('2023-09-30'),
     'test_start' : pd.to_datetime('2023-10-01'), 'test_end' : pd.to_datetime('2023-12-31')},
    # Fold 8
    {'train_start': pd.to_datetime('2017-01-01'), 'train_end': pd.to_datetime('2022-12-31'),
     'val_start'  : pd.to_datetime('2023-01-01'), 'val_end'  : pd.to_datetime('2023-12-31'),
     'test_start' : pd.to_datetime('2024-01-01'), 'test_end' : pd.to_datetime('2024-03-31')},
    # Fold 9
    {'train_start': pd.to_datetime('2017-04-01'), 'train_end': pd.to_datetime('2023-03-31'),
     'val_start'  : pd.to_datetime('2023-04-01'), 'val_end'  : pd.to_datetime('2024-03-31'),
     'test_start' : pd.to_datetime('2024-04-01'), 'test_end' : pd.to_datetime('2024-06-30')},
]

# --- set fold ---
fold_index = 9  
selected_fold = folds[fold_index]

train_mask = (dates >= selected_fold['train_start']) & (dates <= selected_fold['train_end'])
val_mask   = (dates >= selected_fold['val_start'])   & (dates <= selected_fold['val_end'])
test_mask  = (dates >= selected_fold['test_start'])  & (dates <= selected_fold['test_end'])

if not np.any(train_mask): raise ValueError("No training data after split.")
if not np.any(val_mask): raise ValueError("No val data after split.")
if not np.any(test_mask): raise ValueError("No test data after split.")

X_stock_train_seq   = stock_series_data[train_mask]      # Input for scaling & stats feature extraction
X_stock_val_seq    = stock_series_data[val_mask]
X_stock_test_seq    = stock_series_data[test_mask]       # Input for scaling & stats feature extraction

# X_stock_train_curr = stock_current_data[train_mask] # Not used currently
# X_stock_test_curr  = stock_current_data[test_mask]  # Not used currently
X_fin_train = fin_series_data[train_mask]
X_fin_val   = fin_series_data[val_mask]
X_fin_test  = fin_series_data[test_mask]

y_train = matched_targets[train_mask]
y_val   = matched_targets[val_mask]
y_test  = matched_targets[test_mask]

print("Train sequence shape (before scaling):", X_stock_train_seq.shape)
print("Val sequence shape (before scaling):", X_stock_val_seq.shape)
print("Test sequence shape (before scaling):", X_stock_test_seq.shape)
print("Train fin shape:", X_fin_train.shape)
print("Val fin shape:", X_fin_val.shape)
print("Test fin shape:", X_fin_test.shape)
print("Train target size:", len(y_train))
print("Val target size:", len(y_val))
print("Test target size :", len(y_test))
print("Train target distribution (Class 1):", np.mean(y_train))
print("val target distribution (Class 1):", np.mean(y_val))
print("Test target distribution  (Class 1):", np.mean(y_test))

n_train_samples, seq_len, n_features = X_stock_train_seq.shape
n_val_samples = X_stock_val_seq.shape[0]
n_test_samples = X_stock_test_seq.shape[0]

if n_features != n_stock_features: raise ValueError(f"Feature count mismatch: X_stock_train_seq has {n_features} features, expected {n_stock_features}")

# Reshape for scaling (scale across all samples and time steps for each feature)
temp_train = X_stock_train_seq.reshape(-1, n_stock_features)
temp_val   = X_stock_val_seq.reshape(-1, n_stock_features)
temp_test  = X_stock_test_seq.reshape(-1, n_stock_features)

# Apply log1p transformation (handle potential zeros)
temp_train_log = np.log1p(np.maximum(temp_train, 0)) # Ensure non-negative before log
temp_val_log = np.log1p(np.maximum(temp_val, 0)) # Ensure non-negative before log
temp_test_log  = np.log1p(np.maximum(temp_test, 0))

# Handle any resulting NaNs/Infs after log (though maximum should prevent log(0))
temp_train_log = np.nan_to_num(temp_train_log, nan=0.0, posinf=0.0, neginf=0.0)
temp_val_log = np.nan_to_num(temp_val_log, nan=0.0, posinf=0.0, neginf=0.0)
temp_test_log = np.nan_to_num(temp_test_log, nan=0.0, posinf=0.0, neginf=0.0)

stock_scaler_seq = StandardScaler()
temp_train_scaled = stock_scaler_seq.fit_transform(temp_train_log)
temp_val_scaled = stock_scaler_seq.fit_transform(temp_val_log)
temp_test_scaled  = stock_scaler_seq.transform(temp_test_log)

# Handle potential NaNs from StandardScaler (if a feature is constant in train)
if np.isnan(temp_train_scaled).any() or np.isinf(temp_train_scaled).any():
    print("Warning: NaNs/Infs detected after scaling train data. Check for constant features. Filling with 0.")
    temp_train_scaled = np.nan_to_num(temp_train_scaled, nan=0.0, posinf=0.0, neginf=0.0)
if np.isnan(temp_val_scaled).any() or np.isinf(temp_val_scaled).any():
    print("Warning: NaNs/Infs detected after scaling test data. Filling with 0.")
    temp_val_scaled = np.nan_to_num(temp_val_scaled, nan=0.0, posinf=0.0, neginf=0.0)
if np.isnan(temp_test_scaled).any() or np.isinf(temp_test_scaled).any():
    print("Warning: NaNs/Infs detected after scaling test data. Filling with 0.")
    temp_test_scaled = np.nan_to_num(temp_test_scaled, nan=0.0, posinf=0.0, neginf=0.0)

# Reshape back to (n_samples, seq_len, n_features)
X_stock_train_seq_scaled = temp_train_scaled.reshape(n_train_samples, seq_len, n_features)
X_stock_val_seq_scaled = temp_val_scaled.reshape(n_val_samples, seq_len, n_features)
X_stock_test_seq_scaled  = temp_test_scaled.reshape(n_test_samples, seq_len, n_features)

print("Scaled train sequence shape:", X_stock_train_seq_scaled.shape)
print("Scaled test sequence shape:", X_stock_test_seq_scaled.shape)

#seed fixing
import random
import os
seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
print(f"Set random seeds to {seed_value}")


############################################################
# statistics features (wavelet + std) calculation 

print("Calculating wavelet features (energy, std, skew, kurt) using pywt.wavedec...")
#train_wavelet_features = calculate_wavelet_features(X_stock_train_seq_scaled, wavelet='db4', level=3, feature_stats=['energy', 'std', 'skew', 'kurt'])
train_wavelet_features = calculate_wavelet_features(X_stock_train_seq_scaled, wavelet='db4', level=3, feature_stats=['energy', 'std'])
#val_wavelet_features = calculate_wavelet_features(X_stock_val_seq_scaled, wavelet='db4', level=3, feature_stats=['energy', 'std', 'skew', 'kurt'])
val_wavelet_features = calculate_wavelet_features(X_stock_val_seq_scaled, wavelet='db4', level=3, feature_stats=['energy', 'std'])
#test_wavelet_features = calculate_wavelet_features(X_stock_test_seq_scaled, wavelet='db4', level=3, feature_stats=['energy', 'std', 'skew', 'kurt'])
test_wavelet_features = calculate_wavelet_features(X_stock_test_seq_scaled, wavelet='db4', level=3, feature_stats=['energy', 'std'])
train_wavelet_features = np.nan_to_num(train_wavelet_features, nan=0.0, posinf=0.0, neginf=0.0)
val_wavelet_features = np.nan_to_num(val_wavelet_features, nan=0.0, posinf=0.0, neginf=0.0)
test_wavelet_features = np.nan_to_num(test_wavelet_features, nan=0.0, posinf=0.0, neginf=0.0)
print(f"Finished calculating wavelet features. Train Shape: {train_wavelet_features.shape}, Val Shape: {val_wavelet_features.shape}, Test Shape: {test_wavelet_features.shape}")

print("Calculating standard deviations...")
train_stds = np.std(X_stock_train_seq_scaled, axis=1)
val_stds = np.std(X_stock_val_seq_scaled, axis=1)
test_stds = np.std(X_stock_test_seq_scaled, axis=1)
# NaN/Inf prevention
train_stds = np.nan_to_num(train_stds, nan=0.0, posinf=0.0, neginf=0.0)
val_stds = np.nan_to_num(val_stds, nan=0.0, posinf=0.0, neginf=0.0)
test_stds = np.nan_to_num(test_stds, nan=0.0, posinf=0.0, neginf=0.0)
print("Finished calculating standard deviations.")

train_stat_features = np.concatenate([train_wavelet_features, train_stds], axis=1)
val_stat_features = np.concatenate([val_wavelet_features, val_stds], axis=1)
test_stat_features = np.concatenate([test_wavelet_features, test_stds], axis=1)

print("train_stat_features (wavelet+std).shape:", train_stat_features.shape)
print("test_stat_features (wavelet+std).shape :", test_stat_features.shape)


print("\n--- Preparing Final Features for Models ---")
include_stat_features = True 
print(f"Include Statistical Features (Wavelet Features + Std) in Final Features: {include_stat_features}")

# -- Train Data --
feature_list_train = []
if include_stat_features:
    feature_list_train.append(train_stat_features)
if X_fin_train.size > 0: # Check if financial features exist
    feature_list_train.append(X_fin_train)

if not feature_list_train: raise ValueError("No features selected for model training!")
train_features = np.concatenate(feature_list_train, axis=1)

# -- Val Data --
feature_list_val = []
if include_stat_features:
    feature_list_val.append(val_stat_features)
if X_fin_val.size > 0: # Check if financial features exist
    feature_list_val.append(X_fin_val)
# feature_list_test.append(X_stock_test_curr) 

if not feature_list_val: raise ValueError("No features selected for model testing!")
val_features = np.concatenate(feature_list_val, axis=1)

# -- Test Data --
feature_list_test = []
if include_stat_features:
    feature_list_test.append(test_stat_features)
if X_fin_test.size > 0: # Check if financial features exist
    feature_list_test.append(X_fin_test)

if not feature_list_test: raise ValueError("No features selected for model testing!")
test_features = np.concatenate(feature_list_test, axis=1)

print("Combined train_features shape (before scaling):", train_features.shape)
print("Combined val_features shape (before scaling):"  , val_features.shape)
print("Combined test_features shape  (before scaling):", test_features.shape)

# Scale Combined Features <<<
print("\n--- Scaling Combined Features for All Models ---")
feature_scaler = StandardScaler()
train_features_scaled = feature_scaler.fit_transform(train_features)
val_features_scaled = feature_scaler.transform(val_features)
test_features_scaled = feature_scaler.transform(test_features)

# Handle potential NaNs/Infs after scaling combined features
train_features_scaled = np.nan_to_num(train_features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
val_features_scaled = np.nan_to_num(val_features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
test_features_scaled = np.nan_to_num(test_features_scaled, nan=0.0, posinf=0.0, neginf=0.0)

print("Scaled combined train features shape:", train_features_scaled.shape)
print("Scaled combined val features shape:", val_features_scaled.shape)
print("Scaled combined test features shape:", test_features_scaled.shape)
# >>> MODIFICATION END <<<

# --- class imbalance handling ---
n_negatives = np.sum(y_train == 0)
n_positives = np.sum(y_train == 1)
if n_positives == 0 or n_negatives == 0:
    print("Warning: One class is missing in the training data. Setting scale_pos_weight=1.0")
    scale_pos_weight_value = 1.0
else:
    # Ensure n_positives is not zero before division
    scale_pos_weight_value = n_negatives / n_positives if n_positives > 0 else 1.0
print(f"Calculated scale_pos_weight: {scale_pos_weight_value:.4f}")


print("\n--- Training and Evaluating CatBoost ---")

cat_model = CatBoostClassifier(
        iterations=5000,
        depth=7,          
        learning_rate=0.01, 
        random_seed=seed_value,
        verbose=100,
        loss_function='Logloss',
        early_stopping_rounds=1000,
        l2_leaf_reg=100,     
        random_strength=1,   
        bootstrap_type='Bayesian',
        bagging_temperature=10,
        # bootstrap_type='Bernoulli',
        # subsample=0.8,
        eval_metric='AUC',
        use_best_model=True,
        scale_pos_weight=scale_pos_weight_value
)

print("--- Starting CatBoost Training ---")
eval_set = (val_features_scaled, y_val)
cat_model.fit(train_features_scaled, y_train, eval_set=eval_set, verbose=100)

# final evaluation
pred_train_proba_cat = cat_model.predict_proba(train_features_scaled)[:, 1]
pred_test_proba_cat  = cat_model.predict_proba(test_features_scaled)[:, 1]

roc_auc_train_cat = roc_auc_score(y_train, pred_train_proba_cat)
roc_auc_test_cat  = roc_auc_score(y_test, pred_test_proba_cat)
ks_train_cat = calculate_ks(y_train, pred_train_proba_cat) # Changed variable name for clarity
ks_test_cat = calculate_ks(y_test, pred_test_proba_cat)   # Changed variable name for clarity

logloss_train_cat = log_loss(y_train, pred_train_proba_cat)
logloss_test_cat = log_loss(y_test, pred_test_proba_cat)
brier_train_cat = brier_score_loss(y_train, pred_train_proba_cat)
brier_test_cat = brier_score_loss(y_test, pred_test_proba_cat)

results = {} # Initialize results dictionary here
results['CatBoost'] = {'train_auc': roc_auc_train_cat, 'test_auc': roc_auc_test_cat,
                       'train_ks': ks_train_cat, 'test_ks': ks_test_cat,
                       'train_logloss': logloss_train_cat, 'test_logloss': logloss_test_cat,
                       'train_brier': brier_train_cat, 'test_brier': brier_test_cat}


feature_type = "Wavelet Features + Std + Fin" if include_stat_features else "Financial Features Only"
print(f'\n[CatBoost with {feature_type}] Train AUC: {roc_auc_train_cat:.5f} | Train KS: {ks_train_cat:.5f} | Train LogLoss: {logloss_train_cat:.5f} | Train Brier: {brier_train_cat:.5f}')
print(f'[CatBoost with {feature_type}] Test  AUC: {roc_auc_test_cat:.5f}  | Test KS: {ks_test_cat:.5f}  | Test LogLoss: {logloss_test_cat:.5f}  | Test Brier: {brier_test_cat:.5f}')

fpr_train_cat, tpr_train_cat, _ = roc_curve(y_train, pred_train_proba_cat)
fpr_test_cat,  tpr_test_cat,  _ = roc_curve(y_test, pred_test_proba_cat)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr_train_cat, tpr_train_cat, label=f'Train AUC: {roc_auc_train_cat:.5f}')
plt.plot([0,1],[0,1],'k--')
plt.title(f'ROC Curve - Train (CatBoost + {feature_type})')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.legend(); plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(fpr_test_cat, tpr_test_cat, label=f'Test AUC: {roc_auc_test_cat:.5f}')
plt.plot([0,1],[0,1],'k--')
plt.title(f'ROC Curve - Test (CatBoost + {feature_type})')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.show()

plt.figure(figsize=(10, 4))
plt.hist(pred_train_proba_cat, bins=50, alpha=0.7)
plt.title(f'Train Prediction Probabilities (CatBoost + {feature_type})')
plt.xlabel('Predicted Probability (Class 1)'); plt.ylabel('Frequency'); plt.grid(True, axis='y'); plt.show()
plt.figure(figsize=(10, 4))
plt.hist(pred_test_proba_cat, bins=50, alpha=0.7, color='orange')
plt.title(f'Test Prediction Probabilities (CatBoost + {feature_type})')
plt.xlabel('Predicted Probability (Class 1)'); plt.ylabel('Frequency'); plt.grid(True, axis='y'); plt.show()


############################################################
# (LR, RF, XGB, LGBM)
############################################################
print("\n Training and Evaluating Models ---")

# --- Logistic Regression (LR) ---
print("\n--- Training and Evaluating Logistic Regression ---")
try:
    lr_model = LogisticRegression(random_state=seed_value, class_weight='balanced', max_iter=1000, solver='liblinear') # Use balanced weights and specify solver
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

    print(f'[LR] Train AUC: {roc_auc_train_lr:.5f} | Train KS: {ks_train_lr:.5f} | Train LogLoss: {logloss_train_lr:.5f} | Train Brier: {brier_train_lr:.5f}')
    print(f'[LR] Test  AUC: {roc_auc_test_lr:.5f}  | Test KS: {ks_test_lr:.5f}  | Test LogLoss: {logloss_test_lr:.5f}  | Test Brier: {brier_test_lr:.5f}')

    results['LR'] = {'train_auc': roc_auc_train_lr, 'test_auc': roc_auc_test_lr,
                     'train_ks': ks_train_lr, 'test_ks': ks_test_lr,
                     'train_logloss': logloss_train_lr, 'test_logloss': logloss_test_lr,
                     'train_brier': brier_train_lr, 'test_brier': brier_test_lr}
except Exception as e:
    print(f"Error training/evaluating LR: {e}")
    results['LR'] = {'train_auc': np.nan, 'test_auc': np.nan, 'train_ks': np.nan, 'test_ks': np.nan,
                     'train_logloss': np.nan, 'test_logloss': np.nan, 'train_brier': np.nan, 'test_brier': np.nan}


# --- Random Forest (RF) ---
print("\n--- Training and Evaluating Random Forest ---")
try:
    custom_class_weight_for_rf = {0: 1.0, 1: scale_pos_weight_value}
    rf_model = RandomForestClassifier(random_state=seed_value,
                                      class_weight=custom_class_weight_for_rf, # <--- 여기에 적용
                                      n_estimators=200,
                                      n_jobs=-1,
                                      max_depth=7,
                                      min_samples_leaf=5) # Add regularization params
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

    print(f'[RF] Train AUC: {roc_auc_train_rf:.5f} | Train KS: {ks_train_rf:.5f} | Train LogLoss: {logloss_train_rf:.5f} | Train Brier: {brier_train_rf:.5f}')
    print(f'[RF] Test  AUC: {roc_auc_test_rf:.5f}  | Test KS: {ks_test_rf:.5f}  | Test LogLoss: {logloss_test_rf:.5f}  | Test Brier: {brier_test_rf:.5f}')

    results['RF'] = {'train_auc': roc_auc_train_rf, 'test_auc': roc_auc_test_rf,
                     'train_ks': ks_train_rf, 'test_ks': ks_test_rf,
                     'train_logloss': logloss_train_rf, 'test_logloss': logloss_test_rf,
                     'train_brier': brier_train_rf, 'test_brier': brier_test_rf}
except Exception as e:
    print(f"Error training/evaluating RF: {e}")
    results['RF'] = {'train_auc': np.nan, 'test_auc': np.nan, 'train_ks': np.nan, 'test_ks': np.nan,
                     'train_logloss': np.nan, 'test_logloss': np.nan, 'train_brier': np.nan, 'test_brier': np.nan}


# --- XGBoost (XGB) ---
print("\n--- Training and Evaluating XGBoost ---")
try:
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        use_label_encoder=False, # Recommended
        random_state=seed_value,
        n_estimators=1000,       # Allow early stopping
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight_value, # Handle imbalance
        early_stopping_rounds=50 # Use early stopping
    )
    eval_set_xgb = [(val_features_scaled, y_val)] # Use scaled features
    xgb_model.fit(train_features_scaled, y_train, eval_set=eval_set_xgb, verbose=100) # Pass eval_set

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

    print(f'[XGB] Train AUC: {roc_auc_train_xgb:.5f} | Train KS: {ks_train_xgb:.5f} | Train LogLoss: {logloss_train_xgb:.5f} | Train Brier: {brier_train_xgb:.5f}')
    print(f'[XGB] Test  AUC: {roc_auc_test_xgb:.5f}  | Test KS: {ks_test_xgb:.5f}  | Test LogLoss: {logloss_test_xgb:.5f}  | Test Brier: {brier_test_xgb:.5f}')

    results['XGB'] = {'train_auc': roc_auc_train_xgb, 'test_auc': roc_auc_test_xgb,
                      'train_ks': ks_train_xgb, 'test_ks': ks_test_xgb,
                      'train_logloss': logloss_train_xgb, 'test_logloss': logloss_test_xgb,
                      'train_brier': brier_train_xgb, 'test_brier': brier_test_xgb}
except Exception as e:
    print(f"Error training/evaluating XGB: {e}")
    results['XGB'] = {'train_auc': np.nan, 'test_auc': np.nan, 'train_ks': np.nan, 'test_ks': np.nan,
                      'train_logloss': np.nan, 'test_logloss': np.nan, 'train_brier': np.nan, 'test_brier': np.nan}

# --- LightGBM (LGBM) ---
print("\n--- Training and Evaluating LightGBM ---")
try:
    lgbm_model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        random_state=seed_value,
        n_estimators=1000,       # Allow early stopping
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight_value, # Handle imbalance
        n_jobs=-1
    )
    eval_set_lgbm = [(val_features_scaled, y_val)] # Use scaled features
    # Need callbacks for early stopping in LGBM
    callbacks_lgbm = [lgb.early_stopping(stopping_rounds=50, verbose=100)]
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

    print(f'[LGBM] Train AUC: {roc_auc_train_lgbm:.5f} | Train KS: {ks_train_lgbm:.5f} | Train LogLoss: {logloss_train_lgbm:.5f} | Train Brier: {brier_train_lgbm:.5f}')
    print(f'[LGBM] Test  AUC: {roc_auc_test_lgbm:.5f}  | Test KS: {ks_test_lgbm:.5f}  | Test LogLoss: {logloss_test_lgbm:.5f}  | Test Brier: {brier_test_lgbm:.5f}')

    results['LGBM'] = {'train_auc': roc_auc_train_lgbm, 'test_auc': roc_auc_test_lgbm,
                       'train_ks': ks_train_lgbm, 'test_ks': ks_test_lgbm,
                       'train_logloss': logloss_train_lgbm, 'test_logloss': logloss_test_lgbm,
                       'train_brier': brier_train_lgbm, 'test_brier': brier_test_lgbm}
except Exception as e:
    print(f"Error training/evaluating LGBM: {e}")
    results['LGBM'] = {'train_auc': np.nan, 'test_auc': np.nan, 'train_ks': np.nan, 'test_ks': np.nan,
                       'train_logloss': np.nan, 'test_logloss': np.nan, 'train_brier': np.nan, 'test_brier': np.nan}

# --- Feature Importance Visualization for LightGBM ---
import seaborn as sns # <--- Add this import!
print("\n--- Visualizing LightGBM Feature Importance ---")

try:
    # Get feature importances
    importances = lgbm_model.feature_importances_

    # Get feature names from your training data (assuming train_features_scaled is a pandas DataFrame or has .columns)
    # If train_features_scaled is a numpy array, you'll need to know the original feature names.
    # For demonstration, let's assume original_feature_names is a list of your column names.
    # Replace `train_features_scaled.columns` with your actual feature names if it's a numpy array.
    # Example: original_feature_names = ['VKOSPI', 'MONTH', 'YEAR', ...]

    # Assuming `train_features_scaled` is a pandas DataFrame for column names
    # If it's a numpy array, you'll need to define feature_names manually or get them from elsewhere.
    if hasattr(train_features_scaled, 'columns'):
        feature_names = train_features_scaled.columns
    else:
        # Fallback: If train_features_scaled is a numpy array, define dummy feature names
        # YOU MUST REPLACE THESE WITH YOUR ACTUAL FEATURE NAMES IF IT'S A NUMPY ARRAY
        feature_names = [f'Feature_{i}' for i in range(len(importances))]
        print("Warning: `train_features_scaled` does not have column names. Using generic feature names.")
        print("Please ensure `feature_names` list is correctly defined with your actual feature names for accurate plotting.")


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

    # Customizing tick parameters for better visibility
    plt.tick_params(axis='x', colors='dimgray')
    plt.tick_params(axis='y', colors='dimgray')

    # Set x-axis limits to match the example image if needed, or let matplotlib decide
    # plt.xlim(0, 22)

    # Improve layout and display plot
    plt.tight_layout()
    plt.show()

    # Optional: Save the plot to a file
    # plt.savefig('lgbm_feature_importance.png', dpi=300, bbox_inches='tight')

except Exception as e:
    print(f"Error visualizing feature importance: {e}")
    print("Please ensure `lgbm_model` was trained successfully and `train_features_scaled` (or `feature_names`) is correctly defined.")


print("\n--- Model Performance Summary ---")
results_df = pd.DataFrame.from_dict(results, orient='index')
# Display all columns for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(results_df)


print("\n--- lightGBM Feature Importance ---")

try:
    feature_names = []
    if include_stat_features:
        num_wavelet_features = train_wavelet_features.shape[1]
        wavelet_names = [f'WavFeat_{i}' for i in range(num_wavelet_features)]
        feature_names.extend(wavelet_names)

        if train_stds.shape[1] == len(stock_features):
             std_names = [f'Std_{f}' for f in stock_features]
        else: # Fallback to generic names if dimensions mismatch
             print(f"Warning: Mismatch between std feature count ({train_stds.shape[1]}) and stock_features count ({len(stock_features)}). Using generic std names.")
             std_names = [f'StdFeat_{i}' for i in range(train_stds.shape[1])]
        feature_names.extend(std_names)

    if X_fin_train.size > 0: # Check if financial features exist
        feature_names.extend(fin_features)

    if len(feature_names) != train_features_scaled.shape[1]:
        print(f"Error: Final feature name count ({len(feature_names)}) does not match training feature count ({train_features_scaled.shape[1]}). Cannot reliably show feature importances.")
    else:
        importances =  lgbm_model.feature_importances_

        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

        print("\nTop 20 CatBoost Feature Importances:")
        print(importance_df.head(20))

        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20], color='skyblue')
        plt.xlabel("LightGBM Feature Importance")
        plt.ylabel("Feature")
        plt.title("Top 20 LightGBM Feature Importances")
        plt.gca().invert_yaxis() # 중요도 높은 피처를 위로
        plt.tight_layout()
        plt.show()

except AttributeError:
     print("Error: Could not retrieve feature importances. Was the CatBoost model trained successfully?")
except NameError as e:
     print(f"Error: A required variable for feature importance generation is not defined: {e}")
except Exception as e:
    print(f"An error occurred while calculating/displaying CatBoost feature importances: {e}")


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Reshape, Conv1D, MaxPooling1D, LSTM, GRU, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam # Keep Adam import
from tensorflow.keras.callbacks import EarlyStopping # Keep EarlyStopping import


# Calculate class weights for Keras models
if scale_pos_weight_value == 1.0:
    keras_class_weight = None # No weighting needed
    print("Keras class_weight: None (balanced or single class)")
else:
    # Keras expects weights for each class {class_index: weight}
    # Weight for class 0 is typically 1
    # Weight for class 1 is scale_pos_weight
    keras_class_weight = {0: 1.0, 1: scale_pos_weight_value}
    print(f"Keras class_weight: {keras_class_weight}")


############################################################
#  Deep Learning Model Definitions (Using Combined Features)
############################################################
print("\nDefining Deep Learning Models (on Combined Features) ---")

# --- Define Input Shape ---
# Input shape is the number of combined features
input_shape_flat = (train_features_scaled.shape[1],)
# Reshape target for CNN/RNN layers: (batch_size, steps, features)
# We'll treat the flat features as 'steps' with 1 feature per step.
reshape_target = (train_features_scaled.shape[1], 1)
print(f"DL Input Shape (Flat): {input_shape_flat}")
print(f"DL Reshape Target: {reshape_target}")


# --- CNN Model ---
def build_cnn_model(input_shape, reshape_target_cnn, learning_rate=0.001):
    input_layer = Input(shape=input_shape, name='Input_Combined')
    # Reshape flat features into a pseudo-sequence
    reshaped = Reshape(target_shape=reshape_target_cnn, name='Reshape')(input_layer)
    # Apply Conv1D
    conv1 = Conv1D(filters=32, kernel_size=3, activation='relu', name='Conv1D_1')(reshaped)
    pool1 = MaxPooling1D(pool_size=2, name='MaxPool_1')(conv1)
    conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', name='Conv1D_2')(pool1)
    # Global pooling to flatten
    gap = GlobalAveragePooling1D(name='GlobalAvgPool')(conv2)
    # Dense layers
    dense1 = Dense(64, activation='relu', name='Dense_1')(gap)
    dropout1 = Dropout(0.3, name='Dropout_1')(dense1)
    output_layer = Dense(1, activation='sigmoid', name='Output_Sigmoid')(dropout1)

    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
    print("--- CNN Model Summary ---")
    model.summary()
    return model

# --- LSTM Model ---
def build_lstm_model(input_shape, reshape_target_lstm, learning_rate=0.001):
    input_layer = Input(shape=input_shape, name='Input_Combined')
    reshaped = Reshape(target_shape=reshape_target_lstm, name='Reshape')(input_layer)
    # Apply LSTM
    lstm1 = LSTM(units=50, return_sequences=False, name='LSTM_1')(reshaped) # return_sequences=False for the last layer
    # Dense layers
    dense1 = Dense(32, activation='relu', name='Dense_1')(lstm1)
    dropout1 = Dropout(0.3, name='Dropout_1')(dense1)
    output_layer = Dense(1, activation='sigmoid', name='Output_Sigmoid')(dropout1)

    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
    print("--- LSTM Model Summary ---")
    model.summary()
    return model

# --- GRU Model ---
def build_gru_model(input_shape, reshape_target_gru, learning_rate=0.001):
    input_layer = Input(shape=input_shape, name='Input_Combined')
    reshaped = Reshape(target_shape=reshape_target_gru, name='Reshape')(input_layer)
    # Apply GRU
    gru1 = GRU(units=50, return_sequences=False, name='GRU_1')(reshaped)
    # Dense layers
    dense1 = Dense(32, activation='relu', name='Dense_1')(gru1)
    dropout1 = Dropout(0.3, name='Dropout_1')(dense1)
    output_layer = Dense(1, activation='sigmoid', name='Output_Sigmoid')(dropout1)

    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
    print("--- GRU Model Summary ---")
    model.summary()
    return model

# --- Transformer Encoder Block Model ---
def build_transformer_model(input_shape, reshape_target_tf, head_size=256, num_heads=4, ff_dim=4, dropout=0.2, learning_rate=0.001):
    input_layer = Input(shape=input_shape, name='Input_Combined')
    reshaped = Reshape(target_shape=reshape_target_tf, name='Reshape')(input_layer)

    # Optional: Dense projection before Attention
    # projected = Dense(head_size, activation='relu')(reshaped)

    # Simplified Transformer Block (Attention + FeedForward)
    # Layer Normalization 1
    norm1 = LayerNormalization(epsilon=1e-6, name='LayerNorm_1')(reshaped) # Apply norm on reshaped input
    # Multi-Head Attention
    attn_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=head_size // num_heads, dropout=dropout, name='MultiHeadAttention'
    )(norm1, norm1) # Self-attention
    # Skip Connection 1 (Add)
    add1 = tf.keras.layers.Add(name='Add_1')([reshaped, attn_output]) # Add attention output to original reshaped input

    # Layer Normalization 2
    norm2 = LayerNormalization(epsilon=1e-6, name='LayerNorm_2')(add1)
    # Feed Forward Network
    ffn1 = Dense(ff_dim, activation="relu", name='FFN_1')(norm2)
    ffn_dropout = Dropout(dropout, name='FFN_Dropout')(ffn1)
    ffn2 = Dense(reshape_target_tf[-1], name='FFN_2')(ffn_dropout) # Project back to original dimension
    # Skip Connection 2 (Add)
    add2 = tf.keras.layers.Add(name='Add_2')([add1, ffn2]) # Add FFN output to output of first Add layer

    # Pooling and Final Classification Layers
    pool = GlobalAveragePooling1D(name='GlobalAvgPool')(add2)
    dense_out1 = Dense(64, activation="relu", name='Dense_Out_1')(pool)
    dropout_out = Dropout(0.3, name='Dropout_Out')(dense_out1)
    output_layer = Dense(1, activation="sigmoid", name='Output_Sigmoid')(dropout_out)

    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
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
        return inputs + self.position_embeddings

    def compute_mask(self, inputs, mask=None):
        return mask

# --- Transformer Encoder Block Model ---
def build_transformer_model_pos_encoding(input_shape, reshape_target_tf, head_size=64, num_heads=4, ff_dim=4, dropout=0.2, learning_rate=0.01):
    input_layer = Input(shape=input_shape, name='Input_Combined')
    reshaped = Reshape(target_shape=reshape_target_tf, name='Reshape')(input_layer)

    sequence_length = reshape_target_tf[0] 
    output_dim = reshape_target_tf[1]      
    x = Dense(64)(reshaped)  
    x = PositionalEmbedding(sequence_length=sequence_length, output_dim=output_dim)(reshaped)

    # Simplified Transformer Block (Attention + FeedForward)
    # Layer Normalization 1
    norm1 = LayerNormalization(epsilon=1e-6, name='LayerNorm_1')(x)
    # Multi-Head Attention
    attn_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=head_size // num_heads, dropout=dropout, name='MultiHeadAttention'
    )(norm1, norm1) # Self-attention
    # Skip Connection 1 (Add)
    add1 = tf.keras.layers.Add(name='Add_1')([x, attn_output])

    # Layer Normalization 2
    norm2 = LayerNormalization(epsilon=1e-6, name='LayerNorm_2')(add1)
    # Feed Forward Network
    ffn1 = Dense(ff_dim, activation="relu", name='FFN_1')(norm2)
    ffn_dropout = Dropout(dropout, name='FFN_Dropout')(ffn1)
    ffn2 = Dense(reshape_target_tf[-1], name='FFN_2')(ffn_dropout)
    # Skip Connection 2 (Add)
    add2 = tf.keras.layers.Add(name='Add_2')([add1, ffn2])

    # Pooling and Final Classification Layers
    pool = GlobalAveragePooling1D(name='GlobalAvgPool')(add2)
    dense_out1 = Dense(64, activation="relu", name='Dense_Out_1')(pool)
    dropout_out = Dropout(0.3, name='Dropout_Out')(dense_out1)
    output_layer = Dense(1, activation="sigmoid", name='Output_Sigmoid')(dropout_out)

    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=learning_rate) # 필요시 학습률 상향 조정 (e.g., 0.0005)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
    print("--- Transformer Model Summary (with Positional Embedding) ---")
    model.summary()
    return model

def build_tabular_transformer_model(input_shape, head_size=64, num_heads=4, ff_dim=32, embedding_dim=16, dropout=0.2, learning_rate=0.001):

    input_layer = Input(shape=input_shape, name='Input_Flat_Features') # 예: (29,)

    # (None, 29) -> (None, 29, 1)
    reshaped = Reshape((input_shape[0], 1), name='Reshape_for_Embedding')(input_layer)

    # (None, 29, 1) -> (None, 29, 16)
    feature_embedding = Dense(embedding_dim, activation='relu', name='Feature_Embedding')(reshaped)

    norm1 = LayerNormalization(epsilon=1e-6)(feature_embedding)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size // num_heads, dropout=dropout)(norm1, norm1)
    add1 = tf.keras.layers.Add()([feature_embedding, attn_output])

    norm2 = LayerNormalization(epsilon=1e-6)(add1)
    ffn = Dense(ff_dim, activation="relu")(norm2)
    ffn = Dense(embedding_dim)(ffn) # 차원 맞추기
    add2 = tf.keras.layers.Add()([add1, ffn])

    # (None, 29, 16) -> (None, 16)
    pool = GlobalAveragePooling1D(name='GlobalAvgPool')(add2)
    dense_out = Dense(64, activation="relu")(pool)
    dropout_out = Dropout(0.3)(dense_out)
    output_layer = Dense(1, activation="sigmoid", name='Output_Sigmoid')(dropout_out)

    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])

    print("--- Tabular Transformer Model Summary ---")
    model.summary()
    return model

############################################################
# Deep Learning Model Training and Evaluation
############################################################
print("\n Training and Evaluating Deep Learning Models ---")

# --- Training Parameters ---
epochs = 50 # Adjust as needed
batch_size = 128 # Adjust based on memory
patience = 10 # Early stopping patience

# Common Early Stopping Callback
early_stopping = EarlyStopping(monitor='val_loss', # Monitor validation loss
                               patience=patience,
                               restore_best_weights=True, # Restore weights from the best epoch
                               verbose=1)

# --- Train and Evaluate CNN ---
print("\n--- Training and Evaluating CNN ---")
try:
    model_cnn = build_cnn_model(input_shape_flat, reshape_target)
    history_cnn = model_cnn.fit(train_features_scaled, y_train,
                                validation_data=(val_features_scaled, y_val),
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=[early_stopping],
                                class_weight=keras_class_weight, # Use calculated class weights
                                verbose=1) # Set verbose=2 for less output per epoch

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

    print(f'[CNN] Train AUC: {roc_auc_train_cnn:.5f} | Train KS: {ks_train_cnn:.5f} | Train LogLoss: {logloss_train_cnn:.5f} | Train Brier: {brier_train_cnn:.5f}')
    print(f'[CNN] Test  AUC: {roc_auc_test_cnn:.5f}  | Test KS: {ks_test_cnn:.5f}  | Test LogLoss: {logloss_test_cnn:.5f}  | Test Brier: {brier_test_cnn:.5f}')

    results['CNN'] = {'train_auc': roc_auc_train_cnn, 'test_auc': roc_auc_test_cnn,
                      'train_ks': ks_train_cnn, 'test_ks': ks_test_cnn,
                      'train_logloss': logloss_train_cnn, 'test_logloss': logloss_test_cnn,
                      'train_brier': brier_train_cnn, 'test_brier': brier_test_cnn}
except Exception as e:
    print(f"Error training/evaluating CNN: {e}")
    results['CNN'] = {'train_auc': np.nan, 'test_auc': np.nan, 'train_ks': np.nan, 'test_ks': np.nan,
                      'train_logloss': np.nan, 'test_logloss': np.nan, 'train_brier': np.nan, 'test_brier': np.nan}


# --- Train and Evaluate LSTM ---
print("\n--- Training and Evaluating LSTM ---")
try:
    model_lstm = build_lstm_model(input_shape_flat, reshape_target)
    history_lstm = model_lstm.fit(train_features_scaled, y_train,
                                  validation_data=(val_features_scaled, y_val),
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  callbacks=[early_stopping],
                                  class_weight=keras_class_weight,
                                  verbose=1)

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

    print(f'[LSTM] Train AUC: {roc_auc_train_lstm:.5f} | Train KS: {ks_train_lstm:.5f} | Train LogLoss: {logloss_train_lstm:.5f} | Train Brier: {brier_train_lstm:.5f}')
    print(f'[LSTM] Test  AUC: {roc_auc_test_lstm:.5f}  | Test KS: {ks_test_lstm:.5f}  | Test LogLoss: {logloss_test_lstm:.5f}  | Test Brier: {brier_test_lstm:.5f}')

    results['LSTM'] = {'train_auc': roc_auc_train_lstm, 'test_auc': roc_auc_test_lstm,
                       'train_ks': ks_train_lstm, 'test_ks': ks_test_lstm,
                       'train_logloss': logloss_train_lstm, 'test_logloss': logloss_test_lstm,
                       'train_brier': brier_train_lstm, 'test_brier': brier_test_lstm}
except Exception as e:
    print(f"Error training/evaluating LSTM: {e}")
    results['LSTM'] = {'train_auc': np.nan, 'test_auc': np.nan, 'train_ks': np.nan, 'test_ks': np.nan,
                       'train_logloss': np.nan, 'test_logloss': np.nan, 'train_brier': np.nan, 'test_brier': np.nan}


# --- Train and Evaluate GRU ---
print("\n--- Training and Evaluating GRU ---")
try:
    model_gru = build_gru_model(input_shape_flat, reshape_target)
    history_gru = model_gru.fit(train_features_scaled, y_train,
                                validation_data=(val_features_scaled, y_val),
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=[early_stopping],
                                class_weight=keras_class_weight,
                                verbose=1)

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

    print(f'[GRU] Train AUC: {roc_auc_train_gru:.5f} | Train KS: {ks_train_gru:.5f} | Train LogLoss: {logloss_train_gru:.5f} | Train Brier: {brier_train_gru:.5f}')
    print(f'[GRU] Test  AUC: {roc_auc_test_gru:.5f}  | Test KS: {ks_test_gru:.5f}  | Test LogLoss: {logloss_test_gru:.5f}  | Test Brier: {brier_test_gru:.5f}')

    results['GRU'] = {'train_auc': roc_auc_train_gru, 'test_auc': roc_auc_test_gru,
                      'train_ks': ks_train_gru, 'test_ks': ks_test_gru,
                      'train_logloss': logloss_train_gru, 'test_logloss': logloss_test_gru,
                      'train_brier': brier_train_gru, 'test_brier': brier_test_gru}
except Exception as e:
    print(f"Error training/evaluating GRU: {e}")
    results['GRU'] = {'train_auc': np.nan, 'test_auc': np.nan, 'train_ks': np.nan, 'test_ks': np.nan,
                      'train_logloss': np.nan, 'test_logloss': np.nan, 'train_brier': np.nan, 'test_brier': np.nan}


# --- Train and Evaluate Transformer ---
print("\n--- Training and Evaluating Transformer ---")
try:
    input_shape_flat = (train_features_scaled.shape[1],)

    # Adjust hyperparameters if needed
    model_tf = build_tabular_transformer_model(input_shape_flat,
                                              head_size=128, num_heads=4, ff_dim=64, dropout=0.25)
    history_tf = model_tf.fit(train_features_scaled, y_train,
                              validation_data=(val_features_scaled, y_val),
                              epochs=epochs,
                              batch_size=batch_size,
                              callbacks=[early_stopping],
                              class_weight=keras_class_weight,
                              verbose=1)

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

    print(f'[Transformer] Train AUC: {roc_auc_train_tf:.5f} | Train KS: {ks_train_tf:.5f} | Train LogLoss: {logloss_train_tf:.5f} | Train Brier: {brier_train_tf:.5f}')
    print(f'[Transformer] Test  AUC: {roc_auc_test_tf:.5f}  | Test KS: {ks_test_tf:.5f}  | Test LogLoss: {logloss_test_tf:.5f}  | Test Brier: {brier_test_tf:.5f}')

    results['Transformer'] = {'train_auc': roc_auc_train_tf, 'test_auc': roc_auc_test_tf,
                              'train_ks': ks_train_tf, 'test_ks': ks_test_tf,
                              'train_logloss': logloss_train_tf, 'test_logloss': logloss_test_tf,
                              'train_brier': brier_train_tf, 'test_brier': brier_test_tf}
except Exception as e:
    print(f"Error training/evaluating Transformer: {e}")
    results['Transformer'] = {'train_auc': np.nan, 'test_auc': np.nan, 'train_ks': np.nan, 'test_ks': np.nan,
                              'train_logloss': np.nan, 'test_logloss': np.nan, 'train_brier': np.nan, 'test_brier': np.nan}


print("\n Model Performance Summary ---")
# Results from DL models are now included in the 'results' dictionary
results_df = pd.DataFrame.from_dict(results, orient='index')
# Display all columns for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(results_df.sort_values(by='test_auc', ascending=False)) # Sort by test AUC


print("\n Feature Importance ---")

try:
    # 1. 피처 이름 생성 (Code from original Section 16 is unchanged)
    feature_names = []
    if include_stat_features:
        num_wavelet_features = train_wavelet_features.shape[1]
        wavelet_names = [f'WavFeat_{i}' for i in range(num_wavelet_features)]
        feature_names.extend(wavelet_names)
        if train_stds.shape[1] == len(stock_features):
            std_names = [f'Std_{f}' for f in stock_features]
        else:
            print(f"Warning: Mismatch between std feature count ({train_stds.shape[1]}) and stock_features count ({len(stock_features)}). Using generic std names.")
            std_names = [f'StdFeat_{i}' for i in range(train_stds.shape[1])]
        feature_names.extend(std_names)

    if X_fin_train.size > 0:
        feature_names.extend(fin_features)

    # 최종 피처 이름 개수와 실제 학습 피처 개수 확인
    if len(feature_names) != train_features_scaled.shape[1]:
        print(f"Error: Final feature name count ({len(feature_names)}) does not match training feature count ({train_features_scaled.shape[1]}). Cannot reliably show feature importances.")
    else:
        # 2. 변수 중요도 추출
        importances = cat_model.feature_importances_

        # 3. DataFrame 생성 및 정렬
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

        # 4. 결과 출력 (상위 20개)
        print("\nTop 20 CatBoost Feature Importances:")
        print(importance_df.head(20))

        # 5. 결과 시각화 (상위 20개)
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['Feature'][:20], importance_df['Importance'][:20], color='skyblue')
        plt.xlabel("CatBoost Feature Importance")
        plt.ylabel("Feature")
        plt.title("Top 20 CatBoost Feature Importances")
        plt.gca().invert_yaxis() # 중요도 높은 피처를 위로
        plt.tight_layout()
        plt.show()

except AttributeError:
    print("Error: Could not retrieve feature importances. Was the CatBoost model trained successfully?")
except NameError as e:
    print(f"Error: A required variable for feature importance generation is not defined: {e}")
except Exception as e:
    print(f"An error occurred while calculating/displaying CatBoost feature importances: {e}")


print("\nScript finished.")