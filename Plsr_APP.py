#import modules
import pandas as pd
import numpy as np
import time
from datetime import timedelta
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer, StandardScaler
from typing import Tuple, List, Dict, Any, Optional
from Cross_validation import run_cross_validation_analysis
from Backtesting import run_multi_model_backtest
from Plotgrop import run_plot_group_new,run_plot_display_multi_algorithm_results,run_create_prediction_comparison_chart
from Plotgrop import run_plot_backtest_results
# import seaborn as sns

tic = time.time()
def _preprocess_data( X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """數據預處理：移除NaN並檢查數據充足性"""
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
    X_valid = X[mask]
    Y_valid = Y[mask]
    
    n_samples = X_valid.shape[0]
    if n_samples < 2:
        raise ValueError(f"篩選後資料點不足 ({n_samples}), 至少需 2 筆")
    
    return X_valid, Y_valid, n_samples

def _determine_max_factor( n_sample: int, n_features: int, max_factor: int = 8) -> int:
    """確定最大Factor數量（與交叉驗證邏輯一致）"""
    if n_sample < 8:
        max_factor = max(1, n_sample - 2)
    
    max_factor = min(max_factor, n_features)
    
    if max_factor < 1:
        raise ValueError(f"無法進行分析：數據點數 ({n_sample}) 或特徵數 ({n_features}) 不足")
    
    return max_factor

def _fit_single_factor( X: np.ndarray, Y: np.ndarray, n_component: int) -> Tuple[PLSRegression, np.ndarray]:
    """單個Factor的PLS建模"""
    pls = PLSRegression(n_components=n_component, scale=False)
    pls.fit(X, Y)
    Y_pred = pls.predict(X)
    return pls, Y_pred

def _calculate_regression_stats( Y_true: np.ndarray, Y_pred: np.ndarray, 
                                  comp_cols: List[str]) -> Dict[str, Dict[str, float]]:
    """計算回歸統計信息"""
    stats = {}
    
    # 計算各成分的個別統計
    for idx, comp in enumerate(comp_cols):
        y_true = Y_true[:, idx]
        y_pred = Y_pred[:, idx]
        
        r2 = r2_score(y_true, y_pred)
        coeffs = np.polyfit(y_true, y_pred, 1)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        stats[comp] = {
            'r2': r2,
            'slope': coeffs[0],
            'intercept': coeffs[1],
            'rmse': rmse
        }
    
    # 計算 explained variance（比照 cross_validation 算法）
    # 1. 標準化所有成分的 y_prediction 和 y_reference
    # scaler_true = StandardScaler()
    # scaler_pred = StandardScaler()
    
    # Y_true_scaled = scaler_true.fit_transform(Y_true)
    # Y_pred_scaled = scaler_pred.fit_transform(Y_pred)
    # NEW Scale
    scaler_all = StandardScaler()

    Y_true_scaled = scaler_all.fit_transform(Y_true)  # ← 學尺度
    Y_pred_scaled = scaler_all.transform(Y_pred)      # ← 用同一尺度

    
    # 2. 計算所有成分的 R² 並取平均
    component_r2_values = []
    for idx in range(Y_true.shape[1]):
        r2 = r2_score(Y_true_scaled[:, idx], Y_pred_scaled[:, idx])
        component_r2_values.append(r2)
    
    explained_variance = np.mean(component_r2_values)
    
    # 3. 將 explained_variance 添加到每個成分的統計中
    for comp in comp_cols:
        stats[comp]['explained_variance'] = explained_variance
    
    return stats

def run_pls_factor_scan(X: np.ndarray, Y: np.ndarray, 
                        comp_cols: List[str], 
                        max_factor: int = 8, 
                        progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """
    執行PLS Factor掃描分析
    """
    X_valid, Y_valid, n_samples = _preprocess_data(X, Y)
    n_features = X_valid.shape[1]
    max_factor = _determine_max_factor(n_samples, n_features, max_factor)

    factor_results = {}
    # Factor掃描
    for factor in range(1, max_factor + 1):
        if progress_callback:
            progress_callback(factor, max_factor)
        
        pls, Y_pred = _fit_single_factor(X_valid, Y_valid, factor)
        stats = _calculate_regression_stats(Y_valid, Y_pred, comp_cols)
            
        factor_results[factor] = {
            'model': pls, #模型訓練出來的 參數 ex.y^​=X⋅coef_+intercept 可得 coef_
            'Y_pred': Y_pred,
            'stats': stats
        }

    result = {
            'factor_results': factor_results,
            'X_valid': X_valid,
            'Y_valid': Y_valid,
            'max_factor': max_factor,
            'comp_cols': comp_cols,
            'n_samples': n_samples
        }
        
        
    return result
def run_selected_specturm(X_tmp, unselect):
    mw_row_avg = []
    for i in range(1, 37):
        if i in unselect:
            continue  # 跳過這個 i
        # MW數據
        mw_col = f"MW{i}"#MW_absorb_normalized,_normalized
        if mw_col in X_tmp.columns:
            mw_values = X_tmp[mw_col].dropna()
            mw_row_avg.append(mw_values if not mw_values.isna().all() else np.nan) 
        else:
            mw_row_avg.append(np.nan)
    return np.array(mw_row_avg).T       

def run_selected(X_tmp, time_window, df_timeRef,unselect):
    mw_mat = []
    time_before = pd.Timedelta(minutes=time_window[0])
    time_after = pd.Timedelta(minutes=time_window[1])
    # 找到時間窗口範圍內的所有數據點（非對稱）
    for t in df_timeRef['Time']:
        time_mask = (X_tmp['Time'] >= t - time_before) & (X_tmp['Time'] <= t + time_after)
        mw_row_avg = []
        matched_rows = X_tmp[time_mask]
        for i in range(1, 37):
            if i in unselect:
                continue  # 跳過這個 i
            # MW數據
            mw_col = f"MW{i}"#MW_absorb_normalized,_normalized
            if mw_col in matched_rows.columns:
                mw_values = matched_rows[mw_col].dropna()
                mw_row_avg.append(mw_values.mean() if not mw_values.empty else np.nan)
            else:
                mw_row_avg.append(np.nan)
        mw_mat.append(mw_row_avg)
    return np.array(mw_mat)   
#===============main======================================
#===========import dataset================================
df = pd.read_excel("data_out\data_SC1_25043for_AI.xlsx")

df['Time'] = pd.to_datetime(df['Time'])
#==========================concentration Table==========================
df_timeRef = pd.read_excel("data_out\concentration_list_SC-1藥水 DOE-20251224-含變溫(包含氨水重測)_pls.xlsx")
df_timeRef["Time"] = (df_timeRef["Time"].astype(str).str.replace(" PM", "", regex=False).str.replace(" AM", "", regex=False))
df_timeRef["Time"] = df_timeRef["Time"].astype(str).str.strip()
df_timeRef["Time"] = pd.to_datetime(df_timeRef["Time"],format="mixed",errors="coerce")
#observing dataset
# df.head()
Concentration_slec_colnum = 2
timedata = df.iloc[:,1:2]
# time_window = [timedelta(minutes=5), timedelta(minutes=0)]
time_window = [5, 0]#(minutes)
timedata.head()
#X是所有可能的影響變因
#取得所有的列的0,1,2,3,4欄位
# X_df.head()
# y是目標值
# print(df.columns)
# 清理欄名
df.columns = df.columns.str.strip()
# comp_cols = ['Cu-化驗值', 'NaOH-化驗值', 'HCHO-化驗值','EDTA-化驗值']
comp_cols = ['NH4OH','H2O2']
comp_indices = [comp_cols.index(name) for name in comp_cols if name in comp_cols]

# Y.head()
#==============plot part I Rawdatas################
# prefix = '-MW_absorb_normalized'
# run_plot_group_new(prefix,timedata, X_df)
#==================================################
channel_unselect = [1,2,3,10,11,12,25,26,33,34]
X = run_selected(df, time_window, df_timeRef, channel_unselect)
# Y = df.iloc[:, [i+Concentration_slec_colnum for i in comp_indices]].values
Y_temp = df_timeRef.iloc[:, [i+Concentration_slec_colnum for i in comp_indices]].values
toc = time.time()
word = f"preprocessing花費時間：{toc - tic:.3f} 秒" 
print(f"\x1b[32m{word}\x1b[0m")
# III
# # 1️⃣ 壓縮 y 的分布
pt = PowerTransformer(method="yeo-johnson")
Y_scaler = StandardScaler()
# Y_t = pt.fit_transform(Y_temp)
# # 2️⃣ 再做標準化（仍然建議）
# Y = Y_scaler.fit_transform(Y_t)
#  II
# Y_scaler = StandardScaler()
# Y = Y_scaler.fit_transform(Y_temp) #使用時記得去改plotgrop裡的(line148,line239)
# VI
Y = Y_temp
# VII
# Y_scaler = StandardScaler()#使用時記得去改plotgrop裡的(line154)
# Y_log = np.log1p(Y_temp)
# Y = Y_scaler.fit_transform(Y_log)
# 執行PLS分析
pls_result = run_pls_factor_scan(
                    X, Y, comp_cols, max_factor=8)
# 執行交叉驗證
cv_result = run_cross_validation_analysis(
    X, Y, comp_cols, max_factor=8
)

# 儲存結果（包含時間窗口設定）
# 儲存多算法結果
multi_algorithm_results = {}
multi_algorithm_results["算法X"] = {
    'pls': pls_result,
    'cv': cv_result,
    }
#==============plot part II PLS predict################

run_plot_display_multi_algorithm_results(multi_algorithm_results)
run_create_prediction_comparison_chart( multi_algorithm_results,Y_scaler)

#==============Backtesting Part===========
model_name = "算法X"
factor = 5 # select by user choice
unique_key = f"{model_name}_F{factor}"



model_data = multi_algorithm_results[model_name]
pls_results = model_data.get('pls', {})
factor_results = pls_results.get('factor_results', {})

model_info = factor_results[factor]
pls_model = model_info.get('model')
stats = model_info.get('stats', {})
# 4.2 準備預測數據
# 判斷是否使用參考通道
# X_pred = X.copy() 
X_pred = run_selected_specturm(df, channel_unselect)
backtest_result = run_multi_model_backtest(X_pred,factor, model_data, comp_cols,
                                            pls_model ,stats, unique_key,model_name)

#==============plot part III Backtesting################

run_plot_backtest_results(backtest_result, df, backtest_result[f"算法X_F{factor}"]['comp_names'], df_timeRef,Y_scaler,pt)
print('Done!!!!!!!!!!!!!!!!!!!!')








