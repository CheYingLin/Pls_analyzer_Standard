import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any, Callable, Optional

def determine_cv_strategy( n_samples: int) -> Tuple[Any, str, int]:
        """
        根據樣本數量自動確定最優CV策略
        
        決策邏輯:
        - n ≤ 20: Leave-One-Out CV (最大化數據利用)
        - n > 20: 20-fold CV (平衡效率和精度)
        
        Parameters:
        -----------
        n_samples : int
            樣本數量
        
        Returns:
        --------
        Tuple[Any, str, int]
            (cv_object, cv_type, n_folds)
        """
        if n_samples <= 20:
            return LeaveOneOut(), "Leave-One-Out", n_samples
        else:
            return KFold(n_splits=20, shuffle=True, random_state=42), "20-fold", 20
# === 私有方法 ===
    
def _calculate_total_explained_variance_standardized( y_true, y_pred):
    """
    計算標準化Y的總體解釋變異量
    
    此方法專門用於計算Total EV，確保跨成分公平比較
    輸入的y_true和y_pred必須是已經標準化後的數據
    
    Parameters:
    -----------
    y_true : np.ndarray
        標準化後的真實值 (n_samples, n_components)
    y_pred : np.ndarray
        標準化後的預測值 (n_samples, n_components)
    
    Returns:
    --------
    float
        總體解釋變異量 (0-1之間)
    """
    n_samples, n_components = y_true.shape
    
    # 總變異量（標準化後每個成分方差≈1）
    # ss_tot = n_samples * n_components #這裡不保證方差一定等於不建議用這個
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    
    # 殘差平方和
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # 解釋變異量
    explained_variance = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return explained_variance

def cross_validate_single_factor( X: np.ndarray, Y_standardized: np.ndarray, 
                                   Y_original: np.ndarray, n_components: int, 
                                   comp_cols: List[str]) -> Dict[str, Any]:
    """
    累積交叉驗證主函數（修正版）
    
    核心原則：
    1. 使用原始尺度數據訓練PLS模型
    2. 使用原始尺度預測值計算所有指標（R²、RMSE）
    3. 僅在計算Total EV時額外使用標準化數據
    
    Parameters:
    -----------
    X : np.ndarray
        光譜數據 (n_samples, n_features)
    Y_standardized : np.ndarray
        標準化成分數據 (已廢棄，保留以兼容接口)
    Y_original : np.ndarray
        原始尺度成分數據 (n_samples, n_components)
    n_components : int
        Factor數量
    comp_cols : List[str]
        成分名稱列表
    
    Returns:
    --------
    Dict[str, Any]
        完整的累積交叉驗證結果
    """
    # 自動確定CV策略
    cv_object, cv_type, n_folds = determine_cv_strategy(X.shape[0])
    
    # 初始化累積數據容器（僅保存原始尺度）
    y_true_list = []
    y_pred_list = []
    
    # 執行累積交叉驗證循環
    for train_idx, val_idx in cv_object.split(X):
        # 數據分割
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y_original[train_idx], Y_original[val_idx]
        
        # PLS模型訓練（使用原始尺度數據）
        pls = PLSRegression(n_components=n_components, scale=False)
        pls.fit(X_train, Y_train)
        
        # 預測（原始尺度）
        Y_pred = pls.predict(X_val)
        
        # 累積結果
        y_true_list.append(Y_val)
        y_pred_list.append(Y_pred)
    
    # 合併累積數據
    y_true_original = np.vstack(y_true_list)
    y_pred_original = np.vstack(y_pred_list)
    
    # 計算原始尺度指標（R²、RMSE）
    r2_original = []
    rmse_original = []
    
    for i in range(len(comp_cols)):
        try:
            # R²（原始尺度）
            r2 = r2_score(y_true_original[:, i], y_pred_original[:, i])
            r2_original.append(r2 if np.isfinite(r2) else 0.0)
            
            # RMSE（原始尺度）
            rmse = np.sqrt(np.mean((y_true_original[:, i] - y_pred_original[:, i]) ** 2))
            rmse_original.append(rmse if np.isfinite(rmse) else 0.0)
        except Exception as e:
            print(f"警告：成分 {comp_cols[i]} 計算失敗: {e}")
            r2_original.append(0.0)
            rmse_original.append(0.0)
    
    # 額外計算：標準化尺度的Total EV（僅用於EV計算）
    # 使用全局標準化器（確保跨成分公平比較）
    global_scaler = StandardScaler()
    y_true_standardized = global_scaler.fit_transform(y_true_original)
    y_pred_standardized = global_scaler.transform(y_pred_original)
    
    # 計算總體解釋變異量（標準化尺度）
    total_explained_variance = _calculate_total_explained_variance_standardized(
        y_true_standardized, y_pred_standardized)
    
    # 返回結果
    return {
        # 主要結果（原始尺度）- 用於所有業務指標
        'mean_cv_scores_original': r2_original,
        'rmse_means': rmse_original,
        'all_y_true_original': y_true_original,
        'all_y_pred_original': y_pred_original,
        
        # EV結果（標準化尺度）- 僅用於Total EV計算
        'total_explained_variance': total_explained_variance,
        'all_y_true': y_true_standardized,  # 保留以兼容現有代碼
        'all_y_pred': y_pred_standardized,  # 保留以兼容現有代碼
        
        # 其他信息
        'cv_type': cv_type,
        'k_folds': n_folds,
        'mean_cv_scores': r2_original,  # 兼容舊版接口
        'std_cv_scores': [0.0] * len(comp_cols),
        'std_cv_scores_original': [0.0] * len(comp_cols),
        'rmse_stds': [0.0] * len(comp_cols)
    }

def recommend_best_factor( factor_results: Dict[int, Dict[str, Any]]) -> int:
    """推薦最佳Factor"""
    if not factor_results:
        return 1
    
    factors = sorted(factor_results.keys())
    factor_evs = [(factor, factor_results[factor]['total_explained_variance']) 
                    for factor in factors]
    
    # 找到EV最高的Factor
    max_ev = max(ev for _, ev in factor_evs)
    
    # 檢查是否有EV相近的Factor（差異<0.01）
    similar_factors = []
    for factor, ev in factor_evs:
        if abs(ev - max_ev) < 0.01:
            similar_factors.append((factor, ev))
    
    if len(similar_factors) > 1:
        # 選擇較小的Factor（簡潔性原則）
        best_factor = min(similar_factors, key=lambda x: x[0])[0]
    else:
        best_factor = next(factor for factor, ev in factor_evs if ev == max_ev)
    
    return best_factor

def run_cross_validation_analysis( X: np.ndarray, Y: np.ndarray, 
                                         comp_cols: List[str],  
                                         max_factor: int = 8, 
                                         progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        執行交叉驗證分析（核心分析邏輯）
        
        此方法專注於純粹的分析邏輯，不包含UI相關代碼
        重新命名以避免與主程式方法混淆
        
        Parameters:
        -----------
        X : np.ndarray
            光譜數據
        Y : np.ndarray
            成分數據
        comp_cols : List[str]
            成分名稱列表
        algorithm_info : Dict[str, str]
            算法信息字典
        max_factor : int
            最大Factor數量
        progress_callback : Optional[Callable]
            進度回調函數
        
        Returns:
        --------
        Dict[str, Any]
            交叉驗證結果
        """
        # 數據預處理：移除包含NaN的行
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
        X_valid = X[mask]
        Y_valid = Y[mask]
        
        # 檢查數據點是否足夠
        n_samples = X_valid.shape[0]
        if n_samples < 3:
            raise ValueError(f"篩選後資料點不足 ({n_samples}), 至少需 3 筆進行交叉驗證")
        
        # 動態確定最大Factor數量
        if n_samples < 8:
            max_factor = max(1, n_samples - 2)
        
        # 確保最大Factor不超過特徵數量
        n_features = X_valid.shape[1]
        max_factor = min(max_factor, n_features)
        
        if max_factor < 1:
            raise ValueError(f"無法進行分析：數據點數 ({n_samples}) 或特徵數 ({n_features}) 不足")
        
        # Y數據標準化
        scaler = StandardScaler()
        Y_standardized = scaler.fit_transform(Y_valid)
        
        # 存儲每個Factor的結果
        factor_results = {}
        
        # 輪流計算 Factor 1 到 max_factor
        for factor in range(1, max_factor + 1):
            if progress_callback:
                progress_callback(factor, max_factor)
            
            # 交叉驗證單個Factor
            result = cross_validate_single_factor(
                X_valid, Y_standardized, Y_valid, factor, comp_cols
            )
            factor_results[factor] = result
        
        # 推薦最佳Factor
        best_factor = recommend_best_factor(factor_results)
        
        # 存儲結果
        results: Dict[str, Dict[str, Any]] = {}  # 儲存交叉驗證結果
        # cv_key = f"custom_{algorithm_info.get('subalg', 'default')}"
        cv_key = f"custom_算法X"
        results[cv_key] = {
            'factor_results': factor_results,
            'best_factor': best_factor,
            'max_factor': max_factor,
            'n_samples': n_samples,
            'comp_cols': comp_cols,
            'algorithm_info': "算法X"
        }
        
        return results[cv_key]