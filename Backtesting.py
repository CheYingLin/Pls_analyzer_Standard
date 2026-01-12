import numpy as np
from typing import Tuple, List, Dict, Any, Callable, Optional

def run_multi_model_backtest(X: np.ndarray ,factor: int, model_data, comp_cols: List[str], 
                             pls_model, stats, unique_key, model_name):
    predictions_dict = {}
    X_pred = X    
        # 4.3 計算係數
    coefs = pls_model.coef_ if pls_model.coef_.shape[0] != len(comp_cols) else pls_model.coef_.T

    # 4.4 計算截距 - 從訓練結果中獲取
    # model_result = multi_algorithm_results[model_name]
    model_result = model_data
    X_valid = model_result['pls']['X_valid']
    Y_valid = model_result['pls']['Y_valid']

    # 計算訓練數據的均值
    X_mean = X_valid.mean(axis=0)
    Y_mean = Y_valid.mean(axis=0)

    # 計算截距
    intercepts = Y_mean - X_mean.dot(coefs)

    # 4.5 執行預測
    Y_pred = X_pred.dot(coefs) + intercepts

    # 4.6 存儲結果
    predictions_dict[unique_key] = {
        'predictions': Y_pred,
        'comp_names': comp_cols,
        'stats': stats,
        'model_name': model_name,
        'factor': factor
    }
    return{
            f"{model_name}_F{factor}": {
                    'predictions': Y_pred,
                    'comp_names': comp_cols,
                    'stats': {},
                    'model_name': model_name,
                    'factor': factor
                }
            }# 轉換為新格式並調用多模型繪圖函數