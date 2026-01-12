import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error, r2_score

def run_plot_group( prefix,timedata, merged_df):
    fig, axs = plt.subplots(6, 6, figsize=(17.5, 9.5),dpi = 100)
    for i in range(36):
        # col = f"{prefix}{i+1}"
        col = f"{i+1}{prefix}"
        if col in merged_df.columns:
            ax = axs[i // 6, i % 6]
            y_data = merged_df[col]
            x_data = timedata["time"] if "time" in timedata.columns else range(len(y_data))
            ax.plot(x_data, y_data)
            ax.set_title(col)
            ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

def run_plot_group_new( prefix,timedata, merged_df): 
    import matplotlib.pyplot as plt

    plots_per_page = 12
    rows, cols = 3, 4

    figs = []  # 收集 figure

    for page in range(3):
        fig, axs = plt.subplots(rows, cols, figsize=(17.5, 9.5),dpi = 80)
        axs = axs.flatten()

        for i in range(plots_per_page):
            idx = page * plots_per_page + i
            if idx >= 36:
                break

            col = f"{idx+1}{prefix}"
            ax = axs[i]

            if col in merged_df.columns:
                y = merged_df[col]
                x = timedata["time"] if "time" in timedata.columns else range(len(y))
                ax.plot(x, y)
                ax.set_title(col)
                ax.tick_params(axis='x', rotation=45)

        fig.tight_layout()
        figs.append(fig)

    # 🔥 一次顯示全部
    plt.show()

   

def run_plot_display_multi_algorithm_results(multi_results):
    for algorithm_name, results in multi_results.items():               
        # 在該算法分頁中創建第四層Tab (Factor vs EV / 預測對比)
        pls_result = results['pls']
        cv_result = results['cv']
        """創建 Factor vs EV 趨勢圖"""
        comp_cols = pls_result['comp_cols']
        max_factor = pls_result['max_factor']

        # 準備數據
        factors = list(range(1, max_factor + 1))

        # PLS EV數據 - 取第一個成分的EV
        pls_ev_data = []
        for factor in factors:
            if factor in pls_result['factor_results']:
                first_comp = comp_cols[0]
                ev = pls_result['factor_results'][factor]['stats'][first_comp]['explained_variance']
                pls_ev_data.append(ev)
            else:
                pls_ev_data.append(np.nan)  

        # CV EV數據 - 使用total_explained_variance
        cv_ev_data = []
        for factor in factors:
            if factor in cv_result['factor_results']:
                ev = cv_result['factor_results'][factor].get('total_explained_variance', 0)
                cv_ev_data.append(ev)
            else:
                cv_ev_data.append(np.nan)      

        # 繪製圖表
        matplotlib.rc('font', family='serif', serif=['ABC', 'MingLiU']) 
        matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        fig, ax = plt.subplots(figsize=(10, 5))
        # 繪製兩條線
        ax.plot(factors, pls_ev_data, 'o-', label='PLS EV', linewidth=2, markersize=8, color='blue')
        ax.plot(factors, cv_ev_data, 's--', label='CV EV', linewidth=2, markersize=8, color='red')
        
        # 標記最佳Factor            
        best_factor = cv_result['best_factor']
        ax.axvline(x=best_factor, color='green', linestyle=':', alpha=0.7, linewidth=2, 
                   label=f'Best Factor: {best_factor}')
        
        # 在最佳Factor點添加星號標記
        if best_factor <= len(pls_ev_data):
            ax.plot(best_factor, pls_ev_data[best_factor-1], 'g*', markersize=15)
        if best_factor <= len(cv_ev_data):
            ax.plot(best_factor, cv_ev_data[best_factor-1], 'g*', markersize=15)
        
        ax.set_xlabel('Factor', fontsize=12)
        ax.set_ylabel('Explained Variance', fontsize=12)
        ax.set_title(f'Factor vs Explained Variance ({algorithm_name})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(factors)
        
        # 設置Y軸範圍
        all_ev_values = [v for v in pls_ev_data + cv_ev_data if not np.isnan(v)]
        if all_ev_values:
            y_min = min(all_ev_values) * 0.95
            y_max = max(all_ev_values) * 1.05
            ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.show()

def run_create_prediction_comparison_chart(multi_results,Y_scaler):
    for algorithm_name, results in multi_results.items():               
        # 在該算法分頁中創建第四層Tab (Factor vs EV / 預測對比)
        pls_result = results['pls']
        cv_result = results['cv']

        factor = cv_result['best_factor']
        comp_cols = pls_result['comp_cols']
        
        # 獲取PLS和CV結果
        pls_factor_result = pls_result['factor_results'].get(factor)
        cv_factor_result = cv_result['factor_results'].get(factor)

        # 創建子圖
        n_comp = len(comp_cols)
        cols = min(n_comp, 2)
        rows = (n_comp + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if n_comp == 1:
            axes = [axes]
        else:
            axes = np.array(axes).flatten()
        
        Y_true = pls_result['Y_valid']
        pls_Y_pred = pls_factor_result['Y_pred']
        cv_Y_pred = cv_factor_result['all_y_pred_original']
        # Y_true = Y_scaler.inverse_transform(pls_result['Y_valid'])
        # pls_Y_pred = Y_scaler.inverse_transform(pls_factor_result['Y_pred'])
        # cv_Y_pred = Y_scaler.inverse_transform(cv_factor_result['all_y_pred_original'])
        # Y_true_log = Y_scaler.inverse_transform(pls_result['Y_valid'])
        # pls_Y_pred_log = Y_scaler.inverse_transform(pls_factor_result['Y_pred'])
        # cv_Y_pred_log = Y_scaler.inverse_transform(cv_factor_result['all_y_pred_original'])
        # Y_true = np.expm1(Y_true_log)
        # pls_Y_pred = np.expm1(pls_Y_pred_log)
        # cv_Y_pred = np.expm1( cv_Y_pred_log)

        # Y_true[Y_true < 0] = 0  # ============理論上不需要，但可保險用
        # pls_Y_pred[pls_Y_pred < 0] = 0
        # cv_Y_pred[pls_Y_pred < 0] = 0

        for idx, comp in enumerate(comp_cols):
            ax = axes[idx]
            y_true = Y_true[:, idx]
            pls_y_pred = pls_Y_pred[:, idx]
            cv_y_pred = cv_Y_pred[:, idx]
            
            
            
            # 繪製散點圖
            ax.scatter(y_true, pls_y_pred, alpha=0.6, label='PLS', color='blue')
            ax.scatter(y_true, cv_y_pred, alpha=0.6, label='CV', color='red', marker='s')
            
            # 計算y=x線的範圍
            ax_xlim = ax.get_xlim()
            ax_ylim = ax.get_ylim()
            plot_min = min(ax_xlim[0], ax_ylim[0])
            plot_max = max(ax_xlim[1], ax_ylim[1])
            
            # 繪製y=x線
            ax.plot([plot_min, plot_max], [plot_min, plot_max], 'k-', 
                    linewidth=0.5, alpha=0.8, label='y=x')
            
            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(plot_min, plot_max)
            
            # 計算統計
            pls_r2 = r2_score(y_true, pls_y_pred)
            cv_r2 = r2_score(y_true, cv_y_pred)
            pls_rms = np.sqrt(mean_squared_error(y_true, pls_y_pred))
            cv_rms = np.sqrt(mean_squared_error(y_true, cv_y_pred))
            bias = np.mean(cv_y_pred - y_true, axis=0)
            print(bias)
            # 添加統計信息
            ax.text(0.05, 0.95, 
                    f'PLS: R²={pls_r2:.3f}, rmse={pls_rms:.3f} \nCV: R²={cv_r2:.3f}, rmse={cv_rms:.3f}', 
                    transform=ax.transAxes, va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            ax.set_title(f'{comp} (Factor {factor})')
            ax.set_xlabel('參考值')
            ax.set_ylabel('預測值')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隱藏多餘的子圖
        for ax in axes[n_comp:]:
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.show()

def run_plot_backtest_results(predictions_dict, df_time, comp_cols, df_timeRef,Y_scaler ,pt,selected_component=None):
    """繪製多模型對比回測結果圖表（單圖模式）
    
    Args:
        predictions_dict: 預測結果字典
        comp_cols: 成分名稱列表
        selected_component: 選擇要顯示的成分名稱，如果為 None 則顯示第一個成分
    """
    # time_data = df_timeRef['Time'].values
    time_data = df_time['Time'].values
    
    # # 確定要顯示的成分
    # if selected_component is None:
    #     selected_component = comp_cols[0] if comp_cols else None
                
    # # 獲取成分索引
    # comp_idx = comp_cols.index(selected_component)
    
    # 創建單一圖表
    matplotlib.rc('font', family='serif', serif=['ABC', 'MingLiU']) 
    matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    for page in range(len(comp_cols)):
        # 確定要顯示的成分
        if selected_component is None:
            selected_component = comp_cols[page] if comp_cols else None
        else:    
            selected_component = comp_cols[page] if comp_cols else None

        fig = plt.figure(figsize=(14, 4))
        ax = plt.subplot(1, 1, 1)
        # 準備顏色
        colors = plt.cm.tab10(np.linspace(0, 1, 10))  # tab10 色盤
        
        # 為每個模型繪製預測線
        for model_idx, (unique_key, pred_data) in enumerate(predictions_dict.items()):
            Y_pred = pred_data['predictions']
            # Y_pred = Y_scaler.inverse_transform(pred_data['predictions'])
            # Y_pred = pt.inverse_transform(pred_data['predictions'])
            model_name = pred_data['model_name']
            factor = pred_data['factor']
            stats = pred_data.get('stats', {})
            
            # 選擇顏色
            color = colors[model_idx % len(colors)]
            
            # 獲取該成分的 R² 統計數據（如果有）
            r2_score = None
            if stats and selected_component in stats:
                comp_stats = stats[selected_component]
                if isinstance(comp_stats, dict) and 'R2' in comp_stats:
                    r2_score = comp_stats['R2']
            
            # 構建圖例標籤
            if r2_score is not None:
                label = f"{unique_key} (R²={r2_score:.3f})"
            else:
                label = unique_key
            
            # 繪製散點（使用 plot，但 linestyle='none' 移除連線）
            ax.plot(
                time_data, 
                Y_pred[:, page],
                linestyle='none',
                color=color,
                marker='.',
                markersize=5,
                label=label,
                alpha=0.5
            )
        
        # 繪製參考數據（實際值）- 最後繪製，顯示在最上層
        if df_timeRef is not None:
            try:
                # 獲取參考數據的時間和成分值
                if 'Time' in df_timeRef.columns and selected_component in df_timeRef.columns:
                    ref_time = df_timeRef['Time'].values
                    ref_values = df_timeRef[selected_component].values
                    
                    # 繪製參考數據為紅色星形標記
                    ax.plot(
                        ref_time,
                        ref_values,
                        linestyle='none',
                        color='red',
                        marker='o',
                        markersize=6,
                        markerfacecolor='none', 
                        label='Reference Data',
                        alpha=0.9,
                        zorder=100
                    )
            except Exception as e:
                print(f"無法繪製參考數據: {e}")
        
        # 設置圖表標題和標籤
        ax.set_title(
            f"{selected_component} 多模型回測對比",
            fontsize=11,
            fontweight='bold'
        )
        ax.set_xlabel('Time', fontsize=9)
        ax.set_ylabel('Predicted Value', fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 添加圖例（放在子圖外側右方或下方）
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
            framealpha=0.9
        )
        
        # 調整佈局以防止重疊
    plt.tight_layout()
    plt.show()