# from process_plot_data import *
import matplotlib.pyplot as plt
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, classification_report, mean_absolute_error, mean_squared_log_error

# Add multi-currency pair processing functionality
from multi_feature_process import *
from matplotlib import rcParams
rcParams['font.family'] = 'cmr10'
rcParams['axes.formatter.use_mathtext'] = True  # Use mathtext for tick labels
rcParams['text.usetex'] = True

def process_multi_currency_pairs(file_path, all_midas_features):
    """
    Process multi-currency data and generate risk signal alerts for each currency pair.
    """
    # Load multi-currency data
    multi_pairs = pd.read_excel(file_path, parse_dates=["date"])

    # Set date as index
    multi_pairs.set_index('date', inplace=True)

    # Define currency pairs to process
    # currency_pairs = ['USDJPY', 'USDCNH', 'JPYCNY', 'EURUSD', 'EURJPY', 'EURCNH']
    currency_pairs = ['USDJPY', 'EURUSD', 'EURJPY', 'USDCNH', 'JPYCNY']

    # Store results and models for each pair
    models = {}
    results = {}

    for pair in currency_pairs:
        print(f"\nProcessing currency pair: {pair}")
        # Extract the pair's data
        pair_data = pd.DataFrame()
        pair_data['Open'] = multi_pairs[f'{pair}_open']
        pair_data['High'] = multi_pairs[f'{pair}_high']
        pair_data['Low'] = multi_pairs[f'{pair}_low']
        pair_data['Close'] = multi_pairs[f'{pair}_close']

        # Calculate daily returns
        pair_data['FX_Daily_Returns'] = pair_data['Close'].pct_change()

        # Calculate risk score
        pair_data = risk_score_definitions(pair_data)

        # Add technical indicators
        # SMA & EMA
        pair_data['sma_20'] = pair_data['Close'].rolling(window=20).mean()
        pair_data['ema_50'] = pair_data['Close'].ewm(span=50, adjust=False).mean()

        # Bollinger Bands
        pair_data["MiddleBand"] = pair_data["Close"].rolling(window=20).mean()
        pair_data["UpperBand"] = pair_data["MiddleBand"] + 2 * pair_data["Close"].rolling(window=20).std()
        pair_data["LowerBand"] = pair_data["MiddleBand"] - 2 * pair_data["Close"].rolling(window=20).std()

        # RSI
        pair_data['rsi_14'] = ta.rsi(pair_data['Close'], length=14)

        # ATR (lagged)
        pair_data['atr_14_lagged'] = ta.atr(pair_data['High'], pair_data['Low'], pair_data['Close'], length=14).shift(1)

        # Lagged features
        pair_data['FX_Daily_Returns_lag1'] = pair_data['FX_Daily_Returns'].shift(1)
        pair_data['FX_Daily_Returns_lag3'] = pair_data['FX_Daily_Returns'].shift(3)

        # Rolling statistics
        pair_data['FX_Returns_rolling_mean_10'] = pair_data['FX_Daily_Returns'].rolling(window=10).mean().shift(5)
        pair_data['FX_Returns_rolling_std_10'] = pair_data['FX_Daily_Returns'].rolling(window=10).std().shift(5)

        # Merge with external features if available
        if pair in all_midas_features:
            pair_data = pair_data.merge(all_midas_features[pair], left_index=True, right_index=True, how='left')

        pair_data = pair_data.dropna()

        selected_features = pair_data.drop(
            columns=['FX_risk_score', 'Open', 'High', 'Low', 'Close', 'FX_Daily_Returns', 'Future_Direction']
        ).columns

        # Correlation filtering
        corr_matrix_abs = pair_data[selected_features].corr().abs()
        corr_threshold = 0.85
        upper = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool))

        # Compute feature variances for numeric columns
        numeric_cols = pair_data[selected_features].select_dtypes(include=[np.number]).columns
        feature_variances = pair_data[numeric_cols].var()

        # Drop highly correlated features with lower variance
        to_drop = set()
        for column in upper.columns:
            for row in upper.index:
                if upper.loc[row, column] > corr_threshold:
                    drop_feature = row if feature_variances[row] < feature_variances[column] else column
                    to_drop.add(drop_feature)

        # Filter features
        original_filtered_features = [col for col in selected_features if col not in to_drop]
        X = pair_data[original_filtered_features]

        # Target variable for regression
        y_reg = pair_data['FX_risk_score'].values

        '''try importance'''
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y_reg)
        feature_importances = rf_model.feature_importances_

        feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
        feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
        print(feature_importance_df)
        importance_thereshold = 0.005
        important_features = feature_importance_df[feature_importance_df["Importance"] > importance_thereshold][
            "Feature"].tolist()
        X = X[important_features]
        ''''rrrrrfff'''

        # Train-test split
        split_index = int(len(pair_data) * 0.8)
        train_data = pair_data.iloc[:split_index]
        test_data = pair_data.iloc[split_index:]

        X_train, y_train_reg = train_data[X.columns], train_data['FX_risk_score']
        X_test, y_test_reg = test_data[X.columns], test_data['FX_risk_score']

        # Classification target variable
        y_class = pd.qcut(y_reg, q=3, labels=[0, 1, 2])
        _, _, y_train_class, y_test_class = train_test_split(
            X, y_class, test_size=0.2, shuffle=False, random_state=None
        )

        # Train regression model
        reg_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
        reg_model.fit(X_train, y_train_reg)

        # Regression prediction
        y_pred_reg = pd.Series(reg_model.predict(X_test), index=y_test_reg.index)

        # Evaluate regression model
        rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
        mae = mean_absolute_error(y_test_reg, y_pred_reg)

        # Train classification model
        clf_model = XGBClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.9, colsample_bytree=0.8
        )
        clf_model.fit(X_train, y_train_class)

        # Classification prediction
        y_pred_class = clf_model.predict(X_test)

        # Evaluate classification model
        class_report = classification_report(y_test_class, y_pred_class, output_dict=True)

        # Visualization output folder
        output_dir = os.path.expanduser(f'~/Desktop/results/{pair}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save regression prediction plot
        plot_real_pred_reg(y_test_reg, y_pred_reg, f'{output_dir}/volatility_pred_real.png')

        # Visualize risk classes
        threshold_type = 'value'
        visualize_detection_class(
            y_test_reg, y_pred_reg, y_test_class, y_pred_class,
            threshold_type, True, True, 0.003, 0.95,
            f'{output_dir}/classify_{threshold_type}.png',
            f'{output_dir}/classify_CI.png',
            f'{output_dir}/classify_class2.png'
        )

        # Store models and results
        models[pair] = {'reg_model': reg_model, 'clf_model': clf_model}
        results[pair] = {
            'rmse': rmse,
            'mae': mae,
            'classification_report': class_report,
            'test_data': test_data,
            'y_test_reg': y_test_reg,
            'y_pred_reg': y_pred_reg,
            'y_test_class': y_test_class,
            'y_pred_class': y_pred_class
        }

        print(f"{pair} finished. RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        print(f"Classification Report:\n{classification_report(y_test_class, y_pred_class)}")

        # Backtesting logic
        # print(f"\nBacktest analysis for {pair} ...")
        #
        # # Convert NumPy array to Series
        # y_pred_class_series = pd.Series(y_pred_class, index=test_data.index)
        #
        # # Backtest
        # backtest_results = backtest_risk_signals(
        #     test_data,
        #     y_pred_class_series,  # Series version
        #     risk_threshold=2,     # High risk threshold
        #     stop_loss_pct=0.02,   # 2% stop loss
        #     take_profit_pct=0.05  # 5% take profit
        # )
        #
        # # Save backtest results
        # results[pair]['backtest_results'] = backtest_results
        #
        # # Visualize backtest results
        # visualize_backtest_results(
        #     pair,
        #     backtest_results,
        #     y_pred_class_series,
        #     test_data['Close'],
        #     os.path.expanduser(f'~/Desktop/results/{pair}')
        # )
        #
        # print(f"{pair} backtest done")
        # print(f"Total trades: {backtest_results['total_trades']}")
        # print(f"Win rate: {backtest_results['win_rate']:.2f}%")
        # print(f"Average return: {backtest_results['avg_return']:.2f}%")
        # print(f"Max drawdown: {backtest_results['max_drawdown']:.2f}%")
        # print(f"Sharpe ratio: {backtest_results['sharpe_ratio']:.2f}")

    # ------ Multi-currency systemic risk analysis ------

    # ================== 1. 系统性风险可视化区块 begin ==================
    # 本区块输出内容：rolling相关性曲线、同步高风险数量、静态相关性热力图
    # 对应前端：“系统性风险可视化”页面模块

    #  1.1 统计系统性风险矩阵
    risk_class_matrix = pd.DataFrame({
        pair: pd.Series(res['y_pred_class'], index=res['y_test_reg'].index)
        for pair, res in results.items()
    }).dropna()  # Keep only dates where all pairs have data

    # 1.2 滚动相关性分析 (30天)
    window = 30
    avg_corrs = []
    for i in range(window - 1, len(risk_class_matrix)):
        sub_df = risk_class_matrix.iloc[i - window + 1:i + 1]
        corr = sub_df.corr()
        mask = ~np.eye(len(corr), dtype=bool)
        avg_corr = corr.values[mask].mean()
        avg_corrs.append(avg_corr)
    risk_class_matrix['rolling_avg_corr'] = [np.nan] * (window - 1) + avg_corrs

    # 1.3 同步高风险数量统计
    risk_class_matrix['n_high_risk'] = (risk_class_matrix == 2).sum(axis=1)

    # 1.4 静态相关系数矩阵 (热力图数据)
    static_corr_matrix = risk_class_matrix.iloc[:, :-2].corr()

    # 1.5  打包结果，供前端提取
    risk_systemic_results = {
        'risk_class_matrix': risk_class_matrix,      # 行为日期，列为币种及rolling/n_high_risk
        'static_corr_matrix': static_corr_matrix     # 各币种间风险信号相关性
    }

    print(f"\n==== Multi-currency Systemic Risk Analysis ====")
    print(f"Max 30-day rolling average correlation: {np.nanmax(risk_class_matrix['rolling_avg_corr']):.3f}")
    print(f"Max simultaneous high risk pairs: {risk_class_matrix['n_high_risk'].max()} / {len(currency_pairs)}")

    # ------ End of systemic risk analysis ------

    return models, results, risk_systemic_results

multi_pairs_file = os.path.expanduser('/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/multi/all_currency.xlsx')
print(f"Processing multi-currency data: {multi_pairs_file}")
models, results, risk_systemic_results = process_multi_currency_pairs(multi_pairs_file, all_midas_features)

# 1.4 静态相关系数矩阵 可视化
corr_matrix = risk_systemic_results['static_corr_matrix']
# Set up the heatmap background (no annotations yet)
plt.figure(figsize=(8, 6))
ax = sns.heatmap(
    corr_matrix,
    annot=False,  # Turn off default annotations
    cmap='YlGnBu',
    cbar=True,
    xticklabels=corr_matrix.columns,
    yticklabels=corr_matrix.index
)

# Loop through the DataFrame and add your own custom annotations
for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        value = corr_matrix.iloc[i, j]

        # Example: custom rule for what to annotate
        # Show blank if correlation is 1.00 (diagonal), otherwise show rounded
        text = "" if i == j else f"{value:.2f}"

        # Put text in center of heatmap cell
        ax.text(j + 0.5, i + 0.5, text,
                ha='center', va='center', color='black')

plt.title("Static Risk Signal Correlation Coefficient Matrix")
plt.savefig('Risk_signal_corr_matrix.png', dpi=300)
plt.show()

# 1.3 同步高风险数量统计 可视化
risk_systemic_results['risk_class_matrix']['n_high_risk'].plot(
    title="Daily Synchronous High Risk Currency Pair", figsize=(10, 6))
plt.savefig('daily_Synchronous_high_risk_pairs.png', dpi=300)
plt.show()

# ================== 2. 各货币对风险详情区块 begin ==================
# 本区块输出内容：每个币种回归/分类模型效果、各指标、预测走势、回测数据
# 对应前端：“各货币对风险详情”页面模块
# 2.1 汇总主要指标（以表格形式给前端）
pair_metrics = []
for pair, res in results.items():
    pair_metrics.append({
        'pair': pair,
        'rmse': res['rmse'],
        'mae': res['mae'],
        'f1': res['classification_report']['weighted avg']['f1-score'],
        # 可扩展更多字段如'backtest', 'annual_return'等
    })
# => pair_metrics 可直接传给前端表格组件
# 2.2 每币种风险信号预测走势图片

def plot_pair_risk_signal(pair, y_test_reg, y_pred_reg, output_dir):
    """
    Plot and save the predicted and true FX risk score for a single pair.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_reg.index, y_test_reg.values, label='True Risk Score', color='blue')
    plt.plot(y_pred_reg.index, y_pred_reg.values, label='Predicted Risk Score', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Risk Score')
    plt.title(f'Risk Signal Prediction: {pair}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{pair}_risk_signal_trend.png', dpi=200)
    plt.close()

# --- 代码批量保存每个币种的风险信号走势图片 ---
for pair, res in results.items():
    # 自动分配输出目录
    output_dir = os.path.expanduser(f'~/Desktop/results/{pair}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_pair_risk_signal(pair, res['y_test_reg'], res['y_pred_reg'], output_dir)


# ================== 2. 各货币对风险详情区块 end ==================

# ================== 3. 系统性共振因子分析区块 begin ==================
# 本区块输出内容：连续同步高风险区间、涉及币种、共性主导因子
# 对应前端：“系统性共振因子分析”页面模块

# 3.1 识别连续同步高风险区间 & 对应货币对


# 1) 找出所有“单独天”：哪些日期上“同步高风险货币对数量 > 3，这些天是“瞬时”高风险事件（可以不连续），是点事件集合。
high_risk_dates = risk_systemic_results['risk_class_matrix'].loc[
    risk_systemic_results['risk_class_matrix']['n_high_risk'] > 3
].index

print("Dates with synchronous high risk:")
print(high_risk_dates)

# 2) 对应货币对
pairs = [col for col in risk_systemic_results['risk_class_matrix'].columns
         if col not in ['rolling_avg_corr', 'n_high_risk']]
# For high risk dates, check which pairs are in risk class 2
sync_high_risk_details = risk_systemic_results['risk_class_matrix'].loc[high_risk_dates, pairs] == 2
# Display high risk pairs for each day
for date, row in sync_high_risk_details.iterrows():
    high_pairs = row[row].index.tolist()
    print(f"{date.date()} Synchronous high risk currency pairs: {high_pairs}")

#3）高风险区间检测： 即检测是否有连续大于7天（min_days可调）的高风险状态，输出每段连续区间的起止日期。
n_high = risk_systemic_results['risk_class_matrix']['n_high_risk']
is_high_risk = (n_high > 3).astype(int).values
# Step 2: Detect continuous intervals of 1s
min_days = 7
starts = []
ends = []
i = 0
while i < len(is_high_risk):
    if is_high_risk[i]:
        j = i
        while j < len(is_high_risk) and is_high_risk[j]:
            j += 1
        # Interval length must be at least min_days
        if j - i >= min_days:
            starts.append(n_high.index[i])
            ends.append(n_high.index[j - 1])
        i = j
    else:
        i += 1

# Step 3: Output intervals
for k in range(len(starts)):
    print(f"High risk interval {k + 1}: {starts[k].date()} to {ends[k].date()}, total { (ends[k] - starts[k]).days + 1 } days")
high_risk_periods = pd.DataFrame({'start': starts, 'end': ends})
print('high_risk_periods', high_risk_periods)

from datetime import date
# Assume high_risk_periods is your DataFrame
high_risk_periods = [
    {'start': pd.to_datetime(row['start']), 'end': pd.to_datetime(row['end'])}
    for idx, row in high_risk_periods.iterrows()
]

# def feature_importance_in_high_risk(
#         models, results, high_risk_periods,
#         pair_mode='all',  # 'all', 'best', or specify like 'USDCNH'
#         model_type='clf_model',  # 'clf_model' or 'reg_model'
#         topn=10,
#         verbose=True
# ):
#     """
#     Attribute feature importance for high risk intervals.
#     Can analyze all pairs, the best F1-score pair, or a specified pair.
#     Returns:
#         importance_dict: {pair: {period_label: DataFrame(sorted importance)}}
#     """
#     # Automatically select best F1-score pair if needed
#     if pair_mode == 'best':
#         best_pair = None
#         best_f1 = -1
#         for pair in results:
#             class_report = results[pair]['classification_report']
#             if 'weighted avg' in class_report:
#                 f1 = class_report['weighted avg']['f1-score']
#                 if f1 > best_f1:
#                     best_f1 = f1
#                     best_pair = pair
#         pair_list = [best_pair]
#         if verbose:
#             print(f"Best classified pair: {best_pair}, F1-score={best_f1:.3f}")
#     elif pair_mode == 'all':
#         pair_list = list(models.keys())
#     else:  # Specify e.g. 'USDCNH'
#         pair_list = [pair_mode]
#
#     importance_dict = {}
#
#     for pair in pair_list:
#         model = models[pair][model_type]
#         test_data = results[pair]['test_data']
#         # Auto-extract feature names
#         if hasattr(model, "get_booster"):
#             feature_names = model.get_booster().feature_names
#         else:
#             feature_names = test_data.drop(
#                 columns=['FX_risk_score', 'Open', 'High', 'Low', 'Close', 'FX_Daily_Returns', 'Future_Direction'],
#                 errors='ignore').columns.tolist()
#         importance_dict[pair] = {}
#         for period in high_risk_periods:
#             start, end = period['start'], period['end']
#             period_data = test_data.loc[(test_data.index >= start) & (test_data.index <= end), feature_names]
#             period_label = f"{start.date()}~{end.date()}"
#             if len(period_data) == 0:
#                 if verbose:
#                     print(f"{pair} has no test data in period {period_label}.")
#                 continue
#             # Try SHAP
#             expected_features = list(model.get_booster().feature_names)
#             period_data = test_data.loc[(test_data.index >= start) & (test_data.index <= end)]
#             period_data = period_data.reindex(columns=expected_features)
#             period_data.columns = period_data.columns.astype(str)
#             try:
#                 import shap
#                 explainer = shap.TreeExplainer(model)
#                 shap_values = explainer.shap_values(period_data)
#                 # For multi-class classification, average across classes
#                 if isinstance(shap_values, list):  # multi-class
#                     shap_abs_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
#                 else:
#                     shap_abs_mean = np.abs(shap_values).mean(axis=0)
#                 importance = pd.DataFrame({
#                     'feature': feature_names,
#                     'mean_abs_shap': shap_abs_mean
#                 }).sort_values(by='mean_abs_shap', ascending=False)
#                 if verbose:
#                     print(f"\n{pair} in interval {period_label} SHAP feature contribution Top{topn}:")
#                     print(importance.head(topn))
#                 importance_dict[pair][period_label] = importance.head(topn)
#             except Exception as e:
#                 print("SHAP calculation failed, falling back to global importance. Reason:", e)
#                 importances = getattr(model, 'feature_importances_', None)
#                 if importances is not None:
#                     importance = pd.DataFrame({
#                         'feature': feature_names,
#                         'importance': importances
#                     }).sort_values(by='importance', ascending=False)
#                     if verbose:
#                         print(f"\n{pair} in interval {period_label} global feature importance Top{topn}:")
#                         print(importance.head(topn))
#                     importance_dict[pair][period_label] = importance.head(topn)
#                 else:
#                     if verbose:
#                         print(f"Unable to extract feature importance for {pair} in interval {period_label}: {e}")
#             return importance_dict #v1

#v2
def feature_importance_in_high_risk(
        models, results, high_risk_periods,
        pair_mode,  # 'all', 'best', 'synchronous_high_risk', or specify e.g. 'USDCNH'
        model_type,  # 'clf_model' or 'reg_model'
        topn,
        risk_class_matrix,  # Needed for 'synchronous_high_risk' mode
        high_risk_class,
        sync_threshold,  # e.g., at least 3 pairs high risk simultaneously
        verbose
):
    """
    Attribute feature importance for high risk intervals.
    Modes:
        - 'all': Each pair individually
        - 'best': Only best F1-score pair
        - pair name: That pair only
        - 'synchronous_high_risk': Find intersection of top features among all pairs flagged as high risk
    Returns:
        importance_dict:
            For normal: {pair: {period_label: DataFrame(sorted importance)}}
            For 'synchronous_high_risk':
                {date/period: {'high_risk_pairs': [...], 'common_features': [...], 'detail': {pair: top_features}}}
    """
    import shap

    # Select pairs for analysis
    if pair_mode == 'best':
        best_pair = None
        best_f1 = -1
        for pair in results:
            class_report = results[pair]['classification_report']
            if 'weighted avg' in class_report:
                f1 = class_report['weighted avg']['f1-score']
                if f1 > best_f1:
                    best_f1 = f1
                    best_pair = pair
        pair_list = [best_pair]
        if verbose:
            print(f"Best classified pair: {best_pair}, F1-score={best_f1:.3f}")
    elif pair_mode == 'all':
        pair_list = list(models.keys())
    elif pair_mode == 'synchronous_high_risk':
        if risk_class_matrix is None:
            raise ValueError("risk_class_matrix is required for synchronous_high_risk mode")
        # 1. Identify all dates where #high risk pairs >= threshold
        sync_high_risk_dates = risk_class_matrix.index[(risk_class_matrix == high_risk_class).sum(axis=1) >= sync_threshold]
        if len(sync_high_risk_dates) == 0:
            print("No synchronous high-risk periods found.")
            return {}
        results_dict = {}
        for date in sync_high_risk_dates:
            pairs_in_risk = risk_class_matrix.columns[risk_class_matrix.loc[date] == high_risk_class].tolist()
            detail = {}
            top_features_list = []
            for pair in pairs_in_risk:
                model = models[pair][model_type]
                test_data = results[pair]['test_data']
                # Prepare input for SHAP/global feature importance
                if hasattr(model, "get_booster"):
                    feature_names = model.get_booster().feature_names
                else:
                    feature_names = test_data.drop(
                        columns=['FX_risk_score', 'Open', 'High', 'Low', 'Close', 'FX_Daily_Returns', 'Future_Direction'],
                        errors='ignore').columns.tolist()
                row_data = test_data.loc[[date]]
                row_data = row_data.reindex(columns=feature_names)
                row_data.columns = row_data.columns.astype(str)
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(row_data)
                    if isinstance(shap_values, list):  # multi-class
                        # Take the class with the highest predicted probability (for this row)
                        class_idx = shap_values[0].shape[0] == 1 and np.argmax(model.predict_proba(row_data)) or 0
                        shap_this = np.abs(shap_values[class_idx]).flatten()
                    else:
                        shap_this = np.abs(shap_values).flatten()
                    feature_scores = pd.DataFrame({
                        'feature': feature_names,
                        'shap_value': shap_this
                    }).sort_values(by='shap_value', ascending=False)
                    top_features = feature_scores['feature'].head(topn).tolist()
                except Exception as e:
                    # Fallback: use global importance if SHAP fails
                    print(f"SHAP failed for {pair} on {date}: {e}")
                    importances = getattr(model, 'feature_importances_', None)
                    if importances is not None:
                        feature_scores = pd.DataFrame({
                            'feature': feature_names,
                            'importance': importances
                        }).sort_values(by='importance', ascending=False)
                        top_features = feature_scores['feature'].head(topn).tolist()
                    else:
                        top_features = []
                detail[pair] = top_features
                top_features_list.append(set(top_features))
            # Find intersection: features that are top for **all** pairs in risk
            if len(top_features_list) > 1:
                common_features = set.intersection(*top_features_list)
            elif len(top_features_list) == 1:
                common_features = top_features_list[0]
            else:
                common_features = set()
            results_dict[date] = {
                'high_risk_pairs': pairs_in_risk,
                'common_features': list(common_features),
                'detail': detail
            }
            if verbose:
                print(f"\n[Date: {date.date()}] Synchronous high risk in pairs: {pairs_in_risk}")
                print("Common Top Features:", list(common_features) if common_features else "(none)")
                for p, feats in detail.items():
                    print(f" - {p}: {feats}")
        return results_dict
    else:
        pair_list = [pair_mode]

    # -- Original: for 'all' or single-pair mode --
    importance_dict = {}
    for pair in pair_list:
        model = models[pair][model_type]
        test_data = results[pair]['test_data']
        if hasattr(model, "get_booster"):
            feature_names = model.get_booster().feature_names
        else:
            feature_names = test_data.drop(
                columns=['FX_risk_score', 'Open', 'High', 'Low', 'Close', 'FX_Daily_Returns', 'Future_Direction'],
                errors='ignore').columns.tolist()
        importance_dict[pair] = {}
        for period in high_risk_periods:
            start, end = period['start'], period['end']
            period_data = test_data.loc[(test_data.index >= start) & (test_data.index <= end), feature_names]
            period_label = f"{start.date()}~{end.date()}"
            if len(period_data) == 0:
                if verbose:
                    print(f"{pair} has no test data in period {period_label}.")
                continue
            # SHAP logic
            period_data = test_data.loc[(test_data.index >= start) & (test_data.index <= end)]
            period_data = period_data.reindex(columns=feature_names)
            period_data.columns = period_data.columns.astype(str)
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(period_data)
                if isinstance(shap_values, list):
                    shap_abs_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                else:
                    shap_abs_mean = np.abs(shap_values).mean(axis=0)
                importance = pd.DataFrame({
                    'feature': feature_names,
                    'mean_abs_shap': shap_abs_mean
                }).sort_values(by='mean_abs_shap', ascending=False)
                if verbose:
                    print(f"\n{pair} in interval {period_label} SHAP feature contribution Top{topn}:")
                    print(importance.head(topn))
                importance_dict[pair][period_label] = importance.head(topn)
            except Exception as e:
                print("SHAP calculation failed, falling back to global importance. Reason:", e)
                importances = getattr(model, 'feature_importances_', None)
                if importances is not None:
                    importance = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values(by='importance', ascending=False)
                    if verbose:
                        print(f"\n{pair} in interval {period_label} global feature importance Top{topn}:")
                        print(importance.head(topn))
                    importance_dict[pair][period_label] = importance.head(topn)
                else:
                    if verbose:
                        print(f"Unable to extract feature importance for {pair} in interval {period_label}: {e}")
    return importance_dict


# for v1-feature importance
# feature_importance_in_high_risk(models, results, high_risk_periods, pair_mode='all')
# feature_importance_in_high_risk(models, results, high_risk_periods, pair_mode='best')
# feature_importance_in_high_risk(models, results, high_risk_periods, pair_mode='USDCNH')
# imp_dict = feature_importance_in_high_risk(models, results, high_risk_periods, pair_mode='all', verbose=False)

# for v2-feature importance
#
pair_cols = [col for col in risk_systemic_results['risk_class_matrix'].columns if col not in ['rolling_avg_corr', 'n_high_risk']]
risk_class_matrix = risk_systemic_results['risk_class_matrix'][pair_cols]
common_features = feature_importance_in_high_risk(
    models,
    results,
    high_risk_periods, 'synchronous_high_risk', 'clf_model',
    10, risk_class_matrix, 2, 4,  True
)


# ================== 3. 系统性共振因子分析区块 end ==================


# ================== 4. 多货币对投资组合权重优化   ==================

# --- 1. 构造对齐的收益率矩阵（每列为一个货币对，行对应相同日期，去除缺失）---
# (可用于多货币投资收益展示)
return_matrix = pd.DataFrame({
    pair: results[pair]['test_data']['FX_Daily_Returns']
    for pair in results
}).dropna()


# 2. 构造风险信号分类矩阵（与return_matrix对齐，便于结合风险信号进行权重调整）---
# # (用于风控版权重优化)
risk_class_matrix = risk_systemic_results['risk_class_matrix'][list(results.keys())]


# --- 3. 经典/风险过滤版投资组合权重优化函数 ---
from scipy.optimize import minimize

def optimize_portfolio(
    return_matrix,
    risk_class_matrix=None,
    method='markowitz',  # 'markowitz' or 'risk_filtered'
    risk_free_rate=0.0,
    allow_short=False,
    max_risk_weight=0.0,  # Maximum weight for high risk (0 means exclude)
    verbose=True
):

    data = return_matrix.dropna()
    if risk_class_matrix is not None:
        risk_class_matrix = risk_class_matrix.loc[data.index]

    mu = data.mean().values
    cov = data.cov().values
    n = data.shape[1]
    bounds = [(-1.0 if allow_short else 0.0, 1.0) for _ in range(n)]
    cons = ({'type': 'eq', 'fun': lambda w: w.sum() - 1})

    def neg_sharpe(w):
        port_return = np.dot(mu, w)
        port_std = np.sqrt(np.dot(w, np.dot(cov, w)))
        sharpe = (port_return - risk_free_rate) / (port_std + 1e-8)
        return -sharpe

    if method == 'markowitz':
        opt = minimize(neg_sharpe, np.ones(n)/n, bounds=bounds, constraints=cons)
        w_opt = opt.x

    elif method == 'risk_filtered':
        risk_signal = (risk_class_matrix == 2).sum() / len(risk_class_matrix)
        risk_mask = (risk_signal > 0.3)  # Example: exclude if >30% high risk
        effective_bounds = []
        for i in range(n):
            if risk_mask.iloc[i]:
                effective_bounds.append((0.0, max_risk_weight))
            else:
                effective_bounds.append(bounds[i])
        opt = minimize(neg_sharpe, np.ones(n)/n, bounds=effective_bounds, constraints=cons)
        w_opt = opt.x
    else:
        raise ValueError("method should be 'markowitz' or 'risk_filtered'")

    if verbose:
        print("优化后权重 (Optimized weights):", dict(zip(data.columns, w_opt)))
        port_return = np.dot(mu, w_opt)
        port_std = np.sqrt(np.dot(w_opt, np.dot(cov, w_opt)))
        print(f"组合预期收益 (Expected Return): {port_return:.4f}, "
              f"预期波动率 (Volatility): {port_std:.4f}, "
              f"夏普比率 (Sharpe Ratio): {(port_return - risk_free_rate) / (port_std + 1e-8):.4f}")
    return pd.Series(w_opt, index=data.columns)

# --- 4. 示例：调用权重优化函数，输出结果（可接入前端“智能资产配置/一键权重建议”模块）---
weights = optimize_portfolio(
    return_matrix,
    risk_class_matrix=risk_class_matrix,
    method='risk_filtered',
    # method='markowitz',
    max_risk_weight=0.05
)



# ===================== previous Main Script Block =====================
# from original_functions import  *
# # from original_functions import *
# if __name__ == "__main__":
#     # ... (your earlier data prep, feature engineering, model training)
#
#     # 1. Display dashboard and technical indicators for the main test set (single currency example)
#     display_dashboard(test_data, y_pred_reg, y_pred_class)
#     display_technical_indicators(test_data)
#
#     # 2. If predictions are available, ensure you have updated them
#     y_pred_reg = reg_model.predict(X_test)
#     y_pred_class = clf_model.predict(X_test)
#
#     # 3. Multi-currency pipeline
#     multi_pairs_file = os.path.expanduser(
#         '/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/multi/all_currency.xlsx'
#     )
#     print(f"Processing multi-currency data: {multi_pairs_file}")
#     models, results, risk_systemic_results = process_multi_currency_pairs(multi_pairs_file, all_midas_features)
#
#     # 4. Example: manually assign portfolio weights (or use optimizer results)
#     weights = {
#         'USDJPY': 0.15,
#         'USDCNH': 0.25,
#         'JPYEUR': 0.10,
#         'JPYCNY': 0.10,
#         'EURUSD': 0.20,
#         'EURJPY': 0.10,
#         'EURCNH': 0.10
#     }
#
#     # 5. Analyze risk distribution by currency pair (function must be defined elsewhere)
#     risk_distribution, weighted_risk = analyze_currency_pair_risk_distribution(results, weights)
#
#     # 6. Print risk distribution results
#     print("\nRisk distribution for each currency pair:")
#     for pair, dist in risk_distribution.items():
#         print(
#             f"{pair} - Low risk: {dist['low_risk_pct']:.2f}%, Medium risk: {dist['medium_risk_pct']:.2f}%, "
#             f"High risk: {dist['high_risk_pct']:.2f}%, Weight: {dist['weight'] * 100:.2f}%"
#         )
#
#     print("\nWeighted risk distribution:")
#     print(
#         f"Low risk: {weighted_risk['low_risk']:.2f}%, Medium risk: {weighted_risk['medium_risk']:.2f}%, "
#         f"High risk: {weighted_risk['high_risk']:.2f}%"
#     )
#
#     # 7. Visualize risk distribution (file path is an example)
#     visualize_risk_distribution(
#         risk_distribution,
#         weighted_risk,
#         os.path.expanduser(
#             '/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/results/risk_distribution.png'
#         )
#     )
#
#     # 8. Print evaluation summary
#     print("\nMulti-currency FX risk warning model evaluation summary:")
#     for pair, res in results.items():
#         print(
#             f"{pair} - RMSE: {res['rmse']:.4f}, MAE: {res['mae']:.4f}, "
#             f"F1 (weighted): {res['classification_report']['weighted avg']['f1-score']:.4f}"
#         )
#
#     # 9. If USDCNH is present, perform and print detailed backtest analysis
#     if 'USDCNH' in results and 'backtest_results' in results['USDCNH']:
#         print("\nUSDCNH currency pair backtest analysis")
#
#         usdcnh_results = results['USDCNH']
#         usdcnh_backtest = usdcnh_results['backtest_results']
#
#         print("\nUSDCNH best strategy backtest statistics:")
#         print(f"Strategy type: {usdcnh_backtest['strategy']}")
#         print(f"Total trades: {usdcnh_backtest['total_trades']}")
#         print(f"Win rate: {usdcnh_backtest['win_rate']:.2f}%")
#         print(f"Average return: {usdcnh_backtest['avg_return']:.2f}%")
#         print(f"Annualized return: {usdcnh_backtest['annual_return']:.2f}%")
#         print(f"Max drawdown: {usdcnh_backtest['max_drawdown']:.2f}%")
#         print(f"Sharpe ratio: {usdcnh_backtest['sharpe_ratio']:.2f}")
#
#         # Profit/loss ratio analysis
#         trades = usdcnh_backtest.get('trades', pd.DataFrame())
#         if len(trades) > 0 and 'pnl_pct' in trades:
#             profit_trades = trades[trades['pnl_pct'] > 0]
#             loss_trades = trades[trades['pnl_pct'] <= 0]
#             if len(profit_trades) > 0 and len(loss_trades) > 0:
#                 avg_profit = profit_trades['pnl_pct'].mean() * 100
#                 avg_loss = loss_trades['pnl_pct'].mean() * 100
#                 profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
#                 print(f"Average profit: {avg_profit:.2f}%")
#                 print(f"Average loss: {avg_loss:.2f}%")
#                 print(f"Profit/loss ratio: {profit_loss_ratio:.2f}")
#
#         # Show top 5 trade details
#         if len(trades) > 0:
#             print("\nTop 5 USDCNH trades:")
#             trade_display = trades.head()
#             if 'pnl_pct' in trade_display:
#                 trade_display['pnl_pct'] = trade_display['pnl_pct'] * 100
#             print(trade_display)

