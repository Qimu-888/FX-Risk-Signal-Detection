from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def apply_scaler(dataframe, columns, scaler_type, feature_range=(0, 1)):
    if scaler_type == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("scaler_type must be either 'minmax' or 'standard'")
    dataframe[columns] = scaler.fit_transform(dataframe[columns])
    return dataframe


'''Moving Average'''
def add_MAs_for_columns(data, column_names, window_sizes):
    for column_name in column_names:
        for window_size in window_sizes:
            moving_average_column_name = f"{column_name}_MA_{window_size}"
            data[moving_average_column_name] = data[column_name].rolling(window=window_size).mean()
    return data


'''Adjust the width of column'''
def save_adjust_column_width(data, file_name, index_set=True):
    # data.to_csv(file_name, index=index_set)
    data.to_excel(file_name, index=index_set)

    from openpyxl import load_workbook
    #adjust the width of column
    workbook = load_workbook(file_name)
    sheet = workbook.active
    for column in sheet.columns:
        max_length = 0
        column_letter = column[0].column_letter  # 获取列字母
        for cell in column:
            if cell.value:
                max_length = max(max_length, len(str(cell.value)))
        adjusted_width = max_length + 2
        sheet.column_dimensions[column_letter].width = adjusted_width
    workbook.save(file_name)




def compute_monthly_avg(df, columns, date_col):
    # Compute the monthly average for the specified columns
    # Ensure the date column is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])

    # Set the date column as the index for resampling
    df.set_index(date_col, inplace=True)
    monthly_avg = df[columns].resample('M').mean()

    # Reset index and format the 'Month' column
    monthly_avg_df = monthly_avg.reset_index()
    monthly_avg_df.rename(columns={date_col: 'Month'}, inplace=True)
    monthly_avg_df['Month'] = monthly_avg_df['Month'].dt.to_period('M').astype(str)
    monthly_avg_df.rename(columns={col: f"{col}_monthly_avg" for col in columns}, inplace=True)

    return monthly_avg_df



def merge_all_features_with_fx(fx_monthly, folder_path, output_file):
    files = os.listdir(folder_path)

    merged_df = fx_monthly.copy()
    for file in files:
        if file.endswith('.xlsx'):
            file_path = os.path.join(folder_path, file)
            print(f"Processing file: {file}")

            df = pd.read_excel(file_path)

            # drop the 1st col
            df.drop(columns=df.columns[0], inplace=True)

            merged_df = merged_df.merge(df, left_index=True, right_index=True, how='left')

    #save the final file
    save_adjust_column_width(merged_df, output_file, False)
    print(f"Merged data saved to {output_file}")
    return merged_df


def merge_selected_folders_with_fx(fx_monthly, folder_paths, output_file):
    """
    Merges data from selected folders with the fx_monthly DataFrame.

    Parameters:
    fx_monthly (pd.DataFrame): The main DataFrame to merge with.
    folder_paths (list): List of folder paths containing Excel files.
    output_file (str): Path to save the final merged file.

    Returns:
    pd.DataFrame: The merged DataFrame.
    """
    merged_df = fx_monthly.copy()
    total_files = sum(len(os.listdir(folder)) for folder in folder_paths)
    from tqdm.auto import tqdm  # Works better in some environments

    with tqdm(total=total_files, desc="Processing Files", unit="file") as pbar:
        for folder_path in folder_paths:
            files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
            for file in files:
                file_path = os.path.join(folder_path, file)
                print(f"Processing file: {file} in folder: {folder_path}")

                df = pd.read_excel(file_path)
                df.drop(columns=df.columns[0], inplace=True)  # Drop first column

                merged_df = merged_df.merge(df, left_index=True, right_index=True, how='left')
                pbar.update(1)  # Update progress bar

    # Save the final file
    save_adjust_column_width(merged_df, output_file, False)
    print(f"Merged data saved to {output_file}")
    return merged_df




def compute_multiple_rolling_correlation(data, fx_column, feature_cols, window=3,
                                         plot_individual=True, plot_combined=True):
    """
    计算 FX 波动率与多个宏观经济变量的 Rolling Correlation，并绘制单独 & 组合图。

    参数:
    - data (pd.DataFrame): 包含 FX 波动率和宏观经济变量的数据集
    - fx_column (str): FX 波动率的列名
    - feature_cols (list): 需要计算 rolling correlation 的宏观经济变量列名列表
    - window (int): Rolling 窗口大小，默认 3（单位与数据一致，如月度数据则为 3 个月）
    - plot_individual (bool): 是否单独绘制每个 Rolling Correlation 图（默认 True）
    - plot_combined (bool): 是否在一张图上绘制所有特征的 Rolling Correlation（默认 True）

    返回:
    - pd.DataFrame: 包含 Rolling Correlation 计算结果的 DataFrame
    """
    # 确保数据按时间排序
    # data = data.sort_values("Date").copy()

    # 初始化 DataFrame 存储 Rolling Correlation 计算结果
    result_df = data[["Date"]].copy()

    # 颜色列表（如果 features 多，可以自动扩展）
    colors = ["b", "g", "r", "c", "m", "y", "k", "purple", "orange", "brown"]

    # 检查 feature_cols 是否是列表
    if not isinstance(feature_cols, list):
        raise ValueError("feature_cols 参数必须是一个包含多个列名的列表")

    # 计算并绘制每个宏观经济变量的 Rolling Correlation
    for i, feature_col in enumerate(feature_cols):
        # 计算 Rolling Correlation
        rolling_corr = data[fx_column].rolling(window=window).corr(data[feature_col])

        # 存储计算结果
        col_name = f"Rolling_Corr_{fx_column}_{feature_col}"
        result_df[col_name] = rolling_corr

        # 单独绘制每个 Rolling Correlation 图
        if plot_individual:
            plt.figure(figsize=(30, 12))
            plt.plot(data["Date"], rolling_corr, label=f"{fx_column} vs {feature_col}", color=colors[i % len(colors)])
            plt.xticks(rotation=45)  # 45度倾斜显示

            plt.axhline(y=0, color="black", linestyle="--")  # 零基准线
            plt.xlabel("Date")
            plt.ylabel("Rolling Correlation")
            plt.title(f"Rolling Correlation Between {fx_column} and {feature_col}")
            plt.legend()
            plt.show()

    # 在一张图上绘制所有 Rolling Correlation 结果
    if plot_combined:
        plt.figure(figsize=(40, 15))
        for i, feature_col in enumerate(feature_cols):
            plt.plot(data["Date"], result_df[f"Rolling_Corr_{fx_column}_{feature_col}"],
                     label=f"{fx_column} vs {feature_col}", color=colors[i % len(colors)])
            plt.xticks(rotation=45)  # 45度倾斜显示

        plt.axhline(y=0, color="black", linestyle="--")  # 零基准线
        plt.xlabel("Date")
        plt.ylabel("Rolling Correlation")
        plt.title(f"Rolling Correlation of {fx_column} with Multiple Factors")
        plt.legend()
        plt.show()

    return result_df


import shap

# import numpy as np
# import pandas as pd
import matplotlib.ticker as mticker


def plot_shap_summary_and_bar(shap_values, X, save_prefix="shap_importance", figsize=(50, 15)):
    """
    绘制 SHAP Summary Plot 和 Bar Plot
    参数:
    - shap_values: SHAP 计算结果 (shap.Explanation)
    - X: 训练或测试数据 (DataFrame)
    - save_prefix: 保存文件的前缀
    - figsize: 图像大小 (默认 30x12)
    """

    # 1️⃣ 转换 SHAP 值为 DataFrame
    shap_df = pd.DataFrame(shap_values.values, columns=X.columns)

    # 2️⃣ 计算 SHAP 重要性（按特征求平均绝对值）
    shap_importance = shap_df.abs().mean().sort_values(ascending=False)
    feature_names = shap_importance.index

    # 3️⃣ 获取 SHAP 值的颜色映射（红色：正，蓝色：负）
    shap_colors = np.where(shap_df.mean().loc[feature_names] > 0, "red", "blue")

    # 4️⃣ 创建 Summary Plot（散点分布）
    fig, ax = plt.subplots(figsize=figsize)
    for i, feature in enumerate(feature_names):
        values = shap_df[feature].values
        y_positions = np.full_like(values, i)
        colors = np.where(values > 0, "red", "blue")
        plt.scatter(values, y_positions, alpha=0.6, color=colors, s=20)

    # 5️⃣ X 轴使用科学计数法
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.xaxis.offsetText.set_fontsize(12)

    # 6️⃣ 设定 Y 轴特征名称
    plt.yticks(range(len(feature_names)), feature_names, fontsize=14)

    # 7️⃣ 设定 X 轴和标题
    plt.xlabel("SHAP Value Impact", fontsize=16)
    plt.ylabel("Feature", fontsize=16)
    plt.title("SHAP Summary Plot", fontsize=18)
    plt.grid(alpha=0.5, linestyle="--")

    # 8️⃣ 保存 Summary Plot
    plt.savefig(f'results/{save_prefix}_summary.png', dpi=300)
    plt.show()

    # === 画 Bar Plot ===
    # 1️⃣ 提取 SHAP 重要性并排序
    shap_importance = abs(shap_values.values).mean(axis=0)
    shap_summary = sorted(zip(shap_importance, X.columns), reverse=True)

    # 2️⃣ 创建 Bar Plot
    fig, ax = plt.subplots(figsize=figsize)
    importance_values, feature_labels = zip(*shap_summary)
    ax.barh(feature_labels, importance_values, color="red")

    # 3️⃣ X 轴使用科学计数法
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.xaxis.offsetText.set_fontsize(12)

    # 4️⃣ 设定 X 轴和标题
    plt.xlabel("Mean(|SHAP Value|)", fontsize=16)
    plt.ylabel("Features", fontsize=16)
    plt.title("SHAP Feature Importance", fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.5)

    # 5️⃣ 保存 Bar Plot
    plt.savefig(f'results/{save_prefix}_bar.png', dpi=300)
    plt.show()

def plot_corr_feature(corr_matrix, title, file_name):
    plt.figure(figsize=(40, 35))
    ax = sns.heatmap(corr_matrix, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix.columns)):
            text = f"{corr_matrix.iloc[i, j]:.2f}"  # 取2位小数
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', fontsize=12, color="black")
    plt.title(title)
    plt.savefig(file_name, dpi= 300)
    plt.show()

def create_lag_features(df, feature_list, lags):
    """
    为选定的特征创建滞后特征 (Lag Features)
    参数:
    - df: DataFrame, 需要添加滞后特征的数据集
    - feature_list: List, 需要添加滞后特征的列名
    - lags: int, 滞后阶数（默认3，表示添加 t-1, t-2, t-3）

    返回:
    - df_lagged: DataFrame, 含有滞后特征的新数据集
    """
    df_lagged = df.copy()
    for feature in feature_list:
        for lag in range(1, lags + 1):
            df_lagged[f"{feature}_lag{lag}"] = df_lagged[feature].shift(lag)

    # 处理缺失值（由于 shift 操作，前 n 行会有 NaN）
    df_lagged = df_lagged.dropna()

    return df_lagged
#
#
# lags = 2
# X_lagged = create_lag_features(X, important_features, lags)
#
# # 让 y 也去掉前 `lags` 个数据点
# y_aligned = y.iloc[lags:].reset_index(drop=True)
# X_aligned = X_lagged.reset_index(drop=True)  # 确保索引对齐
#
# # 重新赋值
# X = X_aligned
# y = y_aligned
#
# print(f"X shape: {X.shape}, y shape: {y.shape}")


def create_lagged_target(y, lags):
    """
    为目标变量 `y` 生成滞后特征
    参数：
    - y: 目标变量 (Series)
    - lags: 需要创建的滞后步数
    返回：
    - lagged_y: DataFrame，包含 `y` 和滞后特征
    """
    df = pd.DataFrame(y.copy())

    for lag in range(1, lags + 1):
        df[f"y_lag{lag}"] = df[y.name].shift(lag)

    return df.dropna()  # 删除 NaN 行（前 lags 行会有缺失值）



def risk_score_definitions(data, option='roll_val', window_size=20):
    n = 5
    data["FX_Daily_Returns"] = data["Close"].pct_change()
    data['Future_Direction'] = np.sign(data['FX_Daily_Returns'].shift(-n))

    if option == 'roll_val':
        data['FX_risk_score'] = data["FX_Daily_Returns"].rolling(window_size).std().shift(-n)
        # data['FX_risk_score'] = data['FX_Daily_Returns'].shift(-n).rolling(window=n).std()
        # data['FX_risk_score'] = data['Close'].shift(-n)

    if option == 'extreme':#z-score
        # data['fx_abs_return'] = data["FX_Daily_Returns"].abs()
        data['FX_risk_score'] = ((data["FX_Daily_Returns"].abs()- data["FX_Daily_Returns"].abs().rolling(window_size).mean()) /
                                 data["FX_Daily_Returns"].abs().rolling(window_size).std())
    return data


# def midas_transform(data, quarterly_vars, monthly_vars, omega1=1.5, omega2=2.5):
def midas_transform(month_data,qr_data, monthly_vars, quarterly_vars, type_monthly, type_quarterly, omega1=1.5, omega2=2.5, gamma=1.3,
                    lag_month = 6, lag_quarter = 4):

    # Ensure the index is datetime
    if not isinstance(month_data.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")

    def weights_func(lags, omega1, omega2, gamma, type):
        weights = None
        if type == 'single':
            k_values = np.arange(1, lags + 1)  # Lag indices (1 to max lags)
            # Compute the numerator using the new one-parameter formula
            numerator = (1 - k_values / lags) ** (gamma - 1)
            # Compute the denominator (sum of all numerators)
            denominator = np.sum(numerator)

            # Compute final normalized weights
            weights = numerator / denominator

        elif type == 'multi':
            # k_values = np.arange(1, lags + 1)  # Lag indices (1 to max lags)
            # # Compute the numerator for each k
            # numerator = (k_values / (lags + 1)) ** (omega1 - 1) * (1 - k_values / (lags + 1)) ** (omega2 - 1)
            # # Compute the denominator (sum of all numerators)
            # denominator = np.sum(numerator)
            # weights = numerator / denominator
            #     """Generate Beta Polynomial Weights for MIDAS Regression."""
            j = np.arange(1, lags + 1)  # Lag indices (1 to max lags)
            weights = (j ** omega1) * ((lags - j + 1) ** omega2)  # Beta polynomial formula
            weights = weights / weights.sum()  # Normalize weights
        return weights

    # Generate Weights for Quarterly and Monthly Variables
    quarterly_weights = weights_func(lag_quarter, omega1, omega2, gamma, type_quarterly)  # 12-quarter lags
    monthly_weights = weights_func(lag_month, omega1, omega2, gamma, type_monthly)  # 6-month lags

    # Manually Create Lagged Features (12 Lags for Quarterly, 6 Lags for Monthly)
    for col in quarterly_vars:
        for lag in range(1, lag_quarter+1):  # 4-quarter lags
            qr_data[f"{col}_lag_{lag}"] = qr_data[col].shift(lag * 90)  # Convert quarter to days

    for col in monthly_vars:
        for lag in range(1, lag_month+1):  # 6-month lags
            month_data[f"{col}_lag_{lag}"] = month_data[col].shift(lag * 30)  # Convert months to days

    # Apply MIDAS Weighting
    midas_features = pd.DataFrame(index=month_data.index)  # Store transformed features

    for col in quarterly_vars:
        weighted_sum_qr = sum(qr_data[f"{col}_lag_{lag}"] * quarterly_weights[lag - 1] for lag in range(1, lag_quarter+1))
        midas_features[f"midas_{col}_signal"] = weighted_sum_qr

    for col in monthly_vars:
        weighted_sum_month = sum(month_data[f"{col}_lag_{lag}"] * monthly_weights[lag - 1] for lag in range(1, lag_month+1))
        midas_features[f"midas_{col}_signal"] = weighted_sum_month

    print("✅ MIDAS transformation completed successfully!")
    midas_features.index = pd.to_datetime(midas_features.index)  # 确保索引是 datetime 格式
    midas_features.index.name = "Date"  # 设置索引名称为 Date

    midas_features = midas_features.dropna()
    return midas_features


def plot_real_pred_reg(y_real, y_pred, filename):
    plt.figure(figsize=(20, 8))
    plt.plot(y_real, label='True Values')
    plt.plot(y_pred, label='Predicted Values')
    plt.title("True vs Predicted Values")
    plt.xlabel("Samples")
    plt.ylabel("FX_risk_score")
    plt.legend()
    plt.savefig(filename, dpi = 300)
    plt.show()

def visualize_detection_class(y_real_reg, y_pred_reg, y_real_class, y_pred_class, threshold_type, CI_plot, class2_plot, value_threshold,quantile_threshold,
                              filename_threshold, filename_CI, filename_class):
    threshold_volatility = None
    if threshold_type == 'value':
        threshold_volatility = value_threshold
    elif threshold_type == 'quantile':
        threshold_volatility = np.quantile(y_pred_reg, quantile_threshold)

    # 筛选出预测波动大于阈值的节点
    high_risk_dates = y_pred_reg[y_pred_reg > threshold_volatility].index

    # 绘制真实值与预测值的对比图
    plt.figure(figsize=(20, 8))
    plt.plot(y_real_reg.index, y_real_reg, label='True Values', color='blue', alpha=0.7)
    plt.plot(y_pred_reg.index, y_pred_reg, label='Predicted Values', color='orange', alpha=0.7)

    # 标记预测波动大于阈值的节点
    for date in high_risk_dates:
        plt.axvline(x=date, color='red', linestyle='--', alpha=0.5)
        plt.scatter(date, y_real_reg.loc[date], color='red', zorder=5)
        # plt.text(date, y_test_reg.loc[date], f'True: {y_test_reg.loc[date]:.4f}\nPred: {y_pred_reg.loc[date]:.4f}',
        #          fontsize=9, ha='right', va='bottom', color='red')

    plt.title("True vs Predicted FX Risk Score with Threshold Annotations")
    plt.xlabel("Date")
    plt.ylabel("FX Risk Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename_threshold, dpi = 300)
    plt.show()

    if CI_plot == True:
        # 计算预测值的均值和标准差
        pred_mean = np.mean(y_pred_reg)
        pred_std = np.std(y_pred_reg)

        # 计算 95% 置信区间
        ci_lower = pred_mean - 1.96 * pred_std  # 下限
        ci_upper = pred_mean + 1.96 * pred_std  # 上限

        # 筛选出预测值在 95% CI 之外的异常值
        outliers_dates = y_pred_reg[(y_pred_reg < ci_lower) | (y_pred_reg > ci_upper)].index
        # 绘制真实值与预测值的对比图
        plt.figure(figsize=(20, 8))
        plt.plot(y_real_reg.index, y_real_reg, label='True Values', color='blue', alpha=0.7)
        plt.plot(y_pred_reg.index, y_pred_reg, label='Predicted Values', color='orange', alpha=0.7)

        # 标记预测值在 95% CI 之外的异常值
        for date in outliers_dates:
            plt.axvline(x=date, color='red', linestyle='--', alpha=0.5)
            plt.scatter(date, y_real_reg.loc[date], color='red', zorder=5)
            # plt.text(date, y_test_reg.loc[date],
            #          f'True: {y_test_reg.loc[date]:.4f}\nPred: {y_pred_reg.loc[date]:.4f}',
            #          fontsize=9, ha='right', va='bottom', color='red')

        # 添加 95% 置信区间线
        plt.axhline(y=ci_lower, color='green', linestyle='--', label='95% CI Lower Bound')
        plt.axhline(y=ci_upper, color='green', linestyle='--', label='95% CI Upper Bound')

        plt.title("True vs Predicted FX Risk Score with 95% CI Outliers Annotations")
        plt.xlabel("Date")
        plt.ylabel("FX Risk Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename_CI, dpi = 300)
        plt.show()

    if class2_plot == True:
        # 筛选出高风险节点（y_class = 2）
        # 将 y_class 转换为 Series（如果它还不是 Series）

        # 将分类结果转换为 Series
        y_pred_class_series = pd.Series(y_pred_class, index=y_pred_reg.index)
        y_test_class_series = pd.Series(y_real_class, index=y_real_reg.index)

        # 筛选出高风险节点
        high_risk_dates_pred = y_pred_class_series[y_pred_class_series == 2].index  # 预测的高风险节点
        high_risk_dates_real = y_test_class_series[y_test_class_series == 2].index  # 真实的高风险节点
        print("Number of high risk real points:", len(high_risk_dates_real))

        # 绘制真实值与预测值的对比图
        plt.figure(figsize=(20, 8))
        plt.plot(y_real_reg.index, y_real_reg, label='True Values', color='blue', alpha=0.7)
        plt.plot(y_pred_reg.index, y_pred_reg, label='Predicted Values', color='orange', alpha=0.7)

        # 标记真实的高风险节点（黑点）
        plt.scatter(high_risk_dates_real, y_real_reg.loc[high_risk_dates_real],
                    color='black', label='High Risk (Real)', zorder=5)
        # plt.scatter(high_risk_dates_real, y_real_reg.loc[high_risk_dates_real], color='red', zorder=5)

        # 标记预测的高风险节点（竖线）
        for date in high_risk_dates_pred:
            plt.axvline(x=date, color='red', linestyle='--', alpha=0.5,
                        label='High Risk (Predicted)' if date == high_risk_dates_pred[0] else "")  # 竖线


        plt.title("True vs Predicted FX Risk Score with Class Annotations")
        plt.xlabel("Date")
        plt.ylabel("FX Risk Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename_class, dpi = 300)
        plt.show()







