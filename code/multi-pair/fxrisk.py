from process_plot_data import *
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier


from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_log_error
import xgboost as xgb
# import sys
# print(sys.path)

# from midaspy.iolib import *

# import MIDASpy as mp
# print(dir(mp))  # Check available attributes

# print(MIDASpy.__file__)

# Load data
fx_daily = pd.read_excel('/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/usd_cnh_2015.xlsx', parse_dates=["Date"])

fx_daily = risk_score_definitions(fx_daily)
us_cpu = pd.read_excel('/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/features/subEPU/US-CPU_2015.xlsx')
window_size = 10

# FX calculations
#daily
daily_folders = ['/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/features/GPR', '/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/features/other_assets',
                 '/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/features/sentiment_score', '/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/features/IR']
daily_index = pd.read_excel('/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/index_daily.xlsx')
merged_df_daily = merge_selected_folders_with_fx(daily_index, daily_folders, '/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/features/merged/merged_daily_data.xlsx')
daily_vars = ['GPRD','GPRD_ACT','GPRD_THREAT','USD-Gold','CNY-Gold','Shanghai_Gold_Daily_Volatility','Oil_Price','oil-Daily-Volatility',
              'sentiment_score','Difference_CNY_CNH_IR','Difference_USD_CNH_IR']
merged_df_daily = merged_df_daily[daily_vars]
merged_df_daily.to_csv('merged_daily_data.csv', index= False)

'''2-for monthly：EPU,subEPU, FER, currency_suppy, and quarterly data'''
monthly_folders = ['/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/features/CPI', '/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/features/EPU', "/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/features/subEPU",
                   "/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/features/BoP", '/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/features/employment', '/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/features/FER',
                   '/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/features/currency_supply']
merged_df_month = merge_selected_folders_with_fx(us_cpu, monthly_folders, '/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/features/merged/merged_monthly_data.xlsx')
# Define monthly variables
monthly_vars = [
    'US-CPU_x', 'US_CPI_NSA_YoY_Monthly', 'CN-CPI_YoY_Monthly', 'US-TPU', 'CN-EPU', 'Monthly_HKEPU_Index', 'CN-TPU', 'GEPU_ppp', 
    'US-EPU-News_Based_Index', 'US-MPU_Access_World_News', 'CA-US', 'CA-CN', 'US_EMPLOYMENT_RATIO', 'FER-Growth_Rate-US', 
    'FER-Growth_Rate-CN', 'CN-Currency_supply_M2_millionUSD', 'US-Currency_supply_M2_millionUSD'
]

# Convert 'date' column to datetime and set as index
merged_df_month['date'] = pd.to_datetime(merged_df_month['date'])

merged_df_month = merged_df_month.set_index('date')

new_index = pd.date_range(start=merged_df_month.index.min(), end='2024-12-31', freq='D')
# Resample to daily frequency and forward fill
merged_df_month = merged_df_month.reindex(new_index).ffill()
merged_df_month = merged_df_month[monthly_vars]


qr_folders = ['/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/features/GDP']
qr_index = pd.read_excel('/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/qr_index.xlsx')
merged_df_qr = merge_selected_folders_with_fx(qr_index, qr_folders, '/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/features/merged/merged_qr_data.xlsx')
merged_df_qr['Date'] = pd.to_datetime(merged_df_qr['Date'])
merged_df_qr = merged_df_qr.set_index('Date')

qr_vars = ['GDP_YoY_Quarterly_China', 'GDP_YoY_Quarterly_US']
merged_df_qr = merged_df_qr[qr_vars]

new_index_qr = pd.date_range(start=merged_df_qr.index.min(), end='2024-12-31', freq='D')
merged_df_qr = merged_df_qr.reindex(new_index_qr).ffill()
merged_df_qr.to_csv('merged_qr_data_test.csv', index= False)

midas_features = midas_transform(merged_df_month, merged_df_qr, monthly_vars, qr_vars,
                                 'multi','single',1.5,2.5,1.5,
                                 6,4)

midas_features.to_csv('midas_features.csv', index= True)


# fx_daily.index = pd.to_datetime(fx_daily.index)
# merged_df_daily.index = pd.to_datetime(merged_df_daily.index)


data = fx_daily.merge(merged_df_daily, left_index=True, right_index=True, how='left')

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
# midas_features.drop(columns=midas_features.columns[0], inplace=True)
data = midas_features.merge(data, left_index=True, right_index=True, how='left')
# data = data.merge(merged_df_month, left_index=True, right_index=True, how='left')

data.to_csv('data.csv', index= True)

'''3-ADD TECHNICAL INDICATORS'''
data['sma_20'] = data['Close'].rolling(window=20).mean()
data['ema_50'] = data['Close'].ewm(span=50, adjust=False).mean()
#
# # Compute Bollinger Bands
data["MiddleBand"] = data["Close"].rolling(window=20).mean()
data["UpperBand"] = data["MiddleBand"] + 2 * data["Close"].rolling(window=20).std()
data["LowerBand"] = data["MiddleBand"] - 2 * data["Close"].rolling(window=20).std()

# Compute RSI (Relative Strength Index)
data['rsi_14'] = ta.rsi(data['Close'], length=14)

# 添加MACD计算 (新增)
exp1 = data['Close'].ewm(span=12, adjust=False).mean()
exp2 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = exp1 - exp2
data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']

# Compute ATR (Average True Range)
# data['atr_14'] =
data['atr_14_lagged'] = ta.atr(data['High'], data['Low'], data['Close'], length=14).shift(1)

# lag features
# 滞后特征
data['FX_Daily_Returns_lag1'] = data['FX_Daily_Returns'].shift(1)
data['FX_Daily_Returns_lag3'] = data['FX_Daily_Returns'].shift(3)

# 滚动统计量
data['FX_Returns_rolling_mean_10'] = data['FX_Daily_Returns'].rolling(window=10).mean().shift(5)
data['FX_Returns_rolling_std_10'] = data['FX_Daily_Returns'].rolling(window=10).std().shift(5)


print('before data.shape', data.shape)
data = data.dropna()
print('data.shape', data.shape)
# data.to_csv('after data.csv', index= True)

#daily variables
# #daily variables

'''time-based features'''
# Time-Based Features
# data['day_of_week'] = data.index.dayofweek
# data['month'] = data.index.month
# data['quarter'] = data.index.quarter





'''STEP 4: FEATURE SELECTION '''
selected_features = data.drop(columns=['FX_risk_score','Open','High','Low','Close', 'FX_Daily_Returns','Future_Direction']).columns
print('selected_features', selected_features)
# corr
corr_matrix = data[selected_features].corr()

# plot_corr_feature(corr_matrix,'correlation matrix','results/corr_test.png')

corr_matrix_abs = data[selected_features].corr().abs()

corr_threshold = 0.85
# 只保留上三角部分，避免重复
upper = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool))

# feature variances
feature_variances = data[selected_features].var()

# find features with high correlation, retain the feature with higher variance
to_drop = set()
for column in upper.columns:
    for row in upper.index:
        if upper.loc[row, column] > corr_threshold:
            drop_feature = row if feature_variances[row] < feature_variances[column] else column
            to_drop.add(drop_feature)

print('to_drop', to_drop)


#filtered_features = [col for col in selected_features if col not in to_drop]

# 重命名特征列为数字索引

# 修改后（正确）:
# 1. 先获取经过相关性筛选的原始特征名
original_filtered_features = [col for col in selected_features if col not in to_drop]

# 2. 创建数字特征名映射（f0, f1, f2...）
#filtered_features = [f"f{i}" for i in range(len(original_filtered_features))]
filtered_features = [str(i) for i in range(len(original_filtered_features))] 
# 3. 重命名数据框列
#X = data[original_filtered_features].rename(
#    columns={original: new for original, new in zip(original_filtered_features, filtered_features)}
#)

# 不需要重命名特征列为数字索引
# 直接使用原始特征名
X = data[original_filtered_features]  # 使用原始特征名
"""
# 【关键修正】将 data 中的原始列名替换为重命名后的列名
data.rename(
    columns=dict(zip(original_filtered_features, filtered_features)), 
    inplace=True
)
# ------------------------------------------------------------------
"""
# 确认列名已更新
#print("检查 data 列名是否包含 f0-f31:", data.columns.tolist()[:5])  # 应该显示 ['f0', 'f1', ...]

#X = data[filtered_features]
# 验证列名已修改
#print("当前特征列名:", X.columns.tolist()[:5])  # 应该输出 ['f0', 'f1', 'f2', 'f3', 'f4']
#print("Features after correlation filtering:", filtered_features)
# plot_corr_feature(data[filtered_features].corr(),'correlation matrix after','results/corr_after.png')

#X = data[filtered_features]
#X = data[selected_features]
y = data['FX_risk_score'].values

#Rf-rank
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

feature_importances = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
print(feature_importance_df)
importance_thereshold = 0.005
important_features = feature_importance_df[feature_importance_df["Importance"] > importance_thereshold]["Feature"].tolist()

# X = X[important_features]
# print('important_features', important_features)
# X = data[selected_features]



# Apply StandardScaler (ONLY on Continuous Features)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
y_reg = data['FX_risk_score'].values
scaler = MinMaxScaler(feature_range=(0, 1))  # 映射到 [0, 1]
y_normalized = scaler.fit_transform(y_reg.reshape(-1, 1)).flatten()


# y_class = pd.cut(y_normalized, bins=[-0.1, 0.3, 0.7, 1.1], labels=[0, 1, 2])
#quantile
y_class = pd.qcut(y_reg, q=3, labels=[0, 1, 2])

print('y_class.value_counts()', y_class.value_counts())
print("NaN of y_class:", y_class.isnull().sum())
print('X.shape, Y_reg.shape', X.shape, y_reg.shape)
# X_monthly = merged_df_month.values



'''5-train'''
# X = X_scaled
# Train-Test Split
# 假设你的数据是按时间排序的

print('X.columns', X.columns)
split_index = int(len(data) * 0.8)
train_data = data.iloc[:split_index]
test_data = data.iloc[split_index:]

X_train, y_train_reg = train_data[X.columns], train_data['FX_risk_score']
X_test, y_test_reg = test_data[X.columns], test_data['FX_risk_score']

fig, ax = plt.subplots(figsize=(12, 6))
y_train_reg.plot(ax=ax, label='Training Set', title='Train and Test Split')
y_test_reg.plot(ax=ax, label='Test Set')
ax.axvline(split_index, color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])

import os

# 确保目录存在
output_dir = os.path.expanduser('/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/results')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 保存文件
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'train_test_split_{}.png'.format(split_index)), dpi=300)
plt.show()

# X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, shuffle=False, random_state=None)

_, _, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, shuffle=False, random_state=None)
print(y_train_class.value_counts())

# Train Regression Model (XGBoost)
# reg_model = RandomForestRegressor(n_estimators=100, random_state=42)

reg_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
reg_model.fit(X_train, y_train_reg)

# 训练回归模型时
#reg_model.fit(X_train, y_train_reg, feature_names=X_train.columns.tolist())



# y_pred_reg =reg_model.predict(X_test)
y_pred_reg = pd.Series(reg_model.predict(X_test), index=y_test_reg.index)

'''6-evaluate reg'''
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f"RMSE: {rmse:.3e}, MSE: {mse:.3e}")

def log_ratio_error(y_true, y_pred, epsilon=1e-10):
    return np.mean(np.abs(np.log((y_pred + epsilon) / (y_true + epsilon))))
log_ratio = log_ratio_error(y_test_reg, y_pred_reg)
print(f"Log Ratio Error: {log_ratio:.4f}")
mae = mean_absolute_error(y_test_reg, y_pred_reg)
print(f"MAE: {mae:.6f}")
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
smape_value = smape(y_test_reg, y_pred_reg)
print(f"SMAPE: {smape_value:.4f}%")

'''7-train and evaluate classification'''
# Train Classification Model (Random Forest)
# clf_model = RandomForestClassifier(n_estimators=500, random_state=42)
clf_model = XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.01, subsample= 0.9, colsample_bytree= 0.8)
# Best Parameters: {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.9}

clf_model.fit(X_train, y_train_class)
# 训练分类模型时
#clf_model.fit(X_train, y_train_class, feature_names=X_train.columns.tolist())
y_pred_class = clf_model.predict(X_test)
# Evaluate Classification Performance
print(classification_report(y_test_class, y_pred_class))


#reg_model.get_booster().feature_names = X_train.columns.tolist()
#clf_model.get_booster().feature_names = X_train.columns.tolist()

#print("Regression 特征名称:", reg_model.get_booster().feature_names)
#print("Classification 特征名称:", clf_model.get_booster().feature_names)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, make_scorer
# from xgboost import XGBClassifier
#
# # Define parameter grid for tuning
# param_grid = {
#     'n_estimators': [100, 300, 500],
#     'max_depth': [3, 5],
#     'learning_rate': [0.01, 0.1],
#     'subsample': [0.7, 0.8, 0.9],
#     'colsample_bytree': [0.7, 0.8, 0.9]
# }


#连到主页市场概率与风险预警得分
def display_dashboard(data, y_pred_reg, y_pred_class):
    """显示仪表盘指标"""
    
    # 计算波动率
    volatility = data['FX_Daily_Returns'].std() * np.sqrt(252) * 100  # 年化波动率,转为百分比
    
    # 获取最新的交易量
    volume = data['Volume'].iloc[-1] if 'Volume' in data else 23.5  # 如果没有交易量数据,使用示例值
    
    # 判断趋势方向
    trend = '上升' if data['Close'].iloc[-1] > data['Close'].iloc[-2] else '下降'
    
    # 获取最新的风险评分和风险等级
    risk_score = y_pred_reg[-1]
    risk_level = y_pred_class[-1]
    
    # 计算风险等级的置信度(使用最近5天的预测)
    recent_predictions = y_pred_class[-5:]
    confidence = (recent_predictions == risk_level).mean() * 100
    
    print("\n=== 仪表盘 ===")
    
    # 1. 市场警报
    if volatility > 1.0:  # 波动率阈值设为1%
        print("\n市场警报:")
        print(f"USD/CNH 波动率超过阈值")
        print("💡 智小汇建议您：建议减少高风险货币类型，增加避险资产配置，谨慎操作。")
    
    # 2. 市场趋势
    print("\n市场趋势:")
    # 这里可以添加趋势图的数据点,但在控制台只显示数值
    print(f"最新波动率: {volatility:.3f}")
    
    # 3. 市场概览
    print("\n市场概览:")
    print(f"波动率: {volatility:.1f}%")
    print(f"交易量: {volume:.1f}M")
    print(f"趋势方向: {trend}")
    
    # 4. 风险预警
    print("\n风险预警:")
    print(f"风险得分: {risk_score:.5f} ↑")
    risk_level_str = '高' if risk_level == 2 else '中' if risk_level == 1 else '低'
    print(f"风险等级: {risk_level_str} ({confidence:.0f}%)")


# 显示仪表盘
display_dashboard(test_data, y_pred_reg, y_pred_class)

def display_technical_indicators(data):
    """显示技术指标"""
    print("\n技术指标:")
    
    # RSI计算 (已在原代码中)
    rsi = data['rsi_14'].iloc[-1]
    rsi_signal = '中性' if 30 < rsi < 70 else '超卖' if rsi <= 30 else '超买'
    print(f"RSI: {rsi:.0f} ({rsi_signal})")
    
    # MACD计算
    # 如果原代码中没有MACD，需要添加计算
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_value = macd.iloc[-1]
    macd_signal = '买入' if macd_value > 0 else '卖出'
    print(f"MACD: {macd_value:.5f} ({macd_signal})")
    
    # 布林带计算 (已在原代码中)
    middle_band = data["MiddleBand"].iloc[-1]
    upper_band = data["UpperBand"].iloc[-1]
    lower_band = data["LowerBand"].iloc[-1]
    current_price = data['Close'].iloc[-1]
    
    # 判断布林带状态
    if current_price > upper_band:
        bb_signal = '超买'
    elif current_price < lower_band:
        bb_signal = '超卖'
    else:
        bb_signal = '震荡'
    
    print(f"布林带: {lower_band:.2f}-{upper_band:.2f} ({bb_signal})")



'''
# 7 Visualize: XG feature importance
'''
xgb.plot_importance(reg_model, importance_type="gain", max_num_features=10)
plt.title("XGBoost Feature Importance")
plt.show()




'''8 Visualize: pred vs real and mark high risk timestamp'''
plot_real_pred_reg(y_test_reg, y_pred_reg,'/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/results/xg_volatility_pred_real.png')

threshold_type  = 'value'
visualize_detection_class(y_test_reg, y_pred_reg,y_test_class, y_pred_class,threshold_type,True,True, 0.003,0.95,f'/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/results/xg_classify_{threshold_type}.png',
                          f'/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/results/xg_classify_CI.png', f'/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/results/xg_classify_class2.png')


# 添加多货币对处理功能
def process_multi_currency_pairs(file_path):
    """处理多货币对数据,并为每个货币对生成风险信号预警"""
    # 加载多货币对数据
    multi_pairs = pd.read_excel(file_path, parse_dates=["date"])
    
    # 设置日期为索引
    multi_pairs.set_index('date', inplace=True)
    
    # 定义需要处理的货币对列表
    currency_pairs = ['USDJPY', 'USDCNH', 'JPYEUR', 'JPYCNY', 'EURUSD', 'EURJPY', 'EURCNH']
    
    # 存储每个货币对的结果
    models = {}
    results = {}
    
    for pair in currency_pairs:
        print(f"\n处理货币对: {pair}")
        # 提取当前货币对的数据
        pair_data = pd.DataFrame()
        pair_data['Open'] = multi_pairs[f'{pair}_open']
        pair_data['High'] = multi_pairs[f'{pair}_high']
        pair_data['Low'] = multi_pairs[f'{pair}_low']
        pair_data['Close'] = multi_pairs[f'{pair}_close']
        
        # 计算每日收益率
        pair_data['FX_Daily_Returns'] = pair_data['Close'].pct_change()
        
        # 计算风险评分
        pair_data = risk_score_definitions(pair_data)
        
        # 添加技术指标
        # SMA和EMA
        pair_data['sma_20'] = pair_data['Close'].rolling(window=20).mean()
        pair_data['ema_50'] = pair_data['Close'].ewm(span=50, adjust=False).mean()
        
        # 布林带
        pair_data["MiddleBand"] = pair_data["Close"].rolling(window=20).mean()
        pair_data["UpperBand"] = pair_data["MiddleBand"] + 2 * pair_data["Close"].rolling(window=20).std()
        pair_data["LowerBand"] = pair_data["MiddleBand"] - 2 * pair_data["Close"].rolling(window=20).std()
        
        # RSI
        pair_data['rsi_14'] = ta.rsi(pair_data['Close'], length=14)
        
        # ATR
        pair_data['atr_14_lagged'] = ta.atr(pair_data['High'], pair_data['Low'], pair_data['Close'], length=14).shift(1)
        
        # 滞后特征
        pair_data['FX_Daily_Returns_lag1'] = pair_data['FX_Daily_Returns'].shift(1)
        pair_data['FX_Daily_Returns_lag3'] = pair_data['FX_Daily_Returns'].shift(3)
        
        # 滚动统计量
        pair_data['FX_Returns_rolling_mean_10'] = pair_data['FX_Daily_Returns'].rolling(window=10).mean().shift(5)
        pair_data['FX_Returns_rolling_std_10'] = pair_data['FX_Daily_Returns'].rolling(window=10).std().shift(5)
        
        # 如果有外部特征数据，可以合并
        # 这里假设我们有合并好的特征数据，如果需要也可以为每个货币对单独处理
        if 'merged_df_daily' in locals() and 'midas_features' in locals():
            pair_data = pair_data.merge(merged_df_daily, left_index=True, right_index=True, how='left')
            pair_data = midas_features.merge(pair_data, left_index=True, right_index=True, how='left')
        
        # 删除缺失值
        pair_data = pair_data.dropna()
        
        # 特征选择
        selected_features = pair_data.drop(columns=['FX_risk_score','Open','High','Low','Close', 'FX_Daily_Returns','Future_Direction']).columns
        
        # 相关性筛选
        corr_matrix_abs = pair_data[selected_features].corr().abs()
        corr_threshold = 0.85
        upper = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool))
        
        # 计算特征方差
        feature_variances = pair_data[selected_features].var()
        
        # 找出高相关性特征，保留方差更高的
        to_drop = set()
        for column in upper.columns:
            for row in upper.index:
                if upper.loc[row, column] > corr_threshold:
                    drop_feature = row if feature_variances[row] < feature_variances[column] else column
                    to_drop.add(drop_feature)
        
        # 过滤特征
        original_filtered_features = [col for col in selected_features if col not in to_drop]
        X = pair_data[original_filtered_features]
        
        # 目标变量
        y_reg = pair_data['FX_risk_score'].values
        
        # 数据分割
        split_index = int(len(pair_data) * 0.8)
        train_data = pair_data.iloc[:split_index]
        test_data = pair_data.iloc[split_index:]
        
        X_train, y_train_reg = train_data[X.columns], train_data['FX_risk_score']
        X_test, y_test_reg = test_data[X.columns], test_data['FX_risk_score']
        
        # 分类目标变量
        y_class = pd.qcut(y_reg, q=3, labels=[0, 1, 2])
        _, _, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, shuffle=False, random_state=None)
        
        # 训练回归模型
        reg_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
        reg_model.fit(X_train, y_train_reg)
        
        # 回归预测
        y_pred_reg = pd.Series(reg_model.predict(X_test), index=y_test_reg.index)
        
        # 评估回归模型
        rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
        mae = mean_absolute_error(y_test_reg, y_pred_reg)
        
        # 训练分类模型
        clf_model = XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.9, colsample_bytree=0.8)
        clf_model.fit(X_train, y_train_class)
        
        # 分类预测
        y_pred_class = clf_model.predict(X_test)
        
        # 评估分类模型
        class_report = classification_report(y_test_class, y_pred_class, output_dict=True)
        
        # 可视化
        output_dir = os.path.expanduser(f'~/Desktop/results/{pair}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存预测结果
        plot_real_pred_reg(y_test_reg, y_pred_reg, f'{output_dir}/volatility_pred_real.png')
        
        # 可视化风险类别
        threshold_type = 'value'
        visualize_detection_class(
            y_test_reg, y_pred_reg, y_test_class, y_pred_class,
            threshold_type, True, True, 0.003, 0.95,
            f'{output_dir}/classify_{threshold_type}.png',
            f'{output_dir}/classify_CI.png',
            f'{output_dir}/classify_class2.png'
        )
        
        # 存储模型和结果
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
        
        print(f"{pair} 处理完成，RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        print(f"分类报告:\n{classification_report(y_test_class, y_pred_class)}")
        
        # 回测分析（在分类和回归模型都已经训练好之后）
        print(f"\n进行 {pair} 的回测分析...")
        
        # 将NumPy数组转换为带索引的Series
        y_pred_class_series = pd.Series(y_pred_class, index=test_data.index)
        
        # 进行回测
        backtest_results = backtest_risk_signals(
            test_data, 
            y_pred_class_series,  # 使用转换后的Series
            risk_threshold=2,  # 高风险阈值
            stop_loss_pct=0.02,  # 2%止损
            take_profit_pct=0.05  # 5%止盈
        )
        
        # 保存回测结果
        results[pair]['backtest_results'] = backtest_results
        
        # 可视化回测结果
        visualize_backtest_results(
            pair, 
            backtest_results, 
            y_pred_class_series, 
            test_data['Close'], 
            os.path.expanduser(f'~/Desktop/results/{pair}')
        )
        
        print(f"{pair} 回测完成")
        print(f"总交易次数: {backtest_results['total_trades']}")
        print(f"胜率: {backtest_results['win_rate']:.2f}%")
        print(f"平均收益: {backtest_results['avg_return']:.2f}%")
        print(f"最大回撤: {backtest_results['max_drawdown']:.2f}%")
        print(f"夏普比率: {backtest_results['sharpe_ratio']:.2f}")
    
    return models, results

def analyze_currency_pair_risk_distribution(results, weights=None):
    """
    分析多货币对的风险信号分布和加权风险评估
    
    参数:
    results: 货币对处理结果字典
    weights: 各货币对权重字典，如果不提供则平均分配权重
    
    返回:
    risk_distribution: 每个货币对的风险分布
    weighted_risk: 加权后的综合风险评估
    """
    risk_distribution = {}
    
    # 如果未提供权重，则平均分配
    if weights is None:
        pairs = list(results.keys())
        weights = {pair: 1.0/len(pairs) for pair in pairs}
    
    # 计算每个货币对的风险分布
    for pair, result in results.items():
        # 获取预测的风险类别
        y_pred_class = result['y_pred_class']
        
        # 计算各类别的百分比
        class_counts = pd.Series(y_pred_class).value_counts(normalize=True) * 100
        
        # 确保所有类别都存在（即使是0%）
        for i in range(3):  # 假设有3个风险类别(0,1,2)
            if i not in class_counts.index:
                class_counts[i] = 0.0
        
        risk_distribution[pair] = {
            'low_risk_pct': class_counts.get(0, 0.0),
            'medium_risk_pct': class_counts.get(1, 0.0),
            'high_risk_pct': class_counts.get(2, 0.0),
            'weight': weights[pair]
        }
    
    # 计算加权风险指标
    weighted_risk = {
        'low_risk': sum(dist['low_risk_pct'] * dist['weight'] for pair, dist in risk_distribution.items()),
        'medium_risk': sum(dist['medium_risk_pct'] * dist['weight'] for pair, dist in risk_distribution.items()),
        'high_risk': sum(dist['high_risk_pct'] * dist['weight'] for pair, dist in risk_distribution.items())
    }
    
    return risk_distribution, weighted_risk

def visualize_risk_distribution(risk_distribution, weighted_risk, output_path):
    """可视化货币对风险分布和加权风险"""
    # 1. 创建风险分布条形图
    pairs = list(risk_distribution.keys())
    low_risk_values = [risk_distribution[pair]['low_risk_pct'] for pair in pairs]
    medium_risk_values = [risk_distribution[pair]['medium_risk_pct'] for pair in pairs]
    high_risk_values = [risk_distribution[pair]['high_risk_pct'] for pair in pairs]
    weights = [risk_distribution[pair]['weight'] * 100 for pair in pairs]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # 货币对风险分布条形图
    x = np.arange(len(pairs))
    width = 0.25
    
    ax1.bar(x - width, low_risk_values, width, label='低风险', color='green')
    ax1.bar(x, medium_risk_values, width, label='中等风险', color='orange')
    ax1.bar(x + width, high_risk_values, width, label='高风险', color='red')
    
    ax1.set_title('各货币对风险分布', fontsize=16)
    ax1.set_xlabel('货币对', fontsize=14)
    ax1.set_ylabel('百分比 (%)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(pairs)
    ax1.legend()
    
    # 货币对权重分布饼图
    ax2.pie(weights, labels=pairs, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    ax2.set_title('distribution of the weights for curency pairs', fontsize=16)
    
    # 加权风险分布饼图
    weighted_values = [weighted_risk['low_risk'], weighted_risk['medium_risk'], weighted_risk['high_risk']]
    ax3.pie(weighted_values, labels=['low risk', 'medium risk', 'high risk'],
            autopct='%1.1f%%', startangle=90, colors=['green', 'orange', 'red'])
    ax3.axis('equal')
    ax3.set_title('加权风险分布', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    # 创建时间序列上的风险分布
    for pair, result in results.items():
        # 获取预测的风险类别和时间索引
        y_pred_class = result['y_pred_class']
        dates = result['y_test_reg'].index
        
        # 创建时间序列数据框
        risk_ts = pd.DataFrame({
            'date': dates,
            'risk_class': y_pred_class
        })
        risk_ts.set_index('date', inplace=True)
        
        # 创建热图
        plt.figure(figsize=(12, 4))
        plt.pcolormesh(risk_ts.index, [0], risk_ts['risk_class'].values.reshape(1, -1), 
                       cmap=plt.cm.get_cmap('RdYlGn_r', 3), vmin=0, vmax=2)
        plt.colorbar(ticks=[0, 1, 2], label='风险等级')
        plt.yticks([])
        plt.title(f'{pair} 风险信号时间分布')
        plt.tight_layout()
        plt.savefig(os.path.expanduser(f'~/Desktop/results/{pair}/risk_timeline.png'), dpi=300)
        plt.close()


def backtest_risk_signals(pair_data, risk_predictions, strategy_type='enhanced', 
                          risk_threshold=2, stop_loss_pct=0.02, take_profit_pct=0.04):
    """
    根据风险信号进行回测分析，支持多种策略
    
    参数:
    pair_data: 包含价格数据的DataFrame
    risk_predictions: 风险类别预测结果(0=低风险, 1=中等风险, 2=高风险)
    strategy_type: 策略类型 ('basic', 'trend_filter', 'enhanced', 'combined')
    risk_threshold: 触发交易的风险阈值
    stop_loss_pct: 止损百分比
    take_profit_pct: 止盈百分比
    
    返回:
    backtest_results: 回测结果字典
    """
    # 检查risk_predictions是否为numpy数组，如果是则创建带索引的Series
    if isinstance(risk_predictions, np.ndarray):
        risk_predictions = pd.Series(risk_predictions, index=pair_data.index)
    
    # 确保索引对齐
    backtest_df = pd.DataFrame(index=risk_predictions.index)
    backtest_df['close'] = pair_data.loc[risk_predictions.index, 'Close']
    backtest_df['risk_class'] = risk_predictions
    
    # 添加技术指标用于增强策略
    if strategy_type in ['trend_filter', 'enhanced', 'combined']:
        # 计算短期和长期移动平均线判断趋势
        backtest_df['sma_20'] = backtest_df['close'].rolling(window=20).mean()
        backtest_df['sma_50'] = backtest_df['close'].rolling(window=50).mean()
        backtest_df['trend'] = np.where(backtest_df['sma_20'] > backtest_df['sma_50'], 1, -1)
        
        # 计算波动率用于动态止损/止盈
        backtest_df['volatility'] = backtest_df['close'].pct_change().rolling(window=20).std() * np.sqrt(20)
        
        # 添加RSI指标
        delta = backtest_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        backtest_df['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算风险信号变化
        backtest_df['risk_change'] = backtest_df['risk_class'].diff()
    
    # 丢弃包含NaN的行
    backtest_df = backtest_df.dropna()
    
    if len(backtest_df) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'equity_curve': pd.Series([1.0]),
            'trades': pd.DataFrame()
        }
    
    # 初始化变量
    position = 0  # 0: 无仓位, 1: 多头, -1: 空头
    entry_price = 0
    entry_date = None
    trades = []
    daily_returns = []
    equity = [1.0]  # 初始资金为1单位
    dates = [backtest_df.index[0]]
    max_holding_days = 20  # 最大持仓天数
    holding_days = 0
    
    # 回测循环
    for i in range(1, len(backtest_df)):
        date = backtest_df.index[i]
        curr_price = backtest_df['close'].iloc[i]
        prev_price = backtest_df['close'].iloc[i-1]
        risk_class = backtest_df['risk_class'].iloc[i]
        
        # 根据不同策略类型设置动态参数
        if strategy_type in ['enhanced', 'combined']:
            # 动态止损/止盈基于波动率
            current_volatility = backtest_df['volatility'].iloc[i]
            dynamic_stop_loss = max(stop_loss_pct, current_volatility * 1.0)  # 波动率的1倍
            dynamic_take_profit = max(take_profit_pct, current_volatility * 2.0)  # 波动率的2倍
        else:
            dynamic_stop_loss = stop_loss_pct
            dynamic_take_profit = take_profit_pct
        
        # 计算当日收益率（如果有持仓）
        if position != 0:
            pct_change = (curr_price / prev_price - 1) * position
            daily_returns.append(pct_change)
            equity.append(equity[-1] * (1 + pct_change))
            dates.append(date)
            holding_days += 1
        else:
            daily_returns.append(0)
            equity.append(equity[-1])
            dates.append(date)
            holding_days = 0
        
        # 检查持仓时间限制
        if position != 0 and holding_days >= max_holding_days:
            trades.append({
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': date,
                'exit_price': curr_price,
                'position': position,
                'pnl_pct': ((curr_price - entry_price) / entry_price) * position,
                'exit_reason': '持仓时间限制',
                'risk_class': backtest_df['risk_class'].iloc[i-holding_days]
            })
            position = 0
            holding_days = 0
            continue
            
        # 检查止损/止盈
        if position != 0:
            price_change = (curr_price - entry_price) / entry_price
            
            # 止损
            if (position == 1 and price_change < -dynamic_stop_loss) or \
               (position == -1 and price_change > dynamic_stop_loss):
                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': date,
                    'exit_price': curr_price,
                    'position': position,
                    'pnl_pct': price_change * position,
                    'exit_reason': '止损',
                    'risk_class': backtest_df['risk_class'].iloc[i-holding_days]
                })
                position = 0
                holding_days = 0
                continue
                
            # 止盈
            if (position == 1 and price_change > dynamic_take_profit) or \
               (position == -1 and price_change < -dynamic_take_profit):
                trades.append({
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': date,
                    'exit_price': curr_price,
                    'position': position,
                    'pnl_pct': price_change * position,
                    'exit_reason': '止盈',
                    'risk_class': backtest_df['risk_class'].iloc[i-holding_days]
                })
                position = 0
                holding_days = 0
                continue

        # 根据不同策略生成交易信号
        if strategy_type == 'basic':
            # 基础策略：简单基于风险类别交易
            if position == 0:  # 没有持仓时
                if risk_class == 0:  # 低风险信号，做多
                    position = 1
                    entry_price = curr_price
                    entry_date = date
                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'position': position,
                        'risk_class': risk_class
                    })
                elif risk_class == risk_threshold:  # 高风险信号，做空
                    position = -1
                    entry_price = curr_price
                    entry_date = date
                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'position': position,
                        'risk_class': risk_class
                    })
            elif position == 1 and risk_class == risk_threshold:  # 持有多头但出现高风险，平仓
                trades[-1].update({
                    'exit_date': date,
                    'exit_price': curr_price,
                    'pnl_pct': (curr_price - entry_price) / entry_price,
                    'exit_reason': '风险信号'
                })
                position = 0
            elif position == -1 and risk_class == 0:  # 持有空头但出现低风险，平仓
                trades[-1].update({
                    'exit_date': date,
                    'exit_price': curr_price,
                    'pnl_pct': (entry_price - curr_price) / entry_price,
                    'exit_reason': '风险信号'
                })
                position = 0
                
        elif strategy_type == 'trend_filter':
            # 趋势过滤策略：只在趋势方向一致时交易
            trend = backtest_df['trend'].iloc[i]
            
            if position == 0:  # 没有持仓时
                if risk_class == 0 and trend == 1:  # 低风险信号 + 上升趋势，做多
                    position = 1
                    entry_price = curr_price
                    entry_date = date
                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'position': position,
                        'risk_class': risk_class
                    })
                elif risk_class == risk_threshold and trend == -1:  # 高风险信号 + 下降趋势，做空
                    position = -1
                    entry_price = curr_price
                    entry_date = date
                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'position': position,
                        'risk_class': risk_class
                    })
            elif position == 1 and (risk_class == risk_threshold or trend == -1):  # 持有多头但出现高风险或趋势反转，平仓
                trades[-1].update({
                    'exit_date': date,
                    'exit_price': curr_price,
                    'pnl_pct': (curr_price - entry_price) / entry_price,
                    'exit_reason': '信号反转'
                })
                position = 0
            elif position == -1 and (risk_class == 0 or trend == 1):  # 持有空头但出现低风险或趋势反转，平仓
                trades[-1].update({
                    'exit_date': date,
                    'exit_price': curr_price,
                    'pnl_pct': (entry_price - curr_price) / entry_price,
                    'exit_reason': '信号反转'
                })
                position = 0
                
        elif strategy_type == 'enhanced':
            # 增强策略：综合风险信号、趋势和风险变化强度
            trend = backtest_df['trend'].iloc[i]
            rsi = backtest_df['rsi'].iloc[i]
            risk_change = backtest_df['risk_change'].iloc[i] if not pd.isna(backtest_df['risk_change'].iloc[i]) else 0
            
            if position == 0:  # 没有持仓时
                # 低风险 + 上升趋势 + RSI不超买 = 做多
                if risk_class == 0 and trend == 1 and rsi < 70:
                    position = 1
                    entry_price = curr_price
                    entry_date = date
                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'position': position,
                        'risk_class': risk_class
                    })
                # 高风险 + 下降趋势 + RSI不超卖 = 做空    
                elif risk_class == risk_threshold and trend == -1 and rsi > 30:
                    position = -1
                    entry_price = curr_price
                    entry_date = date
                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'position': position,
                        'risk_class': risk_class
                    })
            # 平仓条件更严格，需要多重信号确认        
            elif position == 1:
                if (risk_class == risk_threshold and trend == -1) or rsi > 75:
                    trades[-1].update({
                        'exit_date': date,
                        'exit_price': curr_price,
                        'pnl_pct': (curr_price - entry_price) / entry_price,
                        'exit_reason': '综合信号反转'
                    })
                    position = 0
            elif position == -1:
                if (risk_class == 0 and trend == 1) or rsi < 25:
                    trades[-1].update({
                        'exit_date': date,
                        'exit_price': curr_price,
                        'pnl_pct': (entry_price - curr_price) / entry_price,
                        'exit_reason': '综合信号反转'
                    })
                    position = 0
                    
        elif strategy_type == 'combined':
            # 组合策略：结合所有优化手段，降低交易频率，提高精确度
            trend = backtest_df['trend'].iloc[i]
            rsi = backtest_df['rsi'].iloc[i]
            
            # 检查前3天的风险信号是否一致（连续信号确认）
            if i >= 3:
                prev_3_signals = backtest_df['risk_class'].iloc[i-3:i].values
                consistent_signal = len(set(prev_3_signals)) == 1
            else:
                consistent_signal = False
                
            if position == 0 and consistent_signal:  # 没有持仓且信号一致
                # 多重条件确认做多
                if (risk_class == 0 and trend == 1 and rsi < 40 and
                    curr_price > backtest_df['sma_20'].iloc[i]):
                    position = 1
                    entry_price = curr_price
                    entry_date = date
                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'position': position,
                        'risk_class': risk_class
                    })
                # 多重条件确认做空    
                elif (risk_class == risk_threshold and trend == -1 and rsi > 60 and
                      curr_price < backtest_df['sma_20'].iloc[i]):
                    position = -1
                    entry_price = curr_price
                    entry_date = date
                    trades.append({
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'position': position,
                        'risk_class': risk_class
                    })
            # 多头平仓条件        
            elif position == 1:
                if risk_class == risk_threshold or trend == -1 or rsi > 75:
                    trades[-1].update({
                        'exit_date': date,
                        'exit_price': curr_price,
                        'pnl_pct': (curr_price - entry_price) / entry_price,
                        'exit_reason': '组合信号反转'
                    })
                    position = 0
            # 空头平仓条件        
            elif position == -1:
                if risk_class == 0 or trend == 1 or rsi < 25:
                    trades[-1].update({
                        'exit_date': date,
                        'exit_price': curr_price,
                        'pnl_pct': (entry_price - curr_price) / entry_price,
                        'exit_reason': '组合信号反转'
                    })
                    position = 0
    
    # 处理未平仓的交易
    if position != 0:
        last_date = backtest_df.index[-1]
        last_price = backtest_df['close'].iloc[-1]
        trades[-1].update({
            'exit_date': last_date,
            'exit_price': last_price,
            'pnl_pct': ((last_price - entry_price) / entry_price) * position,
            'exit_reason': '回测结束'
        })
    
    # 转换为DataFrame方便分析
    trades_df = pd.DataFrame(trades)
    
    # 如果没有交易，返回基本结果
    if len(trades_df) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'equity_curve': pd.Series(equity, index=dates),
            'trades': trades_df,
            'strategy': strategy_type
        }
    
    # 计算胜率
    if 'pnl_pct' in trades_df:
        win_rate = (trades_df['pnl_pct'] > 0).mean() * 100
        avg_return = trades_df['pnl_pct'].mean() * 100
    else:
        win_rate = 0
        avg_return = 0
    
    # 计算净值曲线和最大回撤
    equity_series = pd.Series(equity, index=dates)
    max_drawdown = calculate_max_drawdown(equity_series) * 100
    
    # 计算夏普比率（假设无风险利率为0）
    returns_series = pd.Series(daily_returns)
    sharpe_ratio = (returns_series.mean() / returns_series.std()) * np.sqrt(252) if returns_series.std() != 0 else 0
    
    # 计算年化收益率
    days = (dates[-1] - dates[0]).days
    if days > 0:
        annual_return = ((equity[-1] / equity[0]) ** (365 / days) - 1) * 100
    else:
        annual_return = 0
    
    # 汇总结果
    backtest_results = {
        'total_trades': len(trades_df),
        'win_rate': win_rate,
        'avg_return': avg_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'equity_curve': equity_series,
        'trades': trades_df,
        'strategy': strategy_type
    }
    
    return backtest_results



def calculate_max_drawdown(equity_curve):
    """计算最大回撤"""
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return drawdown.min()

def visualize_backtest_results(pair, backtest_results, risk_signals, price_data, output_dir):
    """可视化回测结果"""
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 绘制权益曲线
    plt.figure(figsize=(15, 8))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
    
    # 权益曲线
    equity_curve = backtest_results['equity_curve']
    ax1.plot(equity_curve.index, equity_curve.values, label='策略收益', color='blue', linewidth=2)
    
    # 价格曲线（次坐标轴）
    ax_price = ax1.twinx()
    ax_price.plot(price_data.index, price_data.values, color='gray', alpha=0.5, label='价格')
    
    # 标记交易点
    trades = backtest_results['trades']
    if len(trades) > 0 and 'entry_date' in trades.columns and 'exit_date' in trades.columns:
        for i, trade in trades.iterrows():
            if 'position' in trade and trade['position'] == 1:  # 多头
                if 'exit_date' in trade and pd.notna(trade['exit_date']):
                    if 'pnl_pct' in trade and trade['pnl_pct'] > 0:
                        # 盈利多头
                        ax1.fill_between(
                            pd.date_range(trade['entry_date'], trade['exit_date']),
                            0, 1, transform=ax1.get_xaxis_transform(),
                            alpha=0.2, color='green'
                        )
                    else:
                        # 亏损多头
                        ax1.fill_between(
                            pd.date_range(trade['entry_date'], trade['exit_date']),
                            0, 1, transform=ax1.get_xaxis_transform(),
                            alpha=0.2, color='red'
                        )
                ax1.scatter(trade['entry_date'], equity_curve.loc[trade['entry_date']], 
                           marker='^', color='green', s=100)
                if 'exit_date' in trade and pd.notna(trade['exit_date']):
                    ax1.scatter(trade['exit_date'], equity_curve.loc[trade['exit_date']], 
                               marker='v', color='red', s=100)
            elif 'position' in trade and trade['position'] == -1:  # 空头
                if 'exit_date' in trade and pd.notna(trade['exit_date']):
                    if 'pnl_pct' in trade and trade['pnl_pct'] > 0:
                        # 盈利空头
                        ax1.fill_between(
                            pd.date_range(trade['entry_date'], trade['exit_date']),
                            0, 1, transform=ax1.get_xaxis_transform(),
                            alpha=0.2, color='green'
                        )
                    else:
                        # 亏损空头
                        ax1.fill_between(
                            pd.date_range(trade['entry_date'], trade['exit_date']),
                            0, 1, transform=ax1.get_xaxis_transform(),
                            alpha=0.2, color='red'
                        )
                ax1.scatter(trade['entry_date'], equity_curve.loc[trade['entry_date']], 
                           marker='v', color='red', s=100)
                if 'exit_date' in trade and pd.notna(trade['exit_date']):
                    ax1.scatter(trade['exit_date'], equity_curve.loc[trade['exit_date']], 
                               marker='^', color='green', s=100)
    
    # 风险信号热图
    risk_data = pd.DataFrame({'risk_class': risk_signals}, index=risk_signals.index)
    ax2.pcolormesh(risk_data.index, [0], risk_data['risk_class'].values.reshape(1, -1),
                  cmap=plt.cm.get_cmap('RdYlGn_r', 3), vmin=0, vmax=2)
    
    # 添加标题和标签
    ax1.set_title(f'{pair} 风险信号回测结果', fontsize=16)
    ax1.set_ylabel('策略资产净值', fontsize=12)
    ax_price.set_ylabel('价格', fontsize=12)
    ax2.set_ylabel('风险信号', fontsize=12)
    
    # 添加图例
    ax1.legend(loc='upper left')
    ax_price.legend(loc='upper right')
    
    # 添加回测统计结果文本
    stats_text = (
        f"总交易次数: {backtest_results['total_trades']}\n"
        f"胜率: {backtest_results['win_rate']:.2f}%\n"
        f"平均收益: {backtest_results['avg_return']:.2f}%\n"
        f"年化收益: {backtest_results['annual_return']:.2f}%\n"
        f"最大回撤: {backtest_results['max_drawdown']:.2f}%\n"
        f"夏普比率: {backtest_results['sharpe_ratio']:.2f}"
    )
    ax1.text(0.02, 0.02, stats_text, transform=ax1.transAxes, 
             bbox=dict(facecolor='white', alpha=0.7), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{pair}_backtest_results.png'), dpi=300)
    plt.close()
    
    # 2. 绘制月度收益热图
    if len(equity_curve) > 30:  # 只有足够的数据才绘制
        # 计算每日收益率
        daily_returns = equity_curve.pct_change().dropna()
        
        # 重采样为月度收益
        monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # 创建月度收益热图
        monthly_returns_matrix = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns.values
        })
        
        # 透视表形式
        heatmap_data = monthly_returns_matrix.pivot(index='year', columns='month', values='return')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data * 100, annot=True, fmt='.2f', cmap='RdYlGn',
                   center=0, vmin=-10, vmax=10)
        plt.title(f'{pair} 月度收益率热图 (%)', fontsize=16)
        plt.xlabel('月份', fontsize=12)
        plt.ylabel('年份', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{pair}_monthly_returns.png'), dpi=300)
        plt.close()
    
    # 3. 交易统计饼图
    if len(trades) > 0 and 'exit_reason' in trades.columns:
        exit_reason_counts = trades['exit_reason'].value_counts()
        
        plt.figure(figsize=(10, 6))
        plt.pie(exit_reason_counts.values, labels=exit_reason_counts.index, autopct='%1.1f%%')
        plt.title(f'{pair} 交易平仓原因分布', fontsize=16)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{pair}_exit_reasons.png'), dpi=300)
        plt.close()

# 在主程序中调用
if __name__ == "__main__":
    # 原有代码处理
    # ... 

    # 在主程序中调用
    # 在显示仪表盘后添加技术指标显示
    display_dashboard(test_data, y_pred_reg, y_pred_class)
    display_technical_indicators(test_data)

    # 在主程序中调用
    # 假设我们已经有了预测结果
    y_pred_reg = reg_model.predict(X_test)
    y_pred_class = clf_model.predict(X_test)


    
    # 添加多货币对处理
    multi_pairs_file = os.path.expanduser('/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/multi/all_currency.xlsx')
    print(f"正在处理多货币对数据: {multi_pairs_file}")
    models, results = process_multi_currency_pairs(multi_pairs_file)
    
    # 设置货币对权重（可以根据重要性或交易量进行调整）
    # 例如 USDCNH 权重更高
    weights = {
        'USDJPY': 0.15,
        'USDCNH': 0.25,
        'JPYEUR': 0.10,
        'JPYCNY': 0.10,
        'EURUSD': 0.20,
        'EURJPY': 0.10,
        'EURCNH': 0.10
    }
    
    # 分析风险分布
    risk_distribution, weighted_risk = analyze_currency_pair_risk_distribution(results, weights)
    
    # 输出风险分布结果
    print("\n各货币对风险分布:")
    for pair, dist in risk_distribution.items():
        print(f"{pair} - 低风险: {dist['low_risk_pct']:.2f}%, 中等风险: {dist['medium_risk_pct']:.2f}%, 高风险: {dist['high_risk_pct']:.2f}%, 权重: {dist['weight']*100:.2f}%")
    
    print("\n加权风险分布:")
    print(f"低风险: {weighted_risk['low_risk']:.2f}%, 中等风险: {weighted_risk['medium_risk']:.2f}%, 高风险: {weighted_risk['high_risk']:.2f}%")
    
    # 可视化风险分布
    visualize_risk_distribution(risk_distribution, weighted_risk, 
                               os.path.expanduser('/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/results/risk_distribution.png'))
    
    # 输出原有的结果摘要
    print("\n多货币对风险预警模型评估摘要:")
    for pair, res in results.items():
        print(f"{pair} - RMSE: {res['rmse']:.4f}, MAE: {res['mae']:.4f}, F1(加权): {res['classification_report']['weighted avg']['f1-score']:.4f}")

    # 单独对USDCNH进行回测分析
    if 'USDCNH' in results:
        print("\n专注于USDCNH货币对的回测分析")
        
        usdcnh_results = results['USDCNH']
        usdcnh_backtest = usdcnh_results['backtest_results']
        
        # 打印详细的回测统计
        print("\nUSDCNH最佳策略回测统计:")
        print(f"策略类型: {usdcnh_backtest['strategy']}")
        print(f"总交易次数: {usdcnh_backtest['total_trades']}")
        print(f"胜率: {usdcnh_backtest['win_rate']:.2f}%")
        print(f"平均收益: {usdcnh_backtest['avg_return']:.2f}%")
        print(f"年化收益: {usdcnh_backtest['annual_return']:.2f}%")
        print(f"最大回撤: {usdcnh_backtest['max_drawdown']:.2f}%")
        print(f"夏普比率: {usdcnh_backtest['sharpe_ratio']:.2f}")
        
        # 分析盈亏比
        if len(usdcnh_backtest['trades']) > 0 and 'pnl_pct' in usdcnh_backtest['trades']:
            profit_trades = usdcnh_backtest['trades'][usdcnh_backtest['trades']['pnl_pct'] > 0]
            loss_trades = usdcnh_backtest['trades'][usdcnh_backtest['trades']['pnl_pct'] <= 0]
            
            if len(profit_trades) > 0 and len(loss_trades) > 0:
                avg_profit = profit_trades['pnl_pct'].mean() * 100
                avg_loss = loss_trades['pnl_pct'].mean() * 100
                profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')
                
                print(f"平均盈利: {avg_profit:.2f}%")
                print(f"平均亏损: {avg_loss:.2f}%")
                print(f"盈亏比: {profit_loss_ratio:.2f}")
        
        # 显示前5笔交易详情
        if len(usdcnh_backtest['trades']) > 0:
            print("\nUSDCNH前5笔交易:")
            trade_display = usdcnh_backtest['trades'].head()
            if 'pnl_pct' in trade_display:
                trade_display['pnl_pct'] = trade_display['pnl_pct'] * 100
            print(trade_display)
