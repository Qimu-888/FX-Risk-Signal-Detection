from process_plot_data import *
import pandas as pd
import pandas_ta as ta
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
fx_daily = pd.read_excel('FX/fx_m/usd_cnh_2015.xlsx', parse_dates=["Date"])

fx_daily = risk_score_definitions(fx_daily)
us_cpu = pd.read_excel('features/subEPU/US-CPU_2015.xlsx')
window_size = 10

# FX calculations
# fx_daily[f"FX_Volatility_roll{window_size}"] = fx_daily["FX_Daily_Returns"].rolling(window=window_size).std()
daily_folders = ['features/GPR', 'features/other_assets', 'features/sentiment_score', 'features/IR']
daily_index = pd.read_excel('index_daily.xlsx')
merged_df_daily = merge_selected_folders_with_fx(daily_index, daily_folders, 'features/merged/merged_daily_data.xlsx')
daily_vars = ['GPRD','GPRD_ACT','GPRD_THREAT','USD-Gold','CNY-Gold','Shanghai_Gold_Daily_Volatility','Oil_Price','oil-Daily-Volatility',
              'sentiment_score','Difference_CNY_CNH_IR','Difference_USD_CNH_IR']
merged_df_daily = merged_df_daily[daily_vars]
merged_df_daily.to_csv('merged_daily_data.csv', index= False)

'''2-for monthly and quarterly data'''
monthly_folders = ['features/CPI', 'features/EPU', "features/subEPU", "features/BoP", 'features/employment', 'features/FER', 'features/currency_supply']
merged_df_month = merge_selected_folders_with_fx(us_cpu, monthly_folders, 'features/merged/merged_monthly_data.xlsx')
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


qr_folders = ['features/GDP']
qr_index = pd.read_excel('qr_index.xlsx')
merged_df_qr = merge_selected_folders_with_fx(qr_index, qr_folders, 'features/merged/merged_qr_data.xlsx')
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
filtered_features = [col for col in selected_features if col not in to_drop]
print("Features after correlation filtering:", filtered_features)
# plot_corr_feature(data[filtered_features].corr(),'correlation matrix after','results/corr_after.png')

X = data[filtered_features]
# X = data[selected_features]
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
plt.tight_layout()
plt.savefig(f'results/train_test_split_{split_index}.png', dpi = 300)
plt.show()


# X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, shuffle=False, random_state=None)

_, _, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, shuffle=False, random_state=None)
print(y_train_class.value_counts())

# Train Regression Model (XGBoost)
# reg_model = RandomForestRegressor(n_estimators=100, random_state=42)

reg_model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
reg_model.fit(X_train, y_train_reg)

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
y_pred_class = clf_model.predict(X_test)
# Evaluate Classification Performance
print(classification_report(y_test_class, y_pred_class))

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, make_scorer
from xgboost import XGBClassifier

# Define parameter grid for tuning
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# # Initialize the XGBoost classifier
# clf_model = XGBClassifier(random_state=42)
#
# # Perform GridSearchCV with default precision scoring (macro or weighted)
# grid_search = GridSearchCV(
#     clf_model, param_grid, cv=5, scoring='precision_weighted', n_jobs=-1, verbose=2
# )
#
# # Fit GridSearch to find the best parameters
# grid_search.fit(X_train, y_train_class)
# # Get the best parameters
# best_params = grid_search.best_params_
# print("Best Parameters:", best_params)

# Train the final model with best parameters
# best_model = XGBClassifier(**best_params, random_state=42)
# best_model.fit(X_train, y_train_class)
#
# # Predict and evaluate performance
# y_pred_class = best_model.predict(X_test)
# print(classification_report(y_test_class, y_pred_class))




'''
# 7 Visualize: XG feature importance
'''
xgb.plot_importance(reg_model, importance_type="gain", max_num_features=10)
plt.title("XGBoost Feature Importance")
plt.show()


'''8 Visualize: pred vs real and mark high risk timestamp'''
plot_real_pred_reg(y_test_reg, y_pred_reg,'results/xg_volatility_pred_real.png')

threshold_type  = 'value'
visualize_detection_class(y_test_reg, y_pred_reg,y_test_class, y_pred_class,threshold_type,True,True, 0.003,0.95,f'results/xg_classify_{threshold_type}.png',
                          f'results/xg_classify_CI.png', f'results/xg_classify_class2.png')


'''9 shap explanation'''
print('X_train.shape', X_train.shape,'X_test.shape', X_test.shape)
print('y_pred_class.shape', y_pred_class.shape)
# explainer = shap.Explainer(reg_model, X_train)



# class
explainer_class = shap.Explainer(clf_model, X_train)
shap_values_class = explainer_class(X_test)
shap_values_2d = shap_values_class[:, :, 2]  # Select only class 0 SHAP values
plot_shap_summary_and_bar(shap_values_2d, X_test, save_prefix="shap_xg_class2", figsize=(28, 10))

#reg
explainer_reg = shap.Explainer(reg_model, X_train)
shap_values_reg = explainer_reg(X_test)
plot_shap_summary_and_bar(shap_values_reg, X_test, save_prefix="shap_xg_reg", figsize=(28, 10))


'''2 output test'''
# from sklearn.multioutput import MultiOutputRegressor
# #two output
# y = data[['FX_risk_score', 'Future_Direction']]
# model = MultiOutputRegressor(XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=8))
# _, _, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=None)
# model.fit(X_train, y_train)
#
# # 预测波动性和方向性
# y_pred = model.predict(X_test)
# pred_volatility = y_pred[:, 0]  # 波动性预测



# log_ratio_2 = log_ratio_error(y_test['FX_risk_score'], pred_volatility)
# print(f"Log Ratio Error 2: {log_ratio_2:.4f}")
#
# pred_direction = y_pred[:, 1]   # 方向性预测
# from sklearn.metrics import accuracy_score
# accuracy_direction = accuracy_score(y_test['Future_Direction'], np.sign(pred_direction))
# print(f"Accuracy (Direction) 2: {accuracy_direction:.4f}")








# test_data['Risk_Signal'] = 0
# threshold_volatility = 0.003
# # print('pred_direction', pred_direction)
# print('pred_volatility.min(), pred_volatility.max()', pred_volatility.min(), pred_volatility.max())
# test_data.loc[(pred_volatility > threshold_volatility) & (pred_direction == -1), 'Risk_Signal'] = -1  # 高风险
# test_data.loc[(pred_volatility > threshold_volatility) & (pred_direction == 1), 'Risk_Signal'] = 1




# correlation = data[['atr_14_lagged', 'FX_risk_score']].corr()
# print('correlation between atr and FX', correlation)
# correlation_matrix = data.corr()
# print(correlation_matrix['FX_risk_score'].sort_values(ascending=False))


# data['Signal'] = y_class.map({0: 1, 1: 0, 2: -1})
# data["Signal"] = data["Signal"].astype(int)
# print(data["Signal"].value_counts())
#
# initial_cash = 10000  # Starting capital
# cash = initial_cash
# position = 0  # Number of units held (long/short)
# trade_size = 200  # Amount traded per position
#
# data["Returns"] = data["Close"].pct_change()  # Daily returns
# data["Strategy_Returns"] = data["Signal"].shift(1) * data["Returns"]  # Apply trading signal to returns
#
# # Simulate cash balance
# for i in range(1, len(data)):
#     if data["Signal"].iloc[i] == 1:  # Buy
#         position = trade_size / data["Close"].iloc[i]  # Buy FX with trade size
#         cash -= trade_size  # Deduct cash spent
#     elif data["Signal"].iloc[i] == -1:  # Sell
#         cash += position * data["Close"].iloc[i]  # Sell and update cash balance
#         position = 0  # Exit position
#
# # Final Portfolio Value
# portfolio_value = cash + (position * data["Close"].iloc[-1])  # Final value including held position
# print(f"Final Portfolio Value: ${portfolio_value:.2f}")
# print(f"Net Profit/Loss: ${portfolio_value - initial_cash:.2f}")

# from FX_backtesting import Backtest, Strategy
#
# class FXStrategy(Strategy):
#     def init(self):
#         self.signal = self.I(lambda: data['Signal'].values)
#
#     def next(self):
#         if self.signal[-1] == 1:
#             self.buy(size=1)
#         elif self.signal[-1] == -1:
#             self.sell(size=1)
#
# bt = Backtest(data, FXStrategy, cash=100000, commission=0.002)
# stats = bt.run()
# print('stats', stats)
