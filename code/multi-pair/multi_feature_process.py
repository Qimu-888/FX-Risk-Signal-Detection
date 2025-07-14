import pandas as pd
from process_plot_data import *
fx_daily = pd.read_excel('/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/usd_cnh_2015.xlsx', parse_dates=["Date"])

import os

# 公共路径提取为变量
features_base_path = '/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/multi/new_features'

'''# daily'''
daily_subfolders = ['GPR_daily', 'sentiment_score', 'IR','gold', 'oil','stock_daily']

# 使用 os.path.join 动态拼接完整路径
daily_folders = [os.path.join(features_base_path, subfolder) for subfolder in daily_subfolders]
daily_index = pd.read_excel('/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/index_daily.xlsx')
merged_df_daily = merge_selected_folders_with_fx(daily_index, daily_folders,'merged/merged_daily_data.xlsx')
# merged_df_daily.to_csv('merged/merged_daily_data.csv', index= False)


'''
#monthly: EPU,subEPU, FER, currency_suppy, CPI, BoP, employment, Treasury_bonds
'''
monthly_subfolders = ['CPI', 'EPU', 'subEPU', 'BoP', 'employment', 'FER', 'currency_supply','Treasury_bonds','stock_monthly','GPR_monthly']
monthly_folders = [os.path.join(features_base_path, subfolder) for subfolder in monthly_subfolders]
us_cpu = pd.read_excel('/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/features/subEPU/US-CPU_2015.xlsx')
merged_df_month = merge_selected_folders_with_fx(us_cpu, monthly_folders,'merged/merged_monthly_data.xlsx')


# Resample to daily frequency and forward fill
merged_df_month['date'] = pd.to_datetime(merged_df_month['date'])
merged_df_month = merged_df_month.set_index('date')
new_index = pd.date_range(start=merged_df_month.index.min(), end='2024-12-31', freq='D')

merged_df_month = merged_df_month.reindex(new_index).ffill()
# merged_df_month.to_csv('merged/merged_monthly_data.csv', index= False)


#quarterly: GDP
qr_subfolders = ['GDP_qr']
qr_folders = [os.path.join(features_base_path, subfolder) for subfolder in qr_subfolders]
qr_index = pd.read_excel('/Users/daisymm/PycharmProjects/pythonProject/FX_Vanguard_Backend/FX/qr_index.xlsx')
merged_df_qr = merge_selected_folders_with_fx(qr_index, qr_folders, 'merged/qr_test.xlsx')
#fill
merged_df_qr['Date'] = pd.to_datetime(merged_df_qr['Date'])
merged_df_qr = merged_df_qr.set_index('Date')
new_index_qr = pd.date_range(start=merged_df_qr.index.min(), end='2024-12-31', freq='D')
merged_df_qr = merged_df_qr.reindex(new_index_qr).ffill()
# merged_df_qr.to_csv('merged/qr_test.csv')



currency_pairs = ['USDJPY', 'USDCNH', 'JPYEUR', 'JPYCNY', 'EURUSD', 'EURJPY', 'EURCNH']

global_variables = {
    'daily': ['sentiment_score','GPRD','GPRD_ACT','GPRD_THREAT','oil-Daily-Volatility'],
    'monthly': ['GPR', 'GEPU_current','GEPU_ppp'],
    'quarterly': ['GPRD_ACT']
}


def extract_currencies(pair):
    for i in range(3, len(pair)):
        if pair[:i] in pair[i:]:
            continue
        return pair[:i], pair[i:]
    raise ValueError(f"无法解析货币对: {pair}")

def get_country_specific_vars(df, cur1, cur2):
    return [col for col in df.columns if cur1 in col or cur2 in col]

def get_global_vars(df, global_vars):
    return [col for col in global_vars if col in df.columns]

def build_midas_ready_features(currency_pairs, merged_data, global_variables, midas_transform):
    midas_feature_dict = {}

    for pair in currency_pairs:
        cur1, cur2 = extract_currencies(pair)
        print(f"\n▶️ 正在处理货币对: {pair} ({cur1}, {cur2})")

        # --- 提取每日数据 ---
        daily_df = merged_data['daily']
        daily_country_vars = get_country_specific_vars(daily_df, cur1, cur2)
        daily_global_vars = get_global_vars(daily_df, global_variables.get('daily', []))
        daily_base_cols = [col for col in daily_df.columns if 'date' in col.lower() or daily_df[col].dtype == 'datetime64[ns]']
        merged_df_daily_pair = daily_df[daily_base_cols + daily_country_vars + daily_global_vars].copy()
        # merged_df_daily_pair.to_csv('merged_daily_pair.csv', index=False)

        # --- 提取月度数据 ---
        monthly_df = merged_data['monthly']
        monthly_vars = get_country_specific_vars(monthly_df, cur1, cur2)
        monthly_global_vars = get_global_vars(monthly_df, global_variables.get('monthly', []))
        monthly_all_vars = list(set(monthly_vars + monthly_global_vars))
        merged_df_monthy_pair = monthly_df[monthly_all_vars].copy()


        # --- 提取季度数据 ---
        quarterly_df = merged_data['quarterly']
        quarterly_vars = get_country_specific_vars(quarterly_df, cur1, cur2)
        quarterly_global_vars = get_global_vars(quarterly_df, global_variables.get('quarterly', []))
        quarterly_all_vars = list(set(quarterly_vars + quarterly_global_vars))
        merged_df_qr_pair = quarterly_df[quarterly_all_vars].copy()

        print('monthly_all_vars', monthly_all_vars)

        # --- 做 MIDAS 转换（只用 country-specific 的 monthly 和 quarterly）---
        midas_features = midas_transform(
            merged_df_monthy_pair, merged_df_qr_pair,
            monthly_all_vars, quarterly_all_vars,'multi','single',
            1.5, 2.5, 1.5, 6, 4)

        # merged_features = midas_features.merge(merged_df_daily_pair, left_index=True, right_index=True, how='left')
        # merged_features.index.name = 'date'

        def _ensure_datetime_index(df, date_col='date'):
            # 自动检查是否已经是datetime index，不是就转换
            if date_col in df.columns:
                df = df.copy()
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
            else:
                df = df.copy()
                df.index = pd.to_datetime(df.index)
            # 去重并去掉NaT
            df = df[~df.index.duplicated(keep='first')]
            df = df[~df.index.isna()]
            return df

        midas_features = _ensure_datetime_index(midas_features)
        merged_df_daily_pair = _ensure_datetime_index(merged_df_daily_pair)

        # 推荐以 daily 为主表
        merged_features = merged_df_daily_pair.join(midas_features, how='left')
        # 或以 midas 为主表（通常日度为主更稳）
        # merged_features = midas_features.join(merged_df_daily_pair, how='left')

        merged_features.index.name = 'date'

        merged_features = merged_features.dropna(how='any')
        merged_features.to_excel('merged_test.xlsx', index=True)

        midas_feature_dict[pair] = merged_features
    return midas_feature_dict



merged_data = {
    'daily': merged_df_daily,
    'monthly': merged_df_month,
    'quarterly': merged_df_qr
}

# 用于存储所有货币对的 MIDAS 特征

all_midas_features = build_midas_ready_features(currency_pairs, merged_data, global_variables, midas_transform)
# for pair in currency_pairs:
#     midas_features = build_midas_ready_features([pair], merged_data, global_variables, midas_transform)
#     all_midas_features[pair] = midas_features
