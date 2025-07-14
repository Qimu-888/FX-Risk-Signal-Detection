import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta

# =====================
# 1. 数据预处理
# =====================
def load_and_preprocess(file_path):
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 转换日期格式（假设有date列，如果没有需要调整）
    if 'date' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # 提取各货币对收盘价
    closes = df[[
        'EUR_close', 
        'JPY_close', 
        'CNY_close'
    ]].rename(columns={
        'EUR_close': 'EURUSD',
        'JPY_close': 'JPYUSD', 
        'CNY_close': 'CNYUSD'
    })
    
    # 计算对数收益率
    returns = np.log(closes).diff().dropna()
    return closes, returns

# =====================
# 2. 风险分析引擎
# =====================
class FXCorrelationRiskModel:
    def __init__(self, window=30, thresholds=(0.7, 0.8)):
        self.window = window  # 滚动窗口
        self.thresholds = thresholds  # (警告阈值, 危险阈值)
        
    def calculate_rolling_correlations(self, returns):
        """计算滚动相关系数矩阵"""
        corr_matrices = []
        currencies = returns.columns
        
        for i in range(len(returns)):
            if i < self.window:
                corr = np.full((3,3), np.nan)
            else:
                corr = returns.iloc[i-self.window:i].corr().values
            corr_matrices.append(corr)
            
        return pd.Series(corr_matrices, index=returns.index)
    
    def generate_signals(self, corr_series):
        """生成风险信号"""
        signals = []
        for ts, corr_matrix in corr_series.items():
            if pd.isna(corr_matrix).any():
                signals.append({'risk_level': 0, 'risk_pairs': []})
                continue
                
            # 提取上三角矩阵（避免重复）
            upper_triangle = [
                (0,1), (0,2), (1,2)  # EUR-JPY, EUR-CNY, JPY-CNY
            ]
            high_corr_pairs = []
            
            for i,j in upper_triangle:
                if abs(corr_matrix[i][j]) > self.thresholds[0]:
                    pair = f"{returns.columns[i]}-{returns.columns[j]}"
                    high_corr_pairs.append((
                        pair, 
                        'danger' if corr_matrix[i][j] > self.thresholds[1] else 'warning'
                    ))
            
            # 计算风险等级
            risk_level = min(len(high_corr_pairs)/3, 1.0)  # 3对组合最大风险1.0
            signals.append({
                'timestamp': ts,
                'risk_level': risk_level,
                'details': high_corr_pairs
            })
            
        return pd.DataFrame(signals).set_index('timestamp')

# =====================
# 3. 可视化模块
# =====================
def visualize_risk(signals, latest_corr):
    plt.figure(figsize=(16, 10))
    
    # 风险等级时间序列
    ax1 = plt.subplot(2,1,1)
    signals['risk_level'].plot(color='#FF6B6B', lw=1.5)
    plt.fill_between(signals.index, signals['risk_level'], 
                    color='#FFE8D6', alpha=0.3)
    plt.title('Portfolio Risk Level Over Time', fontsize=14)
    plt.axhline(0.5, color='#FFD93D', ls='--', label='Warning Threshold')
    plt.axhline(0.8, color='#FF2626', ls='--', label='Danger Threshold')
    plt.legend()
    
    # 最新相关系数矩阵
    ax2 = plt.subplot(2,1,2)
    corr_df = pd.DataFrame(
        latest_corr,
        columns=returns.columns,
        index=returns.columns
    )
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, annot=True, mask=mask, cmap='RdBu_r',
               vmin=-1, vmax=1, center=0,
               cbar_kws={'label': 'Correlation Coefficient'},
               annot_kws={'size': 12})
    plt.title('Latest Correlation Matrix (30-day window)', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10, rotation=0)
    
    plt.tight_layout()
    plt.show()

# =====================
# 4. 执行主程序
# =====================
if __name__ == "__main__":
    # 加载数据
    closes, returns = load_and_preprocess("./Desktop/DATA.xlsx")
    
    # 初始化风险模型
    risk_model = FXCorrelationRiskModel(window=30, thresholds=(0.65, 0.8))
    
    # 计算滚动相关性
    corr_series = risk_model.calculate_rolling_correlations(returns)
    
    # 生成风险信号
    risk_signals = risk_model.generate_signals(corr_series)
    
    # 可视化结果
    visualize_risk(risk_signals, corr_series.iloc[-1])
    
    # 输出最新信号
    latest_signal = risk_signals.iloc[-1]
    print(f"\n最新风险信号 [{latest_signal.name.strftime('%Y-%m-%d %H:%M')}]")
    print(f"综合风险等级: {latest_signal['risk_level']:.0%}")
    print("异常相关组合:")
    for pair, level in latest_signal['details']:
        print(f"  - {pair}: {'危险' if level == 'danger' else '警告'}")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# =====================
# 1. 数据预处理模块（增强版）
# =====================
def load_and_preprocess(file_path):
    """加载并预处理数据（增强错误处理）"""
    try:
        # 读取数据并清理空列
        df = pd.read_excel(file_path, parse_dates=['date']).dropna(axis=1, how='all')
        
        # 验证必要列存在
        required_cols = {'EUR_close', 'JPY_close', 'CNY_close', 'date'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"缺少必要列: {missing}")
            
        # 处理日期索引
        df = df.set_index('date').sort_index()
        df = df.ffill().bfill()  # 双向填充
        
        print("数据预处理完成，前3行样例：")
        print(df.head(3))
        return df
        
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        exit()

def calculate_features(df):
    """计算技术指标和相关性（稳健版本）"""
    try:
        # 计算波动率
        for currency in ['EUR', 'JPY', 'CNY']:
            df[f'{currency}_vol'] = df[f'{currency}_close'].rolling(20, min_periods=10).std()
        
        # 计算RSI
        for currency in ['EUR', 'JPY', 'CNY']:
            close_series = df[f'{currency}_close']
            delta = close_series.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(14, min_periods=7).mean()
            avg_loss = loss.rolling(14, min_periods=7).mean()
            
            rs = avg_gain / (avg_loss + 1e-10)
            df[f'{currency}_rsi'] = 100 - (100 / (1 + rs))
        
        # 计算滚动相关性
        corr_features = []
        window_size = 30
        for i in range(len(df)):
            if i < window_size:
                corr_features.append([np.nan]*3)
            else:
                window = df[['EUR_close', 'JPY_close', 'CNY_close']].iloc[i-window_size:i]
                corr_matrix = window.corr()
                corr_features.append([
                    corr_matrix.loc['EUR_close', 'JPY_close'],
                    corr_matrix.loc['EUR_close', 'CNY_close'],
                    corr_matrix.loc['JPY_close', 'CNY_close']
                ])
        
        df[['corr_EUR_JPY', 'corr_EUR_CNY', 'corr_JPY_CNY']] = corr_features
        return df.bfill().dropna()
    
    except Exception as e:
        print(f"特征计算失败: {str(e)}")
        exit()

# =====================
# 2. LSTM模型构建（优化版）
# =====================
class RiskPredictor:
    def __init__(self, time_steps=30, n_features=9):
        self.time_steps = time_steps
        self.n_features = n_features
        self.model = self.build_model()
        
    def build_model(self):
        model = Sequential([
            LSTM(64, return_sequences=True,
                input_shape=(self.time_steps, self.n_features),
                kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            Dropout(0.3),
            LSTM(32),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.AUC(name='auc')]
        )
        return model

# =====================
# 3. 训练流程（带数据验证）
# =====================
def prepare_training_data(df):
    """准备训练数据（带形状校验）"""
    try:
        # 定义特征列
        feature_cols = [
            'EUR_vol', 'EUR_rsi',
            'JPY_vol', 'JPY_rsi',
            'CNY_vol', 'CNY_rsi',
            'corr_EUR_JPY', 'corr_EUR_CNY', 'corr_JPY_CNY'
        ]
        
        # 创建标签（未来5日风险）
        df['risk_label'] = (df['EUR_close'].pct_change(5).shift(-5) < -0.02).astype(int).bfill()
        
        # 标准化数据
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df[feature_cols])
        
        # 创建时间序列数据集
        X, y = [], []
        time_steps = 30
        for i in range(len(scaled_features) - time_steps -5):
            X.append(scaled_features[i:i+time_steps])
            y.append(df['risk_label'].iloc[i+time_steps+5])
            
        X = np.array(X)
        y = np.array(y)
        
        # 数据校验
        if len(X) == 0 or len(y) == 0:
            raise ValueError("训练数据不足，请检查输入数据量")
            
        # 划分训练测试集
        split = int(len(X)*0.8)
        return (X[:split], y[:split]), (X[split:], y[split:]), scaler
        
    except Exception as e:
        print(f"训练数据准备失败: {str(e)}")
        exit()

# =====================
# 4. 实时监控系统（修复版）
# =====================
class RiskMonitor:
    def __init__(self, model, scaler, feature_cols):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.buffer = []
        self.time_steps = 30
        
    def update(self, new_data):
        """更新数据（带缓冲区管理）"""
        self.buffer.append(new_data)
        if len(self.buffer) > self.time_steps:
            self.buffer.pop(0)
            
    def generate_signal(self):
        """生成风险信号（带错误处理）"""
        try:
            if len(self.buffer) < self.time_steps:
                return {
                    "status": "等待数据",
                    "required": self.time_steps - len(self.buffer),
                    "risk_level": 0.0,
                    "recommendation": "数据不足"
                }
                
            # 准备输入数据
            input_data = pd.DataFrame(self.buffer)[self.feature_cols]
            scaled_data = self.scaler.transform(input_data)
            
            # LSTM预测
            lstm_risk = self.model.predict(scaled_data[np.newaxis, ...], verbose=0)[0][0]
            
            # 传统指标
            current_corrs = {
                'EUR_JPY': input_data['corr_EUR_JPY'].iloc[-1],
                'EUR_CNY': input_data['corr_EUR_CNY'].iloc[-1],
                'JPY_CNY': input_data['corr_JPY_CNY'].iloc[-1]
            }
            corr_risk = sum(abs(v) > 0.7 for v in current_corrs.values()) / 3
            
            # 综合风险
            combined_risk = 0.7 * lstm_risk + 0.3 * corr_risk
            return {
                "status": "正常",
                "risk_level": combined_risk,
                "recommendation": self.get_action(combined_risk),
                "correlations": current_corrs
            }
            
        except Exception as e:
            print(f"信号生成失败: {str(e)}")
            return {
                "status": "错误",
                "risk_level": 0.0,
                "recommendation": "系统错误"
            }
    
    @staticmethod
    def get_action(risk_level):
        if risk_level >= 0.8:
            return "立即平仓高风险头寸"
        elif risk_level >= 0.6:
            return "减持相关货币对"
        elif risk_level >= 0.4:
            return "保持监控"
        else:
            return "维持敞口"

# =====================
# 主执行流程（安全版本）
# =====================
if __name__ == "__main__":
    # 加载并预处理数据
    df = load_and_preprocess("./Desktop/DATA.xlsx")
    df = calculate_features(df)
    
    # 准备训练数据
    (X_train, y_train), (X_test, y_test), scaler = prepare_training_data(df)
    print(f"\n训练数据形状: {X_train.shape}")
    
    # 训练模型
    predictor = RiskPredictor()
    print("\n开始模型训练...")
    history = predictor.model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=64,
        verbose=1
    )
    
    # 可视化训练结果
    plt.figure(figsize=(12,6))
    plt.plot(history.history['auc'], label='训练集AUC')
    plt.plot(history.history['val_auc'], label='验证集AUC')
    plt.title('模型训练进度')
    plt.legend()
    plt.show()
    
    # 初始化监控系统
    feature_cols = [
        'EUR_vol', 'EUR_rsi',
        'JPY_vol', 'JPY_rsi',
        'CNY_vol', 'CNY_rsi',
        'corr_EUR_JPY', 'corr_EUR_CNY', 'corr_JPY_CNY'
    ]
    monitor = RiskMonitor(predictor.model, scaler, feature_cols)
    
    # 模拟实时数据流
    print("\n模拟实时监控...")
    for i in range(30, len(df), 5):
        try:
            # 获取最新数据
            new_data = df.iloc[i][feature_cols].to_dict()
            monitor.update(new_data)
            
            # 生成信号
            signal = monitor.generate_signal()
            
            # 打印结果
            print(f"\n日期: {df.index[i].strftime('%Y-%m-%d')}")
            print(f"状态: {signal['status']}")
            if signal['status'] == '正常':
                print(f"综合风险: {signal['risk_level']:.0%}")
                print(f"建议操作: {signal['recommendation']}")
                print("当前相关性:")
                for k, v in signal['correlations'].items():
                    print(f"  {k}: {v:.2f}")
            print("="*50)
            
        except Exception as e:
            print(f"处理第{i}条数据时出错: {str(e)}")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# =====================
# 1. 数据预处理模块
# =====================
def load_and_preprocess(file_path):
    """加载并预处理数据"""
    df = pd.read_excel(file_path, parse_dates=['date']).dropna(axis=1, how='all')
    
    # 验证必要列存在
    required_cols = {'EUR_close', 'JPY_close', 'CNY_close', 'date'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {missing}")
        
    df = df.set_index('date').sort_index().ffill().bfill()
    return df

def calculate_features(df):
    """计算技术指标和相关性"""
    # 计算波动率（20日）
    for currency in ['EUR', 'JPY', 'CNY']:
        df[f'{currency}_vol'] = df[f'{currency}_close'].rolling(20, min_periods=10).std()
    
    # 计算RSI（14日）
    for currency in ['EUR', 'JPY', 'CNY']:
        delta = df[f'{currency}_close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14, min_periods=7).mean()
        avg_loss = loss.rolling(14, min_periods=7).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df[f'{currency}_rsi'] = 100 - (100 / (1 + rs))
    
    # 计算滚动相关性（30日）
    corr_features = []
    window_size = 30
    for i in range(len(df)):
        if i < window_size:
            corr_features.append([np.nan]*3)
        else:
            window = df[['EUR_close', 'JPY_close', 'CNY_close']].iloc[i-window_size:i]
            corr_matrix = window.corr()
            corr_features.append([
                corr_matrix.loc['EUR_close', 'JPY_close'],
                corr_matrix.loc['EUR_close', 'CNY_close'],
                corr_matrix.loc['JPY_close', 'CNY_close']
            ])
    
    df[['corr_EUR_JPY', 'corr_EUR_CNY', 'corr_JPY_CNY']] = corr_features
    return df.bfill().dropna()

# =====================
# 2. 信号生成逻辑
# =====================
class TradingSignalGenerator:
    def __init__(self, df):
        self.df = df
        self.signals = []
    
    def analyze_all(self):
        """批量分析所有数据点"""
        for i in range(len(self.df)):
            if i < 30:  # 跳过前30天数据不足的窗口
                continue
                
            signal = {
                'date': self.df.index[i],
                'EUR': self._analyze_currency('EUR', i),
                'JPY': self._analyze_currency('JPY', i),
                'CNY': self._analyze_currency('CNY', i),
                'correlation_risk': self._get_correlation_risk(i)
            }
            self.signals.append(signal)
        return pd.DataFrame(self.signals).set_index('date')
    
    def _analyze_currency(self, currency, idx):
        """分析单个货币对的交易信号"""
        data = self.df.iloc[idx]
        signal = {}
        
        # 波动率信号
        vol = data[f'{currency}_vol']
        signal['volatility'] = 'High' if vol > 0.015 else 'Normal'
        
        # RSI信号
        rsi = data[f'{currency}_rsi']
        if rsi > 70:
            signal['rsi'] = 'Overbought'
        elif rsi < 30:
            signal['rsi'] = 'Oversold'
        else:
            signal['rsi'] = 'Neutral'
            
        # 价格动量
        pct_5d = self.df[f'{currency}_close'].iloc[idx] / self.df[f'{currency}_close'].iloc[idx-5] - 1
        signal['momentum'] = 'Up' if pct_5d > 0 else 'Down'
        
        return signal
    
    def _get_correlation_risk(self, idx):
        """获取相关性风险"""
        corrs = {
            'EUR_JPY': self.df['corr_EUR_JPY'].iloc[idx],
            'EUR_CNY': self.df['corr_EUR_CNY'].iloc[idx],
            'JPY_CNY': self.df['corr_JPY_CNY'].iloc[idx]
        }
        return {k: 'High' if abs(v) > 0.7 else 'Normal' for k, v in corrs.items()}
    
    def generate_trading_recommendations(self):
        """生成交易建议"""
        recommendations = []
        for signal in self.signals:
            rec = {'date': signal['date']}
            
            # 生成货币对建议
            for currency in ['EUR', 'JPY', 'CNY']:
                status = signal[currency]
                action = []
                
                if status['volatility'] == 'High':
                    action.append('Reduce exposure')
                if status['rsi'] == 'Overbought':
                    action.append('Consider selling')
                elif status['rsi'] == 'Oversold':
                    action.append('Consider buying')
                if status['momentum'] == 'Up':
                    action.append('Trend up')
                else:
                    action.append('Trend down')
                    
                rec[currency] = ' | '.join(action)
            
            # 相关性建议
            high_corr = [k for k, v in signal['correlation_risk'].items() if v == 'High']
            if high_corr:
                rec['correlation_advice'] = f"对冲相关货币对: {', '.join(high_corr)}"
            else:
                rec['correlation_advice'] = '相关性风险正常'
                
            recommendations.append(rec)
            
        return pd.DataFrame(recommendations).set_index('date')


    def generate_hedging_strategy(self):
        """生成对冲策略"""
        # 计算最新相关性
        latest_data = self.df.iloc[-1]
        
        # 计算各货币对波动率
        volatilities = {
            'USD/CNH': latest_data['CNY_vol'],
            'EUR/USD': latest_data['EUR_vol'],
            'USD/JPY': latest_data['JPY_vol']
        }
        
        # 确定主要和对冲货币对
        main_pair = max(volatilities.items(), key=lambda x: x[1])[0]
        
        # 根据相关性选择对冲货币对
        correlations = {
            'EUR/USD': latest_data['corr_EUR_CNY'],
            'USD/JPY': latest_data['corr_JPY_CNY']
        }
        
        hedge_pair = min(correlations.items(), key=lambda x: abs(x[1]))[0]
        
        # 计算对冲有效性
        hedge_effectiveness = self._calculate_hedge_effectiveness(main_pair, hedge_pair)
        
        # 生成对冲策略建议
        strategy = {
            '对冲策略': {
                '主要货币对': {
                    '币种': main_pair,
                    '权重': '70%'
                },
                '对冲货币对': {
                    '币种': hedge_pair,
                    '权重': '30%'
                },
                '对冲效果分析': {
                    '对冲有效性': f"{hedge_effectiveness:.0%}",
                    '建议配比': '7:3',
                    '备选对冲': self._get_alternative_hedge(main_pair, hedge_pair)
                }
            }
        }
        
        return strategy
    
    def _calculate_hedge_effectiveness(self, main_pair, hedge_pair):
        """计算对冲有效性"""
        # 使用30天数据计算对冲效果
        window = self.df.last('30D')
        
        # 简化计算，使用相关性的反向强度作为有效性指标
        correlation = abs(window['corr_EUR_CNY'].mean())
        effectiveness = 1 - correlation
        return min(effectiveness * 1.2, 0.95)  # 调整系数并设置上限
        
    def _get_alternative_hedge(self, main_pair, hedge_pair):
        """获取备选对冲货币对"""
        pairs = {'USD/CNH', 'EUR/USD', 'USD/JPY'}
        return list(pairs - {main_pair, hedge_pair})[0]

# =====================
# 3. 执行主流程
# =====================
if __name__ == "__main__":
    # 加载和处理数据
    df = load_and_preprocess("./Desktop/DATA.xlsx")
    df = calculate_features(df)
    
    # 生成交易信号
    print("\n生成交易信号...")
    signal_generator = TradingSignalGenerator(df)
    signals = signal_generator.analyze_all()
    recommendations = signal_generator.generate_trading_recommendations()
    
    # 保存结果
    recommendations.to_excel("trading_signals.xlsx")
    
    # 打印最新信号
    print("\n最新交易信号:")
    print(recommendations.tail(3))
    
    # 可视化风险趋势
    plt.figure(figsize=(14, 6))
    df['corr_EUR_JPY'].rolling(30).mean().plot(label='EUR-JPY 30日平均相关性')
    df['EUR_vol'].plot(label='EUR波动率', secondary_y=True, alpha=0.3)
    plt.title('市场风险趋势分析')
    plt.legend()
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================
# 1. 数据预处理模块
# =====================
def load_and_preprocess(file_path):
    """加载并预处理数据"""
    df = pd.read_excel(file_path, parse_dates=['date']).dropna(axis=1, how='all')
    
    # 验证必要列存在
    required_cols = {'EUR_close', 'JPY_close', 'CNY_close', 'date'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要列: {missing}")
        
    df = df.set_index('date').sort_index().ffill().bfill()
    return df

def calculate_features(df):
    """计算技术指标"""
    # 计算20日波动率
    for currency in ['EUR', 'JPY', 'CNY']:
        df[f'{currency}_vol'] = df[f'{currency}_close'].rolling(20, min_periods=10).std()
    
    # 计算14日RSI
    for currency in ['EUR', 'JPY', 'CNY']:
        delta = df[f'{currency}_close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14, min_periods=7).mean()
        avg_loss = loss.rolling(14, min_periods=7).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df[f'{currency}_rsi'] = 100 - (100 / (1 + rs))
    
    # 计算30日滚动相关性
    corr_features = []
    for i in range(len(df)):
        if i < 30:
            corr_features.append([np.nan]*3)
        else:
            window = df[['EUR_close', 'JPY_close', 'CNY_close']].iloc[i-30:i]
            corr_matrix = window.corr()
            corr_features.append([
                corr_matrix.loc['EUR_close', 'JPY_close'],
                corr_matrix.loc['EUR_close', 'CNY_close'],
                corr_matrix.loc['JPY_close', 'CNY_close']
            ])
    df[['corr_EUR_JPY', 'corr_EUR_CNY', 'corr_JPY_CNY']] = corr_features
    return df.bfill()

# =====================
# 2. 信号生成逻辑
# =====================
class DailySignalGenerator:
    def __init__(self, df):
        self.df = df.last('30D')  # 获取最近30天数据
        self.signals = []
    
    def generate_signals(self):
        """生成每日交易信号"""
        for date in self.df.index:
            signal = {'日期': date.strftime('%Y-%m-%d')}
            
            # 各货币对信号
            for currency in ['EUR', 'JPY', 'CNY']:
                signal.update(self._get_currency_signal(currency, date))
            
            # 相关性风险
            signal.update(self._get_correlation_risk(date))
            
            self.signals.append(signal)
            
        return pd.DataFrame(self.signals).set_index('日期')
    
    def _get_currency_signal(self, currency, date):
        """获取单个货币信号"""
        data = self.df.loc[date]
        prefix = f"{currency}USD"
        
        # 价格动量 (5日)
        pct_5d = self.df.loc[date, f'{currency}_close'] / self.df[f'{currency}_close'].shift(5).loc[date] - 1
        
        # 生成交易信号
        return {
            f'{prefix}_方向': '买入' if pct_5d > 0 else '卖出',
            f'{prefix}_强度': self._get_strength(data[f'{currency}_vol']),
            f'{prefix}_RSI状态': self._get_rsi_signal(data[f'{currency}_rsi'])
        }
    
    def _get_strength(self, vol):
        """波动率强度"""
        if vol > 0.02: return '高风险'
        elif vol > 0.015: return '中风险'
        else: return '低风险'
    
    def _get_rsi_signal(self, rsi):
        """RSI信号"""
        if rsi > 70: return '超买'
        elif rsi < 30: return '超卖'
        else: return '中性'
    
    def _get_correlation_risk(self, date):
        """获取相关性风险"""
        corrs = {
            'EUR-JPY': self.df.loc[date, 'corr_EUR_JPY'],
            'EUR-CNY': self.df.loc[date, 'corr_EUR_CNY'],
            'JPY-CNY': self.df.loc[date, 'corr_JPY_CNY']
        }
        high_corr = [k for k, v in corrs.items() if abs(v) > 0.7]
        return {
            '最高相关性': high_corr[0] if high_corr else '无',
            '对冲建议': f"对冲 {','.join(high_corr)}" if high_corr else '无需对冲'
        }


# 3. 执行主流程
# =====================
if __name__ == "__main__":
    # 加载和处理数据
    df = load_and_preprocess("./Desktop/DATA.xlsx")
    df = calculate_features(df)
    
    # 检查数据完整性
    if len(df) < 30:
        print("错误：数据不足30天")
        exit()
    
    # 生成信号
    generator = DailySignalGenerator(df)
    signals = generator.generate_signals()

    # 生成对冲策略
    generator = TradingSignalGenerator(df)
    hedging_strategy = generator.generate_hedging_strategy()
    
    # 打印对冲策略
    print("\n对冲策略分析:")
    print(f"主要货币对: {hedging_strategy['对冲策略']['主要货币对']['币种']} "
          f"(权重: {hedging_strategy['对冲策略']['主要货币对']['权重']})")
    print(f"对冲货币对: {hedging_strategy['对冲策略']['对冲货币对']['币种']} "
          f"(权重: {hedging_strategy['对冲策略']['对冲货币对']['权重']})")
    print(f"对冲有效性: {hedging_strategy['对冲策略']['对冲效果分析']['对冲有效性']}")
    print(f"建议配比: {hedging_strategy['对冲策略']['对冲效果分析']['建议配比']}")
    print(f"备选对冲: {hedging_strategy['对冲策略']['对冲效果分析']['备选对冲']}")
    
    # 保存结果
    signals.to_excel("近30天交易信号.xlsx")
    
    # 打印结果
    print("\n最近30天交易信号：")
    print(signals.tail(7))  # 展示最近7天

    
    # 可视化关键指标
    plt.figure(figsize=(14,6))
    df[['EUR_close', 'JPY_close', 'CNY_close']].last('30D').plot(
        secondary_y=['JPY_close', 'CNY_close'],
        title="近30天价格走势"
    )
    plt.show()