import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os

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
        weights = {pair: 1.0 / len(pairs) for pair in pairs}

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

    ax1.bar(x - width, low_risk_values, width, label='low risk', color='green')
    ax1.bar(x, medium_risk_values, width, label='medium risk', color='orange')
    ax1.bar(x + width, high_risk_values, width, label='high risk', color='red')

    ax1.set_title('risk distribution of corresponding currency pairs', fontsize=16)
    ax1.set_xlabel('pairs', fontsize=14)
    ax1.set_ylabel('percent (%)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(pairs)
    ax1.legend()

    # 货币对权重分布饼图
    ax2.pie(weights, labels=pairs, autopct='%1.1f%%', startangle=90)
    ax2.axis('equal')
    ax2.set_title('weight distribution', fontsize=16)

    # 加权风险分布饼图
    weighted_values = [weighted_risk['low_risk'], weighted_risk['medium_risk'], weighted_risk['high_risk']]
    ax3.pie(weighted_values, labels=['low risk', 'medium risk', 'high risk'],
            autopct='%1.1f%%', startangle=90, colors=['green', 'orange', 'red'])
    ax3.axis('equal')
    ax3.set_title('weighted risk distribution', fontsize=16)

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
        plt.colorbar(ticks=[0, 1, 2], label='risk level')
        plt.yticks([])
        plt.title(f'{pair} risk signal')
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
        prev_price = backtest_df['close'].iloc[i - 1]
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
                'risk_class': backtest_df['risk_class'].iloc[i - holding_days]
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
                    'risk_class': backtest_df['risk_class'].iloc[i - holding_days]
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
                    'risk_class': backtest_df['risk_class'].iloc[i - holding_days]
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
                prev_3_signals = backtest_df['risk_class'].iloc[i - 3:i].values
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
    ax1.plot(equity_curve.index, equity_curve.values, label='strategy return', color='blue', linewidth=2)

    # 价格曲线（次坐标轴）
    ax_price = ax1.twinx()
    ax_price.plot(price_data.index, price_data.values, color='gray', alpha=0.5, label='price')

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
    ax1.set_title(f'{pair} risk signal backtesting result', fontsize=16)
    ax1.set_ylabel('net value of the strategy', fontsize=12)
    ax_price.set_ylabel('price', fontsize=12)
    ax2.set_ylabel('risk signal', fontsize=12)

    # 添加图例
    ax1.legend(loc='upper left')
    ax_price.legend(loc='upper right')

    # 添加回测统计结果文本
    stats_text = (
        f"total trading times: {backtest_results['total_trades']}\n"
        f"winning ratio: {backtest_results['win_rate']:.2f}%\n"
        f"average return: {backtest_results['avg_return']:.2f}%\n"
        f"annual return: {backtest_results['annual_return']:.2f}%\n"
        f"maximum drawdown: {backtest_results['max_drawdown']:.2f}%\n"
        f"sharpe ratio: {backtest_results['sharpe_ratio']:.2f}"
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
        plt.title(f'{pair} monthly return hot map (%)', fontsize=16)
        plt.xlabel('month', fontsize=12)
        plt.ylabel('year', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{pair}_monthly_returns.png'), dpi=300)
        plt.close()

    # 3. 交易统计饼图
    if len(trades) > 0 and 'exit_reason' in trades.columns:
        exit_reason_counts = trades['exit_reason'].value_counts()

        plt.figure(figsize=(10, 6))
        plt.pie(exit_reason_counts.values, labels=exit_reason_counts.index, autopct='%1.1f%%')
        plt.title(f'Distribution of trade closing reasons for {pair}', fontsize=16)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{pair}_exit_reasons.png'), dpi=300)
        plt.close()