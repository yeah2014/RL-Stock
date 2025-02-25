import os
import pickle
import pandas as pd
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from rlenv.StockTradingEnv0 import StockTradingEnv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font = fm.FontProperties(fname='font/wqy-microhei.ttc')
# plt.rc('font', family='Source Han Sans CN')
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据流设计：
# 训练/测试数据分离：防止过拟合
# 时间序列排序：保证市场数据时序正确性
# 路径自动替换：通过字符串替换实现数据集切换

#  2. 强化学习流程：
#graph TD
#    A[训练数据] --> B[创建环境]
#    B --> C[初始化PPO2模型]
#    C --> D[策略学习]
#    D --> E[测试数据]
#    E --> F[策略执行]
#    F --> G[收益记录]

# 3. 关键参数说明：

# total_timesteps=1e4: 控制训练强度，可增加至1e5提升模型表现
# MlpPolicy: 使用多层感知机作为策略网络
# tensorboard_log: 保存训练过程数据用于可视化分析

# 4. 扩展建议：
# 添加模型保存/加载功能（save/load方法）
# 实现早停机制（Early Stopping）
# 增加回调函数监控训练过程
# - 添加超参数调节功能

def stock_trade(stock_file):
    """执行单只股票的交易训练和测试
    Args:
        stock_file: 股票数据文件路径（训练集）
    Returns:
        day_profits: 每日收益列表（测试集）
    
    工作原理：
    1. 训练阶段：使用历史数据训练PPO2策略
    2. 测试阶段：在独立测试集上评估策略表现
    3. 数据流：训练数据 -> 环境 -> 智能体 -> 测试数据 -> 收益曲线
    """
    day_profits = [] # 记录每日收益

    # === 数据准备阶段 ===
    # 1. 加载并预处理训练数据
    df = pd.read_csv(stock_file)
    df = df.sort_values('date') # 确保按日期排序

    # === 模型训练阶段 ===
    # 创建向量化环境（PPO2算法要求）
    # 使用lambda延迟环境创建，确保每次训练都是全新环境
    # The algorithms require a vectorized environment to run
    env = DummyVecEnv([lambda: StockTradingEnv(df)])

    # 3. 初始化PPO2模型
    # MlpPolicy: 多层感知机策略网络
    # verbose=0: 关闭训练日志
    # tensorboard_log: 训练日志保存路径
    model = PPO2(MlpPolicy, env, verbose=0, tensorboard_log='./log')

    # 4. 执行训练（1万时间步）
    model.learn(total_timesteps=int(1e4)) #关键参数 total_timesteps=1e4: 控制训练强度，可增加至1e5提升模型表现

    # === 模型测试阶段 ===
    # 5. 加载测试数据集（通过替换路径中的'train'为'test'）
    df_test = pd.read_csv(stock_file.replace('train', 'test'))

    # 6. 创建测试环境
    env = DummyVecEnv([lambda: StockTradingEnv(df_test)])
    obs = env.reset() # 初始化环境观测

    # 7. 执行测试交易
    for i in range(len(df_test) - 1):
        action, _states = model.predict(obs) # 使用训练好的策略预测动作
        action_type = action[0][0]
        amount = action[0][1]
        print('-'*30) # 分隔线
        if action_type < 1:
            print(f'操作买入{amount}')
        elif action_type < 2:
            print(f'操作卖出{amount}')
        
        obs, rewards, done, info = env.step(action) # 执行动作
        profit = env.render() # 获取当前收益
        day_profits.append(profit) # 记录每日收益

        # 8. 环境终止检查（如爆仓等）
        if done: # 当环境返回done=True时提前终止
            break
    return day_profits


def find_file(path, name):
    # print(path, name)
    for root, dirs, files in os.walk(path):
        for fname in files:
            if name in fname:
                return os.path.join(root, fname)


def test_a_stock_trade(stock_code):
    """测试单只股票交易策略并可视化结果
    Args:
        stock_code: 股票代码（如'sh.600036'）
    
    工作原理：
    1. 数据获取：根据股票代码查找训练数据文件
    2. 策略执行：调用stock_trade进行训练和测试
    3. 可视化：生成收益曲线图并保存
    4. 输出：保存PNG格式的收益走势图到img目录
    """
    # 1. 查找训练数据文件（在./stockdata/train目录递归搜索）
    stock_file = find_file('./stockdata/train', str(stock_code))

    # 2. 执行交易策略（包含训练和测试阶段）
    daily_profits = stock_trade(stock_file)

    # 3. 创建可视化图表
    fig, ax = plt.subplots()

    # 绘制收益曲线（带橙色圆形标记）
    ax.plot(daily_profits, '-o', label=stock_code, marker='o', ms=10, alpha=0.7, mfc='orange')

    # 图表样式设置
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel('交易步数', fontproperties=font)
    plt.ylabel('累计收益 (元)', fontproperties=font)
    ax.legend(prop=font, loc='upper left')
    # plt.show()
    plt.savefig(f'./img/{stock_code}.png')
    plt.close()  # 关闭图表释放内存


def multi_stock_trade():
    """批量执行多只股票交易策略
    工作原理：
    1. 股票代码遍历：在指定范围内（600000-603000）遍历所有股票代码
    2. 数据文件查找：在./stockdata/train目录递归搜索对应股票的训练数据
    3. 并行交易执行：对每只股票独立执行训练和测试流程
    4. 结果收集存储：将各股票收益数据序列化保存到.pkl文件
    
    设计特点：
    - 容错机制：单个股票交易失败不影响整体流程
    - 批量处理：支持大规模股票策略验证
    - 结果持久化：便于后续统计分析
    """
    start_code = 600000  # 起始股票代码（沪市A股典型起始代码）
    max_num = 3000       # 最大处理数量（覆盖沪市600000-603000代码段）

    group_result = []    # 存储所有股票的收益数据

    # 遍历股票代码范围（左闭右开区间）
    for code in range(start_code, start_code + max_num):
        # 在训练数据目录中查找对应股票代码的数据文件
        stock_file = find_file('./stockdata/train', str(code))
        if stock_file:
            try:
                # 执行单只股票交易策略（包含训练和测试阶段）
                profits = stock_trade(stock_file)
                group_result.append(profits)  # 记录收益数据
            except Exception as err:
                # 异常处理：打印错误信息但继续执行后续股票
                print(err)
    # 序列化存储结果（使用二进制pickle格式保证存储效率）
    with open(f'code-{start_code}-{start_code + max_num}.pkl', 'wb') as f:
        pickle.dump(group_result, f)


if __name__ == '__main__':
    # multi_stock_trade()
    # test_a_stock_trade('sh.600036')
    test_a_stock_trade('sz.000858')
    # ret = find_file('./stockdata/train', '600036')
    # print(ret)
