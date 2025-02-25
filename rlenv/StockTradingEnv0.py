import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

# 账户余额最大值（使用32位整数最大值防止数值溢出）
MAX_ACCOUNT_BALANCE = 2147483647  
# 单只股票最大持有数量（使用32位整数最大值）
MAX_NUM_SHARES = 2147483647
# 单股价格上限（根据A股市场设定，科创板/创业板需调整）       
MAX_SHARE_PRICE = 5000  
# 最大成交量限制（1000亿股）防止数据异常值
MAX_VOLUME = 1000e8  
# 最大成交额限制（300亿元）用于特征归一化
MAX_AMOUNT = 3e10  
# 最大持仓种类限制（控制风险敞口）            
MAX_OPEN_POSITIONS = 5  
# 最大训练步数（防止无限循环）          
MAX_STEPS = 20000  
# 最大单日涨跌幅限制（100%对应±1.0）         
MAX_DAY_CHANGE = 1  

# 初始账户余额（1万元初始资金）            
INITIAL_ACCOUNT_BALANCE = 1000000 


class StockTradingEnv(gym.Env):
    # 常见的模式包括'human'、'ansi'、'rgb_array'等。'human'模式一般指将环境状态渲染到屏幕，方便人类观察，比如弹出窗口或终端输出。在用户的代码中，`metadata = {'render.modes': ['human']}`说明这个环境支持human渲染模式。
    # 打印人类可读的交易信息
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        """股票交易环境初始化
        Args:
            df: 包含股票历史数据的DataFrame
        """
        super(StockTradingEnv, self).__init__()

        self.df = df # 股票历史数据
        self.reward_range = (0, MAX_ACCOUNT_BALANCE) # 奖励范围

        # 定义动作空间：[动作类型, 数量比例]
        # 动作类型：0-1买入，1-2卖出，2-3持有
        # 数量比例：0-1表示操作资金/持股的比例
        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # 定义状态空间：19个归一化后的特征
        # 包含市场数据、账户状态、持仓信息等
        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(19,), dtype=np.float16)

    def _next_observation(self):
        """构建观测值（状态），所有特征归一化到0-1之间
        包含19个维度：
        0-3: OHCL价格归一化
        4-5: 成交量和成交额归一化
        6: 复权因子（除以10）
        7: 交易状态（0/1）
        8-12: 技术指标（涨跌幅、市盈率、市净率等）
        13-18: 账户状态（余额、最大净值、持仓等）
        """
        obs = np.array([
            # 市场基础数据 (0-3)
            self.df.loc[self.current_step, 'open'] / MAX_SHARE_PRICE,       # 0 开盘价归一化（除以5000）
            self.df.loc[self.current_step, 'high'] / MAX_SHARE_PRICE,       # 1 最高价归一化
            self.df.loc[self.current_step, 'low'] / MAX_SHARE_PRICE,        # 2 最低价归一化
            self.df.loc[self.current_step, 'close'] / MAX_SHARE_PRICE,      # 3 收盘价归一化
            
            # 市场交易量数据 (4-5)
            self.df.loc[self.current_step, 'volume'] / MAX_VOLUME,          # 4 成交量归一化（除以1000亿股）
            self.df.loc[self.current_step, 'amount'] / MAX_AMOUNT,          # 5 成交额归一化（除以300亿元）
            
            # 股票特殊属性 (6-7)
            self.df.loc[self.current_step, 'adjustflag'] / 10,              # 6 复权因子（0-10整数，除以10归一化）
            self.df.loc[self.current_step, 'tradestatus'] / 1,              # 7 交易状态（0停牌/1正常）
            
            # 技术指标 (8-12)
            self.df.loc[self.current_step, 'pctChg'] / 100,                 # 8 涨跌幅（百分比形式，除以100归一）
            self.df.loc[self.current_step, 'peTTM'] / 1e4,                  # 9 市盈率TTM（除以10,000处理异常值）
            self.df.loc[self.current_step, 'pbMRQ'] / 100,                  # 10 市净率MRQ（除以100归一）
            self.df.loc[self.current_step, 'psTTM'] / 100,                  # 11 市销率TTM（除以100归一）
            self.df.loc[self.current_step, 'pctChg'] / 1e3,                 # 12 精细涨跌幅（除以1000处理小数点后三位）
            
            # 账户状态 (13-18)
            self.balance / MAX_ACCOUNT_BALANCE,                             # 13 现金余额比例（当前余额/最大可能余额）
            self.max_net_worth / MAX_ACCOUNT_BALANCE,                       # 14 历史最大净值比例
            self.shares_held / MAX_NUM_SHARES,                              # 15 当前持股数量比例
            self.cost_basis / MAX_SHARE_PRICE,                              # 16 持仓成本比例（每股成本/最高股价）
            self.total_shares_sold / MAX_NUM_SHARES,                        # 17 总卖出股数比例
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),    # 18 总销售额比例（实际销售额/最大可能销售额）
        ])
        return obs

    # 这段代码处理的是智能体在股票交易环境中的买卖动作。
    # 方法接收一个动作参数，根据动作类型（买入或卖出）和数量比例来更新账户余额、持股数量等状态
    def _take_action(self, action):
        """执行买卖操作的核心方法
        Args:
            action: 动作向量 [动作类型, 数量比例]
        """
        # 生成当前步骤的随机交易价格（在当日开盘价和收盘价之间）
        # 注意：使用随机价格可能引入噪声，更真实的模拟应考虑具体时间价格
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df.loc[self.current_step, "open"], self.df.loc[self.current_step, "close"])

        # 解析动作参数
        action_type = action[0]
        amount = action[1]

        # 买入操作分支
        if action_type < 1:
            # 计算最大可买数量（根据当前余额和股价）
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price) # 整数股数
            shares_bought = int(total_possible * amount) # 按比例计算实际购买量
            
            # 计算持仓成本（加权平均法）
            prev_cost = self.cost_basis * self.shares_held # 原有持仓总成本
            additional_cost = shares_bought * current_price # 新增持仓成本

            # 更新账户状态
            self.balance -= additional_cost # 扣除购买金额
            # 计算新的平均持仓成本（总成本 / 总股数）
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought # 增加持股数量

        # 卖出操作分支
        elif action_type < 2:
            # 计算卖出数量（按当前持股比例）
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount) # 整数股数

            # 更新账户状态
            self.balance += shares_sold * current_price # 增加卖出金额
            self.shares_held -= shares_sold # 减少持股数量
            self.total_shares_sold += shares_sold # 更新累计卖出数量
            self.total_sales_value += shares_sold * current_price # 更新累计卖出金额

        # 更新当前净资产（现金余额 + 持仓市值）
        self.net_worth = self.balance + self.shares_held * current_price

        # 更新历史最大净资产（用于风险控制）
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        # 清仓时重置持仓成本（避免除以零错误）
        if self.shares_held == 0:
            self.cost_basis = 0

    # 首先，step方法是强化学习环境中最重要的方法之一，负责执行动作、更新状态、计算奖励和判断是否结束。
    # 首先，self._take_action(action) 是执行具体的买卖操作，这部分在另一个方法中，所以注释中需要说明。
    # 然后，current_step增加，可能触发重置到0，形成循环训练。这里用户注释了loop training，但done被注释掉了，
    # 可能需要说明原因。delay_modifier看起来是用于调整奖励的时间衰减，但当前代码中没有使用，可能需要指出这一点。
    # 奖励计算部分，用户将利润大于0时奖励设为1，否则-100，这可能存在问题，比如奖励设计过于简单，可能导致训练不稳定，
    # 需要在注释中提到潜在问题。然后，如果净资产小于等于0，则结束回合。最后返回观测值、奖励、结束标志和空字典。
    # 接下来，我需要按照用户的注释风格，比如之前的常量注释和_next_observation方法的注释，保持一致性。
    # 添加注释时，要解释每个步骤的作用、参数的意义，以及可能的改进点或注意事项。
    # 例如，在reward计算部分，可以指出当前的二值奖励可能不够合理，建议考虑连续奖励。
    # 同时，done的条件可能过于严格，是否需要其他终止条件。
    # 此外，delay_modifier未被使用，可能需要说明是否应该应用到奖励中。
    # 最后，确保注释清晰，分步骤解释，并遵循用户之前的格式，
    # 比如使用中文注释，详细说明每个变量的作用和设计意图。
    # 同时，检查是否需要使用SEARCH/REPLACE格式，
    # 因为用户要求修改当前选中的代码，所以应该使用该格式，并确保文件路径正确。
    def step(self, action):
        """执行动作并推进环境到下一步
        Args:
            action: 动作向量 [动作类型, 数量比例]
        Returns:
            obs: 新的观测状态
            reward: 当前奖励值
            done: 是否结束训练
            info: 附加信息（本实现为空字典）
        """
        # 1. 执行交易动作（调用私有方法处理买卖操作）
        # Execute one time step within the environment
        self._take_action(action)
        done = False # 默认不终止训练

        # 2. 推进时间步
        self.current_step += 1

        # 3. 检查数据边界（当遍历完所有数据后重置到开头，形成训练循环）
        if self.current_step > len(self.df.loc[:, 'open'].values) - 1:
            self.current_step = 0  # loop training # 重置到数据开头
            # done = True  # 原实现注释掉的终止条件

        # 4. 计算时间衰减因子（当前未实际使用，保留供奖励函数改进）
        delay_modifier = (self.current_step / MAX_STEPS)

        # 5. 计算奖励值（基于当前净资产）
        # 当前简单奖励机制：盈利奖励+1，亏损惩罚-100
        # 潜在问题：二值奖励可能过于简单，可考虑更连续的奖励设计
        # profits
        reward = self.net_worth - INITIAL_ACCOUNT_BALANCE
        reward = 1 if reward > 0 else -100

        # 6. 终止条件判断（净资产归零时终止训练）
        if self.net_worth <= 0:
            done = True # 账户爆仓时终止训练

        # 7. 获取新的观测状态
        obs = self._next_observation()

        # 8. 返回环境反馈（包含新状态、奖励、终止标志、附加信息）
        return obs, reward, done, {}

    # 首先，reset方法是强化学习环境中非常重要的一个方法，用于将环境状态重置为初始状态，以便开始新的训练回合。
    # 用户提供的代码中，reset方法主要完成以下几个任务：
    # 1. 重置账户相关的变量，如余额、净资产、持股数量等。
    # 2. 允许传入新的数据集（new_df参数），这在测试时可能用到。
    # 3. 设置当前步骤为0，或者随机选择一个起始点（但当前代码中随机部分被注释掉了）。
    # 4. 返回初始的观测状态。
    def reset(self, new_df=None):
        """重置环境到初始状态，开始新的训练回合
        Args:
            new_df: 可选的新数据集（用于测试时替换训练数据）
        Returns:
            obs: 初始观测状态
        """
        # 1. 重置账户状态到初始值
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE  # 重置现金余额到初始金额（1万元）
        self.net_worth = INITIAL_ACCOUNT_BALANCE  # 当前净资产
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE  # 历史最大净值（用于风险控制）
        self.shares_held = 0  # 当前持股数量
        self.cost_basis = 0  # 平均持仓成本（每股）
        self.total_shares_sold = 0  # 累计卖出股数
        self.total_sales_value = 0  # 累计卖出金额

        # 2. 数据集处理（测试时替换数据）
        # pass test dataset to environment
        if new_df:
            self.df = new_df # 允许注入新的测试数据集

        # 3. 设置初始时间步
        # 原实现包含随机起始点的设计（被注释），当前使用固定起始点
        # 随机起始有助于避免过拟合特定时间段，但会增加训练难度
        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(
        #     0, len(self.df.loc[:, 'open'].values) - 6)
        self.current_step = 0 # 重置到数据起始位置

        # 4. 返回初始观测值
        return self._next_observation()

    # 查看render方法的现有代码。它打印当前步骤的账户信息，包括余额、持股、净值等
    def render(self, mode='human', close=False):
        """可视化当前环境状态（控制台输出模式）
        Args:
            mode: 渲染模式（支持'human'控制台输出）
            close: 是否关闭渲染（本实现未使用）
        Returns:
            profit: 当前累计利润（用于外部记录）
        """
        # 1. 计算当前利润（净资产 - 初始资金）
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        # 2. 打印交易信息控制台面板
        print(f'Step(当前时间步): {self.current_step}') # 当前时间步
        print(f'Balance(现金余额): {self.balance}') # 现金余额（保留两位小数）
        print(f'Shares held(当前及累计持股): {self.shares_held} (Total sold: {self.total_shares_sold})') # 当前及累计持股
        print(f'Avg cost for held shares(持仓平均成本): {self.cost_basis} (Total sales value(总销售额): {self.total_sales_value})') # 成本与销售额
        print(f'Net worth(当前净值): {self.net_worth} (Max net worth(历史最高净值): {self.max_net_worth})')  # 当前及历史最高净值
        print(f'Profit(累计利润): {profit}') # 累计利润

        # 3. 返回利润值供外部记录（用于可视化分析）
        return profit
