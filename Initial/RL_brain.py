import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(
                state_action[state_action == np.max(state_action)].index)  # 当有多个权重相同的action时，打乱顺序，使得被选取概率相同
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args_):
        pass

    def check_state_exist(self, state):  # 对于state无法估计的时候，检测是否之前访问过，否则将新state添加到QTable
        if state not in self.q_table.index:
            self.q_table = self.q_table._append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r  # 下个状态为terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)


class SarsaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r  # 下个状态为terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)


class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        # 后向观测算法, eligibility trace.
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]
        else:
            q_target = r  # 下个状态为terminal
        q_error = q_target - q_predict
        # 这里开始不同:
        # 对于经历过的 state-action, 我们让他+1,上限为1 证明他是得到 reward 路途中不可或缺的一环\
        self.eligibility_trace.loc[s, :] *= 0
        self.eligibility_trace.loc[s, a] = 1
        # Q table 更新
        self.q_table += self.lr * q_error * self.eligibility_trace
        # 随着时间衰减 eligibility trace 的值, 离获取 reward 越远的步, 他的"不可或缺性"越小
        self.eligibility_trace *= self.lambda_ * self.gamma

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            to_be_append = pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)
            self.q_table = self.q_table._append(to_be_append)
            self.eligibility_trace = self.eligibility_trace._append(to_be_append)
