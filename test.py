from maze_env import Maze
from RL_brain import QLearningTable
from RL_brain import SarsaTable
from RL_brain import SarsaLambdaTable


def update1():
    for episode in range(100):
        observation = env.reset()
        while True:
            env.render()  # 更新可视化环境
            action = RL.choose_action(str(observation))
            observation_, reward, done = env.step(action)
            RL.learn(str(observation), action, reward, str(observation_))
            observation = observation_
            if done:
                break
    # 结束游戏并关闭窗口
    print('game over')
    env.destroy()


def update2():
    for episode in range(100):
        observation = env.reset()
        action = RL.choose_action(str(observation))
        while True:
            env.render()  # 更新可视化环境
            observation_, reward, done = env.step(action)
            action_ = RL.choose_action(str(observation_))
            RL.learn(str(observation), action, reward, str(observation_), action_)
            observation = observation_
            action = action_
            if done:
                break
    # 结束游戏并关闭窗口
    print('game over')
    env.destroy()


def update3():
    for episode in range(100):
        observation = env.reset()
        action = RL.choose_action(str(observation))
        RL.eligibility_trace *= 0
        while True:
            env.render()  # 更新可视化环境
            observation_, reward, done = env.step(action)
            action_ = RL.choose_action(str(observation_))
            RL.learn(str(observation), action, reward, str(observation_), action_)
            observation = observation_
            action = action_
            if done:
                break
    # 结束游戏并关闭窗口
    print('game over')
    env.destroy()


if __name__ == '__main__':
    # 定义环境 env 和 RL 方式
    env = Maze()
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))
    # 开始可视化环境 env
    env.after(100, update3)
    env.mainloop()
