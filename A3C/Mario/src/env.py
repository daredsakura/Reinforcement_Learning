import gym
import gym_super_mario_bros
from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import numpy as np
import subprocess as sp


class Monitor:  # 监视器类，用于记录图像数据并保存为视频文件
    def __init__(self, width, height, saved_path):
        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "80", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        # self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())


def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # RGB色彩空间转换为灰度色彩空间
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):  # 修改环境奖励
    def __init__(self, env=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        if monitor is not None:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        state_, reward, done, info = self.env.step(action)
        if self.monitor:
            self.monitor.record(state_)
        state_ = process_frame(state_)
        reward += (info['score'] - self.curr_score) / 40  # 额外奖励函数，加速收敛
        self.curr_score = info['score']
        if done:
            if info['flag_get']:  # 到达旗帜处
                reward += 50
            else:
                reward -= 50
        # 修改 reward 或 observation
        return state_, (reward / 10), done, info

    def reset(self):
        self.curr_score = 0
        return process_frame(self.env.reset())


class CustomSkipFrame(Wrapper):
    def __init__(self, env=None, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.skip = skip
        self.observation_space = Box(low=0, high=255, shape=(self.skip, 84, 84))

    def step(self, action):

        total_rewards = 0
        states = []
        state_, reward, done, info = self.env.step(action)
        for _ in range(self.skip):
            if not done:
                state_, reward, done, info = self.env.step(action)
                total_rewards += reward
                states.append(state_)
            else:
                states.append(state_)
        states = np.concatenate(states, axis=0, dtype=np.float32)[None, :, :, :]  # 将多个四维数组连接成一个更大的四维数组
        return states, total_rewards, done, info

    def reset(self):
        state = self.env.reset()
        states = np.concatenate([state for _ in range(self.skip)], axis=0, dtype=np.float32)[None, :, :, :]
        # 将1个数组重复连接成一个更大的四维数组
        return states


def create_train_env(world, stage, action_type, output_path=None):
    env_name = "SuperMarioBros-{}-{}-v0".format(world, stage)
    env = gym_super_mario_bros.make(env_name)
    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None
    if action_type == "right":
        actions = RIGHT_ONLY
    elif action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    env = JoypadSpace(env, actions)
    env = CustomReward(env, None)
    env = CustomSkipFrame(env)
    return env, env.observation_space.shape[0], len(actions)
