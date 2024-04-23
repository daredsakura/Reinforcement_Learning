from collections import deque

from gym.wrappers import FrameStack, TransformObservation, GrayScaleObservation
from skimage import transform
import gym_super_mario_bros
from gym.spaces import Box
from gym import Wrapper, ObservationWrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import numpy as np
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)


class ResizeObservation(ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        resize_obs = transform.resize(observation, self.shape)
        # cast float back to uint8
        resize_obs *= 255
        resize_obs = resize_obs.astype(np.uint8)
        return resize_obs


class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class CustomReward(Wrapper):
    def __init__(self, env):
        super(CustomReward, self).__init__(env)
        self._current_score = 0
        self._current_x = 40

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        reward += (info["score"] - self._current_score) / 40.
        self._current_score = info["score"]
        if done:
            if info["flag_get"]:
                reward += 50
            else:
                reward -= 50
        # if self.world == 7 and self.stage == 4:
        #     if (506 <= info["x_pos"] <= 832 and info["y_pos"] > 127) or (
        #             832 < info["x_pos"] <= 1064 and info["y_pos"] < 80) or (
        #             1113 < info["x_pos"] <= 1464 and info["y_pos"] < 191) or (
        #             1579 < info["x_pos"] <= 1943 and info["y_pos"] < 191) or (
        #             1946 < info["x_pos"] <= 1964 and info["y_pos"] >= 191) or (
        #             1984 < info["x_pos"] <= 2060 and (info["y_pos"] >= 191 or info["y_pos"] < 127)) or (
        #             2114 < info["x_pos"] < 2440 and info["y_pos"] < 191) or info["x_pos"] < self.current_x - 500:
        #         reward -= 50
        #         done = True
        # if self.world == 4 and self.stage == 4:
        #     if (info["x_pos"] <= 1500 and info["y_pos"] < 127) or (
        #             1588 <= info["x_pos"] < 2380 and info["y_pos"] >= 127):
        #         reward = -50
        #         done = True

        self._current_x = info["x_pos"]
        return state, reward / 10., done, info

    def reset(self):
        self._current_score = 0
        self._current_x = 40
        return self.env.reset()


def create_env(world, stage, action_type, output_path=None):
    env_name = "SuperMarioBros-{}-{}-v0".format(world, stage)
    env = gym_super_mario_bros.make(env_name)
    if action_type == "right":
        actions = RIGHT_ONLY
    elif action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    # actions = [['right'], ['right', 'A']]
    env = JoypadSpace(env, actions)
    # env = MaxAndSkipEnv(env)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)
    env = CustomReward(env)
    # print(env.observation_space)
    return env, env.observation_space.shape[0], len(actions)


def create_train_env(world, stage, action_type, output_path=None):
    env_name = "SuperMarioBros-{}-{}-v0".format(world, stage)
    env = gym_super_mario_bros.make(env_name)
    if action_type == "right":
        actions = RIGHT_ONLY
    elif action_type == "simple":
        actions = SIMPLE_MOVEMENT
    else:
        actions = COMPLEX_MOVEMENT
    # actions = [['right'], ['right', 'A']]
    env = JoypadSpace(env, actions)
    # env = MaxAndSkipEnv(env)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=84)
    env = TransformObservation(env, f=lambda x: x / 255.)
    env = FrameStack(env, num_stack=4)
    env = CustomReward(env)
    # print(env.observation_space)
    return env
