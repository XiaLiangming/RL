import ale_py

import gymnasium as gym
from gymnasium import spaces

import cv2

import torch
import numpy as np

import keyboard

class Breakout(gym.Wrapper):
    def __init__(self,
                 render_mode=None,
                 width=84,
                 height=84,
                 skip=4,
                 reset_noops=30,
                 grayscale=True,
                 clip_rewards=False,
                 max_episode_steps=10000):

        # Register Arcade Learning Environment (ALE) with Gym
        gym.register_envs(ale_py)

        # Create the base environment
        env = gym.make('BreakoutNoFrameskip-v4', render_mode=render_mode)

        gym.Wrapper.__init__(self, env)
        self._width = width
        self._height = height
        self._skip = skip
        self._action_map = [0, 2, 3]
        self._lives = 0
        self._grayscale = grayscale
        self._clip_rewards = clip_rewards
        self._reset_noops = reset_noops
        self.env._max_episode_steps = max_episode_steps
        self._real_done = True
        self._max_lives = self.env.unwrapped.ale.lives()

        # Define the observation space
        if self._grayscale:
            shp = (self._height, self._width)
        else:
            shp = (self._height, self._width, 3)
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=shp,
                                            dtype=np.uint8)
        
        # Frame buffer for max pooling over last two frames
        self._frame_buffer = np.zeros((2,)+self.observation_space.shape, dtype=np.uint8)
        # 0: NOOP, 1: RIGHT, 2: LEFT
        self.action_space = spaces.Discrete(3)

    def reset(self, **kwargs):
        if self._real_done:
            self.env.reset(**kwargs)[0]
        else:
            self.env.step(0)
        self._lives = self.env.unwrapped.ale.lives()
        random_steps = np.random.randint(self._reset_noops + 1)
        # Randomly move left or right before starting
        if random_steps % 2 == 0:
            action = 3
        elif random_steps % 2 == 1:
            action = 2
        for _ in range(random_steps):
            self.env.step(action)
        frame, _, _, _, _ = self.env.step(1)
        self._frame_buffer[1] = self._resize(frame)
        # Additional random no-op steps
        for i in range(random_steps):
            self.env.step(0)
            self._frame_buffer[i % 2] = self._resize(frame)
        max_frame = torch.from_numpy(self._frame_buffer.max(axis=0))

        return max_frame

    def step(self, action):
        action = self._action_map[action]
        total_reward = 0
        for i in range(self._skip):
            frame, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if i == self._skip - 2: self._frame_buffer[0] = self._resize(frame)
            if i == self._skip - 1: self._frame_buffer[1] = self._resize(frame)
            total_reward += reward
            if done:
                break
        max_frame = torch.from_numpy(self._frame_buffer.max(axis=0))
        self._real_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self._lives:
            done = True
        self._lives = lives
        if self._clip_rewards:
            total_reward = np.sign(total_reward)
        total_reward = torch.tensor(total_reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.bool)
        
        return max_frame, total_reward, done
    
    def get_max_lives(self):
        return self._max_lives
    
    def play(self):
        self.reset()
        keyboard2action = {"down":0, "right":1, "left":2}
        total_reward = 0
        while True:
            self.render()
            action = None
            if keyboard.is_pressed('right'):
                action = keyboard2action['right']
            elif keyboard.is_pressed('left'):
                action = keyboard2action['left']
            elif keyboard.is_pressed('q'):
                print("Exiting the game.")
                self.close()
                break
            else:
                action = keyboard2action['down']
            _, reward, done = self.step(action)
            total_reward += reward.item()
            if done:
                if self._real_done:
                    print(f"Game Over! Total Reward = {total_reward}. Resetting...")
                    total_reward = 0
                self.reset()

    def _resize(self, frame):
        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        
        return frame