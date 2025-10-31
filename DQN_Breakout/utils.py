import torch
import torch.nn as nn

import random
import imageio
import os

import pandas as pd
import matplotlib.pyplot as plt

def evaluate_model(model: nn.Module,
                   Env: type,
                   frame_stacking: int,
                   epsilon: float,
                   device: torch.device,
                   gif_save_path=None) -> float:
    if gif_save_path is not None:
        os.makedirs(os.path.dirname(gif_save_path), exist_ok=True)
        render_mode = 'rgb_array'
    else:
        render_mode = None
    env = Env(clip_rewards=False, render_mode=render_mode)
    total_reward = 0
    max_lives = env.get_max_lives()
    gif_frames = []
    for _ in range(max_lives):
        observation_space = env.observation_space.shape
        state = torch.empty((frame_stacking, *observation_space), dtype=torch.float32)
        frame = env.reset()
        state[0] = frame
        if gif_save_path is not None:
            gif_frames.append(env.render())
        for i in range(1, frame_stacking):
            frame, _, _ = env.step(0)
            state[i] = frame
            if gif_save_path is not None:
                gif_frames.append(env.render())
        done = False
        while not done:
            if random.random() < epsilon:
                action = torch.tensor(env.action_space.sample(), dtype=torch.long, device=device)
            else:
                with torch.no_grad():
                    action = model(state.unsqueeze(0).to(device)).argmax(dim=1).item()
            next_frame, reward, done = env.step(action)
            next_state = torch.cat((state[1:], next_frame.unsqueeze(0)), dim=0)
            state = next_state
            total_reward += reward
            if gif_save_path is not None:
                gif_frames.append(env.render())
        
    if gif_save_path is not None:
        imageio.mimsave(gif_save_path, gif_frames, fps=30)
        print(f"GIF saved to {gif_save_path}.")
    env.close()

    return total_reward

def plot_history(history_path: str,
                 save_path: str,
                 figsize: tuple = (10,6)):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.read_csv(history_path, sep=r"\s+", engine="python")
    plt.figure(figsize=figsize)
    plt.plot(df['Step'], df['AvgEvalReward'], label='AvgEvalReward', linewidth=1.2)
    plt.title("Learning Curve", fontsize=14)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("AvgEvalReward", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Learning curve saved to {save_path}.")