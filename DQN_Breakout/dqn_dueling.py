from buffer import ReplayBuffer
from env import Breakout

import torch
import torch.optim as optim
import torch.nn.functional as F
from model import DuelingDQN
torch.backends.cudnn.benchmark = True

import os
import random

from utils import evaluate_model, plot_history

def train(device: torch.device,
          Env: type,
          Buffer: type,
          Model: type,
          loss_function: callable = F.smooth_l1_loss,
          max_steps: int = 2e7,
          warmup_steps: int = 50000,
          random_steps: int = 50000,
          greedy_steps: int = 1000000,
          train_steps: int = 4,
          eval_steps: int = 10000,
          target_update_steps: int = 10000,
          replay_buffer_size: int = 1000000,
          frame_stacking: int = 4,  
          batch_size: int = 32,
          gamma: float = 0.99,
          epsilon_interval: tuple = (0.1, 1.0),
          eval_episode: int = 10,
          eval_epsilon: float = 0.05,
          grad_clip: float = 1.0,
          history_path: str = "./history/dqn_dueling.csv",
          best_model_save_path: str = "./models/dqn_dueling_best.pth"):

    # Create history directory and model directory if it doesn't exist
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
    # Initialize environments
    env = Env(clip_rewards=False)
    observation_space = env.observation_space.shape
    action_space = env.action_space.n
    # Initialize replay buffer
    buffer = Buffer(capacity=replay_buffer_size,
                    observation_space=observation_space,
                    frame_stacking=frame_stacking,
                    device=device)
    # Initialize online and target models
    online_model = Model(input_shape=(frame_stacking, *observation_space),
                         action_space=action_space).to(device)
    target_model = Model(input_shape=(frame_stacking, *observation_space),
                         action_space=action_space).to(device)
    target_model.load_state_dict(online_model.state_dict())
    # Initialize optimizer
    optimizer = optim.Adam(online_model.parameters(), lr=0.0000625, eps=1.5e-4)

    step = 0
    episode = 0
    epsilon_min, epsilon_max = epsilon_interval
    epsilon = epsilon_max
    delta_epsilon = (epsilon_max - epsilon_min) / greedy_steps
    total_train_reward = 0
    total_loss = 0
    best_eval_reward = -float("inf")

    with open(history_path, "w") as f:
        f.write(f"{'Episode':<10}\t{'AvgTrainReward':<15}\t{'AvgLoss':<15}\t{'Step':<10}\t{'Epsilon':<10}\t{'AvgEvalReward':<15}\n")
    while step < max_steps:
        state = torch.empty((frame_stacking, *observation_space), dtype=torch.float32, pin_memory=True)
        frame = env.reset()
        episode += 1
        for i in range(frame_stacking):
            frame, reward, done  = env.step(0)
            state[i] = frame
            buffer.push(frame, 0, reward, done)

        while not done:
            step += 1
            # Select action
            if step < random_steps:
                action = torch.tensor(env.action_space.sample(), dtype=torch.long, device=device)
            else:
                if random.random() < epsilon:
                    action = torch.tensor(env.action_space.sample(), dtype=torch.long, device=device)
                else:
                    with torch.no_grad():
                        action = online_model(state.unsqueeze(0).to(device)).argmax(dim=1).item()
                # Decay epsilon
                epsilon = max(epsilon - delta_epsilon, epsilon_min)
            # Take action
            next_frame, reward, done = env.step(action)
            buffer.push(next_frame, action, reward, done)
            state = torch.cat((state[1:], next_frame.unsqueeze(0)), dim=0).pin_memory()
            total_train_reward += reward

            if step > warmup_steps:
                # Train
                if step % train_steps == 0:
                    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                    # Compute target Q-values
                    with torch.no_grad():
                        next_q_values = target_model(next_states).max(1)[0].unsqueeze(1)
                        target_q_values = rewards + gamma * next_q_values * (1 - dones)
                    # Optimize model
                    q_values = online_model(states).gather(1, actions)
                    loss = loss_function(q_values, target_q_values)
                    total_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    # Clip gradients
                    for param in online_model.parameters():
                        param.grad.data.clamp_(-grad_clip, grad_clip)
                    optimizer.step()
                # Update target network
                if step % target_update_steps == 0:
                    target_model.load_state_dict(online_model.state_dict())

            # Evaluate
            if step % eval_steps == 0:
                total_eval_reward = 0
                for _ in range(eval_episode):
                    total_eval_reward += evaluate_model(online_model, Env, frame_stacking, eval_epsilon, device)
                avg_eval_reward = total_eval_reward / eval_episode
                if avg_eval_reward > best_eval_reward:
                    best_eval_reward = avg_eval_reward
                    torch.save(online_model.state_dict(), best_model_save_path)
                avg_train_reward = total_train_reward / eval_steps
                avg_loss = total_loss / eval_steps if step > warmup_steps else 0
                with open(history_path, "a") as f:
                    f.write(f"{episode:<10}\t{avg_train_reward:<15.4f}\t{avg_loss:<15.4f}\t{step:<10}\t{epsilon:<10.4f}\t{avg_eval_reward:<15.2f}\n")
                print(f"Episode: {episode}, AvgTrainReward: {avg_train_reward:.4f}, AvgLoss: {avg_loss:.4f}, Step: {step}, Epsilon: {epsilon:.4f}, AvgEvalReward: {avg_eval_reward:.2f}")
                total_train_reward = 0
                total_loss = 0
    env.close()

    return online_model

if __name__ == "__main__":
    frame_stacking = 4
    eval_epsilon = 0.05
    history_path = "./history/dqn_dueling.csv"
    best_model_save_path = "./models/dqn_dueling_best.pth"
    gif_save_path = "./gifs/dqn_dueling.gif"
    plot_save_path = "./plots/dqn_dueling.png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train(device=device,
                  Env=Breakout,
                  Model=DuelingDQN,
                  Buffer=ReplayBuffer,
                  history_path=history_path,
                  best_model_save_path=best_model_save_path)
    total_reward = evaluate_model(model=model,
                                  Env=Breakout,
                                  frame_stacking=frame_stacking,
                                  epsilon=eval_epsilon,
                                  device=device,
                                  gif_save_path=gif_save_path)
    print(f"Total Reward: {total_reward}.")
    plot_history(history_path=history_path,
                 save_path=plot_save_path,
                 figsize=(10,6))
    