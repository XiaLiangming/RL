import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import TimeLimit
import imageio
import pandas as pd
import matplotlib.pyplot as plt

'''Initialize environment'''
def initialize_env(env_name: str, render_mode: str) -> TimeLimit:
    env = gym.make(env_name, render_mode=render_mode)
    env.reset()

    return env

'''Discretize state'''
def discretize_state(state: np.ndarray, max_observation: tuple, observation_bins: tuple) -> tuple:
    clipped_state = np.clip(state, -max_observation, max_observation)

    observation_bin = tuple(np.digitize(clipped_state[i], observation_bins[i]) - 1 for i in range(4))

    return observation_bin

'''Evaluate policy'''
def evaluate(env: TimeLimit, Q: np.ndarray, max_observation: tuple, observation_bins: tuple) -> int:
    state, _ = env.reset()
    action_num = env.action_space.n
    action_space = [i for i in range(action_num)]
    total_steps = 0
    while total_steps < 2e6:
        action = np.argmax(Q[discretize_state(state, max_observation, observation_bins)])
        chosen_action = action_space[action]
        state, _, done, _, _ = env.step(chosen_action)
        if done:
            break
        total_steps += 1

    return total_steps

'''Test learned policy'''
def test(env: TimeLimit, Q: np.ndarray, max_observation: tuple, observation_bins: tuple, save_path: str, fps: int=30) -> np.ndarray:
    state, _ = env.reset()
    frames = []
    action_num = env.action_space.n
    action_space = [i for i in range(action_num)]
    total_steps = 0
    while True:
        total_steps += 1
        action = np.argmax(Q[discretize_state(state, max_observation, observation_bins)])
        chosen_action = action_space[action]
        state, _, done, _, _ = env.step(chosen_action)
        frames.append(env.render())
        if done:
            break
    print(f"Gif saved to {save_path}! Total steps: {total_steps}.")

'''Train Q-learning agent'''
def train(env_train: TimeLimit,
          env_eval: TimeLimit,
          history_path: str,
          max_observation: np.ndarray=np.array([2.4, 10, .2095, 10]),
          observation_bins_num: tuple=(1, 1, 100, 100),
          max_steps: int=1e5,
          greedy_steps: int=1e2,
          alpha_max: float=0.5,
          alpha_min: float=0.1,
          gamma: float=0.99,
          epsilon_max: float=1.0,
          epsilon_min: float=0.05,
          discretize_factor: float=0.6,
          penalty_factor: np.ndarray=np.array([0.5, 0, 10, 0]),
          report_steps: int=1e3,
          eval_episodes: int=5) -> tuple:

    '''Calculate reward'''
    def get_reward(state: np.ndarray, penalty_factor: tuple) -> float:
        reward = 1 - np.sum(penalty_factor * np.abs(state))
        return reward
    
    '''Flip state for symmetry'''
    def flip_state(state: tuple, observation_bins_num: tuple) -> tuple:
        fliped_state = tuple(observation_bins_num[i] - state[i] - 1 for i in range(4))
        return fliped_state
    
    delta_epsilon = (epsilon_max - epsilon_min) / greedy_steps
    epsilon = epsilon_max
    alpha = alpha_max
    delta_alpha = (alpha_max - alpha_min) / max_steps
    action_num = env_train.action_space.n
    action_space = [i for i in range(action_num)]

    observation_bins = ()
    step = 0
    episode = 0
    Q = np.zeros(observation_bins_num + (action_num, ))

    '''Create observation bins'''
    for i in range(4):
        bins = np.linspace(-1, 1, observation_bins_num[i] + 1)
        observation_bins += (np.sign(bins) * abs(bins) ** discretize_factor * max_observation[i], )

    with open(history_path, "w") as f:
        f.write(f"{'Step':<10}\t{'Episode':<10}\t{'Epsilon':<10}\t{'Alpha':<10}\t{'AvgEvalStep':<15}\n")

    while step < max_steps:
        currenet_state, _ = env_train.reset()
        current_state_discretize = discretize_state(currenet_state, max_observation, observation_bins)
        done = False

        while True:
            step += 1
            if step % report_steps == 0:
                total_eval_steps = 0
                for _ in range(eval_episodes):
                    eval_step = evaluate(env_eval, Q, max_observation, observation_bins)
                    total_eval_steps += eval_step
                eval_steps = total_eval_steps // eval_episodes
                with open(history_path, "a") as f:
                    f.write(f"{step:<10}\t{episode + 1:<10}\t{epsilon:<10.4f}\t{alpha:<10.4f}\t{eval_steps:<15}\n")
                print(f"Step {step}, Episode {episode + 1}, Epsilon {epsilon:.4f}, Alpha {alpha:.4f}, Eval Step {eval_steps}.")
            if np.random.random() < epsilon:
                action = np.random.choice(action_num)
            else:
                action = np.argmax(Q[current_state_discretize])
            epsilon = max(epsilon_min, epsilon - delta_epsilon)
            chosen_action = action_space[action]
            next_state, _, done, _, _ = env_train.step(chosen_action)
            
            if done:
                break
            reward = get_reward(next_state, penalty_factor)
            next_state_discretize = discretize_state(next_state, max_observation, observation_bins)

            current_q = Q[current_state_discretize + (action, )]
            next_max_q = np.max(Q[next_state_discretize])
            new_q = (1 - alpha) * current_q + alpha * (reward + gamma * next_max_q)
            Q[current_state_discretize + (action, )] = new_q
            fliped_state = flip_state(current_state_discretize, observation_bins_num)
            Q[fliped_state + (1 - action, )] = new_q
            alpha = max(alpha_min, alpha - delta_alpha)
            current_state_discretize = next_state_discretize

        episode += 1

    return Q, max_observation, observation_bins

'''Plot training history'''
def plot_history(history_path: str, save_path: str):
    df = pd.read_csv(history_path, sep=r"\s+", engine="python")
    plt.figure(figsize=(10, 6))
    plt.plot(df['Step'], df['AvgEvalStep'], linewidth=2)
    plt.title('Training History', fontsize=14)
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel('Average Evaluation Steps', fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

'''Save frames as GIF'''
def save_as_gif(frames: np.ndarray, save_path='./output.gif'):
    imageio.mimsave(save_path, frames, duration=0.1)

if __name__ == '__main__':
    env_train = initialize_env(env_name='CartPole-v1', render_mode='rgb_array')
    env_eval = initialize_env(env_name='CartPole-v1', render_mode='rgb_array')
    history_path = "history.csv"
    Q, max_observation, observation_bins = train(env_train,
                                                 env_eval,
                                                 history_path=history_path)
    env_train.close()
    test(env_eval, Q, max_observation, observation_bins, save_path='result.gif')
    env_eval.close()
    plot_history(history_path, 'history.png')