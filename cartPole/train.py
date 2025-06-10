import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

import handler_qtable as hq


# Hiperparâmetros
learning_rate_a = 0.1
discount_factor_g = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay_rate = 0.995
episodes = 500
max_steps = 200

env = gym.make('CartPole-v1')
obs_space_high = env.observation_space.high
obs_space_low = env.observation_space.low
obs_space_high[1] = 5
obs_space_low[1] = -5
obs_space_high[3] = 5
obs_space_low[3] = -5

def discretize(obs, bins):
    ratios = (obs - obs_space_low) / (obs_space_high - obs_space_low)
    new_obs = (ratios * bins).astype(int)
    return tuple(np.clip(new_obs, 0, bins - 1))

def train_q_learning(din):
    bins = np.array([din]*4)
    q_table = np.random.uniform(low=0, high=1, size=(din, din, din, din, env.action_space.n))

    global epsilon
    epsilon = 1.0
    rewards_per_episode = []

    for ep in range(episodes):
        state, _ = env.reset()
        state_disc = discretize(state, bins)
        total_reward = 0

        for step in range(max_steps):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state_disc])

            next_state, reward, done, truncated, _ = env.step(action)
            next_state_disc = discretize(next_state, bins)

            best_future_q = np.max(q_table[next_state_disc])
            current_q = q_table[state_disc + (action,)]
            q_table[state_disc + (action,)] += learning_rate_a * (reward + discount_factor_g * best_future_q - current_q)

            state_disc = next_state_disc
            total_reward += reward

            if done or truncated:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay_rate)
        rewards_per_episode.append(total_reward)

    avg_reward = np.mean(rewards_per_episode[-50:])
    return avg_reward, rewards_per_episode, q_table

# Treinamento e coleta de dados
din_values = [6, 8, 12, 16, 20]
resultados = {}

for din in din_values:
    print(f"Treinando com din = {din}...")
    avg_reward, rewards, q_table = train_q_learning(din)

    hq.savefile(f'din_{din}.npz',q_table)

    resultados[din] = (avg_reward, rewards)
    print(f"Média de recompensa (últimos 50 episódios) para din={din}: {avg_reward:.2f}")

env.close()

# Gráfico: Recompensas por episódio
plt.figure(figsize=(12, 6))
for din in din_values:
    _, rewards = resultados[din]
    plt.plot(rewards, label=f'din = {din}')
plt.xlabel('Episódio')
plt.ylabel('Recompensa')
plt.title('Recompensas por episódio para diferentes din')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Gráfico: Médias das últimas 50 recompensas
medias_finais = [resultados[din][0] for din in din_values]
plt.figure(figsize=(8, 5))
plt.bar([str(d) for d in din_values], medias_finais, color='skyblue')
plt.xlabel('dimensão da Q-table (din)')
plt.ylabel('Recompensa média (últimos 50 episódios)')
plt.title('Comparação de desempenho médio por dimensão da Q-Table')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
