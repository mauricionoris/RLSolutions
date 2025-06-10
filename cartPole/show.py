import gymnasium as gym
import numpy as np
import time
import handler_qtable as hq


# Hiperparâmetros
learning_rate_a = 0.1
discount_factor_g = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay_rate = 0.995
episodes = 500
max_steps = 200
din = 20  # Dimensão da Q-table

# Ambiente com renderização desativada para treino
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

# Carrega Treinamento
def train_agent(din):
    bins = np.array([din]*4)
    q_table = hq.loadfile(f'din_{din}.npz')

    return q_table, bins




def run_trained_agent(q_table, bins, episodes=3):
    # Agora mostramos o agente treinado com render
    env_render = gym.make('CartPole-v1', render_mode="human")

    for ep in range(episodes):
        state, _ = env_render.reset()
        state_disc = discretize(state, bins)
        total_reward = 0

        for step in range(max_steps):
            action = np.argmax(q_table[state_disc])
            next_state, reward, done, truncated, _ = env_render.step(action)
            state_disc = discretize(next_state, bins)
            total_reward += reward
            time.sleep(0.02)  # Para desacelerar um pouco a simulação

            if done or truncated:
                print(f"🏁 Episódio {ep+1} finalizado com recompensa: {total_reward}")
                break
    env_render.close()
    env.close()



din_values = [6, 8, 12, 16, 20]

for din in din_values:
    # Treinando o agente

    print(f'Exibindo agente treinado com a dimensao: {din} ')
    q_table, bins = train_agent(din)
    run_trained_agent(q_table, bins)
    print('************************************************')

