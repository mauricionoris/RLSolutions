# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 05:17:28 2024

@author: Leonimer

referências:
    https://gymnasium.farama.org/environments/toy_text/frozen_lake/
    
Frozen lake involves crossing a frozen lake from start to goal without 
falling into any holes by walking over the frozen lake. The player may not 
always move in the intended direction due to the slippery nature of the 
frozen lake.
"""

'''
Description
===========
The game starts with the player at location [0,0] of the frozen lake grid world 
with the goal located at far extent of the world e.g. [3,3] for the 
4x4 environment.

Holes in the ice are distributed in set locations when using a pre-determined 
map or in random locations when a random map is generated.

The player makes moves until they reach the goal or fall in a hole.

The lake is slippery (unless disabled) so the player may move perpendicular 
to the intended direction sometimes (see is_slippery).

Randomly generated worlds will always have a path to the goal.

Modos de Renderização Disponíveis
=================================
None (default): no render is computed.

human: Exibe o ambiente em uma janela interativa (requer suporte gráfico).
human: render return None. The environment is continuously rendered in the current display 
or terminal. Usually for human consumption.

ansi: Imprime o estado do ambiente como texto (útil para terminais sem suporte gráfico).
ansi: Return a strings (str) or StringIO.StringIO containing a terminal-style text 
representation for each time step. The text can include newlines and ANSI escape sequences 
(e.g. for colors).

rgb_array: Retorna uma matriz representando a imagem do ambiente (útil para gravações ou 
exibições personalizadas).
rgb_array: return a single frame representing the current state of the environment. 
A frame is a numpy.ndarray with shape (x, y, 3) representing RGB values for an x-by-y 
pixel image.

rgb_array_list: return a list of frames representing the states of the environment since 
the last reset. Each frame is a numpy.ndarray with shape (x, y, 3), as with rgb_array.
'''

import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Treinamento --> is_training = True
# Avaliação --> is_training = False
# is_training = False
is_training = False

# parâmetros do ambiente (env)
size_e = 5
seeds = 1
proba_frozen = .8
max_episode_steps_ = 2000
'''
is_slippery=True: If true the player will move in intended direction with probability 
of 1/3 else will move in either perpendicular direction with equal probability of 1/3 
in both directions.
For example, if action is left and is_slippery is True, then:
P(move left)=1/3
P(move up)=1/3
P(move down)=1/3
'''
is_slip = False
# is_slip = True

# estruturando o ambiente. if is training no render
env = gym.make("FrozenLake-v1", render_mode ='rgb_array' if is_training else 'human',
               max_episode_steps = max_episode_steps_,
               is_slippery=is_slip, desc=generate_random_map(size=size_e, 
                                                             p=proba_frozen, seed=seeds))

# Hiperparâmetros
episodes = 3000  # Número de episódios
alpha = 0.1  # Taxa de aprendizado (alpha or learning rate)
# gamma or discount rate. Near 0: more weight/reward placed on immediate state. 
# Near 1: more on future state
gamma = 0.9  # Fator de desconto  
epsilon = 1  # Taxa de exploração inicial (1 = 100% random actions)
min_epsilon = 0.01  # Epsilon mínimo
#epsilon_decay_rate = 0.001  # Fator de decaimento do epsilon
epsilon_decay_rate = 5/episodes # Fator de decaimento do epsilon

'''
Action Space
============
The action shape is (1,) in the range {0, 3} indicating which direction to move 
the player.

0: Move left
1: Move down
2: Move right
3: Move up

Observation Space
=================
The observation is a value representing the player’s current position as 
current_row * ncols + current_col (where both the row and col start at 0).

For example, the goal position in the 4x4 map can be calculated as 
follows: 3 * 4 + 3 = 15. The number of possible observations is dependent 
on the size of the map. The observation is returned as an int().

Starting State
==============
The episode starts with the player in state [0] (location [0, 0]).
'''
state_space = env.observation_space.n
action_space = env.action_space.n

# Inicializando a tabela Q
if is_training:
    q_table = np.zeros((state_space, action_space))
else:
    print('Carregando o modelo pré-treinado....')
    f = open('frozen_lake_'+str(size_e)+'_'+str(seeds)+'.pkl', 'rb')
    q_table = pickle.load(f)
    f.close()

# Função para escolher a ação (exploration vs. exploitation)
def choose_action(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()  # exploration
    else:
        return np.argmax(q_table[state])  # exploitation

'''
Rewards
=======
Reward schedule:
Reach goal: +1
Reach hole: 0
Reach frozen: 0

Episode End
===========
The episode ends if the following happens:
Termination:
The player moves into a hole.
The player reaches the goal at max(nrow) * max(ncol) - 1 (location [max(nrow)-1, 
                                                                    max(ncol)-1]).
Truncation (when using the time_limit wrapper):
The length of the episode is 100 for 4x4 environment, 200 for 
FrozenLake8x8-v1 environment.
'''
# Treinamento
if is_training:
    # Monitorar métricas durante o treinamento
    rewards_per_episode = []
    steps_per_episode = []
    q_table_history = []
    epsilon_hist = []
    #for episode in tqdm(range(episodes)):
    for episode in range(episodes):
        state = env.reset()[0] # reset state: 0 --> top left corner
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        while not done and not truncated:
            action = choose_action(state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
    
            # Atualizando a tabela Q     
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )

            state = next_state
            total_reward += reward
            steps += 1
    
        # Decaindo o epsilon
        a = epsilon * epsilon_decay_rate
        b = epsilon - a
        epsilon = max(b, min_epsilon)
        epsilon_hist.append(epsilon)
        
        t = 100 * (episode+1) / episodes
        print('episode: %d \ttotal: %.1f%% \tepsilon: %.3f \trewards: %d' % 
              (episode, t, epsilon, total_reward), end='\r')
        
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        q_table_history.append(np.mean(q_table))  # Armazenar média geral da Q-Table
    
    f = open("frozen_lake_"+str(size_e)+'_'+str(seeds)+".pkl","wb")
    pickle.dump(q_table, f)
    f.close()
    print("\nTreinamento concluído!")
    
print("Tabela Q:")
print(q_table)

# Avaliação
if not is_training:
    state = env.reset()[0] # reset state: 0 --> top left corner
    done = False
    env.render()
    i=0
    rw=0
    truncated = False
    while not done and not truncated:
        action = np.argmax(q_table[state])
        next_state, reward, done, truncated, _ = env.step(action)
        
        if state == next_state:
            action = 1
            next_state, reward, done, truncated, _ = env.step(action)
        
        state = next_state

        
        

        i+=1
        print('step time:', i, end='\r')
        rw+=reward
    
    print('\nreward:', rw)

env.close()

# rotina p/ cálculo da média móvel
def media_movel(dados, janela):
  media_movel_lista = []
  for i in range(len(dados) - janela + 1):
    media_movel = sum(dados[i:i+janela])/janela
    media_movel_lista.append(media_movel)
  return media_movel_lista

# Metrics after training
if is_training:
    a= 0
    for i in range(len(rewards_per_episode)):
        if rewards_per_episode[i] == 1.0:
            a+=1
            
    print('\nTotal (+) rewords:', a)
    print('Total (-) rewords:', len(rewards_per_episode) - a)
    
    # Métricas
    
    # Gráfico Total rewards per episode
    # plt.hist(rewards_per_episode, bins=3)
    # plt.xlabel('Rewards')
    # plt.ylabel('Total rewards')
    # plt.title('Total rewards per episode')
    # plt.xticks([0,.5,1])
    # plt.show()
    
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot()
    ax1_bars = [0,1]                           
    ax1.hist( 
        rewards_per_episode, 
        bins=[x for i in ax1_bars for x in (i-0.4,i+0.4)], 
        color='#404080')
    ax1.set_xticks(ax1_bars)
    ax1.set_xticklabels(['reward 0 label','reward 1 label'])
    ax1.set_title("Total rewards")
    # ax1.set_yscale('log')
    ax1.set_ylabel('Total')
    fig.tight_layout()
    plt.show()
    
    # Gráfico Total steps per episode
    plt.title('Total steps per episode')
    plt.xlabel('steps')
    plt.ylabel('nr. de epsódios')
    sns.histplot(steps_per_episode, bins=40, kde=True)
    plt.show()
    
    plt.plot(steps_per_episode, color="green")
    plt.title("steps time")
    plt.xlabel("Episódios")
    plt.ylabel("steps")
    plt.grid(True)
    plt.show()
    
    media_movel_ = media_movel(steps_per_episode, janela=150)
    plt.plot(media_movel_, color="black", alpha=0.8)
    # plt.plot(steps_per_episode, color="green", alpha=.4)
    plt.title("Valor médio dos steps time")
    plt.xlabel("Episódios")
    plt.ylabel("média")
    plt.grid(True)
    plt.show()
    
    # Plotar o decaimento do epsilon ao longo dos epsódios
    # plt.figure(figsize=(8, 6))
    plt.plot(epsilon_hist, color="red")
    plt.title("Decaimento do epsilon")
    plt.xlabel("Episódios")
    plt.ylabel("epsilon")
    plt.grid(True)
    plt.show()
    
    # Plotar a evolução da média geral da Q-Table
    # plt.figure(figsize=(8, 6))
    # plt.plot(q_table_history, label="Média Geral da Q-Table", color="blue")
    plt.plot(q_table_history, color="blue")
    plt.title("Evolução da Média Geral da Q-Table")
    plt.xlabel("Episódios")
    plt.ylabel("Média dos Valores Q")
    #plt.legend()
    plt.grid(True)
    plt.show()


import seaborn as sns  # Para heatmaps sofisticados

# Calcular o valor médio de cada estado
state_means = np.mean(q_table, axis=1)  # Média ao longo das ações

# Remodelar a matriz para o formato do grid
rows=size_e
cols=size_e
state_mean_matrix = state_means.reshape(rows, cols)

# Plotando o heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(state_mean_matrix, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True)
plt.title("Mapa de Calor dos Valores Médios da Q-Table")
plt.xlabel("Colunas")
plt.ylabel("Linhas")
plt.show()
