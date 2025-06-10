from tqdm import tqdm
import numpy as np
import random
import exec_action as a
'''
epsilon_hist = []
q_table_history = []
rewards_per_episode = []

evolution = {}
'''


def epsilon_greedy_policy(Q_state, epsilon, env):
  # if random number > greater than epsilon --> exploitation
  if(random.uniform(0,1) > epsilon):
    action = np.argmax(Q_state)
  # else --> exploration
  else:
    action = env.action_space.sample()
  return action


def fit(env, config):

    epsilon_hist = []
    q_table_history = []
    rewards_per_episode = []

    evolution = {}


    #hyper parameters 
    total_episodes = config.total_episodes
    learning_rate = config.learning_rate
    gamma = config.gamma
    epsilon = config.epsilon
    epsilon_decay_rate = 5/config.total_episodes
    min_epsilon = config.min_epsilon
    limit_of_stubbornness = config.limit_of_stubbornness
    act = config.act
    #todo: validate 

    # Divide position and velocity into 20 discretized segments
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20) # Between -1.2 and 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20) # Between -0.07 and 0.07

    action_space = 3
    state_space = len(pos_space) * len(vel_space)

    print("There are", state_space, "possible states")
    print("There are", action_space, "possible actions")

    Q = np.zeros((state_space, action_space))

    for episode in tqdm(range(total_episodes)):
        # Reset the environment
        state, info_ = env.reset()
        step = 0
        done = False
        total_reward = 0
        while not done:
            action = epsilon_greedy_policy(Q[state],epsilon, env)
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, info, step = a.execute_action(state, action, env, limit_of_stubbornness, act, step)
            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            Q[state][action] = Q[state][action] + learning_rate * (reward + gamma * np.max(Q[new_state]) - Q[state][action])

    
            # atualizo a Q-table
            #q_table[i_state_p, i_state_v, action] = (
                
            #    q_table[i_state_p, i_state_v, action] + alpha  * (
            #        reward + gamma * np.max(q_table[i_new_state_p, i_new_state_v,:]) 
            #        - q_table[i_state_p, i_state_v, action])
            #    
            #        )

            state = new_state
            total_reward += reward

        # Reduce epsilon (because we need less and less exploration)
        epsilon = max(epsilon * (1 - epsilon_decay_rate), min_epsilon)
        # epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        epsilon_hist.append(epsilon)

        q_table_history.append(np.mean(Q))  # Armazenar média geral da Q-Table
        rewards_per_episode.append(total_reward)
        evolution['epsilon_hist'] = epsilon_hist
        evolution['q_table_history'] = q_table_history
        evolution['rewards_per_episode'] = rewards_per_episode
        
        
    return Q, evolution
    