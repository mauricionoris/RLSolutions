import gymnasium as gym

# Load the environment with render mode specified
env = gym.make('CartPole-v1', render_mode="human")

# Initialize the environment to get the initial state
state = env.reset()
print(state)
# Print the state space and action space
print("State space:", env.observation_space)
print("Action space:", env.action_space.n)

action = 1 # empurrãozinho inicial!
done = False
truncated = False
rewards=0
episodes = 1
for i in range(1000): 
    # env.reset()
    # rewards=0
    # while not done or not truncated:
    next_state, reward, done, truncated, info = env.step(action) 
    cart_position = next_state[0]
    cart_velocity = next_state[1]
    pole_angle = next_state[2]
    pole_angular_velocity = next_state[3]
    
    if pole_angle > 0:
        action = 1
    else:
        action = 0
        
    rewards+=reward
        
    # print('rewards: ', rewards, 'episode: ', i)
    
    if done or truncated:
        env.reset()
        print('rewards: ', rewards, 'episode: ', episodes)
        episodes+=1
        rewards=0

env.close()  # Close the environment when done