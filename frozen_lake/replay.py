import numpy as np
import exec_action as a

from IPython.display import clear_output
import time


def view(Q, env, config, mode):
    

    if mode == 'view':
        for i in range(config.n_views):
            state, info = env.reset() # reset state
            done = False
            step=0
            while not done:
                action = np.argmax(Q[state])
                new_state, reward, terminated, truncated, info = a.execute_action(state, action, env, config.limit_of_stubbornness, config.act)
                print(f"step: {step:<5} action: {config.act[action]:<8} reward: {reward} info: info[action_mask] new state: {new_state}")
                state = new_state
                step += 1
                if terminated or truncated:
                    done = True

    if mode == 'notebook':
        for i in range(config.n_views):
            state, info = env.reset() # reset state
            done = False
            step=0
            while not done:
                clear_output(wait=True)
                print(env.render())
                time.sleep(0.5)
                action = np.argmax(Q[state])
                new_state, reward, terminated, truncated, info = a.execute_action(state, action, env, config.limit_of_stubbornness, config.act)
                print(f"step: {step:<5} action: {config.act[action]:<8} reward: {reward} info: {info['action_mask']} new state: {new_state}")
                state = new_state
                step += 1
                if terminated or truncated:
                    done = True
    
    env.close()


