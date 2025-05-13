import numpy as np
import exec_action as a

from IPython.display import clear_output
import time

def view(Q, env, config, mode):

    

    for i in range(config.n_views):
        step=0
        done = False
        state, info = env.reset() # reset state
        while not done:
            if mode == 'notebook':
                clear_output(wait=True)
                print(env.render())
                time.sleep(0.5)

            action = np.argmax(Q[state])
            new_state, reward, done, info, step = a.execute_action(state, action, env, config.limit_of_stubbornness, config.act, step)
            print(f"step: {step:<5} action: {config.act[action]:<8} reward: {reward} info: info: info[action_mask] new state: new state: {new_state}")
            state = new_state

    env.close()


