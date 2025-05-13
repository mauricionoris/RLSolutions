import numpy as np
stubbornness = []

def execute_action(state, action, env, limit_of_stubbornness, act, step):

    done = False
    new_state, reward, terminated, truncated, info = env.step(action)
       
    if new_state == state:
        stubbornness.append((state,action))

    if stubbornness.count((state,action)) > limit_of_stubbornness:
        stubbornness.clear()
        new_action = 1 # right #0 #up #np.argmax(info["action_mask"])
        print("Stuborn action on:", state, act[action], 'replaced by:', act[new_action] )
        new_state, reward, terminated, truncated, info = env.step(new_action)

    step += 1
    if terminated or truncated:
        done = True
        


    return new_state, reward, done, info, step