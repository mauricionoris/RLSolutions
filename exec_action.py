import numpy as np
stubbornness = []

def execute_action(state, action, env, limit_of_stubbornness, act):

    new_state, reward, terminated, truncated, info = env.step(action)

    if new_state == state:
        stubbornness.append((state,action))

    if stubbornness.count((state,action)) > limit_of_stubbornness:
        stubbornness.clear()
        new_action = np.argmax(info["action_mask"])
        print("Stuborn action on:", state, act[action], 'replaced by:', act[new_action] )
        new_state, reward, terminated, truncated, info = env.step(new_action)

    return new_state, reward, terminated, truncated, info