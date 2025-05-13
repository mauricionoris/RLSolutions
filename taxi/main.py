import gymnasium as gym
import numpy as np
from pprint import pprint
import os

import qLearning as q
import replay as r
from hp import Params, get_config_hash

import present as p

def createEnv(mode):
    env = gym.make(config.env_name
                    , render_mode = config.render_mode[mode]
                    , max_episode_steps = config.max_episode_steps_)
    env.reset()

    return env 

def loadfile(h_file):

    
    if not os.path.isfile(h_file):
        return False, None, None
    
    trainfile  = np.load(h_file, allow_pickle=True)
    qtable  = trainfile['array']
    config_trainFile = Params(**trainfile['metadata'].item()['hp'])
    if config_trainFile != config:
        #sanity check of the conditions used for trained if different, delete de file and retrain
        os.remove(h_file)
        return False, None, None

    evolution = trainfile['metadata'].item()['evol']

    return True, qtable, evolution

def savefile(h_file, qtable, evolution):

    md = {}
    md['hp'] = config._asdict()
    md['evol'] = evolution
    np.savez(h_file,array=qtable, metadata=md)

def main(mode):

    pprint(config._asdict(), sort_dicts=True)
    h_file = './temp/' + get_config_hash(config) + '.npz'

    trained, qtable, evolution = loadfile(h_file)
    
    if not trained:
        #train
        env = createEnv('train')
        qtable, evolution = q.fit(env, config)
        env.close()
        savefile(h_file, qtable, evolution)

        

    env = createEnv(mode)
    r.view(qtable, env, config, mode)
    env.close()
    return evolution

def present_charts(evol, mode):

    figs = p.create_charts(evol['epsilon_hist'] , evol['q_table_history']  , evol['rewards_per_episode'])
    p.show(figs)

    if mode == 'view':
        input("Press Enter to exit...")

#################################################################################
config = Params(
    total_episodes     =6000,
    learning_rate      =0.1,
    gamma              =0.8,
    epsilon            =0.99,
    min_epsilon        =0.05,
    map_size           =5,
    seed               =42,
    is_slippery        =False,
    n_runs             =100, 
    action_size        =None,
    state_size         =None,
    proba_frozen       =0.9,
    savefig_folder     ='./fig/',
    env_name           ='Taxi-v3',
    render_mode        = {'train': 'rgb_array', 'notebook': 'ansi', 'view': 'human'},
    max_episode_steps_ =50,
    limit_of_stubbornness = 3,
    n_views=10,
    act = ['down', 'up', 'right', 'left', 'pick', 'drop']
)

if __name__ == "__main__":
    mode = 'view'
    evolution = main(mode)
    present_charts(evolution,mode)




