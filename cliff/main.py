#import gym
import numpy as np
from pprint import pprint
import os

import register as reg

import qLearning as q
import replay as r
from hp import Params, Grid,  get_config_hash

import present as p

envid= 'CliffWalking-v1' # not native on Gym.... imported a custom one !
entry_p = 'cliff_walking_customEnv:CliffWalkingEnv'

def createEnv(mode):

    modes_avalilable = ['view', 'notebook','train']

    if mode not in modes_avalilable:
        raise ValueError("Invalid mode")
    

    if not reg.check_env(envid):
       reg.env_registry(envid, entry_p)


    gym = reg.getgym()
    env = gym.make(config.env_name
                    , render_mode = config.render_mode[mode]
                    , max_episode_steps = config.max_episode_steps_
                    , is_slippery = config.is_slippery)
    env.reset()

    return env 

def loadfile(h_file):

    
    if not os.path.isfile(h_file):
        return False, None, None
    
    trainfile  = np.load(h_file, allow_pickle=True)
    qtable  = trainfile['array']
    config_trainFile = Params.from_dict(trainfile['metadata'].item()['hp']) #Params(**trainfile['metadata'].item()['hp'])

    if config_trainFile.to_dict() != config.to_dict():
        #sanity check of the conditions used for trained if different, delete de file and retrain
        os.remove(h_file)
        return False, None, None

    evolution = trainfile['metadata'].item()['evol']

    return True, qtable, evolution

def savefile(h_file, qtable, evolution):
    
    md = {}
    md['hp'] = config.to_dict()
    pprint(md)
    md['evol'] = evolution
    np.savez(h_file,array=qtable, metadata=md)

def main(mode):

    h_file = './temp/' + get_config_hash(config) + '.npz'

    trained, qtable, evolution = loadfile(h_file)
    
    #print(trained,qtable, evolution)
    #return None,None
    if not trained:
        #train
        env = createEnv('train')
        qtable, evolution = q.fit(env, config)
        env.close()
        savefile(h_file, qtable, evolution)

        

    env = createEnv(mode)
    r.view(qtable, env, config, mode)
    env.close()
    return evolution, qtable

def present_charts(evol, mode, qtable, config):

    figs = p.create_charts(evol['epsilon_hist'] , evol['q_table_history']  , evol['rewards_per_episode'], config.grid, qtable)
    p.show(figs)

    if mode == 'view':
        p.save_pic(figs, config.savefig_folder)
        input("Press Enter to exit...")






#################################################################################
config = Params(
    total_episodes     =1500,
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
    env_name           = envid,
    render_mode        = {'train': 'rgb_array', 'notebook': 'ansi', 'view': 'human'},
    max_episode_steps_ =100,
    limit_of_stubbornness = 3,
    n_views=2,
    act = ['up', 'right', 'down', 'left'],
    starting_state=(3,0),
    grid = Grid((4,12),(3,0),(3,11),[(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),(3,9),(3,10)])
)



if __name__ == "__main__":
    mode = 'view'
    evolution, qtable = main(mode)
    present_charts(evolution,mode, qtable, config)




