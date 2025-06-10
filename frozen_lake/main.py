#import gym
import numpy as np
from pprint import pprint
import os

import register as reg

import qLearning as q
import replay as r
from hp import Params, Grid,  get_config_hash

import present as p
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


envid= 'FrozenLake-v1' # not native on Gym.... imported a custom one !
#entry_p = 'cliff_walking_customEnv:CliffWalkingEnv'

def find_H_positions(grid):

    positions = []
    for row_idx, row in enumerate(grid):
        for col_idx, char in enumerate(row):
            if char == 'H':
                positions.append((row_idx, col_idx))
    return positions

def createEnv(mode):

    modes_avalilable = ['view', 'notebook','train']

    if mode not in modes_avalilable:
        raise ValueError("Invalid mode")
    

    #if not reg.check_env(envid):
    #   reg.env_registry(envid, entry_p)


    gym = reg.getgym()
    env = gym.make(config.env_name
                    , render_mode = config.render_mode[mode]
                    , max_episode_steps = config.max_episode_steps_
                    , is_slippery = config.is_slippery
                    , desc=config.map)
    
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
    #pprint(md)
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

        
    #print(config)
    env = createEnv(mode)
    r.view(qtable, env, config, mode)
    env.close()
    return evolution, qtable

def present_charts(evol, mode, qtable, config, exp_name):

    figs = p.create_charts(evol['epsilon_hist'] , evol['q_table_history']  , evol['rewards_per_episode'], config.grid, qtable, (config.map_size-1,config.map_size-1))
    p.show(figs)

    if mode == 'view':
        p.save_pic(figs, config.savefig_folder, exp_name)
        input("Press Enter to exit...")
        p.close(figs)



#################################################################################
''''
# config = Params(
    total_episodes     =5000,
    learning_rate      =0.1,
    gamma              =0.8,
    epsilon            =0.99,
    min_epsilon        =0.05,
    map_size           =size_e,
    seed               =seed,
    is_slippery        =False,
    n_runs             =100, 
    action_size        =None,
    state_size         =None,
    proba_frozen       =proba_frozen,
    savefig_folder     ='./fig/',
    env_name           = envid,
    render_mode        = {'train': 'rgb_array', 'notebook': 'ansi', 'view': 'human'},
    max_episode_steps_ =100,
    limit_of_stubbornness = 3,
    n_views=5,
    act = ['left', 'down', 'right', 'up'],
    starting_state=(3,0),
    grid = Grid((size_e,size_e),(0,0),(size_e-1,size_e-1),find_H_positions(my_map)),
    map=my_map
)'''

#'0: Move left / #'1: Move down / '2: Move right '3: Move up





def config_build(is_sleep, size_e, seed, proba_frozen):

    
    my_map = generate_random_map(size=size_e, p=proba_frozen, seed=seed)
    return Params(
        total_episodes     =3000,
        learning_rate      =0.1,
        gamma              =0.8,
        epsilon            =0.99,
        min_epsilon        =0.05,
        map_size           =size_e,
        seed               =seed,
        is_slippery        =is_sleep,
        n_runs             =50, 
        action_size        =None,
        state_size         =None,
        proba_frozen       =proba_frozen,
        savefig_folder     ='./fig/',
        env_name           = envid,
        render_mode        = {'train': 'rgb_array', 'notebook': 'ansi', 'view': 'human'},
        max_episode_steps_ =100,
        limit_of_stubbornness = 3,
        n_views=2,
        act = ['left', 'down', 'right', 'up'],
        starting_state=(3,0),
        grid = Grid((size_e,size_e),(0,0),(size_e-1,size_e-1),find_H_positions(my_map)),
        map=my_map)
    


if __name__ == "__main__":

    is_sleep = [False, True]
    size_l = [5, 7, 9, 11, 13, 15]
    seeds = 123
    proba_frozen = [0.9, 0.8, 0.7] 


    for s in is_sleep:
        for size_e in size_l:
            for p_frozen in proba_frozen:

                print(f'''
                        Experimento : 
                             is sleepy: {s}
                                  Size: {size_e}
                              p_frozen: {p_frozen}
                      ''')

                config = config_build(s,size_e,seeds,p_frozen)
                
                mode = 'view'
                
                evolution, qtable = main(mode)

                exp_name = f'is_sleepy-{s}-Size-{size_e}-p_frozen-{p_frozen}' 

                present_charts(evolution,mode, qtable, config, exp_name)
                
        #raise ValueError('para aq')

'''
1. Utilizando os parâmetros:
#size_e = 5, 7, 9, 11, 13 e 15
#seeds = 123
#proba_frozen = 0.9, 0.8, 0.7 
#is_slip = False
#Proceda a sintonia dos hiperparâmetros para o treinamento do agente na tentativa de melhor otimização da trajetória e também no 'descongelamento' do Q-Table.


#2. Tente descobrir por que a tabela Q-Table permace "zerada" mesmo alterando os hiperparâmetros. Faça as modificações necessárias na arquitetura de respostas do ambiente para melhorar o aprendizado buscando uma política ótima.

#3. Faça is_slip = True e repita os procedimentos anteriores.

4. Plote a Q-Table de uma maneira de fácil visualização para acompanhar os treinamentos.
5. Plote os gráficos das métricas mais importantes.
6. Plote o gráfico do ambiente de simulação (grid) com a política ótima representada por setas.
'''


