import gym
from gym.envs.registration import register
#from cliff_walking_customEnv import CliffWalkingEnv 


def env_registry(envid, entry_p):



    # Register the environment
    register(
        id=envid,
        entry_point= entry_p,  # Replace my_module with the actual module name
    )
    print(gym.envs.registration.registry[envid])
    

    # Create an instance of the environment
#    env = gym.make(entry_p)
#    print(env.metadata)
#    env.close()

def env_unregistry(envid):
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for env in env_dict:
        if envid in env:
             print("Remove {} from registry".format(env))
             del gym.envs.registration.registry.env_specs[env]


def check_env(envid):
   return envid in gym.envs.registration.registry

def getgym():
    return gym

#env_unregistry(envid)
#check_env(envid)




