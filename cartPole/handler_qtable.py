import os 
import numpy as np

def loadfile(h_file):

    if not os.path.isfile(h_file):
        return None
    
    trainfile  = np.load(h_file, allow_pickle=True)
    qtable  = trainfile['array']
    return qtable 


 

def savefile(h_file, qtable):
    
    md = {}
    np.savez(h_file,array=qtable, metadata=md)
    