



import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from random import shuffle

from .Model import Jarvis
from .statetofeatures import *

import events as e

PARAMETERS = 'model' 
ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']


def setup(self):
    """
    This is called once when loading each agent.
    Preperation such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.network = Jarvis()

    if self.train:
        self.logger.info("New model training")

    else:
        self.logger.info(f"Load model '{PARAMETERS}'.")
        filename = os.path.join("parameters", f'{PARAMETERS}.pt')
        self.network.load_state_dict(torch.load(filename))
        self.network.eval()
    
   

    self.bomb_buffer = 0
    

def act(self, game_state: dict) -> str:
    
    if game_state is None:
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    features = state_to_features(self, game_state)
    Q = self.network(features)

    if self.train: # Exploration vs exploitation
        eps = self.epsilon_arr[self.episode_counter]
        # random action
        if random.random() <= eps: 
            if eps > 0.1:
                if np.random.randint(10) == 0:    
                    action = np.random.choice(ACTIONS, p=[.167, .167, .167, .167, .166, .166])
                    
                    return action

                
            else:
                action = np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
                
                return action

    action_prob	= np.array(torch.softmax(Q,dim=1).detach().squeeze())
    best_action = ACTIONS[np.argmax(action_prob)]
   
    return best_action