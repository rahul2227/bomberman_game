
import random
from typing import List

import events as e
import os

from collections import deque

import numpy as np
import torch
import torch.optim as optim 
import torch.nn as nn
from .Trainexperience import eps_policy, experience_add, get_score, save_parameters, update_network, add_remaining_experience, train_network
from .statetofeatures import state_to_features
from .Rewardagent import reward_from_events,own_event_reward

import copy


#Hyperparameter for Training
TRAIN_FROM_SCRETCH = True
LOAD = 'model'


EPSILON = (0.5,0.05)
LINEAR_CONSTANT_QUOTIENT = 0.9

DISCOUNTING_FACTOR = 0.6
BUFFERSIZE = 2000 
BATCH_SIZE = 120 

LOSS_FUNCTION = nn.MSELoss()
OPTIMIZER = optim.Adam
LEARNING_RATE = 0.001

TRAINING_EPISODES = 20000



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if not TRAIN_FROM_SCRETCH: #load current parameters
        self.network.load_state_dict(torch.load(f'network_parameters\{LOAD}.pt'))
        self.network.eval()
        


    self.network.initialize_training(LEARNING_RATE, DISCOUNTING_FACTOR, EPSILON,
                                        BUFFERSIZE, BATCH_SIZE, 
                                        LOSS_FUNCTION, OPTIMIZER,
                                        TRAINING_EPISODES)

    self.epsilon_arr = eps_policy(self.network, LINEAR_CONSTANT_QUOTIENT) 
    self.buffer_experience = []

    self.episode_counter = 0
    self.total_episodes = TRAINING_EPISODES

    
    self.game_score = 0      
    self.game_score_arr = []

    self.network_new = copy.deepcopy(self.network) 



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    
    experience_add(self, old_game_state, self_action, new_game_state, events, 5)
    if len(self.buffer_experience) > 0:
        train_network(self)
    
    self.game_score += get_score(events)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
   
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    experience_add(self, last_game_state, last_action, None, events, 5)
    if len(self.buffer_experience) > 0:
        train_network(self)

    update_network(self)
    
    self.game_score += get_score(events)
   

    self.episode_counter += 1
    
    if self.episode_counter % (TRAINING_EPISODES // 100) == 0: #save parameters and the game score array
        
        save_parameters(self,"model")
        np.savetxt(f"game_score_{self.episode_counter}",self.game_score_arr)
    
    
    if e.SURVIVED_ROUND in events:
        self.logger.info("Round survived")


