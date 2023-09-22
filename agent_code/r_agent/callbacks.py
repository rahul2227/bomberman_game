import os
import pickle
from queue import PriorityQueue
import random
from random import shuffle
import tensorflow as tf
from tensorflow import keras

import numpy as np
from agent_code.r_agent.exploration_rule_functions import coin_collector_rules, rb_act, rb_setup
from agent_code.r_agent.state_feature import state_to_features
import settings as s

from agent_code.r_agent.model import DQNAgent, get_next_action


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# Define a constant size for action_features
ACTION_FEATURES_SIZE = 6
DENSE_MODEL_STATE_SIZE = 9
CON_MODEL_STATE_SIZE = (5, 17, 17)



# returns probabilites for an array of game_states
def get_valid_probabilities_list(self, states, features):
    print(features[0].shape)
    if self.train:
        probabilities = self.model.model.predict(np.array(features))
    else:
        probabilities = self.model.predict(np.array(features))
    for i in range(len(probabilities)):
        if min(probabilities[i]) < 0:
            probabilities[i] += abs(min(probabilities[i]))
        probabilities[i] *= get_valid_actions(states[i]) # only allow valid actions
        probabilities[i] /= probabilities[i].sum() # normalize to statistical vector (= sums up to 1)
    return probabilities


def get_valid_probabilities(self, game_state):
    probabilities = get_valid_probabilities_list(self, [game_state], [state_to_features(game_state)])[0]
    return probabilities

def get_next_action(self, game_state):
    probabilities = get_valid_probabilities(self, game_state)
    choice = np.random.choice(ACTIONS, p=probabilities)
    # choice = ACTIONS[np.argmax(probabilities)]

    return probabilities, choice

def get_valid_actions(game_state):
    _, _, bomb, (x, y) = game_state["self"]
    walls = game_state["field"]
    bombs = list(map(lambda x: x[0], game_state["bombs"]))

    actions = np.ones(6)

    if walls[x][y-1] != 0 or (x, y-1) in bombs:
        actions[0] = 0 # can't go up
    if walls[x+1][y] != 0 or (x+1, y) in bombs:
        actions[1] = 0 # can't go right
    if walls[x][y+1] != 0 or (x, y+1) in bombs:
        actions[2] = 0 # can't go down
    if walls[x-1][y] != 0 or (x-1, y) in bombs:
        actions[3] = 0 # can't go left

    # if True:
    if not bomb:
        actions[5] = 0 # can't plant bomb

    return actions




def setup(self): 
    
    rb_setup(self)
    
    if self.train or not os.path.isfile("r-agent-saved-target-model.h5"):
        self.logger.info("Setting up model from scratch.")
        # Configure GPU if available
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # self.model = DQNAgent(state_size=DENSE_MODEL_STATE_SIZE, action_size=6, n_rounds= self.n_rounds, logger=self.logger)
        self.model = DQNAgent(state_size= CON_MODEL_STATE_SIZE, action_size=6, n_rounds= self.n_rounds, logger=self.logger)
        # There are 2 models in this DQNAgent class and we need to use only the target_model with respect to the model. 
    else:
        self.logger.info("Loading model from saved state.")
        self.model = keras.models.load_model("r-agent-saved-target-model.h5")


# TODO : Implement the query for taking only valid probabilities from the model instead of argmax


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    if self.train: 
        random_prob = self.model.epsilon
        self.logger.debug(f'random probab = {self.model.epsilon}')
    if self.train and random.random() < random_prob:
        # coin_collector_rules(self, game_state)
        choice = rb_act(self, game_state)
        
    # Current game state
    # state = state_to_features(game_state=game_state)
    # self.logger.info(f"Shape of the state {state}")

    self.logger.debug("Querying model for action.")
    # encoded_action = self.model.action_to_encoded(state)
    # if self.train:
    #     act_values = self.model.model.predict(state) 
    # else:
    #     act_values = self.model.predict(state)
    # best_action_index = np.argmax(act_values[0]) # TODO : Valid probabilities for actions to be taken
    # best_action = ACTIONS[best_action_index]
    # self.logger.debug(f"value predicted by model {act_values[0]} and best action {best_action}")
    # return np.random.choice(ACTIONS, p=self.model)
    
    probabilities, choice = get_next_action(self, game_state)
    print(probabilities.shape)
    print(choice)
    self.logger.debug(probabilities)
    self.logger.debug(f"Chose action: {choice}")
    return choice

