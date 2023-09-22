import os
import random
import tensorflow as tf
from tensorflow import keras

import numpy as np
from agent_code.r_agent.exploration_rule_functions import rb_act, rb_setup
from agent_code.r_agent.model import build_model, decay_rate
from agent_code.r_agent.parameter import ACTIONS, CON_MODEL_STATE_SIZE, EPSILON_MIN_VAL, INITIAL_EPSILON
from agent_code.r_agent.state_feature import state_to_features
import settings as s


# returns probabilites for an array of game_states
def get_valid_probabilities_list(self, states, features):
    print(features[0].shape)
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
        # self.model = DQNAgent(state_size= CON_MODEL_STATE_SIZE, action_size=6, n_rounds= self.n_rounds, logger=self.logger)
        # There are 2 models in this DQNAgent class and we need to use only the target_model with respect to the model. 
        self.epsilon = INITIAL_EPSILON
        self.epsilon_min = EPSILON_MIN_VAL
        self.epsilon_decay = decay_rate(self)
        self.model = build_model()
        self.target_model = build_model()
        self.target_model.set_weights(self.model.get_weights())
    else:
        self.logger.info("Loading model from saved state.")
        self.model = keras.models.load_model("r-agent-saved-model.h5")
        self.target_model = build_model()
        self.target_model.set_weights(self.model.get_weights())



def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    choice=None
    # todo Exploration vs exploitation
    if self.train and random.random() < self.epsilon:
        choice = rb_act(self, game_state)
        
    self.logger.debug("Querying model for action.")
    
    probabilities, choice = get_next_action(self, game_state)
    # print(probabilities.shape)
    # print(choice)
    self.logger.debug(probabilities)
    self.logger.debug(f"Chose action: {choice}")
    return choice

