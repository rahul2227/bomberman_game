import os
import pickle
from queue import PriorityQueue
import random
from random import shuffle
import tensorflow as tf
from tensorflow import keras

import numpy as np
from typing import Tuple, List
from agent_code.r_agent.exploration_rule_functions import coin_collector_rules
import settings as s

from agent_code.r_agent.model import DQNAgent


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
# Define a constant size for action_features
ACTION_FEATURES_SIZE = 6


def setup(self): 
    if self.train or not os.path.isfile("r-agent-saved-target-model.h5"):
        self.logger.info("Setting up model from scratch.")
        # weights = np.random.rand(len(ACTIONS))
        # self.model = weights / weights.sum()
        # print(self.n_rounds)
        # Configure GPU if available
        physical_devices = tf.config.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        self.model = DQNAgent(state_size=9, action_size=6, n_rounds= self.n_rounds, logger=self.logger)
        # There are 2 models in this DQNAgent class and we need to use only the target_model with respect to the model. 
    else:
        self.logger.info("Loading model from saved state.")
        # with open("r-agent-saved-model.h5", "rb") as file:
            # self.model = pickle.load(file)
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
        coin_collector_rules(self, game_state)
        
    # Current game state
    state = state_to_features(game_state=game_state)
    self.logger.info(f"Shape of the state {state}")

    self.logger.debug("Querying model for action.")
    # encoded_action = self.model.action_to_encoded(state)
    if self.train:
        act_values = self.model.model.predict(state) 
    else:
        act_values = self.model.predict(state)
    best_action_index = np.argmax(act_values[0]) # TODO : Valid probabilities for actions to be taken
    best_action = ACTIONS[best_action_index]
    self.logger.debug(f"value predicted by model {act_values[0]} and best action {best_action}")
    # return np.random.choice(ACTIONS, p=self.model)
    return best_action


def _get_neighboring_tiles(own_coord, radius) -> List[Tuple[int]]:
    x, y = own_coord
    # Finding neighbouring tiles
    neighboring_coordinates = []
    for i in range(1, radius + 1):
        neighboring_coordinates.extend([
            (x, y + i),  # down in the matrix
            (x, y - i),  # up in the matrix
            (x + i, y),  # right in the matrix
            (x - i, y)  # left in the matrix
        ])
    return neighboring_coordinates


# Feature 1: Count the number of walls in the immediate surrounding tiles within a given radius.
def count_walls(own_position, game_state, radius):
    return sum(
        1 for coord in _get_neighboring_tiles(own_position, radius)
        if 0 <= coord[0] < game_state["field"].shape[0] and 0 <= coord[1] < game_state["field"].shape[1]
        and game_state["field"][coord] == -1
    )


# Feature 2: Check for bomb presence in the immediate surrounding tiles within a given radius.
def check_bomb_presence(own_position, game_state, radius):
    return any(
        bomb[0] in _get_neighboring_tiles(own_position, radius)
        and bomb[1] != 0
        for bomb in game_state["bombs"]
    )


# Feature 3: Check for agent presence in the immediate surrounding tiles within a given radius.
def check_agent_presence(own_position, game_state, radius):
    return any(
        agent[3] in _get_neighboring_tiles(own_position, radius)
        for agent in game_state["others"]
    )

# Feature 4: Move makes sense
def moves_making_sense(game_state):
    # Gathering information about the game state
    arena = game_state['field']
    _, score, bombs_left, (x, y) = game_state['self']
    bombs = game_state['bombs']
    bomb_xys = [xy for (xy, t) in bombs]
    others = [xy for (n, s, b, xy) in game_state['others']]
    coins = game_state['coins']
    bomb_map = np.ones(arena.shape) * 5
    for (xb, yb), t in bombs:
        for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
            if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
                bomb_map[i, j] = min(bomb_map[i, j], t)

    # Check which moves make sense at all
    # TODO : make so that we give reward if model takes action from valid moving directions
    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles, valid_actions = [], []
    for d in directions:
        if ((arena[d] == 0) and
                (game_state['explosion_map'][d] < 1) and
                (bomb_map[d] > 0) and
                (not d in others) and
                (not d in bomb_xys)):
            valid_tiles.append(d)
    if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
    if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
    if (x, y - 1) in valid_tiles: valid_actions.append('UP')
    if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
    if (x, y) in valid_tiles: valid_actions.append('WAIT')
    
    # Create a mapping from action names to integers
    action_to_index = {action: index for index, action in enumerate(ACTIONS)}

    # Initialize the valids array with zeros
    valids = [0] * len(ACTIONS)

    # Set 1 for valid actions
    for action in valid_actions:
        index = action_to_index.get(action, -1)  # Get the index for the action
        if index != -1:
            valids[index] = 1
    
    return valids


def state_to_features(game_state) -> np.array:
    if game_state is None:
        print("First game state is None")
        return np.zeros(9)

    own_position = game_state["self"][-1]

    # Calculate features
    wall_counter = count_walls(own_position, game_state, 3)
    bomb_present = check_bomb_presence(own_position, game_state, 3)
    agent_present = check_agent_presence(own_position, game_state, 3)
    action_features = moves_making_sense(game_state=game_state) 
    # print(action_features)
    
    
    # Ensure action_features has a constant size of ACTION_FEATURES_SIZE
    # if len(action_features) < ACTION_FEATURES_SIZE:
    #     action_features.extend([-1] * (ACTION_FEATURES_SIZE - len(action_features)))
    # elif len(action_features) > ACTION_FEATURES_SIZE:
    #     action_features = action_features[:ACTION_FEATURES_SIZE]
    
    # Calculate features
    features = np.array([int(wall_counter > 2), int(bomb_present), int(agent_present)])
    
    # Concatenate the action_features
    features = np.concatenate((features, action_features))
    # print(features)
    
    # Reshape to (1, 9)
    features = features.reshape((1, -1))
    # print(features)

    
    return features
