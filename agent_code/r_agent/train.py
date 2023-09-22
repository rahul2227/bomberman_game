from collections import namedtuple, deque

import pickle
from typing import List
import numpy as np
import tensorflow as tf
from tensorflow import keras
from agent_code.r_agent.parameter import ACTIONS, ALREADY_VISITED, MINI_BATCH_SIZE, NEW_LOCATION_VISITED, SURVIVED_BOMB, SURVIVED_ROUND

import events as e
from .callbacks import get_valid_probabilities_list, state_to_features





def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """    
    self.memory = []
    self.gamma = 0.95
    self.visited_coords = []


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    
    add_custom_events(self, new_game_state, events)
        
    # train and buffer
    if old_game_state is not None:
        # state_to_features is defined in callbacks.py
        self.memory.append((state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events), False))
        # train_model(self)  


def add_custom_events(self, new_game_state, events):
    events.append(SURVIVED_ROUND)

    _, _, _, coords = new_game_state["self"]
    if coords in self.visited_coords:
        events.append(ALREADY_VISITED)
    else:
        events.append(NEW_LOCATION_VISITED)
        self.visited_coords.append(coords)

    if SURVIVED_ROUND in events and e.BOMB_EXPLODED in events:
        events.append(SURVIVED_BOMB)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    # remembering the last rewards
    self.memory.append((state_to_features(last_game_state), last_action, state_to_features(last_game_state), reward_from_events(self, events), True))
    
    # replay
    for i in range(10):
        train_model(self) 
    
    # target model train
    self.target_model.set_weights(self.model.get_weights())

    # Store the model
    # with open("r-agent-saved-model.h5", "wb") as file:
    self.model.save('r-agent-saved-model.h5', save_format='h5')
    self.target_model.save('r-agent-saved-target-model.h5', save_format='h5')


def reward_from_events(self, events: List[str]) -> int: # TODO : Write a helper function to plot reward mapping
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5, 
        # e.KILLED_OPPONENT: 200,
        # e.BOMB_DROPPED: 1,
        e.KILLED_SELF: -100, 
        # e.GOT_KILLED: -50,
        # e.OPPONENT_ELIMINATED: 0.5, 
        # e.SURVIVED_ROUND: 5,
        e.WAITED: -1, 
        e.INVALID_ACTION: -1,
        SURVIVED_ROUND: -.5,
        NEW_LOCATION_VISITED: 1,
        ALREADY_VISITED: -.5
        # e.MOVED_UP: 2,
        # e.MOVED_DOWN: 2,
        # e.MOVED_LEFT: 2,
        # e.MOVED_RIGHT: 2,
        # PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def train_model(self):
    minibatch = MINI_BATCH_SIZE
    if MINI_BATCH_SIZE > len(self.memory):
        minibatch = len(self.memory)
    
    # Randomly sample from self.memory with replacement
    sampled_indices = np.random.choice(len(self.memory), minibatch, replace=True)
    minibatch = [self.memory[i] for i in sampled_indices]
    action_to_index = {action: index for index, action in enumerate(ACTIONS)}
    for state, action, next_state, reward, done in minibatch:
        print(state.shape)
        print(next_state.shape)
        print(self.target_model.summary())
        print(next_state)
        target = reward
        if not done:
            # target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            # next_state = np.expand_dims(next_state, axis=0)  # Add a batch dimension
            # np.array(next_state)
            target = reward + self.gamma * np.amax(self.target_model.predict(next_state)) # TODO : Why the hell the shape is (None, 17, 17) here???????????
        # target_f = self.model.predict(state)
        target_f = get_valid_probabilities_list(self, state, next_state)
        
        action_index = action_to_index[action]
        target_f[0][action_index] = target
        self.model.fit(state, target_f, verbose=0) # epochs = 0 maybe
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay
        
        
def train_target_model(self):
    weights = self.model.get_weights()
    target_weights = self.target_model.get_weights()
    for i in range(len(target_weights)):
        target_weights[i] = weights[i]
    self.target_model.set_weights(target_weights)