from collections import namedtuple, deque

import pickle
from typing import List
from agent_code.r_agent.model_old import DQNAgent
import tensorflow as tf
from tensorflow import keras

import events as e
from .callbacks import state_to_features





def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """    
    #Here I need to send the shape 
    # self.model = DQNAgent(state_size=9, action_size=6) # What can be the state_size?
    # Can it be obtained from the game_environment?
    # self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(SURVIVED_ROUND)
    #     events.append(PLACEHOLDER_EVENT)

    # if len(self.model.memory) > TRANSITION_HISTORY_SIZE:
        # self.transitions.pop(0)
        # self.model.remember_buffer_update()
        
    # state_to_features is defined in callbacks.py
    self.model.remember(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events), False)
    # train
    if old_game_state is not None:
        self.model.replay()  


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    # remembering the last rewards
    self.model.remember(state_to_features(last_game_state), last_action, state_to_features(None), reward_from_events(self, events), True)
    
    # replay
    self.model.replay() 
    
    # target model train
    self.model.target_train()

    # Store the model
    with open("r-agent-saved-model.h5", "wb") as file:
        self.model.model.save('r-agent-saved-model.h5', save_format='h5')
        self.model.target_model.save('r-agent-saved-target-model.h5', save_format='h5')


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
