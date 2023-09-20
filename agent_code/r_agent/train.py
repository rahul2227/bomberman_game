from collections import namedtuple, deque

import pickle
from typing import List
from agent_code.r_agent.model import DQNAgent
import tensorflow as tf
from tensorflow import keras

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """    
    #Here I need to send the shape 
    # self.model = DQNAgent(state_size=9, action_size=6) # What can be the state_size?
    # Can it be obtained from the game_environment?
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events))) # What is this doing exactly and how to utilize this?
    # transition array?
    
    #survival event
    survival = [
        e.KILLED_SELF,
        e.GOT_KILLED,
        e.SURVIVED_ROUND,
    ]
    done = False
    for event in events:
        if event in survival:
            done = True
    
    self.model.remember(state_to_features(old_game_state), self_action, reward_from_events(self, events), state_to_features(new_game_state), done)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    # TODO: Finally update the model on rewards/penalties from the past game to increase it's future performance
    
    # remembering the last rewards
    self.model.remember(state_to_features(last_game_state), last_action, reward_from_events(self, events), None, True)    
    
    # replay
    self.model.replay(batch_size=100)
    
    # target model train
    self.model.target_train()

    # Store the model
    with open("r-agent-saved-model.h5", "wb") as file:
        # pickle.dump(self.model, file)
        # keras.models.save_model(self.model.model, file)
        self.model.model.save('r-agent-saved-model.h5', save_format='h5')


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 14, 
        e.KILLED_OPPONENT: 24,
        e.KILLED_SELF: 0.5, 
        e.GOT_KILLED: 1, 
        e.SURVIVED_ROUND: 5, 
        e.INVALID_ACTION: -1,
        e.MOVED_UP: 1.5,
        e.MOVED_DOWN: 1.5,
        e.MOVED_LEFT: 1.5,
        e.MOVED_RIGHT: 1.5,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
