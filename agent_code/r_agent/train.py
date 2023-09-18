from collections import namedtuple, deque

import pickle
from typing import List
import tensorflow as tf

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
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    # self.model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(5, activation="relu"), # TODO: Need to see the number of inputs for the model, to define the layers
    #     tf.keras.layers.Dense(4, activation=tf.keras.activations.softmax),
    # ])
    
    #Here I need to send the shape 
    self.model = curiosity_agent()
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    
def curiosity_agent():
    # This network is for predicting the intrinsic reward that the agent will get from
    # game exploration
    state_input = tf.keras.layers.Input(shape = (4,)) # shape here is for only exploratory features
    next_state_input = tf.keras.layers.Input(shape = (4,)) # [up, down, left, right]
    action_input = tf.keras.layers.Input(shape = (4,))
    
    # Concatenating the states and actions for forward dynamics(what actions agent will take)
    concatenated_input = tf.keras.layers.concatenate([state_input, action_input], axis=-1)
    
    # prediction of the next state
    forward_model_output = tf.keras.layers.Dense(32, activation='relu')(concatenated_input)
    # forward_model_output = tf.keras.layers.Dense(self.state_shape[0])(forward_model_output)
    forward_model_output = tf.keras.layers.Dense(4)(forward_model_output)
        
    # Predict the action
    inverse_model_output = tf.keras.layers.Dense(32, activation='relu')(state_input)
    # inverse_model_output = tf.keras.layers.Dense(self.n_actions, activation='softmax')(inverse_model_output)
    inverse_model_output = tf.keras.layers.Dense(4, activation='softmax')(inverse_model_output)
    
    forward_loss = tf.reduce_sum(tf.square(forward_model_output - next_state_input), axis=-1)
    inverse_loss = tf.reduce_sum(action_input * tf.math.log(inverse_model_output + 1e-10), axis=-1)
    
    combined_loss = forward_loss + inverse_loss
    curiosity_model = tf.keras.Model(inputs=[state_input, next_state_input, action_input], outputs=combined_loss)
    
    return curiosity_model


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events))) # What is this doing exactly and how to utilize this?
    # transition array?


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

    # Store the model
    with open("r-agent-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -2,
        e.GOT_KILLED: -3,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
