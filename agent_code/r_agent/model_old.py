import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import OneHotEncoder
from collections import deque
import numpy as np
import random
import math
from agent_code.r_agent.callbacks import ACTIONS, get_valid_probabilities_list
from agent_code.r_agent.parameter import MINI_BATCH_SIZE

from agent_code.r_agent.state_feature import state_to_features







class DQNAgent:
    def __init__(self, state_size, action_size, n_rounds, logger):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate = 1.0
        self.epsilon_min = 0.05
        self.n_rounds = n_rounds
        self.logger = logger
        self.epsilon_decay = self.set_decay_rate() # 0.995 # set a dynamic decay rate based on number of rounds
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.target_model = self.build_model()
        # self.update_target_model()
        self.target_train()
        
        
    def set_decay_rate(self) -> float:
        # number of rounds for decay rate
        decay_rate = -math.log((self.epsilon_min + 0.005) / self.epsilon) / self.n_rounds
        self.logger.info(f" n_rounds: {self.n_rounds}")
        self.logger.info(f"Determined exploration decay rate: {decay_rate}")
        return decay_rate


    # def build_model(self):
    #     model = Sequential()
    #     # Define your DQN architecture here
    #     # Example:
    #     model.add(Dense(128, input_dim=self.state_size, activation='relu'))
    #     model.add(Dense(64, activation='relu'))
    #     model.add(Dense(self.action_size, activation='softmax')) 
    #     model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=self.learning_rate), metrics=["accuracy"])
    #     return model
    
    


    def remember(self, state, action, reward, next_state, done): 
        self.memory.append((state, action, reward, next_state, done))
        
    def remember_buffer_update(self):
        self.memory.pop(0)
        
    def encode_actions(actions):
        """
        Encode a list of actions to numerical values.

        :param actions: List of actions
        :return: Dictionary mapping actions to numerical values
        """
        action_to_num = {action: i for i, action in enumerate(actions)}
        return action_to_num

    def replay(self):
        minibatch = MINI_BATCH_SIZE
        if MINI_BATCH_SIZE > len(self.memory):
            minibatch = len(self.memory)
        # print(self.memory)
        # minibatch = np.random.choice(self.memory, batch_size, replace=True)
        # minibatch = np.random.choice(self.memory, MINI_BATCH_SIZE, replace=True)
        # minibatch = random.sample(self.memory, minibatch)
        
        # Randomly sample from self.memory with replacement
        sampled_indices = np.random.choice(len(self.memory), minibatch, replace=True)
        minibatch = [self.memory[i] for i in sampled_indices]
        # print(minibatch[0].shape)
        # state, action, reward, next_state, done = minibatch[0]
        # print(next_state.shape)
        action_to_index = {action: index for index, action in enumerate(ACTIONS)}
        for state, action, next_state, reward, done in minibatch:
            print(state.shape)
            print(next_state.shape)
            print(self.target_model.summary())
            target = reward
            if not done:
                # target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
                next_state = np.expand_dims(next_state, axis=0)  # Add a batch dimension
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state), axis=1) 
            # target_f = self.model.predict(state)
            target_f = get_valid_probabilities_list(self, state, next_state)
            
            action_index = action_to_index[action]
            target_f[0][action_index] = target
            self.model.fit(state, target_f, epochs=10, verbose=0) # epochs = 0 maybe
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)
        

