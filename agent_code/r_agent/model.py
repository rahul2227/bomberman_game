import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import OneHotEncoder
from collections import deque
import numpy as np
import random
import math

# from agent_code.r_agent.train import Transition


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

class DQNAgent:
    def __init__(self, state_size, action_size, n_rounds, logger):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Experience replay buffer 
        # TODO : Can be the same as the Transition size
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
        # self.encoded_actions = self.encode_actions()
        
        
    def set_decay_rate(self) -> float:
        # number of rounds for decay rate
        decay_rate = -math.log((self.epsilon_min + 0.005) / self.epsilon) / self.n_rounds
        self.logger.info(f" n_rounds: {self.n_rounds}")
        self.logger.info(f"Determined exploration decay rate: {decay_rate}")
        return decay_rate
    
    def encode_actions():
        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(sparse=False, categories=[ACTIONS])

        # List to store encoded actions
        encoded_actions_list = []

        # Loop through each action in ACTIONS and encode it
        for action in ACTIONS:
            # Reshape the action into a 2D array
            action = np.array(action).reshape(-1, 1)
            
            # Perform one-hot encoding
            encoded_action = encoder.transform(action)
            
            # Append the encoded action to the list
            encoded_actions_list.append(encoded_action)
            
        return encoded_actions_list
    
    def action_to_encoded():
        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(sparse=False, categories=[ACTIONS])

        # List to store encoded actions
        encoded_actions_list = []

        # Dictionary to map actions to their encoded vectors
        action_to_encoded = {}

        # Loop through each action in ACTIONS and encode it
        for action in ACTIONS:
            # Reshape the action into a 2D array
            action = np.array(action).reshape(-1, 1)
            
            # Perform one-hot encoding
            encoded_action = encoder.transform(action)
            
            # Append the encoded action to the list
            encoded_actions_list.append(encoded_action)
            
            # Map the action to its encoded vector in the dictionary
            action_to_encoded[action[0][0]] = encoded_action
            
        return action_to_encoded
    
    def find_action_index(action, encoded_actions_list):
        # Convert the action to a one-hot encoded vector
        action_vector = np.array(action).reshape(1, -1)
        
        # Loop through the encoded_actions_list to find a match
        for index, encoded_action in enumerate(encoded_actions_list):
            if np.array_equal(encoded_action, action_vector):
                return index
        
        # If the action is not found, return -1 or raise an error as needed
        return -1  # or raise Exception("Action not found")

    def build_model(self):
        model = Sequential()
        # Define your DQN architecture here
        # Example:
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax')) 
        model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, transition): # TODO : Change the reward buffer according to the Transition
        self.memory.append(transition)
        
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

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        action_to_index = {action: index for index, action in enumerate(ACTIONS)}
        for state, action, reward, next_state, done in minibatch:
            target = reward
            # encoded_action = self.action_to_encoded(action)
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0]) # Better Q-learning update rule or Q-table usage?
                # Approximate Q-Learning?
            target_f = self.model.predict(state)
            
            action_index = action_to_index[action]
            target_f[0][action_index] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model.load_weights(filename)
