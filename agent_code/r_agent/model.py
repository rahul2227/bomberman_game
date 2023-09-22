import math
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam

from agent_code.r_agent.parameter import CON_MODEL_STATE_SIZE, EPSILON_MIN_VAL, LEARNING_RATE

def build_model():
    NUM_ACTION = 6
    model = Sequential()
    model.add(Conv2D(16, input_shape=CON_MODEL_STATE_SIZE, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(NUM_ACTION))
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate= LEARNING_RATE))
    return model


def decay_rate(self) -> float:
    # number of rounds for decay rate
    decay_rate = -math.log((EPSILON_MIN_VAL + 0.005) / self.epsilon) / self.n_rounds
    self.logger.info(f" n_rounds: {self.n_rounds}")
    self.logger.info(f"Determined exploration decay rate: {decay_rate}")
    return decay_rate
