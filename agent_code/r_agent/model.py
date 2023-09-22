from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam

def build_model():
    NUM_ACTION = 6
    model = Sequential()
    model.add(Conv2D(16, input_shape=CON_MODEL_STATE_SIZE, kernel_size=(5, 5), padding="same", activation="relu"))
    model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(NUM_ACTION))
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=self.learning_rate))
    return model