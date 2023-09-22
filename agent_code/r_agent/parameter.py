ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Batch size
MINI_BATCH_SIZE = 32

# Define a constant size for action_features
ACTION_FEATURES_SIZE = 6
DENSE_MODEL_STATE_SIZE = 9
CON_MODEL_STATE_SIZE = (5, 17, 17)
LEARNING_RATE = 0.0005

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Batch size
MINI_BATCH_SIZE = 32

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
SURVIVED_ROUND = "SURVIVED_ROUND"