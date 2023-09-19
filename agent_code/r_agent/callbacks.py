import os
import pickle
from queue import PriorityQueue
import random
from random import shuffle
from tensorflow import keras

import numpy as np
from typing import Tuple, List
import settings as s

from agent_code.r_agent.model import DQNAgent


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    if self.train or not os.path.isfile("r-agent-saved-model.h5"):
        self.logger.info("Setting up model from scratch.")
        # weights = np.random.rand(len(ACTIONS))
        # self.model = weights / weights.sum()
        self.model = DQNAgent(state_size=3, action_size=6)
    else:
        self.logger.info("Loading model from saved state.")
        # with open("r-agent-saved-model.h5", "rb") as file:
            # self.model = pickle.load(file)
        self.model = keras.models.load_model("r-agent-saved-model.h5")


def look_for_targets(free_space, start, targets, logger=None):
    """Find direction of the closest target that can be reached via free tiles.

    Performs a breadth-first search of the reachable free tiles until a target is encountered.
    If no target can be reached, the path that takes the agent closest to any target is chosen.

    Args:
        free_space: Boolean numpy array. True for free tiles and False for obstacles.
        start: the coordinate from which to begin the search.
        targets: list or array holding the coordinates of all target tiles.
        logger: optional logger object for debugging.
    Returns:
        coordinate of first step towards the closest target or towards tile closest to any target.
    """
    if len(targets) == 0:
        return None

    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()

    while len(frontier) > 0:
        current = frontier.pop(0)
        # Find distance from current position to all targets, track closest
        d = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if d + dist_so_far[current] <= best_dist:
            best = current
            best_dist = d + dist_so_far[current]
        if d == 0:
            # Found path to a target's exact position, mission accomplished!
            best = current
            break
        # Add unexplored free neighboring tiles to the queue in a random order
        x, y = current
        neighbors = [(x, y) for (x, y) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if free_space[x, y]]
        shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger: logger.debug(f'Suitable target found at {best}')
    # Determine the first step towards the best found target tile
    current = best
    while True:
        if parent_dict[current] == start: return current
        current = parent_dict[current]


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
        # self.logger.debug("Choosing action based on the search")
        # # return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        # # 80%: walk in any direction. 10% wait. 10% bomb.
        # # return np.random.choice(ACTIONS, p=[.225, .225, .225, .225, .1]) # , .1
        
        # # Constants for actions
        # UP, RIGHT, DOWN, LEFT = 'UP', 'RIGHT', 'DOWN', 'LEFT'
        # # Check if there are any coins left to collect
        # if not game_state['coins']:
        #     self.logger.debug("No coins left to collect. Choosing action purely at random.")
        #     return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])  # Adjust action probabilities

        # # Extract the agent's position
        # agent_x, agent_y = game_state['self'][3]

        # # Find the nearest coin using A* search
        # nearest_coin = self.find_nearest_coin(game_state, (agent_x, agent_y))

        # # If a valid path to a coin is found, move towards it
        # if nearest_coin is not None:
        #     coin_x, coin_y = nearest_coin

        #     # Determine the direction to move
        #     if coin_x < agent_x:
        #         return LEFT
        #     elif coin_x > agent_x:
        #         return RIGHT
        #     elif coin_y < agent_y:
        #         return UP
        #     elif coin_y > agent_y:
        #         return DOWN

        # # If no valid path is found, choose an action purely at random
        # self.logger.debug("No valid path to a coin. Choosing action purely at random.")
        # return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])  # Adjust action probabilities
        self.logger.info('Picking action according to rule set')
        # Gather information about the game state
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
        # Disallow the BOMB action if agent dropped a bomb in the same spot recently
        if bombs_left > 0:
            valid_actions.append('BOMB')
        self.logger.debug(f'Valid actions: {valid_actions}')

        # Collect basic action proposals in a queue
        # Later on, the last added action that is also valid will be chosen
        action_ideas = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        shuffle(action_ideas)

        # Compile a list of 'targets' the agent should head towards
        cols = range(1, arena.shape[0] - 1)
        rows = range(1, arena.shape[0] - 1)
        dead_ends = [(x, y) for x in cols for y in rows if (arena[x, y] == 0)
                    and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(0) == 1)]
        crates = [(x, y) for x in cols for y in rows if (arena[x, y] == 1)]
        targets = coins + dead_ends + crates

        # Exclude targets that are currently occupied by a bomb
        targets = [target for target in targets if target not in bomb_xys]

        # Take a step towards the most immediately interesting target
        free_space = arena == 0
        for o in others:
            free_space[o] = False
        d = look_for_targets(free_space, (x, y), targets, self.logger)
        if d == (x, y - 1): action_ideas.append('UP')
        if d == (x, y + 1): action_ideas.append('DOWN')
        if d == (x - 1, y): action_ideas.append('LEFT')
        if d == (x + 1, y): action_ideas.append('RIGHT')
        if d is None:
            self.logger.debug('All targets gone, nothing to do anymore')
            action_ideas.append('WAIT')

        # Add proposal to drop a bomb if at dead end
        if (x, y) in dead_ends:
            action_ideas.append('BOMB')
        # Add proposal to drop a bomb if arrived at target and touching crate
        if d == (x, y) and ([arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]].count(1) > 0):
            action_ideas.append('BOMB')

        # Add proposal to run away from any nearby bomb about to blow
        for (xb, yb), t in bombs:
            if (xb == x) and (abs(yb - y) <= s.BOMB_POWER):
                # Run away
                if (yb > y): action_ideas.append('UP')
                if (yb < y): action_ideas.append('DOWN')
                # If possible, turn a corner
                action_ideas.append('LEFT')
                action_ideas.append('RIGHT')
            if (yb == y) and (abs(xb - x) <= s.BOMB_POWER):
                # Run away
                if (xb > x): action_ideas.append('LEFT')
                if (xb < x): action_ideas.append('RIGHT')
                # If possible, turn a corner
                action_ideas.append('UP')
                action_ideas.append('DOWN')
        # Try random direction if directly on top of a bomb
        for (xb, yb), t in bombs:
            if xb == x and yb == y:
                action_ideas.extend(action_ideas[:4])

        # Pick last action added to the proposals list that is also valid
        while len(action_ideas) > 0:
            a = action_ideas.pop()
            if a in valid_actions:
                self.logger.debug(f"This is a valid returned action {a}")
                return a
        
    # Current game state
    state = state_to_features(game_state=game_state)
    self.logger.info(f"Shape of the state {state}")

    self.logger.debug("Querying model for action.")
    # encoded_action = self.model.action_to_encoded(state)
    if self.train:
        act_values = self.model.model.predict(state) 
    else:
        act_values = self.model.predict(state)
    best_action_index = np.argmax(act_values[0])
    best_action = ACTIONS[best_action_index]
    self.logger.debug(f"value predicted by model {act_values[0]} and best action {best_action}")
    # return np.random.choice(ACTIONS, p=self.model)
    return best_action


# def find_nearest_coin(self, game_state, start):
#     """
#     Find the nearest coin using A* search algorithm.

#     :param game_state: The game state dictionary.
#     :param start: The starting position (agent's position).
#     :return: The position of the nearest coin, or None if no valid path is found.
#     """
#     # Extract the game board and coin positions
#     field = game_state['field']
#     coins = game_state['coins']

#     # Define a function to calculate the Manhattan distance between two points
#     def manhattan_distance(pos1, pos2):
#         return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

#     # Initialize the priority queue for A* search
#     open_list = PriorityQueue()
#     open_list.put((0, start))
#     came_from = {}
#     cost_so_far = {start: 0}

#     while not open_list.empty():
#         current_cost, current_pos = open_list.get()

#         # Check if the current position is a coin
#         if current_pos in coins:
#             return current_pos

#         # Generate neighboring positions
#         neighbors = [(current_pos[0] + 1, current_pos[1]),
#                      (current_pos[0] - 1, current_pos[1]),
#                      (current_pos[0], current_pos[1] + 1),
#                      (current_pos[0], current_pos[1] - 1)]

#         for next_pos in neighbors:
#             x, y = next_pos

#             # Check if the next position is within the game board and not a wall
#             if 0 <= x < field.shape[0] and 0 <= y < field.shape[1] and field[x, y] == 0:
#                 new_cost = cost_so_far[current_pos] + 1

#                 # If the new cost is lower or the position hasn't been visited, update the cost and priority
#                 if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
#                     cost_so_far[next_pos] = new_cost
#                     priority = new_cost + manhattan_distance(next_pos, start)
#                     open_list.put((priority, next_pos))
#                     came_from[next_pos] = current_pos

#     return None

# def state_to_features(game_state: dict) -> np.array:
#     """
#     *This is not a required function, but an idea to structure your code.*

#     Converts the game state to the input of your model, i.e.
#     a feature vector.

#     You can find out about the state of the game environment via game_state,
#     which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
#     what it contains.

#     :param game_state:  A dictionary describing the current game board.
#     :return: np.array
#     """
    
#     # This can be used to send the game state into the training of the model
#     # TODO: Utilize this inside the game_events_occurred for training of the model
    
#     # This is the dict before the game begins and after it ends
#     if game_state is None:
#         return None

#     # For example, you could construct several channels of equal shape, ...
#     channels = []
#     channels.append(game_state)
#     # concatenate them as a feature tensor (they must have the same shape), ...
#     stacked_channels = np.stack(channels)
#     # and return them as a vector
#     return stacked_channels.reshape(-1)


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



def state_to_features(game_state) -> np.array:
    if game_state is None:
        print("First game state is None")
        return np.zeros(2)

    own_position = game_state["self"][-1]

    # Calculate features
    wall_counter = count_walls(own_position, game_state, 3)
    bomb_present = check_bomb_presence(own_position, game_state, 3)
    agent_present = check_agent_presence(own_position, game_state, 3)

    # Calculate feature_id based on features
    features = np.array([int(wall_counter > 2), int(bomb_present), int(agent_present)])
    # feature_id = 2 * features[0] + features[1] + 2 * features[2]
    
    features = features.reshape((1, -1))

    # return feature_id
    return features
