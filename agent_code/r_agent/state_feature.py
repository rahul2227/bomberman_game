from typing import List, Tuple
import numpy as np

# from agent_code.r_agent.callbacks import ACTIONS


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

# Feature 4: Move makes sense
# def moves_making_sense(game_state):
#     # Gathering information about the game state
#     arena = game_state['field']
#     _, score, bombs_left, (x, y) = game_state['self']
#     bombs = game_state['bombs']
#     bomb_xys = [xy for (xy, t) in bombs]
#     others = [xy for (n, s, b, xy) in game_state['others']]
#     coins = game_state['coins']
#     bomb_map = np.ones(arena.shape) * 5
#     for (xb, yb), t in bombs:
#         for (i, j) in [(xb + h, yb) for h in range(-3, 4)] + [(xb, yb + h) for h in range(-3, 4)]:
#             if (0 < i < bomb_map.shape[0]) and (0 < j < bomb_map.shape[1]):
#                 bomb_map[i, j] = min(bomb_map[i, j], t)

#     # Check which moves make sense at all
#     directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
#     valid_tiles, valid_actions = [], []
#     for d in directions:
#         if ((arena[d] == 0) and
#                 (game_state['explosion_map'][d] < 1) and
#                 (bomb_map[d] > 0) and
#                 (not d in others) and
#                 (not d in bomb_xys)):
#             valid_tiles.append(d)
#     if (x - 1, y) in valid_tiles: valid_actions.append('LEFT')
#     if (x + 1, y) in valid_tiles: valid_actions.append('RIGHT')
#     if (x, y - 1) in valid_tiles: valid_actions.append('UP')
#     if (x, y + 1) in valid_tiles: valid_actions.append('DOWN')
#     if (x, y) in valid_tiles: valid_actions.append('WAIT')
    
#     # Create a mapping from action names to integers
#     action_to_index = {action: index for index, action in enumerate(ACTIONS)}

#     # Initialize the valids array with zeros
#     valids = [0] * len(ACTIONS)

#     # Set 1 for valid actions
#     for action in valid_actions:
#         index = action_to_index.get(action, -1)  # Get the index for the action
#         if index != -1:
#             valids[index] = 1
    
#     return valids


# def state_to_features(game_state) -> np.array:
#     if game_state is None:
#         print("First game state is None")
#         return np.zeros(9)

#     own_position = game_state["self"][-1]

#     # Calculate features
#     wall_counter = count_walls(own_position, game_state, 3)
#     bomb_present = check_bomb_presence(own_position, game_state, 3)
#     agent_present = check_agent_presence(own_position, game_state, 3)
#     action_features = moves_making_sense(game_state=game_state) 
#     # print(action_features)
    
    
#     # Ensure action_features has a constant size of ACTION_FEATURES_SIZE
#     # if len(action_features) < ACTION_FEATURES_SIZE:
#     #     action_features.extend([-1] * (ACTION_FEATURES_SIZE - len(action_features)))
#     # elif len(action_features) > ACTION_FEATURES_SIZE:
#     #     action_features = action_features[:ACTION_FEATURES_SIZE]
    
#     # Calculate features
#     features = np.array([int(wall_counter > 2), int(bomb_present), int(agent_present)])
    
#     # Concatenate the action_features
#     features = np.concatenate((features, action_features))
#     # print(features)
    
#     # Reshape to (1, 9)
#     features = features.reshape((1, -1))
#     # print(features)

    
#     return features

# Implementation of new features
def state_to_features(game_state: dict) -> np.array:
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    
    channels = []

    # chanels consist of: 
    #       - walls
    #       - crates
    #       - coins
    #       - players
    #       - bombs

    # create walls channel
    wall_map = np.zeros_like(game_state["field"])
    for x, y in zip(*np.where(game_state["field"] == -1)):
        wall_map[x][y] = 1
    channels.append(wall_map)

    # create crates channel
    crate_map = np.zeros_like(wall_map)
    for x, y in zip(*np.where(game_state["field"] == 1)):
        crate_map[x][y] = 1
    channels.append(crate_map)    

    # create coins channel
    coin_map = np.zeros_like(wall_map)
    for x, y in game_state["coins"]:
        coin_map[x][y] = 1
    channels.append(coin_map)

    # create bomb channel
    bomb_map = np.zeros_like(wall_map)
    for x, y in zip(*np.where(game_state["explosion_map"] == 1)):
        bomb_map[x][y] = 1
        # bomb basically is a wall -> player can't move past it
        wall_map[x][y] = 1

    for (x, y), countdown in game_state["bombs"]:
        bomb_map[x][y] = 1
        for i in range(1, 4):
            if x+i < len(bomb_map[x]) - 1:
                bomb_map[x+i][y] = 1
            if x-i > 0:
                bomb_map[x-i][y] = 1
            if y+i < len(bomb_map[:,y]) - 1:
                bomb_map[x][y+i] = 1
            if y-i > 0:
                bomb_map[x][y-i] = 1

    channels.append(bomb_map)

    # create player channel
    player_map = np.zeros_like(wall_map)
    _, _, _, (x, y) = game_state["self"]
    player_map[x][y] = 1
    for _, _, _, (x, y) in game_state["others"]:
        player_map[x][y] = -1
    channels.append(player_map)

    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels).astype(float)
    # Add an extra dimension for the batch size
    # stacked_channels = np.expand_dims(stacked_channels, axis=0)
    
    return stacked_channels