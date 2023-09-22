import torch
import numpy as np
from collections import deque
import heapq
import random
import bisect


STEP = np.array([[1,0], [-1,0], [0,1], [0,-1]])

DIRECTION = {(1,0):0, (-1,0):1, (0,1):2, (0,-1):3}

MOVE = ["right", "left", "down", "up"]

ACTION = ['RIGHT', 'LEFT', 'DOWN','UP',  'WAIT', 'BOMB']
INVERSE_ACTION = {"RIGHT": 0, "LEFT": 1, "DOWN": 2, "UP": 3, "WAIT": 4, "BOMB": 5}

DISTANCE_MAX = 15 #Search distance of agent

#The number of crates is limit to number 4 
CRATE_POSITION_MAX = 4

#Dead ends are usually local and immediate obstacles
MAX_DEAD_END = 2

MAX_CRATES = MAX_DEAD_END+CRATE_POSITION_MAX

def state_to_features(self, game_state: dict) -> torch.tensor:

    coins_max = 10
    crates_max =200


    # at the beginning and the end:
    if game_state is None:
        return None


    def nearest_enemy(pos):
        
        result = []
        for new_pos in (pos + STEP):
            if field[new_pos[0], new_pos[1]] == 0:
                result.append(new_pos.tolist())
        return result


    def effect_bomb(pos):
        
        destroyed_crates = 0
        for direction in STEP:
            for length in range(1, 4):
                beam = direction*length + pos
                obj = field[beam[0], beam[1]]
                if obj == -1:
                    break
                if (obj == 1) and future_explosion_map[beam[0], beam[1]]==1: 
                    destroyed_crates += 1
        return destroyed_crates


    def map_explosion_fill(explosions, bombs, field):
        
        future_explosion_map = (np.copy(explosions)*-4) + 1 
        for bomb in bombs:
            pos = np.array(bomb[0])
            timer = bomb[1] - 3 
            field[pos[0], pos[1]] = -2 

            for direction in STEP:
                for length in range(0, 4):
                    beam = direction*length + pos
                    obj = field[beam[0], beam[1]]
                    if obj == -1:
                        break
                    if future_explosion_map[beam[0], beam[1]] > timer:
                        future_explosion_map[beam[0], beam[1]] = timer

        return future_explosion_map


    def create_new_future_explosion_map(future_explosion_map, pos):
        
        new_future_explosion_map = np.copy(future_explosion_map)
        
        new_future_explosion_map[new_future_explosion_map < 1] -= 1
        new_future_explosion_map[new_future_explosion_map < -3] = 1
        timer = 0

        for direction in STEP:
            for length in range(0, 4):
                beam = direction*length + pos
                obj = field[beam[0], beam[1]]
                if obj == -1:
                    break
                if new_future_explosion_map[beam[0], beam[1]] > timer:
                    new_future_explosion_map[beam[0], beam[1]] = timer

        return new_future_explosion_map


    def probable_death(pos, future_explosion_map, turns=0, forbidden_fields = None):
        
        q = deque()

        visited = []

        if forbidden_fields is not None:
            for forbidden_pos in forbidden_fields:
                visited.append(forbidden_pos)

        q.append((pos.tolist(), turns))
        while len(q):
            pos, turns = q.popleft()

            
            if turns > 4:
                break

            if pos in visited:
                continue

            
            if turns-1 - future_explosion_map[pos[0], pos[1]] == 3:
                continue
            
            
            if future_explosion_map[pos[0], pos[1]] == 1:
                return False

            visited.append(pos)
            for neighbor in nearest_enemy(pos):
                q.append((neighbor, turns+1))
                
        return True

    #saving jarvis position
    player_pos = np.array(game_state["self"][3])

    # saving known positions of coins
    coins = np.array(game_state["coins"])
    coins_list = coins.tolist()
    number_of_coins = len(coins)

    # saving crate positions
    field = np.array(game_state["field"])
    explosions = np.array(game_state["explosion_map"])
    bombs = game_state["bombs"]

    crates = np.argwhere(field==1)
    number_of_crates = len(crates)
    future_explosion_map = map_explosion_fill(explosions, bombs, field)

    # saving other agent position
    others = []
    for agent in game_state["others"]:
        pos = agent[3]
        others.append(pos)

        field[pos[0], pos[1]] == 2 

    others = np.array(others)
    others_list = others.tolist()

   
    if number_of_coins == 0:
        coins = np.zeros((coins_max, 2))

   
    possible_next_pos = nearest_enemy(player_pos)

    
    inv_coins = [[] for _ in range(4)]
    inv_crate_distances = [[] for _ in range(4)]
    crate_points = [[] for _ in range(4)]
    inv_opponents = [[] for _ in range(4)]

    
    coin_distances_after_step = np.empty((4, coins_max))
    crate_distances_after_step = np.empty((4, MAX_CRATES))
    opponent_distances_after_step = np.empty((4, 4))  # MAX_OPPONENTS = 4

   
    expected_destructions_after_step = np.zeros((4, MAX_CRATES))

    coin_distances_after_step.fill(np.inf)
    crate_distances_after_step.fill(np.inf)
    opponent_distances_after_step.fill(np.inf)

    
    visited = [player_pos.tolist()]

    # queue for breadth first search
    q = []
    for pos in (player_pos + STEP):
        pos = pos.tolist() 
        
        x = pos[0] - player_pos[0]
        y = pos[1] - player_pos[1]
        heapq.heappush(q, (1, pos, DIRECTION[(x,y)]))

   
    number_of_found_crate_positions = np.zeros(4)
    number_of_found_dead_ends = np.zeros(4)
    number_of_found_coins = np.zeros(4)
    number_of_found_opponents = np.zeros(4)

    
    found_one = False
    skipped = [False, False, False, False]

    #  performing the dijkstra algorithm
    while len(q) != 0:
        
       
        distance, pos, direction = heapq.heappop(q)

        
        if (distance > DISTANCE_MAX) and (found_one==True):
            break
        
        
        if pos in visited:
            continue

        
        visited.append(pos)

       
        if distance == 1:
            
            if future_explosion_map[pos[0], pos[1]]==-2:
                crate_points[direction] = np.zeros(MAX_CRATES)
                placebo1 = np.zeros(MAX_CRATES)
                placebo1.fill(-2)
                placebo2 = np.zeros(coins_max)
                placebo2.fill(-2)
                placebo3 = np.zeros(4)
                placebo3.fill(-2)
                inv_crate_distances[direction] = np.copy(placebo1)
                inv_coins[direction] = np.copy(placebo2)
                inv_opponents[direction] = np.copy(placebo3)

                skipped[direction] = True
                continue

            if pos not in possible_next_pos:
                # To know whether we are going towards wall or crate
                crate_points[direction] = np.zeros(MAX_CRATES)
                placebo1 = np.zeros(MAX_CRATES)
                placebo1.fill(-1)
                placebo2 = np.zeros(coins_max)
                placebo2.fill(-1)
                placebo3 = np.zeros(4)
                placebo3.fill(-1)
                inv_crate_distances[direction] = np.copy(placebo1)
                inv_coins[direction] = np.copy(placebo2)
                inv_opponents[direction] = np.copy(placebo3)

                skipped[direction] = True
                continue


       
        is_coin = pos in coins_list 
        if is_coin:
            coin_distances_after_step[direction][int(number_of_found_coins[direction])] = distance
            number_of_found_coins[direction] += 1
        if is_coin and not found_one:
            found_one = True


       
        if (number_of_found_opponents[direction] < 4):
            for possible_opponent in (pos + STEP):
                if (possible_opponent.tolist() in others_list):
                    
                    index_opponents = int(number_of_found_opponents[direction])
                    opponent_distances_after_step[direction][index_opponents] = distance

                    number_of_found_opponents[direction] += 1
                    
                    break


        neighbors = nearest_enemy(pos)

       
        ways_out = 0
        for node in neighbors:
            ways_out += 1
            if (distance+1)<=3 and (future_explosion_map[node[0], node[1]] != 1):
               
                heapq.heappush(q, (distance+1, node, direction))
            heapq.heappush(q, (distance+1, node, direction))

        # crates
        if future_explosion_map[pos[0], pos[1]] != 1: 
            continue

        dead_end = False
        if (ways_out == 1) and (number_of_found_dead_ends[direction] < MAX_DEAD_END):
            
            index_crates = int(number_of_found_crate_positions[direction] + number_of_found_dead_ends[direction])
            crate_distances_after_step[direction][index_crates] = distance
            expected_destructions_after_step[direction][index_crates] = effect_bomb(pos)

            dead_end = True
            number_of_found_dead_ends[direction] += 1
            found_one = True

        
        if (number_of_found_crate_positions[direction] < CRATE_POSITION_MAX) and not dead_end:
            for possible_crate in (pos + STEP):
                if field[possible_crate[0], possible_crate[1]] == 1 and (future_explosion_map[possible_crate[0], possible_crate[1]]==1):
                    
                    index_crates = int(number_of_found_crate_positions[direction] + number_of_found_dead_ends[direction])
                    crate_distances_after_step[direction][index_crates] = distance
                    expected_destructions_after_step[direction][index_crates] = effect_bomb(pos)

                    number_of_found_crate_positions[direction] += 1
                    found_one = True
                    break

    for direction in range(4):
        if skipped[direction]:
            continue
        
        inv_coins[direction] = 1/np.array(coin_distances_after_step[direction])

        
        inv_opponents[direction] = 1/np.array(opponent_distances_after_step[direction])

        
        inv_crate_distances[direction] = 1/np.array(crate_distances_after_step[direction])

       
        crate_points[direction] = np.array(expected_destructions_after_step[direction])

    inv_crate_distances = np.array(inv_crate_distances)

    crate_points = np.array(crate_points)

    inv_coins = np.array(inv_coins)

    inv_opponents = np.array(inv_opponents)

    #Now collecting feature in feature array
    features = []
    # coins feature
    features = np.append(features, np.max(inv_coins, axis=1))

    # crates features
    features = np.append(features, np.max(inv_crate_distances * crate_points, axis=1))

    
    neighboring_chest = False
    neighboring_opponent = False
    if future_explosion_map[player_pos[0], player_pos[1]] == 1:
        for pos in player_pos + STEP:
            if (field[pos[0], pos[1]] == 1) and (future_explosion_map[pos[0], pos[1]] == 1): # free crate
                neighboring_chest = True
            if pos.tolist() in others_list: # oppontent
                neighboring_opponent = True

    # opponent and crate points
    if neighboring_opponent:
        bomb_here = 5 + effect_bomb(player_pos)

    # crate points only
    elif neighboring_chest:
        bomb_here = effect_bomb(player_pos)

    # dropping bomb
    if not neighboring_chest and not neighboring_opponent:
        bomb_here = -1
    
    
    if not game_state["self"][2]:
        bomb_here = -1

    new_future_explosion_map = create_new_future_explosion_map(future_explosion_map, player_pos)

    
    if probable_death(player_pos, new_future_explosion_map):
        
        bomb_here = -1
    
    features = np.append(features, bomb_here)
    
    
    features = np.append(features,-(future_explosion_map[player_pos[0], player_pos[1]]-1))

    # fled away feature
    running = np.zeros(4)
    
    if future_explosion_map[player_pos[0], player_pos[1]] == 1:
        features = np.append(features, running)
    
    else:
        direction = -1
        for pos in player_pos+STEP:
            direction += 1
            if probable_death(pos, future_explosion_map, turns=1, forbidden_fields=[player_pos.tolist()]):
                running[direction] = -1
            elif ((direction == np.argmax(np.max(inv_opponents, axis = 1)))
                and ((np.max(inv_opponents, axis = 1)!=0).all()) 
                and ((1/np.max(inv_opponents, axis = 1)) < 5).any()):
                running[direction] = -0.5
            if field[pos[0], pos[1]] != 0:
                running[direction] = -1

        features = np.append(features, running)

    # wait feature to avoid getting blown away
    danger = np.zeros(4)
    if future_explosion_map[player_pos[0], player_pos[1]] == 1: 
        dim = 0
        for pos in player_pos + STEP:
            if future_explosion_map[pos[0], pos[1]] == -3:
                danger[dim] = -1
            dim += 1
    features = np.append(features, danger)

    # other agent feature
    features = np.append(features, np.max(inv_opponents, axis=1))

    # remaining reward
    features = np.append(features, number_of_coins + 1/3 * number_of_crates)
    
    # rulebased
    self.features = features

    # crate features
    self.destroyed_crates = self.bomb_buffer
    self.bomb_buffer = effect_bomb(player_pos)


    # torch tensor to returne from the features
    features = torch.from_numpy(features).float()

    return features.unsqueeze(0)