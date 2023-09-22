import random
import numpy as np
import torch 
import copy



from .statetofeatures import state_to_features
from .Rewardagent import *
    
INDEX_Action = {'LEFT':0, 'RIGHT':1, 'UP':2, 'DOWN':3, 'WAIT':4, 'BOMB':5}

def eps_policy(network, q):
   
    Net = network.training_episodes
    Net1 = int(Net*q)
    Net2 = Net - Net1
    eps_1 = np.linspace(network.epsilon_begin, network.epsilon_end, Net1)
    if Net1 == Net:
        return eps_1
    eps_2 = np.ones(Net2) * network.epsilon_end
    return np.append(eps_1, eps_2)

def experience_add(self, game_state_old, self_action, game_state_new, events, n=5):
    
    features_old = state_to_features(self, game_state_old)
    if features_old is not None:
        if game_state_new is None:
            new_features = features_old
        else:
            new_features = state_to_features(self, game_state_new)
        reward = reward_from_events(self, events)
        reward += own_event_reward(self, events)

        
        index_action = INDEX_Action[self_action]
        action = torch.zeros(6)
        action[index_action] = 1

        
        add_remaining_experience(self, events, reward, new_features, n)

        self.buffer_experience.append((features_old, action, reward, new_features, 0))
        buffer_elements = len(self.buffer_experience)
        if buffer_elements > self.network.buffer_size:
            self.buffer_experience.pop(0)


def add_remaining_experience(self, events, new_reward, new_features, n):
    '''
    updating rewards of last step
    '''
    steps_back = min(len(self.buffer_experience), n)
    for i in range(1, steps_back+1):
        old_old_features, action, reward, old_new_features, Extra_rewards = self.buffer_experience[-i]
        reward += (self.network.gamma**i)*new_reward
        Extra_rewards += 1
        assert Extra_rewards == i
        self.buffer_experience[-i] = (old_old_features, action, reward, new_features, Extra_rewards)


def train_network(self):
    
    network_new = self.network_new  
    old_network = self.network     
    buffer_experience = self.buffer_experience

   
    buffer_elements = len(buffer_experience)
    batch_size = min(buffer_elements, old_network.batch_size)

    random_i = [random.randrange(buffer_elements) for _ in range(batch_size)]

   
    sub_batch = []
    Y = []
    for i in random_i:
        experience_random = buffer_experience[i]
        sub_batch.append(experience_random)
    
    for b in sub_batch:
        features_old = b[0]
        action = b[1]
        reward = b[2]
        new_features = b[3]
        Extra_rewards = b[4]

        y = reward
        if new_features is not None:
            y += old_network.gamma**(Extra_rewards+1) * torch.max(old_network(new_features))

        Y.append(y)

    Y = torch.tensor(Y)

    #Qs
    states = torch.cat(tuple(b[0] for b in sub_batch))  
    q_values = network_new(states)
    actions = torch.cat([b[1].unsqueeze(0) for b in sub_batch])
    Q = torch.sum(q_values*actions, dim=1)
    
    Residuals = torch.abs(Y-Q)
    batch_size = min(len(Residuals), 50)
    _, indices = torch.topk(Residuals, batch_size)

    Y_reduced = Y[indices]
    Q_reduced = Q[indices]

    
    loss = network_new.loss_function(Q_reduced, Y_reduced)
    network_new.optimizer.zero_grad()
    loss.backward()
    network_new.optimizer.step()


def update_network(self):
    
    self.network = copy.deepcopy(self.network_new)




def save_parameters(self, string):
    
    torch.save(self.network.state_dict(), f"network_parameters/{string}.pt")


def get_score(events):
    '''
    track score
    '''
    game_rewards_original = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
    }
    score = 0
    for eve in events:
        if eve in game_rewards_original:
            score += game_rewards_original[eve]
    return score

