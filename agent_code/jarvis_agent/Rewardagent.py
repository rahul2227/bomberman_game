import events as e

def reward_from_events(self, events) -> int:
    
    reward_game = {
        e.COIN_COLLECTED: 150,
        e.KILLED_OPPONENT: 450,
        e.MOVED_RIGHT: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -1,
        e.INVALID_ACTION: -20,
        e.BOMB_DROPPED: -1,
        e.KILLED_SELF: 0,
        e.GOT_KILLED: -600,
    }

    sum_reward = 0
    for eve in events:
        if eve in reward_game:
            sum_reward += reward_game[eve]
    self.logger.info(f"Got {sum_reward} for events {', '.join(events)}")
    return sum_reward

def own_event_reward(self, events):
    
    sum_reward = 0
    sum_reward += reward_crate(self, events)
    self.logger.info(f"Got {sum_reward} for own events")
    return sum_reward


def reward_crate(self, events):
    
    if e.BOMB_DROPPED in events:
        self.logger.info(f"award for {self.destroyed_crates} that will be destroyed -> +{self.destroyed_crates * 30}")
        return self.destroyed_crates * 30
    return 0



    
    

        



