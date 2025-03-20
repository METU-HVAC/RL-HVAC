import random
from collections import namedtuple, deque
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

import numpy as np

class PriorityReplayMemory(object):

    def __init__(self, capacity, beta=2.0):
        self.capacity = capacity
        self.memory = np.empty(capacity, dtype=object)  # Fixed-size array
        self.priorities = np.zeros(capacity, dtype=np.float64)  # Fixed-size array
        self.beta = beta
        self.index = 0  # Index counter for circular buffer
        self.size = 0  # Current size of the buffer

    def push(self, *args):
        """Save a transition with priority based on push time"""
        self.memory[self.index] = Transition(*args)
        self.priorities[self.index] = self.index  # Priority based on index
        self.index = (self.index + 1) % self.capacity  # Circular buffer index
        self.size = min(self.size + 1, self.capacity)  # Track the current size

    def sample(self, batch_size):
        """Sample transitions with time-based priorities"""
        if self.size == 0:
            return [], []

        priorities = self.priorities[:self.size] #Only use the populated priorities
        probabilities = priorities ** self.beta
        probabilities /= probabilities.sum()

        indices = np.random.choice(self.size, batch_size, p=probabilities)
        samples = [self.memory[i] for i in indices]
        return samples

    def __len__(self):
        return self.size
