import numpy as np
import random
from collections import namedtuple, deque

from actor import Actor
from critic import Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

import copy

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, device, num_agents, state_size, action_size, seed, 
        BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int):       dimension of each state
            action_size (int):      dimension of each action
            seed (int):             random seed
            BUFFER_SIZE (int):      replay buffer size
            BATCH_SIZE (int):       minibatch size
            GAMMA (float):          discount factor
            TAU (float):            for soft update of target parameters
            LR_ACTOR (float):       learning rate of the actor
            LR_CRITIC (float):      learning rate of the critic
        """
        self.device = device
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.BUFFER_SIZE = BUFFER_SIZE
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.TAU = TAU
        self.LR_ACTOR = LR_ACTOR
        self.LR_CRITIC = LR_CRITIC

        # Actor-Network
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic-Network
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Replay memory
        self.memory = ReplayBuffer(device, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
        
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, self.GAMMA)

    def act(self, states):
        """Returns actions for given states as per current policy.
        
        Params
        ======
            states (array_like): current states
        """
        states = torch.from_numpy(states).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        return actions

    def learn(self, experiences, gamma):
        """Update critic and actor using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## update critic network
        next_actions = self.actor_target.forward(next_states)
        q_target = self.critic_target.forward(next_states, next_actions)
        q_target = (1 - dones) * q_target * gamma + rewards
        q_local = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_local, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        ## update actor network
        actions_pred = self.actor_local.forward(states)
        J = -self.critic_local.forward(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        J.backward()
        self.actor_optimizer.step()

        # ------------------- update target networks ------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)                 


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, device, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            device: cpu or gpu
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck noise process"""
    def __init__(self, size, seed, mu=0.0, theta=0.2, sigma=0.1):
        """Initialize parameters and noise process
        Parameters:
        ======
            size (tuple): size of noise
            seed (int): random seed        
        """
        self.size = size
        self.seed = random.seed(seed)
        self.mu = mu * np.ones(self.size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        ds = self.theta * (self.mu - self.state) + self.sigma * np.random.standard_normal(self.size)
        self.state += ds
        return self.state