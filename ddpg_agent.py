import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    shared_actor_local = None
    shared_actor_target = None
    shared_actor_optimizer = None
    
    shared_critic_local = None
    shared_critic_target = None
    shared_critic_optimizer = None

    shared_memory = None

    shared_learn_per_step = 0
    shared_update_times = 0
    def __init__(self, action_size):
        """Initialize an Agent object.
        
        Params
        ======
            action_size (int): dimension of each action
        """
        # Actor Network (w/ Target Network)
        self.actor_local = Agent.shared_actor_local
        self.actor_target = Agent.shared_actor_target
        self.actor_optimizer = Agent.shared_actor_optimizer

        # Critic Network (w/ Target Network)
        self.critic_local = Agent.shared_critic_local
        self.critic_target = Agent.shared_critic_target
        self.critic_optimizer = Agent.shared_critic_optimizer

        # Noise process
        self.noise = OUNoise(action_size)

        # Replay memory
        self.memory = Agent.shared_memory
    
    @staticmethod
    def set_hparams(state_size, action_size, hparams):
        Agent.hparams = hparams

        Agent.batch_size = hparams['batch_size']
        Agent.tau = hparams['tau']
        Agent.gamma = hparams['gamma']
        Agent.shared_learn_per_step = hparams['learn_per_step']
        Agent.shared_update_times = hparams['update_times']

        lr = hparams['lr']
        actor_hidden = hparams['hidden_layers']['actor']

        Agent.shared_actor_local = Actor(state_size, action_size, actor_hidden).to(device)
        Agent.shared_actor_target = Actor(state_size, action_size, actor_hidden).to(device)
        Agent.shared_actor_optimizer = optim.Adam(Agent.shared_actor_local.parameters(), lr=lr['actor'])

        critic_hidden = hparams['hidden_layers']['critic']
        Agent.shared_critic_local = Critic(state_size, action_size, critic_hidden).to(device)
        Agent.shared_critic_target = Critic(state_size, action_size, critic_hidden).to(device)
        Agent.shared_critic_optimizer = optim.Adam(Agent.shared_critic_local.parameters(), lr=lr['critic'], weight_decay=hparams['weight_decay'])
        
        Agent.shared_memory = ReplayBuffer(action_size, hparams['buffer_size'], Agent.batch_size)


    @staticmethod
    def save(prefix):
        torch.save(Agent.shared_actor_local.state_dict(), f'{prefix}_actor.pth')
        torch.save(Agent.shared_critic_local.state_dict(), f'{prefix}_critic.pth')

    @staticmethod
    def load(prefix):
        Agent.shared_actor_local.load_state_dict(torch.load(f'{prefix}_actor.pth'))

    def step(self, t, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        if t % Agent.shared_learn_per_step != 0:
            return
        # Learn, if enough samples are available in memory
        if len(self.memory) > Agent.batch_size:
            for _ in range(Agent.shared_update_times):
                experiences = self.memory.sample()
                self.learn(experiences, Agent.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)                     

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(Agent.tau*local_param.data + (1.0-Agent.tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * (np.random.standard_normal(size=x.shape))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)