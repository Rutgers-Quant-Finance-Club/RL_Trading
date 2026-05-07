import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class StockMarketEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance=1000000, h_max=100,
                 transaction_cost_pct=0.001, turbulence_threshold=None):
        super(StockMarketEnv, self).__init__()
        self.df = df
        self.dates = self.df['date'].unique()
        self.current_step = 0
        self.stock_dim = 30
        self.initial_balance = initial_balance
        self.h_max = h_max
        self.transaction_cost_pct = transaction_cost_pct
        self.turbulence_threshold = turbulence_threshold

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32)
        self.state_dim = 1 + (6 * self.stock_dim)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(self.state_dim,), dtype=np.float32)
        self.balance = self.initial_balance
        self.shares = np.zeros(self.stock_dim)
        self.portfolio_value = self.initial_balance
        self.state = None

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares = np.zeros(self.stock_dim)
        self.portfolio_value = self.initial_balance
        self.state = self._get_observation()
        return self.state

    def _get_observation(self):
        current_data = self.df[self.df['date'] == self.dates[self.current_step]]
        prices = current_data['close'].values
        macd = current_data['macd'].values
        rsi = current_data['rsi'].values
        cci = current_data['cci'].values
        adx = current_data['adx'].values
        state = np.hstack([self.balance, prices, self.shares, macd, rsi, cci, adx])
        return np.float32(state)

    def step(self, actions):
        self.terminal = self.current_step >= len(self.dates) - 1
        if self.terminal:
            return self.state, 0, self.terminal, {}

        current_data = self.df[self.df['date'] == self.dates[self.current_step]]
        prices = current_data['close'].values
        turbulence = current_data['turbulence'].values[0] if 'turbulence' in current_data.columns else 0

        if self.turbulence_threshold is not None and turbulence > self.turbulence_threshold:
            actions = np.where(self.shares > 0, -1, 0)

        actions = (actions * self.h_max).astype(int)

        portfolio_value_t = self.balance + np.sum(prices * self.shares)

        argsort_actions = np.argsort(actions)
        sell_indices = argsort_actions[:np.where(actions < 0)[0].shape[0]]
        buy_indices = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

        for index in sell_indices:
            sell_amount = min(abs(actions[index]), self.shares[index])
            if sell_amount > 0:
                self.shares[index] -= sell_amount
                sale_revenue = prices[index] * sell_amount
                self.balance += sale_revenue * (1 - self.transaction_cost_pct)

        for index in buy_indices:
            cost_per_share = prices[index] * (1 + self.transaction_cost_pct)
            buy_amount = min(actions[index], int(self.balance // cost_per_share))
            if buy_amount > 0:
                self.shares[index] += buy_amount
                self.balance -= prices[index] * buy_amount * (1 + self.transaction_cost_pct)

        self.current_step += 1
        next_data = self.df[self.df['date'] == self.dates[self.current_step]]
        next_prices = next_data['close'].values
        portfolio_value_t_plus_1 = self.balance + np.sum(next_prices * self.shares)

        reward = portfolio_value_t_plus_1 - portfolio_value_t
        self.portfolio_value = portfolio_value_t_plus_1
        self.state = self._get_observation()
        return self.state, reward, self.terminal, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance:.2f}")
        print(f"Portfolio Value: {self.portfolio_value:.2f}")
        print("-" * 30)


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.norm = nn.LayerNorm(state_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim), nn.Tanh()
        )

    def forward(self, state):
        return self.net(self.norm(state))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.norm = nn.LayerNorm(state_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([self.norm(state), action], dim=1))


class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state


class DDPGAgent:
    def __init__(self, state_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, tau=0.005, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer()
        self.noise = OUNoise(action_dim)

    def select_action(self, state, explore=True):
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()
        self.actor.train()
        if explore:
            action += self.noise.sample()
        return np.clip(action, -1.0, 1.0)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = rewards + self.gamma * self.critic_target(next_states, next_actions) * (1 - dones)

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return actor_loss.item(), critic_loss.item()


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    dates = pd.date_range(start="2016-01-01", periods=200)
    mock_data = []
    base_prices = np.random.uniform(50, 300, 30)
    for i, date in enumerate(dates):
        for stock_id in range(30):
            price = base_prices[stock_id] * (1 + 0.001 * i + np.random.randn() * 0.02)
            mock_data.append([
                date, stock_id,
                max(1.0, price),
                np.random.uniform(-5, 5),
                np.random.uniform(20, 80),
                np.random.uniform(-100, 100),
                np.random.uniform(10, 50),
                np.random.uniform(0, 30)
            ])

    df = pd.DataFrame(mock_data, columns=['date', 'stock_id', 'close', 'macd', 'rsi', 'cci', 'adx', 'turbulence'])

    env = StockMarketEnv(df=df, turbulence_threshold=100)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = DDPGAgent(state_dim, action_dim)

    episodes = 200
    print(f"DDPG Trading Agent | state_dim={state_dim} | action_dim={action_dim}")
    print("=" * 60)

    for episode in range(1, episodes + 1):
        state = env.reset()
        agent.noise.reset()
        ep_reward = 0
        done = False
        actor_l, critic_l = None, None

        while not done:
            action = agent.select_action(state, explore=True)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward / 10000.0, next_state, done)
            result = agent.update()
            if result[0] is not None:
                actor_l, critic_l = result
            state = next_state
            ep_reward += reward

        profit_pct = (env.portfolio_value - env.initial_balance) / env.initial_balance * 100

        if episode % 10 == 0:
            a_str = f"{actor_l:.4f}" if actor_l is not None else "—"
            c_str = f"{critic_l:.4f}" if critic_l is not None else "—"
            print(f"Ep {episode:4d} | Actor: {a_str} | Critic: {c_str} | "
                  f"Portfolio: ${env.portfolio_value:,.0f} | Profit: {profit_pct:+.2f}% | "
                  f"Buffer: {len(agent.replay_buffer)}")

    print("=" * 60)
    print("Training complete.")
