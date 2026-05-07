import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class StockMarketEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, df, initial_balance=1000000, h_max=100, transaction_cost_pct=0.001, turbulence_threshold=None):
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
        self.state_dim = 1 + (6 * self.stock_dim) #181 dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
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
        #Constructs the 181-dimensional state vector for the current time step
        current_data = self.df[self.df['date'] == self.dates[self.current_step]]
        
        prices = current_data['close'].values
        macd = current_data['macd'].values
        rsi = current_data['rsi'].values
        cci = current_data['cci'].values
        adx = current_data['adx'].values

        state = np.hstack([
            self.balance, 
            prices, 
            self.shares, 
            macd, 
            rsi, 
            cci, 
            adx
        ])
        return np.float32(state)

    def step(self, actions):
        """Executes the action, updates state, and calculates reward."""
        self.terminal = self.current_step >= len(self.dates) - 1
        
        if self.terminal:
            return self.state, 0, self.terminal, {}
            
        current_data = self.df[self.df['date'] == self.dates[self.current_step]]
        prices = current_data['close'].values
        turbulence = current_data['turbulence'].values[0] if 'turbulence' in current_data.columns else 0
        

        # market crash logic
        if self.turbulence_threshold is not None and turbulence > self.turbulence_threshold:
            actions = np.where(self.shares > 0, -1, 0) 




        # Un-normalize actions from [-1, 1] to [-h_max, h_max]
        actions = actions * self.h_max 
        actions = actions.astype(int)
        
        # Record portfolio value before taking actions (for reward calculation)
        portfolio_value_t = self.balance + np.sum(prices * self.shares)
        
        # Process actions: Separate into Sells (to free up cash first) and Buys
        argsort_actions = np.argsort(actions)
        sell_indices = argsort_actions[:np.where(actions < 0)[0].shape[0]]
        buy_indices = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]



        #sell
        for index in sell_indices:
            action = actions[index]
            sell_amount = min(abs(action), self.shares[index])
            if sell_amount > 0:
                self.shares[index] -= sell_amount
                sale_revenue = prices[index] * sell_amount
                transaction_cost = sale_revenue * self.transaction_cost_pct
                self.balance += (sale_revenue - transaction_cost)
                
        #buy
        for index in buy_indices:
            action = actions[index]
            cost_per_share = prices[index] * (1 + self.transaction_cost_pct)
            buy_amount = min(action, self.balance // cost_per_share)
            
            if buy_amount > 0:
                self.shares[index] += buy_amount
                purchase_cost = prices[index] * buy_amount
                transaction_cost = purchase_cost * self.transaction_cost_pct
                self.balance -= (purchase_cost + transaction_cost)

        #next time step
        self.current_step += 1
        

        #reward
        next_data = self.df[self.df['date'] == self.dates[self.current_step]]
        next_prices = next_data['close'].values
        portfolio_value_t_plus_1 = self.balance + np.sum(next_prices * self.shares)
        reward = portfolio_value_t_plus_1 - portfolio_value_t
        self.portfolio_value = portfolio_value_t_plus_1
        self.state = self._get_observation()
        return self.state, reward, self.terminal, {}

    def render(self, mode='human'):
        """Displays current step information."""
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance:.2f}")
        print(f"Portfolio Value: {self.portfolio_value:.2f}")
        print("-" * 30)




if __name__ == "__main__":
    # Mocking a DataFrame based on the paper's required state features
    # Required columns: date, close, macd, rsi, cci, adx, turbulence
    np.random.seed(42)
    dates = pd.date_range(start="2016-01-01", periods=100)
    mock_data = []
    
    for date in dates:
        for stock_id in range(30):
            mock_data.append([
                date, stock_id, 
                np.random.uniform(10, 500), # close
                np.random.uniform(-5, 5),   # macd
                np.random.uniform(0, 100),  # rsi
                np.random.uniform(-200, 200),# cci
                np.random.uniform(0, 100),  # adx
                np.random.uniform(0, 50)    # turbulence
            ])
            
    df = pd.DataFrame(mock_data, columns=['date', 'stock_id', 'close', 'macd', 'rsi', 'cci', 'adx', 'turbulence'])
    
    # Initialize the Environment
    env = StockMarketEnv(df=df, turbulence_threshold=150)
    state = env.reset()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


# CONTINUOUS A2C
class ContinuousA2C(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ContinuousA2C, self).__init__()
    
        self.state_norm = nn.LayerNorm(state_dim)

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Tanh activation function because it mathematically forces 
        # the output to stay strictly between -1.0 and 1.0.
        self.actor_mean = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Tanh() 
        )

        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        state = self.state_norm(state)
        features = self.shared(state)

        action_mean = self.actor_mean(features)
        action_std = self.actor_log_std.exp().expand_as(action_mean)

        state_value = self.critic(features)

        return action_mean, action_std, state_value

# Portfolio Trading Loop
if __name__ == "__main__":
    print("Initializing 30-Stock Portfolio Environment...")
    
    np.random.seed(42)
    dates = pd.date_range(start="2016-01-01", periods=100)
    mock_data = []
    for date in dates:
        for stock_id in range(30):
            mock_data.append([
                date, stock_id, 
                np.random.uniform(10, 500), np.random.uniform(-5, 5), 
                np.random.uniform(0, 100), np.random.uniform(-200, 200),
                np.random.uniform(0, 100), np.random.uniform(0, 50) 
            ])
    df = pd.DataFrame(mock_data, columns=['date', 'stock_id', 'close', 'macd', 'rsi', 'cci', 'adx', 'turbulence'])
    
    # Create the environment
    env = StockMarketEnv(df=df, turbulence_threshold=150)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]    
    
    model = ContinuousA2C(state_dim, action_dim)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    
    episodes = 500
    gamma = 0.99
    entropy_beta = 0.01

    print("Starting Continuous A2C Training...")

    for episode in range(1, episodes + 1):
        state = env.reset()
        ep_return = 0
        done = False

        model.train() 

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # 1. Forward Pass
            action_mean, action_std, state_value = model(state_tensor)
            
            # 2. Build the 30 Bell Curves and sample 30 actions
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            
            # Math mathematically clip actions to [-1, 1] just in case a wild sample escapes
            action_clipped = torch.clamp(action, -1.0, 1.0)
            
            # Step the environment (passing the 30 actions as a flat numpy array)
            next_state, reward, done, _ = env.step(action_clipped.squeeze(0).numpy())
            
            scaled_reward = reward / 10000.0 
            
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
            _, _, next_state_value = model(next_state_tensor)
            
            # 3. Advantage Math
            td_target = scaled_reward + gamma * next_state_value.detach() * (1 - int(done))
            advantage = td_target - state_value
            
            # 4. Losses
            critic_loss = F.mse_loss(state_value, td_target.detach())
            
            # For continuous, log_prob is the sum of the log probs of all 30 actions
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            
            actor_loss = -(log_prob * advantage.detach()).mean() - (entropy_beta * entropy.mean())
            total_loss = actor_loss + critic_loss
            
            # 5. Backprop
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) 
            optimizer.step()
            
            state = next_state
            ep_return += reward

        if episode % 10 == 0:
            true_portfolio_return = ((env.portfolio_value - env.initial_balance) / env.initial_balance) * 100
            print(f"Episode {episode}: Actor: {actor_loss.item():.4f} | Critic: {critic_loss.item():.4f} | TRUE PROFIT: {true_portfolio_return:.2f}%")