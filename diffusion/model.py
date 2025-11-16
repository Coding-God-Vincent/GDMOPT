import torch
import torch.nn as nn
from .helpers import SinusoidalPosEmb
from tianshou.data import Batch, ReplayBuffer, to_torch

#------------------------------------------------------------------------------------------------------------------#
# With Expert data & Without 的 GDM 都用這個
#------------------------------------------------------------------------------------------------------------------#
class MLP(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        t_dim=16,
        activation='mish'
    ):
        super(MLP, self).__init__()
        _act = nn.Mish if activation == 'mish' else nn.ReLU
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            _act(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.mid_layer = nn.Sequential(
            nn.Linear(hidden_dim + action_dim + t_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    # x : x_t。shape (batch_size, action_dim)
    # time : t of each data。shape (batch_size)
    # state : shape (batch_size, state_dim)
    def forward(self, x, time, state):
        processed_state = self.state_mlp(state)
        t = self.time_mlp(time)
        x = torch.cat([x, t, processed_state], dim=1)
        x = self.mid_layer(x)
        # x = self.final_layer(x)
        return x

#------------------------------------------------------------------------------------------------------------------#
# without Expert data 實惠用到的 Twin Q-networks
#------------------------------------------------------------------------------------------------------------------#
class DoubleCritic(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=256,
            activation='mish'
    ):
        super(DoubleCritic, self).__init__()
        # _act = nn.Mish if activation == 'mish' else nn.ReLU
        _act = nn.ReLU
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.q1_net = nn.Sequential(nn.Linear(hidden_dim + action_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, 1))
        self.q2_net = nn.Sequential(nn.Linear(hidden_dim + action_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      _act(),
                                      nn.Linear(hidden_dim, 1))

    # return shape 一個 tuple ((batch_size, 1), (batch_size, 1))
    def forward(self, state, action):
        processed_state = self.state_mlp(state)
        x = torch.cat([processed_state, action], dim=-1)
        return self.q1_net(x), self.q2_net(x)
    
    # return shape (batch_size)
    def q_min(self, obs, action):
        # *(self.forward(obs, action)) 是把 forward 的那個 tuple 拆開成兩個 (batch_size, 1)
        return torch.min(*self.forward(obs, action))
