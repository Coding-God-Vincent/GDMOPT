import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#------------------------------------------------------------------------------------------------------#
# 對 timestep 進行位置編碼 (positional embedding)，讓模型知道現在是 Diffusion 的第幾步
# 並非一個神經網路，只是因為很常被當作模組呼叫所以繼承 nn.Module 來寫
#------------------------------------------------------------------------------------------------------#

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

#------------------------------------------------------------------------------------------------------#
# 將一個 batch 中各 x_t 的時間步 t 和其對應的係數 a 取出放在一個 tensor 之中
# a -> 跟 t 有關的係數，例如各時間點的 \alpha_bar_t。shape (timesteps)
# t -> 時間點，每一維度的值皆介於 [0, timesteps-1]。shape (batch_size)
#------------------------------------------------------------------------------------------------------#
# * -> 讓值從資料結構中解脫變成獨立的值。ex: *(1, 2, 3) -> 1, 2, 3
def extract(a, t, x_shape):
    b, *_ = t.shape  # b -> timesteps
    out = a.gather(-1, t)  # dim= -1 (即最後一個維度，在這邊 a.shape = (timesteps)) # 這邊就是取出時間點 t 對應的係數 # output_shape (timesteps)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # shape (timesteps, 1, 1, ..., 1) 後面的 1 有 len(x_shape - 1) 
    # 若 x_shape (batch_size, action_dim)，len(x_shape) = 1，那 out.shape 會變成 (b, 1)

#------------------------------------------------------------------------------------------------------#
# 3 types of \beta scheduling
# 回傳 timestpes 個 beta_t，並將這些值放入一個 tensor
#------------------------------------------------------------------------------------------------------#

# beta 利用 cosine 產生平滑的值
def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

# beta 從 1e-4 線性增加到 2e-2
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2, dtype=torch.float32):
    betas = np.linspace(
        beta_start, beta_end, timesteps
    )
    return torch.tensor(betas, dtype=dtype)

# 此 project 中所使用，使隨機變數的變異數保持為 1 的那種
# 保證訊號在整個 Diffusion 過程中變異數的變化平滑可控，適用於 SDE-based diffusion
def vp_beta_schedule(timesteps, dtype=torch.float32):
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return torch.tensor(betas, dtype=dtype)

#-----------------------------------------------------------------------------#
#---------------------------------- losses -----------------------------------#
#-----------------------------------------------------------------------------#

class WeightedLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, targ, weights=1.0):
        '''
            pred, targ : tensor [ batch_size x action_dim ]
        '''
        loss = self._loss(pred, targ)
        weighted_loss = (loss * weights).mean()
        return weighted_loss

class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')


Losses = {
    'l1': WeightedL1,
    'l2': WeightedL2,
}


class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new