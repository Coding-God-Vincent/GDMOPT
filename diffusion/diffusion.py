# Import necessary libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Import additional helper functions and utils
from .helpers import (  # . 代表 "當前檔案所在的資料夾"
    cosine_beta_schedule,
    linear_beta_schedule,
    vp_beta_schedule,
    extract,
    Losses
)
from .utils import Progress, Silent

#%%
# Define the main Diffusion class that inherits from PyTorch's nn.Module
# 定義許多在寫 GDM 時會用到的函式，包括完整的 Denoise 等
class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, model, max_action,
                 beta_schedule='vp', n_timesteps=5,
                 loss_type='l2', clip_denoised=True, bc_coef=False):
        # Call parent constructor
        super(Diffusion, self).__init__()

        #------------------------------------------------------------------------------------------------------------------#
        # Set initial attributes
        #------------------------------------------------------------------------------------------------------------------#
       
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action  # 1，將 action 限縮在 [-max_action, max_action]
        self.model = model
        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised  # boolean, 是否將輸出的動作限制在 [-max_action, max_action]
        self.bc_coef = bc_coef  # true -> with expert data, false -> without expert data

        #------------------------------------------------------------------------------------------------------------------#
        # set \beta, \alpha, \alpha_bar in each timestep t
        #------------------------------------------------------------------------------------------------------------------#

        # Define the diffusion beta schedule
        # betas -> n_timesteps 個 beta 值 in tensor
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)
        # Define alpha parameters related to the beta schedule
        alphas = 1. - betas  # [1-\beta_0, 1-\beta_1, ..., 1-\beta_(n-1)]
        alphas_cumprod = torch.cumprod(alphas, axis=0)  # alpha_bar in each t. [alpha_bar_0, alpha_bar_1, ... alpha_bar_n-1]  # cumprod() : 累乘
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])  # [1, alpha_bar_0, ..., alpha_bar_n-2]

        #------------------------------------------------------------------------------------------------------------------#
        # 將固定不更新但要跟著模型走的 tensor 存入的 register_buffer。當模型被宣告之後，這些被註冊的值也都會在
        # register_buffer : Module 提供的方法。可以用來儲存上述用途的值進模型
        #------------------------------------------------------------------------------------------------------------------#
    
        # Register these values as buffers in the module, which PyTorch will track
        self.register_buffer('betas', betas)  # [beta_t], t = 1~timestep
        self.register_buffer('alphas_cumprod', alphas_cumprod)  # [alpha_bar_t], t = 1~timestep
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)  # [1, alpha_bar_t], t = 1~timestep-1

        # Pre-calculate some quantities for the diffusion process and posterior
        # distribution calculation based on alpha and beta schedules
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # More pre-calculations for the posterior distribution
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # 1. q(x_{t-1}|x_t) 的 variance
        # 在論文中，這個 posterior 的 variance 被設為 \beta_t I，那是這邊的近似，兩種都可以，這邊這樣比較嚴謹
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        # 因為有時 posterior_variance 會算出 0，這邊用一個很小的值替換，避免梯度爆炸，這是為何要用 clipped。
        # 因為使用 log variance 能使得數值穩定、避免負值，也與理論一致，這是為何要用 log。
        # 由上，我們在整個 GDM 之中，我們所謂的 \beta_t I 就是指 posterior_log_variance_clipped。並不會用 posterior_variance
        # Log calculation clipped to avoid log(0)
        # ## log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        # 這個 posterior_log_variance_clipped 存的是個時間點的 log(variance) = log(\sigma^2)
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        
        # 2. q(x_{t-1}|x_t) 的 mean = coef1 * x_0 + coef2 * x_1
        # shape (timesteps)
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        # Select the appropriate loss function from the predefined Losses dictionary
        self.loss_fn = Losses[loss_type]()
        # loss_type = l1 -> self.loss_fn = WeightedL1()
        # loss_type = l2 -> self.loss_fn = WeightedL2()


    #------------------------------------------------------------------------------------------------------------------#
    # 由 x_t, t , epsilon 去 Predict the 近似 x_0 (x_0_hat) by the given diffused state at time t (x_T) and noise
    # coef 會跟 x_t 的每一個維度逐項相乘
    # 從 x_t 和模型預測出的 epsilon 來計算近似的 x_0
    # x_t -> shape (batch_size, action_dim)
    # t -> shape (batch_size), t -> integer
    # noise -> shape (batch_size, action_dim)
    # return shape (batch_size, action_dim)
    #------------------------------------------------------------------------------------------------------------------#
    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.explore_solution, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.bc_coef:  # 代表現在有 expert data，不用做這件事情
            return noise
        else:
            # extract : shape: (batch_size, 1) * (batch_size, action_dim) -> (batch_size, action_dim)
            # ex:
            # a = torch.tensor([[2], [3], [4]]) -> shape (3, 1)
            # b = torch.tensor([[1, 2, 3], [2, 3, 4], [4, 5, 6]]) -> shape(3, 3)
            # a*b = tensor([[ 2,  4,  6],
            #               [ 6,  9, 12],
            #               [16, 20, 24]])
            return ( 
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )

    #------------------------------------------------------------------------------------------------------------------#
    # 由 x_t, t, x_0 計算真實後驗分布 q(x_{t-1}|x_t, x_0) 的 mean、variance、log variance
    # x_start -> 一個 batch 的 x_0。shape (batch_size, action_dim)
    # x_t -> 一個 batch 的 x_t。shape (batch_size, action_dim)
    # t -> 一個 batch 各 x_t 的時間步。shape (batch_size)
    # return : 
    # posterior_mean -> shape (batch_size, action_dim)，每一個 batch 中的每一筆資料會是 action_dim 維度，每一維度都是獨立的高斯分布，故這邊每一維都是該獨立分布的平均值
    # posterior_variance -> shape (batch_size, 1)
    # posterior_log_variance -> shape (batch_size, 1)
    #------------------------------------------------------------------------------------------------------------------#
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    #------------------------------------------------------------------------------------------------------------------#
    # 由 x_t, t, state 得出近似後驗分布的 mean, variance, log_variance
    # self.model(x, t, s) -> 輸出 epsilon，shape (batch_size, action_dim)
    # x -> x_t，shape (batch_size, action_dim)
    # t -> shape (batch_size)
    # s -> state, shape (batch_size, state_dim)
    # return :
    # model_mean : 近似後驗分布的 mean。shape (batch_size, action_dim)
    # posterior_variance : 近似後驗分布的 variance。shape (batch_size, 1)
    # posterior_log_variance : 近似後驗分布的 log_variance。shape (batch_size, 1)
    #------------------------------------------------------------------------------------------------------------------#
    def p_mean_variance(self, x, t, s):
        # model 輸出 epsilon，進一步與 x_t 去算出 x_0_hat。shape (batch_size, action_dim)
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))  

        # 將 x_0_hat 限縮於 [-max_action, max_action]
        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)  # clamp_ -> 會 in-place 修改，clamp -> 會回傳一個新的 tensor
        else:
            assert RuntimeError()
        
        # 輸出近似的後驗分布 p_\theta(x_{t-1}|x_t) 的 mean, variance, log_variance
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    #------------------------------------------------------------------------------------------------------------------#
    # 用 x_t, t, state 做一次 denoise 後取樣出 x_(t-1)
    # x -> x_t。shape (batch_size, action_dim)
    # t -> timestep of each data。shape (batch_size)
    # s -> state。shape (batch_size, state_dim)
    # return -> x_(t-1)。shape (batch_size, action_dim)
    #------------------------------------------------------------------------------------------------------------------#
    # @torch.no_grad()
    # Sample from the prior distribution
    def p_sample(self, x, t, s):
        # x.shape (batch_size, action_dim) 經過 * 被拆成兩個數字 batch_size & action_dim
        # b = batch_size
        b, *_, device = *x.shape, x.device  

        # 根據 x_t, t, state 得出近似的後驗分布的 mean & log_variance
        # 相當於做一次 denoise step 後輸出近似的後驗分布
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)

        # with torch.random.fork_rng():
        #     torch.manual_seed(t)
        #     noise = torch.randn_like(x)
        
        # 除了最後一步不用 (t=0) 之外，其餘每一次 denoise 出來的平均值都要加上一個額外的噪聲，增加多樣性。
        # torch.randn_like(input, dtype, device, requires_grad) -> 輸出一個跟 input 維度相同的常態分佈取樣數值的 tensor
        # 每個維度都有獨立的噪聲
        noise = torch.randn_like(x)  # shape (batch_size, action_dim)

        # no noise when t == 0
        # 這邊是創建一個 mask，t = 0 的那筆資料其 mask 位址會是 0，不會加上噪聲
        # ex: t = [1, 0, 2] -> (t==0) = [false, true, false] -> (1 - (t==0)) = [1, 0, 1]
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))  # shape (batch_size, 1)

        # 用 reparameterization 取樣 -> x_(t-1) = mean + sigma(標準差) * noise,  noise ~ N(0, I)
        # model_log_variance -> log(\sigma^2)
        # 0.5 * log(\sigma^2) = log(\sigma)
        # exp(log(\sigma)) = sigma
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise  # shape (batch_size, action_dim)
    
    #==================================================================================================================#
    #==================================================== inference ===================================================#
    #==================================================================================================================#

    #------------------------------------------------------------------------------------------------------------------#
    # 用在 Inference
    # 做一個完整 timesteps 的 p_sample
    # 即對一個 batch 每一筆資料做一遍完整的 Reverse Process (x_t -> x_0)
    # state : shape (batch_size, state_dim)
    # shape : 要生成的出的 shape，即 GDM 要產出的動作。shape (batch_size, action_dim)
    # verbose : 是否顯示進度條
    # return_diffusion : 若為 True，會返回每個時間步的中間結果
    # return : 最終 x_0。shape (batch_size, action_dim) (還沒 clamp)
    #------------------------------------------------------------------------------------------------------------------#
    # @torch.no_grad()
    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):

        device = self.betas.device
        batch_size = shape[0]  # shape = (batch_size, action_dim)

        # with torch.random.fork_rng():
        #     torch.manual_seed(0)
        #     x = torch.randn(shape, device=device)
        # 產生 noise，shape (batch_size, action_dim)
        x = torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]
        progress = Progress(self.n_timesteps) if verbose else Silent()  # 初始化進度條

        # Reverse Process (batch_size, x_t) -> (batch_size, x_0)
        for i in reversed(range(0, self.n_timesteps)):
            # torch.full -> 將 value (i) 填滿整個 shape 為 size (batch_size,) 的 tensor
            # 所有資料的去躁步數為固定。這在 inference 是正常的
            # 通常 inference 不會一次處理一個 batch，這邊是為了平行化加速
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)  # 做一步去躁  # x -> x_{t-1} of a batch。shape (batch_size, action_dim)

            # max_action = 1.0
            # ====== for inference ======
            # x.clamp_(-self.max_action, self.max_action)
            # actions = torch.abs(x)
            # Aution = actions.detach().numpy()
            # normalized_weights = Aution / np.sum(Aution)
            # total_power = 12
            # actf = normalized_weights * total_power
            # actff = torch.from_numpy(actf).float()
            # print('x', actff)
            # ===========================

            progress.update({'t': i})
            if return_diffusion: diffusion.append(x)

        progress.close()  # 關閉進度條

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x
    
    #------------------------------------------------------------------------------------------------------------------#
    # @torch.no_grad()
    # 一次把一個 batch 的 state 傳入 p_sample_loop 得到一個 batch 的預測 x_0 後 clamp [-max_action, max_action]
    # state : shape (batch_size, action_dim)
    # *args & **kwargs -> 其餘參數 (verbose & return_diffusion)，用在傳遞給 p_sample_loop
    # return : action -> 裁減過的 action。shape (batch_size, action_dim)
    #------------------------------------------------------------------------------------------------------------------#
    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        # Clamping the actions to be between -max_action and max_action
        return action.clamp_(-self.max_action, self.max_action)
        # return action

    #==================================================================================================================#
    #==================================================== trianing ====================================================#
    #==================================================================================================================#

    #------------------------------------------------------------------------------------------------------------------#
    # 在做 forward process。即把一個 batch 的 x_0 加躁到任意時間步 t，得到 x_t
    # x_start : x_0。shape (batch_size, action_dim)
    # t : 各筆資料的目標時間步。shape (batch_size)
    # return : 一個 batch 的 x_t。shape (batch_size, action_dim)
    #------------------------------------------------------------------------------------------------------------------#
    def q_sample(self, x_start, t, noise=None):

        # if noise is not provided, generate random noise
        if noise is None:
            noise = torch.randn_like(x_start)  # generate random noises in every dimensions of shape (batch_size, action_dim)

        # x_t = sqrt(\alpha_t_bar) x_0 + sqrt(1-\alpha_t_bar) \epsilon
        sample = (  
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    #------------------------------------------------------------------------------------------------------------------#
    # 傳入一個 batch 的 x_0, state, t，在這邊會做 forward process，之後將 x_t, t, state 傳入模型輸出噪聲預測，最後算出這整個 batch 的 loss
    # x_start : x_0。shape (batch_size, action_dim)
    # state : shape (batch_size, state_dim)
    # t : timesteps of each data。shape (batch_size)
    #------------------------------------------------------------------------------------------------------------------#
    # Compute the losses based on the predictions from the model
    def p_losses(self, x_start, state, t, weights= 1.0):
        noise = torch.randn_like(x_start)  # \epsilon to be added to x_0。shape (batch_size, action_dim)

        # 跟據 x_0 & t 做 forward process 得到 x_t。shape (batch_size, action_dim)
        x_noisy = self.q_sample(x_start= x_start, t= t, noise= noise)

        # 根據 x_t & t & state 得出預測的噪聲 \epsilon。shape (batch_size, action_dim)
        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape
        
        # 根據 \epsilon 和 x_0 算出本次猜測的 loss

        # 1. 有 Expert data (Behavior Cloning)
        # 直接讓模型輸出 x_0_hat，loss_fn 為 MSE(x_0_hat, x_0)
        # 這種做法隱含的代表 GDM 能夠完美的預測噪聲 (DDPM 那篇有證明過預測噪聲跟預測 x_0 兩件事情是等價的)
        if self.bc_coef:
            loss = self.loss_fn(x_recon, x_start, weights)  # weights = 1 代表所有樣本的權重相同
        
        # 2. 無 Expert data (DRL) -> 與一般 GDM 一樣用 ||(predicted_noise - real_noise)||_2
        # 根本沒用到這個 loss，本篇沒有 expert data 時的，Loss 只靠 Critic 給的 Q 值而已 !
        # 原因是因為在無 Expert Data 時，GDM 已經轉變角色從 "去躁器" 變成 "特殊的產生出策略的網路架構"。所以他也不再需要學去躁，只要學怎麼輸出高分的動作就好。
        else:  
            loss = self.loss_fn(x_recon, noise, weights)
        # loss = self.loss_fn(x_recon, noise, weights)
        return loss

    #------------------------------------------------------------------------------------------------------------------#
    # 隨機挑選 batch 中每一筆資料的時間步，之後會將 x_0, t, state 送進 p_loss 去算整個 batch 的 loss
    # x : x_0。shape (batch_size, action_dim)
    # state : shape (batch_size, state_dim)
    # return -> loss : shape (1)
    #------------------------------------------------------------------------------------------------------------------#
    # Compute the total loss by sampling different timesteps for each data in the batch
    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        # sample a different timestep for each data in the batch
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    #------------------------------------------------------------------------------------------------------------------#
    # 一次把一個 batch 的 state 傳入 p_sample_loop 得到一個 batch 的預測 x_0 後 clamp [-max_action, max_action]
    # state : shape (batch_size, state_dim)
    # return -> 預測的 x_0 (clamped)。shape (batch_size, action_dim)
    #------------------------------------------------------------------------------------------------------------------#
    # Generate a sample from the model
    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)