import torch
import copy
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from typing import Any, Dict, List, Type, Optional, Union
from tianshou.data import Batch, ReplayBuffer, to_torch
from tianshou.policy import BasePolicy
from torch.optim.lr_scheduler import CosineAnnealingLR
from .helpers import (
    Losses
)


'''About tianshou

* tianshou.data.Batch : 
Batch 是 tinashou 中獨特的資料結構。是一種類似字典的資料容器，用來打包一整批的強化學習資料。
ex : 
    Batch(
            obs: array([[1, 2],
                        [3, 4]]),
            act: array([0, 1]),
            rew: array([1. , 0.5]),
            done: array([False,  True])
        )

* tianshou.data.ReplayBuffer : 
ReplayBuffer 是用來儲存 agent 與環境互動資料 (trajectories 或 transitions) 的資料結構。其本身是一個大 Batch。
reward 都只存 single step 的。每一筆資料是有順序性的，因為要算 n-step return。
ex : 
假設有 2 筆 data : 
    data1 : obs= [1, 2]、act= 0、rew= 0.1、done= F、obs_next= [2, 3]
    data2 : obs= [2, 3]、act= 1、rew= 0.3、done= F、obs_next= [3, 4]
則 ReplayBuffer = Batch (
                        obs = np.array([[1, 2], [2, 3]]),
                        act = np.array([0, 1]),
                        rew = np.array([0.1, 0.3]),
                        done = np.array([F, F]),
                        obs_next = np.array([[2, 3], [3, 4]])
                  )
'''

'''About Inherit BasePolicy
只要繼承 tianshou 的 BasePolicy 就一定要有以下幾個函式 : 
1. forward() : 傳入一個 batch，算出各 state 輸出的 Action。return tensor with shape (batch_size, action_dim)
2. process_fn() : 傳入一個 batch，算出該 batch 中各資料的 Target Q (用於更新 Twin Q network)。return Batch (多了一個屬性 : Batch.returns)
3. learn() : 用一個 batch 更新所有的網路
4. update() : 傳入 replay buffer，從中抽取出一個 batch 之後，利用該 batch 更新所有的網路 (用到 process_fn() 產出新 batch 後傳入 learn())
'''

class DiffusionOPT(BasePolicy):

    # python 會自動判斷資料型態，這邊寫出型別提示只是為了方便維護跟閱讀
    def __init__(
            self,
            # dimension of the state aka no. of channels
            state_dim: int,  # 型別提示，只跟他講型態，不用給初始值沒關係
            # actor network
            actor: Optional[torch.nn.Module],  # 這種寫法是來自 typing 模組，意思是 actor 的型態要馬是 torch.nn.Module，要馬是 None
            # optimizer of the actor network
            actor_optim: Optional[torch.optim.Optimizer],
            # dimension of the actor aka no. of channels
            action_dim: int,
            # critic network
            critic: Optional[torch.nn.Module],
            # optimizer of the critic network
            critic_optim: Optional[torch.optim.Optimizer],
            # dist_fn: Type[torch.distributions.Distribution],
            device: torch.device,
            # soft updata parameters (\theta_target <- tau * \theta + (1-tau) * \theta_target)
            tau: float = 0.005,
            # discount factor
            gamma: float = 1,
            reward_normalization: bool = False,
            # n_steps
            estimation_step: int = 1,
            # decay learning rate or not
            lr_decay: bool = False,
            # use lr_maxt steps to decay the learning rate to the ultimate learning rate
            lr_maxt: int = 1000,
            # have expert data or not
            bc_coef: bool = False,
            # 10% chance to add a gaussian noise to the action (enhance exploration)
            # only apply on the training phase with no expert data (we want to imitate the expert data as similar as possible so we don't add noise with expert data)
            exploration_noise: float = 0.1,  # std in gaussian distribution of the noise generator
            **kwargs: Any
    ) -> None:  # -> 會接 return type，這邊 None 是指不會回傳任何東西 (init 本來就不會回傳任何東西，這邊只是寫出來而已，可能因為這篇是教學型論文)
        
        super().__init__(**kwargs)
        assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"

        # Initialize actor network and its optimizer if provided
        if actor is not None and actor_optim is not None:
            self._actor: torch.nn.Module = actor  # Actor network
            self._target_actor = deepcopy(actor)  # Target actor network for stable learning
            self._target_actor.eval()  # Set target actor to evaluation mode
            self._actor_optim: torch.optim.Optimizer = actor_optim  # Optimizer for the actor network
            self._action_dim = action_dim  # Dimensionality of the action space

        # Initialize critic network and its optimizer if provided
        if critic is not None and critic_optim is not None:
            self._critic: torch.nn.Module = critic  # Critic network
            self._target_critic = deepcopy(critic)  # Target critic network for stable learning
            self._critic_optim: torch.optim.Optimizer = critic_optim  # Optimizer for the critic network
            self._target_critic.eval()  # Set target critic to evaluation mode

        # If learning rate decay is applied, initialize learning rate schedulers for both actor and critic
        if lr_decay:
            # CosineAnnealingLR -> 於訓練中動態調整 lr 的函式
            # T_max : 一個周期 (從最大值降到最小值) 所需的步數
            # eta_min : 最小的 lr 數值
            self._actor_lr_scheduler = CosineAnnealingLR(self._actor_optim, T_max=lr_maxt, eta_min=0.)
            self._critic_lr_scheduler = CosineAnnealingLR(self._critic_optim, T_max=lr_maxt, eta_min=0.)

        # Initialize other parameters and configurations
        self._tau = tau  # Soft update coefficient for target networks
        self._gamma = gamma  # Discount factor for future rewards
        self._rew_norm = reward_normalization  # If true, normalize rewards
        self._n_step = estimation_step  # Steps for n-step return estimation
        self._lr_decay = lr_decay  # If true, apply learning rate decay
        self._bc_coef = bc_coef  # Coefficient for policy gradient loss
        self._device = device  # Device to run computations on
        self.noise_generator = GaussianNoise(sigma= exploration_noise)
    
    '''
    * Actor 的 loss (以 single step 為例)
        Actor_loss = - E[ min{ Q^k(s, a) } ], k = 1, 2

    * Critic_k 的 loss (以 single step 為例)
        Critic^k_loss = E[ MSE( Q^k(s, a), (r + \gamma * min{ Q^k_target(s', a'_target) } ) ) ], k = 1, 2
        * a'_target = target_actor(s')
    '''

    #------------------------------------------------------------------------------------------------------------------#
    # 給他當前 batch 的資料，算出各資料的 Critic^k_loss 中的 min{ Q^k_target(s', a'_target) } 用以更新 Q network。(Twin Q network 皆使用這個 target Q 值來進行更新)
    # buffer : Replay buffer
    # indices : 一個 batch 資料的 index。shape (batch_size)
    # return : target_q，各資料的 min{ Q^k_target(s', a'_target) }。shape (batch_size)
    #------------------------------------------------------------------------------------------------------------------#
    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        # 為一個 tianshou.data.batch，裡頭裝一個 batch 的資料
        batch = buffer[indices]  
        # 將一個 batch 的資料傳入 _target_actor 並得到 actions。此及 a'_target。shape (batch_size, action_dim)
        # self() 會呼叫 __call__()，而因為 Diffusion_OPT 這個 class 繼承 BasePolicy，所以會呼叫到 forward() (forward 寫在下面)
        ttt = self(batch, model='_target_actor', input='obs_next').act
        # 取出 batch 中的 next_state。shape (batch_size, state_dim)
        batch.obs_next = to_torch(batch.obs_next, device=self._device, dtype=torch.float32)  # 把 np.array 或 list 轉為 tensor
        # min{ Q^k_target(s', a'_target) }
        # critic 在 diffusion/model.py，那邊可以看到有一個 q_min 會給出兩個 Q 網路的最小的那個。
        target_q = self._target_critic.q_min(batch.obs_next, ttt)
        return target_q  # return the minimum of the dual Q values


    #------------------------------------------------------------------------------------------------------------------#
    # 算出一個 batch 中各筆資料的 Target Q 值。(r0 + \gamma*r1 + \gamma^(2)*r2 + ... + \gamma^(n-1)*r(n-1) + \gamma^(n)*Q(sn, an))
    # 這邊的 Q(sn, an) 是前面 _target_q_() 出來的 min{ Q^k_target(s', a'_target) }
    # batch : 一個 batch 的資料。 (可以透過 batch.obs、obs.act... 取出各筆資料的各項資料)
    # buffer : replay buffer
    # indices : 一個 batch 的資料於 replay buffer 中的 indices。shape (batch_size)
    # return : Batch。結果會放在 Batch.returns
    #------------------------------------------------------------------------------------------------------------------#
    def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
        # Compute n-step return for transitions in the batch
        return self.compute_nstep_return(
            batch,  # 一個 batch 中的資料
            buffer,  # 傳入 buffer & indices 是因為本篇採用 n-step，要知道後面 n 筆資料才有辦法算 n-step return
            indices,
            self._target_q,  # 用來算 Q(s_n, a_n)
            self._gamma,  # discount factor
            self._n_step,  # n_step 步數
            self._rew_norm  # 是否對 reward 做標準化。即所有 single step reward 會一起做一個標準化 (normalized_reward = (reward - mean) / (std + epsilon))
        )

    #------------------------------------------------------------------------------------------------------------------#
    # 從 Replay buffer 中抽取一個 batch 的資料來更新一遍所有的 networks
    # sample_size : batch_size
    # buffer : Replay buffer
    # return result : dict {一個 batch 的 critic loss (Twin Q network 的 loss 總和), 一個 batch 的 Actor loss}
    #------------------------------------------------------------------------------------------------------------------#
    def update(
            self,
            sample_size: int,
            buffer: Optional[ReplayBuffer],
            **kwargs: Any
    ) -> Dict[str, Any]:
        
        # If no replay buffer is provided, return an empty dictionary
        if buffer is None: return {}
        self.updating = True # Indicate that the policy is being updated

        # Sample 一個 batch 的資料
        batch, indices = buffer.sample(sample_size)
        # 計算該 batch 中各資料的 Target Q 值 (用以更新 Twin Q network)
        batch = self.process_fn(batch, buffer, indices)
        # 用一個 batch 的資料去更新所有 networks (Actor & Twin Q network & Target Networks)
        # result : dict {一個 batch 的 critic loss (Twin Q network 的 loss 總和), 一個 batch 的 Actor loss}
        result = self.learn(batch, **kwargs)
        # 更新完一次，更新一次 lr
        if self._lr_decay: # If learning rate decay is enabled, step the learning rate schedulers
            self._actor_lr_scheduler.step()
            self._critic_lr_scheduler.step()
        self.updating = False # Indicate that the policy update has finished
        return result

    #------------------------------------------------------------------------------------------------------------------#
    # 傳入一個 batch 的資料，指定好輸入狀態後回傳 Actor 的結果。最後全部存入 Batch 中回傳該 Batch
    # batch : 一個 batch 的 data
    # input : 可以是 "obs" or "obs_next"。說明你是要使用當前狀態還是下一個狀態
    # model : 可以是 actor or _target_actor。說明你是要使用 actor 還是 _target_actor 來做
    # return : Batch
    #------------------------------------------------------------------------------------------------------------------#
    def forward(
            self,
            batch: Batch,
            state: Optional[Union[dict, Batch, np.ndarray]] = None,
            input: str = "obs",
            model: str = "actor"
    ) -> Batch:
        
        # Convert batch observations to PyTorch tensors
        obs_ = to_torch(batch[input], device=self._device, dtype=torch.float32)
        # Use actor or target actor based on provided model argument
        model_ = self._actor if model == "actor" else self._target_actor
        # Feed observations through the selected model to get action logits
        # model_(obs_) -> model___call__(obs_) by PyTorch -> model_.forward(obs_) in diffusion.py -> model_.sample(obs_)
        # logits.shape = (batch_size, action_dim)
        # 這邊這樣寫是為了配合 tianshou，在他的一些演算法中會有需要回傳隱藏狀態，例如 RNN。這邊確實是不用。
        logits, hidden = model_(obs_), None

        if self._bc_coef:  # 有專家資料 -> 不用加上額外噪聲
            acts = logits
        else:  # 無專家資料 -> 有 10% 要加上額外噪聲增加探索性
            if np.random.rand() < 0.1:
                # Add exploration noise to the actions
                noise = to_torch(self.noise_generator.generate(logits.shape),
                                 dtype=torch.float32, device=self._device)
                # Add the noise to the action
                acts = logits + noise
                acts = torch.clamp(acts, -1, 1)  # 原本 GDM 輸出的 action 就有 clamp 過了，這邊加上 noise 之後又要再 clamp 一次
            else:
                acts = logits

        # dist 是一個
        # PPO、SAC 等演算法會輸出機率分布，再從此機率分布中抽樣出動作，這時就會把該機率分布設為 dist
        # 若今天是使用 PPO 要自己取樣出 act 傳入，也要再傳入 dist，這是因為 dist 要用於 PPO 的重要性採樣
        dist = None  # does not use a probability distribution for actions

        # 設定 Batch.logits = logits, Batch.act = acts...
        return Batch(logits=logits, act=acts, state=obs_, dist=dist)

    #------------------------------------------------------------------------------------------------------------------#
    # 將一個 batch_size 的向量中的各維數字轉為 one-hot-vector。
    # data : 一個存 np.array 的整數 np.array。shape 為 (batch_size)
    # one_hot_dim : one_hot_vector 長度 (data 中最大的數字 + 1，因為這邊出來的 one-hot-vector 是從 0 開始，ex [1, 0, 0] -> 為 0 的 one-hot vector)
    # return shape : (batch_size, one_hot_dim)
    #------------------------------------------------------------------------------------------------------------------#
    def _to_one_hot(
            self,
            data: np.ndarray,
            one_hot_dim: int
    ) -> np.ndarray:
        
        batch_size = data.shape[0]
        # np.eye(N, M= 0, k, dtype= float) -> 用來產生一個單位矩陣。
        # N : 列數、M : 行數 (M = 0 則 M = N，則會輸出一個方陣)、k 偏移量 (k = 0 則 1 在對角線的位置)
        # ex : np.eye(N= 3, M= 0, k= 0)
        # 輸出:
        # [[1., 0., 0.],
        #  [0., 1., 0.],
        #  [0., 0., 1.]]
        # 產生一個 shape (one_hot_dim, one_hot_dim) 的方陣
        one_hot_codes = np.eye(one_hot_dim)  
        # 取出對應的數字的 one_hot_vector
        one_hot_res = [one_hot_codes[data[i]].reshape((1, one_hot_dim)) for i in range(batch_size)]
        return np.concatenate(one_hot_res, axis=0)

    #------------------------------------------------------------------------------------------------------------------#
    # 輸入一個 batch 的資料 (已經有做過 process_fn 的 Batch，裡面已經有 Batch.returns) 來更新一次 Twin Q-Network (不包括 Target Twin Q-network)
    # Critic^k_loss = E[ MSE( Q^k(s, a), (r + \gamma * min{ Q^k_target(s', a'_target) } ) ) ], k = 1, 2
    # batch : 一個 batch 的資料
    # return critic_loss (一個 batch 中兩個 Q-Network 的 loss 和)
    #------------------------------------------------------------------------------------------------------------------#
    def _update_critic(self, batch: Batch) -> torch.Tensor:
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        acts_ = to_torch(batch.act, device=self._device, dtype=torch.float32)
        # target_q : batch 中各筆資料的 target Q 值 (Critic^k_loss 中的 (r + \gamma * min{ Q^k_target(s', a'_target, k = 1, 2))
        target_q = batch.returns  
        # Critic^k_loss 中的 Q^k(s, a), k = 1, 2
        current_q1, current_q2 = self._critic(obs_,acts_)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q) # Compute the MSE losses
        self._critic_optim.zero_grad() # Zero the critic optimizer's gradients
        critic_loss.backward() # Backpropagate the loss
        self._critic_optim.step() # Perform a step of optimization
        return critic_loss

    #------------------------------------------------------------------------------------------------------------------#
    # With Expert data 的情況下，用一個 batch 的資料去更新一次 Actor (update = true 時)
    # Actor_loss = E[ MSE( x_0_hat, exper_action ) ]
    # batch : 一個 batch 中的資料
    # update : 是否在此函式中進行參數更新 (預設為 false，因為想在 learn() 中統一做更新)
    # return 該 batch 中的 Actor_loss
    #------------------------------------------------------------------------------------------------------------------#
    def _update_bc(self, batch: Batch, update: bool = False) -> torch.Tensor:
        # Compute the behavior cloning loss
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        expert_actions = torch.Tensor([info["expert_action"] for info in batch.info]).to(self._device)
        
        # diffusion.py 中的 p_loss()
        '''
            * 有 Expert data (Behavior Cloning): 
                直接讓模型輸出 x_0_hat，loss_fn 為 MSE(x_0_hat, x_0) (x_0 為 expert_actions)
                這種做法隱含的代表 GDM 能夠完美的預測噪聲 (DDPM 那篇有證明過預測噪聲跟預測 x_0 兩件事情是等價的)
                if self.bc_coef:
                    loss = self.loss_fn(x_recon, x_start, weights)  # weights = 1 代表所有樣本的權重相同
        '''
        bc_loss = self._actor.loss(expert_actions, obs_).mean()

        if update:  # Update actor parameters if update flag is True
            self._actor_optim.zero_grad()  # Zero the actor optimizer's gradients
            bc_loss.backward()  # Backpropagate the loss
            self._actor_optim.step()  # Perform a step of optimization
        return bc_loss

    #------------------------------------------------------------------------------------------------------------------#
    # Without Expert data 的情況下，用一個 batch 的資料去更新一次 Actor (update = true 時)
    # Actor_loss = - E[ min{ Q^k(s, a) } ], k = 1, 2
    # batch : 一個 batch 中的資料
    # update : 是否在此函式中進行參數更新 (預設為 false，因為想在 learn() 中統一做更新)
    # return 該 batch 中的 Actor_loss
    #------------------------------------------------------------------------------------------------------------------#
    def _update_policy(self, batch: Batch, update: bool = False) -> torch.Tensor:
        # Compute the policy gradient loss
        obs_ = to_torch(batch.obs, device=self._device, dtype=torch.float32)
        acts_ = to_torch(self(batch).act, device=self._device, dtype=torch.float32)
        # Without expert data 的 Actor_loss 
        # q_min 輸出 shape (batch_size)，為該 batch 中各 action 所算出的 q_min
        pg_loss = - self._critic.q_min(obs_, acts_).mean()
        if update:
            self._actor_optim.zero_grad()
            pg_loss.backward()
            self._actor_optim.step()
        return pg_loss

    #------------------------------------------------------------------------------------------------------------------#
    # soft update target actor 和 target twin Q networks
    #------------------------------------------------------------------------------------------------------------------#
    def _update_targets(self):
        # Perform soft update on target actor and target critic. Soft update is a method of slowly blending
        # the regular and target network to provide more stable learning updates.
        self.soft_update(self._target_actor, self._actor, self._tau)
        self.soft_update(self._target_critic, self._critic, self._tau)

    #------------------------------------------------------------------------------------------------------------------#
    # 用一個 batch 的資料進行所有網路的更新
    # 順序 : 更新 Twin Q network (在 _update_critic() 中更新) -> 更新 Actor Network (在本函式中更新) -> 更新所有的 Target Network (在 _update_targets() 中更新)
    # return dict: { 一個 batch 的 total_critic_loss, 一個 batch 的 total_actor_loss }
    #------------------------------------------------------------------------------------------------------------------#
    def learn(
            self,
            batch: Batch,
            **kwargs: Any
    ) -> Dict[str, List[float]]:
        # Update Twin Q network and return the sum of the MSE losses of the two Q networks
        critic_loss = self._update_critic(batch)

        # Update actor network. Here, we first calculate the policy gradient (pg_loss) and
        # behavior cloning loss (bc_loss) but we do not update the actor network yet.
        # The overall loss is a weighted combination of policy gradient loss and behavior cloning loss.
        if self._bc_coef:  # with expert data
            bc_loss = self._update_bc(batch, update=False)
            overall_loss = bc_loss
        else:  # without expert data
            pg_loss = self._update_policy(batch, update=False)
            overall_loss = pg_loss

        self._actor_optim.zero_grad()
        overall_loss.backward()
        self._actor_optim.step()

        # Update the target networks
        self._update_targets()
        return {
            'loss/critic': critic_loss.item(),  # Returns the critic loss as part of the results
            'overall_loss': overall_loss.item()  # Returns the overall loss as part of the results
        }

#=====================================================================================================================================================================================#
class GaussianNoise:
    """Generates Gaussian noise."""

    def __init__(self, mu= 0.0, sigma= 0.1):
        """
        :param mu: Mean of the Gaussian distribution.
        :param sigma: Standard deviation of the Gaussian distribution.
        """
        self.mu = mu
        self.sigma = sigma

    # 從均勻分布中抽樣出一個值
    def generate(self, shape):
        """
        Generate Gaussian noise based on a shape .

        :param shape: Shape of the noise to generate, typically the action's shape.
        :return: Numpy array with Gaussian noise.
        """
        noise = np.random.normal(self.mu, self.sigma, shape)
        return noise
