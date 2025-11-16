# Import necessary libraries
import argparse
import os
import pprint
import torch
import numpy as np
from datetime import datetime
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.trainer import offpolicy_trainer
from torch.distributions import Independent, Normal
from tianshou.exploration import GaussianNoise
from env import make_aigc_env
from policy import DiffusionOPT
from diffusion import Diffusion
from diffusion.model import MLP, DoubleCritic
import warnings

# Ignore warnings
# 因為大量 warnings 會使得 console 很難讀
# 下面這種是直接隱藏所有 warnings，也可以透過指定，隱藏掉某些 warnings
# ex: warnings.filterwarnings("ignore", module="torch")  # 隱藏掉 torch 這個 module 的 warnings
# ex: warnings.filterwarnings("ignore", category=FutureWarning)  # 隱藏掉 FutureWarning 這個種類的 warnings
warnings.filterwarnings('ignore')

#------------------------------------------------------------------------------------------------------------------#
# Define a function to get command line arguments
#------------------------------------------------------------------------------------------------------------------#
def get_args():
    # Create argument parser
    parser = argparse.ArgumentParser()  # argument collector

    # 前面有 -- or - 幾乎都要加上 default
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument('--algorithm', type=str, default='diffusion_opt')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--buffer-size', type=int, default=1e6)#1e6
    parser.add_argument('-e', '--epoch', type=int, default=1e6)# 1000
    parser.add_argument('--step-per-epoch', type=int, default=1)# 100
    parser.add_argument('--step-per-collect', type=int, default=1)#1000
    # 在輸入時可以用 -b or --batch-size 代表，但在程式中只能用 parser.batch_size 中取得
    parser.add_argument('-b', '--batch-size', type=int, default=512)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--log-prefix', type=str, default='default')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument('--rew-norm', type=int, default=0)
    # parser.add_argument(
    #     '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument(
        '--device', type=str, default='cuda:0')
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument('--watch', action='store_true', default=False)
    parser.add_argument('--lr-decay', action='store_true', default=False)
    parser.add_argument('--note', type=str, default='')

    # for diffusion
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-4)
    parser.add_argument('--tau', type=float, default=0.005)  # for soft update
    # adjust
    parser.add_argument('-t', '--n-timesteps', type=int, default=6)  # for diffusion chain 3 & 8 & 12
    parser.add_argument('--beta-schedule', type=str, default='vp',
                        choices=['linear', 'cosine', 'vp'])  # 會自動看你的輸入是否為 choices 中任一

    # With Expert: bc-coef True
    # Without Expert: bc-coef False
    # parser.add_argument('--bc-coef', default=False) # Apr-04-132705
    parser.add_argument('--bc-coef', default=False)

    # for prioritized experience replay
    parser.add_argument('--prioritized-replay', action='store_true', default=False)
    parser.add_argument('--prior-alpha', type=float, default=0.4)#
    parser.add_argument('--prior-beta', type=float, default=0.4)#

    # Parse arguments and return them
    # args, unknown = parser.parse_known_args()
    # 下面可以透過 args.參數名 取出傳入的參數值
    args = parser.parse_known_args()[0]
    return args

#------------------------------------------------------------------------------------------------------------------#
def main(args=get_args()):
    
    # create environments (包成 gym)
    env, train_envs, test_envs = make_aigc_env(args.training_num, args.test_num)
    args.state_shape = env.observation_space.shape[0]
    args.action_shape = env.action_space.n  # env.action_space 為一個 Discrete 物件，Discrete.n 會回傳該物件內元素數量，即 action 個數
    args.max_action = 1.
    # 確保探索噪聲的強度跟動作空間的範圍大小成比例，若動作空間範圍大則噪聲強度也一併提高
    args.exploration_noise = args.exploration_noise * args.max_action

    # seed
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # train_envs.seed(args.seed)
    # test_envs.seed(args.seed)

    # create actor
    actor_net = MLP(
        state_dim=args.state_shape,
        action_dim=args.action_shape
    )
    # Actor is a Diffusion model
    # 用 GDM 做完整加噪 & 去躁
    actor = Diffusion(
        state_dim=args.state_shape,
        action_dim=args.action_shape,
        model=actor_net,
        max_action=args.max_action,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.n_timesteps,  # denoise steps
        bc_coef = args.bc_coef
    ).to(args.device)
    actor_optim = torch.optim.AdamW(
        actor.parameters(),
        lr=args.actor_lr,
        weight_decay=args.wd  # default 1e-4
    )

    # Create critic
    critic = DoubleCritic(
        state_dim=args.state_shape,
        action_dim=args.action_shape
    ).to(args.device)
    critic_optim = torch.optim.AdamW(
        critic.parameters(),
        lr=args.critic_lr,
        weight_decay=args.wd
    )

    ## Setup logging
    # 紀錄當前時間並轉換為 Jan30-153205 (1 月 30 日 15 時 32 分 05 秒) 這樣的形勢
    time_now = datetime.now().strftime('%b%d-%H%M%S')
    # ex: args.logdir= "logs", args.log_prefix= "expA", time_now= Jan30-153205
    # 則 log_path = 'logs/expA/difusion/Jan30-153205'
    # os.path 最強的地方就是他會自己依照作業系統去加上 / or \，也讓路徑設定變得簡單，每一項用逗號隔開就好了
    log_path = os.path.join(args.logdir, args.log_prefix, "diffusion", time_now)
    # 若要寫入 TensorboardLogger，那一定需要 writer，這些都是 tianshou 的套件
    # 有這兩個東西他就會自己寫 log
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # def dist(*logits):
    #    return Independent(Normal(*logits), 1)

    # Define policy
    # 傳 replay buffer 進去，會自行抽 batch 出來並訓練所有的網路
    # DiffusionOPT 繼承 tianshou 的 BasePolicy，因此可以透過 parameters() 取出所有模型的參數。
    # 也可以透過 torch.save() 存下所有模型的參數等跟 nn.Module 定義出的模型一樣的參數相關函式
    policy = DiffusionOPT(
        args.state_shape,
        actor,  # instance of Diffusion()
        actor_optim,
        args.action_shape,
        critic,
        critic_optim,
        # dist,
        args.device,
        tau=args.tau,
        gamma=args.gamma,
        estimation_step=args.n_step,
        lr_decay=args.lr_decay,
        lr_maxt=args.epoch,
        bc_coef=args.bc_coef,
        action_space=env.action_space,
        exploration_noise = args.exploration_noise,
    )

    # 用在訓練過程被中斷時
    # 載入先前訓練到一半的模型參數並移至 device 上
    if args.resume_path:
        ckpt = torch.load(args.resume_path, map_location=args.device)
        policy.load_state_dict(ckpt)
        print("Loaded agent from: ", args.resume_path)

    # Setup buffer
    if args.prioritized_replay:
        # 優先經驗回放，會根據 TD-error 大小優先採樣更有價值的經驗
        buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.prior_alpha,  # 控制優先採樣的強度。0 代表完全使用均勻採樣，1 表示完全使用優先權採樣
            # 重要性採樣權重的指數，用於修正優先採樣帶來的偏差。因為你一直使用高 TD0-error 的經驗，改變了原始數據的分布
            # 跟 PPO 所做的前後策略的分布差異不同，PPO 中還是要自己做一次修正
            beta=args.prior_beta,  
        )
    else:
        # 所有經驗被平等對待、訓練時採用均勻採樣抽取經驗
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs)
        )

    # Setup collector
    # 像是一個作法定義，會使用 policy 到 train_envs 中做互動後將經驗存入 buffer
    # 會再搭配 tianshou.offpolicy_trainer 來控制要做幾步等等的行為
    train_collector = Collector(policy, train_envs, buffer)
    # 用於評估 policy，不存儲經驗到 buffer
    test_collector = Collector(policy, test_envs)

    # 將最佳的 policy 模型參數儲存至指定位置
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    # Trainer
    if not args.watch:  # args.watch : 1 -> testing, otherwise training
        # result : dict
        # 前 8 個參數是位置參數，必須按照以下的順序傳入
        result = offpolicy_trainer(
            policy,  # 要訓練的策略對象
            train_collector,  
            test_collector,  
            args.epoch,  # max_epoch : 總訓練輪數 (default 1e6)。每一個 epoch 結束後會評估一次模型，若是當前最佳的效能會將其參數儲存
            args.step_per_epoch,  # 一個 epoch 步數 (default = 1)
            # epoch * step_per_epoch = 總訓練步數
            args.step_per_collect,  # 每次調用 collector 蒐集的步數 (default = 1)
            args.test_num,  # episode_per_test，測試用的 episodes 數 (default = 1)
            args.batch_size,  # batch 大小 (default = 512)
            save_best_fn=save_best_fn,  # 保存在整個訓練過程中最佳的那個模型參數。若訓練過程中發現後面有更好的就會直接覆蓋
            logger=logger,  # 日誌紀錄器
            test_in_train=False  # 是否在訓練期間做測試
        )

        # pprint -> pretty print。將資料結構以可讀性高的樣式印出來
        pprint.pprint(result)

    # Watch the performance
    # python main.py --watch --resume-path log/default/diffusion/Jul10-142653/policy.pth
    if __name__ == '__main__':
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1) #, render=args.render
        print(result)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")

#------------------------------------------------------------------------------------------------------------------#
# 主程式，就呼叫 main 並把參數傳遞進去
#------------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    main(get_args())
