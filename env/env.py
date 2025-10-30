import gym
from gym.spaces import Box, Discrete
from tianshou.env import DummyVectorEnv
from .utility import CompUtility
import numpy as np

class AIGCEnv(gym.Env):

    def __init__(self):
        
        # use to switch from 'train' to 'test'
        self._flag = 0

        # Define observation space based on the shape of the state
        # Box -> 用來定義連續空間 [low, high] 的資料結構。shape 是代表取出幾維的資料，每一維的資料皆介於 [low, high]
        self._observation_space = Box(shape=self.state.shape, low=0, high=1)

        # Define action space - discrete space with 10 possible actions
        # action_space 即是通道數目 M
        # Discrete(10) 會回傳一個物件
        # _action_space.n = 10
        # _action_space.sample() -> 從 0~10 中抽出一個數字
        self._action_space = Discrete(2*5)

        # 紀錄當前 episode 已經執行多少步，每 step() 一次會加一
        self._num_steps = 0

        # 紀錄當前 episode 是否結束
        self._terminated = False

        # 紀錄最近一次的 state 向量
        self._laststate = None

        # 紀錄最近一次專家的 action
        self.last_expert_action = None

        # Define the number of steps per episode
        self._steps_per_episode = 1
    
    # 回傳當前狀態 -> np.array of shape (no. channels)
    @property  # property 可以將方法包裝成像屬性一樣，可以直接用 class.oberservation_space 這樣呼叫而不用 class.observation_space()
    def observation_space(self):
        # Return the observation space
        return self._observation_space

    # 回傳 Discrete() 物件
    @property
    def action_space(self):
        # Return the action space
        return self._action_space
    
    # 隨機產生各 channel 的 channel gain 作為本篇模型的輸入
    # state = np.random.uniform(low, high, size= no. channels)
    # 回傳狀態 -> np.array of shape (no. channels)
    @property
    def state(self):
        # Provide the current state to the agent
        # rng = np.random.default_rng(seed=0)  # rng -> numpy 新版隨機數生成器，np.random.default_rng(seed) 會回傳一個 Generator
        # states1 = rng.uniform(1, 2, 5)
        # states2 = rng.uniform(0, 1, 5)
        # np.random.uniform(low, high, size), size -> no. of channels 
        states1 = np.random.uniform(1, 2, 5)
        states2 = np.random.uniform(0, 1, 5)

        reward_in = []
        reward_in.append(0)
        # 為了增加環境的複雜，可以依照自己的需求修改
        # reward 加在 state 裡面是作者的作法，不是 gym 規定
        states = np.concatenate([states1, states2, reward_in])  

        self.channel_gains = np.concatenate([states1, states2])
        self._laststate = states
        return states

    # action -> 為各頻道要分配的功率的 logit (M 維)，CompUtility 會把這個 logit 轉為比例後再轉成真正分配的功率
    def step(self, action):
        # Check if episode has ended
        # 若已經 terminated 還在呼叫 step() 就用後面文字報錯
        assert not self._terminated, "One episodic has terminated"

        # Calculate reward based on last state and action taken
        reward, expert_action, sub_expert_action, real_action = CompUtility(self.channel_gains, action)
        
        # 最後一格放 reward
        self._laststate[-1] = reward
        # 除了最後一格的 reward，更新前面每一格的狀態 (channel_gains)

        # **這超怪，channel gain 不會因為分配多少功率而改變
        # 已修正，改為註解，因為 channel_gains 不應該改變
        # self._laststate[0:-1] = self.channel_gains * real_action

        # self._laststate[0:-1] = self.channel_gains * real_action
        self._num_steps += 1

        # Check if episode should end based on number of steps taken
        # 若超過 episode 的最大步數就 terminate 該 episode
        if self._num_steps >= self._steps_per_episode:
            self._terminated = True

        # Information about number of steps taken
        info = {'num_steps': self._num_steps, 'expert_action': expert_action, 'sub_expert_action': sub_expert_action}
        
        # gym 規定一定要回傳 (next_state, reward, done, info)
        return self._laststate, reward, self._terminated, info

    # 初始化環境
    # 回傳初始狀態 np.array of shape (no. channels)
    def reset(self):
        # Reset the environment to its initial state
        self._num_steps = 0
        self._terminated = False
        state = self.state
        return state, {'num_steps': self._num_steps}

    def seed(self, seed=None):
        # Set seed for random number generation
        np.random.seed(seed)


# 為一個環境工廠，一次幫你建立一個單一環境、training_num 個並行的訓練環境、test_num 個並行的測試環境
def make_aigc_env(training_num=0, test_num=0):
    """Wrapper function for AIGC env.
    :return: a tuple of (single env, training envs, test envs).
    """
    env = AIGCEnv()
    env.seed(0)

    train_envs, test_envs = None, None
    if training_num:
        # Create multiple instances of the environment for training
        train_envs = DummyVectorEnv(
            [lambda: AIGCEnv() for _ in range(training_num)])
        train_envs.seed(0)

    if test_num:
        # Create multiple instances of the environment for testing
        test_envs = DummyVectorEnv(
            [lambda: AIGCEnv() for _ in range(test_num)])
        test_envs.seed(0)
    return env, train_envs, test_envs
