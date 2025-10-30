import numpy as np
import torch
from scipy.stats import nakagami
from scipy.special import gammainc
import math
from scipy.io import savemat
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# 沒用到，創建狀態時就有用均勻分布取樣出 channel gain 了
def rayleigh_channel_gain(ex, sta):
    num_samples = 1
    gain = np.random.normal(ex, sta, num_samples)
    # Square the absolute value to get Rayleigh-distributed gains
    gain = np.abs(gain) ** 2
    return gain

# Function to implement water filling algorithm for power allocation
# s -> state aka channel gains of the channels (shape M)
# return expert，專家分給各通道的功率比 (shape M)，sumdata_rate，專家達到的 sum rate(shape 1)，subexpert，比專家還要爛一點的分配 (shape M)
def water(s, total_power):
    a = total_power
    # Define the channel gain and noise level
    g_n = s
    N_0 = 1  # Assuming a fixed noise-level of 1 for all transmissions, this can be changed based on your requirement

    # Initialize the upper and lower bounds for the bisection search
    L = 0
    U = a + N_0 * np.sum(1 / (g_n + 1e-6))  # Initial guess for upper bound

    # Define the precision for the bisection search
    precision = 1e-6

    # Perform the bisection search for the power level
    while U - L > precision:
        alpha_bar = (L + U) / 2  # Set the current level to be in the middle of bounds
        p_n = np.maximum(alpha_bar - N_0 / (g_n + 1e-6), 0)  # Calculate the power allocation
        P = np.sum(p_n)  # Calculate the total power

        # Check whether the power budget is under or over-utilized
        if P > a:  # If the power budget is over-utilized
            U = alpha_bar  # Move the upper bound to the current power level
        else:  # If the power level is below the power budget
            L = alpha_bar  # Move the lower bound up

    # Calculate the final power allocation
    p_n_final = np.maximum(alpha_bar - N_0 / (g_n + 1e-6), 0)

    # Calculate the data rate for each channel
    SNR = g_n * p_n_final / N_0  # Calculate the SNR
    data_rate = np.log2(1 + SNR)  # Calculate the data rate
    sumdata_rate = np.sum(data_rate)
    # print('p_n_final', p_n_final)
    # print('data_rate', sumdata_rate)
    expert = p_n_final / total_power  # 分配給各 channel 的功率比 (shape M)
    subexpert = p_n_final / total_power + np.random.normal(0, 0.1, len(p_n_final))
    return expert, sumdata_rate, subexpert

# Function to compute utility (reward) for the given state and action
# action 為一個 M 維向量，代表每一個通道分配的 logit
# 這邊會把 action 中各維的數字轉為比例，再依照此比例將 logit 轉換為真實的功率
# State 為一個 M 維向量，代表每一個通道的 channel gain
def CompUtility(State, Aution):
    actions = torch.from_numpy(np.array(Aution)).float()
    actions = torch.abs(actions)
    # actions = torch.sigmoid(actions)
    Aution = actions.numpy()
    total_power = 3
    normalized_weights = Aution / np.sum(Aution)  # 將 logit 轉為分配的比例
    a = normalized_weights * total_power  # a 為各頻道真正分配到的功率總量

    g_n = State
    SNR = g_n * a
    data_rate = np.log2(1 + SNR)  # shape (M), 各通道能達到的資料傳輸速率

    expert_action, sumdata_rate, subopt_expert_action = water(g_n, total_power)

    # **reward 幹嘛減掉 sumdata_rate，論文中有說明 reward 是用 objective value (已修正)
    # reward = np.sum(data_rate) - sumdata_rate
    reward = np.sum(data_rate)

    # **env.py 那邊的 step() 將最後的 Aution 設為 real action，所以應該要是各頻道真實的功率分配才對 (已修正)
    # return reward, expert_action, subopt_expert_action, Aution
    return reward, expert_action, subopt_expert_action, a