### Python version: 9.1

import numpy as np
import pandas as pd
import datetime

import milling_tool_environment
import utilities
from milling_tool_environment import MillingTool_SS_V2, MillingTool_MS_V2
from utilities import compute_metrics, compute_metrics_simple, write_metrics_report, store_results, plot_learning_curve, single_axes_plot
from utilities import two_axes_plot, two_variable_plot, plot_error_bounds, test_script, write_test_results, downsample
from reinforce_classes import PolicyNetwork, Agent

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO, DQN

dt = datetime.datetime.now()
dt_d = dt.strftime('%d-%b-%Y')
dt_t = dt.strftime('%H_%M_%S')
dt_m = dt.strftime('%H%M')

# Version name: <Ver.No>_Sim_<N or W>NB_<episodes>_test-run
# Data Files: Simulated_Dasic_2006_Tool_Wear_Model. Threshold 3mm
# PHM:        PHM_C01_MultiStateEnv_0p12; PHM_C04_MultiStateEnv_0p0975; PHM_C06_MultiStateEnv_0p13

ENVIRONMENT_INFO = 'PHM 2006. Single-var state V2.'
DATA_FILE = 'data\PHM_C06_MultiStateEnv_0p13.csv'
WEAR_THRESHOLD = 0.12 # mm
ADD_NOISE = 1e3 # 0 for no noise. Factor to apply on np.random.rand(). For e.g. 1e2 or 1e3 are factors for higher and lower noise. 
BREAKDOWN_CHANCE = 0.05 # Recommended: 0.05 = 5%
EPISODES = 1200 # Train for N episodes. # Suggested 600

## Read data
df = pd.read_csv(DATA_FILE)
n_records = len(df.index)
# MILLING_OPERATIONS_MAX = 800
MILLING_OPERATIONS_MAX = n_records-1 # Suggested 300
# MILLING_OPERATIONS_MAX = 800

if ADD_NOISE == 1e3 and BREAKDOWN_CHANCE == 0.05:
    lnoise = 'LowNBD'
elif ADD_NOISE == 1e2  and BREAKDOWN_CHANCE == 0.10:
    lnoise = 'HighNBD'
elif ADD_NOISE <= 0 and BREAKDOWN_CHANCE == 0:
    lnoise = 'NoNBD'
else:
    lnoise = 'ArbNBD'

VERSION = f'PHM-C06_{lnoise}_{WEAR_THRESHOLD}_{EPISODES}_{MILLING_OPERATIONS_MAX}'

METRICS_METHOD = 'weighted' # average method = {‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’} 
TEST_INFO = 'Sampled from training data'
TEST_CASES = 40
TEST_ROUNDS = 5
# Milling operation constants
WEAR_THRESHOLD_NORMALIZED = 0.0 # normalized to the max wear threshold

# Policy network learning parameters
gamma = 0.99
alpha = 0.01

RESULTS_FOLDER = 'results/18-May-2023'
CONSOLIDATED_METRICS_FILE = f'{RESULTS_FOLDER}/CONSOLIDATED_METRICS.csv'
RESULTS_FILE = f'{RESULTS_FOLDER}/{VERSION}_test_results_{dt_d}_{dt_m}.csv'
METRICS_FILE = f'{RESULTS_FOLDER}/{VERSION}_metrics.csv'

# RESULTS_FILE = f'results/13-May-2023/{VERSION}_test_results_{dt_d}-{dt_m}.csv'
# METRICS_FILE = f'results/13-May-2023/{VERSION}_metrics_{dt_d}-{dt_m}.csv'

print('\n -- Columns added to results file ', RESULTS_FILE)
results = ['Date', 'Time', 'Round', 'Environment', 'Training_data', 'Wear_Threshold', 'Test_data', 'Algorithm', 'Episodes', 'Normal_cases', 'Normal_error', 
           'Replace_cases', 'Replace_error', 'Overall_error', 
           'Wtd_Precision', 'Wtd_Recall', 'F_Beta_0_5', 'F_Beta_0_75', 'F_1_Score']
write_test_results(results, RESULTS_FILE)

# Normalizing entire df with min-max scaling
WEAR_MIN = df['tool_wear'].min() 
WEAR_MAX = df['tool_wear'].max()
WEAR_THRESHOLD_NORMALIZED = (WEAR_THRESHOLD-WEAR_MIN)/(WEAR_MAX-WEAR_MIN)
df_normalized = (df-df.min())/(df.max()-df.min())
df_normalized['ACTION_CODE'] = df['ACTION_CODE']
print(f'Tool wear data imported ({len(df.index)} records). WEAR_THRESHOLD_NORMALIZED: {WEAR_THRESHOLD_NORMALIZED:4.3f} \n\n')

# Visualize the data
# df.plot(figsize = (10, 6))

x = [n for n in range(n_records)]
y1 = df['tool_wear'].values.tolist()
y2 = df['ACTION_CODE'].values.tolist()
wear_plot = f'{RESULTS_FOLDER}/{VERSION}_wear_plot.png'
title=f'Tool Wear (mm) data\n{VERSION}'
two_axes_plot(x, y1, y2, title=title, x_label='Time', y1_label='Tool Wear (mm)', y2_label='Action code (1=Replace)', xticks=20, file=wear_plot, threshold=WEAR_THRESHOLD)

### Environment
env = MillingTool_SS_V2(df_normalized, WEAR_THRESHOLD_NORMALIZED, MILLING_OPERATIONS_MAX, ADD_NOISE, BREAKDOWN_CHANCE, 1.0, -1.0, -100.0)

### Main loop
rewards_history = []
loss_history = []
training_stats = []

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

agent_RF = Agent(input_dim, output_dim, alpha, gamma)

for episode in range(EPISODES):
    state = env.reset()

    # Sample a trajectory
    for t in range(MILLING_OPERATIONS_MAX): # Max. milling operations desired
        action = agent_RF.act(state)
        state, reward, done, info = env.step(action)
        agent_RF.rewards.append(reward)
        #env.render()
        if done:
            # print('** DONE **', info)
            break

    # Learn during this episode 
    loss = agent_RF.learn() # train per episode
    total_reward = sum(agent_RF.rewards)

    # Record statistics for this episode
    rewards_history.append(total_reward)
    loss_history.append(loss.item()) # Extract values from list of torch items for plotting

    # On-policy - so discard all data 
    agent_RF.onpolicy_reset()

    if (episode%100 == 0):
        # print(f'[{episode:04d}] Loss: {loss:>10.2f} | Reward: {total_reward:>10.2f} | Ep.length: {env.ep_length:04d}')
        print(f'[{episode:04d}] Loss: {loss:>10.2e} | Reward: {total_reward:>10.2e} | Ep.length: {env.ep_length:04d}')
        
x = [i for i in range(EPISODES)]

## Moving average for rewards
ma_window_size = 10
# # Convert error array to pandas series
rewards = pd.Series(rewards_history)
windows = rewards.rolling(ma_window_size)
moving_avg = windows.mean()
moving_avg_lst = moving_avg.tolist()
y1 = rewards
y2 = moving_avg_lst

filename = f'{RESULTS_FOLDER}/{VERSION}_Avg_episode_rewards.png'
two_variable_plot(x, y1, y2, 'Avg. rewards per episode', VERSION, 'Episode', 'Avg. Rewards', 'Moving Avg.', 50, filename)

# plot_error_bounds(x, y1)

filename = f'{RESULTS_FOLDER}/{VERSION}_Episode_Length.png'
single_axes_plot(x, env.ep_length_history, 'Episode length', VERSION, 'Episode', 'No of milling operations', 50, 0.0, filename)

filename = f'{RESULTS_FOLDER}/{VERSION}_Tool_Replacements.png' 
single_axes_plot(x, env.ep_tool_replaced_history, 'Tool replacements per episode', VERSION, 'Episode', 'Replacements', 50, 0.0, filename)

### TEST
idx_replace_cases = df.index[df['ACTION_CODE'] >= 1.0]
idx_normal_cases = df.index[df['ACTION_CODE'] < 1.0]

print('\n === REINFORCE model trained ===\n')
print(80*'-')
print(f'Algorithm\tNormal\terr.%\tReplace\terr.%\tOverall err.%')
print(80*'-')
for test_round in range(TEST_ROUNDS):
    # Create test cases
    idx_replace_cases = np.random.choice(idx_replace_cases, int(TEST_CASES/2), replace=False)
    idx_normal_cases = np.random.choice(idx_normal_cases, int(TEST_CASES/2), replace=False)
    test_cases = [*idx_normal_cases, *idx_replace_cases]
    
    results = test_script(METRICS_METHOD, test_round, df_normalized, 'REINFORCE', EPISODES, env, ENVIRONMENT_INFO, agent_RF, 
                          test_cases, TEST_INFO, DATA_FILE, WEAR_THRESHOLD, RESULTS_FILE)
    write_test_results(results, RESULTS_FILE)
    
print(f'\n- Test results written to file: {RESULTS_FILE}')

algos = ['A2C','DQN','PPO']

SB_agents = []

for SB_ALGO in algos:
    if SB_ALGO.upper() == 'A2C': agent_SB = A2C('MlpPolicy', env)
    if SB_ALGO.upper() == 'DQN': agent_SB = DQN('MlpPolicy', env)
    if SB_ALGO.upper() == 'PPO': agent_SB = PPO('MlpPolicy', env)
    
    print(f'\n{SB_ALGO} - Training and Testing Stable-Baselines-3 {SB_ALGO} algorithm')
    agent_SB.learn(total_timesteps=EPISODES)

    SB_agents.append(agent_SB)
    print(agent_SB)

n = 0
for agent_SB in SB_agents:
    print(f'\n\n - Testing Stable-Baselines-3 {agent_SB}')
    print(80*'-')
    print(f'Algo.\tNormal\tErr.%\tReplace\tErr.%\tOverall err.%')
    print(80*'-')
    for test_round in range(TEST_ROUNDS):
        # Create test cases
        idx_replace_cases = np.random.choice(idx_replace_cases, int(TEST_CASES/2), replace=False)
        idx_normal_cases = np.random.choice(idx_normal_cases, int(TEST_CASES/2), replace=False)
        test_cases = [*idx_normal_cases, *idx_replace_cases]
        results = test_script(METRICS_METHOD, test_round, df_normalized, algos[n], EPISODES, env, ENVIRONMENT_INFO, 
                              agent_SB, test_cases, TEST_INFO, DATA_FILE, WEAR_THRESHOLD, RESULTS_FILE)
        write_test_results(results, RESULTS_FILE)
    n += 1

### Create a consolidated algorithm wise metrics summary
print(80*'-', f'\n Algorithm level consolidated metrics being reported to file:\n {METRICS_FILE}\n', 80*'-')

header_columns = [VERSION]
write_test_results(header_columns, METRICS_FILE)
header_columns = ['Date', 'Time', 'Environment', 'Noise', 'Breakdown_chance', 'Train_data', 'Wear threshold', 'Episodes', 'Terminate on',
                  'Test_info', 'Test_cases', 'Metrics_method']
write_test_results(header_columns, METRICS_FILE)

dt_t = dt.strftime('%H:%M:%S')
noise_info = 'None' if ADD_NOISE == 0 else (1/ADD_NOISE)
header_info = [dt_d, dt_t, ENVIRONMENT_INFO, noise_info, BREAKDOWN_CHANCE, DATA_FILE, WEAR_THRESHOLD, EPISODES, MILLING_OPERATIONS_MAX, TEST_INFO, TEST_CASES, METRICS_METHOD]
write_test_results(header_info, METRICS_FILE)
write_test_results([], METRICS_FILE) # leave a blank line

print('- Experiment related meta info written')

df_algo_results = pd.read_csv(RESULTS_FILE)
# algo_metrics = compute_metrics_simple(df_algo_results)
algo_metrics = compute_metrics(df_algo_results)

write_metrics_report(algo_metrics, METRICS_FILE, 4)
write_test_results([], METRICS_FILE) # leave a blank line
print('- Algorithm level consolidated metrics reported to file')

## ------------------------------------------------------------------------------------------
write_test_results(header_columns, CONSOLIDATED_METRICS_FILE)
write_test_results(header_info, CONSOLIDATED_METRICS_FILE)
write_test_results([], CONSOLIDATED_METRICS_FILE) # leave a blank line
write_metrics_report(algo_metrics, CONSOLIDATED_METRICS_FILE, 4)
write_test_results([120*'-'], CONSOLIDATED_METRICS_FILE) # leave a blank line
print(f'- {CONSOLIDATED_METRICS_FILE} file updated')
print(algo_metrics.round(3))
print('\n\n ================= END OF PROGRAM =================')

