#!/usr/bin/env python
# coding: utf-8

# ## Milling Tool Wear Maintenance Policy using the REINFORCE algorithm
# Ver.10.0 Auto Experiment
print ('\n ====== REINFORCE for Predictive Maintenance. Automated Experiments V.10. ====== \n')
print ('- Loading packages...')
import datetime
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C, PPO, DQN

import milling_tool_environment
from milling_tool_environment import MillingTool_SS_V3, MillingTool_MS_V3
from utilities import compute_metrics, compute_metrics_simple, write_metrics_report, store_results, plot_learning_curve, single_axes_plot, lnoise
import utilities
from utilities import two_axes_plot, two_variable_plot, plot_error_bounds, test_script, write_test_results, downsample
from reinforce_classes import PolicyNetwork, Agent

# Auto experiment file structure
print ('- Loading Experiments...')
df_expts = pd.read_csv('Experiments.csv')
n_expts = len(df_expts.index)

for n_expt in range(n_expts):
    dt = datetime.datetime.now()
    dt_d = dt.strftime('%d-%b-%Y')
    dt_t = dt.strftime('%H_%M_%S')
    dt_m = dt.strftime('%H%M')

    # Load experiment parameters
    ENVIRONMENT_INFO = df_expts['environment'][n_expt]
    DATA_FILE = df_expts['data_file'][n_expt]
    R1 = df_expts['R1'][n_expt]
    R2 = df_expts['R2'][n_expt]
    R3 = df_expts['R3'][n_expt]
    WEAR_THRESHOLD = df_expts['wear_threshold'][n_expt]
    THRESHOLD_FACTOR = df_expts['threshold_factor'][n_expt]
    ADD_NOISE = df_expts['add_noise'][n_expt]
    BREAKDOWN_CHANCE = df_expts['breakdown_chance'][n_expt]
    EPISODES = df_expts['episodes'][n_expt]
    MILLING_OPERATIONS_MAX = df_expts['milling_operations_max'][n_expt]
    ver_prefix = df_expts['version_prefix'][n_expt]
    TEST_INFO = df_expts['test_info'][n_expt]
    TEST_CASES = df_expts['test_cases'][n_expt]
    TEST_ROUNDS = df_expts['test_rounds'][n_expt]
    RESULTS_FOLDER = df_expts['results_folder'][n_expt]

    ## Read data
    df = pd.read_csv(DATA_FILE)
    n_records = len(df.index)
    VERSION = f'{ver_prefix}_{lnoise(ADD_NOISE, BREAKDOWN_CHANCE)}_{WEAR_THRESHOLD}_{THRESHOLD_FACTOR}_{R3}_{EPISODES}_{MILLING_OPERATIONS_MAX}_'
    print(f'\n [{dt_t}] Experiment {n_expt}: {VERSION}')

    METRICS_METHOD = 'binary' # average method = {‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’}
    WEAR_THRESHOLD_NORMALIZED = 0.0 # normalized to the max wear threshold

    # Policy network learning parameters
    gamma = 0.99
    alpha = 0.01

    CONSOLIDATED_METRICS_FILE = f'{RESULTS_FOLDER}/TEST_CONSOLIDATED_METRICS.csv'
    RESULTS_FILE = f'{RESULTS_FOLDER}/{VERSION}_test_results_{dt_d}_{dt_m}.csv'
    METRICS_FILE = f'{RESULTS_FOLDER}/{VERSION}_metrics.csv'

    print('\n- Columns added to results file: ', RESULTS_FILE)
    results = ['Date', 'Time', 'Round', 'Environment', 'Training_data', 'Wear_Threshold', 'Test_data', 'Algorithm', 'Episodes', 'Normal_cases', 'Normal_error',
               'Replace_cases', 'Replace_error', 'Overall_error',
               'Precision', 'Recall', 'F_Beta_0_5', 'F_Beta_0_75', 'F_1_Score']
    write_test_results(results, RESULTS_FILE)


    # ## Data pre-process
    # 1. Add noise
    # 2. Add ACTION_CODE based on tool wear threshold
    # 3. Normalize data base
    # 4. Split into train and test

    # 1. Add noise
    if ADD_NOISE:
        df['tool_wear'] += np.random.normal(0, 1, n_records)/ADD_NOISE

    # 2. Add ACTION code
    df['ACTION_CODE'] = np.where(df['tool_wear'] < WEAR_THRESHOLD, 0.0, 1.0)

    # 3. Normalize
    WEAR_MIN = df['tool_wear'].min()
    WEAR_MAX = df['tool_wear'].max()
    WEAR_THRESHOLD_NORMALIZED = THRESHOLD_FACTOR*(WEAR_THRESHOLD-WEAR_MIN)/(WEAR_MAX-WEAR_MIN)
    df_normalized = (df-df.min())/(df.max()-df.min())
    df_normalized['ACTION_CODE'] = df['ACTION_CODE']
    print(f'- Tool wear data imported ({len(df.index)} records).')

    # 4. Split into train and test
    df_train = downsample(df_normalized, 100)
    df_train.to_csv('TempTrain.csv')
    df_train = pd.read_csv('TempTrain.csv')

    df_test = downsample(df_normalized, 70)
    df_test.to_csv('TempTest.csv')
    df_test = pd.read_csv('TempTest.csv')

    print(f'- Tool wear data split into train ({len(df_train.index)} records) and test ({len(df_test.index)} records).')

    n_records = len(df_train.index)
    x = [n for n in range(n_records)]
    y1 = df_train['tool_wear']
    y2 = df_train['ACTION_CODE']
    wear_plot = f'{RESULTS_FOLDER}/{VERSION}_wear_plot.png'
    title=f'Tool Wear (mm) data\n{VERSION}'
    two_axes_plot(x, y1, y2, title=title, x_label='Time', y1_label='Tool Wear (mm)', y2_label='Action code (1=Replace)', xticks=20, file=wear_plot, threshold=WEAR_THRESHOLD_NORMALIZED)


    # ## Milling Tool Environment -
    # 1. MillingTool_SS: Single state: tool_wear and time
    # 2. MillingTool_MS: Multie-state: force_x; force_y; force_z; vibration_x; vibration_y; vibration_z; acoustic_emission_rms; tool_wear
    # - Note: ACTION_CODE is only used for evaluation later (testing phase) and is NOT passed as part of the environment states

    env = MillingTool_MS_V3(df_train, WEAR_THRESHOLD_NORMALIZED, MILLING_OPERATIONS_MAX, ADD_NOISE, BREAKDOWN_CHANCE, R1, R2, R3)
    env_test = MillingTool_MS_V3(df_test, WEAR_THRESHOLD_NORMALIZED, MILLING_OPERATIONS_MAX, ADD_NOISE, BREAKDOWN_CHANCE, R1, R2, R3)

    # ## REINFORCE RL Algorithm
    ### Main loop
    print('\n* Train REINFORCE model...')
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

    # ### Generate a balanced test set
    idx_replace_cases = df_test.index[df_test['ACTION_CODE'] >= 1.0]
    idx_normal_cases = df_test.index[df_test['ACTION_CODE'] < 1.0]

    # Process results
    # eps = [i for i in range(EPISODES)]
    # store_results(RF_TRAINING_FILE, training_round, eps, rewards_history, env.ep_tool_replaced_history)
    print('- Test REINFORCE model...')
    # print(80*'-')
    # print(f'Algorithm\tNormal\terr.%\tReplace\terr.%\tOverall err.%')
    # print(80*'-')
    for test_round in range(TEST_ROUNDS):
        # Create test cases
        idx_replace_cases = np.random.choice(idx_replace_cases, int(TEST_CASES/2), replace=False)
        idx_normal_cases = np.random.choice(idx_normal_cases, int(TEST_CASES/2), replace=False)
        test_cases = [*idx_normal_cases, *idx_replace_cases]

        results = test_script(METRICS_METHOD, test_round, df_test, 'REINFORCE', EPISODES, env_test, ENVIRONMENT_INFO, agent_RF,
                              test_cases, TEST_INFO, DATA_FILE, WEAR_THRESHOLD, RESULTS_FILE)
        write_test_results(results, RESULTS_FILE)

    print(f'- REINFORCE Test results written to file: {RESULTS_FILE}.\n')

    # ## Stable-Baselines Algorithms
    print('\n* Train Stable-Baselines-3 A2C, DQN and PPO models...')

    algos = ['A2C','DQN','PPO']
    SB_agents = []

    for SB_ALGO in algos:
        if SB_ALGO.upper() == 'A2C': agent_SB = A2C('MlpPolicy', env)
        if SB_ALGO.upper() == 'DQN': agent_SB = DQN('MlpPolicy', env)
        if SB_ALGO.upper() == 'PPO': agent_SB = PPO('MlpPolicy', env)

        print(f'- Training Stable-Baselines-3 {SB_ALGO} algorithm...')
        agent_SB.learn(total_timesteps=EPISODES)
        SB_agents.append(agent_SB)

    n = 0
    for agent_SB in SB_agents:
        print(f'- Testing Stable-Baselines-3 {SB_ALGO} model...')
        # print(80*'-')
        # print(f'Algo.\tNormal\tErr.%\tReplace\tErr.%\tOverall err.%')
        # print(80*'-')
        for test_round in range(TEST_ROUNDS):
            # Create test cases
            idx_replace_cases = np.random.choice(idx_replace_cases, int(TEST_CASES/2), replace=False)
            idx_normal_cases = np.random.choice(idx_normal_cases, int(TEST_CASES/2), replace=False)
            test_cases = [*idx_normal_cases, *idx_replace_cases]
            results = test_script(METRICS_METHOD, test_round, df_test, algos[n], EPISODES, env_test, ENVIRONMENT_INFO,
                                  agent_SB, test_cases, TEST_INFO, DATA_FILE, WEAR_THRESHOLD, RESULTS_FILE)
            write_test_results(results, RESULTS_FILE)
        n += 1

    ### Create a consolidated algorithm wise metrics summary

    print(f'* Test Report: Algorithm level consolidated metrics will be written to: {METRICS_FILE}.')

    header_columns = [VERSION]
    write_test_results(header_columns, METRICS_FILE)
    header_columns = ['Date', 'Time', 'Environment', 'Noise', 'Breakdown_chance', 'Train_data', 'env.R1', 'env.R2', 'env.R3', 'Wear threshold', 'Look-ahead Factor', 'Episodes', 'Terminate on', 'Test_info', 'Test_cases', 'Metrics_method', 'Version']
    write_test_results(header_columns, METRICS_FILE)

    dt_t = dt.strftime('%H:%M:%S')
    noise_info = 'None' if ADD_NOISE == 0 else (1/ADD_NOISE)
    header_info = [dt_d, dt_t, ENVIRONMENT_INFO, noise_info, BREAKDOWN_CHANCE, DATA_FILE, env.R1, env.R2, env.R3, WEAR_THRESHOLD, THRESHOLD_FACTOR, EPISODES, MILLING_OPERATIONS_MAX, TEST_INFO, TEST_CASES, METRICS_METHOD, VERSION]
    write_test_results(header_info, METRICS_FILE)
    write_test_results([], METRICS_FILE) # leave a blank line

    print('- Experiment related meta info written.')

    df_algo_results = pd.read_csv(RESULTS_FILE)
    # algo_metrics = compute_metrics_simple(df_algo_results)
    algo_metrics = compute_metrics(df_algo_results)

    write_metrics_report(algo_metrics, METRICS_FILE, 4)
    write_test_results([], METRICS_FILE) # leave a blank line
    print('- Algorithm level consolidated metrics reported to file.')

    write_test_results(header_columns, CONSOLIDATED_METRICS_FILE)
    write_test_results(header_info, CONSOLIDATED_METRICS_FILE)
    write_test_results([], CONSOLIDATED_METRICS_FILE) # leave a blank line
    write_metrics_report(algo_metrics, CONSOLIDATED_METRICS_FILE, 4)
    write_test_results([120*'-'], CONSOLIDATED_METRICS_FILE) # leave a blank line
    print(f'- {CONSOLIDATED_METRICS_FILE} file updated.')
    print(algo_metrics.round(3))
    print(f'- End Experiment {n_expt}.\n')

print('\n\n ================= END OF PROGRAM =================')
