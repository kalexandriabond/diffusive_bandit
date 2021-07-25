import pandas as pd; import numpy as np; import matplotlib.pyplot as plt;
import seaborn as sns

n_trials = 1000

p_opt = 0.85; p_subopt = 1 - p_opt
reward_values = [1, 0]
trials = np.arange(n_trials)

lambda_param = 30


n_poisson_samples = n_trials // lambda_param

epoch_lengths = np.random.poisson(lambda_param,
size=n_poisson_samples)

change_point_idx = np.cumsum(epoch_lengths)


if change_point_idx[-1] > n_trials:
    change_point_idx = change_point_idx[:-1]

epoch_lengths = np.append(epoch_lengths, n_trials-change_point_idx[-1])

change_points = np.zeros_like(trials)
change_points[change_point_idx] = 1

epoch_dfs = []
target_identities = [0,1]


for idx, epoch_len in enumerate(epoch_lengths):

    trials = np.arange(epoch_len)
    reward_values.reverse()
    target_identities.reverse()

    probability_df = pd.DataFrame({'trial': trials,
    'reward_target_0': np.nan, 'reward_target_1': np.nan,
     'optimal_target_identity': np.nan})

    optimal_target_reward = np.random.choice(reward_values,
    epoch_len, p=[p_opt, p_subopt])

    suboptimal_target_reward = np.random.choice(reward_values,
    epoch_len, p=[p_subopt, p_opt])

    probability_df[['reward_target_0']] = optimal_target_reward
    probability_df[['reward_target_1']] = suboptimal_target_reward

    probability_df['cum_prob_target_0'] = np.cumsum(probability_df.reward_target_0) / n_trials
    probability_df['cum_prob_target_1'] = np.cumsum(probability_df.reward_target_1) / n_trials

    probability_df['optimal_target_identity'] = target_identities[0]

    print(probability_df.cum_prob_target_0, probability_df.optimal_target_identity)

    epoch_dfs.append(probability_df)

alpha = 0.1


plt.scatter(probability_df.trial, probability_df.reward_target_0,
color='green', alpha=alpha)
plt.scatter(probability_df.trial, probability_df.reward_target_1,
color='red', alpha=alpha)

plt.plot(probability_df.trial, probability_df.cum_prob_target_0, color='green')
plt.plot(probability_df.trial, probability_df.cum_prob_target_1, color='red')

sns.despine()
plt.show()
