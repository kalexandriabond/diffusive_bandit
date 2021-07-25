# method from Daw 2006
# written by KB


# The slots paid off points (to be exchanged for money) noisily around four different means.
# Unlike standard slots, the mean payoffs changed randomly and independently from trial to trial,
# with subjects finding information about the current worth of a slot only through sampling it actively.
# This feature of the experimental design, together with a model-based analysis,
# allowed us to study exploratory and exploitative decisions under uniform conditions, in the context of a single task.

import numpy as np; import matplotlib.pyplot as plt; import seaborn as sns
import pandas as pd; import os; from scipy import stats; from glob import glob

home = os.path.join(os.path.expanduser('~'), 'Documents/diffusive_bandit')


sub = 6
n_bandit_arms = 2
trial_duration = 1.5

n_runs = 2
max_run_duration = 8.92 # in m, allowing 5 s for HRF after final trial

n_trials = 600
final_n_trials = 60

# n_trials = 1000


# decay_parameter_min = 0.95
# decay_parameter_max = 0.9836
# step = 0.01
decay_parameter = 0.98

diffusion_noise_mean, diffusion_noise_sigma = 0, 6 # manipulate diffusion noise sigma as proxy for conflict. sigma should be relative to point range.
theta = 50

initial_mu_arm1, initial_mu_arm2 = 20, 80

f_image_dir = glob(os.path.join(home, 'images/symm_greebles/f*.tif'))
m_image_dir = glob(os.path.join(home, 'images/symm_greebles/m*.tif'))

f_image_names = [os.path.basename(f_n)for f_n in f_image_dir]
m_image_names = [os.path.basename(m_n)for m_n in m_image_dir]


f_image_list = np.random.choice(f_image_names, final_n_trials); m_image_list = np.random.choice(m_image_names, final_n_trials)



# point_range = np.arange(start=0, stop=101)

point_range = np.arange(start=0, stop=11)
point_sigma = 4

def gen_itis(iti_min=4, iti_max=10, rate_param=2.8, n_trials=n_trials):
    lower, upper, scale = iti_min, iti_max, rate_param
    X = stats.truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)
    ITIs = X.rvs(n_trials)
    return ITIs

def generate_decaying_gaussian_random_walk(n_trials, decay_parameter, diffusion_noise_mean, theta,
diffusion_noise_sigma, point_range, initial_mu, scaling_factor=10):


    initial_mu_sample = np.random.normal(loc=initial_mu, scale=point_sigma)

    trialwise_mus = []
    trials = np.arange(n_trials)

    for i in trials:

        diffusion_noise = np.random.normal(diffusion_noise_mean, diffusion_noise_sigma)

        if i == 0:
            mu_i = (decay_parameter*initial_mu_sample) + (1-decay_parameter)*theta + diffusion_noise
        else:
            mu_i = (decay_parameter*trialwise_mus[-1]) + (1-decay_parameter)*theta + diffusion_noise

        mu_i_rounded = int(np.round(mu_i, decimals=0))
        trialwise_mus.append(mu_i_rounded)

    return trials, trialwise_mus

trialwise_mus = np.array([-1])
run_duration = max_run_duration + 1

# while (np.sum(np.sign(trialwise_mus) == 1) != len(trialwise_mus)) or (run_duration > max_run_duration):  # make sure values are positive
while (run_duration > max_run_duration):  # make sure values are positive

    for run in np.arange(n_runs)+1:

        bandit_arm_point_dfs = []

        for arm in np.arange(n_bandit_arms).astype(int):

            if arm == 0:
                identity = 'm'
                initial_mu = initial_mu_arm1
            elif arm == 1:
                identity = 'f'
                initial_mu = initial_mu_arm2

            diffusion_parameters = {'n_trials': n_trials, 'decay_parameter': decay_parameter,
            'diffusion_noise_mean': diffusion_noise_mean, 'theta': theta,
            'diffusion_noise_sigma': diffusion_noise_sigma,
            'point_range': point_range, 'initial_mu': initial_mu}

            trials, trialwise_mus = generate_decaying_gaussian_random_walk(**diffusion_parameters)

            trialwise_mu_subsample = trialwise_mus[::10]  # subsample. too lazy to scale pattern.
            trials = np.arange(len(trialwise_mu_subsample))
            bandit_arm_point_df = pd.DataFrame({'trial': trials, 'mu': trialwise_mu_subsample})
            bandit_arm_point_df['identity'] = identity

            bandit_arm_point_dfs.append(bandit_arm_point_df)


        all_bandit_arm_points_df = pd.concat(bandit_arm_point_dfs)

        all_bandit_arm_points_df['mu'] = all_bandit_arm_points_df.mu

        all_bandit_arm_points_df_unmelted = all_bandit_arm_points_df.pivot(index='trial',columns=['identity'], values=['mu']).reset_index(drop=True)

        level_one, level_two = all_bandit_arm_points_df_unmelted.columns.get_level_values(0).astype(str), all_bandit_arm_points_df_unmelted.columns.get_level_values(1).astype(str)
        all_bandit_arm_points_df_unmelted.columns = level_one + level_two

        all_bandit_arm_points_df_unmelted.columns = ['reward_f', 'reward_m']
        all_bandit_arm_points_df_unmelted[['reward_f', 'reward_m']] = all_bandit_arm_points_df_unmelted[['reward_f', 'reward_m']] // 10
        all_bandit_arm_points_df_unmelted['trial'] = range(len(all_bandit_arm_points_df_unmelted))
        all_bandit_arm_points_df_unmelted['run'] = run

        all_bandit_arm_points_df_unmelted['theta'] = theta
        all_bandit_arm_points_df_unmelted['decay_parameter'] = decay_parameter
        all_bandit_arm_points_df_unmelted['diffusion_noise_sigma'] = diffusion_noise_sigma
        all_bandit_arm_points_df_unmelted['diffusion_noise_mean'] = diffusion_noise_mean

        all_bandit_arm_points_df_unmelted['reward_delta'] = all_bandit_arm_points_df_unmelted.reward_m - all_bandit_arm_points_df_unmelted.reward_f
        all_bandit_arm_points_df_unmelted['high_value_identity'] = all_bandit_arm_points_df_unmelted[['reward_f', 'reward_m']].idxmax(axis=1)
        all_bandit_arm_points_df_unmelted['high_value_identity'] = all_bandit_arm_points_df_unmelted.high_value_identity.str.split('reward_', expand=True)[1]


        all_bandit_arm_points_df_unmelted['m_image'] = m_image_list
        all_bandit_arm_points_df_unmelted['f_image'] = f_image_list

        all_bandit_arm_points_df_unmelted['ITI'] = gen_itis()[::10]


        run_duration = ((all_bandit_arm_points_df_unmelted.ITI.sum() + (len(all_bandit_arm_points_df)*trial_duration))/60)

        print('Timing of run in minutes', run_duration)


        all_bandit_arm_points_df.mu = all_bandit_arm_points_df.mu // 10

        plt.figure(figsize=(10, 3))
        # sns.set_context('poster')
        sns.lineplot(x = 'trial', y = 'mu', hue = 'identity',
        data = all_bandit_arm_points_df)
        plt.ylabel('mean payoff')
        plt.title('n_arms={}, decay_center={}, diffusion_noise_sigma={}, decay_param={}, run={}'.format(n_bandit_arms,
         theta, diffusion_noise_sigma, decay_parameter, run))
        sns.despine()
        # plt.vlines(x=trials[::theta], ymin=point_range.min(), ymax=point_range.max())
        plt.tight_layout()
        plt.savefig(os.path.join(home, 'experimental_parameters/reward_schedule_figs/sub{}_reward_schedule_diffusion{}_theta{}_decay{}_run{}.png').format(sub, diffusion_noise_sigma, theta, decay_parameter, run))
        plt.show()

        plt.figure(figsize=(10, 3))
        # sns.set_context('poster')
        sns.lineplot(x = 'trial', y = 'reward_delta', data=all_bandit_arm_points_df_unmelted)
        plt.ylabel(r'$\Delta$ mean payoff (r(M) - r(F))')
        plt.title('n_arms={}, decay_center={}, diffusion_noise_sigma={}, decay_param={}, run={}'.format(n_bandit_arms,
         theta, diffusion_noise_sigma, decay_parameter, run))
        sns.despine()
        # plt.vlines(x=trials[::theta], ymin=point_range.min(), ymax=point_range.max())
        plt.tight_layout()
        plt.savefig(os.path.join(home, 'experimental_parameters/reward_schedule_figs/sub{}_delta_reward_schedule_diffusion{}_theta{}_decay{}_run{}.png').format(sub, diffusion_noise_sigma, theta, decay_parameter, run))
        plt.show()

        all_bandit_arm_points_df_unmelted.to_csv(os.path.join(home,
        'experimental_parameters/reward_parameters/sub{}_diffusive_bandit_run{}.csv').format(sub, run),
        index=False)
