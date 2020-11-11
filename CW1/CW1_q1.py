
#### Question 1 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

from CW_classes import MC_World
from CW_classes import DP_Policy
from CW_classes import TD_World



print('Running CW1 q1')
full_start_time = time.time()
# Creating GridWorld
cid_nums = 379
x = int(str(cid_nums)[0])
y = int(str(cid_nums)[1])

p = 0.25 + 0.5 * (x+1)/10
gamma = 0.2 + 0.5 * y/10
threshold = 0.0001

# Value of trace






exit()

print('CW1 q2 misc graphs')
reward_state_rew = ((1, 2), 10)     # State 2
penalty_state_rew = ((4, 3), -100)  # State 11
absorbing_locs = [reward_state_rew[0], penalty_state_rew[0]]
special_rewards = [reward_state_rew[1], penalty_state_rew[1]]

# MC
epsilon_init = 0.01
alpha = 0.01
use_first_visit=True
visit_text = '(First-visit)' if use_first_visit else '(Every-visit)'
decay_alpha = False
repeats = 30

dp_world = DP_Policy(absorbing_locs=absorbing_locs, special_rewards=special_rewards, p_transition=p)
Policy = np.zeros((dp_world.state_size, dp_world.action_size))
policy_opt_DPpit, V_optimal_DPpit, epochs_DPpit = dp_world.policy_iteration(Policy, discount=gamma, threshold=threshold)
policy_opt_DPvit, V_optimal_DPvit, epochs_DPvit = dp_world.value_iteration(gamma, threshold)
dp_world.draw_value(V_optimal_DPvit, title=r'DP: Value, $\gamma$={} threshold={}'.format(gamma, threshold), save='DP_value_world.png')


rep_values_mc = np.load('/Users/oliviagallup/Desktop/Kode/Imperial/Y4_RL/MC_30_rep/rep_values_mc.npy')
rep_policies_mc = np.load('/Users/oliviagallup/Desktop/Kode/Imperial/Y4_RL/MC_30_rep/rep_policies_mc.npy')
rep_all_rmse_mc = np.load('/Users/oliviagallup/Desktop/Kode/Imperial/Y4_RL/MC_30_rep/rep_all_rmse_mc.npy')
rep_all_rmse_mc = np.concatenate((rep_all_rmse_mc, np.load('/Users/oliviagallup/Desktop/Kode/Imperial/Y4_RL/MC_30_reps_ii/rep_all_rmse_mc.npy')), axis=0)
rep_all_total_returns_mc = np.load('/Users/oliviagallup/Desktop/Kode/Imperial/Y4_RL/MC_30_rep/rep_all_total_returns_mc.npy')

rep_all_rmse_sarsa = np.load('/Users/oliviagallup/Desktop/Kode/Imperial/Y4_RL/TD_rep30/rep_all_rmse_sarsa.npy')
rep_all_total_returns_sarsa = np.load('/Users/oliviagallup/Desktop/Kode/Imperial/Y4_RL/TD_rep30/rep_all_total_returns_sarsa.npy')
rep_policies_sarsa = np.load('/Users/oliviagallup/Desktop/Kode/Imperial/Y4_RL/TD_rep30/rep_policies_sarsa.npy')
rep_values_sarsa = np.load('/Users/oliviagallup/Desktop/Kode/Imperial/Y4_RL/TD_rep30/rep_values_sarsa.npy')



mc_world = MC_World(absorbing_locs, special_rewards, p, use_first_visit=use_first_visit)

# MC Varying: moving average
alpha_collection = [0.01, 0.2, 0.4]
epsilon_collection = [0.01, 0.2, 0.8]
alphas = alpha_collection * len(epsilon_collection)
epsilon_inits = np.sort(epsilon_collection * len(alpha_collection))
N = 150
MA_all_total_returns_mc = [] 
for var in range(np.shape(rep_all_total_returns_mc)[0]):
    ma = []
    for rep in range(np.shape(rep_all_total_returns_mc)[1]):
        ma.append(np.convolve(rep_all_total_returns_mc[var][rep], np.ones((N,))/N, mode='valid'))
    MA_all_total_returns_mc.append(ma)
MA_rep_all_rmse_mc= [] 
for var in range(np.shape(rep_all_rmse_mc)[0]):
    ma = []
    for rep in range(np.shape(rep_all_rmse_mc)[1]):
        ma.append(np.convolve(rep_all_rmse_mc[var][rep], np.ones((N,))/N, mode='valid'))
    MA_rep_all_rmse_mc.append(ma)

titles = [r'$\alpha$={} and $\epsilon$={}'.format(a, e) for (a, e) in zip(alphas, epsilon_inits)]
for var in range(len(epsilon_collection)):
    s = len(epsilon_collection)
    mc_world.draw_learningcurve_repvars(MA_all_total_returns_mc[var*s:(var*s)+s], title_text=(r'MC Online: Discounted Returns varying $\alpha$ and $\epsilon$, {} reps & {} smooth'.format(repeats, N)), var_labels=titles[var*s:(var*s)+s], axislabels=('Episodes', 'Returns'), new_fig=False, save='MC_returns_e{}.png'.format(epsilon_collection[var]))
    mc_world.draw_learningcurve_repvars(MA_rep_all_rmse_mc[var*s:(var*s)+s], title_text=(r'MC Online: RMSE varying $\alpha$ and $\epsilon$, {} reps & {} smooth'.format(repeats, N)), var_labels=titles[var*s:(var*s)+s], axislabels=('Episodes', 'Root Mean Square Error'), new_fig=False, save='MC_rmse_e{}.png'.format(epsilon_collection[var]))



policy_mc = rep_policies_mc[0]
policy_mc = np.average(policy_mc, axis=0)
V_mc = rep_values_mc[0]
V_mc = np.average(V_mc, axis=0)
# mc_world.draw_stochastic_policy(rep_policies_mc[0][0], rep_values_mc[0][0], title=r'MC Online: Policy, 1st repeat {} $\alpha$={} $\epsilon$={}'.format(visit_text, alpha, epsilon_init), save='MC_policy_world_{}r.png'.format(1))
# mc_world.draw_value(rep_values_mc[0][0], title=r'MC Online Value, 1st repeat {} $\alpha$={} $\epsilon$={}'.format(visit_text, alpha, epsilon_init), save='MC_value_world_{}r.png'.format(1))

# mc_world.draw_stochastic_policy(policy_mc, V_mc, title=r'MC Online: Policy, {} repeats {} $\alpha$={} $\epsilon$={}'.format(repeats, visit_text, alpha, epsilon_init), save='MC_policy_world_{}r.png'.format(repeats))
# mc_world.draw_value(V_mc, title=r'MC Online Value, {} repeats {} $\alpha$={} $\epsilon$={}'.format(repeats, visit_text, alpha, epsilon_init), save='MC_value_world_{}r.png'.format(repeats))

td_world = TD_World(absorbing_locs, special_rewards, p, use_first_visit=use_first_visit)
td_world.draw_stochastic_policy(rep_policies_mc[0][0], rep_values_mc[0][0], title=r'MC Online: Policy, 1st repeat {} $\alpha$={} $\epsilon$={}'.format(visit_text, alpha, epsilon_init), save='MC_policy_world_{}r.png'.format(1))

# TD Varying: moving average
N = 150
MA_all_total_returns_sarsa = [] 
for var in range(np.shape(rep_all_total_returns_sarsa)[0]):
    ma = []
    for rep in range(np.shape(rep_all_total_returns_sarsa)[1]):
        ma.append(np.convolve(rep_all_total_returns_sarsa[var][rep], np.ones((N,))/N, mode='valid'))
    MA_all_total_returns_sarsa.append(ma)
MA_rep_all_rmse_sarsa = [] 
for var in range(np.shape(rep_all_rmse_sarsa)[0]):
    ma = []
    for rep in range(np.shape(rep_all_rmse_sarsa)[1]):
        ma.append(np.convolve(rep_all_rmse_sarsa[var][rep], np.ones((N,))/N, mode='valid'))
    MA_rep_all_rmse_sarsa.append(ma)

titles = [r'$\alpha$={} and $\epsilon$={}'.format(a, e) for (a, e) in zip(alphas, epsilon_inits)]
for var in range(len(epsilon_collection)):
    s = len(epsilon_collection)
    td_world.draw_learningcurve_repvars(MA_all_total_returns_sarsa[var*s:(var*s)+s], title_text=(r'TD SARSA: Discounted Returns varying $\alpha$ and $\epsilon$, {} reps & {} smooth'.format(repeats, N)), var_labels=titles[var*s:(var*s)+s], axislabels=('Episodes', 'Returns'), new_fig=False, save='TD-S_returns_e{}.png'.format(epsilon_collection[var]))
    td_world.draw_learningcurve_repvars(MA_rep_all_rmse_sarsa[var*s:(var*s)+s], title_text=(r'TD SARSA: RMSE varying $\alpha$ and $\epsilon$, {} reps & {} smooth'.format(repeats, N)), var_labels=titles[var*s:(var*s)+s], axislabels=('Episodes', 'Root Mean Square Error'), new_fig=False, save='TD-S_rmse_e{}.png'.format(epsilon_collection[var]))




################# Comparison of Learners #################

# Vars
labels_mc, labels_td = [], []
mean_rmse_mc, mean_returns_mc, std_rmse_mc, labels_mc = [], [], [], []
for i, (all_rmse_mc, all_total_returns_mc) in enumerate(zip(rep_all_rmse_mc, rep_all_total_returns_mc)):
    labels_mc.append(r'MC online $\alpha$={} $\epsilon$={}'.format(alphas[i], epsilon_inits[i]))
    # flat_rmse_mc = np.reshape(all_rmse_mc,(np.prod(np.shape(all_rmse_mc)[:-1]), np.shape(all_rmse_mc)[-1]))
    # flat_returns_mc = np.reshape(all_total_returns_mc, (np.prod(np.shape(all_total_returns_mc)[:-1]), np.shape(all_total_returns_mc)[-1]))
    mean_rmse_mc.append(np.mean(all_rmse_mc, axis=0))
    mean_returns_mc.append(np.mean(all_total_returns_mc, axis=0))
    std_rmse_mc.append(np.std(all_rmse_mc, axis=0))

mean_rmse_td, mean_returns_td, std_rmse_td, labels_td = [], [], [], []
for i, (all_rmse_td, all_total_returns_td) in enumerate(zip(rep_all_rmse_sarsa, rep_all_total_returns_sarsa)):
    labels_td.append(r'TD SARSA $\alpha$={} $\epsilon$={}'.format(alphas[i], epsilon_inits[i]))
    mean_rmse_td.append(np.mean(all_rmse_td, axis=0))
    mean_returns_td.append(np.mean(all_total_returns_td, axis=0))
    std_rmse_td.append(np.std(all_rmse_td, axis=0))

vmin_rmse = np.min(np.concatenate((rep_all_rmse_mc, rep_all_rmse_sarsa)))
vmin_returns = np.min(np.concatenate((rep_all_total_returns_mc, rep_all_total_returns_sarsa)))
vmax_rmse = np.max(np.concatenate((rep_all_rmse_mc, rep_all_rmse_sarsa)))
vmax_returns = np.max(np.concatenate((rep_all_total_returns_mc, rep_all_total_returns_sarsa)))

# vmin_rmse_mean = np.min(np.concatenate((mean_rmse_mc, mean_rmse_td)))
# vmin_returns_mean = np.min(np.concatenate((mean_returns_mc, mean_returns_td)))
# vmax_rmse_mean = np.max(np.concatenate((mean_rmse_mc, mean_rmse_td)))
# vmax_retruns_mean = np.max(np.concatenate((mean_returns_mc, mean_returns_td)))





print('2.e.2 Comparing MC and TD')
# Optimal value function estimation error
# vmin_rmse = np.min(np.concatenate((mean_rmse_mc, mean_rmse_td)))
# vmin_returns = np.min(np.concatenate((mean_returns_mc, mean_returns_td)))
# vmax_rmse = np.max(np.concatenate((mean_rmse_mc, mean_rmse_td)))
# vmax_retruns = np.max(np.concatenate((mean_returns_mc, mean_returns_td)))

N = 50
MA_all_total_returns_mc = [] 
for var in range(np.shape(rep_all_total_returns_mc)[0]):
    ma = []
    for rep in range(np.shape(rep_all_total_returns_mc)[1]):
        ma.append(np.convolve(rep_all_total_returns_mc[var][rep], np.ones((N,))/N, mode='valid'))
    MA_all_total_returns_mc.append(ma)
MA_rep_all_rmse_mc= [] 
for var in range(np.shape(rep_all_rmse_mc)[0]):
    ma = []
    for rep in range(np.shape(rep_all_rmse_mc)[1]):
        ma.append(np.convolve(rep_all_rmse_mc[var][rep], np.ones((N,))/N, mode='valid'))
    MA_rep_all_rmse_mc.append(ma)

titles = [r'$\alpha$={} and $\epsilon$={}'.format(a, e) for (a, e) in zip(alphas, epsilon_inits)]
for var in range(len(epsilon_collection)):
    s = len(epsilon_collection)
    mc_world.draw_learningcurve_repvars(MA_all_total_returns_mc[var*s:(var*s)+s], title_text=(r'MC Online: Discounted Returns varying $\alpha$ and $\epsilon$, {} reps & {} smooth'.format(repeats, N)), var_labels=titles[var*s:(var*s)+s], axislabels=('Episodes', 'Returns'), new_fig=False, save='MC_returns_e{}.png'.format(epsilon_collection[var]))
    mc_world.draw_learningcurve_repvars(MA_rep_all_rmse_mc[var*s:(var*s)+s], title_text=(r'MC Online: RMSE varying $\alpha$ and $\epsilon$, {} reps & {} smooth'.format(repeats, N)), var_labels=titles[var*s:(var*s)+s], axislabels=('Episodes', 'Root Mean Square Error'), new_fig=False, save='MC_rmse_e{}.png'.format(epsilon_collection[var]))


# MC
plt.figure()
for i in range(len(alpha_collection)):
    # plt.scatter(MA_all_total_returns_mc[i], MA_rep_all_rmse_mc[i], label=labels_mc[i])
    # plt.scatter(rep_all_total_returns_mc[i], rep_all_rmse_mc[i], label=labels_mc[i])
    plt.scatter(mean_returns_mc[i], mean_rmse_mc[i], label=labels_mc[i])
    # plt.scatter(all_total_returns_mc[i], all_rmse_mc[i], label=labels_mc[i])
# plt.xlim([vmin_returns,vmax_returns])
# plt.ylim([vmin_rmse,vmax_rmse])
plt.xlabel('Returns')
plt.ylabel('RMSE')
plt.legend(loc='lower left')
plt.title('MC: RMSE vs Returns per episode, 30 repeats')
plt.savefig('Comparison_MC.png')
# plt.show()

# TD
plt.figure()
for i in range(len(alpha_collection)):
    # plt.scatter(MA_all_total_returns_sarsa[i], MA_rep_all_rmse_sarsa[i], label=labels_td[i])
    # plt.scatter(rep_all_total_returns_sarsa[i], rep_all_rmse_sarsa[i], label=labels_td[i])
    plt.scatter(mean_returns_td[i], mean_rmse_td[i], label=labels_td[i])
    # plt.scatter(all_total_returns_td[i], all_rmse_td[i], label=labels_td[i])
# plt.xlim([vmin_returns,vmax_returns])
# plt.ylim([vmin_rmse,vmax_rmse])
plt.xlabel('Returns')
plt.ylabel('RMSE')
plt.legend(loc='lower right')
plt.title('TD SARSA: RMSE vs Returns per episode, 30 repeats')
plt.savefig('Comparison_TD.png')
# plt.show()

# BOTH
plt.figure()
for i in range(len(alpha_collection)):
    # plt.scatter(all_total_returns_mc[i], all_rmse_mc[i], label=labels_mc[i])
    # plt.scatter(all_total_returns_td[i], all_rmse_td[i], label=labels_td[i])
    plt.scatter(mean_returns_mc[i], mean_rmse_mc[i], label=labels_mc[i], alpha=0.6)
    plt.scatter(mean_returns_td[i], mean_rmse_td[i], label=labels_td[i], alpha=0.6)
# plt.xlim([vmin_returns,vmax_returns])
# plt.ylim([vmin_rmse,vmax_rmse])
plt.xlabel('Returns')
plt.ylabel('RMSE')
plt.legend(bbox_to_anchor=(1.5, 1), loc='upper right')
plt.title('MC and TD SARSA: RMSE vs Returns per episode, 30 repeats')
plt.savefig('Comparison_MC_TD.png', bbox_inches='tight')


# for 

# np.save('all_rmse_mc.npy', all_rmse_mc)
# np.save('all_total_returns_mc.npy', all_total_returns_mc)
# np.save('all_rmse_td.npy', all_rmse_td)
# np.save('all_total_returns_td.npy', all_total_returns_td)


# MA and mean
labels_mc, labels_td = [], []
mean_rmse_mc, mean_returns_mc, std_rmse_mc, labels_mc = [], [], [], []
for i, (all_rmse_mc, all_total_returns_mc) in enumerate(zip(MA_rep_all_rmse_mc, MA_all_total_returns_mc)):
    labels_mc.append(r'MC online $\alpha$={} $\epsilon$={}'.format(alphas[i], epsilon_inits[i]))
    # flat_rmse_mc = np.reshape(all_rmse_mc,(np.prod(np.shape(all_rmse_mc)[:-1]), np.shape(all_rmse_mc)[-1]))
    # flat_returns_mc = np.reshape(all_total_returns_mc, (np.prod(np.shape(all_total_returns_mc)[:-1]), np.shape(all_total_returns_mc)[-1]))
    mean_rmse_mc.append(np.mean(all_rmse_mc, axis=0))
    mean_returns_mc.append(np.mean(all_total_returns_mc, axis=0))
    std_rmse_mc.append(np.std(all_rmse_mc, axis=0))

mean_rmse_td, mean_returns_td, std_rmse_td, labels_td = [], [], [], []
for i, (all_rmse_td, all_total_returns_td) in enumerate(zip(MA_rep_all_rmse_sarsa, MA_all_total_returns_sarsa)):
    labels_td.append(r'TD SARSA $\alpha$={} $\epsilon$={}'.format(alphas[i], epsilon_inits[i]))
    mean_rmse_td.append(np.mean(all_rmse_td, axis=0))
    mean_returns_td.append(np.mean(all_total_returns_td, axis=0))
    std_rmse_td.append(np.std(all_rmse_td, axis=0))


print('2.e.1 RMSE for MC and TD')
# Estimation error v episodes
plt.figure()  # figsize=(5,8)
for i in range(len(alpha_collection)):
    plt.plot(mean_rmse_mc[i], label=labels_mc[i])
    plt.fill_between(list(range(len(mean_rmse_mc[i]))), (mean_rmse_mc[i] - std_rmse_mc[i]), (mean_rmse_mc[i] + std_rmse_mc[i]), alpha=0.6)
    plt.plot(mean_rmse_td[i], label=labels_td[i])
    plt.fill_between(list(range(len(mean_rmse_td[i]))), (mean_rmse_td[i] - std_rmse_td[i]), (mean_rmse_td[i] + std_rmse_td[i]), alpha=0.6)
plt.legend(bbox_to_anchor=(1.5, 1), loc='upper right', borderaxespad=0.)
plt.xlabel('Episodes')
plt.ylabel('RMSE')
plt.title('Comparison of RMSE between MC and TD, 30 repeats')
plt.savefig('Comparison_learningcurve.png', bbox_inches='tight')
# plt.show()

print('Everything took ', time.time() - full_start_time)
