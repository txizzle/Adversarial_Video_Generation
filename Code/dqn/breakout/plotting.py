import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle

# Plotting for performance on Breakout
pixel_data = pickle.load(open('breakout_dqn_2000000_data.pkl','rb'))
pixel_t = pixel_data['t_log']
pixel_mean_rewards = pixel_data['mean_reward_log']
pixel_best_rewards = pixel_data['best_mean_log']

lookahead_data = pickle.load(open('breakout_tree_1400000_data.pkl','rb')) # tree
lookahead_t = lookahead_data['t_log']
lookahead_mean_rewards = lookahead_data['mean_reward_log']
lookahead_best_rewards = lookahead_data['best_mean_log']

pixel_plot= plt.figure()
pixel_mean_rew, = plt.plot(pixel_t, pixel_mean_rewards, label='DQN Mean 100-Ep Reward')
pixel_best_rew, = plt.plot(pixel_t, pixel_best_rewards, label='DQN Best Mean Reward')
la_mean_rew, = plt.plot(lookahead_t, lookahead_mean_rewards, label='Tree Mean 100-Ep Reward')
la_best_rew, = plt.plot(lookahead_t, lookahead_best_rewards, label='Tree Best Mean Reward')
plt.suptitle('DQN vs. DQN + Tree Performance on Breakout', fontsize=20)
plt.xlabel('Timesteps')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel('Reward')
plt.legend(loc=0)
# plt.show()
pp = PdfPages('breakout_plot.pdf')
pp.savefig(pixel_plot)
pp.close()