import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from operator import truediv

# Plotting for performance on Pong
pixel_data = pickle.load(open('pong_dqn_5000000_data.pkl','rb'))
pixel_t = pixel_data['t_log']
pixel_mean_rewards = pixel_data['mean_reward_log']
pixel_best_rewards = pixel_data['best_mean_log']

lookahead_data = pickle.load(open('pong_tree2_200000_data.pkl','rb')) # tree
lookahead_t = lookahead_data['t_log']
lookahead_mean_rewards = lookahead_data['mean_reward_log']
lookahead_best_rewards = lookahead_data['best_mean_log']

pixel_plot= plt.figure()
pixel_mean_rew, = plt.plot(pixel_t, pixel_mean_rewards, label='DQN Mean 100-Episode Reward')
pixel_best_rew, = plt.plot(pixel_t, pixel_best_rewards, label='DQN Best Mean Reward')
la_mean_rew, = plt.plot(lookahead_t, lookahead_mean_rewards, label='Tree Mean 100-Episode Reward')
la_best_rew, = plt.plot(lookahead_t, lookahead_best_rewards, label='Tree Best Mean Reward')
plt.suptitle('DQN vs. DQN + Tree Performance on Pong', fontsize=20)
plt.xlabel('Timesteps')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel('Reward')
plt.legend(loc=4)
pp = PdfPages('pong_plot.pdf')
pp.savefig(pixel_plot)
pp.close()

actions_plot = plt.figure()
diff_log = lookahead_data['diff']
same_log = lookahead_data['same']


same, = plt.plot(lookahead_t, same_log[1:], label='Same Actions')
diff, = plt.plot(lookahead_t, diff_log[1:], label='Different Actions')
plt.suptitle('DQN vs. DQN + Tree Actions', fontsize=20)
plt.xlabel('Timesteps')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel('Number of Actions')
plt.legend(loc=0)
pp = PdfPages('pong_actions_plot.pdf')
pp.savefig(actions_plot)
pp.close()