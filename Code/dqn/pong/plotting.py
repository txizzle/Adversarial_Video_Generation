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

tree_data = pickle.load(open('pong_tree_2800000_data.pkl','rb')) # tree
tree_t = tree_data['t_log']
tree_mean_rewards = tree_data['mean_reward_log']
tree_best_rewards = tree_data['best_mean_log']

tree2_data = pickle.load(open('pong_tree2_1000000_data.pkl','rb')) # tree w/ depth 2
tree2_t = tree2_data['t_log']
tree2_mean_rewards = tree2_data['mean_reward_log']
tree2_best_rewards = tree2_data['best_mean_log']

pixel_plot= plt.figure()
pixel_mean_rew, = plt.plot(pixel_t, pixel_mean_rewards, label='DQN Mean 100-Episode Reward')
pixel_best_rew, = plt.plot(pixel_t, pixel_best_rewards, label='DQN Best Mean Reward')
tree_mean, = plt.plot(tree_t, tree_mean_rewards, label='Tree d=1 Mean 100-Episode Reward')
tree_best, = plt.plot(tree_t, tree_best_rewards, label='Tree d=1 Best Mean Reward')
tree2_mean, = plt.plot(tree2_t, tree2_mean_rewards, label='Tree d=2 Mean 100-Episode Reward')
tree2_best, = plt.plot(tree2_t, tree2_best_rewards, label='Tree d=2 Best Mean Reward')
plt.suptitle('DQN vs. DQN + Tree Performance on Pong', fontsize=20)
plt.xlabel('Timesteps')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel('Reward')
plt.legend(loc=4)
pp = PdfPages('pong_plot.pdf')
pp.savefig(pixel_plot)
pp.close()

actions_plot = plt.figure()
diff_log = tree_data['diff']
same_log = tree_data['same']


same, = plt.plot(tree_t, same_log[1:], label='Same Actions')
diff, = plt.plot(tree_t, diff_log[1:], label='Different Actions')
plt.suptitle('DQN vs. DQN + Tree Actions', fontsize=20)
plt.xlabel('Timesteps')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylabel('Number of Actions')
plt.legend(loc=0)
pp = PdfPages('pong_actions_plot.pdf')
pp.savefig(actions_plot)
pp.close()