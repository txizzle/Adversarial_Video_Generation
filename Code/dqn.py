from __future__ import division, print_function, unicode_literals
from avg_runner import AVGRunner

import utils
import constants as c
import matplotlib.pyplot as plt

# Handle arguments (before slow imports so --help can be fast)
import argparse
parser = argparse.ArgumentParser(description="Train a DQN net to play MsMacman.")
parser.add_argument("-n", "--number-steps", type=int, help="total number of training steps", default=10000)
parser.add_argument("-l", "--learn-iterations", type=int, help="number of game iterations between each training step", default=3)
parser.add_argument("-s", "--save-steps", type=int, help="number of training steps between saving each checkpoint", default=50)
parser.add_argument("-c", "--copy-steps", type=int, help="number of training steps between each copy of the critic to the actor", default=25)
parser.add_argument("-m", "--model-path", help="load path for the model")
parser.add_argument("-r", "--render", help="render training", action="store_true", default=False)
parser.add_argument("-p", "--path", help="path of the checkpoint file", default="my_dqn.ckpt")
parser.add_argument("-t", "--test", help="test (no learning and minimal epsilon)", action="store_true", default=False)
parser.add_argument("-v", "--verbosity", action="count", help="increase output verbosity", default=0)
args = parser.parse_args()

from collections import deque
import gym
import numpy as np
import numpy.random as rnd
import os
import tensorflow as tf
from tensorflow.contrib.layers import convolution2d, fully_connected

env = gym.make("Freeway-v0")
done = True  # env needs to be reset

# TensorFlow - Construction phase
input_height = 88
input_width = 80
input_channels = 1  # we only look at one frame at a time, so ghosts and power pellets really are invisible when they blink
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"] * 3
conv_activation = [tf.nn.relu] * 3
n_hidden_inputs = 64 * 11 * 10  # conv3 has 64 maps of 11x10 each
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n  # MsPacman has 9 actions: upper left, up, upper right, left, and so on.
initializer = tf.contrib.layers.variance_scaling_initializer() # He initialization
learning_rate = 0.01

def q_network(X_state, scope):
    prev_layer = X_state
    conv_layers = []
    with tf.variable_scope(scope) as scope:
        for n_maps, kernel_size, stride, padding, activation in zip(conv_n_maps, conv_kernel_sizes, conv_strides, conv_paddings, conv_activation):
            prev_layer = convolution2d(prev_layer, num_outputs=n_maps, kernel_size=kernel_size, stride=stride, padding=padding, activation_fn=activation, weights_initializer=initializer)
            conv_layers.append(prev_layer)
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_inputs])
        hidden = fully_connected(last_conv_layer_flat, n_hidden, activation_fn=hidden_activation, weights_initializer=initializer)
        outputs = fully_connected(hidden, n_outputs, activation_fn=None)
    trainable_vars = {var.name[len(scope.name):]: var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)}
    return outputs, trainable_vars

X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])
actor_q_values, actor_vars = q_network(X_state, scope="q_networks/actor")    # acts
critic_q_values, critic_vars = q_network(X_state, scope="q_networks/critic") # learns

copy_ops = [actor_var.assign(critic_vars[var_name])
            for var_name, actor_var in actor_vars.items()]
copy_critic_to_actor = tf.group(*copy_ops)

X_action = tf.placeholder(tf.int32, shape=[None])
y = tf.placeholder(tf.float32, shape=[None, 1])
q_value = tf.reduce_sum(critic_q_values * tf.one_hot(X_action, n_outputs),
                        reduction_indices=1, keep_dims=True)
cost = tf.reduce_mean(tf.square(y - q_value))
global_step = tf.Variable(0, trainable=False, name="global_step")
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(cost, global_step=global_step)

init = tf.initialize_all_variables()
saver = tf.train.Saver()

# Replay memory, epsilon-greedy policy and observation preprocessing
replay_memory_size = 1000
replay_memory = deque([], maxlen=replay_memory_size)

def sample_memories(batch_size):
    indices = rnd.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)

epsilon_min = 0.05
epsilon_max = 1.0 if not args.test else epsilon_min
epsilon_decay_steps = args.number_steps // 2
epsilon = epsilon_max

def epsilon_greedy(q_values, epsilon):
    if rnd.rand() < epsilon:
        return rnd.randint(n_outputs) # random action
    else:
        return np.argmax(q_values) # optimal action

mspacman_color = np.array([210, 164, 74]).mean()

def preprocess_observation_dqn(obs):
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.mean(axis=2) # to greyscale
    img[img==mspacman_color] = 0 # Improve contrast
    img = (img - 128) / 128 - 1 # normalize from -1. to 1.
    return img.reshape(88, 80, 1)

# TensorFlow - Execution phase
n_steps = args.number_steps
learning_start_iteration = 1000
learning_every_n_iterations = args.learn_iterations
batch_size = 50
discount_rate = 0.95
skip_start = 90
iteration = 0
rewards_history = []

temp_counter = 1

action = 0

# x: [batch_size x self.height x self.width x (3 * (c.HIST_LEN))]
num_test_rec = 1  # number of recursive predictions to make on test
num_steps = 1000001
frames_history = np.zeros((1, c.FULL_HEIGHT, c.FULL_WIDTH, 3*(c.HIST_LEN + num_test_rec)))
#dynamics_model = AVGRunner(num_steps, args.model_path, num_test_rec)
dynamics_model = AVGRunner(num_steps, None, num_test_rec)
with tf.Session() as sess:
    if os.path.isfile(args.path):
        saver.restore(sess, args.path)
    else:
        init.run()
    while True:
        step = global_step.eval()
        if step >= n_steps:
            break
        iteration += 1
        if args.verbosity > 0:
            print("\rIteration {} | Training step {}/{} ({:.1f}%) | avg reward={:.2f} | epsilon={:.2f}".format(iteration, step, n_steps, step * 100 / n_steps, np.mean(rewards_history[-5000:]), epsilon), end="")
        if done: # game over, start again
            obs = env.reset()
            for skip in range(skip_start): # skip boring game iterations at the start of each game
                obs, reward, done, info = env.step(0)
            state = preprocess_observation_dqn(obs)
        if args.render:
            env.render()

        prev_action = action

        # Actor evaluates what to do
        q_values = actor_q_values.eval(feed_dict={X_state: [state]})
        epsilon = max(epsilon_min, epsilon_max - (epsilon_max - epsilon_min) * global_step.eval() / epsilon_decay_steps)
        action = epsilon_greedy(q_values, epsilon)

        # plt.imshow(obs)
        # plt.show()
        # print(action)

        # Actor plays
        obs, reward, done, info = env.step(action)
        next_state = preprocess_observation_dqn(obs)
        if args.test:
            continue

        # plt.imshow(obs)
        # plt.show()

        # Predict next frame of game
        # print(obs.shape) # (210, 160, 3)
        # plt.imshow(obs)
        # plt.show()

        # obs2 = dynamics_model.predict(obs)
        # plt.imshow(obs2)
        # plt.show()

        frames_history = np.roll(frames_history, -3, axis=3)
        frames_history[0,:,:,-3:] = utils.normalize_frames(obs.reshape((1,)+obs.shape))
        if iteration % 1000 == 0: # TODO: replace 1000 with a constante
            for a in range(c.ACTION_SPACE):
                pred = utils.denormalize_frames(dynamics_model.predict(frames_history, a, print_out=False))[0]
                plt.imsave('./Temp5/%06i_%i.png'%(iteration, a), pred)

        # if step > 0.9*n_steps:
        #     pred = utils.denormalize_frames(dynamics_model.predict(frames_history, 0, print_out=False))[0]
        #     plt.imsave('./Temp5/gen/g_%06i.png'%temp_counter, pred)
        #     plt.imsave('./Temp5/ren/r_%06i.png'%(temp_counter-1), obs)
        #     temp_counter += 1

        if iteration > c.HIST_LEN + num_test_rec:
            dynamics_model.train(frames_history, action, print_out=False) # prev_action because we are trying to predict newest frame?
        #print(pred.shape)

        # plt.imshow(frames_history)
        # plt.show()

        rewards_history.append(reward)

        # Let's memorize what just happened
        replay_memory.append((state, action, reward, next_state, 1.0 - done))
        state = next_state

        # Critic learns
        if iteration > learning_start_iteration and iteration % learning_every_n_iterations == 0:
            X_state_val, X_action_val, rewards, X_next_state_val, continues = sample_memories(batch_size)
            next_q_values = actor_q_values.eval(feed_dict={X_state: X_next_state_val})
            y_val = rewards + continues * discount_rate * np.max(next_q_values, axis=1, keepdims=True)
            training_op.run(feed_dict={X_state: X_state_val, X_action: X_action_val, y: y_val})

            if global_step.eval() % args.save_steps == 0:
                saver.save(sess, args.path)
            if global_step.eval() % args.copy_steps == 0:
                copy_critic_to_actor.run()