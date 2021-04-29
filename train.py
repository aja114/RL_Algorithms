import os
import logging
from warnings import simplefilter

logging.disable(logging.WARNING)
simplefilter(action='ignore', category=UserWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
import sys

# res_folder = "test"
# env_name = "BipedalWalker-v3"
# algorithm = 'td3'

algorithm = sys.argv[1]
env_name = sys.argv[2]
res_folder = f'Data/{algorithm}_'+sys.argv[3]

if algorithm == 'td3':
    from td3_agent_class import Agent
elif algorithm == 'ddpg':
    from ddpg_agent_class import Agent

if os.path.exists(f"{res_folder}"):
    os.system(f"rm -f {res_folder}/*")
else:
    os.mkdir(f"{res_folder}")

os.system(f"touch {res_folder}/result.csv")
with open(f"{res_folder}/result.csv", 'w') as f:
    f.write("episodes,num_steps,reward\n")

tf.disable_eager_execution()
tf.set_random_seed(3)

# Using flags for usefulness as they can be set when running the code from the command line
tf.app.flags.DEFINE_float('learning_rate_Actor', 0.0001,
                          'Learning rate for the policy estimator')
tf.app.flags.DEFINE_float('learning_rate_Critic', 0.0001,
                          'Learning rate for the state-value estimator')
tf.app.flags.DEFINE_float('gamma', 0.99, 'Future discount factor')
tf.app.flags.DEFINE_float(
    'tau', 0.001, 'Update rate for the target networks parameter')
tf.app.flags.DEFINE_float('actor_noise_dev', 0.1, 'Standard deviation for the exploration noise component')
tf.app.flags.DEFINE_float('target_noise_dev', 0.2, 'Standard deviation for the smoothing noise component')
tf.app.flags.DEFINE_float('noise_clip', 0.5, 'clip val for the smoothing noise component')
tf.app.flags.DEFINE_integer('batch_size', 50, 'Batch size for the updates')
tf.app.flags.DEFINE_integer('update_after', int(1*1e3), 'when to start the updates')
tf.app.flags.DEFINE_integer('update_every', 50, 'frequency at which to perform the updates')
tf.app.flags.DEFINE_integer('start_steps', int(5*1e4), 'start sampling from the networks')
tf.app.flags.DEFINE_integer('policy_delay', 4, 'policy delay')

TF_FLAGS = tf.app.flags.FLAGS

# Create and train the agent
agent = Agent(TF_FLAGS, env_name, res_folder)
total_rewards = agent.train(num_episodes=20001, display_step=1000, max_iterations=1000)
