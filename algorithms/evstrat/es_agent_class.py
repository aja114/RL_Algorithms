import numpy as np
import tensorflow.compat.v1 as tf
import random

from ..base import agent_class, noise_class

class Agent(agent_class.Agent):

    def __init__(self, TF_FLAGS, env_name, res_folder, id, weights=None):
        agent_class.Agent.__init__(self, TF_FLAGS, env_name, res_folder)

        scope = f'agent_{id}'

        with tf.variable_scope(scope):
            self.states = tf.placeholder(tf.float32, shape=(
                None, self.env.get_state_size()), name='state')
            self.actions = self.action_network()

        self.params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    def set_session(self, session):
        self.session = session

    def action_network(self):
        first_layer_size = 128
        second_layer_size = 128

        h1 = tf.layers.dense(
            self.states, first_layer_size, tf.nn.relu, use_bias=True,
            kernel_initializer=tf.initializers.glorot_normal(),
            bias_initializer=tf.zeros_initializer()
        )

        h2 = tf.layers.dense(
            h1, second_layer_size, tf.nn.relu, use_bias=True,
            kernel_initializer=tf.initializers.glorot_normal(),
            bias_initializer=tf.zeros_initializer()
        )

        actions = tf.layers.dense(
            h2, self.env.get_action_size(), activation=tf.tanh,
            kernel_initializer=tf.initializers.glorot_normal()
        )

        return actions

    def get_action(self, state):
        feed_dict = {self.states: state}
        return self.session.run(self.actions, feed_dict)

    def get_action_training(self, state):
        feed_dict = {self.states: state}
        return self.session.run(self.actions, feed_dict)

    def upd_weights(self, v):
        upd_w = []
        s = 0
        for w in self.params:
            e = np.prod(w.shape)
            upd_w.append(w.assign(tf.add(w, v[s:s+e].reshape(w.shape))))
            s = s+e
        self.session.run(upd_w)

    def set_weights(self, weights, v):
        if v is None:
            set_w = [w.assign(bw) for w, bw in zip(self.params, weights)]
        else:
            set_w = []
            s = 0
            for w, bw in zip(self.params, weights):
                e = np.prod(w.shape)
                set_w.append(w.assign(tf.add(bw, v[s:s+e].reshape(w.shape))))
                s = s+e
        self.session.run(set_w)


class Population:
    def __init__(self, TF_FLAGS, env_name, res_folder, pop_size=100, num_episodes=200, lr=1e-7):

        # Set the parameters of the population
        self.pop_size = pop_size
        self.lr = lr
        self.num_episodes = num_episodes

        # Create the population - Agent 0 is used to track the best weights
        self.best_agent = Agent(TF_FLAGS, env_name, res_folder, 0)
        self.best_weights = self.best_agent.params
        self.num_param = np.sum([np.prod(l.shape) for l in self.best_weights])

        self.pop = [Agent(TF_FLAGS, env_name, res_folder, i+1)
                    for i in range(pop_size)]

        self.init_noise()
        self.init_session()

    def init_session(self):
        self.session = self.get_session()
        self.best_agent.set_session(self.session)
        self.session.run(tf.global_variables_initializer())
        for agent in self.pop:
            agent.set_session(self.session)

    def init_noise(self):
        self.noise_mean = 0
        self.noise_std = 0.0001
        self.noise = noise_class.GaussNoise(
            self.noise_mean, 1, size=(self.pop_size, self.num_param))

    def get_session(self, session=None):
        if session is None:
            session = tf.Session()
        return session

    def get_best_weights(self):
        self.best_weights = self.best_agent.params
        # print("Agent 0", self.session.run(self.best_weights[0]))

    def transfer_weights(self):
        self.curr_noise = self.noise()
        self.curr_noise_std = self.noise_std*self.curr_noise

        for i, agent in enumerate(self.pop):
            agent.set_weights(self.best_weights, self.curr_noise_std[i])
            # print(f"Agent {i+1}")
            # print(f"Noise {i+1}: ", self.curr_noise[i])
            # print(f"Weights {i+1}: ", self.session.run(agent.params[0]))

    def train_one_episode(self):
        rewards = np.zeros(self.pop_size)
        for i, agent in enumerate(self.pop):
            rewards[i] = agent.play_one_episode()

        return rewards

    def train(self):
        mean_rewards = np.zeros(self.num_episodes)
        for ep in range(self.num_episodes):
            print(f"Episode #{ep}")

            self.get_best_weights()
            self.transfer_weights()
            rewards = self.train_one_episode()
            self.update_agent(rewards)

            mean_rewards[ep] = np.mean(rewards)
            print("MEAN REWARD: ", np.mean(rewards))
            print("BEST EVAL: ", self.best_agent.play_one_episode())

        print(mean_rewards)

    def update_agent(self, rewards):
        upd = self.lr * (1 / (self.pop_size * self.noise_std)) * \
            self.curr_noise.T @ rewards
        self.best_agent.upd_weights(upd)
        # print("Updates: ", upd)
