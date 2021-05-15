import numpy as np
import tensorflow.compat.v1 as tf
import random

from ..base import env_class, agent_class
from ..base.noise_class import OUNoise
from .actor_class import Actor
from .critic_class import Critic


class Agent(agent_class.Agent):

    def __init__(self, TF_FLAGS, env_name, res_folder):
        ''' This class build the Agent that learns in the environment via the actor-critic algorithm. '''

        agent_class.Agent.__init__(self, TF_FLAGS, env_name, res_folder)

        # Define the actor network and a "stationary" target for the training
        self.actor_target = Actor(
            scope='target', target_network=None, env=self.env, flags=TF_FLAGS)
        self.actor = Actor(
            scope='local', target_network=self.actor_target, env=self.env, flags=TF_FLAGS)

        # Define the critic network and a "stationary" target for the training
        self.critic_target = Critic(
            scope='target', target_network=None, env=self.env, flags=TF_FLAGS)
        self.critic = Critic(
            scope='local', target_network=self.critic_target, env=self.env, flags=TF_FLAGS)

        # Start the TF sessions
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

        # Pass it to the four networks in total
        self.critic.set_session(self.session)
        self.actor.set_session(self.session)
        self.actor_target.set_session(self.session)
        self.critic_target.set_session(self.session)

        # Initialise network weights
        self.critic.init_target_network()
        self.actor.init_target_network()

        # Assert the correct initialisation of both local and target
        self.network_similarity(self.critic, self.critic_target)
        self.network_similarity(self.actor, self.actor_target)

        # Create the noise object to add to the actor network
        self.actor_noise = OUNoise(mean=np.zeros(
            1), std_deviation=float(self.TF_FLAGS.actor_noise_dev) * np.ones(1))

    def get_action(self, state):
        return self.actor.get_action(state)

    def get_action_training(self, state):
        return self.actor.get_action(state.reshape(
            1, -1), self.actor_noise).reshape(-1)

    def update_agent(self):
        if len(self.memory) > self.TF_FLAGS.batch_size:
            # Randomly chose a batch from the replay buffer
            indexes = random.sample(
                range(len(self.memory) - 1), self.TF_FLAGS.batch_size)

            states = np.array([self.memory[i][0] for i in indexes])
            actions = np.array([self.memory[i][1] for i in indexes])
            rewards = np.array([self.memory[i][2] for i in indexes])
            next_states = np.array([self.memory[i][3] for i in indexes])
            dones = np.array([self.memory[i][4] for i in indexes])

            # print("length memory: ", len(self.memory))
            # print("states: ", states.shape)
            # print("actions: ", actions.shape)
            # print("next_states: ", next_states.shape)
            # print("done: ", dones.shape)
            # print("reward: ", rewards.shape)

            # update the parameters every 50 steps on average
            self.update_critic(states, actions, rewards, next_states, dones)
            self.update_actor(states, actions, rewards, next_states, dones)

            # slowly converge toward the target
            self.critic.update_target_parameter()
            self.actor.update_target_parameter()

    def update_critic(self, states, actions, rewards, next_states, dones):
        next_actions = np.array(
            self.actor.target_network.get_action(next_states))
        q_nexts = self.critic.target_network.calculate_Q(
            next_states, next_actions)
        q_targets = rewards + self.TF_FLAGS.gamma * q_nexts

        # print("next actions: ", next_actions.shape)
        # print("q_nexts: ", q_targets.shape)
        # print("rewards shape: ", rewards.shape)
        # print("q_targets: ", q_targets.shape)

        self.critic.train(states, actions, q_targets)

    def update_actor(self, states, actions, rewards, next_states, dones):
        predicted_action = np.array(self.actor.get_action(states))
        q_gradients = self.critic.compute_gradients(states, predicted_action)

        # print("predicted actions: ", predicted_action.shape)
        # print("q_gradients: ", q_gradients.shape)

        self.actor.train(states, q_gradients)

    def network_similarity(self, network1, network2):
        for lp, tp in zip(network1.param, network2.param):
            try:
                assert np.array_equal(
                    self.session.run(lp), self.session.run(tp))
            except AssertionError as e:
                e.args += (f'{network1.scope} and {network2.scope} don\'t have the same weights',)
                raise
