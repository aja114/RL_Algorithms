import numpy as np
import tensorflow.compat.v1 as tf
import random

from ..base import env_class, agent_class
from .dqn_class import DQN

class Agent(agent_class.Agent):

    def __init__(self, TF_FLAGS, env_name, res_folder):
        ''' This class build the Agent that learns in the environment via the DQN algorithm. '''
        super().__init__(TF_FLAGS, env_name, res_folder)

        self.epsilon = 0.1

        # Define the actor network and a "stationary" target for the training
        self.dqn_target = DQN(
            scope='target', target_network=None, env=self.env, flags=TF_FLAGS)
        self.dqn = DQN(
            scope='local', target_network=self.dqn_target, env=self.env, flags=TF_FLAGS)

        # Start the TF sessions
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.dqn.set_session(self.session)
        self.dqn_target.set_session(self.session)


        # Initialise network weights
        self.dqn.init_target_network()

        # Assert the correct initialisation of both local and target
        self.network_similarity(self.dqn, self.dqn_target)

    def get_action(self, state):
        return self.dqn.get_action(state)

    def get_action_training(self, states):
        if np.random.rand() <= self.epsilon:
            return self.env.get_random_action()
        else:
            return self.get_action(states.reshape(-1, self.env.get_state_size()))[0]

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

            # update the parameters every 50 steps on average
            self.update_q_network(states, actions, rewards, next_states, dones)

            # slowly converge toward the target
            self.dqn.update_target_parameter()

    def update_q_network(self, states, actions, rewards, next_states, dones):
        q_nexts = self.dqn.get_q_nexts(next_states)
        q_targets = rewards + self.TF_FLAGS.gamma * q_nexts * (1 - dones)

        # print("dones: ", dones[:5])
        # print("q_nexts: ", q_nexts[:5])
        # print("q_targets: ", q_targets[:5])

        self.dqn.train(states, actions, q_targets)

    def network_similarity(self, network1, network2):
        for lp, tp in zip(network1.param, network2.param):
            try:
                assert np.array_equal(
                    self.session.run(lp), self.session.run(tp))
            except AssertionError as e:
                e.args += (f'{network1.scope} and {network2.scope} don\'t have the same weights',)
                raise
