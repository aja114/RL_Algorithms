import numpy as np
import tensorflow.compat.v1 as tf
import random
from tensorflow.python.summary.writer.writer import FileWriter


class Critic:

    def __init__(self, scope, target_network, env, flags):
        """
        This class implements the Critic for the stochastic policy gradient model.
        The critic provides a state-value for the current state environment where
        the agent operates.

        :param scope: within this scope the parameters will be defined
        :param target_network: instance of the Actor(target-network class)
        :param env: instance of the openAI environment
        :param FLAGS: TensorFlow flags which contain thevalues for hyperparameters

        """

        self.TF_FLAGS = flags
        self.env = env
        self.scope = 'critic_' + scope

        if 'target' in scope:

            with tf.variable_scope(scope):

                self.states = tf.placeholder(tf.float32, shape=(
                    None, self.env.get_state_size()), name='states')
                self.actions = tf.placeholder(tf.float32, shape=(
                    None, self.env.get_action_size()), name='actions')
                self.q = self.create_network(scope='q_target_network')
                self.param = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/q_target_network')

        elif 'local' in scope:

            with tf.variable_scope(scope):

                # Add the target network instance
                self.target_network = target_network

                # Create the placeholders for the inputs to the network
                self.states = tf.placeholder(tf.float32, shape=(
                    None, self.env.get_state_size()), name='states')
                self.actions = tf.placeholder(tf.float32, shape=(
                    None, self.env.get_action_size()), name='actions')

                # Create the network with the goal of predicting the action-value function
                self.q = self.create_network(scope='q_network')
                self.q_targets = tf.placeholder(
                    tf.float32, shape=(None,), name='q_targets')

                # The parameters of the network
                self.param = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/q_network')

                # print("Critic Param: ", self.param)

                with tf.name_scope('q_network_loss'):
                    # Difference between targets value and calculated ones by the model
                    self.loss = tf.losses.mean_squared_error(
                        self.q_targets, self.q)

                with tf.name_scope('train_q_network'):
                    # Optimiser for the training of the critic network
                    self.train_opt = tf.train.AdamOptimizer(
                        self.TF_FLAGS.learning_rate_Critic).minimize(self.loss)

                with tf.name_scope('q_network_gradient'):
                    # Compute the gradients to be used for the actor model training
                    self.actor_loss = -tf.math.reduce_mean(self.q)
                    self.gradients = tf.gradients(self.actor_loss, self.actions)

                with tf.name_scope('update_q_target'):
                    # Perform a soft update of the parameters: Critic network parameters = Local Parameters (LP) and Target network parameters (TP)
                    # TP = tau * LP + (1-tau) * TP
                    self.update_opt = [tp.assign(tf.multiply(self.TF_FLAGS.tau, lp)+tf.multiply(
                        1-self.TF_FLAGS.tau, tp)) for tp, lp in zip(self.target_network.param, self.param)]

                with tf.name_scope('initialize_q_target_network'):
                    # Set the parameters of the local network equal to the target one
                    # LP = TP
                    self.init_target_op = [tp.assign(lp) for tp, lp in zip(
                        self.target_network.param, self.param)]

                # FileWriter('logs/train', graph=self.train_opt.graph).close()

    def create_network(self, scope):
        '''Build the neural network that estimates the action-values '''
        first_layer_size = 400
        second_layer_size = 300

        with tf.variable_scope(scope):

            state_action = tf.concat([self.states, self.actions], axis=-1)

            h1 = tf.layers.dense(state_action, first_layer_size, tf.nn.relu, use_bias=True,
                                 kernel_initializer=tf.initializers.glorot_normal(),
                                 bias_initializer=tf.zeros_initializer()
                                 )

            h2 = tf.layers.dense(h1, second_layer_size, tf.nn.relu, use_bias=True,
                                 kernel_initializer=tf.initializers.glorot_normal(),
                                 bias_initializer=tf.zeros_initializer()
                                 )

            q = tf.squeeze(tf.layers.dense(h2, 1, None,
                                           kernel_initializer=tf.initializers.glorot_normal()),
                           axis=1)

        return q

    def compute_gradients(self, states, actions):
        '''Compute the gradients of the action_value estimator neural network '''

        feed_dict = {
            self.states: states,
            self.actions: actions
        }

        q_gradient = self.session.run(self.gradients, feed_dict)[0]

        # if random.randint(0,20)%10==0:
        #     print("ACTOR LOSS: ", self.session.run(self.actor_loss, feed_dict))

        return q_gradient

    def calculate_Q(self, states, actions):
        '''Compute the action-value '''

        feed_dict = {self.states: states,
                     self.actions: actions}

        q_next = self.session.run(self.q, feed_dict)

        return q_next

    def train(self, states, action, targets):
        '''Train the critic network '''
        feed_dict = {
            self.states: states,
            self.actions: action,
            self.q_targets: targets
        }

        # print("CRITIC ESTIMATES: ", self.session.run(self.q, feed_dict).shape)
        # if random.randint(0,20)%10==0:
        #     print("CRITIC LOSS: ", self.session.run(self.loss, feed_dict))

        self.session.run(self.train_opt, feed_dict)

    def set_session(self, session):
        '''Set the session '''
        self.session = session

    def init_target_network(self):
        '''Initialize the parameters of the target-network '''
        self.session.run(self.init_target_op)

    def update_target_parameter(self):
        '''Update the parameters of the target-network '''
        self.session.run(self.update_opt)
