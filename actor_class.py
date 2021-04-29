import numpy as np
import tensorflow.compat.v1 as tf


class Actor:

    def __init__(self, scope, target_network, env, flags):
        """
        This class implements the actor for the deterministic policy gradients model.
        The actor class determines the action that the agent must take in a environment.

        :param scope: within this scope the parameters will be defined
        :param target_network: instance of the Actor(target-network class)
        :param env: instance of the openAI environment
        :param FLAGS: TensorFlow flags which contain values for hyperparameters

        """

        self.TF_FLAGS = flags
        self.env = env
        self.scope = 'actor_' + scope
        self.min_action = self.env.get_env().action_space.low
        self.max_action = self.env.get_env().action_space.high

        if 'target' in scope:
            with tf.variable_scope(scope):

                self.states = tf.placeholder(tf.float32, shape=(
                    None, self.env.get_state_size()), name='states')
                self.action = self.create_network()
                self.param = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

        elif 'local' in scope:

            with tf.variable_scope(scope):

                # Add the target network instance
                self.target_network = target_network

                # Create the placeholders for the input to the network
                self.states = tf.placeholder(tf.float32, shape=(
                    None, self.env.get_state_size()), name='states')

                # Create the network with the goal of improving the action taken with respect to the critic choice
                self.action = self.create_network()
                self.q_network_gradient = tf.placeholder(tf.float32, shape=(
                    None, self.env.get_action_size()), name='q_network_gradients')

                # The parameters of the network
                self.param = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

                # print("Actor Param: ", self.param)

                with tf.name_scope('policy_gradients'):
                    self.policy_gradient = tf.gradients(
                        self.action, self.param, self.q_network_gradient)
                    self.policy_gradient_normalized = list(
                        map(lambda x: tf.div(x, self.TF_FLAGS.batch_size), self.policy_gradient))

                with tf.name_scope('train_policy_network'):
                    # self.train_opt = tf.train.AdamOptimizer(
                    #     self.TF_FLAGS.learning_rate_Actor).apply_gradients(zip(self.policy_gradient_normalized, self.param))
                    self.train_opt = tf.train.AdamOptimizer(
                        self.TF_FLAGS.learning_rate_Actor).apply_gradients(zip(self.policy_gradient_normalized, self.param))

                with tf.name_scope('update_actor_target'):
                    # Perform a soft update of the parameters: Actor network parameters = Local Parameters (LP) and Target network parameters (TP)
                    # TP = tau * LP + (1-tau) * TP
                    self.update_opt = [tp.assign(tf.multiply(self.TF_FLAGS.tau, lp)+tf.multiply(
                        1-self.TF_FLAGS.tau, tp)) for tp, lp in zip(self.target_network.param, self.param)]

                with tf.name_scope('initialize_actor_target_network'):
                    # Set the parameters of the local network equal to the target one
                    # LP = TP
                    self.init_target_op = [tp.assign(lp) for tp, lp in zip(
                        self.target_network.param, self.param)]

    def create_network(self):
        '''Build the neural network that estimates the action for a given state '''
        first_layer_size = 400
        second_layer_size = 300

        h1 = tf.layers.dense(
            self.states, first_layer_size, tf.nn.relu, use_bias=True,
            kernel_initializer=tf.initializers.glorot_normal(),
            bias_initializer=tf.zeros_initializer()
        )

        # h1 = tf.layers.batch_normalization(h1)

        h2 = tf.layers.dense(
            h1, second_layer_size, tf.nn.relu, use_bias=True,
            kernel_initializer=tf.initializers.glorot_normal(),
            bias_initializer=tf.zeros_initializer()
        )

        # h2 = tf.layers.batch_normalization(h2)

        actions = tf.layers.dense(
            h2, self.env.get_action_size(), activation=tf.tanh,
            kernel_initializer=tf.initializers.glorot_normal()
        )

        scalled_actions = actions * self.max_action

        return scalled_actions

    def set_session(self, session):
        '''Set the session '''
        self.session = session

    def init_target_network(self):
        '''Initialize the parameters of the target-network '''
        self.session.run(self.init_target_op)

    def update_target_parameter(self):
        '''Update the parameters of the target-network using a soft update'''
        self.session.run(self.update_opt)

    def get_action(self, states, noise=None):
        '''Get an action for a certain state '''
        feed_dict = {
            self.states: states
        }

        action = self.session.run(self.action, feed_dict)

        # print("get_action action: ", action)

        if noise is not None:
            # Adding noise to action and make sure action is within bounds
            noisy_action = np.clip(action + noise(), self.min_action, self.max_action)
            # print("get_action noisy action: ", action)
            return noisy_action

        return action

    def train(self, states, q_gradient):
        '''Train the actor network '''
        feed_dict = {
            self.q_network_gradient: q_gradient,
            self.states: states
        }

        self.session.run(self.train_opt, feed_dict)
