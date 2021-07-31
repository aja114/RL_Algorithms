import numpy as np
import tensorflow.compat.v1 as tf


class DQN:
    def __init__(self, scope, target_network, env, flags):
        """
        This class implements the deep Q-Learning agent

        :param scope: within this scope the parameters will be defined
        :param target_network: instance of the DQN target-network class
        :param env: instance of the openAI environment
        :param FLAGS: TensorFlow flags which contain values for hyperparameters

        """

        self.TF_FLAGS = flags
        self.env = env
        self.scope = 'dqn' + scope

        if 'target' in scope:
            with tf.variable_scope(scope):

                self.states = tf.placeholder(tf.float32, shape=(
                    None, self.env.get_state_size()), name='states')

                self.q = self.create_network(scope='q_network_target')
                self.param = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/q_network_target')

        elif 'local' in scope:

            with tf.variable_scope(scope):
                # Add the target network instance
                self.target_network = target_network

                # Create the placeholders for the inputs to the network
                self.states = tf.placeholder(
                    tf.float32, shape=(None, self.env.get_state_size()),
                    name='states')
                self.actions = tf.placeholder(
                    tf.uint8, shape=(None, ), name='actions')
                self.q_targets = tf.placeholder(
                    tf.float32, shape=(None,), name='q_targets')

                # Create the network with the goal of predicting the
                # action-value function
                self.q = self.create_network(scope='q_network')

                # The parameters of the network
                self.param = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/q_network')


                with tf.name_scope('q_network_loss'):
                    # Difference between targets value and calculated ones by
                    # the model
                    actions_one_hot = tf.one_hot(self.actions, self.env.get_action_size(),  1.0, 0.0, name='action_one_hot')
                    q = tf.reduce_sum(self.q * actions_one_hot, reduction_indices=1, name='q_acted')
                    self.loss = tf.losses.mean_squared_error(
                        q, self.q_targets)

                with tf.name_scope('train_q_network'):
                    self.train_opt = tf.train.AdamOptimizer(
                        self.TF_FLAGS.critic_learning_rate).minimize(self.loss)

                with tf.name_scope('update_q_target'):
                    # Perform a soft update of the parameters: Critic network parameters = Local Parameters (LP) and Target network parameters (TP)
                    # TP = tau * LP + (1-tau) * TP
                    self.update_opt = [tp.assign(tf.multiply(self.TF_FLAGS.tau, lp) + tf.multiply(
                        1 - self.TF_FLAGS.tau, tp)) for tp, lp in zip(self.target_network.param, self.param)]

                with tf.name_scope('initialize_q_target_network'):
                    # Set the parameters of the local network equal to the target one
                    # LP = TP
                    self.init_target_op = [tp.assign(lp) for tp, lp in zip(
                        self.target_network.param, self.param)]

                with tf.name_scope('q_nexts'):
                    self.q_nexts = tf.math.reduce_max(self.target_network.q, axis=1)

    def create_network(self, scope):
        '''Build the neural network that estimates the action for a given state '''
        first_layer_size = 400
        second_layer_size = 300

        with tf.variable_scope(scope):
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
                h2, self.env.get_action_size(), activation=None,
                kernel_initializer=tf.initializers.glorot_normal()
            )

        return actions

    def set_session(self, session):
        '''Set the session '''
        self.session = session

    def init_target_network(self):
        '''Initialize the parameters of the target-network '''
        self.session.run(self.init_target_op)

    def update_target_parameter(self):
        '''Update the parameters of the target-network using a soft update'''
        self.session.run(self.update_opt)

    def get_action(self, states):
        '''Get an action for a certain state '''
        feed_dict = {
            self.states: states
        }
        q_values = self.session.run(self.q, feed_dict)
        action = np.argmax(q_values, axis=1)

        return action

    def get_q_nexts(self, next_states):
        feed_dict = {
            self.target_network.states: next_states
        }

        q_nexts = self.session.run(self.q_nexts, feed_dict)

        return q_nexts

    def train(self, states, actions, targets):
        '''Train the q network '''
        feed_dict = {
            self.states: states,
            self.actions: actions,
            self.q_targets: targets
        }

        self.session.run(self.train_opt, feed_dict)

        # loss = self.session.run(self.loss, feed_dict)
        # q_values = self.session.run(self.q, feed_dict)
        #
        # print("loss: ", loss)
        # print("q_values: ", np.mean(q_values))
