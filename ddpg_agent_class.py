import numpy as np
import tensorflow.compat.v1 as tf
import random

from env_class import Environment
from actor_class import Actor
from critic_class import Critic
from noise_class import OUActionNoise


class Agent:

    def __init__(self, TF_FLAGS, env_name, res_folder):
        ''' This class build the Agent that learns in the environment via the actor-critic algorithm. '''

        self.env = Environment(env_name)
        self.TF_FLAGS = TF_FLAGS
        self.res_folder = res_folder

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
        self.actor_noise = OUActionNoise(mean=np.zeros(
            1), std_deviation=float(self.TF_FLAGS.actor_noise_dev) * np.ones(1))

        # experience replay buffer
        self.memory = []
        self.memory_size = 1000000

        self.step_count = 0

    def update_agent(self):
        if len(self.memory) > self.TF_FLAGS.batch_size:
            # Randomly chose a batch from the replay buffer
            indexes = random.sample(
                range(len(self.memory)-1), self.TF_FLAGS.batch_size)

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
        next_actions = np.array(self.actor.target_network.get_action(next_states))
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
                assert np.array_equal(self.session.run(lp), self.session.run(tp))
            except AssertionError as e:
                e.args += (f'{network1.scope} and {network2.scope} don\'t have the same weights',)
                raise

    def add2memory(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

        # Discard the first half of the memory if the buffer is full
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[int(self.memory_size)//2:]

    def train_one_episode(self, max_iterations=500, render=False):
        ''' Play an episode and train the agent '''

        state = self.env.reset()

        done = False
        total_reward = 0
        iters = 0

        update_after = self.TF_FLAGS.update_after
        update_every = self.TF_FLAGS.update_every
        start_steps = self.TF_FLAGS.start_steps

        # Loop for the episode until
        # 1. An end state is reached
        # 2. The maximum number of iterations is reached
        while not done and iters < max_iterations:
            self.step_count += 1

            # Sample an action from the actor distribution
            prev_state = state
            if self.step_count > start_steps:
                action = self.actor.get_action(state.reshape(1, -1), self.actor_noise).reshape(-1)
            else:
                action = self.env.env.action_space.sample()

            # Obtain a <state, reward, done> tuple from the environment
            state, reward, done, _ = self.env.get_env().step(action)

            # print("prev state: ", prev_state.shape)
            # print("action: ", action.shape)
            # print("state: ", state.shape)
            # print("done: ", done)
            # print("reward: ", reward)

            total_reward += reward

            if render:
                self.env.render()

            self.add2memory(prev_state, action, reward, state, done)

            if self.step_count >= update_after and self.step_count%update_every==0:
                for _ in range(update_every):
                    self.update_agent()

            iters += 1

        return total_reward

    def train(self, num_episodes=100, display_step=10, max_iterations=500):
        '''Run and train the agent for a particular number of episodes. '''

        print("\n"+"*"*100)
        print("TRAINING START\n")
        total_rewards = []

        for n in range(0, num_episodes):

            if n % display_step == 0 and n >= display_step:
                avg_reward = sum(
                    total_rewards[n-display_step: n]) / display_step
                print("episodes: %i, num steps: %i, avg_reward (last: %i episodes): %.2f" %
                      (n, self.step_count, display_step, avg_reward))
                # total_reward = self.train_one_episode(max_iterations, render=True)
                if n>500 and avg_reward > 0:
                    total_reward = self.train_one_episode(max_iterations, render=True)
                    self.env.make_gif(f"{self.res_folder}/episode_number_{n}")
                else:
                    total_reward = self.train_one_episode(max_iterations, render=False)

            else:
                total_reward = self.train_one_episode(max_iterations)

            with open(f"{self.res_folder}/result.csv", 'a') as f:
                f.write(f"{n},{self.step_count},{total_reward}\n")

            total_rewards.append(total_reward)

        print("\n"+"*"*100)
        print("TRAINING END\n")

        return total_rewards

    def play_one_episode(self, max_iterations=500):
        '''Runs and records one episode using the trained actor and critic'''
        # Get the initial state and reshape it
        state = self.env.reset()
        state = state.reshape(1, self.env.get_state_size())
        done = False
        iters = 0
        total_reward = 0

        # Loop for the episode
        while not done and iters < max_iterations:

            # Sample an action from the gauss distribution
            action = self.actor.get_action(state)

            # Obtain a <state, reward, done> tuple from the environment
            state, reward, done, _ = self.env.get_env().step(action.reshape(-1))
            state = state.reshape(1, self.env.get_state_size())
            total_reward += reward

            self.env.render()
            iters += 1

        return self.env.make_gif()
