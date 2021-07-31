import numpy as np
import tensorflow.compat.v1 as tf
import random
from .env_class import Environment

class Agent:

    def __init__(self, TF_FLAGS, env_name, res_folder):
        ''' This class build the Agent that learns in the environment via the actor-critic algorithm. '''

        self.env = Environment(env_name)
        self.TF_FLAGS = TF_FLAGS
        self.res_folder = res_folder

        print(self.TF_FLAGS)

        # experience replay buffer
        self.memory = []
        self.memory_size = self.TF_FLAGS.buffer_size

        self.step_count = 0

    def get_action(self, state):
        return self.env.get_random_action()

    def get_action_training(self, state):
        return self.env.get_random_action()

    def update_agent(self):
        pass

    def add2memory(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

        # Discard the first half of the memory if the buffer is full
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[int(self.memory_size) // 2:]

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
                action = self.get_action_training(state)
            else:
                action = self.env.env.action_space.sample()

            # Obtain a <state, reward, done> tuple from the environment
            state, reward, done, _ = self.env.get_env().step(action)
            total_reward += reward

            if render:
                self.env.render()

            self.add2memory(prev_state, action, reward, state, done)

            if self.step_count >= update_after and self.step_count % update_every == 0:
                for _ in range(update_every):
                    self.update_agent()

            iters += 1

        return total_reward

    def train(self, num_episodes=100, display_step=10, max_iterations=500):
        '''Run and train the agent for a particular number of episodes. '''

        print("\n" + "*" * 100)
        print("TRAINING START\n")
        total_rewards = []

        for n in range(0, num_episodes):

            if n % display_step == 0 and n >= display_step:
                avg_reward = sum(
                    total_rewards[n - display_step: n]) / display_step
                print("episodes: %i, num steps: %i, avg_reward (last: %i episodes): %.2f" %
                      (n, self.step_count, display_step, avg_reward))
                total_reward = self.train_one_episode(
                    max_iterations, render=True)
                self.env.make_gif(f"{self.res_folder}/episode_number_{n}")
            else:
                total_reward = self.train_one_episode(max_iterations)

            with open(f"{self.res_folder}/result.csv", 'a') as f:
                f.write(f"{n},{self.step_count},{total_reward}\n")

            total_rewards.append(total_reward)

        print("\n" + "*" * 100)
        print("TRAINING END\n")

        return total_rewards

    def play_one_episode(self, max_iterations=500, render=False):
        '''Runs and records one episode using actions from the agent'''
        # Get the initial state
        state = self.env.reset()
        state = state.reshape(1, self.env.get_state_size())
        done = False
        iters = 0
        total_reward = 0

        # Loop for the episode
        while not done and iters < max_iterations:

            # Sample an action from the agent
            action = self.get_action(state)

            # Obtain a <state, reward, done> tuple from the environment
            state, reward, done, _ = self.env.get_env().step(action.reshape(-1))
            state = state.reshape(1, self.env.get_state_size())
            total_reward += reward

            if render:
                self.env.render()

            iters += 1

        return total_reward
