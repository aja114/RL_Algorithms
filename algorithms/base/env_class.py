import gym
import numpy as np
import imageio
from IPython.display import Image

class Environment:

    def __init__(self, env_name):
        """
        This class implements a wrapper arround the OpenAI Gym Pendulum
        """
        self.env = gym.make(env_name)
        self.state_size = len(self.env.observation_space.sample())
        self.action_size = len(self.env.action_space.sample())

        # Used to save a visual of one episode
        self.images = []

    def reset(self):
        '''Wrapper arround the reset function which also reset the images to be recorded'''
        self.images = []
        initial_state = self.env.reset()
        return initial_state

    def render(self):
        '''Wrapper arround the render function
        which adds the every image of the rendering to a list'''
        img = self.env.render(mode='rgb_array')
        self.images.append(img)

    def get_env(self):
        '''Getter function for the OpenAI Gym instance '''
        return self.env

    def get_state_size(self):
        '''Getter function for the state-size in the environment '''
        return self.state_size

    def get_action_size(self):
        '''Getter function for the state-size in the environment '''
        return self.action_size

    def get_random_action(self):
        '''Getter function for the state-size in the environment '''
        return self.env.action_space.sample()

    def make_gif(self, filename="render"):
        imageio.mimsave(filename + '.gif', [np.array(img) for i, img in enumerate(self.images) if i%2 == 0], fps=29)
        return Image(open(filename + '.gif','rb').read())

    def run_random_episode(self, render=False):
        current_state = self.reset()

        final_state = False
        iters = 0
        total_reward = 0

        while not final_state and iters < 500:
            if render:
                self.render()

            action = self.env.action_space.sample()
            current_state, reward, final_state, info = self.env.step(action)
            iters += 1
            total_reward += reward

        self.env.close()

        if render:
            return (total_reward, self.make_gif("random"))
        else:
            return total_reward
