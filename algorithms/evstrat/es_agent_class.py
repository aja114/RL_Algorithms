import numpy as np
import tensorflow.compat.v1 as tf
import random

from ..base import env_class, agent_class


class Agent(agent_class.Agent):

    def __init__(self, TF_FLAGS, env_name, res_folder):
        ''' This class build the Agent that learns in the environment via the actor-critic algorithm. '''

        agent_class.Agent.__init__(self, TF_FLAGS, env_name, res_folder)

    def get_action(self, state):
        pass

    def get_action_training(self, state):
        pass

    def update_agent(self):
        pass
