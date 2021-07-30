if __name__ == "__main__":
    import sys
    import tensorflow.compat.v1 as tf
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import pandas as pd
    import os
    import logging
    import argparse
    from warnings import simplefilter

    from algorithms.utils.utils import get_data_location, get_env_list, prepare_data_directory, store_training_config

    logging.disable(logging.WARNING)
    simplefilter(action='ignore', category=UserWarning)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    env_list = get_env_list()
    alg_list = ['es', 'td3', 'ddpg']

    parser = argparse.ArgumentParser(description='Process user inputs')
    parser.add_argument("-a", "--algorithm",
                        action='store', required=True,
                        choices=alg_list,
                        help=f"The algorithm to be used for the \
                        training. Pick from: {', '.join(alg_list)}")
    parser.add_argument("-e", "--env_name",
                        action='store', required=True,
                        help="The environment on which to train the agent. \
                        Pick from the list of OpenAI Gym environments")
    parser.add_argument("-n", "--training_name", action='store',
                        required=True,
                        help="The name for the directory where training \
                        informations are stored")
    parser.add_argument('-alr', '--actor_learning_rate', action='store',
                        default=0.001, type=float,
                        help='Learning rate for the policy estimator')
    parser.add_argument('-clr', '--critic_learning_rate',
                        default=0.001, type=float,
                        help='Learning rate for the state value estimator')
    parser.add_argument('-g', '--gamma',
                        default=0.99, type=float,
                        help='Future discount factor')
    parser.add_argument('-t', '--tau',
                        default=0.001, type=float,
                        help='Update rate for the target networks parameter')
    parser.add_argument('-bas', '--batch_size',
                        default=50, type=int,
                        help='Batch size for the updates')
    parser.add_argument('-and', '--actor_noise_dev',
                        default=0.1, type=float,
                        help='Standard deviation for the exploration noise component')
    parser.add_argument('-tnd', '--target_noise_dev',
                        default=0.2, type=float,
                        help='Standard deviation for the smoothing noise component')
    parser.add_argument('-rs', '--random_seed',
                        default=3, type=int,
                        help='random seed for the experiment')
    parser.add_argument('-ua', '--update_after',
                        default=int(1 * 1e3), type=int,
                        help='when to start the updates')
    parser.add_argument('-ue', '--update_every',
                        default=50, type=int,
                        help='frequency at which to perform the updates')
    parser.add_argument('-nc', '--noise_clip',
                        default=0.5, type=float,
                        help='clip val for the smoothing noise component')
    parser.add_argument('-bus', '--buffer_size',
                        default=int(1 * 1e7), type=int,
                        help='Size for the replay memory buffer')
    parser.add_argument('-ss', '--start_steps',
                        default=int(5 * 1e4), type=int,
                        help='start sampling from the networks')
    parser.add_argument('-pd', '--policy_delay',
                        default=4, type=int,
                        help='policy delay')

    flags = parser.parse_args()

    if flags.env_name not in env_list:
        parser.error("-e --env_name has to be a valid Gym environment.")

    res_path = get_data_location(flags.training_name, flags.algorithm)
    # prepare_data_directory(res_path)
    # store_training_config(res_path, vars(flags))

    tf.disable_eager_execution()
    tf.set_random_seed(flags.random_seed)

    if flags.algorithm == 'td3':
        from algorithms.contcontrol.td3_agent_class import Agent
    elif flags.algorithm == 'ddpg':
        from algorithms.contcontrol.ddpg_agent_class import Agent
    elif flags.algorithm == 'es':
        from algorithms.evstrat.es_agent_class import Population

    # Create and train the agent
    pop = Population(flags, flags.env_name, res_path)
    pop.train()
