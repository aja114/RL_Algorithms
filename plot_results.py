#!/Users/utilisateur/opt/anaconda3/bin/python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sys import argv
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


results_dirs = []
direc = os.chdir('Data')
if len(argv) > 1:
    dirs = argv[1:]
    print(dirs)
    for d in dirs:
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "result.csv")):
            results_dirs.append(d)
else:
    for d in os.listdir():
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "result.csv")):
            results_dirs.append(d)

for d in results_dirs:
    algorithm = d.split('_')[0].upper()
    name = ' '.join(d.split('_')[1:])
    res_file = os.path.join(d, "result.csv")
    df = pd.read_csv(res_file)
    df = df.rename(
        columns={df.columns[0]: "episode", df.columns[-1]: "total_reward"})
    ep = df.episode
    r = df.total_reward
    window_size = int(len(ep) / 20)
    ma = moving_average(r, window_size)

    plt.figure(figsize=(15, 7))
    plt.xlabel("episodes")
    plt.ylabel("rewards")
    plt.title(f"{algorithm} results for the {name} environment")
    plt.scatter(ep, r, marker='+', c='b', s=30, linewidth=1,
                alpha=0.5, label="total rewards")
    plt.plot(ep[window_size - 1:], ma, c='r',
             alpha=0.7, label="reward moving average")
    plt.legend()
    plt.savefig(os.path.join(d, "results.png"))
    plt.show()
