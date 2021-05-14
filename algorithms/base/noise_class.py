import numpy as np

class OUNoise:

    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.count_use = 0
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        if self.count_use > 100000:
            self.count_use = 0
            self.std_dev = self.std_dev / 1.5

        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x

        self.count_use += 1

        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class GaussNoise:

    def __init__(self, mean=0, std_deviation=0.1, clip=0.3, size=None):
        self.mean = mean
        self.std = std_deviation
        self.size = size
        self.c = clip

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = np.random.normal(self.mean, self.std, self.size)
        x = np.clip(x, -self.c, self.c)

        return x
