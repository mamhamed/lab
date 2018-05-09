import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np

# D = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
D = np.concatenate([np.zeros(70), np.ones(30)])

with pm.Model() as model:
    prior = pm.Uniform('p', 0, 1)
    x_obs = pm.Bernoulli('y', prior, observed=D)
    beta_binomial_trace = pm.sample(draws=10000)
    plt.hist(beta_binomial_trace['p'], bins=200)
    plt.show()