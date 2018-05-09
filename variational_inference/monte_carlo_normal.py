import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np

D = np.random.normal(loc=3, scale=2, size=10)


with pm.Model() as model:
    prior_mu = pm.Uniform('mu', 0, 5)
    prior_sd = pm.Uniform('sd', 0, 5)

    x_obs = pm.Normal('y', prior_mu, prior_sd, observed=D)
    trace = pm.sample(draws=10000)
    plt.hist2d(trace['mu'], trace['sd'], bins=200)
    plt.show()