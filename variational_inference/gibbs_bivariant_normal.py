import edward as ed
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np


"""
Inspired by https://wiseodd.github.io/techblog/2015/10/09/gibbs-sampling/
"""

sess = tf.Session()

mu = [3., 2.]
cov = [[1, -0.6], [-0.6, 1]]
scale = tf.cholesky(cov)
mvn = ed.models.MultivariateNormalTriL(loc=mu, scale_tril=scale)

data = mvn.sample(5000).eval(session=sess)
plt.hist2d([x[0] for x in data], [x[1] for x in data], bins=100)
plt.show()

x0 = tf.placeholder(shape=[], dtype=tf.float32)
x1 = tf.placeholder(shape=[], dtype=tf.float32)
normal0_given1 = ed.models.Normal(loc=mu[0]+cov[0][1]/cov[1][1]*(x1-mu[1]), scale=cov[1][1]-cov[0][1]**2/cov[0][0])
normal1_given0 = ed.models.Normal(loc=mu[1]+cov[1][0]/cov[0][0]*(x0-mu[0]), scale=cov[0][0]-cov[1][0]**2/cov[1][1])
N = 1000
with sess:
    x1_sample = 0
    data_sampled = [[0, 0]] * N
    for i in range(N):
        x0_sample = sess.run(normal0_given1.sample(1), {x1: x1_sample})[0]
        x1_sample = sess.run(normal1_given0.sample(1), {x0: x0_sample})[0]
        data_sampled[i] = [x0_sample, x1_sample]

        print i

    plt.hist2d([x[0] for x in data_sampled], [x[1] for x in data_sampled], bins=30)
    plt.show()
