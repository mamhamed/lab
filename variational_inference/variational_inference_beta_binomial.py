import edward as ed
from edward.models import Bernoulli, Beta, Uniform
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# D = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
D = np.concatenate([np.zeros(70), np.ones(30)])

p = Uniform(0., 1.)

ed_beta_binomial = Bernoulli(probs=p, sample_shape=len(D))

qp = Beta(concentration1=tf.nn.softplus(tf.get_variable("alpha", [])),
          concentration0=tf.nn.softplus(tf.get_variable("beta", []))
          )

inference = ed.KLqp({p: qp},
                    {ed_beta_binomial: D})

inference.run(n_iter=1000)

plt.hist(qp.sample(10000).eval(), bins=200)
plt.show()