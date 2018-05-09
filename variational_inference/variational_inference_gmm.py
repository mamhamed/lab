import edward as ed
from edward.models import InverseGamma, Normal, Dirichlet, Categorical, Empirical
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


"""
Note: ed.KLqp is known not to work very well with Gamma or Dirichlet variational approximations. For high-dimensional and large GMM problems,
you should generally stick with Gibbs sampling or coordinate ascent VI
"""
N = 30000

true_mu = np.array([-3.0, 0.0, 3.0], np.float32)
true_sigma = np.array([1.0, 1.0, 1.0], np.float32)

true_pi = np.array([.2, .3, .5], np.float32)
K = len(true_pi)
true_z = np.random.choice(np.arange(K), p=true_pi, size=N)
x = true_mu[true_z] + np.random.randn(N) * true_sigma[true_z]

# plt.hist(x, bins=200)
# plt.show()

# we like to calculate posterior p(\theta|x) where \theta=[mu_1,..., mu_3, sigma_1,...,sigma_3, z_1,...,z_3]

# Model
pi = Dirichlet(np.ones(K, np.float32))
mu = Normal(0.0, 9.0, sample_shape=[K])
sigma = InverseGamma(1.0, 1.0, sample_shape=[K])

c = Categorical(logits=tf.log(pi) - tf.log(1.0 - pi), sample_shape=N)
ed_x = Normal(loc=tf.gather(mu, c), scale=tf.gather(sigma, c))

# parameters
q_pi = Dirichlet(tf.nn.softplus(tf.get_variable("qpi", [K], initializer=tf.constant_initializer(1.0 / K))))
q_mu = Normal(loc=tf.get_variable("qmu", [K]), scale=1.0)
q_sigma = Normal(loc=tf.nn.softplus(tf.get_variable("qsigma", [K])), scale=1.0)


inference = ed.KLqp(latent_vars={mu: q_mu, sigma: q_sigma},
                    data={ed_x: x}
                    )  # this will fail if we include qpi: pi

inference.run(n_iter=1000)

print q_pi.value().eval()

print q_mu.value().eval()
print q_sigma.value().eval()
