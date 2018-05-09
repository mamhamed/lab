import edward as ed
from edward.models import Normal, InverseGamma
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

N = 1000
D = np.random.normal(loc=3., scale=2., size=N)

p_mu = Normal(0., 1.)
p_s = InverseGamma(1., 1.)  # https://en.wikipedia.org/wiki/Inverse-gamma_distribution

ed_normal = Normal(loc=p_mu, scale=p_s, sample_shape=N)

q1 = Normal(loc=tf.get_variable("mu", []), scale=1.0)
q2 = Normal(loc=tf.nn.softplus(tf.get_variable("sd", [])), scale=1.0)

inference = ed.KLqp(latent_vars={p_mu: q1, p_s: q2},
                    data={ed_normal: D}
                    )

inference.run(n_iter=10000)

print np.mean(D)
print np.std(D)
plt.hist2d(q1.sample(10000).eval(), q2.sample(10000).eval(), bins=200)
plt.show()
