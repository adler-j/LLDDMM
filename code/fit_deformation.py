import adler
adler.util.gpu.setup_one_gpu()

from adler.odl.phantom import random_phantom
from adler.tensorflow import prelu, cosine_decay, ssim, psnr

import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow
from util import bilinear_sampler

sess = tf.InteractiveSession()


spc = odl.uniform_discr([-1, -1], [1, 1], [128, 128], interp='linear', dtype='float32')
img = odl.phantom.shepp_logan(spc, True)
displacement = spc.tangent_bundle.element([lambda x: 0.1 * x[0] ** 2,
                                           lambda x: 0.1 * x[1] ** 2])

img_deform = spc.element(odl.deform.linearized._linear_deform(img,
                                                              displacement))

img.show('img')
img_deform.show('img_deform')

x_0, y_0 = np.meshgrid(np.linspace(-1, 1, 128),
                       np.linspace(-1, 1, 128),
                       sparse=True)

f_0 = tf.constant(img.asarray()[None, ..., None])
f_1 = tf.constant(img_deform.asarray()[None, ..., None])

n = 10
v = tf.Variable(tf.random_normal([1, 128, 128, 2 * n], stddev=0.000001))

f_values = [f_0]

f_iter = f_0
for ni in range(n):
    xi = v[..., 2 * ni] + x_0[None, ...]
    yi = v[..., 2 * ni + 1] + y_0[None, ...]
    f_iter = bilinear_sampler(f_iter, xi, yi)

    f_values.append(f_iter)


with tf.name_scope('optimizer'):
    loss = 100 * tf.nn.l2_loss(v) + tf.nn.l2_loss(f_iter - f_1)
    opt_func = tf.train.AdamOptimizer(learning_rate=1e-4)
    optimizer = opt_func.minimize(loss)


callback = odl.solvers.CallbackShow(step=10) * (spc ** (n + 1)).element

sess.run(tf.global_variables_initializer())

for i in range(1000):
    _, f_values_result, loss_result = sess.run([optimizer, f_values, loss])

    print(loss_result)
    callback(f_values_result)
