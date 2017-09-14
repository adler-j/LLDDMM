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


def dx(t):
    return t[:, 1:, :, :] - t[:, :-1, :, :]
def dy(t):
    return t[:, :, 1:, :] - t[:, :, :-1, :]


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
                       sparse=False)

f_0 = tf.constant(img.asarray()[None, ..., None])
f_1 = tf.constant(img_deform.asarray()[None, ..., None])

n = 5
v = tf.Variable(tf.random_normal([1, 128, 128, 2 * n], stddev=0.000001))

x_0_img = tf.constant(x_0[None, ..., None], dtype='float32')
y_0_img = tf.constant(y_0[None, ..., None], dtype='float32')

x_iter = x_0_img
y_iter = y_0_img
for ni in range(n):
    xi = v[..., 2 * ni] + x_0[None, ...]
    yi = v[..., 2 * ni + 1] + y_0[None, ...]
    x_iter = bilinear_sampler(x_iter, xi, yi)
    y_iter = bilinear_sampler(y_iter, xi, yi)

f_result = bilinear_sampler(f_0, x_iter[..., 0], y_iter[..., 0])

with tf.name_scope('optimizer'):
    loss = (tf.nn.l2_loss(v) +
            10 * tf.nn.l2_loss(dx(v)) +
            10 * tf.nn.l2_loss(dy(v)) +
            tf.nn.l2_loss(f_result - f_1))
    opt_func = tf.train.AdamOptimizer(learning_rate=1e-3)
    optimizer = opt_func.minimize(loss)


callback = odl.solvers.CallbackShow(step=1) * spc.element

sess.run(tf.global_variables_initializer())

for i in range(10000):
    _, f_result_result, loss_result = sess.run([optimizer, f_result, loss])

    print(loss_result)
    callback(f_result_result)
