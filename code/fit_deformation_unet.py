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


def random_data(ref=False):
    if ref:
        img = odl.phantom.shepp_logan(spc, True)
        displacement = spc.tangent_bundle.element([lambda x: 0.05 * x[0] ** 2 + 0.1 * x[1],
                                                   lambda x: 0.05 * x[1] ** 2])
    else:
        img = adler.odl.phantom.random_phantom(spc)
        a, b, c, d = 0.1 * np.random.randn(4)

        displacement = spc.tangent_bundle.element([lambda x: a * x[0] + b * x[0] ** 2,
                                                   lambda x: c * x[1] + d * x[1] ** 2])

    img_deform = spc.element(odl.deform.linearized._linear_deform(img,
                                                                  displacement))

    return img, img_deform

x_0, y_0 = np.meshgrid(np.linspace(-1, 1, 128),
                       np.linspace(-1, 1, 128),
                       sparse=True)

f_0 = tf.placeholder(tf.float32, [1, 128, 128, 1])
f_1 = tf.placeholder(tf.float32, [1, 128, 128, 1])
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')

f = tf.concat([f_0, f_1], -1)

n = 10
v = adler.tensorflow.reference_unet(f, 2 * n,
                                    use_batch_norm=False,
                                    is_training=is_training)

v = v / (10 * n)

f_values = [f_0]

f_iter = f_0
for ni in range(n):
    xi = v[..., 2 * ni] + x_0[None, ...]
    yi = v[..., 2 * ni + 1] + y_0[None, ...]
    f_iter = bilinear_sampler(f_iter, xi, yi)

    f_values.append(f_iter)


with tf.name_scope('optimizer'):
    loss = (tf.nn.l2_loss(v) +
            10 * tf.nn.l2_loss(dx(v)) +
            10 * tf.nn.l2_loss(dy(v)) +
            tf.nn.l2_loss(f_iter - f_1))
    opt_func = tf.train.AdamOptimizer(learning_rate=1e-4)
    optimizer = opt_func.minimize(loss)


callback = odl.solvers.CallbackShow(step=10) * (spc ** (n + 1)).element

sess.run(tf.global_variables_initializer())


img_v, img_deform_v = random_data(True)
img_v.show('img')
img_deform_v.show('img_deform')

for i in range(100000):
    img, img_deform = random_data(False)

    _ = sess.run([optimizer],
                 feed_dict={f_0: img.asarray()[None, ..., None],
                            f_1: img_deform.asarray()[None, ..., None],
                            is_training: True})

    f_values_result, loss_result = sess.run([f_values, loss],
                                            feed_dict={f_0: img_v.asarray()[None, ..., None],
                                                       f_1: img_deform_v.asarray()[None, ..., None],
                                                       is_training: False})

    print(loss_result)
    callback(f_values_result)
