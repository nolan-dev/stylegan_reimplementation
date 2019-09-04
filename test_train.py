# out of date

import glob
import os

import tensorflow as tf
import numpy as np

from models import Generator, Discriminator
from train import non_saturating_loss, l2_gp, get_interpolates, \
    wgan_gp, wasserstein_loss, drift_penalty, r1_gp, weight_following_ema_ops


class TestLosses(tf.test.TestCase):
    # Non-saturating
    def test_non_saturating_loss_good_discriminator(self):
        fake_logits = tf.constant([-1000.])
        real_logits = tf.constant([1000.])
        loss_disciminator, loss_generator = non_saturating_loss(real_logits, fake_logits)
        self.assertAllClose(loss_disciminator, 0.)
        self.assertAllGreater(loss_generator, 100.)

    def test_non_saturating_loss_reverse_discriminator(self):
        fake_logits = tf.constant([1000.])
        real_logits = tf.constant([-1000.])
        loss_disciminator, loss_generator = non_saturating_loss(real_logits, fake_logits)
        self.assertAllGreater(loss_disciminator, 100.)
        self.assertAllClose(loss_generator, 0.)

    def test_non_saturating_loss_chance_discriminator(self):
        fake_logits = tf.constant([0.])
        real_logits = tf.constant([0.])
        loss_disciminator, loss_generator = non_saturating_loss(real_logits, fake_logits)
        difference = tf.math.abs(loss_disciminator - loss_generator)
        self.assertAllLess(difference, 1.)  # D and G loss aren't too far away

    # Wasserstein
    def test_wasserstein_loss_good_discriminator(self):
        fake_logits = tf.constant([-1000.])
        real_logits = tf.constant([1000.])
        loss_disciminator, loss_generator = wasserstein_loss(real_logits, fake_logits)
        self.assertAllLess(loss_disciminator, -100)
        self.assertAllGreater(loss_generator, 100.)

    def test_wasserstein_loss_reverse_discriminator(self):
        fake_logits = tf.constant([1000.])
        real_logits = tf.constant([-1000.])
        loss_disciminator, loss_generator = wasserstein_loss(real_logits, fake_logits)
        self.assertAllGreater(loss_disciminator, 100.)
        self.assertAllLess(loss_generator, -100.)

    def test_wasserstein_loss_chance_discriminator(self):
        fake_logits = tf.constant([0.])
        real_logits = tf.constant([0.])
        loss_disciminator, loss_generator = wasserstein_loss(real_logits, fake_logits)
        difference = tf.math.abs(loss_disciminator - loss_generator)
        self.assertAllLess(difference, 1.)  # D and G loss aren't too far away

    def test_l2_gp_single(self):
        input = tf.constant([1.])
        output = input*2.
        penalty = l2_gp(input, output)

        self.assertAllEqual(penalty, 1.)

    def test_l2_gp_single_tensor(self):
        input = tf.constant([1., 2., 3.])
        output = tf.tensordot(input, [2., 2., -1.], axes=1)

        penalty = l2_gp(input, output)

        self.assertAllEqual(penalty, 4.)

    def test_l2_gp_batch(self):
        input = tf.constant([[1., 2., 3., 0.], [2., 3., 4., 1.]])
        output = tf.tensordot(input, [input[0][1], 2., -2., 2.], axes=1)
        # grads is [ 2.,  5., -2.,  2.],
        #          [ 2.,  2., -2.,  2.]
        penalty = tf.reduce_mean(l2_gp(input, output))

        self.assertAllClose(penalty, 17.41724, rtol=1e-4, atol=1e-4)

    def test_l2_gp_reduce(self):
        input = tf.constant([[1., 2.], [2., 3.]], shape=[1, 2, 2, 1])
        output = tf.reduce_mean(input*3)
        penalty = l2_gp(input, output)

        self.assertAllEqual(penalty, [.25])

    def test_l2_gp_variable(self):
        input = tf.constant([1.])
        output = input*tf.Variable(initial_value=2.)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            penalty = l2_gp(input, output)

        self.assertAllEqual(penalty, 1.)

    def test_get_interps(self):
        fake = tf.constant([1.])
        real = tf.constant([0.])
        tf.set_random_seed(1.)
        result = get_interpolates(real, fake)
        self.assertAllGreater(result, 0.)
        self.assertAllLess(result, 1.)

        tf.set_random_seed(1.)
        result1 = get_interpolates(real, fake)
        self.assertAllGreater(result, 0.)
        self.assertAllLess(result, 1.)

        tf.set_random_seed(1.)
        result2 = get_interpolates(real, fake)
        self.assertAllGreater(result, 0.)
        self.assertAllLess(result, 1.)

        tf.set_random_seed(1.)
        result3 = get_interpolates(real, fake)
        self.assertAllGreater(result, 0.)
        self.assertAllLess(result, 1.)

        tf.set_random_seed(1.)
        result4 = get_interpolates(real, fake)
        self.assertAllGreater(result, 0.)
        self.assertAllLess(result, 1.)

        self.assertNotAllClose(result1, result2)
        self.assertNotAllClose(result2, result3)
        self.assertNotAllClose(result3, result4)

    def test_wgan_gp(self):
        dis_model = lambda x, alpha: tf.reduce_mean(x*tf.Variable(initial_value=3.)*x[0][0][0][0])
        fake_image = tf.constant([2., 2., 2., 2.], shape=[1, 2, 2, 1])
        real_image = tf.constant([1., 1., 1., 1.], shape=[1, 2, 2, 1])
        penalty = wgan_gp(fake_image,
                          real_image,
                          dis_model,
                          alpha=1.,
                          alpha_interpolates=tf.constant(.5, shape=[real_image.get_shape().as_list()[0], 1]))
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
        self.assertAllClose(penalty, [24.53162], rtol=1e-4, atol=1e-4)

    def test_r1_gp(self):
        dis_model = lambda x, alpha: tf.reduce_mean(x*tf.Variable(initial_value=3.)*x[0][0][0][0])
        fake_image = tf.constant([2., 2., 2., 2.], shape=[1, 2, 2, 1])
        real_image = tf.constant([1., 1., 1., 1.], shape=[1, 2, 2, 1])
        penalty = r1_gp(fake_image,
                        real_image,
                        dis_model,
                        alpha=1.,
                        alpha_interpolates=tf.constant(.5, shape=[real_image.get_shape().as_list()[0], 1]))
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
        self.assertAllClose(penalty*2, [(9/4. + 6/4)**2 + 3*(3./4)**2], rtol=1e-4, atol=1e-4)

    def test_drift_penalty(self):
        test_input = tf.constant([1., 2., 3., 0.])
        penalty = drift_penalty(test_input)
        self.assertAllEqual(penalty, [1., 4., 9., 0.])

        test_input = tf.reshape(test_input, [1, 4])
        penalty = tf.reduce_mean(drift_penalty(test_input), axis=-1)
        self.assertAllEqual(penalty, [3.5])


class TestWeightFollowingEMAOps(tf.test.TestCase):
    def test_weight_following_ema_ops(self):
        class TestModel(tf.keras.Model):
            def __init__(self):
                super(TestModel, self).__init__()
                self.dense = tf.keras.layers.Dense(1, kernel_initializer=tf.ones_initializer())
            def call(self, x):
                return self.dense(x)
        test_model = TestModel()
        average_model = TestModel()
        average_model(tf.ones(shape=[1, 2]))
        result = test_model(tf.ones(shape=[1, 2]))
        initialization_ops = weight_following_ema_ops(average_model, test_model, decay=0.)
        train_op = [tf.assign(weight, weight+1) for weight in test_model.weights]
        with tf.control_dependencies(train_op):
            averaging_ops = weight_following_ema_ops(average_model, test_model)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(initialization_ops)
            self.assertAllEqual(tf.identity(average_model.weights[0]), [[1.], [1.]])
            sess.run(averaging_ops)
            self.assertAllEqual(tf.identity(average_model.weights[0]), tf.constant([[1.01], [1.01]]))




# class TestTraining(tf.test.TestCase):
#     def test_build_loss(self):
#         res = 4
#         input_regex = os.path.join("test_files", "*.png")
#         batch_size = 2
#         num_samples = 2 * 2 * 20
#         optimizer_g = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
#         optimizer_d = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
#         loss_fn = non_saturating_loss
#         gen_model = Generator(res)
#         dis_model = Discriminator(res)
#
#         files = glob.glob(input_regex)
#         loss_discriminator, loss_generator, fake_image, real_image, alpha_ph = \
#             build_loss(files,
#                        num_samples,
#                        batch_size,
#                        False,
#                        gen_model,
#                        dis_model,
#                        loss_fn)
#
#         train_step_d = optimizer_d.minimize(loss_discriminator, var_list=dis_model.trainable_variables)
#         train_step_g = optimizer_g.minimize(loss_generator, var_list=gen_model.trainable_variables)
#
#         with self.test_session() as sess:
#             sess.run(tf.global_variables_initializer())
#             old_gen_vars = [v.eval(session=sess) for v in gen_model.trainable_variables]
#             old_dis_vars = [v.eval(session=sess) for v in dis_model.trainable_variables]
#             old_dis_loss = loss_discriminator.eval(session=sess, feed_dict={alpha_ph: 1.})
#             for i in range(0, 30):
#                 sess.run(train_step_d, feed_dict={alpha_ph: 1.})
#             new_gen_vars = [v.eval(session=sess) for v in gen_model.trainable_variables]
#             new_dis_vars = [v.eval(session=sess) for v in dis_model.trainable_variables]
#             for old, new in zip(old_gen_vars, new_gen_vars):
#                 self.assertAllEqual(old, new)
#             self.assertNotAllClose(old_dis_vars, new_dis_vars)
#             self.assertAllClose(old_dis_vars, new_dis_vars, rtol=.1, atol=.1)
#             self.assertGreater(old_dis_loss, loss_discriminator.eval(
#                 session=sess,
#                 feed_dict={alpha_ph: 1.}))
#
#             old_gen_loss = loss_generator.eval(session=sess, feed_dict={alpha_ph: 1.})
#             for i in range(0, 30):
#                 sess.run(train_step_g, feed_dict={alpha_ph: 1.})
#             new_gen_vars = [v.eval(session=sess) for v in gen_model.trainable_variables]
#             new_dis_vars2 = [v.eval(session=sess) for v in dis_model.trainable_variables]
#             for old, new in zip(new_dis_vars, new_dis_vars2):
#                 self.assertAllEqual(old, new)
#             self.assertNotAllClose(old_gen_vars, new_gen_vars, rtol=1e-5)
#             self.assertAllClose(old_gen_vars, new_gen_vars, rtol=.1, atol=.1)
#             self.assertGreater(old_gen_loss, loss_generator.eval(
#                 session=sess,
#                 feed_dict={alpha_ph: 1.}))


if __name__ == "__main__":
    if not os.path.exists("test_files"):
        os.makedirs("test_files")
    tf.test.main()
