# out of date

import tensorflow as tf
import numpy as np

from ops import name_scope
from ops import adaptive_instance_norm
from ops import upsample
from ops import pixel_norm
from ops import minibatch_stddev
from ops import downsample
from ops import apply_binomial_filter

class TestPixelNorm(tf.test.TestCase):
    def test_pixel_norm_output(self):
        test_input = tf.constant([1., 2., 3., 4, 5., 6., 7., 8], shape=[1, 2, 2, 2])
        output = pixel_norm(test_input)
        target = [1./np.sqrt((1.**1+2.**2)/2), 2./np.sqrt((1.**2+2.**2)/2),
                  3./np.sqrt((3.**2+4.**2)/2), 4./np.sqrt((3.**2+4.**2)/2),
                  5./np.sqrt((5.**2+6.**2)/2), 6./np.sqrt((5.**2+6.**2)/2),
                  7./np.sqrt((7.**2+8.**2)/2), 8./np.sqrt((7.**2+8.**2)/2)]
        target = np.reshape(target, [1, 2, 2, 2])
        self.assertAllClose(output, target)


class TestNameScopeDecorator(tf.test.TestCase):
    def test_name_scope_decorator(self):
        decorator = name_scope("test")

        def test_func(float_arg):
            self.assertEqual(tf.get_default_graph().get_name_scope(), "test")
            return tf.constant(float_arg)

        decorated_func = decorator(test_func)
        result = decorated_func(2.)
        self.assertAllEqual(result, tf.constant(2.))
        self.assertEqual(result.name, "test/Const:0")


class TestAdaptiveInstanceNorm(tf.test.TestCase):
    def test_adaptive_instance_norm_ys2_ybneg1(self):
        # channels first is a bit more intuitive to read, but
        # must transpose to get correct shape for normal channel
        # last operations. transpose moves it from
        # (batch, channels, h, w) to (batch, h, w, channels)
        test_input = tf.transpose(
            tf.constant([[  # channel 1:
                [[-3., 2.],
                 [8., -3.]],
                # channel 2:
                [[-3., 2.],
                 [8., -3.]]
            ]]),
            (0, 2, 3, 1))  # (batch, h, w, c)
        test_ys = tf.constant([2., 2.])
        test_yb = tf.constant([-1., -1.])
        x = adaptive_instance_norm(test_input, test_ys, test_yb)
        x = tf.transpose(x, (0, 3, 1, 2))
        self.assertAllClose(x, tf.constant([[[-2.76690442, -0.5582739],
                                             [2.09208273, -2.76690442]],

                                            [[-2.76690442, -0.5582739],
                                             [2.09208273, -2.76690442]]], shape=(1, 2, 2, 2)))

    def test_adaptive_instance_norm_ys1_yb0(self):
        # channels first is a bit more intuitive to read, but
        # must transpose to get correct shape for normal channel
        # last operations. transpose moves it from
        # (batch, channels, h, w) to (batch, h, w, channels)
        test_input = tf.transpose(
            tf.constant([[  # channel 1:
                [[-1., 0.],
                 [2., -1.]],
                # channel 2:
                [[-1., 0.],
                 [2., -1.]]
            ]]),
            (0, 2, 3, 1))  # (batch, h, w, c)
        test_ys = tf.constant([1., 1.])
        test_yb = tf.constant([0., 0.])
        x = adaptive_instance_norm(test_input, test_ys, test_yb)
        x = tf.transpose(x, (0, 3, 1, 2))
        self.assertAllClose(x, tf.constant([[[-0.81649658, 0.],
                                             [1.63299316, -0.81649658]],

                                            [[-0.81649658, 0.],
                                             [1.63299316, -0.81649658]]], shape=(1, 2, 2, 2)))
        x_mean, x_var = tf.nn.moments(x, axes=(2, 3))
        self.assertAllClose(x_mean, tf.constant([[0., 0.]]))
        self.assertAllClose(x_var, tf.constant([[1., 1.]]))


class TestUpsample(tf.test.TestCase):
    def test_upsample_nn(self):
        test_input_spatial = [[0., 1.],
                              [2., 3.]]
        test_input = tf.transpose(tf.constant([[test_input_spatial]*3]*2, dtype=tf.float32),
                                  (0, 2, 3, 1))  # b, h, w, c
        x = upsample(test_input, method='nearest_neighbor')
        spatial_target = [[0., 0., 1., 1.],
                          [0., 0., 1., 1.],
                          [2., 2., 3., 3.],
                          [2., 2., 3., 3.]]
        target_array = tf.constant([[spatial_target]*3]*2)  # b, c, h, w
        x = tf.transpose(x, (0, 3, 1, 2))  # b, c, h, w
        self.assertAllEqual(x, target_array)

    def test_upsample_nn_factor4(self):
        test_input_spatial = [[0., 1.],
                              [2., 3.]]
        test_input = tf.transpose(tf.constant([[test_input_spatial]*3]*2, dtype=tf.float32),
                                  (0, 2, 3, 1))  # b, h, w, c
        x = upsample(test_input, method='nearest_neighbor', factor=4)
        spatial_target = [[0., 0., 0., 0., 1., 1., 1., 1.],
                          [0., 0., 0., 0., 1., 1., 1., 1.],
                          [0., 0., 0., 0., 1., 1., 1., 1.],
                          [0., 0., 0., 0., 1., 1., 1., 1.],
                          [2., 2., 2., 2., 3., 3., 3., 3.],
                          [2., 2., 2., 2., 3., 3., 3., 3.],
                          [2., 2., 2., 2., 3., 3., 3., 3.],
                          [2., 2., 2., 2., 3., 3., 3., 3.]]
        target_array = tf.constant([[spatial_target]*3]*2)  # b, c, h, w
        x = tf.transpose(x, (0, 3, 1, 2))  # b, c, h, w
        self.assertAllEqual(x, target_array)

    def test_upsample_bilinear(self):
        test_input_spatial = [[0., .1],
                              [.2, .3]]
        test_input = tf.transpose(tf.constant([[test_input_spatial]*3]*2, dtype=tf.float32),
                                  (0, 2, 3, 1))  # b, h, w, c
        x = upsample(test_input, method='bilinear')
        # skimage.transform.resize (mode='edge') result (a bit different than tf.image.resize_bilinear)
        spatial_target = [[0., 0.025, 0.075, 0.1],
                          [0.05, 0.075, 0.125, 0.15],
                          [0.15, 0.175, 0.225, 0.25],
                          [0.2, 0.225, 0.275, 0.3]]
        target_array = tf.constant([[spatial_target]*3]*2)  # b, c, h, w
        x = tf.transpose(x, (0, 3, 1, 2))  # b, c, h, w
        self.assertAllClose(x, target_array, atol=.02)

    def test_upsample_nn_inverted_by_avg_pool(self):
        test_input = tf.constant(np.random.normal(0., 1., size=[2, 4, 4, 3]), dtype=tf.float32)
        up_x = upsample(test_input, "nearest_neighbor")
        down_x = tf.nn.avg_pool(up_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        self.assertAllEqual(down_x, test_input)

    # # bilinear interpolation probably performs poorly on random data
    # def test_upsample_bilinear_inverted_by_bilinear(self):
    #     test_input = tf.constant(np.random.normal(0., 1., size=[2, 8, 8, 3]), dtype=tf.float32)
    #     up_x = upsample(test_input, "bilinear")
    #     down_x = downsample(up_x, "bilinear")
    #     np.set_printoptions(threshold=np.nan, suppress=True)
    #     with self.test_session() as sess:
    #         print(sess.run(test_input))
    #         print("******")
    #         print(sess.run(down_x))
    #     self.assertAllClose(down_x, test_input)

    def test_upsample_bilinear_inverted_by_bilinear(self):
        test_input = tf.reshape(tf.constant(np.arange(0, 2*8*8*3)/(2*8*8*3), dtype=tf.float32),
                                [2, 8, 8, 3])
        up_x = upsample(test_input, "bilinear")
        down_x = downsample(up_x, "bilinear")
        np.set_printoptions(threshold=np.nan, suppress=True)
        self.assertAllClose(down_x, test_input, atol=.02)

class TestApplyBinomialFilter(tf.test.TestCase):
    def test_apply_binomial_filter_gradients(self):
        img = tf.random_normal([2, 128, 128, 3], 0., 1.)
        filtered = apply_binomial_filter(img)
        gradients = tf.gradients(filtered, img)
        self.assertAllClose(tf.reduce_mean(gradients), 1., atol=.1)

        self.fail("unfinished")
        second_gradients = tf.gradients(gradients, img)
        self.assertAllClose(second_gradients, [0.])


class TestDownsample(tf.test.TestCase):
    def test_downsample_avg(self):
        test_input_spatial = [[0., 0., 1., 1.],
                              [0., 0., 1., 1.],
                              [2., 2., 3., 3.],
                              [2., 2., 3., 3.]]
        test_input = tf.transpose(tf.constant([[test_input_spatial]*3]*2),
                                  (0, 2, 3, 1))  # b, h, w, c
        x = downsample(test_input, method='nearest_neighbor')
        spatial_target = [[0., 1.],
                          [2., 3.]]
        target_array = tf.constant([[spatial_target]*3]*2)  # b, c, h, w
        #x = tf.transpose(x, [0, 3, 1, 2])  # b, c, h, w
        target_array = tf.transpose(target_array, [0, 2, 3, 1])
        self.assertAllEqual(x, target_array)

    def test_downsample_avg_factor_4(self):
        test_input_spatial = [[0., 0., 0., 0., 1., 1., 1., 1.],
                              [0., 0., 0., 0., 1., 1., 1., 1.],
                              [0., 0., 0., 0., 1., 1., 1., 1.],
                              [0., 0., 0., 0., 1., 1., 1., 1.],
                              [2., 2., 2., 2., 3., 3., 3., 3.],
                              [2., 2., 2., 2., 3., 3., 3., 3.],
                              [2., 2., 2., 2., 3., 3., 3., 3.],
                              [2., 2., 2., 2., 3., 3., 3., 3.]]
        test_input = tf.transpose(tf.constant([[test_input_spatial]*3]*2),
                                  (0, 2, 3, 1))  # b, h, w, c
        x = downsample(test_input, method='nearest_neighbor', factor=4)
        spatial_target = [[0., 1.],
                          [2., 3.]]
        target_array = tf.constant([[spatial_target]*3]*2)  # b, c, h, w
        #x = tf.transpose(x, [0, 3, 1, 2])  # b, c, h, w
        target_array = tf.transpose(target_array, [0, 2, 3, 1])
        self.assertAllEqual(x, target_array)

    def test_downsample_bilinear(self):
        test_input_spatial = np.resize(np.arange(0, 16*16)/256., [16, 16]).tolist()
        test_input = tf.transpose(tf.constant([[test_input_spatial]*3]*2),
                                  (0, 2, 3, 1)) # b, h, w, c
        x = downsample(test_input, "bilinear")
        # skimage.transform.resize result (a bit different than tf.image.resize_bilinear)
        spatial_target = [[0.03320313, 0.04101563, 0.04882813, 0.05664063, 0.06445313, 0.07226563, 0.08007813, 0.08789063],
                          [0.15820313, 0.16601563, 0.17382813, 0.18164063, 0.18945313, 0.19726563, 0.20507813, 0.21289063],
                          [0.28320313, 0.29101563, 0.29882813, 0.30664063, 0.31445313, 0.32226563, 0.33007813, 0.33789063],
                          [0.40820312, 0.41601563, 0.42382813, 0.43164063, 0.43945313, 0.44726563, 0.45507813, 0.46289063],
                          [0.53320312, 0.54101562, 0.54882812, 0.55664063, 0.56445313, 0.57226563, 0.58007813, 0.58789063],
                          [0.65820312, 0.66601562, 0.67382813, 0.68164063, 0.68945313, 0.69726563, 0.70507813, 0.71289063],
                          [0.78320312, 0.79101562, 0.79882812, 0.80664062, 0.81445312, 0.82226563, 0.83007813, 0.83789063],
                          [0.90820312, 0.91601562, 0.92382812, 0.93164062, 0.93945312, 0.94726563, 0.95507813, 0.96289063]]
        target_array = tf.constant([[spatial_target]*3]*2)  # b, c, h, w
        target_array = tf.transpose(target_array, [0, 2, 3, 1])
        self.assertAllClose(x, target_array, atol=.02)


class TestMinibatchStddev(tf.test.TestCase):
    def test_minibatch_stddev(self):
        test_input = tf.constant(np.arange(0, 6), dtype=tf.float32) # 3x2x2x2
        test_input = tf.reshape(test_input, [3, 2, 1, 1])
        avg_stddev = tf.sqrt(8/3.)
        target = tf.concat([test_input,
                            tf.ones_like(test_input)*avg_stddev], axis=-1)
        output = minibatch_stddev(test_input)
        self.assertEqual(output.get_shape().as_list(), [3, 2, 1, 2])
        self.assertAllEqual(output, target)


if __name__ == "__main__":
    tf.test.main()
