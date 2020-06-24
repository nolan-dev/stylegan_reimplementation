import tensorflow as tf
import numpy as np
# includes code from https://github.com/NVlabs/stylegan for comparison purposes

def name_scope(name):
    """
    :param name: enclose the function in tf.name_scope with given name
    :return: decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with tf.name_scope(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


@name_scope("adaptive_instance_norm")
def adaptive_instance_norm(x, ys, yb):
    """
    :param x: (batch, h, w, c)
    :param ys: (c)
    :param yb: (c)
    :return: (batch, h, w, c)
    """
    batch_size, height, width, channels = x.get_shape().as_list()
    x_mean, x_var = tf.nn.moments(x, axes=(1, 2))

    x_mean = tf.reshape(x_mean, [batch_size, 1, 1, channels])
    x_std = tf.reshape(tf.sqrt(x_var), [batch_size, 1, 1, channels])
    x -= x_mean
    x /= x_std + 1e-8
    """
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)
    x -= tf.reduce_mean(x, axis=[2, 3], keepdims=True)
    epsilon = tf.constant(1e-8, dtype=x.dtype, name='epsilon')
    x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[2, 3], keepdims=True) + epsilon)
    x = tf.cast(x, orig_dtype)
    """
    
    ys = tf.reshape(ys, [batch_size, 1, 1, channels])
    x *= ys
    yb = tf.reshape(yb, [batch_size, 1, 1, channels])
    x += yb
    return x


@name_scope("pixel_norm")
def pixel_norm(x):
    return x * tf.rsqrt(tf.reduce_mean(tf.square(x), keep_dims=True, axis=-1)+1e-8)


#@tf.RegisterGradient("DepthwiseConv2dNativeBackpropInput")
#def _DepthwiseConv2dNativeBackpropInputGrad(op, grad):
#    return None, None, tf.zeros_like(op.inputs[2])


@name_scope("binomial_filter")
def apply_binomial_filter_(img):
    shape = img.get_shape().as_list()
    binomial_filter = tf.constant([[1., 2., 1.],
                                   [2., 4., 2.],
                                   [1., 2., 1.]], dtype=tf.float32) / 16.
    filter_for_each_channel = tf.tile(tf.reshape(binomial_filter, [3, 3, 1, 1]),
                                      [1, 1, shape[-1], 1])
    img = tf.pad(img, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
    img = tf.nn.depthwise_conv2d(img,
                                 filter_for_each_channel,
                                 strides=[1, 1, 1, 1],
                                 padding='VALID')
    return img


@name_scope("upsample")
def upsample(img, method, factor=2):
    """
    :param img: spatial data to upscale
    :param method: bilinear or nearest_neighbor
    :return: spatial data upsampled by a factor of 2
    """
    shape = img.get_shape().as_list()
    img = tf.expand_dims(img, axis=-1)
    img = tf.tile(img, [1, 1, 1, 1, factor**2])
    img = tf.reshape(img, shape=[-1, shape[1], shape[2], shape[3], factor, factor])
    img = tf.transpose(img, perm=[0, 1, 4, 2, 5, 3])  # batch, height, 2, width, 2, channels
    img = tf.reshape(img, shape=[-1, shape[1]*factor, shape[2]*factor, shape[3]])
    if method == 'bilinear':
        # basic tensorflow resize bilinear has no second order derivatives
        # img = tf.image.resize_bilinear(img, [shape[1]*2, shape[2]*2])
        # This method is similar to skimage.transform.resize, and implemented
        # in a way similar to how the paper describes:
        img = apply_binomial_filter(img)
    elif method != 'nearest_neighbor':
        raise ValueError("unknown resize method %s"%method)
    return img

def apply_binomial_filter(x, f=[1,2,1], normalize=True):
    with tf.variable_scope('Blur2D'):
        @tf.custom_gradient
        def func(x):
            y = apply_binomial_filter_(x)
            @tf.custom_gradient
            def grad(dy):
                def gradgrad(ddx):
                    return apply_binomial_filter_(ddx)
                dx = apply_binomial_filter_(dy)
                return dx, gradgrad
            return y, grad
        return func(x)

def _blur2d_nv(x, f=[1,2,1], normalize=True, flip=False, stride=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(stride, int) and stride >= 1

    # Finalize filter kernel.
    f = np.array(f, dtype=np.float32)
    if f.ndim == 1:
        f = f[:, np.newaxis] * f[np.newaxis, :]
    assert f.ndim == 2
    if normalize:
        f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1]
    f = f[:, :, np.newaxis, np.newaxis]
    f = np.tile(f, [1, 1, int(x.shape[-1]), 1])

    # No-op => early exit.
    if f.shape == (1, 1) and f[0,0] == 1:
        return x

    # Convolve using depthwise_conv2d.
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)  # tf.nn.depthwise_conv2d() doesn't support fp16
    f = tf.constant(f, dtype=x.dtype, name='filter')
    strides = [1, stride, stride, 1]
    x = tf.nn.depthwise_conv2d(x, f, strides=strides, padding='SAME', data_format='NHWC')
    x = tf.cast(x, orig_dtype)
    return x

def downsample_nv(x):
    s = tf.shape(x)
    x = tf.reshape(x, [-1, s[1] // 2, 2, s[2] // 2, 2, s[3]])
    x = tf.reduce_mean(x, axis=[2, 4], keepdims=False)
    return x

@name_scope("downsample")
def downsample(img, method='bilinear', factor=2):
    """
    :param img: spatial data to downsample
    :param method: bilinear or nearest_neighbor (average)
    :return: spatial data downsampled by a factor of 2
    """
    shape = img.get_shape().as_list()
    if method == 'bilinear':
        # basic tensorflow resize bilinear has no second order derivatives:
        # img = tf.image.resize_bilinear(img, [shape[1]//2, shape[2]//2])
        # This method is similar to skimage.transform.resize, and implemented
        # in a way similar to how the paper describes:
        img = apply_binomial_filter(img)
    elif method != 'nearest_neighbor':
        raise ValueError("unknown resize method %s"%method)
    img = tf.nn.avg_pool(img, ksize=[1, factor, factor, 1], strides=[1, factor, factor, 1], padding='SAME')
    return img


@name_scope("minibatch_stddev")
def minibatch_stddev(x):
    shape = x.get_shape().as_list()
    #_, var = tf.nn.moments(x, axes=0)
    var = tf.reduce_mean((x-tf.reduce_mean(x, axis=0, keepdims=True))**2, axis=0, keepdims=True) + 1e-8
    stddev = tf.sqrt(var)
    average = tf.reduce_mean(stddev, keepdims=True)
    stddev_map = tf.tile(average, [shape[0], shape[1], shape[2], 1])
    return tf.concat([x, stddev_map], axis=-1)
