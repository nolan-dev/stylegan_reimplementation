import tensorflow as tf
import numpy as np

from ops import upsample, downsample, pixel_norm, minibatch_stddev, adaptive_instance_norm


class Conv2D(tf.keras.layers.Layer):
    """
    Conv2D with option for equalized learning rate (from pggan paper)
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 use_bias=True,
                 equalized_lr=False,
                 gain=np.sqrt(2),
                 kernel_initializer='he_normal',
                 bias_initializer='zeros',
                 **kwargs):
        self.filters = filters
        if np.ndim(kernel_size) == 0:
            self.kernel_size = [kernel_size, kernel_size]
        else:
            self.kernel_size = kernel_size
        if len(strides) == 2:
            self.strides = [1, strides[0], strides[1], 1]
        else:
            self.strides = strides
        self.padding = padding.upper()
        self.use_bias = use_bias
        self.equalized_lr = equalized_lr
        self.gain = gain
        if self.equalized_lr:
            self.kernel_initializer = tf.keras.initializers.RandomNormal(0., 1.)
        else:
            self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        super(Conv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape.as_list()
        self.kernel = self.add_weight(name='kernel',
                                      shape=[self.kernel_size[0], self.kernel_size[1],
                                             input_shape[-1], self.filters],
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=[self.filters],
                                        initializer=self.bias_initializer,
                                        trainable=True)

        if self.equalized_lr:
            fan_in = (self.kernel_size[0]*self.kernel_size[1])*input_shape[-1]
            self.wscale = tf.constant(self.gain / np.sqrt(fan_in), dtype=tf.float32, name="wscale")
        super(Conv2D, self).build(input_shape)

    def call(self, x):
        if self.equalized_lr:
            kernel = self.kernel*self.wscale
        else:
            kernel = self.kernel
        x = tf.nn.conv2d(x, kernel, self.strides, self.padding)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        return x


class Dense(tf.keras.layers.Layer):
    """
    Dense with option for equalized learning rate (from pggan paper)
    """
    def __init__(self,
                 units,
                 use_bias=True,
                 equalized_lr=False,
                 gain=np.sqrt(2),
                 lrmult=1.,
                 kernel_initializer='he_normal',
                 bias_initializer='zeros',
                 **kwargs):
        self.units = units
        self.use_bias = use_bias
        self.equalized_lr = equalized_lr
        self.gain = gain
        self.lrmult = lrmult
        if self.equalized_lr:
            self.kernel_initializer = tf.keras.initializers.RandomNormal(0., 1 / lrmult)
        else:
            self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        super(Dense, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape.as_list()
        self.kernel = self.add_weight(name='kernel',
                                      shape=[input_shape[-1], self.units],
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=[self.units],
                                        initializer=self.bias_initializer,
                                        trainable=True)

        if self.equalized_lr:
            fan_in = input_shape[-1]
            self.wscale = self.lrmult * self.gain / np.sqrt(fan_in)
        super(Dense, self).build(input_shape)

    def call(self, x):
        if self.equalized_lr:
            kernel = self.kernel*self.wscale
        else:
            kernel = self.kernel
        x = tf.matmul(x, kernel)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        return x


class BiasLayer(tf.keras.layers.Layer):
    """
    Dense with option for equalized learning rate (from pggan paper)
    """
    def __init__(self,
                 bias_initializer='zeros',
                 **kwargs):
        self.bias_initializer = bias_initializer
        super(BiasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape.as_list()
        self.bias = self.add_weight(name='bias',
                                    shape=[input_shape[-1]],
                                    initializer=self.bias_initializer,
                                    trainable=True)

        super(BiasLayer, self).build(input_shape)

    def call(self, x):
        x = tf.nn.bias_add(x, self.bias)
        return x


class LearnedInput(tf.keras.layers.Layer):
    def __init__(self, shape=None, **kwargs):
        if shape is None:
            self.shape = [4, 4, 512]
        else:
            self.shape = shape
        super(LearnedInput, self).__init__(**kwargs)

    def build(self, input_shape=None):
        self.value = self.add_variable(name='value', shape=self.shape,
                                       initializer='ones', #tf.keras.initializers.RandomNormal(0., 1.),
                                       dtype=tf.float32, trainable=True)

    def call(self, input=None):
        #  tf.summary.histogram("learned_input", self.value)
        return self.value


class NoiseInput(tf.keras.layers.Layer):
    def __init__(self, shape=None, constant=False, per_pixel=False, **kwargs):
        self.per_pixel = per_pixel
        self.constant = constant
        if shape is None:
            self.shape = [4, 4, 512]
        else:
            self.shape = shape

        self.constant = constant
        if constant:
            self.noise_image = tf.constant(np.random.normal(0., 1., [1, self.shape[0]*self.shape[1]]), dtype=tf.float32)
        super(NoiseInput, self).__init__(**kwargs)

    def build(self, input_shape=None):
        if self.per_pixel:
            target_shape = [self.shape[0]*self.shape[1], self.shape[0]*self.shape[1]*self.shape[2]]
        else:
            target_shape = [1, 1, self.shape[2]]
        self.scale_factor = self.add_variable(name='scale_factor', shape=target_shape,
                                              initializer='zeros',
                                              dtype=tf.float32, trainable=True)

    def call(self, input=None):
        if self.constant:
            noise_image = self.noise_image
        else:
            noise_image = tf.random_normal([1, self.shape[0]*self.shape[1]], 0., 1., dtype=tf.float32)
        if self.per_pixel:
            x = tf.matmul(noise_image, self.scale_factor)
        else:
            noise_image = tf.reshape(noise_image, [self.shape[0], self.shape[1], 1])
            x = tf.multiply(tf.tile(noise_image, [1, 1, self.shape[2]]), self.scale_factor)
        return tf.reshape(x, [1, self.shape[0], self.shape[1], self.shape[2]])


class MappingNetwork(tf.keras.Model):
    def __init__(self, layer_dim=512, num_layers=8, equalized_lr=True):
        super(MappingNetwork, self).__init__()
        self.fc_layers = []
        for i in range(0, num_layers):
            self.fc_layers.append(Dense(layer_dim, equalized_lr=equalized_lr, lrmult=.01, name="FC_%d" % i))

    def call(self, x):
        x = pixel_norm(x)
        for l in self.fc_layers:
            x = tf.nn.leaky_relu(l(x), alpha=.2)
        #  tf.summary.histogram("mapping_network_outputs", x)
        return x


class IntermediateLatentToStyle(tf.keras.Model):
    def __init__(self, channels, name="intermediate_latent_to_style"):
        super(IntermediateLatentToStyle, self).__init__(name=name)
        self.to_ys = Dense(channels, equalized_lr=True, bias_initializer='zeros', name='to_ys', gain=1.)
        self.to_yb = Dense(channels, equalized_lr=True, bias_initializer='zeros', name='to_yb', gain=1.)

    def call(self, w):
        ys = self.to_ys(w)+1.
        yb = self.to_yb(w)
        #  tf.summary.histogram("to_ys", ys)
        #  tf.summary.histogram("to_yb", ys)
        return ys, yb


class Generator(tf.keras.Model):
    def __init__(self, res_w, output_res_w=None, use_pixel_norm=False, equalized_lr=True, use_mapping_network=True,
                 traditional_input=False, add_noise=True, start_shape=(4, 4), resize_method='bilinear',
                 cond_layers=None, map_cond=False, include_fmap_add_ops=False, constant_noise=False):
        super(Generator, self).__init__()
        if output_res_w is None:
            self.output_res_w = res_w
        else:
            self.output_res_w = output_res_w
        self.model_res_w = res_w
        self.use_pixel_norm = use_pixel_norm
        self.add_noise = add_noise
        self.resize_method = resize_method
        self.use_mapping_network = use_mapping_network
        self.start_shape = start_shape
        self.cond_layers = cond_layers
        self.map_cond = map_cond
        self.include_fmap_add_ops = include_fmap_add_ops
        if not traditional_input:
            self.learned_input = LearnedInput(shape=[start_shape[0],
                                                     start_shape[1], 512], name="learned_input")  # todo: move to 4x4x512
        else:
            self.learned_input = None
        if res_w > 4:
            self.toRGB_lower = Conv2D(3, [1, 1], padding='valid',
                                      equalized_lr=equalized_lr, name="toRGB%d" % (res_w // 2))
        self.toRGB = Conv2D(3, [1, 1], padding='valid',
                            equalized_lr=equalized_lr, name="toRGB%d" % res_w)
        self.model_layers = []
        res_h = start_shape[0]
        res_w = start_shape[1]
        filters = 512
        layer_counter = 0
        while res_w <= self.model_res_w:
            if res_w > 32:
                filters = filters // 2
            if self.learned_input is not None and layer_counter == 0:
                first_conv = None
            elif self.learned_input is None and layer_counter == 0:
                first_conv = Conv2D(filters, [4, 4],
                                    padding='valid',
                                    equalized_lr=equalized_lr,
                                    name="convRes%d_1" % res_w,
                                    use_bias=False)
            else:
                first_conv = Conv2D(filters, [3, 3],
                                    padding='same',
                                    equalized_lr=equalized_lr,
                                    name="convRes%d_1" % res_w,
                                    use_bias=False)

            self.model_layers.append([first_conv,
                                     NoiseInput(shape=[res_h, res_w, filters],
                                                constant=constant_noise, name="Noise%d_1" % res_w)
                                      if self.add_noise else None,
                                      BiasLayer(name="Bias%d_1" % res_w),
                                     IntermediateLatentToStyle(filters, name="ILS%d_1" % res_w)
                                      if self.use_mapping_network else None,
                                     Conv2D(filters, [3, 3], padding='same',
                                            equalized_lr=equalized_lr,
                                            name="convRes%d_2" % res_w,
                                            use_bias=False),
                                     NoiseInput(shape=[res_h, res_w, filters],
                                                constant=constant_noise, name="Noise%d_2" % res_w )
                                      if self.add_noise else None,
                                      BiasLayer(name="Bias%d_2" % res_w),
                                     IntermediateLatentToStyle(filters, name="ILS%d_2" % res_w)
                                      if self.use_mapping_network else None])
            res_h *= 2
            res_w *= 2
            layer_counter += 1

    def call(self, alpha, zs=None, intermediate_ws=None, mapping_network=None, cgan_w=None,
             crossover_list=None, random_crossover=False):
        """
        :param alpha:
        :param zs:
        :param intermediate_ws:
        :param mapping_network:
        :param cgan_w:
        :param crossover_list:
        :param random_crossover:
        :return:
        """
        intermediate_mode = (intermediate_ws is not None)
        mixing_mode = isinstance(zs, list) or isinstance(intermediate_ws, list)
        style_mixing = random_crossover or crossover_list is not None
        if zs is None and intermediate_ws is None:
            raise ValueError("Need z or intermediate")
        if self.use_mapping_network and (mapping_network is None and intermediate_ws is None):
            raise ValueError("No mapping network supplied to generator call")

        if not mixing_mode:
            if intermediate_mode:
                intermediate_ws = [intermediate_ws]
            else:
                zs = [zs]
        if not intermediate_mode:
            intermediate_ws = []
            for z in zs:
                z_shape = z.get_shape().as_list()
                if self.use_pixel_norm:
                    z = pixel_norm(z)  # todo: verify correct
                if len(z_shape) == 2:  # [batch size, z dim]
                    if self.map_cond and cgan_w is not None:
                        z = tf.concat([z, cgan_w], -1)
                    intermediate_latent = mapping_network(z)
                    z = tf.expand_dims(z, 1)
                    z = tf.expand_dims(z, 1)
                    #z = tf.reshape(z, [z_shape[0], 1, 1, -1])
                else:  # [batch size, 1, 1, z dim]
                    z_flat = tf.squeeze(z, axis=[1, 2])
                    if self.map_cond and cgan_w is not None:
                        z_flat = tf.concat([z_flat, cgan_w], -1)
                    intermediate_latent = mapping_network(z_flat)
                intermediate_ws.append(intermediate_latent)

        if len(intermediate_ws) > 1 and not random_crossover and crossover_list is None:
                raise ValueError("Need crossover for mixing mode")



        if cgan_w is not None and not self.map_cond:
           intermediate_latent_cond = tf.concat([intermediate_latent, cgan_w], -1)
        else:
           intermediate_latent_cond = None
        intermediate_latent_cond = None

        batch_size = tf.shape(intermediate_ws[0])[0]
        latent_size = tf.shape(intermediate_ws[0])[1]
        if self.learned_input is not None:
            z = tf.expand_dims(self.learned_input(None), axis=0)
            x = tf.tile(z, [batch_size, 1, 1, 1])
        else:
            x = tf.pad(z, [[0, 0], [3, 3], [3, 3], [0, 0]])
        if self.model_res_w == 1:  # for testing purposes
            return x
        current_res = self.start_shape[1]

        # Inefficient implementation, will have to redo
        with tf.name_scope("style_mixing"):
            intermediate_for_layer_list = []
            if random_crossover:
                intermediate_for_layer_list = []
                intermediate_mixing_schedule = tf.random.uniform([batch_size], 0, len(self.model_layers), dtype=tf.int32)
                intermediate_mixing_schedule = tf.transpose(
                    tf.one_hot(intermediate_mixing_schedule, depth=len(self.model_layers), dtype=tf.int32))
                intermediate_multiplier_for_current_layer = tf.zeros([batch_size], dtype=tf.int32)
                for i in range(0, len(self.model_layers)):
                    intermediate_multiplier_for_current_layer = tf.bitwise.bitwise_or(
                        intermediate_multiplier_for_current_layer,
                        intermediate_mixing_schedule[i])
                    intermediate_multiplier = tf.cast(intermediate_multiplier_for_current_layer,
                                                                        dtype=tf.float32)
                    intermediate_multiplier = tf.expand_dims(intermediate_multiplier, 1)
                    intermediate_for_layer_list.append(
                        (1-intermediate_multiplier)*intermediate_ws[0] +
                        intermediate_multiplier*intermediate_ws[1])
            elif crossover_list:
                for i in range(0, len(self.model_layers)):
                    intermediate_index = 0
                    for c in crossover_list:
                        if i >= c:
                            intermediate_index += 1
                    intermediate_for_layer_list.append(intermediate_ws[intermediate_index])
        to_rgb_lower = 0.
        layer_counter = 0
        # shape: [num_layers, batch_size, len(intermediate_w)]

        # for i in range(0, len(self.model_layers)):
        #     latents_to_swap = tf.random.categorical([batch_size, 2])
        #         ([batch_size, latent_size], minval=0, maxval=1, dtype=tf.int32, )
        #     intermediate_for_layer_list
        # if random_crossover:
        #     crossover_layer = tf.random_uniform([tf.shape(intermediate_ws[0])[0], 1], 0, len(self.model_layers),
        #                                         dtype=tf.int32)
        for conv1, noise1, bias1, tostyle1, conv2, noise2, bias2, tostyle2 in self.model_layers:
            with tf.name_scope("Res%d"%current_res):
                #apply_conditioning = intermediate_latent_cond is not None and \
                #    (self.cond_layers is None or
                #     layer_counter in self.cond_layers)
                apply_conditioning = False

                if (self.include_fmap_add_ops):
                    x += tf.zeros([tf.shape(x)], dtype=tf.float32, name="FmapRes%d")
                if layer_counter != 0 or self.learned_input is None:
                    x = conv1(x)
                if self.add_noise:
                    with tf.name_scope("noise_add1"):
                        noise_inputs = noise1(False)
                        assert(x.get_shape().as_list()[1:] == noise_inputs.get_shape().as_list()[1:])
                        x += noise_inputs
                x = bias1(x)
                x = tf.nn.leaky_relu(x, alpha=.2)
                if self.use_pixel_norm:
                    x = pixel_norm(x)

                if apply_conditioning:
                    ys, yb = tostyle1(intermediate_latent_cond)
                else:
                    if style_mixing:
                        ys, yb = tostyle1(intermediate_for_layer_list[layer_counter])
                    else:
                        ys, yb = tostyle1(intermediate_ws[0])
                x = adaptive_instance_norm(x, ys, yb)

                x = conv2(x)
                if self.use_pixel_norm:
                    x = pixel_norm(x)
                if self.add_noise:
                    with tf.name_scope("noise_add2"):
                        noise_inputs = noise2(False)
                        assert(x.get_shape().as_list()[1:] == noise_inputs.get_shape().as_list()[1:])
                        x += noise_inputs
                x = bias2(x)
                x = tf.nn.leaky_relu(x, alpha=.2)

                if apply_conditioning:
                    ys, yb = tostyle2(intermediate_latent_cond)
                else:
                    if style_mixing:
                        ys, yb = tostyle2(intermediate_for_layer_list[layer_counter])
                    else:
                        ys, yb = tostyle2(intermediate_ws[0])
                x = adaptive_instance_norm(x, ys, yb)

                if current_res == self.model_res_w // 2:
                    to_rgb_lower = upsample(self.toRGB_lower(x), method=self.resize_method)
                if current_res != self.model_res_w:
                    x = upsample(x, method=self.resize_method)
                layer_counter += 1
                current_res *= 2
        to_rgb = self.toRGB(x)
        output = to_rgb_lower + alpha * (to_rgb - to_rgb_lower)
        if self.output_res_w//self.model_res_w >= 2:
            output = upsample(output, method='nearest_neighbor',
                              factor=self.output_res_w//self.model_res_w)
        return output


class Discriminator(tf.keras.Model):
    def __init__(self, res, equalized_lr=True, end_shape=(4, 4), do_minibatch_stddev=True, resize_method='bilinear',
                 cgan_nclasses=None, label_list=None):
        super(Discriminator, self).__init__()
        self.res = res
        self.do_minibatch_stddev = do_minibatch_stddev
        self.resize_method = resize_method
        self.end_shape = end_shape
        self.label_list = label_list

        conv_layers = []
        current_res = end_shape[1]
        current_filters = 512
        while current_res <= res:
            if current_res == end_shape[1]:
                conv_layers.append([Conv2D(current_filters, end_shape, padding='same',
                                           equalized_lr=equalized_lr,
                                           name="convRes%d_1" % current_res),
                                    Conv2D(current_filters, end_shape, padding='valid',
                                           equalized_lr=equalized_lr,
                                           name="convRes%d_2" % current_res)])
            else:
                conv_layers.append([Conv2D(current_filters // 2 if current_res > 32 else current_filters,
                                           [3, 3], padding='same', equalized_lr=equalized_lr,
                                           name="convRes%d_1" % current_res),
                                    Conv2D(current_filters,
                                           [3, 3], padding='same', equalized_lr=equalized_lr,
                                           name="convRes%d_2" % current_res)])
            if current_res > 32:
                current_filters = current_filters // 2
            current_res *= 2

        conv_layers.reverse()
        self.conv_layers = conv_layers
        if res > 4:
            self.fromRGB_lower = Conv2D(current_filters if current_filters == 512 else current_filters*2,
                                        [1, 1], padding='valid', equalized_lr=equalized_lr,
                                        name="fromRGB%d" % (res // 2))
        self.fromRGB = Conv2D(current_filters, [1, 1], padding='valid',
                              equalized_lr=equalized_lr, name="fromRGB%d" % res)
        self.fc_layer = Dense(1, equalized_lr=equalized_lr, gain=1, name="fc_layer")
        if cgan_nclasses is not None:
            if label_list is None:
                self.embedding = Dense(cgan_nclasses, equalized_lr=equalized_lr, gain=1,
                                       use_bias=False, name="embedding_layer")
            else:
                self.class_dense_map = {}
                for label in label_list:
                    if label.multi_dim is False:
                        self.class_dense_map[label.name] = \
                            (Dense(1, equalized_lr=equalized_lr, gain=1,
                             name=label.name))
                    else:
                        self.class_dense_map[label.name] = \
                            (Dense(label.num_classes, equalized_lr=equalized_lr, gain=1,
                             name=label.name))

        else:
            self.embedding = None

    def call(self, x, alpha, y=None):
        """
        :param x: image to analyze
        :param alpha: how much weight to give to the current resolution's output vs previous resolution
        :return: classification logit (low number for fake, high for real)
        """
        width = x.get_shape()[2]
        if width != self.res:
            x = downsample(x, 'nearest_neighbor', factor=width // self.res)
        input_lowres = downsample(x, method=self.resize_method)
        x = tf.nn.leaky_relu(self.fromRGB(x), alpha=.2)
        current_res = self.res
        for conv1, conv2 in self.conv_layers:
            if current_res == self.res // 2:
                x_lower = tf.nn.leaky_relu(self.fromRGB_lower(input_lowres), alpha=.2)
                x = x_lower + alpha * (x - x_lower)
            if current_res == self.end_shape[1] and self.do_minibatch_stddev:
                x = minibatch_stddev(x)
            x = tf.nn.leaky_relu(conv1(x), alpha=.2)
            x = tf.nn.leaky_relu(conv2(x), alpha=.2)
            if current_res != self.end_shape[1]:
                x = downsample(x, method=self.resize_method)
            current_res = current_res // 2
        x = tf.reshape(x, [-1, 512])
        logit = self.fc_layer(x)
        if y is not None and self.label_list is None: # proj discrim
            if self.embedding is None:
                raise ValueError("need y value when using cgan")
            conditional_dotprod = tf.reduce_sum(tf.multiply(y, self.embedding(x)),
                                                axis=1, keep_dims=True)
            tf.summary.scalar("conditional_dotprod", tf.reduce_mean(conditional_dotprod))
            logit += conditional_dotprod
            return logit, None
        elif self.label_list is not None: # acgan
            class_logits = {}
            for label in self.label_list:
                class_logits[label.name] = self.class_dense_map[label.name](x)
            return logit, class_logits

        return logit, None # no conditional
