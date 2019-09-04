import tensorflow as tf
import numpy as np
import math
import os
import csv
import json

from tensorflow.python.tools import freeze_graph
from PIL import Image
from utils import build_label_list_from_file
from train import build_models
from train import restore_models_and_optimizers_and_alpha, TrainHps

# todo: shouldn't this be run when importing train?
TrainHps.__new__.__defaults__ = (None,) * len(TrainHps._fields)


def save_sampling_graph(hps, save_paths, graph_dir):
    _, mapping_network, _, sampling_model = build_models(hps,
                                                         hps.current_res_w,
                                                         use_ema_sampling=True,
                                                         num_classes=0,
                                                         label_list=None)

    #sample_latent = tf.placeholder(tf.float32, shape=[1, 512], name="input_latent")  # tf.random_normal
    sample_latent = tf.random_normal([1, 512], 0., 1., name="z");
    intermediate_w = tf.identity(mapping_network(sample_latent), "w")
    sample_img_tensor = sampling_model(1., intermediate_ws=intermediate_w)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        alpha = restore_models_and_optimizers_and_alpha(sess, None, None, mapping_network,
                                                        sampling_model, None, None, None, save_paths)
        sample_img_tensor = tf.clip_by_value(sample_img_tensor, -1., 1.)  # essential due to how tf.summary.image scales values
        sample_img_tensor = tf.cast((sample_img_tensor+1)*127.5, tf.uint8, name="out")
        #png_tensor = tf.image.encode_png(tf.squeeze(sample_img_tensor, axis=0), name="output")
        tf.saved_model.simple_save(sess, graph_dir,
                                   inputs={'z': sample_latent},
                                   outputs={'out': sample_img_tensor})
    """
    freeze_graph ^
        --input_saved_model_dir=saved_graph ^
        --output_graph=generator.pb ^
        --output_node_names=output
    """

def save_image_grid(images, file_path, sess):
    images_shape = images.get_shape().as_list()
    total_images = images_shape[0]
    image_square_side = math.sqrt(total_images)
    if math.floor(image_square_side) != image_square_side:
        image_square_side = int(math.floor(image_square_side)+1)
        num_padding_images = image_square_side**2 - total_images
        padding_images = tf.ones([num_padding_images, images_shape[1], images_shape[2], images_shape[3]])
        images = tf.concat([images, padding_images], axis=0)
    else:
        image_square_side = int(math.floor(image_square_side))
    images = tf.clip_by_value(images, -1., 1.)  # essential due to how tf.summary.image scales values

    grid = tf.contrib.gan.eval.image_grid(
        images,
        grid_shape=[image_square_side, image_square_side],
        image_shape=images.get_shape().as_list()[1:3])
    grid = tf.squeeze(tf.cast((grid+1.)*127.5, tf.uint8))

    png_data = sess.run(tf.image.encode_png(grid))

    with open(file_path, "wb") as f:
        f.write(png_data)

    return grid


def sample_multiple_mix(hps, sample_dir, mix_layer, num_samples):
    print("******************************")
    print("Resolution (w): %d, Alpha %.02f" % (hps.current_res_w, 1.0))
    print("******************************")

    with tf.Session() as sess:
        [sample_img1, sample_img2, sample_img_mix], [intermediate_w1, intermediate_w2] = sample_style_mix(hps, sess, mix_layer)
        sample_img1_tensor = tf.clip_by_value(sample_img1, -1., 1.)  # essential due to how tf.summary.image scales values
        sample_img1_tensor = tf.cast((sample_img1_tensor+1)*127.5, tf.uint8)

        sample_img2_tensor = tf.clip_by_value(sample_img2, -1., 1.)  # essential due to how tf.summary.image scales values
        sample_img2_tensor = tf.cast((sample_img2_tensor+1)*127.5, tf.uint8)

        sample_img_mix_tensor = tf.clip_by_value(sample_img_mix, -1., 1.)  # essential due to how tf.summary.image scales values
        sample_img_mix_tensor = tf.cast((sample_img_mix_tensor+1)*127.5, tf.uint8)
        counter = 0
        with open(os.path.join(sample_dir, "latents_gen.csv"), "w") as f_csv:
            writer = csv.writer(f_csv, delimiter=',')
            for i in range(0, num_samples, hps.batch_size):
                images1, intermediates1, images2, intermediates2, images_mix = \
                    sess.run([sample_img1_tensor, intermediate_w1,
                              sample_img2_tensor, intermediate_w2,
                              sample_img_mix_tensor])
                for j in range(0, hps.batch_size):
                    fname = "sample1_%d.png" % (j+i)
                    image = images1[j]
                    intermediate1 = intermediates1[j]
                    img = Image.fromarray(image, 'RGB')
                    img.save(os.path.join(sample_dir, fname))
                    writer.writerow([fname] + [str(v) for v in intermediate1])

                    fname = "sample2_%d.png" % (j+i)
                    image = images2[j]
                    intermediate2 = intermediates2[j]
                    img = Image.fromarray(image, 'RGB')
                    img.save(os.path.join(sample_dir, fname))
                    writer.writerow([fname] + [str(v) for v in intermediate2])

                    fname = "sample_mix_%d.png" % (j+i)
                    image = images_mix[j]
                    img = Image.fromarray(image, 'RGB')
                    img.save(os.path.join(sample_dir, fname))
           #for image, latent in zip(images, intermediates:
           #    with open

def sample_style_mix(hps, sess, mix_layer):
    tiled_class_latent_batch = None
    tiled_class_latent_many = None
    _, mapping_network, _, sampling_model = build_models(hps,
                                                         hps.current_res_w,
                                                         use_ema_sampling=True)

    sample_latent1 = tf.random_normal([int(hps.batch_size), 512], 0., 1.)
    if hps.map_cond:
        sample_latent1 = tf.concat([sample_latent1, tiled_class_latent_batch], axis=-1)
    sample_latent2 = tf.random_normal([int(hps.batch_size), 512], 0., 1.)
    if hps.map_cond:
        sample_latent2 = tf.concat([sample_latent2, tiled_class_latent_batch], axis=-1)

    many_latent = tf.random_normal([10000, 512], 0., 1)
    if hps.map_cond:
        many_latent = tf.concat([many_latent, tiled_class_latent_many], axis=-1)
    average_w = tf.reduce_mean(mapping_network(many_latent), axis=0)
    intermediate_w1 = average_w + hps.psi_w*(mapping_network(sample_latent1) - average_w)
    sample_img1 = sampling_model(1., intermediate_ws=intermediate_w1)
    intermediate_w2 = mapping_network(sample_latent2) #average_w + hps.psi_w*(mapping_network(sample_latent2) - average_w)
    sample_img2 = sampling_model(1., intermediate_ws=intermediate_w2)
    sample_img_mix = sampling_model(1.,
                                    intermediate_ws=[intermediate_w1, intermediate_w2],
                                    crossover_list=[mix_layer])
    alpha = restore_models_and_optimizers_and_alpha(sess, None, None, mapping_network,
                                                    sampling_model, None, None, None, hps.save_paths)
    return [sample_img1, sample_img2, sample_img_mix], [intermediate_w1, intermediate_w2]


def sample(hps, sess):
    if hps.label_file is not None:
        label_list, total_classes = build_label_list_from_file(hps.label_file)
        if hps.conditional_type != "acgan":
            label_list = None
        class_latent_str = input("Enter %d class values (comma separated):" % total_classes)
        if class_latent_str == "":
            class_latent_str = "1." + ",0."*(total_classes-1)
        class_latent = [float(v) for v in class_latent_str.split(",")]
        tiled_class_latent_batch = [class_latent] * int(hps.batch_size)
        tiled_class_latent_many = [class_latent] * 10000
    else:
        label_list = None
        total_classes = 0
        tiled_class_latent_batch = None
        tiled_class_latent_many = None
    _, mapping_network, _, sampling_model = build_models(hps,
                                                         hps.current_res_w,
                                                         use_ema_sampling=True,
                                                         num_classes=total_classes,
                                                         label_list=label_list)

    sample_latent = tf.random_normal([int(hps.batch_size), 512], 0., 1.)
    if hps.map_cond:
        sample_latent = tf.concat([sample_latent, tiled_class_latent_batch], axis=-1)

    many_latent = tf.random_normal([10000, 512], 0., 1)
    if hps.map_cond:
        many_latent = tf.concat([many_latent, tiled_class_latent_many], axis=-1)
    average_w = tf.reduce_mean(mapping_network(many_latent), axis=0)
    intermediate_w = average_w + hps.psi_w*(mapping_network(sample_latent) - average_w)
    sample_img = sampling_model(1., intermediate_ws=intermediate_w)
    alpha = restore_models_and_optimizers_and_alpha(sess, None, None, mapping_network,
                                                    sampling_model, None, None, None, hps.save_paths)
    return sample_img, intermediate_w


def sample_multiple(hps, sample_dir, num_samples):
    print("******************************")
    print("Resolution (w): %d, Alpha %.02f" % (hps.current_res_w, 1.0))
    print("******************************")

    with tf.Session() as sess:
        sample_img_tensor, intermediates_tensor = sample(hps, sess)
        sample_img_tensor = tf.clip_by_value(sample_img_tensor, -1., 1.)  # essential due to how tf.summary.image scales values
        sample_img_tensor = tf.cast((sample_img_tensor+1)*127.5, tf.uint8)
        counter = 0
        with open(os.path.join(sample_dir, "latents_gen.csv"), "w") as f_csv:
            writer = csv.writer(f_csv, delimiter=',')
            for i in range(0, num_samples, hps.batch_size):
                images, intermediates = sess.run([sample_img_tensor, intermediates_tensor])
                for j in range(0, hps.batch_size):
                    fname = "sample_%d.png" % (j+i)
                    image = images[j]
                    intermediate = intermediates[j]

                    img = Image.fromarray(image, 'RGB')
                    img.save(os.path.join(sample_dir, fname))
                    writer.writerow([fname] + [str(v) for v in intermediate])
           #for image, latent in zip(images, intermediates:
           #    with open


def sample_with_intermediate(hps_path, intermediate, save_paths):
    with open(hps_path, "r") as f:
        hps_dict = json.load(f)
    hps = TrainHps(**hps_dict)
    print("******************************")
    print("Resolution (w): %d, Alpha %.02f" % (hps.current_res_w, 1.0))
    print("******************************")

    # if hps.label_file is not None:
    #     label_list, total_classes = build_label_list_from_file(hps.label_file)
    #     if hps.conditional_type != "acgan":
    #         label_list = None
    #     class_latent_str = input("Enter %d class values (comma separated):" % total_classes)
    #     if class_latent_str == "":
    #         class_latent_str = "1." + ",0."*(total_classes-1)
    #     class_latent = [float(v) for v in class_latent_str.split(",")]
    #     tiled_class_latent_batch = [class_latent]
    #     tiled_class_latent_many = [class_latent] * 10000
    # else:
    #     label_list = None
    #     total_classes = 0
    #     tiled_class_latent_batch = None
    #     tiled_class_latent_many = None
    _, _, _, sampling_model = build_models(hps,
                                           hps.current_res_w,
                                           use_ema_sampling=True,
                                           num_classes=0,
                                           label_list=None)

    sample_img_tensor = sampling_model(1., intermediate_ws=intermediate)
    with tf.Session() as sess:
        alpha = restore_models_and_optimizers_and_alpha(sess, None, None, None,
                                                        sampling_model, None, None, None, save_paths)

        sample_img_tensor = tf.clip_by_value(sample_img_tensor, -1., 1.)  # essential due to how tf.summary.image scales values
        sample_img_tensor = tf.cast((sample_img_tensor+1)*127.5, tf.uint8)
        images = sess.run(sample_img_tensor)
    return images

def sample_grid(hps, sample_dir):
    print("******************************")
    print("Resolution (w): %d, Alpha %.02f" % (hps.current_res_w, 1.0))
    print("******************************")
    with tf.Session() as sess:
        sample_img, _ = sample(hps, sess)
        grid = save_image_grid(sample_img, os.path.join(sample_dir, "image_grid.png"), sess)
        return grid
