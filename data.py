import tensorflow as tf

def make_raw_dataset(files, batch_size):
    """
    :param files: list of paths containing image files to read into dataset
    :return: dataset that reads those files
    """
    def input_parser_raw(x):
        return tf.image.decode_png(tf.read_file(x), channels=3)

    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(input_parser_raw,
                                                               batch_size, drop_remainder=True))
    return dataset


def make_record_dataset(files, batch_size):
    """
    :param files: list of paths containing tfrecords files to read into dataset
    :return: dataset that reads those files
    """
    def input_parser_record(x):
        example = tf.parse_single_example(serialized=x,
                                          features={"data": tf.FixedLenFeature([], dtype=tf.string),
                                                    "shape": tf.FixedLenFeature([3], tf.int64)})
        return tf.reshape(tf.decode_raw(example["data"], tf.uint8), example["shape"])
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.apply(tf.data.experimental.map_and_batch(input_parser_record,
                                                               batch_size, drop_remainder=True))
    return dataset


def make_record_dataset_nvidia(files, current_res, batch_size, epochs_per_res,
                               label_list=None, num_shards=None, shard_index=None):
    """
    :param files: list of paths containing tfrecords files to read into dataset
    :return: dataset that reads those files
    """
    def input_parser_record(x):
        features = {"data": tf.io.FixedLenFeature([], dtype=tf.string),
                    "shape": tf.io.FixedLenFeature([3], tf.int64)}
        if label_list is not None:
            for label in label_list:
                features[label.name] = tf.io.FixedLenFeature([], tf.float32)
        example = tf.parse_single_example(serialized=x,
                                          features=features)

        decoded = tf.decode_raw(example['data'], tf.uint8)
        #reshaped = tf.transpose(tf.reshape(decoded, example['shape']), [1, 2, 0])
        reshaped = tf.transpose(tf.reshape(decoded, [3, current_res*2, current_res]), [1, 2, 0])
        float_pixels = (tf.cast(reshaped, tf.float32) / 127.5) - 1.
        flipped = tf.image.random_flip_left_right(float_pixels)
        example['data'] = flipped
        if label_list is not None:
            for label in label_list:
                if label.multi_dim:
                    example[label.name] = tf.one_hot(tf.cast(example[label.name], tf.int32),
                                                     label.num_classes)
        return example

    dataset = tf.data.TFRecordDataset([f for f in files if "res%d" % current_res in f])
    if num_shards is not None:
        dataset = dataset.shard(num_shards, shard_index)
    dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=2000, count=epochs_per_res))
    dataset = dataset.apply(tf.data.experimental.map_and_batch(input_parser_record,
                                                               batch_size, drop_remainder=True,
                                                               num_parallel_batches=4))
    return dataset


def get_dataset(files, current_res, epochs_per_res, batch_size, label_list=None, num_shards=None, shard_index=None):
    """
    :param files_regex: path specifying which files to read (example: ./data/*.jpg)
    :return: dataset that reads those files
    """
    if ".png" in files[0] or ".jpg" in files[0]:
        dataset = make_raw_dataset(files)
    elif ".tfrecords" in files[0]:
        dataset = make_record_dataset_nvidia(files, current_res, batch_size, epochs_per_res, label_list=label_list,
                                             num_shards=num_shards, shard_index=shard_index)
    else:  # TODO: make compatible with list of patterns
        raise ValueError("files_regex looks for unknown file types")
    dataset = dataset.prefetch(batch_size)
    return dataset
