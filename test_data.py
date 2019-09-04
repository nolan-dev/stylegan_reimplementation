# out of date

import tensorflow as tf
import numpy as np
import os
import glob
from data import get_dataset, preprocess_dataset


class TestDatasetOps(tf.test.TestCase):
    def test_get_dataset_raw(self):
        with self.test_session():
            test_image1 = tf.constant(np.arange(4 * 4 * 3), shape=[4, 4, 3], dtype=tf.uint8)
            encoded = tf.image.encode_png(test_image1)
            image = encoded.eval()
            print(os.getcwd())
            with open(os.path.join("test_files", "test1.png"), "wb") as f:
                f.write(image)

            test_image2 = tf.constant(np.flip(np.arange(4 * 4 * 3), axis=0), shape=[4, 4, 3], dtype=tf.uint8)
            encoded = tf.image.encode_png(test_image2)
            image = encoded.eval()
            with open(os.path.join("test_files", "test2.png"), "wb") as f:
                f.write(image)

            files = glob.glob(os.path.join("test_files", "test*.png"))
            dataset = get_dataset(files)

            it = dataset.make_one_shot_iterator()
            self.assertAllClose(it.get_next(), test_image1)
            self.assertAllClose(it.get_next(), test_image2)

    def test_get_dataset_tfrecords(self):
            with self.test_session():
                test_image1 = tf.constant(np.arange(4 * 4 * 3), shape=[4, 4, 3], dtype=tf.uint8)

                test_image2 = tf.constant(np.flip(np.arange(4 * 4 * 3), axis=0), shape=[4, 4, 3], dtype=tf.uint8)
                writer = tf.python_io.TFRecordWriter(os.path.join("test_files", "test.tfrecords"))
                testimage1_bytes_list = tf.train.BytesList(value=[test_image1.eval().tobytes()])
                example1 = tf.train.Example(
                    features=tf.train.Features(
                        feature={'data': tf.train.Feature(bytes_list=testimage1_bytes_list),
                                 'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[4, 4, 3]))}
                    )
                )
                testimage2_bytes_list = tf.train.BytesList(value=[test_image2.eval().tobytes()])
                example2 = tf.train.Example(
                    features=tf.train.Features(
                        feature={'data': tf.train.Feature(bytes_list=testimage2_bytes_list),
                                 'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[4, 4, 3]))}
                    )
                )
                writer.write(example1.SerializeToString())
                writer.write(example2.SerializeToString())
                writer.close()

                files = glob.glob(os.path.join("test_files", "*.tfrecords"))
                dataset = get_dataset(files)
                it = dataset.make_one_shot_iterator()
                self.assertAllClose(it.get_next(), test_image1)
                self.assertAllClose(it.get_next(), test_image2)

    def test_preprocess_dataset_batch2_float_raw(self):
        with self.test_session():
            test_image1 = tf.constant(np.arange(4 * 4 * 3), shape=[4, 4, 3], dtype=tf.uint8)

            test_image2 = tf.constant(np.flip(np.arange(4 * 4 * 3), axis=0), shape=[4, 4, 3], dtype=tf.uint8)
            writer = tf.python_io.TFRecordWriter(os.path.join("test_files", "test.tfrecords"))
            testimage1_bytes_list = tf.train.BytesList(value=[test_image1.eval().tobytes()])
            example1 = tf.train.Example(
                features=tf.train.Features(
                    feature={'data': tf.train.Feature(bytes_list=testimage1_bytes_list),
                             'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[4, 4, 3]))}
                )
            )
            testimage2_bytes_list = tf.train.BytesList(value=[test_image2.eval().tobytes()])
            example2 = tf.train.Example(
                features=tf.train.Features(
                    feature={'data': tf.train.Feature(bytes_list=testimage2_bytes_list),
                             'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[4, 4, 3]))}
                )
            )
            writer.write(example1.SerializeToString())
            writer.write(example2.SerializeToString())
            writer.close()
            files = glob.glob(os.path.join("test_files", "*.tfrecords"))
            dataset = get_dataset(files)

            dataset = preprocess_dataset(dataset, size=[64, 64], batch_size=2,
                                         float_pixels=True)

            it = dataset.make_one_shot_iterator()
            data = it.get_next().eval()
            self.assertEqual(data.shape, (2, 64, 64, 3))
            self.assertAllClose(max(data.flatten()), max(test_image1.eval().flatten()) / 127.5 - 1.)
            self.assertAllClose(min(data.flatten()), min(test_image1.eval().flatten()) / 127.5 - 1.)

    def test_preprocess_dataset_batch2_float_tfrecord(self):
        with self.test_session():
            test_image1 = tf.constant(np.arange(4 * 4 * 3) * 5, shape=[4, 4, 3], dtype=tf.uint8)
            encoded = tf.image.encode_png(test_image1)
            image1 = encoded.eval()
            with open(os.path.join("test_files", "test1.png"), "wb") as f:
                f.write(image1)

            test_image2 = tf.constant(np.flip(np.arange(4 * 4 * 3) * 5, axis=0), shape=[4, 4, 3],
                                      dtype=tf.uint8)
            encoded = tf.image.encode_png(test_image2)
            image2 = encoded.eval()
            with open(os.path.join("test_files", "test2.png"), "wb") as f:
                f.write(image2)

            files = glob.glob(os.path.join("test_files", "test*.png"))
            dataset = get_dataset(files)

            dataset = preprocess_dataset(dataset, size=[64, 64], batch_size=2,
                                         float_pixels=True)

            it = dataset.make_one_shot_iterator()
            data = it.get_next().eval()
            self.assertEqual(data.shape, (2, 64, 64, 3))
            self.assertAllClose(max(data.flatten()), max(test_image1.eval().flatten()) / 127.5 - 1.)
            self.assertAllClose(min(data.flatten()), min(test_image1.eval().flatten()) / 127.5 - 1.)


if __name__ == "__main__":
    if not os.path.exists("test_files"):
        os.makedirs("test_files")
    tf.test.main()
