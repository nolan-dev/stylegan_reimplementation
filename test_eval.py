import tensorflow as tf
import os
from eval import save_image_grid, sample


class TestSaveImageGrid(tf.test.TestCase):
    def test_save_image_grid(self):
        images = tf.zeros([8, 32, 32, 3])
        with self.test_session() as sess:
            grid = save_image_grid(images, os.path.join("test_files", "image_grid.png"), sess)
        self.assertAllEqual(grid[0:64, 0:64], tf.fill(grid[0:64, 0:64].get_shape(), 127))
        self.assertAllEqual(grid[64:, 64:], tf.fill(grid[64:, 64:].get_shape(), 255))
