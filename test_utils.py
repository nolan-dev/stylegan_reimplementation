import os

import tensorflow as tf

from utils import filter_vars_with_checkpoint


class TestUtils(tf.test.TestCase):
    def test_get_variables_in_checkpoint(self):
        chkpt_path = os.path.join("test_files", "chkpt")
        testvar = tf.Variable(2, name="test1", dtype=tf.float32)
        testvar2 = tf.Variable(2, name="test2", dtype=tf.float32)
        assign_op = tf.assign(testvar, 3.)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(var_list=[testvar])
            saver.save(sess, chkpt_path)
            filtered_vars = filter_vars_with_checkpoint(chkpt_path, [testvar, testvar2])
            self.assertAllEqual(filtered_vars, [testvar])
            sess.run(assign_op)
            self.assertEqual(sess.run(testvar), 3.)
            saver = tf.train.Saver(var_list=filtered_vars)
            saver.restore(sess, chkpt_path)
            self.assertEqual(sess.run(testvar), 2.)


if __name__ == "__main__":
    if not os.path.exists("test_files"):
        os.makedirs("test_files")
    tf.test.main()
