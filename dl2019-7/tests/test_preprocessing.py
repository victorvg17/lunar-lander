import unittest
from train import preprocessing, skip_frames
from config import Config
import numpy as np


class TestPreprocessingMethods(unittest.TestCase):
    def test_preprocessing_dimension_CNN(self):
        N_train = 80
        N_valid = 20
        height = 20
        width = 30
        channels = 3

        X_train = np.random.uniform(size=(N_train, height, width, channels))
        y_train = np.random.uniform(size=(N_train, 1))
        X_valid = np.random.uniform(size=(N_valid, height, width, channels))
        y_valid = np.random.uniform(size=(N_valid, 1))

        conf = Config()
        conf.is_fcn = False  # CNN case
        hist_len = conf.history_length
        sk_no = conf.skip_frames

        # preprocess data
        X_tr_out, y_tr_out, X_va_out, y_va_out = preprocessing(
            X_train, y_train, X_valid, y_valid, conf)

        no_im_tr = ((N_train - (hist_len + 1)) // sk_no) + 1
        no_im_val = ((N_valid - (hist_len + 1)) // sk_no) + 1

        self.assertEqual(np.shape(X_tr_out),
                         (no_im_tr, height, width, hist_len + 1))
        self.assertEqual(np.shape(X_va_out),
                         (no_im_val, height, width, hist_len + 1))
        self.assertEqual(np.shape(y_tr_out), (no_im_tr, 1))
        self.assertEqual(np.shape(y_va_out), (no_im_val, 1))

    def test_preprocessing_dimension_FCN(self):
        N_train = 80
        N_valid = 20
        state_dim = 8
        X_train = np.random.random((N_train, state_dim))
        X_valid = np.random.random((N_valid, state_dim))
        y_train = np.random.random((N_train, 1))
        y_valid = np.random.random((N_valid, 1))

        conf = Config()
        conf.is_fcn = True  # FCN case
        sk_no = conf.skip_frames

        # preprocess data
        X_tr_out, y_tr_out, X_va_out, y_va_out = preprocessing(
            X_train, y_train, X_valid, y_valid, conf)

        expected_no_tr = ((N_train - 1) // sk_no) + 1
        expected_no_val = ((N_valid - 1) // sk_no) + 1

        self.assertEqual(X_tr_out.shape, (expected_no_tr, 8))
        self.assertEqual(X_va_out.shape, (expected_no_val, 8))


if __name__ == '__main__':
    unittest.main()