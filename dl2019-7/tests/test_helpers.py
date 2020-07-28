import unittest
from train import calculate_weights
import numpy as np
import torch


class TestHelperMethods(unittest.TestCase):
    def test_calculate_weights(self):
        y_train = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                            3]).reshape(-1, 1)
        weights = calculate_weights(y_train)
        expected_weights = torch.Tensor([0.5000, 0.2500, 0.2500,
                                         0.2000]).view(-1, 1)
        self.assertTrue(torch.all(torch.eq(weights, expected_weights)))