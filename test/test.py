import unittest

import numpy as np
from scipy.ndimage import generic_filter

from main import smoothen_image

class Tester(unittest.TestCase):

    def test_smoothen(self):
        image = np.array([[0,1,1],
                          [1,0,1],
                          [1,0,0]],dtype=np.uint8)
        smoothened = smoothen_image(image)
        self.assertTrue(np.array_equal(smoothened, [[0,1,1],
                                      [1,1,1],
                                      [1,0,0]]))

        image2 = np.array([[0,1,1],
                           [0,0,1],
                           [0,0,0]],dtype=np.uint8)
        smoothened2 = smoothen_image(image2)
        self.assertTrue(np.array_equal(smoothened2, [[0,1,1],
                                      [0,0,1],
                                      [0,0,0]]))

    def random_test(self):
        import functools
        fp = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], np.uint8)
        median_filter = functools.partial(generic_filter,
                                          function=np.median,
                                          footprint=fp)

        image = np.array([[0,1,1],
                          [1,0,1],
                          [1,0,0]],dtype=np.uint8)

        print(median_filter(image))