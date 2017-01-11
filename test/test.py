import unittest

import numpy as np

from main import smoothen_image

class Tester(unittest.TestCase):

    def test_smoothen(self):
        image = np.array([[0,1,1],
                          [1,0,1],
                          [1,0,0]],dtype=np.uint8)

        smoothened = smoothen_image(image)
        self.assertEqual(smoothened, [[0,1,1],
                                      [1,1,1],
                                      [1,0,0]])