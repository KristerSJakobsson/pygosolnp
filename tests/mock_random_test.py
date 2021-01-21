import unittest
from unittest.mock import patch

from tests.mock.mock_random import MockRandom
from random import Random


class TestMockRandom(unittest.TestCase):

    def test_random(self):
        mock_random = MockRandom(x=442)
        first_random_number = mock_random.random()
        self.assertEqual(0.36640772799152665, first_random_number)
        random_normal_no_variance = mock_random.normalvariate(mu=1, sigma=0)
        self.assertEqual(1, random_normal_no_variance)
        random_normal = mock_random.normalvariate(mu=1, sigma=1)
        self.assertEqual(0.7481635417542626, random_normal)

    @patch(target="random.Random", new=MockRandom)
    def test_patch_random_mock(self):
        mock_random = Random(442)
        first_random_number = mock_random.random()
        self.assertEqual(0.36640772799152665, first_random_number)
        random_normal_no_variance = mock_random.normalvariate(mu=1, sigma=0)
        self.assertEqual(1, random_normal_no_variance)
        random_normal = mock_random.normalvariate(mu=1, sigma=1)
        self.assertEqual(0.7481635417542626, random_normal)
