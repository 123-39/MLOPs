""" tests """
# pylint: disable=E0401, E0611


import unittest

from src.data import read_dataset, split_train_val_data
from src.entities import SplittingParams


class TestProject(unittest.TestCase):

    def test_read_dataset(self):

        data = read_dataset('data/raw/heart_cleveland_upload.csv')

        self.assertEqual(297, len(data))

    def test_split_data(self):

        data = read_dataset('data/raw/heart_cleveland_upload.csv')

        splitting_params = SplittingParams(random_state=42, test_size=0.1)

        train, test = split_train_val_data(data, splitting_params)

        self.assertEqual(267, len(train))
        self.assertEqual(30, len(test))


if __name__ == '__main__':

    unittest.main()
