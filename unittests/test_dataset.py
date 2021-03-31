import unittest
from unittest import TestCase

from dataset import DatasetC1

import router


class TestDatasetC1(TestCase):

    def test_get_data(self):
        """
        Check if processed datasets are available
        :return:
        """
        dset = DatasetC1(name='small', path=router.dataset_small, process_data=False)
        try:
            df = dset.get_data()
        except:
            self.fail()


if __name__ == '__main__':
    unittest.main()
