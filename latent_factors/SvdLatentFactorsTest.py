import unittest
from unittest.mock import Mock

import numpy as np

from PredictionStrategy import PredictionStrategy
from latent_factors.SvdLatentFactors import SvdLatentFactors


class SvdLatentFactorsTest(unittest.TestCase):
    rating_matrix = np.array([
        [0, 3, 1, 0, 5, 2, 0, 0, 5],
        [0, 0, 2, 5, 4, 3, 1, 0, 4],
        [5, 5, 1, 4, 3, 5, 2, 1, 3],
        [4, 4, 5, 3, 0, 3, 1, 2, 4]
    ])

    instances_to_predict = np.array([(1, 1), (2, 1), (2, 2), (2, 8), (4, 5)])


    def test_predict(self):
        data_loader = Mock()
        data_loader.get_prediction_instances = Mock(return_value=self.instances_to_predict)

        data_loader.get_ratings_matrix = Mock(return_value=self.rating_matrix)

        prediction_strategy: PredictionStrategy = SvdLatentFactors(3)

        prediction_strategy.add_data_loader(data_loader)

        U, D, V = prediction_strategy.predict()

        print(U @ D @ V)

        # print(U @ D @ V)
        # print(U.shape)
        # print(D.shape)
        # print(V.shape)





if __name__ == '__main__':
    unittest.main()
