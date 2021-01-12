import unittest
from unittest.mock import Mock

from collaborative_filtering.RowPearsonSimilarityMatrix import RowPearsonSimilarityMatrix
from collaborative_filtering.global_baseline.UserGlobalBaselineCollaborativeFiltering import UserGlobalBaselineCollaborativeFiltering


import numpy as np

from commons.PredictionStrategy import PredictionStrategy


class TestUserGlobalBaselineCollaborativeFiltering(unittest.TestCase):

    rating_matrix = np.array([
        [0, 3, 1, 0, 5, 2, 0, 0, 5],
        [0, 0, 2, 5, 4, 3, 1, 0, 4],
        [5, 5, 1, 4, 3, 5, 2, 1, 3],
        [4, 4, 5, 3, 0, 3, 1, 2, 4]
    ])

    row_to_user_id = [1, 2, 3, 4]

    col_to_movie_id = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    user_id_to_row = {1: 0, 2: 1, 3: 2, 4: 3}

    movie_id_to_col = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8}

    instances_to_predict = np.array([(1, 1), (2, 1), (2, 2), (2, 8), (4, 5)])

    def test_prediction(self):
        data_loader = Mock()
        data_loader.get_prediction_instances = Mock(return_value = self.instances_to_predict)
        data_loader.get_rating_matrix_user_and_movie_data = Mock(return_value = (self.row_to_user_id, self.col_to_movie_id))
        data_loader.get_rating_matrix_user_and_movie_index_translation_dict = Mock(return_value = (self.user_id_to_row, self.movie_id_to_col))
        data_loader.get_ratings_matrix = Mock(return_value = self.rating_matrix)

        sim_matrix_row = RowPearsonSimilarityMatrix(self.rating_matrix)


        prediction_strategy: PredictionStrategy = UserGlobalBaselineCollaborativeFiltering(
            k_neighbors=2,
            sim_matrix=sim_matrix_row
        )


        prediction_strategy.add_data_loader(data_loader)

        prediction_strategy.perform_precomputations()

        expected_prediction = {
            (1, 1): 4.97,
            (2, 1): 4.52,
            (2, 2): 3.84,
            (2, 8): 1.34,
            (4, 5): 3.6
        }

        actual_prediction = prediction_strategy.predict()

        print(actual_prediction)


        for expected_rating, actual_rating in zip(expected_prediction.values(), actual_prediction.values()):
            self.assertAlmostEqual(expected_rating, actual_rating, delta=0.01)


if __name__ == '__main__':
    unittest.main()
