import unittest
from unittest.mock import Mock

from commons.RatingPredictor import RatingPredictor
from collaborative_filtering.lsh.ItemLshCollaborativeFiltering import ItemLshCollaborativeFiltering
from collaborative_filtering.lsh.UserLshCollaborativeFiltering import UserLshCollaborativeFiltering
from data_handling.DiskPersistor import DiskPersistor
from commons.FormulaFactory import FormulaFactory


import numpy as np

from commons.PredictionStrategy import PredictionStrategy


class TestUserCollaborativeFiltering(unittest.TestCase):

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

        user_colab: PredictionStrategy = UserLshCollaborativeFiltering(
            k_neighbors=5,
            signiture_length=4,
            max_query_distance=16,
            formula_factory=FormulaFactory(),
            random_seed=3
        )

        item_colab: PredictionStrategy = ItemLshCollaborativeFiltering(
            k_neighbors=5,
            signiture_length=4,
            max_query_distance=16,
            formula_factory=FormulaFactory(),
            random_seed=3
        )


        predictor: RatingPredictor = RatingPredictor(data_loader, DiskPersistor(), "test_predictor", [user_colab, item_colab])

        predictor.perform_precomputations()


        expected_prediction = {
            (1, 1): 3.2,
            (2, 1): 3.22,
            (2, 2): 2.75,
            (2, 8): 2.33,
            (4, 5): 3.39
        }

        actual_prediction = predictor.make_average_prediction(weights=[0.3, 0.7])

        print(actual_prediction)


        for expected_rating, actual_rating in zip(expected_prediction.values(), actual_prediction.values()):
            self.assertAlmostEqual(expected_rating, actual_rating, delta=0.01)


if __name__ == '__main__':
    unittest.main()
